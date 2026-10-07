#!/usr/bin/env python3
"""Regenerate the deterministic tagged-Cauchy native-runtime unit fixture.

Builds one owned center's edge list by hand (no neighbor search, no ASE
cell) and evaluates it directly with
:class:`ye3t_methods.atomistic.tagged_cauchy_linear.RealMomentEvaluator` -- the exact
real-arithmetic evaluator the native C++ loader/evaluator is required to
match -- via ``torch`` autograd on the edge displacements themselves, so the
exported ``edge_gradients_eV_per_A`` are ``dE/d(displacement)`` for that
center with no other atom in the system (this is what
``tests/test_tagged_cauchy_cpu.cpp`` compares the native evaluator against).

Requires ``descriptor_selection: null``. If ``RealMomentEvaluator(compiled,
role_bindings, None)``'s ``descriptor_count`` does not already match
``len(beta)``, the exported ``real_moment_program`` was built with a pooled
``combination_matrix`` (as both current v2 exports document); that
matrix is reconstructed deterministically from ``(compiled, role_bindings)``
via ``ye3t.couplings.tagged_cauchy.pooled_tagged_basis``/
``pooled_feature_matrix`` and passed back into ``RealMomentEvaluator``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ye3t.couplings.tagged_cauchy import pooled_feature_matrix, pooled_tagged_basis
from ye3t_methods.atomistic.equivariant_calc.angular_basis import ComplexSphericalHarmonicsBasis
from ye3t_methods.atomistic.equivariant_calc.radial_basis import _pace_cheb_exp_cos_table_with_derivative
from ye3t_methods.atomistic.lifted_cauchy_linear import _artifact_channels
from ye3t_methods.atomistic.tagged_cauchy_linear import RealMomentEvaluator, load_tagged_model

_ANGULAR_BASIS = ComplexSphericalHarmonicsBasis()


def build_phi_complex(model, channels, displacements, edge_species_index):
    """``[edges, channels, max_width]`` complex ordinary-ACE primitives."""

    species_index = {name: index for index, name in enumerate(model.species_order)}
    radial_count = max(int(channel["radial_channel"]) for channel in channels) + 1
    max_width = max(2 * int(channel["l"]) + 1 for channel in channels)

    dist = torch.linalg.norm(displacements, dim=-1)
    radial_values, _ = _pace_cheb_exp_cos_table_with_derivative(
        dist,
        rc=model.cutoff,
        cutoff_width=float(model.radial_config.get("cutoff_width", 0.0)),
        lmbda=float(model.radial_config["lmbda"]),
        radial_count=radial_count,
    )

    angular_cache = {}
    channel_tensors = []
    for channel in channels:
        angular_l = int(channel["l"])
        radial_n = int(channel["radial_channel"])
        neighbor_species = str(channel["neighbor_species"])
        if angular_l not in angular_cache:
            angular_cache[angular_l] = _ANGULAR_BASIS.cartesian_values(
                angular_l, displacements
            ).transpose(0, 1)
        angular = angular_cache[angular_l]
        width = int(angular.shape[-1])
        radial_column = radial_values[:, radial_n].to(angular.dtype)
        value = radial_column.unsqueeze(-1) * angular
        species_mask = (edge_species_index == species_index[neighbor_species]).unsqueeze(-1)
        value = torch.where(species_mask, value, torch.zeros_like(value))
        if width < max_width:
            pad = value.new_zeros(tuple(value.shape[:-1]) + (max_width - width,))
            value = torch.cat((value, pad), dim=-1)
        channel_tensors.append(value)
    return torch.stack(channel_tensors, dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path, help="path to a ye3t_tagged_cauchy_slice_v2 JSON")
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--edge-count", type=int, default=9)
    args = parser.parse_args()

    model = load_tagged_model(str(args.model))
    if model.descriptor_selection is not None:
        raise SystemExit(
            "generate_tagged_cauchy_fixture.py only supports descriptor_selection=null "
            "models (RealMomentEvaluator(compiled, role_bindings, None) must reproduce "
            "the artifact's embedded real_moment_program exactly); got a non-null "
            "descriptor_selection."
        )
    channels = _artifact_channels(model.compiled)
    central_species = model.species_order[0]

    rng = np.random.default_rng(args.seed)
    n = int(args.edge_count)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.uniform(low=0.28 * model.cutoff, high=0.96 * model.cutoff, size=n)
    displacements_np = directions * radii[:, None]
    edge_species_np = rng.integers(0, len(model.species_order), size=n)

    edge_species_index = torch.tensor(edge_species_np, dtype=torch.long)
    displacements = torch.tensor(displacements_np, dtype=torch.float64, requires_grad=True)

    phi_complex = build_phi_complex(model, channels, displacements, edge_species_index)
    from ye3t_methods.atomistic.tagged_cauchy_linear import complex_to_artifact_real

    phi_real = complex_to_artifact_real(phi_complex, model.compiled)
    A_real = phi_real.sum(dim=0, keepdim=True)
    src = torch.zeros(n, dtype=torch.long)

    combination_matrix = None
    evaluator = RealMomentEvaluator(model.compiled, model.role_bindings, model.descriptor_selection)
    if int(evaluator.descriptor_count) != int(model.beta.shape[0]):
        # The exported real_moment_program was built with a pooled-feature
        # combination_matrix (see export_tagged_model's real_moment_
        # combination_matrix/real_moment_selection): reconstruct the same
        # pooling deterministically from (compiled, role_bindings) via
        # ye3t.couplings.tagged_cauchy.pooled_tagged_basis, matching what
        # both tagged_cauchy_k1_v2_reference.json and tagged_cauchy_k2_v2_
        # reference.json document as the export recipe.
        basis = pooled_tagged_basis(model.compiled, model.role_bindings)
        descriptor_count = len(model.compiled.payload["descriptors"])
        combination_matrix = pooled_feature_matrix(basis, descriptor_count)
        evaluator = RealMomentEvaluator(
            model.compiled, model.role_bindings, model.descriptor_selection,
            combination_matrix=combination_matrix,
        )
    if int(evaluator.descriptor_count) != int(model.beta.shape[0]):
        raise SystemExit(
            f"RealMomentEvaluator feature_count {evaluator.descriptor_count} still does "
            f"not match len(beta) {model.beta.shape[0]} after pooled-basis reconstruction"
        )
    features = evaluator.evaluate_real(phi_real, A_real, src, 1)
    beta = model.beta.to(features.dtype)
    energy = (features[0] * beta).sum() + float(model.offsets[central_species])
    (gradient,) = torch.autograd.grad(energy, displacements)

    self_hash = json.loads(args.model.read_text(encoding="utf-8"))["self_hash"]
    fixture = {
        "schema": "ye3t_tagged_cauchy_cpu_fixture_v1",
        "model_path": str(args.model),
        "model_self_hash": self_hash,
        "seed": int(args.seed),
        "tag_count": int(model.evaluator.tag_count),
        "central_species": central_species,
        "edges": [
            {
                "neighbor_species": model.species_order[int(edge_species_np[i])],
                "displacement_A": displacements_np[i].tolist(),
            }
            for i in range(n)
        ],
        "real_moment_program_feature_count": int(features.shape[1]),
        "pooled_combination_matrix_used": combination_matrix is not None,
        "energy_eV": float(energy.detach()),
        "edge_gradients_dE_dd_eV_per_A": gradient.detach().tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}: energy={fixture['energy_eV']:.15g} eV, {n} edges")


if __name__ == "__main__":
    main()
