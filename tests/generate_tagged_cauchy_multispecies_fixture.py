#!/usr/bin/env python3
"""Synthetic two-species tagged-Cauchy fixture (written before any real
multi-species export exists).

Requires installed ``ye3t`` and ``ye3t-ace`` (not yet public).

`ye3t.couplings.first_lifted_cauchy_scalar_request` already accepts an
`elements` tuple (not just a single `element`) and lays out one channel per
`(element, radial_channel)` pair, uniform `l`/radial params across every
element -- the radial definition is one ChebExpCos parameter set for all
pairs. This script calls that real
compiler pipeline (`count`/`plan`/`compile`) directly with
`elements=("Ta","W")` (bypassing `ye3t_ace.tagged_cauchy_linear.
compile_tagged_cauchy_artifact`, which only exposes a single `element`), so
the resulting `CompiledLiftedCauchyScalar` is a genuine, fully self-
consistent multi-species artifact (correct resource report, internal
self_hash, descriptors that actually couple Ta and W channels) -- not a
hand-patched approximation of one.

`beta` is then written as the per-species `{species: [feature_count doubles]}`
mapping (two independently-random vectors), and `real_moment_program` is the
real `ye3t_ace.tagged_cauchy_linear.real_moment_program` exact-sympy
lowering of that compiled artifact (so its own zero-imaginary-residual proof
covers the construction).

Two single-center edge-list fixtures are generated (one Ta-center, one
W-center), each with neighbors of BOTH species, evaluated in Python via
`RealMomentEvaluator` and compared against the native C++ evaluator by
`tests/test_tagged_cauchy_cpu.cpp` (same fixture schema as
`generate_tagged_cauchy_fixture.py`) to 1e-10.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ye3t.couplings import compile as compile_coupling
from ye3t.couplings import count as count_coupling
from ye3t.couplings import first_lifted_cauchy_scalar_request
from ye3t.couplings import plan as plan_coupling
from ye3t.execution_plan import compile_tagged_moment_execution_portfolio
from ye3t_ace.equivariant_calc.angular_basis import ComplexSphericalHarmonicsBasis
from ye3t_ace.equivariant_calc.radial_basis import _pace_cheb_exp_cos_table_with_derivative
from ye3t_ace.lifted_cauchy_linear import _artifact_channels
from ye3t_ace.tagged_cauchy_linear import (
    DEFAULT_TA_FAMILIES,
    TA_CUTOFF,
    TA_RADIAL_CONFIG,
    RealMomentEvaluator,
    complex_to_artifact_real,
)

_ANGULAR_BASIS = ComplexSphericalHarmonicsBasis()


def _canonical_json_sha256(body):
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def compile_multi_species(elements, radial_channel_count=2, angular_l=1):
    request = dict(
        first_lifted_cauchy_scalar_request(
            int(radial_channel_count), elements=tuple(elements), angular_l=int(angular_l),
            family_ids=tuple(DEFAULT_TA_FAMILIES),
        )
    )
    request["role_dimension"] = 2  # tag_count 1 (role 0 = edge tag, role 1 = density)
    request["emit_canonical"] = True
    request["emit_factored"] = False
    request["emit_ordered_reference"] = False
    report = count_coupling(request)
    compiler_plan = plan_coupling(report)
    return compile_coupling(compiler_plan)


def build_phi_complex(species_order, channels, displacements, edge_species_index, cutoff, radial_config):
    species_index = {name: index for index, name in enumerate(species_order)}
    radial_count = max(int(channel["radial_channel"]) for channel in channels) + 1
    max_width = max(2 * int(channel["l"]) + 1 for channel in channels)

    dist = torch.linalg.norm(displacements, dim=-1)
    radial_values, _ = _pace_cheb_exp_cos_table_with_derivative(
        dist, rc=cutoff, cutoff_width=float(radial_config.get("cutoff_width", 0.0)),
        lmbda=float(radial_config["lmbda"]), radial_count=radial_count,
    )

    angular_cache = {}
    channel_tensors = []
    for channel in channels:
        angular_l = int(channel["l"])
        radial_n = int(channel["radial_channel"])
        neighbor_species = str(channel["neighbor_species"])
        if angular_l not in angular_cache:
            angular_cache[angular_l] = _ANGULAR_BASIS.cartesian_values(angular_l, displacements).transpose(0, 1)
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


def evaluate_one_center(compiled, role_bindings, channels, species_order, cutoff, radial_config,
                        beta_by_species, offsets, central_species, edge_species_names, displacements_np):
    n = len(edge_species_names)
    edge_species_np = np.array([species_order.index(name) for name in edge_species_names], dtype=np.int64)
    edge_species_index = torch.tensor(edge_species_np, dtype=torch.long)
    displacements = torch.tensor(displacements_np, dtype=torch.float64, requires_grad=True)

    phi_complex = build_phi_complex(species_order, channels, displacements, edge_species_index, cutoff, radial_config)
    phi_real = complex_to_artifact_real(phi_complex, compiled)
    A_real = phi_real.sum(dim=0, keepdim=True)
    src = torch.zeros(n, dtype=torch.long)

    evaluator = RealMomentEvaluator(compiled, role_bindings, None)
    features = evaluator.evaluate_real(phi_real, A_real, src, 1)
    beta = torch.tensor(beta_by_species[central_species], dtype=torch.float64)
    if int(features.shape[1]) != int(beta.shape[0]):
        raise SystemExit(
            f"RealMomentEvaluator feature_count {features.shape[1]} != beta length {beta.shape[0]}"
        )
    energy = (features[0] * beta).sum() + float(offsets[central_species])
    (gradient,) = torch.autograd.grad(energy, displacements)
    return float(energy.detach()), gradient.detach().tolist(), int(evaluator.descriptor_count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_output", type=Path)
    parser.add_argument("fixture_ta_output", type=Path)
    parser.add_argument("fixture_w_output", type=Path)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--edge-count", type=int, default=10)
    args = parser.parse_args()

    species_order = ["Ta", "W"]
    cutoff = float(TA_CUTOFF)
    radial_config = dict(TA_RADIAL_CONFIG)
    role_bindings = (("edge", 0), ("density", 0))
    tag_count = 1

    compiled = compile_multi_species(species_order, radial_channel_count=2, angular_l=1)
    channels = _artifact_channels(compiled)
    observed_species = sorted({str(c["neighbor_species"]) for c in channels})
    if observed_species != sorted(species_order):
        raise SystemExit(
            f"compiled artifact channels reference species {observed_species}, expected {species_order}"
        )

    evaluator = RealMomentEvaluator(compiled, role_bindings, None)
    program = evaluator.program
    feature_count = int(program["feature_count"])
    print(
        f"compiled multi-species artifact: {len(channels)} channels, "
        f"{feature_count} features, {len(program['terms'])} terms, "
        f"real_density_key_count={len(program['real_density_keys'])}, "
        f"real_moment_key_count={len(program['real_moment_keys'])}"
    )

    rng = np.random.default_rng(args.seed)
    beta_by_species = {
        "Ta": rng.uniform(-50.0, 50.0, size=feature_count).tolist(),
        "W": rng.uniform(-50.0, 50.0, size=feature_count).tolist(),
    }
    offsets = {"Ta": -7.5, "W": -11.25}

    channel_real_form_id_of = {
        int(r["channel_index"]): str(r["real_form_id"]) for r in compiled.payload["channel_real_form_ids"]
    }
    channel_real_forms = [
        {
            "channel_index": int(channel["channel_index"]),
            "l": int(channel["l"]),
            "neighbor_species": str(channel["neighbor_species"]),
            "radial_channel": int(channel["radial_channel"]),
            "real_form_id": channel_real_form_id_of[int(channel["channel_index"])],
        }
        for channel in channels
    ]
    radial_definition = {
        "kind": "pace_cheb_exp_cos",
        "parameters": {
            "rc": cutoff,
            "cutoff_width": float(radial_config.get("cutoff_width", 0.0)),
            "lmbda": float(radial_config["lmbda"]),
            "radial_count": max(int(c["radial_channel"]) for c in channels) + 1,
        },
        "column_mapping": (
            "PACE ChebExpCos native radial labels are 1-based; evaluator output "
            "column j (0-based, j = 0 .. radial_count-1) is PACE n = j + 1."
        ),
    }

    body = {
        "schema": "ye3t_tagged_cauchy_slice_v2",
        "compiled_artifact": compiled.to_dict(),
        "role_bindings": [[k, v] for k, v in role_bindings],
        "descriptor_selection": None,
        "beta": beta_by_species,
        "offsets": offsets,
        "offset_mode": "fitted_species_offsets",
        "radial_config": radial_config,
        "cutoff": cutoff,
        "species_order": species_order,
        "tag_count": tag_count,
        "context_policy": "inclusive",
        "pooling": "ordered_sum",
        "real_moment_program": program,
        "tagged_execution_portfolio": compile_tagged_moment_execution_portfolio(
            program, beta_by_species
        ),
        "channel_real_forms": channel_real_forms,
        "radial_definition": radial_definition,
        "synthetic_fixture_note": (
            "Hand-built two-species (Ta, W) artifact compiled directly via "
            "ye3t.couplings.{count,plan,compile} with elements=('Ta','W') "
            "(first_lifted_cauchy_scalar_request's existing multi-element "
            "support), before any real multi-species export exists; "
            "see generate_tagged_cauchy_multispecies_fixture.py."
        ),
    }
    self_hash = _canonical_json_sha256(body)
    payload = dict(body)
    payload["self_hash"] = self_hash
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    print(f"wrote {args.model_output}: feature_count={feature_count}, self_hash={self_hash}")

    rng_edges = np.random.default_rng(args.seed + 1)

    def random_edges(n, species_choices):
        directions = rng_edges.normal(size=(n, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radii = rng_edges.uniform(low=0.28 * cutoff, high=0.96 * cutoff, size=n)
        displacements = directions * radii[:, None]
        species_names = [species_choices[i % len(species_choices)] for i in range(n)]
        rng_edges.shuffle(species_names)
        return displacements, species_names

    for label, central_species, output_path in (
        ("ta_center_mixed_neighbors", "Ta", args.fixture_ta_output),
        ("w_center_mixed_neighbors", "W", args.fixture_w_output),
    ):
        displacements_np, edge_species_names = random_edges(int(args.edge_count), ["Ta", "W"])
        energy, gradients, descriptor_count = evaluate_one_center(
            compiled, role_bindings, channels, species_order, cutoff, radial_config,
            beta_by_species, offsets, central_species, edge_species_names, displacements_np,
        )
        fixture = {
            "schema": "ye3t_tagged_cauchy_cpu_fixture_v1",
            "model_path": (
                args.model_output.name
                if args.model_output.resolve().parent == output_path.resolve().parent
                else str(args.model_output)
            ),
            "model_self_hash": self_hash,
            "seed": int(args.seed),
            "tag_count": tag_count,
            "central_species": central_species,
            "edges": [
                {"neighbor_species": edge_species_names[i], "displacement_A": displacements_np[i].tolist()}
                for i in range(len(edge_species_names))
            ],
            "real_moment_program_feature_count": descriptor_count,
            "energy_eV": energy,
            "edge_gradients_dE_dd_eV_per_A": gradients,
            "label": label,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        species_counts = {s: edge_species_names.count(s) for s in species_order}
        print(f"wrote {output_path}: {label}, energy={energy:.15g} eV, neighbor species counts {species_counts}")


if __name__ == "__main__":
    main()
