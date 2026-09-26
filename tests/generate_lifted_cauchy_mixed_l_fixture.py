"""Regenerate the deterministic mixed-l lifted-Cauchy native fixture.

Requires installed ``ye3t`` and ``ye3t-ace`` (not yet public).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ase import Atoms

from ye3t.couplings import compile as compile_coupling
from ye3t.couplings import lifted_cauchy_fixed_content_scalar_request
from ye3t_ace import YE3TDescriptors, export_lifted_cauchy_linear_bundle
from ye3t_ace.lifted_cauchy_linear import (
    LIFTED_CAUCHY_MIXED_L_SOURCE_FAMILY,
    lifted_cauchy_model_from_descriptor,
)


def _packed(values, blocks):
    rows = []
    for block in blocks:
        position = int(block["channel_position"])
        width = int(block["real_component_count"])
        rows.append(values[:, position, :, :width].reshape(len(values), -1))
    return torch.cat(tuple(rows), dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    channels = (
        {
            "channel_id": "n0_l0",
            "neighbor_species": "Ta",
            "radial_channel": 0,
            "l": 0,
            "source_family_id": LIFTED_CAUCHY_MIXED_L_SOURCE_FAMILY,
        },
        {
            "channel_id": "n0_l1",
            "neighbor_species": "Ta",
            "radial_channel": 0,
            "l": 1,
            "source_family_id": LIFTED_CAUCHY_MIXED_L_SOURCE_FAMILY,
        },
        {
            "channel_id": "n0_l2",
            "neighbor_species": "Ta",
            "radial_channel": 0,
            "l": 2,
            "source_family_id": LIFTED_CAUCHY_MIXED_L_SOURCE_FAMILY,
        },
    )
    request = lifted_cauchy_fixed_content_scalar_request(
        channels,
        (1, 2, 2),
        role_dimension=2,
    )
    compiled = compile_coupling(request)
    labels = tuple(
        descriptor["label"] for descriptor in compiled.payload["descriptors"]
    )
    nontrivial_labels = tuple(
        label
        for label in labels
        if sum(len(tuple(kappa)) > 1 for kappa in label["block_kappas"])
        >= 2
        and sum(int(value) == 1 for value in label["block_Lambdas"]) >= 2
    )
    if not nontrivial_labels:
        raise RuntimeError(
            "The mixed-l fixture must contain compiler-owned nontrivial "
            "internal kappa and nonzero intermediate angular momentum."
        )
    descriptor = YE3TDescriptors.ye3t_basis(
        {
            "elements": ("Ta",),
            "type_map": {"Ta": 0},
            "lifted_cauchy": {
                "compiled": compiled.to_dict(),
                "source": {
                    "schema": "ye3t_lifted_cauchy_joint_source_v3",
                    "cutoff_A": 5.2,
                    "periodic_image_mode": "nonperiodic",
                    "neighbor_backend": "ase",
                },
            },
        }
    )
    model = lifted_cauchy_model_from_descriptor(
        descriptor,
        {
            "coefficients": (
                tuple(
                    np.linspace(
                        -0.007, 0.011, len(compiled.payload["descriptors"])
                    )
                ),
            ),
            "offsets": (0.2,),
            "realization": "canonical",
            "source_realization": "factorized",
        },
    )
    model.fit_metadata = {
        "purpose": "deterministic_mixed_l_native_correctness_fixture",
        "training": "not_fitted",
        "publication_timing": False,
    }
    factored_model = lifted_cauchy_model_from_descriptor(
        descriptor,
        {
            "coefficients": model.coefficients.detach().cpu().numpy(),
            "offsets": model.offsets.detach().cpu().numpy(),
            "realization": "factored",
            "source_realization": "factorized",
        },
    )
    direct_model = lifted_cauchy_model_from_descriptor(
        descriptor,
        {
            "coefficients": model.coefficients.detach().cpu().numpy(),
            "offsets": model.offsets.detach().cpu().numpy(),
            "realization": "canonical",
            "source_realization": "direct",
        },
    )
    export_lifted_cauchy_linear_bundle(model, args.output)
    native = json.loads(
        (args.output / "native_runtime.ye3t.json").read_text(encoding="utf-8")
    )
    blocks = native["coordinate_convention"]["channel_blocks"]

    positions = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.8, 0.1, -0.2),
            (-0.4, 2.0, 0.3),
            (0.2, -0.7, 2.1),
        ),
        dtype=float,
    )
    atoms = Atoms("Ta4", positions=positions, pbc=False)
    result = model.evaluate_atoms(atoms, forces=True, stress=True)
    factored_result = factored_model.evaluate_atoms(atoms, forces=True, stress=True)
    direct_result = direct_model.evaluate_atoms(atoms, forces=True, stress=True)
    source_force_error = torch.max(
        torch.abs(direct_result["forces"] - result["forces"])
    ).item()
    if source_force_error > 2.0e-9:
        raise RuntimeError(
            "Mixed-l direct and factorized source autograd forces disagree by "
            f"{source_force_error:.6e}."
        )
    feature_force_error = torch.max(
        torch.abs(factored_result["forces"] - result["forces"])
    ).item()
    if feature_force_error > 2.0e-9:
        raise RuntimeError(
            "Mixed-l canonical and factored autograd forces disagree by "
            f"{feature_force_error:.6e}."
        )
    atom_types = torch.zeros(len(positions), dtype=torch.long)
    positions_t = torch.as_tensor(
        positions, dtype=torch.float64
    ).requires_grad_(True)
    density, context = model.source.materialize(
        positions_t,
        atom_types,
        source_realization="direct",
    )
    features, source_adjoint = model.evaluator.vjp(
        density,
        model.coefficients[0].expand(len(positions), -1),
        realization="canonical",
        method="explicit",
    )
    _autograd_features, autograd_source_adjoint = model.evaluator.vjp(
        density.detach().requires_grad_(True),
        model.coefficients[0].expand(len(positions), -1),
        realization="canonical",
        method="autograd",
    )
    density_adjoint_error = torch.max(
        torch.abs(source_adjoint - autograd_source_adjoint)
    ).item()
    if density_adjoint_error > 2.0e-9:
        raise RuntimeError(
            "Mixed-l explicit and autograd density adjoints disagree by "
            f"{density_adjoint_error:.6e}."
        )
    atomic_energies = (features * model.coefficients[0]).sum(dim=1) + model.offsets[0]
    edge_index = context["edge_index"]
    edge_adjoint = source_adjoint.index_select(0, edge_index[0])
    edge_gradients = torch.einsum(
        "ecsq,ecsqd->ed", edge_adjoint, context["edge_dx"]
    )
    direct_pullback = model.source.vjp(
        source_adjoint, context, atom_count=len(positions), include_strain=True
    )
    direct_forces = -direct_pullback["position_gradient"]
    autograd_source_gradient = torch.autograd.grad(
        (density * source_adjoint.detach()).sum(), positions_t, retain_graph=True
    )[0]
    source_vjp_error = torch.max(
        torch.abs(
            direct_pullback["position_gradient"] - autograd_source_gradient
        )
    ).item()
    if source_vjp_error > 2.0e-9:
        raise RuntimeError(
            "Mixed-l analytic and autograd source VJPs disagree by "
            f"{source_vjp_error:.6e}."
        )
    channel_errors = []
    for channel in range(int(density.shape[1])):
        seed = torch.zeros_like(source_adjoint)
        seed[:, channel] = source_adjoint[:, channel]
        analytical_channel = model.source.vjp(
            seed, context, atom_count=len(positions), include_strain=False
        )["position_gradient"]
        automatic_channel = torch.autograd.grad(
            (density[:, channel] * seed[:, channel].detach()).sum(),
            positions_t,
            retain_graph=True,
        )[0]
        channel_errors.append(
            torch.max(torch.abs(analytical_channel - automatic_channel)).item()
        )
    if max(channel_errors) > 2.0e-9:
        raise RuntimeError(
            "Mixed-l source VJP channel error exceeds tolerance: "
            f"{channel_errors}."
        )
    explicit_force_error = torch.max(
        torch.abs(direct_forces - result["forces"])
    ).item()
    if explicit_force_error > 2.0e-9:
        raise RuntimeError(
            "Mixed-l explicit and autograd canonical forces disagree by "
            f"{explicit_force_error:.6e}; density-adjoint error is "
            f"{density_adjoint_error:.6e}; source-VJP error is "
            f"{source_vjp_error:.6e}; per-channel errors are {channel_errors}."
        )
    numerical_forces = torch.zeros_like(direct_forces)
    step = 1.0e-5
    for atom in range(len(positions)):
        for axis in range(3):
            delta = torch.zeros_like(positions_t)
            delta[atom, axis] = step
            plus = model(positions_t.detach() + delta, atom_types)
            minus = model(positions_t.detach() - delta, atom_types)
            numerical_forces[atom, axis] = -(plus - minus) / (2.0 * step)
    coordinate_fd_error = torch.max(
        torch.abs(direct_forces - numerical_forces)
    ).item()
    if coordinate_fd_error > 2.0e-6:
        raise RuntimeError(
            "Mixed-l analytic force and coordinate finite difference disagree "
            f"by {coordinate_fd_error:.6e}."
        )
    reference = {
        "schema": "ye3t_lifted_cauchy_native_fixture_v2",
        "description": "Validation-only mixed-l nontrivial-internal fixture; not a fitted potential.",
        "descriptor_count": len(labels),
        "nontrivial_internal_descriptor_count": len(nontrivial_labels),
        "density_adjoint_max_abs_error": density_adjoint_error,
        "source_vjp_max_abs_error": source_vjp_error,
        "source_vjp_channel_max_abs_errors": channel_errors,
        "coordinate_force_fd_max_abs_error_eV_per_A": coordinate_fd_error,
        "positions_A": positions.tolist(),
        "atom_types": atom_types.tolist(),
        "total_energy_eV": float(atomic_energies.sum().detach()),
        "atomic_energies_eV": atomic_energies.detach().tolist(),
        "canonical_source_values_q": _packed(density.detach(), blocks).tolist(),
        "canonical_source_adjoints_dE_dq": _packed(
            source_adjoint.detach(), blocks
        ).tolist(),
        "directed_edge_centers": edge_index[0].detach().tolist(),
        "directed_edge_neighbors": edge_index[1].detach().tolist(),
        "directed_edge_displacements_A": context["displacements"].detach().tolist(),
        "directed_edge_gradients_dE_dd_eV_per_A": edge_gradients.detach().tolist(),
        "forces_eV_per_A": direct_forces.detach().tolist(),
        "strain_derivative_eV": direct_pullback[
            "strain_derivative"
        ].detach().tolist(),
        "lammps_global_virial_eV": (
            -direct_pullback["strain_derivative"].detach()
        ).tolist(),
    }
    (args.output / "reference.json").write_text(
        json.dumps(reference, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
