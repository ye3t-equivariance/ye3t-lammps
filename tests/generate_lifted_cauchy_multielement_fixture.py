"""Regenerate the deterministic two-element lifted-Cauchy fixture.

Requires installed ``ye3t`` and ``ye3t-ace`` (not yet public).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ase import Atoms

from ye3t.couplings import compile as compile_coupling
from ye3t.couplings import first_lifted_cauchy_scalar_request
from ye3t_ace import YE3TDescriptors, export_lifted_cauchy_linear_bundle
from ye3t_ace.lifted_cauchy_linear import (
    LIFTED_CAUCHY_JOINT_SOURCE_FAMILY,
    lifted_cauchy_model_from_descriptor,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    request = first_lifted_cauchy_scalar_request(
        2,
        source_family_id=LIFTED_CAUCHY_JOINT_SOURCE_FAMILY,
        family_ids=(
            "NT_NU4_K22_L0",
            "NT_NU2_MU2_SIGN_L1x1",
        ),
    )
    request["channels"] = (
        {
            "channel_id": 0,
            "neighbor_species": "Ta",
            "radial_channel": 0,
            "l": 1,
            "source_family_id": LIFTED_CAUCHY_JOINT_SOURCE_FAMILY,
        },
        {
            "channel_id": 1,
            "neighbor_species": "W",
            "radial_channel": 0,
            "l": 1,
            "source_family_id": LIFTED_CAUCHY_JOINT_SOURCE_FAMILY,
        },
    )
    compiled = compile_coupling(request)
    cross_descriptors = tuple(
        descriptor
        for descriptor in compiled.payload["descriptors"]
        if set(descriptor["label"]["block_channel_indices"]) == {0, 1}
    )
    if not cross_descriptors:
        raise RuntimeError("The two-element fixture has no cross-channel descriptor.")

    descriptor = YE3TDescriptors.ye3t_basis(
        {
            "elements": ("Ta", "W"),
            "type_map": {"Ta": 2, "W": 0},
            "lifted_cauchy": {
                "compiled": compiled.to_dict(),
                "source": {
                    "schema": "ye3t_lifted_cauchy_joint_source_v2",
                    "cutoff_A": 5.2,
                    "periodic_image_mode": "all_images",
                    "neighbor_backend": "ase",
                },
            },
        }
    )
    descriptor_count = len(compiled.payload["descriptors"])
    coefficients = np.stack(
        (
            np.linspace(-0.85, 1.05, descriptor_count),
            np.linspace(0.65, -1.15, descriptor_count),
        )
    )
    model = lifted_cauchy_model_from_descriptor(
        descriptor,
        {
            "coefficients": coefficients,
            "offsets": (0.31, -0.27),
            "realization": "factored",
            "source_realization": "factorized",
        },
    )
    model.fit_metadata = {
        "purpose": "deterministic_multielement_native_correctness_fixture",
        "training": "not_fitted",
        "publication_timing": False,
    }
    export_lifted_cauchy_linear_bundle(model, args.output)

    positions = np.asarray(
        (
            (5.00, 5.00, 5.00),
            (5.28, 7.52, 5.28),
            (7.24, 5.28, 4.58),
            (6.82, 4.16, 7.10),
            (4.20, 6.10, 7.70),
            (8.60, 7.80, 6.10),
        ),
        dtype=float,
    )
    atoms = Atoms(
        ("Ta", "W", "Ta", "W", "Ta", "W"),
        positions=positions,
        cell=np.diag((12.0, 12.0, 12.0)),
        pbc=True,
    )
    neighbor_data = model.source.neighbor_data(atoms)
    positions_t = torch.as_tensor(neighbor_data.positions, dtype=torch.float64)
    atom_types = torch.as_tensor(neighbor_data.atom_types, dtype=torch.long)
    edge_index = torch.as_tensor(neighbor_data.edge_index, dtype=torch.long)
    cell = torch.as_tensor(neighbor_data.cell, dtype=torch.float64)
    shifts = torch.as_tensor(neighbor_data.shifts, dtype=torch.float64)
    density, context = model.source.materialize(
        positions_t,
        atom_types,
        edge_index=edge_index,
        cell=cell,
        shifts=shifts,
        pbc=atoms.pbc,
        source_realization="direct",
    )
    features = model.evaluator.evaluate(density, realization="canonical")
    type_positions = torch.empty_like(atom_types)
    for atom_type, position in model.type_position.items():
        type_positions[atom_types == atom_type] = position
    selected_coefficients = model.coefficients.index_select(0, type_positions)
    selected_offsets = model.offsets.index_select(0, type_positions)
    features_vjp, source_adjoint = model.evaluator.vjp(
        density,
        selected_coefficients,
        realization="canonical",
        method="explicit",
    )
    if not torch.allclose(features, features_vjp, atol=2.0e-13, rtol=2.0e-13):
        raise RuntimeError("Canonical value and VJP feature paths disagree.")
    atomic_energies = (
        features * selected_coefficients
    ).sum(dim=1) + selected_offsets
    counterfactual = features @ model.coefficients.T + model.offsets
    if torch.max(torch.abs(counterfactual[:, 0] - counterfactual[:, 1])).item() < 1.0e-6:
        raise RuntimeError("The two central-species readout heads are vacuous.")

    group_norms = torch.linalg.vector_norm(density, dim=(2, 3))
    if torch.min(group_norms).item() < 1.0e-8:
        raise RuntimeError("Every center must receive nonzero Ta and W source blocks.")
    edge_values = context["edge_values"]
    edge_types = atom_types.index_select(0, edge_index[1])
    for edge, neighbor_type in enumerate(edge_types.tolist()):
        expected_channel = 0 if neighbor_type == model.type_map["Ta"] else 1
        wrong_channel = 1 - expected_channel
        if torch.count_nonzero(edge_values[edge, expected_channel]).item() == 0:
            raise RuntimeError("An active species source block vanished.")
        if torch.count_nonzero(edge_values[edge, wrong_channel]).item() != 0:
            raise RuntimeError("A wrong-species source block is nonzero.")

    edge_adjoint = source_adjoint.index_select(0, edge_index[0])
    edge_gradients = torch.einsum(
        "ecsq,ecsqd->ed", edge_adjoint, context["edge_dx"]
    )
    response_norms = {}
    for center_species, center_type in model.type_map.items():
        for neighbor_species, neighbor_type in model.type_map.items():
            mask = (atom_types.index_select(0, edge_index[0]) == center_type) & (
                edge_types == neighbor_type
            )
            value = torch.linalg.vector_norm(edge_gradients[mask]).item()
            response_norms[f"{center_species}_{neighbor_species}"] = value
            if value < 1.0e-10:
                raise RuntimeError(
                    f"The {center_species}-{neighbor_species} response is vacuous."
                )

    result = model.evaluate_atoms(atoms, forces=True, stress=True)
    reference = {
        "schema": "ye3t_lifted_cauchy_multielement_native_fixture_v1",
        "description": (
            "Validation-only nontrivial-internal-kappa Ta/W semantic fixture; "
            "not a fitted potential."
        ),
        "cell_A": np.asarray(atoms.cell).tolist(),
        "pbc": [True, True, True],
        "symbols": list(atoms.get_chemical_symbols()),
        "training_type_map": dict(model.type_map),
        "central_species_order": list(model.central_species_order),
        "positions_A": positions.tolist(),
        "atom_types": atom_types.tolist(),
        "descriptor_count": descriptor_count,
        "cross_channel_descriptor_indices": [
            int(record["descriptor_index"]) for record in cross_descriptors
        ],
        "source_group_norms": group_norms.detach().tolist(),
        "response_norms_eV_per_A": response_norms,
        "counterfactual_atomic_energies_eV": counterfactual.detach().tolist(),
        "total_energy_eV": float(atomic_energies.sum().detach()),
        "atomic_energies_eV": atomic_energies.detach().tolist(),
        "canonical_source_values_q": density.detach().reshape(len(atoms), -1).tolist(),
        "canonical_source_adjoints_dE_dq": (
            source_adjoint.detach().reshape(len(atoms), -1).tolist()
        ),
        "directed_edge_centers": edge_index[0].detach().tolist(),
        "directed_edge_neighbors": edge_index[1].detach().tolist(),
        "directed_edge_displacements_A": context["displacements"].detach().tolist(),
        "directed_edge_gradients_dE_dd_eV_per_A": edge_gradients.detach().tolist(),
        "forces_eV_per_A": result["forces"].detach().tolist(),
        "strain_derivative_eV": result["strain_derivative"].detach().tolist(),
        "lammps_global_virial_eV": (
            -result["strain_derivative"].detach()
        ).tolist(),
    }
    (args.output / "reference.json").write_text(
        json.dumps(reference, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
