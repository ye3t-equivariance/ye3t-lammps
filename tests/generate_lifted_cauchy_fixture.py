"""Regenerate the deterministic lifted-Cauchy native-runtime fixture.

Requires installed ``ye3t`` and ``ye3t-methods``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ase import Atoms

from ye3t.couplings import compile as compile_coupling
from ye3t.couplings import first_lifted_cauchy_scalar_request
from ye3t_methods.atomistic import YE3TDescriptors, export_lifted_cauchy_linear_bundle
from ye3t_methods.atomistic.lifted_cauchy_linear import (
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
    compiled = compile_coupling(request)
    descriptor = YE3TDescriptors.ye3t_basis(
        {
            "elements": ("Ta",),
            "type_map": {"Ta": 0},
            "lifted_cauchy": {
                "compiled": compiled.to_dict(),
                "source": {
                    "schema": "ye3t_lifted_cauchy_joint_source_v2",
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
                        -0.7, 1.1, len(compiled.payload["descriptors"])
                    )
                ),
            ),
            "offsets": (0.2,),
            "realization": "factored",
            "source_realization": "factorized",
        },
    )
    model.fit_metadata = {
        "purpose": "deterministic_native_correctness_fixture",
        "training": "not_fitted",
        "publication_timing": False,
    }

    export_lifted_cauchy_linear_bundle(model, args.output)

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
    atom_types = torch.zeros(len(positions), dtype=torch.long)
    positions_t = torch.as_tensor(positions, dtype=torch.float64)
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
    atomic_energies = (features * model.coefficients[0]).sum(dim=1) + model.offsets[0]
    edge_index = context["edge_index"]
    edge_adjoint = source_adjoint.index_select(0, edge_index[0])
    edge_gradients = torch.einsum(
        "ecsq,ecsqd->ed", edge_adjoint, context["edge_dx"]
    )
    reference = {
        "schema": "ye3t_lifted_cauchy_native_fixture_v1",
        "description": "Validation-only nontrivial-internal-kappa Ta fixture; not a fitted potential.",
        "positions_A": positions.tolist(),
        "atom_types": atom_types.tolist(),
        "total_energy_eV": float(atomic_energies.sum().detach()),
        "atomic_energies_eV": atomic_energies.detach().tolist(),
        "canonical_source_values_q": (
            density.detach().reshape(len(positions), -1).tolist()
        ),
        "canonical_source_adjoints_dE_dq": (
            source_adjoint.detach().reshape(len(positions), -1).tolist()
        ),
        "directed_edge_centers": edge_index[0].detach().tolist(),
        "directed_edge_neighbors": edge_index[1].detach().tolist(),
        "directed_edge_displacements_A": (
            context["displacements"].detach().tolist()
        ),
        "directed_edge_gradients_dE_dd_eV_per_A": (
            edge_gradients.detach().tolist()
        ),
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
