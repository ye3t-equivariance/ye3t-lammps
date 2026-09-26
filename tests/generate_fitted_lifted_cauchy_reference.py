#!/usr/bin/env python3
"""Generate an independent Python reference for the fitted Ta CPU fixture.

Requires installed ``ye3t`` and ``ye3t-ace`` (not yet public).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ase import Atoms

from ye3t_ace import load_lifted_cauchy_linear_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    loaded = load_lifted_cauchy_linear_bundle(
        args.bundle,
        realization="canonical",
        source_realization="direct",
    )
    model = loaded["lifted_model"]
    if tuple(model.central_species_order) != ("Ta",):
        raise ValueError("The fitted Ta fixture requires one Ta readout head.")

    positions = np.asarray(
        (
            (0.3, 0.8, 0.6),
            (2.1, 0.9, 0.4),
            (19.9, 2.8, 0.9),
            (0.5, 0.1, 2.7),
        ),
        dtype=float,
    )
    atoms = Atoms(
        "Ta4",
        positions=positions,
        cell=np.eye(3) * 20.0,
        pbc=True,
    )
    result = model.evaluate_atoms(atoms, forces=True, stress=True)

    neighbor_data = model.source.neighbor_data(atoms)
    positions_t = torch.as_tensor(neighbor_data.positions, dtype=torch.float64)
    cell_t = torch.as_tensor(neighbor_data.cell, dtype=torch.float64)
    atom_types = torch.as_tensor(neighbor_data.atom_types, dtype=torch.long)
    edge_index = torch.as_tensor(neighbor_data.edge_index, dtype=torch.long)
    shifts = torch.as_tensor(neighbor_data.shifts, dtype=torch.float64)
    density, context = model.source.materialize(
        positions_t,
        atom_types,
        edge_index=edge_index,
        cell=cell_t,
        shifts=shifts,
        pbc=atoms.pbc,
        source_realization="direct",
    )
    weights = model.coefficients[0].to(density).expand(len(atoms), -1)
    features, source_adjoint = model.evaluator.vjp(
        density,
        weights,
        realization="canonical",
        method="explicit",
    )
    atomic_energies = (features * weights).sum(dim=1) + model.offsets[0]
    edge_adjoint = source_adjoint.index_select(0, edge_index[0])
    edge_gradients = torch.einsum(
        "ecsq,ecsqd->ed", edge_adjoint, context["edge_dx"]
    )

    reference = {
        "schema": "ye3t_fitted_lifted_cauchy_reference_v1",
        "description": (
            "Canonical compiler/Python reference for the fitted periodic Ta CPU "
            "fixture; the additive ZBL reference is intentionally excluded."
        ),
        "bundle_model_self_hash": loaded["model_payload"]["self_hash"],
        "composite_artifact_hash": model.artifact_hash,
        "realization": "canonical",
        "source_realization": "direct",
        "positions_A": positions.tolist(),
        "cell_A": atoms.cell.array.tolist(),
        "pbc": atoms.pbc.tolist(),
        "atom_types": atom_types.tolist(),
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(reference, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
