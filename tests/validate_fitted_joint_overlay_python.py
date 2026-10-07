#!/usr/bin/env python3
"""Compare a fitted joint-overlay LAMMPS snapshot with YE3T-ACE."""

import argparse
import json
from pathlib import Path

import numpy as np
from ase import Atoms

from ye3t_methods.atomistic import load_linear_ace_calculator
from ye3t_methods.atomistic import load_lifted_cauchy_linear_bundle


METAL_NKTV2P = 1.6021765e6


def _read_snapshot(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    box_line = lines.index(
        next(line for line in lines if line.startswith("ITEM: BOX"))
    )
    bounds = [
        tuple(map(float, lines[box_line + offset].split()[:2]))
        for offset in (1, 2, 3)
    ]
    atom_line = lines.index(
        next(line for line in lines if line.startswith("ITEM: ATOMS"))
    )
    fields = lines[atom_line].split()[2:]
    records = []
    for line in lines[atom_line + 1 :]:
        if line.startswith("ITEM:"):
            break
        records.append(dict(zip(fields, map(float, line.split()))))
    records.sort(key=lambda row: int(row["id"]))
    cell = np.diag([upper - lower for lower, upper in bounds])
    positions = np.asarray(
        [[row[key] for key in ("x", "y", "z")] for row in records]
    )
    return Atoms("Ta" * len(records), positions=positions, cell=cell, pbc=True), records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--ordinary", type=Path)
    parser.add_argument("--lifted", type=Path, required=True)
    parser.add_argument("--reference-snapshot", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    atoms, records = _read_snapshot(args.snapshot)
    if args.reference_snapshot is not None:
        reference_atoms, reference_records = _read_snapshot(
            args.reference_snapshot
        )
        if len(reference_atoms) != len(atoms) or not np.allclose(
            reference_atoms.positions, atoms.positions, atol=1.0e-12, rtol=0.0
        ):
            raise AssertionError("Reference and total LAMMPS snapshots differ in geometry.")
        for total, reference in zip(records, reference_records):
            if int(total["id"]) != int(reference["id"]):
                raise AssertionError("Reference and total snapshots differ in atom IDs.")
            for field in ("c_atom_energy", "fx", "fy", "fz"):
                total[field] -= reference[field]
            for index in range(1, 7):
                field = f"c_atom_stress[{index}]"
                total[field] -= reference[field]
    ordinary_energy = 0.0
    ordinary_forces = np.zeros((len(atoms), 3), dtype=np.float64)
    ordinary_strain = np.zeros((3, 3), dtype=np.float64)
    if args.ordinary is not None:
        ordinary = load_linear_ace_calculator(
            args.ordinary,
            force_method="autograd",
            backend="pytorch",
            strict_backend=True,
            validate_backend=True,
            factorized_descriptor_runtime_policy="auto",
        )
        ordinary_atoms = atoms.copy()
        ordinary_atoms.calc = ordinary
        ordinary_energy = float(ordinary_atoms.get_potential_energy())
        ordinary_forces = np.asarray(
            ordinary_atoms.get_forces(), dtype=np.float64
        )
        ordinary_strain = (
            np.asarray(ordinary_atoms.get_stress(voigt=False), dtype=np.float64)
            * atoms.get_volume()
        )

    lifted = load_lifted_cauchy_linear_bundle(args.lifted)["lifted_model"]
    lifted_result = lifted.evaluate_atoms(atoms, forces=True, stress=True)
    python_energy = ordinary_energy + float(lifted_result["energy"].detach())
    python_forces = ordinary_forces + lifted_result["forces"].detach().cpu().numpy()
    python_virial = -(
        ordinary_strain
        + lifted_result["strain_derivative"].detach().cpu().numpy()
    )

    lammps_energy = sum(row["c_atom_energy"] for row in records)
    lammps_forces = np.asarray(
        [[row[key] for key in ("fx", "fy", "fz")] for row in records]
    )
    lammps_virial = np.asarray(
        [
            -sum(row[f"c_atom_stress[{index}]"] for row in records)
            / METAL_NKTV2P
            for index in range(1, 7)
        ]
    )
    python_virial6 = np.asarray(
        (
            python_virial[0, 0],
            python_virial[1, 1],
            python_virial[2, 2],
            python_virial[0, 1],
            python_virial[0, 2],
            python_virial[1, 2],
        )
    )
    report = {
        "schema": "ye3t_python_lammps_joint_overlay_parity_v1",
        "atom_count": len(atoms),
        "ordinary_component_included": args.ordinary is not None,
        "reference_snapshot_subtracted": args.reference_snapshot is not None,
        "energy_absolute_error_eV": abs(python_energy - lammps_energy),
        "force_max_absolute_error_eV_per_A": float(
            np.max(np.abs(python_forces - lammps_forces))
        ),
        "virial_max_absolute_error_eV": float(
            np.max(np.abs(python_virial6 - lammps_virial))
        ),
    }
    limits = {
        "energy_absolute_error_eV": 2.0e-8,
        "force_max_absolute_error_eV_per_A": 2.0e-7,
        "virial_max_absolute_error_eV": 2.0e-6,
    }
    for key, limit in limits.items():
        if not np.isfinite(report[key]) or report[key] > limit:
            raise AssertionError(f"{key}={report[key]:.3e} exceeds {limit:.3e}")
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.json.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
