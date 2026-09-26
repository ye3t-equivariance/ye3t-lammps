#!/usr/bin/env python3
"""Verify the deterministic lifted-Cauchy fixture through one-rank LAMMPS."""

import argparse
import json
import math
from pathlib import Path
import re


METAL_NKTV2P = 1.6021765e6
ENERGY_ATOL = 5.0e-10
VALUE_ATOL = 5.0e-8
VALUE_RTOL = 5.0e-9
FORCE_FD_ATOL = 2.0e-5
VIRIAL_FD_ATOL_EV = 2.0e-5


def read_dump(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    frames = []
    cursor = 0
    while cursor < len(lines):
        if lines[cursor] != "ITEM: TIMESTEP":
            raise ValueError(f"{path}: expected timestep header at line {cursor + 1}")
        timestep = int(lines[cursor + 1])
        cursor += 2
        if lines[cursor] != "ITEM: NUMBER OF ATOMS":
            raise ValueError(f"{path}: missing atom-count header")
        count = int(lines[cursor + 1])
        cursor += 2
        if not lines[cursor].startswith("ITEM: BOX BOUNDS"):
            raise ValueError(f"{path}: missing box header")
        cursor += 4
        if not lines[cursor].startswith("ITEM: ATOMS "):
            raise ValueError(f"{path}: missing atom header")
        fields = lines[cursor].split()[2:]
        cursor += 1
        records = {}
        for _ in range(count):
            values = lines[cursor].split()
            cursor += 1
            if len(values) != len(fields):
                raise ValueError(f"{path}: malformed atom record")
            record = dict(zip(fields, map(float, values)))
            atom_id = int(record["id"])
            if atom_id in records:
                raise ValueError(f"{path}: duplicate atom ID {atom_id}")
            records[atom_id] = record
        frames.append((timestep, records))
    if not frames:
        raise ValueError(f"{path}: no dump frames")
    return frames


def marker(path, name):
    text = Path(path).read_text(encoding="utf-8")
    values = re.findall(rf"{re.escape(name)}=([-+0-9.eE]+)", text)
    if len(values) != 1:
        raise ValueError(f"{path}: expected one {name} marker, found {len(values)}")
    return float(values[0])


def require_close(actual, expected, label, atol=VALUE_ATOL, rtol=VALUE_RTOL):
    tolerance = atol + rtol * max(abs(actual), abs(expected))
    if not math.isfinite(actual) or abs(actual - expected) > tolerance:
        raise AssertionError(
            f"{label}: {actual:.17g} != {expected:.17g} "
            f"(absolute error {abs(actual - expected):.3e}, tolerance {tolerance:.3e})"
        )


def expected_per_atom_virial(reference):
    count = len(reference["positions_A"])
    result = [[0.0] * 6 for _ in range(count)]
    for center, neighbor, displacement, gradient in zip(
        reference["directed_edge_centers"],
        reference["directed_edge_neighbors"],
        reference["directed_edge_displacements_A"],
        reference["directed_edge_gradients_dE_dd_eV_per_A"],
    ):
        dx, dy, dz = displacement
        gx, gy, gz = gradient
        edge = (
            -dx * gx,
            -dy * gy,
            -dz * gz,
            -dx * gy,
            -dx * gz,
            -dy * gz,
        )
        for atom in (center, neighbor):
            for component, value in enumerate(edge):
                result[atom][component] += 0.5 * value
    return result


def reference_virial6(reference):
    matrix = reference["lammps_global_virial_eV"]
    return (matrix[0][0], matrix[1][1], matrix[2][2], matrix[0][1], matrix[0][2], matrix[1][2])


def validate_snapshot(path, log_path, reference, expected_vatom, policy):
    frames = read_dump(path)
    if len(frames) != 1:
        raise AssertionError(f"{path}: expected one snapshot frame")
    _, records = frames[0]
    if sorted(records) != list(range(1, len(reference["positions_A"]) + 1)):
        raise AssertionError(f"{path}: unexpected atom IDs")

    energy_sum = 0.0
    stress_sum = [0.0] * 6
    maximum_energy_error = 0.0
    maximum_force_error = 0.0
    maximum_atom_virial_error = 0.0
    for atom_id, expected_energy, expected_force, expected_atom_virial in zip(
        range(1, len(records) + 1),
        reference["atomic_energies_eV"],
        reference["forces_eV_per_A"],
        expected_vatom,
    ):
        record = records[atom_id]
        require_close(
            record["c_atom_energy"],
            expected_energy,
            f"{policy} atom {atom_id} energy",
            ENERGY_ATOL,
            5.0e-11,
        )
        maximum_energy_error = max(
            maximum_energy_error, abs(record["c_atom_energy"] - expected_energy)
        )
        energy_sum += record["c_atom_energy"]
        for component, field in enumerate(("fx", "fy", "fz")):
            require_close(
                record[field],
                expected_force[component],
                f"{policy} atom {atom_id} {field}",
            )
            maximum_force_error = max(
                maximum_force_error,
                abs(record[field] - expected_force[component]),
            )
        for component in range(6):
            field = f"c_atom_stress[{component + 1}]"
            atom_virial = -record[field] / METAL_NKTV2P
            stress_sum[component] += atom_virial
            require_close(
                atom_virial,
                expected_atom_virial[component],
                f"{policy} atom {atom_id} virial {component}",
            )
            maximum_atom_virial_error = max(
                maximum_atom_virial_error,
                abs(atom_virial - expected_atom_virial[component]),
            )

    require_close(
        energy_sum,
        reference["total_energy_eV"],
        f"{policy} sum atomic energy",
        ENERGY_ATOL,
        5.0e-11,
    )
    require_close(
        marker(log_path, "YE3T_LIFTED_PE"),
        reference["total_energy_eV"],
        f"{policy} total energy",
        ENERGY_ATOL,
        5.0e-11,
    )

    expected_global = reference_virial6(reference)
    names = ("XX", "YY", "ZZ", "XY", "XZ", "YZ")
    maximum_global_virial_error = 0.0
    for component, name in enumerate(names):
        require_close(
            stress_sum[component],
            expected_global[component],
            f"{policy} summed per-atom virial {name}",
        )
        global_virial = marker(log_path, f"YE3T_LIFTED_VIRIAL_{name}")
        require_close(
            global_virial,
            expected_global[component],
            f"{policy} global pressure virial {name}",
        )
        maximum_global_virial_error = max(
            maximum_global_virial_error,
            abs(global_virial - expected_global[component]),
        )

    force_fd = marker(log_path, "YE3T_LIFTED_NUMDIFF_FORCE_MAX_ABS")
    virial_fd_bar = marker(log_path, "YE3T_LIFTED_NUMDIFF_VIRIAL_L2_BAR")
    virial_fd_eV = marker(log_path, "YE3T_LIFTED_NUMDIFF_VIRIAL_L2_EV")
    if force_fd > FORCE_FD_ATOL:
        raise AssertionError(f"{policy} force finite-difference error {force_fd:.3e}")
    if virial_fd_eV > VIRIAL_FD_ATOL_EV:
        raise AssertionError(
            f"{policy} virial finite-difference error {virial_fd_eV:.3e} eV "
            f"({virial_fd_bar:.3e} bar)"
        )
    return records, {
        "atomic_energy_max_abs_eV": maximum_energy_error,
        "force_max_abs_eV_per_A": maximum_force_error,
        "per_atom_virial_max_abs_eV": maximum_atom_virial_error,
        "global_virial_max_abs_eV": maximum_global_virial_error,
        "force_finite_difference_max_abs_eV_per_A": force_fd,
        "virial_finite_difference_l2_bar": virial_fd_bar,
        "virial_finite_difference_l2_eV": virial_fd_eV,
    }


def validate_replay(path, snapshot, policy):
    frames = read_dump(path)
    if len(frames) < 5:
        raise AssertionError(f"{path}: expected at least five replay frames")
    fields = tuple(next(iter(snapshot.values())))
    for timestep, records in frames:
        if records.keys() != snapshot.keys():
            raise AssertionError(f"{policy} replay timestep {timestep}: atom IDs differ")
        for atom_id in records:
            for field in fields:
                if field in {"id", "type"}:
                    continue
                require_close(
                    records[atom_id][field],
                    snapshot[atom_id][field],
                    f"{policy} replay timestep {timestep} atom {atom_id} {field}",
                    5.0e-9,
                    5.0e-11,
                )


def validate_isolated(path, log_path, policy):
    frames = read_dump(path)
    if len(frames) != 1 or list(frames[0][1]) != [1]:
        raise AssertionError(f"{path}: expected one isolated atom")
    record = frames[0][1][1]
    require_close(record["c_atom_energy"], 0.2, f"{policy} isolated atom energy", 1.0e-14, 1.0e-14)
    require_close(marker(log_path, "YE3T_LIFTED_ISOLATED_PE"), 0.2, f"{policy} isolated total energy", 1.0e-14, 1.0e-14)
    for field in ("fx", "fy", "fz"):
        require_close(record[field], 0.0, f"{policy} isolated {field}", 1.0e-14, 0.0)
    for component in range(1, 7):
        require_close(
            record[f"c_atom_stress[{component}]"],
            0.0,
            f"{policy} isolated stress {component}",
            1.0e-14,
            0.0,
        )


def compare_policies(direct, factorized):
    maximum_difference = {
        "atomic_energy_max_abs_eV": 0.0,
        "coordinate_max_abs_A": 0.0,
        "force_max_abs_eV_per_A": 0.0,
        "per_atom_virial_max_abs_eV": 0.0,
    }
    for atom_id in direct:
        for field in direct[atom_id]:
            if field in {"id", "type"}:
                continue
            require_close(
                direct[atom_id][field],
                factorized[atom_id][field],
                f"direct/factorized atom {atom_id} {field}",
                5.0e-9,
                5.0e-11,
            )
            difference = abs(direct[atom_id][field] - factorized[atom_id][field])
            if field == "c_atom_energy":
                key = "atomic_energy_max_abs_eV"
            elif field in {"fx", "fy", "fz"}:
                key = "force_max_abs_eV_per_A"
            elif field.startswith("c_atom_stress["):
                key = "per_atom_virial_max_abs_eV"
                difference /= METAL_NKTV2P
            else:
                key = "coordinate_max_abs_A"
            maximum_difference[key] = max(maximum_difference[key], difference)
    return maximum_difference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("output")
    args = parser.parse_args()

    output = Path(args.output)
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    expected_vatom = expected_per_atom_virial(reference)
    results = {}
    snapshots = {}
    for policy in ("direct", "factorized", "model_default", "direct_unsorted"):
        snapshots[policy], results[policy] = validate_snapshot(
            output / f"{policy}.snapshot.dump",
            output / f"log.{policy}",
            reference,
            expected_vatom,
            policy,
        )
        validate_replay(output / f"{policy}.replay.dump", snapshots[policy], policy)
        if policy in {"direct", "factorized"}:
            validate_isolated(
                output / f"{policy}.isolated.dump",
                output / f"log.{policy}.isolated",
                policy,
            )
    parity = {
        "direct_vs_factorized_max_abs": compare_policies(
            snapshots["direct"], snapshots["factorized"]
        ),
        "factorized_vs_model_default_max_abs": compare_policies(
            snapshots["factorized"], snapshots["model_default"]
        ),
        "direct_vs_unsorted_max_abs": compare_policies(
            snapshots["direct"], snapshots["direct_unsorted"]
        ),
    }
    print(
        json.dumps(
            {"passed": True, "parity": parity, "policies": results},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
