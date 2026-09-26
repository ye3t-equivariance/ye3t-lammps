#!/usr/bin/env python3
"""Validate lifted PairYE3T MPI, two-element, and bounded-NVE fixtures."""

import argparse
import json
import math
from pathlib import Path

from test_lifted_cauchy_lammps import (
    ENERGY_ATOL,
    METAL_NKTV2P,
    expected_per_atom_virial,
    marker,
    read_dump,
    reference_virial6,
    require_close,
)


def one_frame(path):
    frames = read_dump(path)
    if len(frames) != 1:
        raise AssertionError(f"{path}: expected one frame, found {len(frames)}")
    return frames[0][1]


def compare_records(left, right, label, ignored=("id", "type", "proc")):
    if left.keys() != right.keys():
        raise AssertionError(f"{label}: atom IDs differ")
    maxima = {
        "coordinate_A": 0.0,
        "velocity_A_per_ps": 0.0,
        "atomic_energy_eV": 0.0,
        "force_eV_per_A": 0.0,
        "per_atom_virial_eV": 0.0,
    }
    for atom_id in left:
        if left[atom_id].keys() != right[atom_id].keys():
            raise AssertionError(f"{label}: atom {atom_id} fields differ")
        for field in left[atom_id]:
            if field in ignored:
                continue
            actual = left[atom_id][field]
            expected = right[atom_id][field]
            if field.startswith("c_atom_stress["):
                actual /= METAL_NKTV2P
                expected /= METAL_NKTV2P
            require_close(
                actual,
                expected,
                f"{label} atom {atom_id} {field}",
                5.0e-8,
                5.0e-9,
            )
            difference = abs(actual - expected)
            if field in {"x", "y", "z"}:
                key = "coordinate_A"
            elif field in {"vx", "vy", "vz"}:
                key = "velocity_A_per_ps"
            elif field == "c_atom_energy":
                key = "atomic_energy_eV"
            elif field in {"fx", "fy", "fz"}:
                key = "force_eV_per_A"
            elif field.startswith("c_atom_stress["):
                key = "per_atom_virial_eV"
            else:
                continue
            maxima[key] = max(maxima[key], difference)
    return maxima


def validate_multielement(output, reference):
    expected_vatom = expected_per_atom_virial(reference)
    expected_global = reference_virial6(reference)
    names = ("XX", "YY", "ZZ", "XY", "XZ", "YZ")
    variants = (
        "normal.direct.rank1",
        "normal.direct.rank2",
        "normal.factorized.rank1",
        "normal.factorized.rank2",
        "reversed.direct.rank1",
        "reversed.factorized.rank1",
    )
    snapshots = {}
    maxima = {}
    for variant in variants:
        records = one_frame(output / f"multi.{variant}.dump")
        snapshots[variant] = records
        maximum_energy = 0.0
        maximum_force = 0.0
        maximum_vatom = 0.0
        for atom_id, expected_energy, expected_force, atom_virial in zip(
            range(1, len(records) + 1),
            reference["atomic_energies_eV"],
            reference["forces_eV_per_A"],
            expected_vatom,
        ):
            record = records[atom_id]
            require_close(
                record["c_atom_energy"],
                expected_energy,
                f"{variant} atom {atom_id} energy",
                ENERGY_ATOL,
                5.0e-11,
            )
            maximum_energy = max(
                maximum_energy, abs(record["c_atom_energy"] - expected_energy)
            )
            for component, field in enumerate(("fx", "fy", "fz")):
                require_close(
                    record[field],
                    expected_force[component],
                    f"{variant} atom {atom_id} {field}",
                )
                maximum_force = max(
                    maximum_force, abs(record[field] - expected_force[component])
                )
            for component in range(6):
                actual = -record[f"c_atom_stress[{component + 1}]"] / METAL_NKTV2P
                require_close(
                    actual,
                    atom_virial[component],
                    f"{variant} atom {atom_id} virial {component}",
                )
                maximum_vatom = max(maximum_vatom, abs(actual - atom_virial[component]))
        log_path = output / f"log.multi.{variant}"
        require_close(
            marker(log_path, "YE3T_LIFTED_MULTI_PE"),
            reference["total_energy_eV"],
            f"{variant} total energy",
            ENERGY_ATOL,
            5.0e-11,
        )
        for component, name in enumerate(names):
            require_close(
                marker(log_path, f"YE3T_LIFTED_MULTI_VIRIAL_{name}"),
                expected_global[component],
                f"{variant} global virial {name}",
            )
        net_force = [
            sum(record[field] for record in records.values())
            for field in ("fx", "fy", "fz")
        ]
        if math.sqrt(sum(value * value for value in net_force)) > 2.0e-10:
            raise AssertionError(f"{variant}: nonzero net force {net_force}")
        maxima[variant] = {
            "atomic_energy_max_abs_eV": maximum_energy,
            "force_max_abs_eV_per_A": maximum_force,
            "per_atom_virial_max_abs_eV": maximum_vatom,
        }

    normal_types = [snapshots["normal.direct.rank1"][index]["type"] for index in range(1, 7)]
    reversed_types = [snapshots["reversed.direct.rank1"][index]["type"] for index in range(1, 7)]
    if normal_types != [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]:
        raise AssertionError("Normal LAMMPS type assignment is wrong.")
    if reversed_types != [2.0, 1.0, 2.0, 1.0, 2.0, 1.0]:
        raise AssertionError("Reversed LAMMPS type assignment is wrong.")

    parity = {}
    anchor = snapshots["normal.direct.rank1"]
    for variant in variants[1:]:
        parity[variant] = compare_records(anchor, snapshots[variant], variant)
    if not reference["cross_channel_descriptor_indices"]:
        raise AssertionError("The reference has no cross-channel descriptor.")
    if min(reference["response_norms_eV_per_A"].values()) <= 1.0e-10:
        raise AssertionError("The reference has a vacuous chemical response class.")
    for row in reference["source_group_norms"]:
        if min(row) <= 1.0e-8:
            raise AssertionError("The reference has a vacuous center/source group.")
    return {"reference_maxima": maxima, "parity_max_abs": parity}


def validate_migration(output):
    variants = (
        "direct.rank1",
        "direct.rank2",
        "direct.rank4",
        "factorized.rank1",
        "factorized.rank2",
        "factorized.rank4",
    )
    snapshots = {}
    migration = {}
    for variant in variants:
        initial = one_frame(output / f"migration.{variant}.initial.dump")
        final = one_frame(output / f"migration.{variant}.final.dump")
        snapshots[variant] = (initial, final)
        ranks = int(variant.rsplit("rank", 1)[1])
        processes = {int(record["proc"]) for record in initial.values()}
        if ranks > 1 and len(processes) != ranks:
            raise AssertionError(
                f"{variant}: only {len(processes)} of {ranks} ranks own atoms"
            )
        changed = [
            atom_id
            for atom_id in initial
            if initial[atom_id]["proc"] != final[atom_id]["proc"]
        ]
        if ranks > 1 and not changed:
            raise AssertionError(f"{variant}: no atom migrated between MPI domains")
        force_norm = math.sqrt(
            sum(
                record[field] * record[field]
                for record in final.values()
                for field in ("fx", "fy", "fz")
            )
        )
        if force_norm <= 1.0e-10:
            raise AssertionError(f"{variant}: force response is vacuous")
        net_force = [
            sum(record[field] for record in final.values())
            for field in ("fx", "fy", "fz")
        ]
        if math.sqrt(sum(value * value for value in net_force)) > 2.0e-8:
            raise AssertionError(f"{variant}: nonzero net force {net_force}")
        migration[variant] = changed

    parity = {}
    anchor_initial, anchor_final = snapshots["direct.rank1"]
    for variant in variants[1:]:
        initial, final = snapshots[variant]
        parity[f"{variant}.initial"] = compare_records(
            anchor_initial, initial, f"{variant} initial"
        )
        parity[f"{variant}.final"] = compare_records(
            anchor_final, final, f"{variant} final"
        )
    return {"migrating_atom_ids": migration, "parity_max_abs": parity}


def minimum_periodic_distance(records, box_length=20.0):
    atom_ids = sorted(records)
    minimum = math.inf
    for position, left_id in enumerate(atom_ids):
        left = records[left_id]
        for right_id in atom_ids[position + 1 :]:
            right = records[right_id]
            displacement = []
            for field in ("x", "y", "z"):
                value = right[field] - left[field]
                value -= round(value / box_length) * box_length
                displacement.append(value)
            minimum = min(minimum, math.sqrt(sum(value * value for value in displacement)))
    return minimum


def validate_nve(output):
    variants = (
        "factorized.rank1.dt",
        "factorized.rank4.dt",
        "direct.rank1.dt",
        "factorized.rank1.half_dt",
    )
    trajectories = {}
    drift = {}
    for variant in variants:
        frames = read_dump(output / f"nve.{variant}.dump")
        if len(frames) < 2:
            raise AssertionError(f"{variant}: expected an NVE trajectory")
        atom_ids = frames[0][1].keys()
        for timestep, records in frames:
            if records.keys() != atom_ids:
                raise AssertionError(f"{variant} step {timestep}: atom IDs changed")
            for record in records.values():
                for value in record.values():
                    if not math.isfinite(value):
                        raise AssertionError(f"{variant}: nonfinite trajectory value")
            if minimum_periodic_distance(records) < 1.0:
                raise AssertionError(f"{variant}: atoms approached too closely")
        trajectories[variant] = frames
        log_path = output / f"log.nve.{variant}"
        initial = marker(log_path, "YE3T_LIFTED_NVE_INITIAL_ETOTAL")
        final = marker(log_path, "YE3T_LIFTED_NVE_FINAL_ETOTAL")
        drift[variant] = abs(final - initial)
        if drift[variant] > 5.0e-8:
            raise AssertionError(f"{variant}: excessive bounded-NVE drift {drift[variant]:.3e}")

    anchor = trajectories["factorized.rank1.dt"][-1][1]
    parity = {
        "factorized.rank4.dt": compare_records(
            anchor,
            trajectories["factorized.rank4.dt"][-1][1],
            "NVE rank1/rank4",
        ),
        "direct.rank1.dt": compare_records(
            anchor,
            trajectories["direct.rank1.dt"][-1][1],
            "NVE direct/factorized",
        ),
    }
    if drift["factorized.rank1.half_dt"] > max(
        1.2 * drift["factorized.rank1.dt"], 5.0e-12
    ):
        raise AssertionError("Half-timestep NVE drift did not remain bounded.")
    return {"absolute_energy_drift_eV": drift, "final_state_parity_max_abs": parity}


def validate_restart(output):
    parity = {}
    rank_results = {}
    for ranks in (1, 4):
        continuous = one_frame(output / f"restart.rank{ranks}.continuous.dump")
        rehydrated = one_frame(output / f"restart.rank{ranks}.rehydrated.dump")
        parity[f"rank{ranks}"] = compare_records(
            continuous,
            rehydrated,
            f"restart rank{ranks}",
        )
        rank_results[ranks] = continuous
    parity["rank1_vs_rank4"] = compare_records(
        rank_results[1],
        rank_results[4],
        "restart rank1/rank4",
    )
    return {"final_state_parity_max_abs": parity}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("output")
    args = parser.parse_args()
    output = Path(args.output)
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    result = {
        "passed": True,
        "multielement": validate_multielement(output, reference),
        "migration": validate_migration(output),
        "bounded_nve": validate_nve(output),
        "restart_rehydration": validate_restart(output),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
