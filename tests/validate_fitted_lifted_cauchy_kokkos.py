#!/usr/bin/env python3
"""Validate a fitted lifted-Cauchy bundle across CPU and Kokkos execution."""

import argparse
import json
import math
from pathlib import Path

from test_lifted_cauchy_lammps import marker, read_dump
from test_lifted_cauchy_mpi import compare_records, one_frame


def require_below(value, limit, label):
    if not math.isfinite(value) or value > limit:
        raise AssertionError(f"{label}: {value:.17g} exceeds {limit:.17g}")


def compare_series(left_path, right_path, label):
    left = read_dump(left_path)
    right = read_dump(right_path)
    if len(left) != len(right) or [step for step, _ in left] != [
        step for step, _ in right
    ]:
        raise AssertionError(f"{label}: dump timesteps differ")
    maxima = {}
    for (step, left_records), (_, right_records) in zip(left, right, strict=True):
        current = compare_records(
            left_records, right_records, f"{label} timestep {step}"
        )
        for key, value in current.items():
            maxima[key] = max(maxima.get(key, 0.0), value)
    return maxima


def validate_numdiff(output):
    results = {}
    for backend in ("cpu", "kk"):
        log = output / f"log.{backend}.numdiff"
        force_error = marker(log, "YE3T_LIFTED_NUMDIFF_FORCE_MAX_ABS")
        virial_error = marker(log, "YE3T_LIFTED_NUMDIFF_VIRIAL_L2_EV")
        require_below(force_error, 2.0e-7, f"{backend} force finite difference")
        require_below(virial_error, 2.0e-6, f"{backend} virial finite difference")
        results[backend] = {
            "energy_eV": marker(log, "YE3T_LIFTED_PE"),
            "force_max_abs_eV_per_A": force_error,
            "virial_l2_eV": virial_error,
        }
    require_below(
        abs(results["cpu"]["energy_eV"] - results["kk"]["energy_eV"]),
        5.0e-10,
        "CPU/Kokkos total energy",
    )
    snapshot = compare_records(
        one_frame(output / "cpu.numdiff.snapshot.dump"),
        one_frame(output / "kk.numdiff.snapshot.dump"),
        "fitted CPU/Kokkos snapshot",
    )
    replay = compare_series(
        output / "cpu.numdiff.replay.dump",
        output / "kk.numdiff.replay.dump",
        "fitted CPU/Kokkos replay",
    )
    return {"finite_difference": results, "snapshot_max_abs": snapshot,
            "replay_max_abs": replay}


def validate_migration(output):
    frames = {}
    migrating = {}
    for ranks in (1, 2):
        label = f"kk.migration.rank{ranks}"
        initial = one_frame(output / f"{label}.initial.dump")
        final = one_frame(output / f"{label}.final.dump")
        frames[ranks] = (initial, final)
        owners = {int(record["proc"]) for record in initial.values()}
        if len(owners) != ranks:
            raise AssertionError(
                f"{label}: {len(owners)} MPI ranks own atoms, expected {ranks}"
            )
        changed = [
            atom_id
            for atom_id in initial
            if initial[atom_id]["proc"] != final[atom_id]["proc"]
        ]
        if ranks > 1 and not changed:
            raise AssertionError(f"{label}: no atom crossed an MPI domain")
        net_force = [
            sum(record[field] for record in final.values())
            for field in ("fx", "fy", "fz")
        ]
        require_below(
            math.sqrt(sum(value * value for value in net_force)),
            2.0e-8,
            f"{label} net force",
        )
        migrating[str(ranks)] = changed
    rank1_initial, rank1_final = frames[1]
    rank2_initial, rank2_final = frames[2]
    return {
        "migrating_atom_ids": migrating,
        "rank2_initial_max_abs": compare_records(
            rank1_initial, rank2_initial, "Kokkos rank1/rank2 initial"
        ),
        "rank2_final_max_abs": compare_records(
            rank1_final, rank2_final, "Kokkos rank1/rank2 final"
        ),
    }


def validate_multielement(output):
    anchor = one_frame(output / "cpu.multi.normal.dump")
    parity = {}
    for label in (
        "kk.multi.normal.rank1",
        "kk.multi.normal.rank2",
        "kk.multi.reversed.rank1",
    ):
        records = one_frame(output / f"{label}.dump")
        parity[label] = compare_records(anchor, records, label)
        force_norm = math.sqrt(
            sum(
                record[field] * record[field]
                for record in records.values()
                for field in ("fx", "fy", "fz")
            )
        )
        if force_norm <= 1.0e-10:
            raise AssertionError(f"{label}: force response is vacuous")
    return {"parity_max_abs": parity}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report = {
        "schema": "ye3t_fitted_lifted_cauchy_kokkos_validation_v1",
        "numdiff": validate_numdiff(args.output),
        "migration": validate_migration(args.output),
        "multielement": validate_multielement(args.output),
        "publication_timing": False,
        "oversubscribed_multi_rank": True,
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json is not None:
        args.json.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
