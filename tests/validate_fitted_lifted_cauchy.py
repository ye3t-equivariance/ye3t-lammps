#!/usr/bin/env python3
"""Validate a fitted lifted-Cauchy bundle across CPU source paths and MPI ranks."""

import argparse
import json
import math
from pathlib import Path

from test_lifted_cauchy_lammps import (
    expected_per_atom_virial,
    marker,
    read_dump,
    validate_snapshot,
)
from test_lifted_cauchy_mpi import compare_records, one_frame


def require_below(value, limit, label):
    if not math.isfinite(value) or value > limit:
        raise AssertionError(f"{label}: {value:.17g} exceeds {limit:.17g}")


def validate_numdiff(output):
    results = {}
    for policy in ("direct", "factorized"):
        log = output / f"log.numdiff.{policy}.rank1"
        force_error = marker(log, "YE3T_LIFTED_NUMDIFF_FORCE_MAX_ABS")
        virial_error = marker(log, "YE3T_LIFTED_NUMDIFF_VIRIAL_L2_EV")
        require_below(force_error, 2.0e-7, f"{policy} force finite difference")
        require_below(virial_error, 2.0e-6, f"{policy} virial finite difference")
        results[policy] = {
            "energy_eV": marker(log, "YE3T_LIFTED_PE"),
            "force_max_abs_eV_per_A": force_error,
            "virial_l2_eV": virial_error,
        }
    direct = one_frame(output / "numdiff.direct.rank1.snapshot.dump")
    factorized = one_frame(output / "numdiff.factorized.rank1.snapshot.dump")
    parity = compare_records(direct, factorized, "fitted direct/factorized numdiff")
    return {"finite_difference": results, "source_parity_max_abs": parity}


def validate_migration(output):
    variants = tuple(
        f"{policy}.rank{ranks}"
        for policy in ("direct", "factorized")
        for ranks in (1, 2, 4)
    )
    frames = {}
    migration = {}
    for variant in variants:
        initial = one_frame(output / f"migration.{variant}.initial.dump")
        final = one_frame(output / f"migration.{variant}.final.dump")
        frames[variant] = (initial, final)
        ranks = int(variant.rsplit("rank", 1)[1])
        owners = {int(record["proc"]) for record in initial.values()}
        if len(owners) != ranks:
            raise AssertionError(
                f"{variant}: {len(owners)} MPI ranks own atoms, expected {ranks}"
            )
        changed = [
            atom_id
            for atom_id in initial
            if initial[atom_id]["proc"] != final[atom_id]["proc"]
        ]
        if ranks > 1 and not changed:
            raise AssertionError(f"{variant}: no atom crossed an MPI domain")
        net_force = [
            sum(record[field] for record in final.values())
            for field in ("fx", "fy", "fz")
        ]
        require_below(
            math.sqrt(sum(value * value for value in net_force)),
            2.0e-8,
            f"{variant} net force",
        )
        migration[variant] = changed

    anchor_initial, anchor_final = frames["direct.rank1"]
    parity = {}
    for variant in variants[1:]:
        initial, final = frames[variant]
        parity[f"{variant}.initial"] = compare_records(
            anchor_initial, initial, f"{variant} initial"
        )
        parity[f"{variant}.final"] = compare_records(
            anchor_final, final, f"{variant} final"
        )
    return {"migrating_atom_ids": migration, "parity_max_abs": parity}


def validate_python_reference(output, reference_path):
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    expected_vatom = expected_per_atom_virial(reference)
    results = {}
    for policy in ("direct", "factorized"):
        _records, results[policy] = validate_snapshot(
            output / f"numdiff.{policy}.rank1.snapshot.dump",
            output / f"log.numdiff.{policy}.rank1",
            reference,
            expected_vatom,
            policy,
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    report = {
        "schema": "ye3t_fitted_lifted_cauchy_cpu_validation_v1",
        "numdiff": validate_numdiff(args.output),
        "migration": validate_migration(args.output),
    }
    if args.reference is not None:
        report["python_reference"] = validate_python_reference(
            args.output, args.reference
        )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json is not None:
        args.json.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
