#!/usr/bin/env python3
"""Capture (or diff) a high-precision energy/force snapshot of the
native tagged-Cauchy CPU evaluator on the k1/k2 reference structures, for
a "no numerical change beyond floating-point reassociation, at most
1e-12 relative" acceptance check. Deliberately independent of the existing
parity checks (which compare against the reference JSON / Python oracle at
looser tolerances, 1e-8-ish): this script
diffs the CPU evaluator directly against itself, step JSON vs. baseline
JSON, at whatever precision `%.17g` thermo/dump output gives LAMMPS.

Reuses `validate_fitted_tagged_cauchy.render_single_point_deck` (read-only
import; that module is not modified).
"""

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from test_lifted_cauchy_lammps import marker, read_dump  # noqa: E402
from validate_fitted_tagged_cauchy import render_single_point_deck, run_lmp  # noqa: E402


def capture(*, lmp, model_path, reference_path, output_dir, tag):
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    structures = reference["structures"]
    element = "Ta"  # both k1/k2 artifacts are single-species Ta
    snapshot = {}
    for label, structure in structures.items():
        positions = structure["positions_angstrom"]
        cell = structure.get("cell_angstrom")
        pbc = structure.get("pbc", (False, False, False))
        cutoff = 10.0  # only used for the non-periodic box padding branch
        deck_path = output_dir / f"in.snapshot.{tag}.{label}"
        dump_path = output_dir / f"snapshot.{tag}.{label}.dump"
        log_path = output_dir / f"log.snapshot.{tag}.{label}"
        screen_path = output_dir / f"screen.snapshot.{tag}.{label}"
        deck_path.write_text(
            render_single_point_deck(
                model_path=model_path, element=element, cutoff=cutoff, positions=positions,
                cell=cell, pbc=pbc, snapshot_path=dump_path, numdiff=False,
                marker_prefix="YE3T_TAGGED",
            ),
            encoding="utf-8",
        )
        run_lmp(str(lmp), deck_path, log_path, screen_path)
        frames = read_dump(dump_path)
        _, records = frames[0]
        total_energy = marker(log_path, "YE3T_TAGGED_PE")
        snapshot[label] = {
            "total_energy_eV": total_energy,
            "per_atom": {
                str(atom_id): {
                    "energy_eV": record["c_atom_energy"],
                    "force_eV_per_A": [record["fx"], record["fy"], record["fz"]],
                }
                for atom_id, record in records.items()
            },
        }
    return snapshot


def diff(baseline, current):
    max_energy_abs = 0.0
    max_energy_rel = 0.0
    max_force_abs = 0.0
    max_force_rel = 0.0
    worst = None
    for label, base_structure in baseline.items():
        cur_structure = current[label]
        e_abs = abs(cur_structure["total_energy_eV"] - base_structure["total_energy_eV"])
        e_rel = e_abs / max(abs(base_structure["total_energy_eV"]), 1.0e-300)
        if e_rel > max_energy_rel:
            max_energy_rel = e_rel
            worst = f"{label} total_energy"
        max_energy_abs = max(max_energy_abs, e_abs)
        for atom_id, base_atom in base_structure["per_atom"].items():
            cur_atom = cur_structure["per_atom"][atom_id]
            for component in range(3):
                base_f = base_atom["force_eV_per_A"][component]
                cur_f = cur_atom["force_eV_per_A"][component]
                f_abs = abs(cur_f - base_f)
                f_rel = f_abs / max(abs(base_f), 1.0e-300)
                if f_abs > 1.0e-300 and f_rel > max_force_rel:
                    max_force_rel = f_rel
                    worst = f"{label} atom {atom_id} force[{component}]"
                max_force_abs = max(max_force_abs, f_abs)
    return {
        "max_energy_abs_eV": max_energy_abs,
        "max_energy_rel": max_energy_rel,
        "max_force_abs_eV_per_A": max_force_abs,
        "max_force_rel": max_force_rel,
        "worst_relative_at": worst,
    }


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    cap = sub.add_parser("capture")
    cap.add_argument("output", type=Path)
    cap.add_argument("--lmp", type=Path, required=True)
    cap.add_argument("--model-k1", type=Path, required=True)
    cap.add_argument("--reference-k1", type=Path, required=True)
    cap.add_argument("--model-k2", type=Path, required=True)
    cap.add_argument("--reference-k2", type=Path, required=True)
    cap.add_argument("--json", type=Path, required=True)

    cmp_parser = sub.add_parser("diff")
    cmp_parser.add_argument("baseline_json", type=Path)
    cmp_parser.add_argument("current_json", type=Path)

    args = parser.parse_args()

    if args.mode == "capture":
        args.output.mkdir(parents=True, exist_ok=True)
        report = {
            "k1": capture(lmp=args.lmp, model_path=args.model_k1, reference_path=args.reference_k1,
                          output_dir=args.output, tag="k1"),
            "k2": capture(lmp=args.lmp, model_path=args.model_k2, reference_path=args.reference_k2,
                          output_dir=args.output, tag="k2"),
        }
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    else:
        baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        current = json.loads(args.current_json.read_text(encoding="utf-8"))
        result = {tag: diff(baseline[tag], current[tag]) for tag in ("k1", "k2")}
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
