#!/usr/bin/env python3
"""CPU timing harness for the tagged-Cauchy native evaluator, on a
128-atom and a 1024-atom BCC Ta cell (single rank), plus the `.yace`
reference for comparison. Deliberately independent of
`tests/validate_fitted_tagged_cauchy.py`'s `measure_timing` (which only
covers the fixed 128-atom cell): this script parameterizes the BCC lattice
replication count so the same deck shape covers both required cell sizes,
mirroring the sized-timing approach already used for
`tests/validate_fitted_tagged_cauchy_kokkos.py`'s timing check, minus every
Kokkos-specific piece.

Not a correctness gate (see `validate_fitted_tagged_cauchy.py --skip-timing`
for that); only measures `Loop time of` from a plain `run N post no` MD
loop, mirroring the existing tagged-Cauchy timing convention exactly.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

MASS_TA = 180.94788


def render_timing_deck(*, pair_style_line, pair_coeff_line, steps, replication):
    lines = [
        "units metal", "atom_style atomic", "boundary p p p",
        "atom_modify map yes sort 0 0.0", "newton on", "",
        "variable a equal 3.300", "lattice bcc ${a}",
        f"region cell block 0 {replication} 0 {replication} 0 {replication} units lattice",
        "create_box 1 cell", "create_atoms 1 box", f"mass 1 {MASS_TA}",
        "reset_atoms id sort yes",
        "velocity all create 300.0 173285 mom yes rot yes dist gaussian loop all", "",
        "neighbor 0.3 bin", "neigh_modify every 1 delay 0 check yes", "",
        pair_style_line, pair_coeff_line, "",
        "timestep 0.001", "fix integrate all nve", "thermo 50", f"run {steps} post no",
    ]
    return "\n".join(lines) + "\n"


def measure(*, lmp, label, pair_style_line, pair_coeff_line, output_dir, replication, steps=200):
    n_atoms = 2 * replication ** 3
    deck_path = output_dir / f"in.timing.{label}"
    log_path = output_dir / f"log.timing.{label}"
    screen_path = output_dir / f"screen.timing.{label}"
    deck_path.write_text(
        render_timing_deck(pair_style_line=pair_style_line, pair_coeff_line=pair_coeff_line,
                           steps=steps, replication=replication),
        encoding="utf-8",
    )
    start = time.perf_counter()
    result = subprocess.run(
        [str(lmp), "-screen", str(screen_path), "-log", str(log_path), "-in", str(deck_path)],
        timeout=300, capture_output=True, text=True,
    )
    wall_seconds = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(
            f"lmp failed (exit {result.returncode}) for {label}\n"
            f"stdout tail: {result.stdout[-2000:]}\nstderr tail: {result.stderr[-2000:]}"
        )
    text = log_path.read_text(encoding="utf-8")
    loop_seconds = None
    for line in text.splitlines():
        if line.startswith("Loop time of"):
            loop_seconds = float(line.split()[3])
            break
    seconds = loop_seconds if loop_seconds is not None else wall_seconds
    return {
        "label": label, "replication": replication, "n_atoms": n_atoms, "steps": steps,
        "loop_seconds": loop_seconds, "wall_seconds": wall_seconds,
        "microseconds_per_atom_step": seconds * 1.0e6 / (n_atoms * steps),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--lmp", type=Path, required=True)
    parser.add_argument("--model-k1", type=Path)
    parser.add_argument("--model-k2", type=Path)
    parser.add_argument("--yace-model", type=Path)
    parser.add_argument("--label", default="run", help="tag for this measurement pass, e.g. baseline/step1")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    report = {"schema": "ye3t_tagged_cauchy_cpu_timing_v1", "pass_label": args.label, "results": {}}

    for tag, model_path in (("k1", args.model_k1), ("k2", args.model_k2)):
        if model_path is None:
            continue
        for replication, cell_label in ((4, "128atom"), (8, "1024atom")):
            key = f"{tag}_{cell_label}"
            report["results"][key] = measure(
                lmp=args.lmp, label=f"{args.label}_{key}",
                pair_style_line="pair_style ye3t model_family tagged_cauchy chunksize 32",
                pair_coeff_line=f"pair_coeff * * {model_path} Ta",
                output_dir=args.output, replication=replication,
            )

    if args.yace_model:
        for replication, cell_label in ((4, "128atom"), (8, "1024atom")):
            key = f"yace_{cell_label}"
            report["results"][key] = measure(
                lmp=args.lmp, label=f"{args.label}_{key}",
                pair_style_line="pair_style ye3t model_family yace chunksize 32",
                pair_coeff_line=f"pair_coeff * * {args.yace_model} Ta",
                output_dir=args.output, replication=replication,
            )

    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
