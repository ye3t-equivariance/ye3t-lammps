#!/usr/bin/env python3
"""Evaluate cubic elastic/Born diagnostics for a fitted YE3T+ZBL model."""

import argparse
import json
import math
from pathlib import Path
import re
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "fitted_joint_overlay"
PRESSURE_BAR_TO_GPA = 1.0e-4


def marker(path, name):
    text = Path(path).read_text(encoding="utf-8")
    values = re.findall(rf"^{re.escape(name)}\s+([-+0-9.eE]+)$", text, re.MULTILINE)
    if len(values) != 1:
        raise ValueError(f"{path}: expected one {name} marker, found {len(values)}")
    return float(values[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmp", type=Path, required=True)
    parser.add_argument("--composite-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lattice-constant", type=float, required=True)
    parser.add_argument(
        "--strains", type=float, nargs="+", default=(1.0e-4, 5.0e-4)
    )
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()

    lmp = args.lmp.resolve()
    model = args.composite_bundle.resolve() / "model.ye3t.json"
    if not lmp.is_file() or not model.is_file():
        raise FileNotFoundError("LAMMPS executable or composite model is missing.")
    if shutil.which(args.mpiexec) is None:
        raise FileNotFoundError(f"MPI launcher not found: {args.mpiexec}")
    if args.lattice_constant <= 0.0 or args.ranks < 1:
        raise ValueError("Lattice constant/rank count must be positive.")
    strains = tuple(sorted(set(float(value) for value in args.strains)))
    if not strains or any(not math.isfinite(value) or value <= 0.0 for value in strains):
        raise ValueError("Every strain amplitude must be finite and positive.")

    payload = json.loads(model.read_text(encoding="utf-8"))
    metadata = payload.get("fit_metadata", {})
    reference = metadata.get("reference_potential", {})
    if (
        metadata.get("targets") != "zbl_residual_energy_and_forces"
        or reference.get("pair_style") != "zbl"
        or float(reference.get("inner_cutoff_A", -1.0)) != 4.0
        or float(reference.get("outer_cutoff_A", -1.0)) != 4.8
        or reference.get("atomic_numbers") != {"Ta": 73}
    ):
        raise ValueError("Model metadata does not bind the Ta ZBL 4.0/4.8 reference.")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    modes = ("xx", "yy", "zz", "xy", "xz", "yz")
    pressure_names = ("PXX", "PYY", "PZZ", "PXY", "PXZ", "PYZ")
    raw = {}
    for strain in strains:
        strain_key = format(strain, ".17g")
        raw[strain_key] = {}
        for mode in modes:
            raw[strain_key][mode] = {}
            for sign, signed_strain in (("minus", -strain), ("plus", strain)):
                case = output / f"strain_{strain:.1e}_{mode}_{sign}"
                case.mkdir()
                command = [
                    args.mpiexec,
                    "-n",
                    str(args.ranks),
                    str(lmp),
                    "-screen",
                    str(case / "screen.log"),
                    "-log",
                    str(case / "log.lammps"),
                    "-in",
                    str(FIXTURE / "in.overlay_elastic_point"),
                    "-var",
                    "plugin_setup",
                    str(FIXTURE / "plugin.builtin.inc"),
                    "-var",
                    "pair_setup",
                    str(FIXTURE / "pair.lifted_zbl_ta.inc"),
                    "-var",
                    "lifted_model",
                    str(model),
                    "-var",
                    "source_realization",
                    "factorized",
                    "-var",
                    "a",
                    format(args.lattice_constant, ".17g"),
                    "-var",
                    "mode",
                    mode,
                    "-var",
                    "strain",
                    format(signed_strain, ".17g"),
                ]
                subprocess.run(command, check=True, timeout=args.timeout_seconds)
                log = case / "log.lammps"
                raw[strain_key][mode][sign] = {
                    name.lower(): marker(log, f"YE3T_ELASTIC_{name}")
                    for name in pressure_names
                }

    estimates = []
    for strain in strains:
        strain_key = format(strain, ".17g")
        c11 = []
        c12 = []
        c44 = []
        for axis, mode in enumerate(("xx", "yy", "zz")):
            minus = raw[strain_key][mode]["minus"]
            plus = raw[strain_key][mode]["plus"]
            diagonal = ("pxx", "pyy", "pzz")
            for response, pressure in enumerate(diagonal):
                stiffness = -(
                    plus[pressure] - minus[pressure]
                ) / (2.0 * strain) * PRESSURE_BAR_TO_GPA
                (c11 if response == axis else c12).append(stiffness)
        for mode, pressure in (("xy", "pxy"), ("xz", "pxz"), ("yz", "pyz")):
            minus = raw[strain_key][mode]["minus"]
            plus = raw[strain_key][mode]["plus"]
            c44.append(
                -(plus[pressure] - minus[pressure])
                / (2.0 * strain)
                * PRESSURE_BAR_TO_GPA
            )
        row = {
            "strain_amplitude": strain,
            "C11_GPa": sum(c11) / len(c11),
            "C12_GPa": sum(c12) / len(c12),
            "C44_GPa": sum(c44) / len(c44),
            "C11_equivalent_values_GPa": c11,
            "C12_equivalent_values_GPa": c12,
            "C44_equivalent_values_GPa": c44,
        }
        row["born_C11_minus_C12_GPa"] = row["C11_GPa"] - row["C12_GPa"]
        row["born_C11_plus_2C12_GPa"] = row["C11_GPa"] + 2.0 * row["C12_GPa"]
        row["born_C44_GPa"] = row["C44_GPa"]
        row["born_stable"] = all(
            row[key] > 0.0
            for key in (
                "born_C11_minus_C12_GPa",
                "born_C11_plus_2C12_GPa",
                "born_C44_GPa",
            )
        )
        estimates.append(row)

    report = {
        "schema": "ye3t_fitted_lifted_cubic_elastic_v1",
        "model_self_hash": payload["self_hash"],
        "lattice_constant_A": args.lattice_constant,
        "mpi_ranks": args.ranks,
        "estimates": estimates,
        "raw_pressures_bar": raw,
    }
    (output / "elastic.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
