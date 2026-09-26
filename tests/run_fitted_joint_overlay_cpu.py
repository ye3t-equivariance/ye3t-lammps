#!/usr/bin/env python3
"""Run the bounded CPU/MPI matrix for one fitted YE3T ZBL overlay."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "fitted_joint_overlay"


def _existing_file(path, label):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmp", type=Path, required=True)
    parser.add_argument("--plugin", type=Path)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--fit", type=Path)
    model_group.add_argument("--composite-bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--long-nve-steps", type=int, default=0)
    parser.add_argument("--long-nve-ranks", type=int, default=1)
    parser.add_argument("--trajectory-interval", type=int, default=100)
    args = parser.parse_args()

    lmp = _existing_file(args.lmp, "LAMMPS executable")
    plugin = (
        None
        if args.plugin is None
        else _existing_file(args.plugin, "YE3T plugin")
    )
    plugin_setup = FIXTURE / (
        "plugin.builtin.inc" if plugin is None else "plugin.dynamic.inc"
    )
    ordinary = None
    ordinary_python = None
    lifted_python = None
    pair_setup = FIXTURE / "pair.overlay_no_reference.inc"
    reference_setup = None
    if args.composite_bundle is not None:
        bundle = Path(args.composite_bundle).resolve()
        lifted = _existing_file(
            bundle / "model.ye3t.json", "composite YE3T model"
        )
        lifted_python = bundle
        model_payload = json.loads(lifted.read_text(encoding="utf-8"))
        if model_payload.get("schema") != (
            "ye3t_lifted_cauchy_composite_linear_bundle_v1"
        ):
            raise ValueError("--composite-bundle requires a composite YE3T bundle.")
        if (
            model_payload.get("central_species_order") != ["Ta"]
            or model_payload.get("type_map") != {"Ta": 0}
        ):
            raise ValueError("The composite Ta fixture requires one Ta type/readout.")
        fit_metadata = model_payload.get("fit_metadata", {})
        if fit_metadata.get("targets") != "zbl_residual_energy_and_forces":
            raise ValueError("Composite model was not fit to declared ZBL residuals.")
        reference = fit_metadata.get("reference_potential")
        pair_setup = FIXTURE / "pair.lifted_zbl_ta.inc"
        reference_setup = FIXTURE / "pair.zbl_reference_ta.inc"
    else:
        fit = Path(args.fit).resolve()
        ordinary = _existing_file(
            fit / "model.ye3t" / "ordinary.yace", "ordinary .yace component"
        )
        ordinary_python = _existing_file(
            fit / "ordinary_model.pt", "ordinary Python component"
        )
        lifted = _existing_file(
            fit / "lifted_component.ye3t" / "model.ye3t.json",
            "lifted native component",
        )
        lifted_python = fit / "lifted_component.ye3t"
        overlay_manifest = json.loads(
            _existing_file(
                fit / "lammps_hybrid_overlay.json", "hybrid/overlay manifest"
            ).read_text(encoding="utf-8")
        )
        reference = overlay_manifest.get("reference_potential")
    if reference is not None:
        if (
            reference.get("pair_style") != "zbl"
            or float(reference.get("inner_cutoff_A")) != 4.0
            or float(reference.get("outer_cutoff_A")) != 4.8
            or reference.get("type_order", ["Ta"]) != ["Ta"]
            or reference.get("atomic_numbers") != {"Ta": 73}
        ):
            raise ValueError("This Ta fixture requires the frozen ZBL 4.0/4.8/Z=73 reference.")
        if args.composite_bundle is None:
            pair_setup = FIXTURE / "pair.overlay_zbl_ta.inc"
        reference_setup = FIXTURE / "pair.zbl_reference_ta.inc"
    if shutil.which(args.mpiexec) is None:
        raise FileNotFoundError(f"MPI launcher not found: {args.mpiexec}")
    if args.timeout_seconds < 1:
        raise ValueError("--timeout-seconds must be positive.")
    if args.long_nve_steps < 0 or args.long_nve_ranks < 1:
        raise ValueError("Long-NVE steps/ranks must be nonnegative/positive.")
    if args.trajectory_interval < 1:
        raise ValueError("--trajectory-interval must be positive.")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)

    def run_case(label, ranks, input_name, variables):
        case = output / label
        case.mkdir()
        command = [
            args.mpiexec,
            "-n",
            str(ranks),
            str(lmp),
            "-screen",
            str(case / "screen.log"),
            "-log",
            str(case / "log.lammps"),
            "-in",
            str(FIXTURE / input_name),
            "-var",
            "plugin_path",
            "none" if plugin is None else str(plugin),
            "-var",
            "plugin_setup",
            str(plugin_setup),
            "-var",
            "ordinary_model",
            "none" if ordinary is None else str(ordinary),
            "-var",
            "lifted_model",
            str(lifted),
        ]
        effective_variables = {"pair_setup": pair_setup}
        effective_variables.update(variables)
        for key, value in effective_variables.items():
            command.extend(("-var", str(key), str(value)))
        subprocess.run(command, check=True, timeout=args.timeout_seconds)

    static_cases = (
        ("overlay_static_factorized_rank1_v3", 1, "factorized"),
        ("overlay_static_direct_rank1_v3", 1, "direct"),
        ("overlay_static_factorized_rank2_v3", 2, "factorized"),
        ("overlay_static_factorized_rank4_v3", 4, "factorized"),
    )
    for label, ranks, source in static_cases:
        run_case(
            label,
            ranks,
            "in.overlay_static",
            {
                "source_realization": source,
                "dump_path": output / label / "overlay.snapshot.dump",
            },
        )
    reference_snapshot = None
    if reference_setup is not None:
        label = "overlay_static_zbl_reference_rank1_v1"
        reference_snapshot = output / label / "overlay.snapshot.dump"
        run_case(
            label,
            1,
            "in.overlay_static",
            {
                "pair_setup": reference_setup,
                "source_realization": "factorized",
                "dump_path": reference_snapshot,
            },
        )
    run_case(
        "overlay_numdiff_rank1_v2",
        1,
        "in.overlay_energy",
        {"source_realization": "factorized", "delta": "1.0e-5"},
    )
    for suffix, lattice_constant in (
        ("3p0", "3.0"),
        ("3p1", "3.1"),
        ("3p2", "3.2"),
        ("3p3", "3.3"),
        ("3p4", "3.4"),
        ("3p5", "3.5"),
        ("3p6", "3.6"),
    ):
        run_case(
            f"overlay_eos_a{suffix}",
            1,
            "in.overlay_eos",
            {"source_realization": "factorized", "a": lattice_constant},
        )
    run_case(
        "overlay_strain_plus",
        1,
        "in.overlay_eos",
        {"source_realization": "factorized", "a": "3.3000033"},
    )
    run_case(
        "overlay_strain_minus",
        1,
        "in.overlay_eos",
        {"source_realization": "factorized", "a": "3.2999967"},
    )
    for suffix, separation in (
        ("1p5", "1.5"),
        ("1p8", "1.8"),
        ("2p2", "2.2"),
        ("2p5", "2.5"),
        ("2p86", "2.86"),
        ("3p2", "3.2"),
    ):
        run_case(
            f"overlay_pair_r{suffix}",
            1,
            "in.overlay_pair_scan",
            {"source_realization": "factorized", "separation": separation},
        )
    run_case(
        "overlay_nve_dt1fs",
        1,
        "in.overlay_nve",
        {"source_realization": "factorized", "dt": "0.001", "steps": 100},
    )
    run_case(
        "overlay_nve_dt0p5fs",
        1,
        "in.overlay_nve",
        {"source_realization": "factorized", "dt": "0.0005", "steps": 200},
    )
    if args.long_nve_steps:
        run_case(
            "overlay_nve_long",
            args.long_nve_ranks,
            "in.overlay_nve_trajectory",
            {
                "source_realization": "factorized",
                "dt": "0.001",
                "steps": args.long_nve_steps,
                "trajectory_interval": args.trajectory_interval,
                "trajectory_path": output
                / "overlay_nve_long"
                / "trajectory.dump",
            },
        )

    parity_command = [
        sys.executable,
        str(ROOT / "tests" / "validate_fitted_joint_overlay_python.py"),
        "--snapshot",
        str(
            output
            / "overlay_static_factorized_rank1_v3"
            / "overlay.snapshot.dump"
        ),
        "--ordinary",
        str(ordinary_python),
        "--lifted",
        str(lifted_python),
        "--json",
        str(output / "python_lammps_parity.json"),
    ]
    if ordinary_python is None:
        ordinary_flag = parity_command.index("--ordinary")
        del parity_command[ordinary_flag : ordinary_flag + 2]
    if reference_snapshot is not None:
        parity_command.extend(("--reference-snapshot", str(reference_snapshot)))
    subprocess.run(parity_command, check=True)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tests" / "validate_fitted_joint_overlay.py"),
            str(output),
            "--json",
            str(output / "validation.json"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
