#!/usr/bin/env python3
"""Calibrate PairYE3T/Kokkos AUTO on the GPU that will run the model."""

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import make_kokkos_auto_replay as replay_tools
import run_gpu_scaling as runner_tools


DEFAULT_TEAM_SIZES = (32, 64, 128, 256)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path, manifest):
    Path(path).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def append_passthrough(command, args):
    command.extend(("--launcher", args.launcher))
    command.append("--rank-flag=" + args.rank_flag)
    command.extend(("--device-policy", args.device_policy))
    for value in args.launcher_arg:
        command.append("--launcher-arg=" + value)


def model_paths(args, script_dir):
    if bool(args.model) != bool(args.plan):
        raise RuntimeError("--model and --plan must be supplied together")
    if args.model:
        evaluators = tuple(args.evaluator or ("direct", "block"))
        return Path(args.model).resolve(), Path(args.plan).resolve(), None, evaluators
    matrix_path = script_dir / "gpu_scaling_cases.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    cases = matrix.get("cases", {})
    if args.bundle not in cases:
        raise RuntimeError("unknown GPU bundle: " + args.bundle)
    case = cases[args.bundle]
    bundle_dir = runner_tools.resolve_bundle_directory(script_dir, case["model_directory"])
    evaluators = tuple(
        evaluator
        for evaluator in case["legal_evaluators"]
        if evaluator != "auto"
    )
    if args.evaluator:
        illegal = set(args.evaluator) - set(evaluators)
        if illegal:
            raise RuntimeError(
                "bundle does not provide evaluator(s): "
                + ",".join(sorted(illegal))
            )
        evaluators = tuple(args.evaluator)
    if args.with_pace and not case.get("pace_product", {}).get("supported", False):
        raise RuntimeError(
            "PACE is unavailable for this bundle; rerun with --without-pace"
        )
    return (
        bundle_dir / "model.yace",
        bundle_dir / "manifest.json",
        args.bundle,
        evaluators,
    )


def runner_command(
    args,
    runner,
    output,
    base_cells,
    schedule,
    team_size,
    bundle,
    evaluators,
    auto_replay=None,
):
    command = [
        sys.executable,
        str(runner),
        "--lmp",
        str(Path(args.lmp).resolve()),
        "--output",
        str(output),
        "--mode",
        "performance",
        "--ranks",
        "1",
        "--device-ids",
        args.device_ids,
        "--base-cells",
        str(base_cells),
        "--warmup-steps",
        str(args.warmup_steps),
        "--timed-steps",
        str(args.timed_steps),
        "--repetitions",
        str(args.repetitions),
        "--ye3t-chunksize",
        str(args.ye3t_chunksize),
        "--pace-chunksize",
        str(args.pace_chunksize),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--memory-headroom",
        str(args.memory_headroom),
        "--block-schedule",
        schedule,
    ]
    if bundle is None:
        command.extend(("--model", str(Path(args.model).resolve())))
        command.extend(("--plan", str(Path(args.plan).resolve())))
    else:
        command.extend(("--bundle", bundle))
    for evaluator in evaluators:
        command.extend(("--evaluator", evaluator))
    if auto_replay is not None:
        command.extend(("--auto-replay", str(auto_replay)))
    if team_size is not None:
        command.extend(("--block-team-size", str(team_size)))
    if args.with_pace:
        command.append("--with-pace")
    append_passthrough(command, args)
    return command


def validation_command(args, runner, output, replay, bundle, evaluators):
    command = [
        sys.executable,
        str(runner),
        "--lmp",
        str(Path(args.lmp).resolve()),
        "--output",
        str(output),
        "--mode",
        "correctness",
        "--ranks",
        "1",
        "--device-ids",
        args.device_ids,
        "--auto-replay",
        str(replay),
        "--base-cells",
        str(max(args.base_cells)),
        "--warmup-steps",
        "1",
        "--timed-steps",
        "3",
        "--repetitions",
        "1",
        "--ye3t-chunksize",
        str(args.ye3t_chunksize),
        "--pace-chunksize",
        str(args.pace_chunksize),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--memory-headroom",
        str(args.memory_headroom),
    ]
    if bundle is None:
        command.extend(("--model", str(Path(args.model).resolve())))
        command.extend(("--plan", str(Path(args.plan).resolve())))
    else:
        command.extend(("--bundle", bundle))
    for evaluator in evaluators:
        command.extend(("--evaluator", evaluator))
    command.extend(("--evaluator", "auto"))
    if args.with_pace:
        command.append("--with-pace")
    append_passthrough(command, args)
    return command


def run_record(command, label, manifest_path, manifest, dry_run):
    record = {
        "command": command,
        "command_shell": shlex.join(command),
        "label": label,
        "status": "planned" if dry_run else "running",
    }
    manifest["commands"].append(record)
    write_manifest(manifest_path, manifest)
    if dry_run:
        return
    completed = subprocess.run(command, check=False)
    record["returncode"] = completed.returncode
    record["status"] = "passed" if completed.returncode == 0 else "failed"
    write_manifest(manifest_path, manifest)
    if completed.returncode != 0:
        raise RuntimeError(label + " failed with exit code " + str(completed.returncode))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark every certified catalogue evaluator and block schedule "
            "on one target GPU, freeze a device-bound AUTO replay, and validate it."
        )
    )
    parser.add_argument("--lmp", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bundle", default="compact")
    parser.add_argument("--model")
    parser.add_argument("--plan")
    parser.add_argument(
        "--evaluator",
        action="append",
        choices=("direct", "block", "scalar_power", "coupled_product"),
    )
    parser.add_argument("--device-ids", default="0")
    parser.add_argument(
        "--device-policy", choices=("explicit", "launcher"), default="explicit"
    )
    parser.add_argument("--launcher", default=os.environ.get("YE3T_MPIEXEC", "mpiexec"))
    parser.add_argument("--launcher-arg", action="append", default=[])
    parser.add_argument("--rank-flag", default="-n")
    parser.add_argument("--base-cells", action="append", type=int)
    parser.add_argument("--work-major-team-size", action="append", type=int)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--timed-steps", type=int, default=300)
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--ye3t-chunksize", type=int, default=4096)
    parser.add_argument("--pace-chunksize", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--memory-headroom", type=float, default=0.15)
    parser.add_argument("--lock-file")
    parser.add_argument("--minimum-centers-per-rank", type=int)
    parser.add_argument("--maximum-centers-per-rank", type=int)
    parser.add_argument(
        "--minimum-lower-confidence-speedup", type=float, default=0.03
    )
    pace = parser.add_mutually_exclusive_group()
    pace.add_argument("--with-pace", dest="with_pace", action="store_true")
    pace.add_argument("--without-pace", dest="with_pace", action="store_false")
    parser.set_defaults(with_pace=True)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-confirmation", action="store_true")
    parser.add_argument("--require-faster-than-pace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.base_cells = args.base_cells or [3, 6, 8, 10]
    args.team_sizes = args.work_major_team_size or list(DEFAULT_TEAM_SIZES)
    return args


def validate_arguments(args):
    integers = (
        *args.base_cells,
        *args.team_sizes,
        args.warmup_steps,
        args.timed_steps,
        args.repetitions,
        args.ye3t_chunksize,
        args.pace_chunksize,
        args.timeout_seconds,
    )
    if min(integers) < 1:
        raise RuntimeError(
            "all sizes, steps, repetitions, chunks, and timeouts must be positive"
        )
    if len(set(args.base_cells)) != len(args.base_cells):
        raise RuntimeError("--base-cells values must be unique")
    if len(set(args.team_sizes)) != len(args.team_sizes):
        raise RuntimeError("work-major team sizes must be unique")
    if any(size not in DEFAULT_TEAM_SIZES for size in args.team_sizes):
        raise RuntimeError("work-major team sizes must be 32, 64, 128, or 256")
    if args.repetitions < 8:
        raise RuntimeError("AUTO calibration requires at least eight paired repetitions")
    if not 0.0 < args.memory_headroom < 1.0:
        raise RuntimeError("--memory-headroom must lie strictly between zero and one")
    if args.minimum_lower_confidence_speedup < 0.0:
        raise RuntimeError("the confidence speedup threshold must be nonnegative")
    if args.evaluator and len(set(args.evaluator)) != len(args.evaluator):
        raise RuntimeError("--evaluator values must be unique")
    if args.require_faster_than_pace and not args.with_pace:
        raise RuntimeError("--require-faster-than-pace requires --with-pace")


def holdout_cells(base_cells):
    measured = set(base_cells)
    for candidate in range(max(base_cells) - 1, min(base_cells), -1):
        if candidate not in measured:
            return candidate
    return None


def main():
    args = parse_arguments()
    validate_arguments(args)
    script_dir = Path(__file__).resolve().parent
    runner = script_dir / "run_gpu_scaling.py"
    replay_maker = script_dir / "make_kokkos_auto_replay.py"
    lmp = Path(args.lmp).resolve()
    if not lmp.is_file() or not os.access(lmp, os.X_OK):
        raise RuntimeError("--lmp must name an executable file")
    model, plan, bundle, evaluators = model_paths(args, script_dir)
    if "direct" not in evaluators or len(evaluators) < 2:
        raise RuntimeError(
            "calibration requires direct and at least one certified non-direct evaluator"
        )
    artifacts = (
        ("model", model),
        ("plan", plan),
        ("runner", runner),
        ("replay maker", replay_maker),
    )
    for label, path in artifacts:
        if not path.is_file():
            raise RuntimeError(label + " does not name a file: " + str(path))

    output = Path(args.output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise RuntimeError("--output must be absent or an empty directory")
    output.mkdir(parents=True, exist_ok=True)
    lock_path = (
        Path(args.lock_file).resolve()
        if args.lock_file
        else Path("/tmp/ye3t_lammps_gpu_calibration.lock")
    )
    lock_stream = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another YE3T GPU calibration holds the global lock") from error
    lock_stream.write(str(os.getpid()) + "\n")
    lock_stream.flush()
    manifest_path = output / "calibration_manifest.json"
    manifest = {
        "schema": "ye3t_kokkos_auto_calibration_run_v1",
        "status": "dry_run" if args.dry_run else "running",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "lmp": {"path": str(lmp), "sha256": sha256_file(lmp)},
            "model": {"path": str(model), "sha256": sha256_file(model)},
            "plan": {"path": str(plan), "sha256": sha256_file(plan)},
        },
        "configuration": {
            "base_cells": sorted(args.base_cells),
            "bundle": bundle or "custom",
            "device_ids": args.device_ids,
            "device_policy": args.device_policy,
            "evaluators": list(evaluators),
            "memory_headroom": args.memory_headroom,
            "lock_file": str(lock_path),
            "minimum_lower_confidence_speedup": args.minimum_lower_confidence_speedup,
            "repetitions": args.repetitions,
            "team_sizes": args.team_sizes,
            "timed_steps": args.timed_steps,
            "warmup_steps": args.warmup_steps,
            "with_pace": args.with_pace,
            "ye3t_chunksize": args.ye3t_chunksize,
        },
        "commands": [],
    }
    write_manifest(manifest_path, manifest)

    benchmark_results = []
    try:
        for base_cells in sorted(args.base_cells):
            base_schedule = "fused" if "block" in evaluators else "default"
            base_name = base_schedule
            case_output = output / "calibration" / (
                "cells-" + str(base_cells).zfill(3)
            ) / base_name
            command = runner_command(
                args,
                runner,
                case_output,
                base_cells,
                base_schedule,
                None,
                bundle,
                evaluators,
            )
            label = (
                "calibration cells=" + str(base_cells) + " schedule=" + base_name
            )
            run_record(command, label, manifest_path, manifest, args.dry_run)
            benchmark_results.append(case_output / "result.json")

            if "block" in evaluators:
                for team_size in args.team_sizes:
                    name = "work_major-" + str(team_size)
                    case_output = output / "calibration" / (
                        "cells-" + str(base_cells).zfill(3)
                    ) / name
                    command = runner_command(
                        args,
                        runner,
                        case_output,
                        base_cells,
                        "work_major",
                        team_size,
                        bundle,
                        ("direct", "block"),
                    )
                    label = (
                        "calibration cells=" + str(base_cells) + " schedule=" + name
                    )
                    run_record(
                        command, label, manifest_path, manifest, args.dry_run
                    )
                    benchmark_results.append(case_output / "result.json")

        minimum_centers = args.minimum_centers_per_rank
        maximum_centers = args.maximum_centers_per_rank
        if minimum_centers is None:
            minimum_centers = 2 * min(args.base_cells) ** 3
        if maximum_centers is None:
            maximum_centers = 2 * max(args.base_cells) ** 3
        if minimum_centers < 1 or maximum_centers < minimum_centers:
            raise RuntimeError("the replay center-count envelope is invalid")

        replay = output / "kokkos_auto_replay.json"
        make_command = [
            sys.executable,
            str(replay_maker),
            "--model",
            str(model),
            "--manifest",
            str(plan),
            "--output",
            str(replay),
            "--rank",
            "1",
            "--minimum-centers-per-rank",
            str(minimum_centers),
            "--maximum-centers-per-rank",
            str(maximum_centers),
            "--minimum-lower-confidence-speedup",
            str(args.minimum_lower_confidence_speedup),
        ]
        for benchmark in benchmark_results:
            make_command.extend(("--benchmark", str(benchmark)))
        run_record(make_command, "freeze AUTO replay", manifest_path, manifest, args.dry_run)
        replay_payload = (
            None
            if args.dry_run
            else json.loads(replay.read_text(encoding="utf-8"))
        )

        confirmation_cells = holdout_cells(args.base_cells)
        if not args.skip_confirmation and confirmation_cells is not None:
            confirmation_output = output / "confirmation"
            command = runner_command(
                args,
                runner,
                confirmation_output,
                confirmation_cells,
                "default",
                None,
                bundle,
                ("direct", "auto"),
                auto_replay=replay,
            )
            run_record(
                command,
                "independent timing confirmation cells=" + str(confirmation_cells),
                manifest_path,
                manifest,
                args.dry_run,
            )
            if not args.dry_run:
                confirmation = json.loads(
                    (confirmation_output / "result.json").read_text(encoding="utf-8")
                )
                replay_tools.validate_benchmark_authorization(
                    confirmation_output / "result.json", confirmation
                )
                comparisons = confirmation["summary"]["by_rank"]["1"][
                    "paired_comparisons"
                ]
                direct_lower = comparisons["direct_over_auto"][
                    "bootstrap_95_percent_speedup_interval"
                ][0]
                pace_lower = None
                if args.with_pace:
                    pace_lower = comparisons["pace_product_over_auto"][
                        "bootstrap_95_percent_speedup_interval"
                    ][0]
                if (
                    replay_payload["calibration"]["selected_evaluator"]
                    != "direct"
                    and direct_lower <= 0.0
                ):
                    raise RuntimeError(
                        "selected non-direct AUTO did not beat direct on the holdout"
                    )
                if args.require_faster_than_pace and pace_lower <= 0.0:
                    raise RuntimeError(
                        "AUTO did not beat pace/kk on the independent holdout"
                    )
                manifest["confirmation"] = {
                    "base_cells": confirmation_cells,
                    "direct_over_auto_lower_95": direct_lower,
                    "pace_over_auto_lower_95": pace_lower,
                    "result": str(confirmation_output / "result.json"),
                }

        if not args.skip_validation:
            validate_output = output / "validation"
            command = validation_command(
                args, runner, validate_output, replay, bundle, evaluators
            )
            run_record(command, "validate AUTO replay", manifest_path, manifest, args.dry_run)

        if args.dry_run:
            manifest["status"] = "dry_run"
        else:
            manifest["replay"] = {
                "path": str(replay),
                "sha256": sha256_file(replay),
                "replay_sha256": replay_payload["replay_sha256"],
                "selected_evaluator": replay_payload["calibration"][
                    "selected_evaluator"
                ],
            }
            manifest["status"] = "passed"
        write_manifest(manifest_path, manifest)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        write_manifest(manifest_path, manifest)
        raise

    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "replay": (
                    None
                    if args.dry_run
                    else str(output / "kokkos_auto_replay.json")
                ),
                "status": manifest["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("calibrate_kokkos_auto.py: " + str(error), file=sys.stderr)
        sys.exit(1)
