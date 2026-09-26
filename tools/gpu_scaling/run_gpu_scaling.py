#!/usr/bin/env python3
"""Run fail-closed PairYE3T/Kokkos evaluator and GPU-scaling studies."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone


EVALUATORS = ("direct", "block", "scalar_power", "coupled_product", "auto")
CASE_MATRIX_NAME = "gpu_scaling_cases.json"


def examples_directory(script_dir):
    return script_dir.parents[1] / "examples" / "PACKAGES" / "ye3t"


def resolve_bundle_directory(script_dir, directory):
    candidates = (
        examples_directory(script_dir) / "models" / directory,
        script_dir.parents[1] / "tests" / "fixtures" / "high_rank" / directory,
    )
    for candidate in candidates:
        if (candidate / "model.yace").is_file():
            return candidate
    raise RuntimeError("model bundle not found: " + directory)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def factor_3d(count):
    if count < 1:
        raise ValueError("rank count must be positive")
    factors = [1, 1, 1]
    remaining = count
    divisor = 2
    primes = []
    while divisor * divisor <= remaining:
        while remaining % divisor == 0:
            primes.append(divisor)
            remaining //= divisor
        divisor += 1
    if remaining > 1:
        primes.append(remaining)
    for value in reversed(primes):
        index = factors.index(min(factors))
        factors[index] *= value
    return tuple(sorted(factors, reverse=True))


def cell_shape(scaling, base_cells, ranks):
    if scaling == "strong":
        return (base_cells, base_cells, base_cells)
    factors = factor_3d(ranks)
    return tuple(base_cells * factor for factor in factors)


def parse_loop_timing(text, measured_steps, expected_atoms, expected_ranks):
    pattern = re.compile(
        r"Loop time of\s+([0-9.eE+-]+)\s+on\s+(\d+)\s+procs\s+for\s+"
        r"(\d+)\s+steps\s+with\s+(\d+)\s+atoms"
    )
    matches = []
    for match in pattern.finditer(text):
        loop_seconds = float(match.group(1))
        ranks = int(match.group(2))
        steps = int(match.group(3))
        atoms = int(match.group(4))
        if steps == measured_steps and atoms == expected_atoms:
            matches.append((loop_seconds, ranks, steps, atoms))
    if not matches:
        raise RuntimeError("could not find the measured LAMMPS loop-time record")
    loop_seconds, ranks, steps, atoms = matches[-1]
    if ranks != expected_ranks:
        raise RuntimeError("LAMMPS timing record has the wrong MPI rank count")

    timing_start = text.rfind("MPI task timing breakdown:")
    sections = {}
    if timing_start >= 0:
        for line in text[timing_start:].splitlines():
            columns = [column.strip() for column in line.split("|")]
            if columns and columns[0] in {
                "Pair",
                "Neigh",
                "Comm",
                "Output",
                "Modify",
                "Sync",
                "Other",
            }:
                try:
                    sections[columns[0].lower() + "_seconds"] = float(columns[2])
                except (IndexError, ValueError):
                    pass

    memory_matches = re.findall(
        r"Per MPI rank memory allocation \(min/avg/max\) =\s+"
        r"([0-9.eE+-]+)\s+\|\s+([0-9.eE+-]+)\s+\|\s+([0-9.eE+-]+)\s+Mbytes",
        text,
    )
    memory = None
    if memory_matches:
        minimum, average, maximum = memory_matches[-1]
        memory = {
            "minimum_mib": float(minimum),
            "average_mib": float(average),
            "maximum_mib": float(maximum),
        }

    def last_metric(pattern_text):
        values = re.findall(pattern_text, text)
        return float(values[-1]) if values else None

    result = {
        "loop_seconds": loop_seconds,
        "mpi_ranks": ranks,
        "steps": steps,
        "atoms": atoms,
        "loop_microseconds_per_atom_step": loop_seconds * 1.0e6 / (steps * atoms),
        "memory_per_rank": memory,
        "average_local_atoms": last_metric(r"Nlocal:\s+([0-9.eE+-]+)\s+ave"),
        "average_ghost_atoms": last_metric(r"Nghost:\s+([0-9.eE+-]+)\s+ave"),
        "average_full_neighbors": last_metric(r"FullNghs:\s+([0-9.eE+-]+)\s+ave"),
    }
    result.update(sections)
    return result


def parse_prefixed_marker(line, prefix):
    if not line.startswith(prefix):
        raise RuntimeError("invalid PairYE3T Kokkos marker")
    result = {}
    for field in line[len(prefix) :].split(", "):
        key, separator, value = field.partition(" ")
        if not separator or not key or not value:
            raise RuntimeError("malformed PairYE3T Kokkos marker field")
        result[key] = value
    return result


def parse_marker(line):
    return parse_prefixed_marker(line, "YE3T Kokkos device-plan probe: ")


def require_kokkos_neighbor_contract(text, style):
    if (
        "pair " + style + "/kk, perpetual" not in text
        or "attributes: full, newton on, kokkos_device" not in text
        or "pair build: full/bin/kk/device" not in text
    ):
        raise RuntimeError(
            style + "/kk did not report full/Newton-on Kokkos-device ownership"
        )


def _marker_integer(marker, name):
    try:
        return int(marker.get(name, ""))
    except ValueError as error:
        raise RuntimeError("PairYE3T did not report a valid " + name) from error


def parse_ye3t_dispatch(text, requested, auto_expectation=None):
    require_kokkos_neighbor_contract(text, "ye3t")
    markers = [
        parse_marker(line)
        for line in text.splitlines()
        if line.startswith("YE3T Kokkos device-plan probe: ")
    ]
    if not markers:
        raise RuntimeError("PairYE3T did not report a Kokkos device-plan probe")
    marker = markers[-1]
    if marker.get("execution_space") not in {"Cuda", "HIP"}:
        raise RuntimeError("PairYE3T did not execute a supported GPU backend")
    if "YE3T CPU dispatch:" in text:
        raise RuntimeError("PairYE3T entered the CPU evaluator during a GPU run")
    if marker.get("copy_sentinel_maximum_error") != "0":
        raise RuntimeError("PairYE3T device-plan copy sentinel failed")
    if marker.get("selection_rank_consensus") != "passed":
        raise RuntimeError("PairYE3T selection identity did not reach MPI consensus")
    evaluator = marker.get("evaluator", "")
    if requested == "auto":
        if not evaluator.startswith("auto["):
            raise RuntimeError("PairYE3T AUTO did not report its selected evaluator")
        if auto_expectation:
            for field in (
                "evaluator",
                "active_families",
                "planner_profile",
                "planner_algorithm",
                "planner_status",
                "calibration_hash",
                "decision_reason",
            ):
                if field in auto_expectation and marker.get(field) != auto_expectation[field]:
                    raise RuntimeError(
                        "PairYE3T AUTO does not match the checked case field " + field
                    )
    elif evaluator != requested:
        raise RuntimeError("PairYE3T did not dispatch the requested evaluator")
    if marker.get("requested_policy") != requested:
        raise RuntimeError("PairYE3T reported the wrong requested evaluator policy")

    family_contract = {
        "block": ("active_block", "block_routes"),
        "scalar_power": ("active_scalar_power", "scalar_routes"),
        "coupled_product": ("active_coupled_product", "coupled_plans"),
    }
    family_fields = {
        "direct": ("active_direct", "direct_monomial_storage"),
        "block": ("active_block", "block_routes"),
        "scalar_power": ("active_scalar_power", "scalar_routes"),
        "coupled_product": ("active_coupled_product", "coupled_plans"),
    }
    active_families = []
    for family, (active_field, count_field) in family_fields.items():
        active = marker.get(active_field)
        count = _marker_integer(marker, count_field)
        if active not in {"yes", "no"} or (active == "yes") != (count > 0):
            raise RuntimeError(
                "PairYE3T active-family flag disagrees with " + count_field
            )
        if active == "yes":
            active_families.append(family)
    reported_families = marker.get("active_families", "").split("+")
    if reported_families != active_families:
        raise RuntimeError("PairYE3T evaluator and active-family records disagree")
    if requested == "auto":
        if evaluator != "auto[" + "+".join(active_families) + "]":
            raise RuntimeError("PairYE3T AUTO bracket families are inconsistent")
        selected_evaluator = (
            auto_expectation.get("selected_evaluator")
            if auto_expectation
            else None
        )
        if selected_evaluator:
            optimized = [
                family for family in active_families if family != "direct"
            ]
            if selected_evaluator == "direct":
                valid_selection = active_families == ["direct"]
            else:
                valid_selection = optimized == [selected_evaluator]
            if not valid_selection:
                raise RuntimeError(
                    "PairYE3T AUTO active families disagree with the replay selection"
                )
        for name in (
            "planner_profile",
            "planner_algorithm",
            "planner_status",
            "calibration_hash",
            "decision_reason",
        ):
            if not marker.get(name):
                raise RuntimeError("PairYE3T AUTO omitted " + name)
    elif requested == "direct":
        if marker.get("active_direct") != "yes" or any(
            marker.get(name) != "no"
            for name in (
                "active_block",
                "active_scalar_power",
                "active_coupled_product",
            )
        ):
            raise RuntimeError("PairYE3T direct mode reported an optimized family")
    elif requested in family_contract:
        active_field, count_field = family_contract[requested]
        route_count = _marker_integer(marker, count_field)
        if marker.get(active_field) != "yes" or route_count < 1:
            raise RuntimeError(
                "PairYE3T forced " + requested + " mode executed no optimized routes"
            )
        other_optimized = {
            "block": "active_block",
            "scalar_power": "active_scalar_power",
            "coupled_product": "active_coupled_product",
        }
        if any(
            marker.get(field) == "yes"
            for family, field in other_optimized.items()
            if family != requested
        ):
            raise RuntimeError(
                "PairYE3T forced " + requested + " activated another optimized family"
            )
    if requested == "scalar_power" and marker.get("scalar_math_probe") != "passed":
        raise RuntimeError("PairYE3T scalar-power device math probe failed")
    if requested == "coupled_product" and marker.get("coupled_math_probe") != "passed":
        raise RuntimeError("PairYE3T coupled-product device math probe failed")
    required_hashes = (
        "source_model",
        "direct_logical_plan",
        "flat_plan",
        "device_schedule",
        "semantic_selection",
        "VJP_schedule",
    )
    for name in required_hashes:
        if not re.fullmatch(r"[0-9a-f]{64}", marker.get(name, "")):
            raise RuntimeError("PairYE3T did not report a valid " + name + " hash")
    if not marker.get("source_policy"):
        raise RuntimeError("PairYE3T did not report its source policy")
    if _marker_integer(marker, "source_serial_terms") < 1:
        raise RuntimeError("PairYE3T reported invalid source serial work")
    return marker


def parse_pace_dispatch(text):
    if "Recursive evaluator is used" in text:
        raise RuntimeError("ML-PACE entered the recursive evaluator")
    if (
        "KOKKOS mode with Kokkos version" not in text
        or "Product evaluator is used" not in text
        or "pair pace/kk" not in text
    ):
        raise RuntimeError("ML-PACE did not execute pace/kk product")
    require_kokkos_neighbor_contract(text, "pace")
    return {"evaluator": "pace_product", "execution_space": "Kokkos GPU"}


def normalize_pci_bus_id(value):
    match = re.fullmatch(
        r"(?:[0-9a-fA-F]{4,8}:)?([0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7])",
        value,
    )
    if not match:
        raise RuntimeError("invalid GPU PCI bus ID: " + value)
    return match.group(1).lower()


def parse_runtime_device_records(text, expected_ranks):
    prefix = "YE3T Kokkos rank-device: "
    records = [
        parse_prefixed_marker(line, prefix)
        for line in text.splitlines()
        if line.startswith(prefix)
    ]
    if len(records) != expected_ranks:
        raise RuntimeError("PairYE3T did not report one runtime GPU record per rank")
    ranks = sorted(_marker_integer(record, "world_rank") for record in records)
    if ranks != list(range(expected_ranks)):
        raise RuntimeError("PairYE3T runtime GPU records do not cover the MPI world")
    summary_prefix = "YE3T Kokkos rank-device summary: "
    summaries = [
        parse_prefixed_marker(line, summary_prefix)
        for line in text.splitlines()
        if line.startswith(summary_prefix)
    ]
    if len(summaries) != 1:
        raise RuntimeError("PairYE3T did not report one runtime GPU-map summary")
    summary = summaries[0]
    if _marker_integer(summary, "world_size") != expected_ranks:
        raise RuntimeError("PairYE3T runtime GPU-map summary has the wrong world size")
    if not re.fullmatch(r"[0-9a-f]{64}", summary.get("rank_map", "")):
        raise RuntimeError("PairYE3T runtime GPU-map summary has an invalid hash")
    return records, summary


def validate_runtime_device_records(runtime_records, probe_records):
    probes = {record["global_rank"]: record for record in probe_records}
    for runtime in runtime_records:
        rank = _marker_integer(runtime, "world_rank")
        probe = probes.get(rank)
        if probe is None:
            raise RuntimeError("runtime GPU record has no matching rank-map probe")
        if runtime.get("hostname") != probe.get("hostname"):
            raise RuntimeError("runtime and probe hostnames disagree")
        if runtime.get("uuid", "").lower() != probe["gpu"]["uuid"].lower():
            raise RuntimeError("runtime and probe GPU UUIDs disagree")
        if normalize_pci_bus_id(runtime.get("pci_bus_id", "")) != normalize_pci_bus_id(
            probe["gpu"]["pci_bus_id"]
        ):
            raise RuntimeError("runtime and probe GPU PCI bus IDs disagree")


def parse_capacity_records(text, expected_ranks, require_exact, headroom):
    prefix = "YE3T Kokkos capacity: "
    records = [
        parse_prefixed_marker(line, prefix)
        for line in text.splitlines()
        if line.startswith(prefix)
    ]
    if len(records) != expected_ranks:
        raise RuntimeError("PairYE3T did not report one capacity record per rank")
    ranks = sorted(_marker_integer(record, "world_rank") for record in records)
    if ranks != list(range(expected_ranks)):
        raise RuntimeError("PairYE3T capacity records do not cover the MPI world")
    for record in records:
        free_bytes = _marker_integer(record, "free_device_bytes")
        total_bytes = _marker_integer(record, "total_device_bytes")
        if total_bytes <= 0 or free_bytes / total_bytes < headroom:
            raise RuntimeError("PairYE3T post-allocation capacity violates GPU headroom")
        if require_exact:
            if (
                _marker_integer(record, "exact_chunk_required") != 1
                or _marker_integer(record, "center_reduced") != 0
                or _marker_integer(record, "edge_reductions") != 0
            ):
                raise RuntimeError("PairYE3T performance run reduced its requested chunk")
    return records


def gpu_aware_mpi_status(text):
    if "Disabling GPU-aware MPI" in text:
        return "disabled_by_lammps"
    lines = [line.strip() for line in text.splitlines() if "GPU-aware MPI" in line]
    return lines[-1] if lines else "not_reported"


def distribution(values):
    median = statistics.median(values)
    return {
        "minimum": min(values),
        "median": median,
        "maximum": max(values),
        "median_absolute_deviation": statistics.median(
            abs(value - median) for value in values
        ),
        "samples": len(values),
        "values": values,
    }


def paired_bootstrap(records, baseline, candidate, samples=20000):
    indexed = {}
    for record in records:
        if record["status"] != "passed" or record["engine"] not in {
            baseline,
            candidate,
        }:
            continue
        indexed[(record["engine"], record["repetition"])] = record["timing"][
            "loop_seconds"
        ]
    repetitions = sorted(
        repetition
        for engine, repetition in indexed
        if engine == baseline and (candidate, repetition) in indexed
    )
    ratios = [
        indexed[(baseline, repetition)] / indexed[(candidate, repetition)]
        for repetition in repetitions
    ]
    if not ratios:
        return None
    generator = random.Random(0)
    draws = []
    for unused in range(samples):
        sample = [ratios[generator.randrange(len(ratios))] for ignored in ratios]
        draws.append(statistics.median(sample))
    draws.sort()
    lower = draws[int(0.025 * (samples - 1))]
    upper = draws[int(0.975 * (samples - 1))]
    ratio = statistics.median(ratios)
    return {
        "paired_repetitions": len(ratios),
        "median_baseline_over_candidate": ratio,
        "median_speedup_fraction": ratio - 1.0,
        "bootstrap_95_percent_ratio_interval": [lower, upper],
        "bootstrap_95_percent_speedup_interval": [lower - 1.0, upper - 1.0],
        "bootstrap_samples": samples,
        "bootstrap_seed": 0,
    }


def read_meminfo():
    values = {}
    path = Path("/proc/meminfo")
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, rest = line.partition(":")
        if separator:
            values[key] = rest.strip()
    return values


def command_version(command, arguments):
    if not command:
        return {"available": False}
    try:
        completed = subprocess.run(
            [command] + arguments,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": True, "error": str(error)}
    output = (completed.stdout + "\n" + completed.stderr).strip()
    return {
        "available": True,
        "returncode": completed.returncode,
        "first_line": output.splitlines()[0] if output else "",
    }


def run_bounded(command, timeout_seconds, cwd=None, environment=None):
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
    return subprocess.CompletedProcess(
        command,
        124 if timed_out else process.returncode,
        stdout,
        stderr,
    )


def launcher_prefix(args, ranks):
    return [args.launcher] + args.launcher_arg + [args.rank_flag, str(ranks)]


def rank_probe(args, script_dir, output, ranks):
    rank_dir = output / "preflight" / ("ranks-" + str(ranks))
    rank_dir.mkdir(parents=True)
    command = launcher_prefix(args, ranks) + [
        sys.executable,
        str(script_dir / "gpu_rank_wrapper.py"),
        "probe",
        "--expected-world",
        str(ranks),
        "--device-policy",
        args.device_policy,
        "--device-ids",
        args.device_ids,
        "--output-dir",
        str(rank_dir),
    ]
    completed = run_bounded(command, args.timeout_seconds)
    (rank_dir / "launcher.stdout").write_text(completed.stdout, encoding="utf-8")
    (rank_dir / "launcher.stderr").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        suffix = ": " + detail[-1] if detail else ""
        raise RuntimeError(
            "MPI GPU rank-map probe failed for " + str(ranks) + " ranks" + suffix
        )
    paths = sorted(rank_dir.glob("rank-*.json"))
    if len(paths) != ranks:
        raise RuntimeError("MPI GPU rank-map probe did not produce one record per rank")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    validate_rank_map(args, records, ranks)
    return {"command": command, "records": records}


def validate_rank_map(args, records, ranks):
    global_ranks = [record.get("global_rank") for record in records]
    if sorted(global_ranks) != list(range(ranks)):
        raise RuntimeError("rank-map records do not cover the complete MPI world")
    local_keys = set()
    uuids = []
    physical_devices = []
    wrapped = False
    processes = {}
    for record in records:
        if record.get("schema") != "ye3t_gpu_rank_map_v1":
            raise RuntimeError("rank-map record has an unsupported schema")
        if record.get("world_size") != ranks:
            raise RuntimeError("rank-map record has the wrong MPI world size")
        local_key = (record.get("hostname"), record.get("local_rank"))
        if local_key in local_keys:
            raise RuntimeError("two MPI tasks report the same host-local rank")
        local_keys.add(local_key)
        uuid = record.get("gpu", {}).get("uuid", "")
        pci_bus_id = record.get("gpu", {}).get("pci_bus_id", "")
        if not uuid or not pci_bus_id:
            raise RuntimeError("rank-map record lacks a physical GPU UUID or PCI bus ID")
        uuids.append(uuid)
        physical_devices.append((record.get("hostname"), pci_bus_id))
        wrapped = wrapped or bool(record.get("mapping_wrapped"))
        total = float(record["gpu"]["memory_total_mib"])
        free = float(record["gpu"]["memory_free_mib"])
        if total <= 0.0 or free / total < args.memory_headroom:
            raise RuntimeError("GPU preflight does not retain the requested memory headroom")
        for process in record.get("compute_processes", []):
            processes[(uuid, process["pid"])] = process
    duplicate_gpu = len(set(uuids)) != len(uuids)
    duplicate_physical_device = len(set(physical_devices)) != len(physical_devices)
    if args.mode == "performance":
        if wrapped or duplicate_gpu or duplicate_physical_device:
            raise RuntimeError(
                "performance mode requires one unique physical GPU per MPI rank"
            )
        if processes:
            raise RuntimeError("performance mode requires idle selected GPUs")
    elif (
        wrapped or duplicate_gpu or duplicate_physical_device
    ) and not args.allow_oversubscription:
        raise RuntimeError(
            "duplicate GPU use requires --allow-oversubscription in correctness mode"
        )


def case_command(args, script_dir, ranks, engine, shape, paths):
    input_path = (
        script_dir / "in.ye3t.pace-gpu-benchmark"
        if engine == "pace_product"
        else (
            script_dir / "in.ye3t.auto-replay-gpu-benchmark"
            if engine == "auto" and "auto_replay" in paths
            else script_dir / "in.ye3t.gpu-benchmark"
        )
    )
    lammps = [
        str(paths["lmp"]),
        "-k",
        "on",
        "g",
        "1",
        "-pk",
        "kokkos",
        "neigh",
        "half",
        "-sf",
        "kk",
        "-in",
        str(input_path),
        "-var",
        "model",
        str(paths["model"]),
        "-var",
        "element",
        args.element,
        "-var",
        "lattice_constant",
        format(args.lattice_constant, ".17g"),
        "-var",
        "cells_x",
        str(shape[0]),
        "-var",
        "cells_y",
        str(shape[1]),
        "-var",
        "cells_z",
        str(shape[2]),
        "-var",
        "warmup_steps",
        str(args.warmup_steps),
        "-var",
        "timed_steps",
        str(args.timed_steps),
    ]
    if engine == "pace_product":
        lammps.extend(("-var", "pace_chunksize", str(args.pace_chunksize)))
    else:
        lammps.extend(
            (
                "-var",
                "plan_manifest",
                str(paths["plan"]),
                "-var",
                "block_policy",
                engine,
                "-var",
                "ye3t_chunksize",
                str(args.ye3t_chunksize),
            )
        )
        if engine == "auto" and "auto_replay" in paths:
            lammps.extend(("-var", "auto_replay", str(paths["auto_replay"])))
    wrapper = [
        sys.executable,
        str(script_dir / "gpu_rank_wrapper.py"),
        "launch",
        "--expected-world",
        str(ranks),
        "--device-policy",
        args.device_policy,
        "--device-ids",
        args.device_ids,
        "--",
    ]
    return launcher_prefix(args, ranks) + wrapper + lammps


def run_case(args, script_dir, output, paths, ranks, repetition, engine, shape):
    case_dir = output / ("ranks-" + str(ranks)) / (
        "rep-" + str(repetition).zfill(2) + "-" + engine
    )
    case_dir.mkdir(parents=True)
    log_path = case_dir / "log.lammps"
    screen_path = case_dir / "screen.txt"
    stdout_path = case_dir / "launcher.stdout"
    stderr_path = case_dir / "launcher.stderr"
    parity_path = case_dir / "parity.dump"
    command = case_command(args, script_dir, ranks, engine, shape, paths)
    command.extend(("-var", "parity_dump", str(parity_path)))
    command.extend(("-log", str(log_path), "-screen", str(screen_path)))
    environment = os.environ.copy()
    environment.update(
        {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    )
    if args.mode == "performance" and engine != "pace_product":
        environment["YE3T_KOKKOS_REQUIRE_EXACT_CHUNKSIZE"] = "1"
    if engine == "block" and args.block_schedule != "default":
        environment["YE3T_KOKKOS_BLOCK_SCHEDULE"] = args.block_schedule
    if engine == "block" and args.block_team_size is not None:
        environment["YE3T_KOKKOS_BLOCK_TEAM_SIZE"] = str(args.block_team_size)
    started = time.monotonic()
    completed = run_bounded(
        command, args.timeout_seconds, cwd=case_dir, environment=environment
    )
    elapsed = time.monotonic() - started
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    record = {
        "ranks": ranks,
        "repetition": repetition,
        "engine": engine,
        "command": command,
        "wall_seconds_including_startup": elapsed,
        "returncode": completed.returncode,
        "status": "failed" if completed.returncode else "passed",
        "paths": {
            "case": str(case_dir),
            "log": str(log_path),
            "parity_dump": str(parity_path),
            "screen": str(screen_path),
        },
    }
    if completed.returncode != 0:
        record["failure"] = (
            "LAMMPS or MPI launcher timed out"
            if completed.returncode == 124
            else "LAMMPS returned nonzero"
        )
        return record
    text = log_path.read_text(encoding="utf-8")
    if not parity_path.is_file():
        raise RuntimeError("GPU case did not produce its pre-timing parity dump")
    record["parity_dump_sha256"] = sha256(parity_path)
    atoms = 2 * shape[0] * shape[1] * shape[2]
    try:
        record["timing"] = parse_loop_timing(
            text, args.timed_steps, atoms, ranks
        )
        if engine == "pace_product":
            record["dispatch"] = parse_pace_dispatch(text)
        else:
            record["dispatch"] = parse_ye3t_dispatch(
                text,
                engine,
                (
                    args.replay_expectation
                    if engine == "auto" and args.replay_expectation is not None
                    else (
                        args.case_record.get("auto_expectation")
                        if engine == "auto"
                        else None
                    )
                ),
            )
            runtime_devices, runtime_summary = parse_runtime_device_records(
                text, ranks
            )
            validate_runtime_device_records(
                runtime_devices, args.rank_maps[str(ranks)]["records"]
            )
            if record["dispatch"].get("rank_device_map") != runtime_summary["rank_map"]:
                raise RuntimeError(
                    "PairYE3T plan marker and runtime GPU-map hashes disagree"
                )
            record["runtime_devices"] = runtime_devices
            record["runtime_device_summary"] = runtime_summary
            record["capacity"] = parse_capacity_records(
                text,
                ranks,
                args.mode == "performance",
                args.memory_headroom,
            )
            if record["dispatch"]["source_model"] != sha256(paths["model"]):
                raise RuntimeError("PairYE3T source-model identity does not match model bytes")
            free_bytes = float(record["dispatch"].get("free_device_bytes", "nan"))
            total_bytes = float(record["dispatch"].get("total_device_bytes", "nan"))
            if not (total_bytes > 0.0 and free_bytes / total_bytes >= args.memory_headroom):
                raise RuntimeError("PairYE3T device-plan probe violates memory headroom")
        record["gpu_aware_mpi"] = gpu_aware_mpi_status(text)
        record["hashes"] = {
            "log_sha256": sha256(log_path),
            "screen_sha256": sha256(screen_path),
            "stdout_sha256": sha256(stdout_path),
            "stderr_sha256": sha256(stderr_path),
        }
    except Exception as error:
        record["status"] = "failed_validation"
        record["failure"] = str(error)
    return record


def build_parity_certificate(records, rank_counts, engines, paths):
    if len(engines) < 2:
        return {
            "schema": "ye3t_gpu_pretiming_parity_v1",
            "status": "not_applicable_single_engine",
        }
    comparisons = []
    for ranks in rank_counts:
        repetitions = sorted(
            {
                record["repetition"]
                for record in records
                if record["ranks"] == ranks and record["status"] == "passed"
            }
        )
        for repetition in repetitions:
            by_engine = {
                record["engine"]: record
                for record in records
                if record["ranks"] == ranks
                and record["repetition"] == repetition
                and record["status"] == "passed"
            }
            missing = set(engines) - set(by_engine)
            if missing:
                raise RuntimeError(
                    "pre-timing parity is missing evaluator(s): "
                    + ",".join(sorted(missing))
                )
            reference_engine = (
                "pace_product" if "pace_product" in by_engine else "direct"
            )
            candidates = [
                engine for engine in engines if engine != reference_engine
            ]
            reference_dump = Path(by_engine[reference_engine]["paths"]["parity_dump"])
            verifier_config = reference_dump.parent / (
                f"parity_verifier_ranks{ranks}_rep{repetition}_config.json"
            )
            verifier_config.write_text(
                json.dumps(
                    {
                        "check": "dumps",
                        "reference": str(reference_dump),
                        "candidates": [
                            str(by_engine[engine]["paths"]["parity_dump"])
                            for engine in candidates
                        ],
                    }
                ),
                encoding="utf-8",
            )
            environment = dict(os.environ)
            environment["CONFIG_PATH"] = str(verifier_config)
            command = [sys.executable, str(paths["verifier"])]
            completed = subprocess.run(
                command, capture_output=True, text=True, env=environment
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise RuntimeError(
                    "pre-timing energy/force/virial parity failed: "
                    + detail
                )
            result = json.loads(completed.stdout)
            if not all(item.get("passed") for item in result["comparisons"]):
                raise RuntimeError("pre-timing parity verifier returned a failed row")
            comparisons.append(
                {
                    "candidates": candidates,
                    "comparison": result,
                    "dump_sha256": {
                        engine: by_engine[engine]["parity_dump_sha256"]
                        for engine in engines
                    },
                    "mpi_ranks": ranks,
                    "reference": reference_engine,
                    "repetition": repetition,
                }
            )
    payload = {
        "artifacts": {
            name: sha256(paths[name])
            for name in ("lmp", "model", "plan", "runner", "verifier")
            if name in {"lmp", "model", "plan", "runner", "verifier"}
        },
        "comparisons": comparisons,
        "schema": "ye3t_gpu_pretiming_parity_v1",
        "status": "passed",
    }
    encoded = json.dumps(
        payload, allow_nan=False, ensure_ascii=True, sort_keys=True,
        separators=(",", ":")
    ).encode("ascii")
    payload["certificate_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def validate_fixed_identities(records, case_record):
    fields = (
        "source_model",
        "direct_logical_plan",
        "flat_plan",
        "semantic_selection",
        "device_schedule",
        "VJP_schedule",
        "evaluator",
        "active_families",
        "planner_profile",
        "planner_algorithm",
        "planner_status",
        "calibration_hash",
        "decision_reason",
        "direct_monomial_storage",
        "block_routes",
        "scalar_routes",
        "coupled_plans",
        "device_class",
        "execution_space",
        "kokkos_layout",
        "block_schedule",
        "block_team_size",
        "block_scratch_layout",
        "block_scratch_bytes",
        "source_policy",
        "VJP_policy",
    )
    identities = {}
    for engine in EVALUATORS:
        selected = [
            record["dispatch"]
            for record in records
            if record["engine"] == engine and record["status"] == "passed"
        ]
        if not selected:
            continue
        identity = {field: selected[0].get(field) for field in fields}
        for marker in selected[1:]:
            if any(marker.get(field) != identity[field] for field in fields):
                raise RuntimeError(
                    engine + " changed semantic or device identity across the scaling series"
                )
        identities[engine] = identity
    source_models = {identity["source_model"] for identity in identities.values()}
    direct_plans = {
        identity["direct_logical_plan"] for identity in identities.values()
    }
    if len(source_models) > 1 or len(direct_plans) > 1:
        raise RuntimeError("YE3T evaluators did not share one logical model and plan")
    expectation = case_record.get("auto_expectation", {})
    if expectation.get("require_direct_flat_plan_identity"):
        if "direct" in identities and "auto" in identities:
            if identities["direct"]["flat_plan"] != identities["auto"]["flat_plan"]:
                raise RuntimeError(
                    "checked conservative AUTO did not retain the direct flat plan"
                )
    return identities


def summarize(records, rank_counts, engines, scaling, performance_statistics):
    summary = {"by_rank": {}, "scaling": {}}
    for ranks in rank_counts:
        rank_records = [record for record in records if record["ranks"] == ranks]
        engine_summary = {}
        for engine in engines:
            passed = [
                record
                for record in rank_records
                if record["engine"] == engine and record["status"] == "passed"
            ]
            if not passed:
                engine_summary[engine] = {"status": "no_passed_samples"}
                continue
            engine_summary[engine] = {
                "status": "passed",
                "loop_seconds": distribution(
                    [record["timing"]["loop_seconds"] for record in passed]
                ),
                "loop_microseconds_per_atom_step": distribution(
                    [
                        record["timing"]["loop_microseconds_per_atom_step"]
                        for record in passed
                    ]
                ),
            }
            for section in ("pair", "neigh", "comm", "modify", "sync"):
                key = section + "_seconds"
                values = [record["timing"][key] for record in passed if key in record["timing"]]
                if values:
                    engine_summary[engine][key] = distribution(values)
        ye3t_passed = {
            engine: data["loop_seconds"]["median"]
            for engine, data in engine_summary.items()
            if engine != "pace_product" and data.get("status") == "passed"
        }
        best = (
            min(ye3t_passed, key=ye3t_passed.get)
            if performance_statistics and ye3t_passed
            else None
        )
        comparisons = {}
        if performance_statistics and "direct" in ye3t_passed:
            for engine in ye3t_passed:
                if engine != "direct":
                    comparisons["direct_over_" + engine] = paired_bootstrap(
                        rank_records, "direct", engine
                    )
        if (
            performance_statistics
            and engine_summary.get("pace_product", {}).get("status") == "passed"
        ):
            for engine in ye3t_passed:
                comparisons["pace_product_over_" + engine] = paired_bootstrap(
                    rank_records, "pace_product", engine
                )
        summary["by_rank"][str(ranks)] = {
            "engines": engine_summary,
            "fastest_ye3t_evaluator": best,
            "paired_comparisons": comparisons,
        }

    if not performance_statistics:
        summary["performance_statistics"] = "suppressed_for_correctness_mode"
        return summary

    base = min(rank_counts)
    for engine in engines:
        base_data = summary["by_rank"][str(base)]["engines"].get(engine, {})
        if base_data.get("status") != "passed":
            continue
        base_time = base_data["loop_seconds"]["median"]
        rows = []
        for ranks in sorted(rank_counts):
            data = summary["by_rank"][str(ranks)]["engines"].get(engine, {})
            if data.get("status") != "passed":
                continue
            current = data["loop_seconds"]["median"]
            speedup = base_time / current
            if scaling == "strong":
                efficiency = speedup / (ranks / base)
            else:
                efficiency = speedup
            rows.append(
                {
                    "ranks": ranks,
                    "median_loop_seconds": current,
                    "speedup_from_base": speedup,
                    "parallel_efficiency": efficiency,
                }
            )
        summary["scaling"][engine] = {
            "kind": scaling,
            "base_ranks": base,
            "rows": rows,
        }
    return summary


def write_human_summary(path, result):
    lines = [
        "# PairYE3T/Kokkos GPU run summary",
        "",
        "- Mode: `" + result["configuration"]["mode"] + "`",
        "- Scaling: `" + result["configuration"]["scaling"] + "`",
        "- Publication timing: `" + str(result["publication_timing"]).lower() + "`",
        "- Pre-timing parity: `" + result["parity_certificate"]["status"] + "`",
        "- MPI ranks: `" + ",".join(str(value) for value in result["configuration"]["rank_counts"]) + "`",
        "- Evaluators: `" + ",".join(result["configuration"]["engines"]) + "`",
        "- Model SHA256: `" + result["artifacts"]["model"]["sha256"] + "`",
        "",
        "## Fastest YE3T evaluator by rank",
        "",
        "| ranks | evaluator |",
        "|---:|---|",
    ]
    for ranks in result["configuration"]["rank_counts"]:
        best = result["summary"]["by_rank"][str(ranks)]["fastest_ye3t_evaluator"]
        lines.append("| " + str(ranks) + " | " + str(best) + " |")
    lines.extend(
        (
            "",
            "The JSON result is authoritative. Correctness/oversubscribed runs are not",
            "publication timing or multi-GPU scaling evidence.",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def load_case_matrix(path):
    matrix = json.loads(path.read_text(encoding="utf-8"))
    if matrix.get("schema") != "ye3t_gpu_scaling_cases_v1":
        raise RuntimeError("GPU scaling case matrix has an unsupported schema")
    cases = matrix.get("cases")
    if not isinstance(cases, dict) or not cases:
        raise RuntimeError("GPU scaling case matrix contains no cases")
    for name, record in cases.items():
        legal = record.get("legal_evaluators")
        defaults = record.get("default_evaluators")
        if (
            not isinstance(name, str)
            or not isinstance(record.get("model_directory"), str)
            or not isinstance(legal, list)
            or not legal
            or not isinstance(defaults, list)
            or not defaults
            or any(evaluator not in EVALUATORS for evaluator in legal)
            or any(evaluator not in legal for evaluator in defaults)
        ):
            raise RuntimeError("GPU scaling case matrix contains an invalid case")
    return matrix


def resolve_paths(args, script_dir, case_matrix):
    if (args.model is None) != (args.plan is None):
        raise RuntimeError("--model and --plan must be supplied together")
    if args.model is None:
        if args.bundle not in case_matrix["cases"]:
            raise RuntimeError("unknown checked GPU bundle: " + args.bundle)
        args.case_record = case_matrix["cases"][args.bundle]
        directory = args.case_record["model_directory"]
        defaults = args.case_record["default_evaluators"]
        bundle_dir = resolve_bundle_directory(script_dir, directory)
        model = bundle_dir / "model.yace"
        plan = bundle_dir / "manifest.json"
        evaluators = tuple(args.evaluator or defaults)
        illegal = set(evaluators) - set(args.case_record["legal_evaluators"])
        if illegal:
            raise RuntimeError(
                "bundle " + args.bundle + " does not contain evaluator(s): "
                + ",".join(sorted(illegal))
            )
        pace = args.case_record.get("pace_product", {})
        if args.with_pace and not pace.get("supported", False):
            raise RuntimeError(
                "PACE product is unavailable for bundle " + args.bundle + ": "
                + pace.get("reason", "no checked support record")
            )
    else:
        args.case_record = {}
        model = Path(args.model).resolve()
        plan = Path(args.plan).resolve()
        evaluators = tuple(args.evaluator or ("direct", "auto"))
    if len(set(evaluators)) != len(evaluators):
        raise RuntimeError("evaluator list contains duplicates")
    for evaluator in evaluators:
        if evaluator not in EVALUATORS:
            raise RuntimeError("unknown evaluator: " + evaluator)
    paths = {
        "lmp": Path(args.lmp).resolve(),
        "model": model.resolve(),
        "plan": plan.resolve(),
        "runner": Path(__file__).resolve(),
        "verifier": (examples_directory(script_dir) / "verify_examples.py").resolve(),
        "rank_wrapper": (script_dir / "gpu_rank_wrapper.py").resolve(),
        "ye3t_input": (script_dir / "in.ye3t.gpu-benchmark").resolve(),
        "ye3t_auto_replay_input": (
            script_dir / "in.ye3t.auto-replay-gpu-benchmark"
        ).resolve(),
        "pace_input": (script_dir / "in.ye3t.pace-gpu-benchmark").resolve(),
        "case_matrix": (script_dir / CASE_MATRIX_NAME).resolve(),
    }
    if args.install_record:
        paths["install_record"] = Path(args.install_record).resolve()
    if args.auto_replay:
        paths["auto_replay"] = Path(args.auto_replay).resolve()
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(name + " does not name a file: " + str(path))
    if not os.access(paths["lmp"], os.X_OK):
        raise RuntimeError("LAMMPS executable is not executable")
    args.replay_expectation = None
    if "auto_replay" in paths:
        replay = json.loads(paths["auto_replay"].read_text(encoding="utf-8"))
        replay_hash = replay.get("replay_sha256")
        if replay.get("schema") != "ye3t_kokkos_auto_replay_v2" or not re.fullmatch(
            r"[0-9a-f]{64}", replay_hash or ""
        ):
            raise RuntimeError("--auto-replay is not a valid YE3T Kokkos replay")
        replay_executable = replay.get("runtime", {}).get(
            "lammps_executable_sha256"
        )
        if replay_executable != sha256(paths["lmp"]):
            raise RuntimeError(
                "--auto-replay was calibrated for a different LAMMPS executable"
            )
        args.replay_expectation = {
            "planner_profile": "kokkos_gpu_device_replay_v1",
            "planner_algorithm": "device_bound_candidate_replay_v1",
            "planner_status": "selected",
            "calibration_hash": replay_hash,
            "decision_reason": "offline_device_bound_interleaved_calibration",
            "selected_evaluator": replay.get("calibration", {}).get(
                "selected_evaluator"
            ),
        }
        if args.replay_expectation["selected_evaluator"] not in {
            "direct",
            "block",
            "scalar_power",
            "coupled_product",
        }:
            raise RuntimeError("--auto-replay has an invalid selected evaluator")
    return paths, evaluators


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmp", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bundle", default="compact")
    parser.add_argument("--model")
    parser.add_argument("--plan")
    parser.add_argument("--auto-replay")
    parser.add_argument("--element", default="Ta")
    parser.add_argument("--evaluator", action="append", choices=EVALUATORS)
    parser.add_argument("--with-pace", action="store_true")
    parser.add_argument("--skip-pace-reason")
    parser.add_argument("--mode", choices=("correctness", "performance"), required=True)
    parser.add_argument("--allow-oversubscription", action="store_true")
    parser.add_argument("--publication", action="store_true")
    parser.add_argument("--ranks", action="append", type=int)
    parser.add_argument("--device-policy", choices=("explicit", "launcher"), default="explicit")
    parser.add_argument("--device-ids", default="0")
    parser.add_argument("--launcher", default=os.environ.get("YE3T_MPIEXEC", "mpiexec"))
    parser.add_argument("--launcher-arg", action="append", default=[])
    parser.add_argument("--rank-flag", default="-n")
    parser.add_argument("--scaling", choices=("strong", "weak"), default="strong")
    parser.add_argument("--base-cells", type=int)
    parser.add_argument("--lattice-constant", type=float, default=3.3)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--timed-steps", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--ye3t-chunksize", type=int, default=4096)
    parser.add_argument("--pace-chunksize", type=int, default=4096)
    parser.add_argument(
        "--block-schedule", choices=("default", "fused", "work_major"),
        default="default"
    )
    parser.add_argument("--block-team-size", type=int)
    parser.add_argument("--lock-file")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--memory-headroom", type=float, default=0.15)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--install-record")
    args = parser.parse_args()
    args.rank_counts = args.ranks or [1]
    defaults = (
        (3, 1, 5, 1)
        if args.mode == "correctness"
        else (10, 20, 300, 8)
    )
    if args.base_cells is None:
        args.base_cells = defaults[0]
    if args.warmup_steps is None:
        args.warmup_steps = defaults[1]
    if args.timed_steps is None:
        args.timed_steps = defaults[2]
    if args.repetitions is None:
        args.repetitions = defaults[3]
    return args


def validate_arguments(args, evaluators):
    integers = (
        args.base_cells,
        args.warmup_steps,
        args.timed_steps,
        args.repetitions,
        args.ye3t_chunksize,
        args.pace_chunksize,
        args.timeout_seconds,
        *args.rank_counts,
    )
    if min(integers) < 1:
        raise RuntimeError("all sizes, ranks, steps, repetitions, and timeouts must be positive")
    if len(set(args.rank_counts)) != len(args.rank_counts):
        raise RuntimeError("--ranks values must be unique")
    if not 0.0 < args.memory_headroom < 1.0:
        raise RuntimeError("--memory-headroom must lie strictly between zero and one")
    if args.mode == "performance" and args.allow_oversubscription:
        raise RuntimeError("performance mode cannot allow GPU oversubscription")
    if args.auto_replay and "auto" not in evaluators:
        raise RuntimeError("--auto-replay requires the auto evaluator")
    if args.block_team_size is not None and (
        args.block_schedule != "work_major"
        or args.block_team_size not in (32, 64, 128, 256)
    ):
        raise RuntimeError(
            "--block-team-size requires work_major and must be 32, 64, 128, or 256"
        )
    if args.publication:
        if args.mode != "performance":
            raise RuntimeError("--publication requires performance mode")
        if set(args.rank_counts) != {1, 2, 4}:
            raise RuntimeError("publication scaling requires 1, 2, and 4 ranks")
        if args.repetitions < 6 or args.warmup_steps < 20 or args.timed_steps < 300:
            raise RuntimeError("publication timing requires at least 6 repeats, 20 warmup steps, and 300 measured steps")
        if not args.with_pace or "direct" not in evaluators or "auto" not in evaluators:
            raise RuntimeError("publication timing requires PACE, YE3T direct, and YE3T AUTO")
        if not args.auto_replay:
            raise RuntimeError("publication timing requires --auto-replay")
        if args.continue_on_error:
            raise RuntimeError("publication timing cannot continue past failed cases")
    if args.skip_pace_reason and args.with_pace:
        raise RuntimeError("--with-pace and --skip-pace-reason are mutually exclusive")


def main():
    args = parse_arguments()
    script_dir = Path(__file__).resolve().parent
    case_matrix = load_case_matrix(script_dir / CASE_MATRIX_NAME)
    paths, evaluators = resolve_paths(args, script_dir, case_matrix)
    validate_arguments(args, evaluators)

    launcher = shutil.which(args.launcher)
    if launcher is None:
        raise RuntimeError("MPI launcher not found: " + args.launcher)
    args.launcher = str(Path(launcher).resolve())
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi is required for fail-closed CUDA mapping")

    output = Path(args.output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise RuntimeError("--output must be absent or an empty directory")
    output.mkdir(parents=True, exist_ok=True)
    lock_path = (
        Path(args.lock_file).resolve()
        if args.lock_file
        else (
            Path("/tmp/ye3t_lammps_gpu_publication.lock")
            if args.publication
            else output.parent / ".ye3t_gpu_scaling.lock"
        )
    )
    lock_stream = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another YE3T GPU runner holds the output-parent lock") from error
    lock_stream.write(str(os.getpid()) + "\n")
    lock_stream.flush()

    help_result = subprocess.run(
        [str(paths["lmp"]), "-help"], capture_output=True, text=True, timeout=60
    )
    (output / "lammps-help.txt").write_text(
        help_result.stdout + help_result.stderr, encoding="utf-8"
    )
    if help_result.returncode != 0 or "ye3t/kk" not in help_result.stdout:
        raise RuntimeError("LAMMPS executable does not register pair_style ye3t/kk")
    if args.with_pace and "pace/kk" not in help_result.stdout:
        raise RuntimeError("--with-pace requires pair_style pace/kk")

    engines = list(evaluators)
    if args.with_pace:
        engines.append("pace_product")
    preflight = {
        "schema": "ye3t_lammps_gpu_preflight_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "probing",
        "mode": args.mode,
        "publication_requested": args.publication,
        "publication_timing": False,
        "scaling": args.scaling,
        "rank_counts": args.rank_counts,
        "device_policy": args.device_policy,
        "device_ids": args.device_ids,
        "allow_oversubscription": args.allow_oversubscription,
        "memory_headroom_fraction": args.memory_headroom,
        "lock_file": str(lock_path),
        "rank_maps": {},
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "load_average": list(os.getloadavg()),
            "meminfo": read_meminfo(),
            "incoming_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "software": {
            "python": platform.python_version(),
            "mpi_launcher": command_version(args.launcher, ["--version"]),
            "nvidia_smi": command_version("nvidia-smi", ["--version"]),
            "nvcc": command_version(shutil.which("nvcc"), ["--version"]),
            "nsys": command_version(shutil.which("nsys"), ["--version"]),
            "ncu": command_version(shutil.which("ncu"), ["--version"]),
            "gpu_aware_mpi": "determined_from_each_lammps_log",
        },
        "artifacts": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "pace": {
            "included": args.with_pace,
            "omission_reason": args.skip_pace_reason,
        },
    }
    (output / "preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rank_maps = {}
    try:
        for ranks in args.rank_counts:
            rank_maps[str(ranks)] = rank_probe(args, script_dir, output, ranks)
            preflight["rank_maps"] = rank_maps
            (output / "preflight.json").write_text(
                json.dumps(preflight, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    except Exception as error:
        preflight["status"] = "refused"
        preflight["refusal_reason"] = str(error)
        (output / "preflight.json").write_text(
            json.dumps(preflight, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    preflight["status"] = "passed"
    (output / "preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.rank_maps = rank_maps

    records = []
    for ranks in args.rank_counts:
        shape = cell_shape(args.scaling, args.base_cells, ranks)
        for repetition in range(args.repetitions):
            offset = repetition % len(engines)
            order = engines[offset:] + engines[:offset]
            for engine in order:
                print(
                    "Running ranks=" + str(ranks) + " repetition=" + str(repetition)
                    + " engine=" + engine,
                    flush=True,
                )
                record = run_case(
                    args, script_dir, output, paths, ranks, repetition, engine, shape
                )
                records.append(record)
                (output / "records.partial.json").write_text(
                    json.dumps(records, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                if record["status"] != "passed" and not args.continue_on_error:
                    raise RuntimeError(
                        "GPU run failed validation: " + str(record.get("failure", "unknown"))
                    )

    identities = validate_fixed_identities(
        records, {} if args.replay_expectation is not None else args.case_record
    )
    parity_certificate = build_parity_certificate(
        records, args.rank_counts, engines, paths
    )
    summary = summarize(
        records,
        args.rank_counts,
        engines,
        args.scaling,
        args.mode == "performance",
    )
    failed = [record for record in records if record["status"] != "passed"]
    publication_timing = bool(
        args.publication
        and not failed
        and parity_certificate.get("status") == "passed"
    )
    result = {
        "schema": "ye3t_lammps_gpu_scaling_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "publication_timing": publication_timing,
        "publication_timing_eligible": args.mode == "performance" and not failed,
        "configuration": {
            "mode": args.mode,
            "scaling": args.scaling,
            "rank_counts": args.rank_counts,
            "engines": engines,
            "bundle": args.bundle if args.model is None else "custom",
            "base_cells": args.base_cells,
            "cells_by_rank": {
                str(ranks): list(cell_shape(args.scaling, args.base_cells, ranks))
                for ranks in args.rank_counts
            },
            "warmup_steps": args.warmup_steps,
            "timed_steps": args.timed_steps,
            "repetitions": args.repetitions,
            "ye3t_chunksize": args.ye3t_chunksize,
            "pace_chunksize": args.pace_chunksize,
            "block_schedule": args.block_schedule,
            "block_team_size": args.block_team_size,
            "auto_replay": str(paths["auto_replay"]) if "auto_replay" in paths else None,
            "newton_pair": True,
            "neighbor_ownership": "complete centered environments; Kokkos global half",
            "dtype": "float64/complex128",
        },
        "artifacts": preflight["artifacts"],
        "rank_maps": rank_maps,
        "fixed_ye3t_identities": identities,
        "pace": preflight["pace"],
        "parity_certificate": parity_certificate,
        "records": records,
        "summary": summary,
        "failed_records": len(failed),
    }
    result_path = output / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    preflight["publication_timing"] = publication_timing
    (output / "preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_human_summary(output / "SUMMARY.md", result)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failed:
        raise RuntimeError(str(len(failed)) + " GPU case(s) failed or were unsupported")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("run_gpu_scaling.py: " + str(error), file=sys.stderr)
        sys.exit(1)
