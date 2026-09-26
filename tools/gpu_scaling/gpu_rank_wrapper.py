#!/usr/bin/env python3
"""Bind one MPI local rank to one GPU and record its physical identity."""

import argparse
import csv
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from datetime import datetime, timezone


GLOBAL_RANK_KEYS = (
    "YE3T_GLOBAL_RANK",
    "OMPI_COMM_WORLD_RANK",
    "PMI_RANK",
    "SLURM_PROCID",
    "MV2_COMM_WORLD_RANK",
    "PALS_RANKID",
)
LOCAL_RANK_KEYS = (
    "YE3T_LOCAL_RANK",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "MPI_LOCALRANKID",
    "SLURM_LOCALID",
    "MV2_COMM_WORLD_LOCAL_RANK",
    "PALS_LOCAL_RANKID",
    "PMI_LOCAL_RANK",
)
WORLD_SIZE_KEYS = (
    "YE3T_WORLD_SIZE",
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "SLURM_NTASKS",
    "MV2_COMM_WORLD_SIZE",
    "PALS_NRANKS",
)


def _rank_value(keys, label, environment):
    values = []
    for key in keys:
        if key not in environment or environment[key] == "":
            continue
        try:
            value = int(environment[key])
        except ValueError as error:
            raise RuntimeError(key + " is not an integer") from error
        if value < 0:
            raise RuntimeError(key + " must not be negative")
        values.append((key, value))
    if not values:
        raise RuntimeError("could not determine MPI " + label)
    distinct = {value for unused, value in values}
    if len(distinct) != 1:
        detail = ", ".join(key + "=" + str(value) for key, value in values)
        raise RuntimeError("inconsistent MPI " + label + " variables: " + detail)
    return values[0][1]


def rank_context(environment=None):
    environment = os.environ if environment is None else environment
    return {
        "global_rank": _rank_value(GLOBAL_RANK_KEYS, "global rank", environment),
        "local_rank": _rank_value(LOCAL_RANK_KEYS, "local rank", environment),
        "world_size": _rank_value(WORLD_SIZE_KEYS, "world size", environment),
    }


def parse_device_ids(value):
    tokens = [token.strip() for token in value.split(",")]
    if not tokens or any(not token for token in tokens):
        raise RuntimeError("--device-ids must be a comma-separated nonempty list")
    for token in tokens:
        if any(character.isspace() for character in token):
            raise RuntimeError("GPU device identifiers may not contain whitespace")
        if token.startswith("-"):
            raise RuntimeError("GPU device identifiers may not begin with '-'")
    return tokens


def select_device(policy, device_ids, local_rank, environment=None):
    environment = os.environ if environment is None else environment
    incoming = environment.get("CUDA_VISIBLE_DEVICES", "")
    if policy == "launcher":
        tokens = parse_device_ids(incoming)
        if len(tokens) != 1:
            raise RuntimeError(
                "launcher device policy requires exactly one CUDA_VISIBLE_DEVICES entry per rank"
            )
        return {
            "device": tokens[0],
            "mapping_wrapped": False,
            "incoming_cuda_visible_devices": incoming,
        }
    tokens = parse_device_ids(device_ids)
    return {
        "device": tokens[local_rank % len(tokens)],
        "mapping_wrapped": local_rank >= len(tokens),
        "incoming_cuda_visible_devices": incoming,
    }


def _run_nvidia_smi(arguments):
    completed = subprocess.run(
        ["nvidia-smi"] + arguments,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if completed.returncode != 0:
        raise RuntimeError("nvidia-smi failed: " + completed.stderr.strip())
    return completed.stdout


def query_gpu(device):
    fields = (
        "uuid,pci.bus_id,name,memory.total,memory.free,driver_version,compute_cap,"
        "pstate,power.limit"
    )
    output = _run_nvidia_smi(
        ["-i", device, "--query-gpu=" + fields, "--format=csv,noheader,nounits"]
    )
    rows = list(csv.reader(line for line in output.splitlines() if line.strip()))
    if len(rows) != 1 or len(rows[0]) != 9:
        raise RuntimeError("nvidia-smi returned an unexpected GPU identity record")
    values = [value.strip() for value in rows[0]]
    try:
        memory_total_mib = float(values[3])
        memory_free_mib = float(values[4])
    except ValueError as error:
        raise RuntimeError("nvidia-smi returned nonnumeric GPU memory") from error
    try:
        power_limit_watts = float(values[8])
    except ValueError:
        power_limit_watts = None
    return {
        "uuid": values[0],
        "pci_bus_id": values[1],
        "name": values[2],
        "memory_total_mib": memory_total_mib,
        "memory_free_mib": memory_free_mib,
        "driver_version": values[5],
        "compute_capability": values[6],
        "performance_state": values[7],
        "power_limit_watts": power_limit_watts,
        "power_limit_raw": values[8],
    }


def query_compute_processes(device):
    output = _run_nvidia_smi(
        [
            "-i",
            device,
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    records = []
    for row in csv.reader(line for line in output.splitlines() if line.strip()):
        if len(row) != 3:
            raise RuntimeError("nvidia-smi returned an unexpected process record")
        pid, name, used = [value.strip() for value in row]
        records.append(
            {"pid": int(pid), "process_name": name, "used_memory_mib": float(used)}
        )
    return records


def probe_record(args):
    context = rank_context()
    if context["world_size"] != args.expected_world:
        raise RuntimeError("MPI world size does not match --expected-world")
    selection = select_device(
        args.device_policy, args.device_ids, context["local_rank"]
    )
    gpu = query_gpu(selection["device"])
    return {
        "schema": "ye3t_gpu_rank_map_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        **context,
        "device_policy": args.device_policy,
        "requested_device_ids": args.device_ids,
        **selection,
        "assigned_cuda_visible_devices": selection["device"],
        "gpu": gpu,
        "compute_processes": query_compute_processes(selection["device"]),
    }


def write_probe(record, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / ("rank-" + str(record["global_rank"]).zfill(6) + ".json")
    temporary = output_dir / ("." + destination.name + "." + str(os.getpid()) + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def launch(args):
    context = rank_context()
    if context["world_size"] != args.expected_world:
        raise RuntimeError("MPI world size does not match --expected-world")
    selection = select_device(
        args.device_policy, args.device_ids, context["local_rank"]
    )
    environment = os.environ.copy()
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = selection["device"]
    if not args.command:
        raise RuntimeError("launch mode requires a command after '--'")
    os.execvpe(args.command[0], args.command, environment)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    for name in ("probe", "launch"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--expected-world", type=int, required=True)
        subparser.add_argument(
            "--device-policy", choices=("explicit", "launcher"), required=True
        )
        subparser.add_argument("--device-ids", default="")
        if name == "probe":
            subparser.add_argument("--output-dir", type=Path, required=True)
        else:
            subparser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.expected_world < 1:
        raise RuntimeError("--expected-world must be positive")
    if args.device_policy == "explicit" and not args.device_ids:
        raise RuntimeError("explicit device policy requires --device-ids")
    if args.action == "probe":
        write_probe(probe_record(args), args.output_dir.resolve())
    else:
        if args.command and args.command[0] == "--":
            args.command = args.command[1:]
        launch(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("gpu_rank_wrapper.py: " + str(error), file=sys.stderr)
        sys.exit(1)
