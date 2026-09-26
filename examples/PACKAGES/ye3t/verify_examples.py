#!/usr/bin/env python3
"""Verify ML-YE3T LAMMPS example outputs using only the Python standard library.

The check to run and its inputs are read from the JSON file named by the
CONFIG_PATH environment variable, for example:

    CONFIG_PATH=parity.json python3 verify_examples.py

with parity.json containing

    {"check": "dumps",
     "reference": "dump.ye3t.pace-product",
     "candidates": ["dump.ye3t.direct", "dump.ye3t.block"]}

Available checks and their inputs:

    dumps               reference, candidates (list): last-frame parity of
                        positions, atomic/total energy, forces, per-atom and
                        total virial
    numdiff             logs (list): fix numdiff / numdiff/virial markers
    replay              dumps (list): fixed-position repeated evaluations
    md                  log: finite thermo output and bounded energy drift
    virial_consistency  dump, log: per-atom stress sum against the global
                        pair pressure

Tolerances default to the values in TOLERANCES below and can be overridden
per call by adding the same keys to the JSON file. The report is printed as
JSON and the exit status is 0 on success, 1 on a tolerance failure, and 2 on
a malformed input.
"""

import json
import math
import os
from pathlib import Path
import re
import sys


# LAMMPS compute stress/atom returns pressure*volume.  In metal units its
# components are eV multiplied by nktv2p (update.cpp), with the stress sign.
METAL_NKTV2P = 1.6021765e6

TOLERANCES = {
    "dumps": {
        "position_atol": 1.0e-12,
        "atomic_energy_atol": 1.0e-9,
        "total_energy_atol": 1.0e-8,
        "force_atol": 1.0e-8,
        "virial_atol": 1.0e-8,
        "total_virial_atol": 1.0e-8,
        "total_virial_per_atom_atol": 5.0e-11,
    },
    "numdiff": {
        "force_atol": 1.0e-3,
        "virial_atol": 1.0e-1,
    },
    "replay": {
        "minimum_frames": 6,
        "position_atol": 1.0e-12,
        "atomic_energy_atol": 1.0e-9,
        "force_atol": 1.0e-8,
        "virial_atol": 1.0e-8,
    },
    "md": {
        "minimum_steps": 20,
        "energy_drift_atol": 1.0e-4,
    },
    "virial_consistency": {
        "atol": 1.0e-8,
    },
}


def read_dump_frames(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    cursor = 0
    frames = []
    while cursor < len(lines):
        if lines[cursor] != "ITEM: TIMESTEP":
            raise ValueError(f"{path}: expected ITEM: TIMESTEP at line {cursor + 1}")
        cursor += 2
        if cursor >= len(lines) or lines[cursor] != "ITEM: NUMBER OF ATOMS":
            raise ValueError(f"{path}: missing atom-count header")
        count = int(lines[cursor + 1])
        cursor += 2
        if cursor >= len(lines) or not lines[cursor].startswith("ITEM: BOX BOUNDS"):
            raise ValueError(f"{path}: missing box header")
        cursor += 4
        if cursor >= len(lines) or not lines[cursor].startswith("ITEM: ATOMS "):
            raise ValueError(f"{path}: missing atom header")
        fields = lines[cursor].split()[2:]
        cursor += 1
        records = {}
        for _ in range(count):
            values = lines[cursor].split()
            cursor += 1
            if len(values) != len(fields):
                raise ValueError(f"{path}: malformed atom row")
            record = dict(zip(fields, values))
            atom_id = int(record["id"])
            for field, value in record.items():
                if field in {"id", "type", "element"}:
                    continue
                try:
                    number = float(value)
                except ValueError as exc:
                    raise ValueError(
                        f"{path}: non-numeric {field} value for atom {atom_id}"
                    ) from exc
                if not math.isfinite(number):
                    raise ValueError(
                        f"{path}: non-finite {field} value for atom {atom_id}"
                    )
            if atom_id in records:
                raise ValueError(f"{path}: duplicate atom ID {atom_id}")
            records[atom_id] = record
        frames.append(records)
    if not frames:
        raise ValueError(f"{path}: dump has no frames")
    return frames


def read_last_dump_frame(path):
    return read_dump_frames(path)[-1]


def maximum_difference(reference, candidate, fields):
    result = 0.0
    for atom_id in reference:
        for field in fields:
            result = max(
                result,
                abs(float(reference[atom_id][field]) - float(candidate[atom_id][field])),
            )
    return result


def compare_dumps(settings):
    reference = read_last_dump_frame(settings["reference"])
    energy_field = "c_atom_energy"
    force_fields = ("fx", "fy", "fz")
    position_fields = ("x", "y", "z")
    virial_fields = tuple(f"c_atom_stress[{index}]" for index in range(1, 7))
    required = {"id", "type", energy_field, *force_fields, *position_fields, *virial_fields}
    output = []
    failed = False

    for candidate_path in settings["candidates"]:
        candidate = read_last_dump_frame(candidate_path)
        if set(reference) != set(candidate):
            raise ValueError(f"{candidate_path}: atom IDs differ from the reference")
        for atom_id in reference:
            if int(reference[atom_id]["type"]) != int(candidate[atom_id]["type"]):
                raise ValueError(f"{candidate_path}: atom type differs for ID {atom_id}")
            if not required.issubset(reference[atom_id]) or not required.issubset(candidate[atom_id]):
                raise ValueError(f"{candidate_path}: required dump columns are missing")

        position = maximum_difference(reference, candidate, position_fields)
        atomic_energy = maximum_difference(reference, candidate, (energy_field,))
        force = maximum_difference(reference, candidate, force_fields)
        virial = maximum_difference(reference, candidate, virial_fields) / METAL_NKTV2P
        total_virial = max(
            abs(
                math.fsum(float(row[field]) for row in reference.values())
                - math.fsum(float(row[field]) for row in candidate.values())
            )
            / METAL_NKTV2P
            for field in virial_fields
        )
        reference_total = math.fsum(float(row[energy_field]) for row in reference.values())
        candidate_total = math.fsum(float(row[energy_field]) for row in candidate.values())
        total_energy = abs(reference_total - candidate_total)
        effective_total_virial_atol = (
            settings["total_virial_atol"]
            + len(reference) * settings["total_virial_per_atom_atol"]
        )
        passed = (
            position <= settings["position_atol"]
            and atomic_energy <= settings["atomic_energy_atol"]
            and total_energy <= settings["total_energy_atol"]
            and force <= settings["force_atol"]
            and virial <= settings["virial_atol"]
            and total_virial <= effective_total_virial_atol
        )
        failed = failed or not passed
        output.append(
            {
                "candidate": str(candidate_path),
                "maximum_absolute_atomic_energy_eV": atomic_energy,
                "maximum_absolute_force_component_eV_per_A": force,
                "maximum_absolute_position_A": position,
                "maximum_absolute_virial_component_eV": virial,
                "passed": passed,
                "total_energy_difference_eV": total_energy,
                "total_virial_component_difference_eV": total_virial,
                "total_virial_effective_atol_eV": effective_total_virial_atol,
            }
        )

    print(json.dumps({"comparisons": output, "schema": "ye3t_example_dump_parity_v1"}, indent=2))
    return 1 if failed else 0


def read_numdiff(path):
    text = Path(path).read_text(encoding="utf-8")
    force = re.findall(r"YE3T_NUMDIFF_FORCE_MAX_ABS=([-+0-9.eE]+)", text)
    virial = re.findall(r"YE3T_NUMDIFF_VIRIAL_L2=([-+0-9.eE]+)", text)
    if len(force) != 1 or len(virial) != 1:
        raise ValueError(f"{path}: expected one force and one virial marker")
    return float(force[0]), float(virial[0])


def compare_numdiff(settings):
    output = []
    failed = False
    for path in settings["logs"]:
        force, virial = read_numdiff(path)
        passed = force <= settings["force_atol"] and virial <= settings["virial_atol"]
        failed = failed or not passed
        output.append(
            {
                "force_maximum_absolute_error_eV_per_A": force,
                "log": str(path),
                "passed": passed,
                "virial_pressure_l2_error_bar": virial,
            }
        )
    print(json.dumps({"checks": output, "schema": "ye3t_example_numdiff_v1"}, indent=2))
    return 1 if failed else 0


def check_replay(settings):
    energy_field = "c_atom_energy"
    force_fields = ("fx", "fy", "fz")
    position_fields = ("x", "y", "z")
    virial_fields = tuple(f"c_atom_stress[{index}]" for index in range(1, 7))
    required = {"id", "type", energy_field, *force_fields, *position_fields, *virial_fields}
    output = []
    failed = False

    for path in settings["dumps"]:
        frames = read_dump_frames(path)
        reference = frames[0]
        maximum_position = 0.0
        maximum_atomic_energy = 0.0
        maximum_force = 0.0
        maximum_virial = 0.0
        for frame in frames[1:]:
            if set(reference) != set(frame):
                raise ValueError(f"{path}: atom IDs changed during replay")
            for atom_id in reference:
                if int(reference[atom_id]["type"]) != int(frame[atom_id]["type"]):
                    raise ValueError(f"{path}: atom type changed for ID {atom_id}")
                if not required.issubset(reference[atom_id]) or not required.issubset(
                    frame[atom_id]
                ):
                    raise ValueError(f"{path}: required dump columns are missing")
            maximum_position = max(
                maximum_position,
                maximum_difference(reference, frame, position_fields),
            )
            maximum_atomic_energy = max(
                maximum_atomic_energy,
                maximum_difference(reference, frame, (energy_field,)),
            )
            maximum_force = max(
                maximum_force,
                maximum_difference(reference, frame, force_fields),
            )
            maximum_virial = max(
                maximum_virial,
                maximum_difference(reference, frame, virial_fields) / METAL_NKTV2P,
            )
        passed = (
            len(frames) >= settings["minimum_frames"]
            and maximum_position <= settings["position_atol"]
            and maximum_atomic_energy <= settings["atomic_energy_atol"]
            and maximum_force <= settings["force_atol"]
            and maximum_virial <= settings["virial_atol"]
        )
        failed = failed or not passed
        output.append(
            {
                "dump": str(path),
                "frames": len(frames),
                "maximum_absolute_atomic_energy_replay_eV": maximum_atomic_energy,
                "maximum_absolute_force_component_replay_eV_per_A": maximum_force,
                "maximum_absolute_position_replay_A": maximum_position,
                "maximum_absolute_virial_component_replay_eV": maximum_virial,
                "passed": passed,
            }
        )
    print(json.dumps({"checks": output, "schema": "ye3t_example_replay_v1"}, indent=2))
    return 1 if failed else 0


def read_thermo_table(path, required_fields):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    tables = []
    for index, line in enumerate(lines):
        fields = line.split()
        if (
            not fields
            or fields[0] != "Step"
            or not set(required_fields).issubset(fields)
        ):
            continue
        rows = []
        for row_line in lines[index + 1 :]:
            values = row_line.split()
            if len(values) != len(fields):
                break
            try:
                row = {field: float(value) for field, value in zip(fields, values)}
            except ValueError:
                break
            if not all(math.isfinite(value) for value in row.values()):
                raise ValueError(f"{path}: non-finite MD thermo value")
            rows.append(row)
        if rows:
            tables.append(rows)
    if not tables:
        raise ValueError(f"{path}: no matching thermo table found")
    return tables[-1]


def read_md_thermo(path):
    return read_thermo_table(path, ("Atoms", "TotEng"))


def check_virial_consistency(settings):
    frame = read_last_dump_frame(settings["dump"])
    virial_fields = tuple(f"c_atom_stress[{index}]" for index in range(1, 7))
    for atom_id, record in frame.items():
        if not set(virial_fields).issubset(record):
            raise ValueError(f"{settings['dump']}: virial columns missing for atom {atom_id}")

    pressure_fields = tuple(f"c_pair_pressure[{index}]" for index in range(1, 7))
    row = read_thermo_table(settings["log"], ("Volume", *pressure_fields))[-1]
    per_atom_virial = [
        math.fsum(float(record[field]) for record in frame.values()) / METAL_NKTV2P
        for field in virial_fields
    ]
    global_virial = [
        row["Volume"] * row[field] / METAL_NKTV2P for field in pressure_fields
    ]
    residuals = [
        abs(per_atom + global_value)
        for per_atom, global_value in zip(per_atom_virial, global_virial)
    ]
    maximum = max(residuals)
    passed = maximum <= settings["atol"]
    print(
        json.dumps(
            {
                "dump": str(settings["dump"]),
                "global_pair_virial_eV": global_virial,
                "log": str(settings["log"]),
                "maximum_absolute_global_per_atom_virial_residual_eV": maximum,
                "passed": passed,
                "per_atom_stress_sum_eV": per_atom_virial,
                "schema": "ye3t_example_virial_consistency_v1",
            },
            indent=2,
        )
    )
    return 0 if passed else 1


def check_md(settings):
    rows = read_md_thermo(settings["log"])
    first = rows[0]
    last = rows[-1]
    atoms = first["Atoms"]
    if atoms <= 0.0 or any(row["Atoms"] != atoms for row in rows):
        raise ValueError(f"{settings['log']}: invalid or changing atom count")
    elapsed_steps = int(round(last["Step"] - first["Step"]))
    energy_drift = max(abs(row["TotEng"] - first["TotEng"]) for row in rows) / atoms
    passed = (
        elapsed_steps >= settings["minimum_steps"]
        and energy_drift <= settings["energy_drift_atol"]
    )
    output = {
        "atoms": int(atoms),
        "elapsed_steps": elapsed_steps,
        "maximum_absolute_total_energy_drift_eV_per_atom": energy_drift,
        "passed": passed,
        "schema": "ye3t_example_md_v1",
        "temperature_maximum_K": max(row.get("Temp", 0.0) for row in rows),
        "temperature_minimum_K": min(row.get("Temp", 0.0) for row in rows),
        "thermo_rows": len(rows),
    }
    print(json.dumps(output, indent=2))
    return 0 if passed else 1


CHECKS = {
    "dumps": (compare_dumps, ("reference", "candidates")),
    "numdiff": (compare_numdiff, ("logs",)),
    "replay": (check_replay, ("dumps",)),
    "md": (check_md, ("log",)),
    "virial_consistency": (check_virial_consistency, ("dump", "log")),
}


def load_settings():
    config_path = os.environ.get("CONFIG_PATH", "")
    if not config_path:
        raise ValueError("set CONFIG_PATH to a JSON file describing the check")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("check") not in CHECKS:
        raise ValueError(
            f"{config_path}: 'check' must be one of " + ", ".join(sorted(CHECKS))
        )
    check, inputs = CHECKS[config["check"]]
    settings = dict(TOLERANCES[config["check"]])
    settings.update(config)
    for name in inputs:
        if name not in settings:
            raise ValueError(f"{config_path}: missing input '{name}'")
    for name, value in settings.items():
        if name in TOLERANCES[config["check"]] and not isinstance(value, (int, float)):
            raise ValueError(f"{config_path}: '{name}' must be numeric")
    return check, settings


def main():
    try:
        check, settings = load_settings()
        return check(settings)
    except (OSError, ValueError) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
