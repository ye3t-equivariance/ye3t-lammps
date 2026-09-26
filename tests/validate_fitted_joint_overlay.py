#!/usr/bin/env python3
"""Validate a fitted PACE plus lifted-YE3T hybrid/overlay replay."""

import argparse
import json
import math
from pathlib import Path
import re

from test_lifted_cauchy_mpi import compare_records, one_frame


class GateValidationError(AssertionError):
    """A failed validation gate with evidence that is still safe to report."""

    def __init__(self, message, evidence):
        super().__init__(message)
        self.evidence = evidence


def marker(path, name):
    text = Path(path).read_text(encoding="utf-8")
    values = re.findall(rf"^{re.escape(name)}\s+([-+0-9.eE]+)$", text, re.MULTILINE)
    if len(values) != 1:
        raise ValueError(f"{path}: expected one {name} marker, found {len(values)}")
    return float(values[0])


def validate_eos(output):
    points = []
    for token in ("3p0", "3p1", "3p2", "3p3", "3p4", "3p5", "3p6"):
        log = output / f"overlay_eos_a{token}" / "log.lammps"
        points.append(
            {
                "lattice_constant_A": marker(log, "YE3T_OVERLAY_EOS_A"),
                "energy_eV_per_atom": marker(log, "YE3T_OVERLAY_EOS_EPA"),
                "pressure_bar": marker(log, "YE3T_OVERLAY_EOS_PRESS"),
            }
        )
    values = [point["energy_eV_per_atom"] for point in points]
    minimum = min(range(len(points)), key=values.__getitem__)
    if minimum in {0, len(points) - 1}:
        raise AssertionError("BCC EOS minimum is not bracketed.")
    if not all(values[index] > values[index + 1] for index in range(minimum)):
        raise AssertionError("BCC EOS is not decreasing before its minimum.")
    if not all(
        values[index] < values[index + 1]
        for index in range(minimum, len(points) - 1)
    ):
        raise AssertionError("BCC EOS is not increasing after its minimum.")
    if not (
        points[minimum]["pressure_bar"] > 0.0
        and points[minimum + 1]["pressure_bar"] < 0.0
    ):
        raise AssertionError("BCC EOS pressure does not bracket zero near the minimum.")
    return {"points": points, "minimum_grid_index": minimum}


def validate_pair_scan(output):
    tokens = ("1p5", "1p8", "2p2", "2p5", "2p86", "3p2")
    points = []
    for token in tokens:
        log = output / f"overlay_pair_r{token}" / "log.lammps"
        points.append(
            {
                "separation_A": marker(log, "YE3T_OVERLAY_PAIR_R"),
                "energy_eV": marker(log, "YE3T_OVERLAY_PAIR_PE"),
                "force_x_atom_1_eV_per_A": marker(
                    log, "YE3T_OVERLAY_PAIR_FX1"
                ),
            }
        )
    repulsive = points[:-1]
    force_passed = all(
        point["force_x_atom_1_eV_per_A"] < 0.0 for point in repulsive
    )
    energy_passed = all(
        repulsive[index]["energy_eV"] > repulsive[index + 1]["energy_eV"]
        for index in range(len(repulsive) - 1)
    )
    evidence = {
        "points": points,
        "minimum_tested_separation_A": 1.5,
        "repulsive_force_passed": force_passed,
        "monotonic_energy_passed": energy_passed,
    }
    failures = []
    if not force_passed:
        failures.append("force is not consistently repulsive")
    if not energy_passed:
        failures.append("energy is not monotonic")
    if failures:
        raise GateValidationError(
            "Close-range pair " + " and ".join(failures) + ".", evidence
        )
    return evidence


def validate_nve(output):
    cases = {}
    for name, directory in (
        ("1.0_fs", "overlay_nve_dt1fs"),
        ("0.5_fs", "overlay_nve_dt0p5fs"),
    ):
        log = output / directory / "log.lammps"
        atoms = int(marker(log, "YE3T_OVERLAY_NVE_ATOMS"))
        timestep = marker(log, "YE3T_OVERLAY_NVE_DT")
        steps = int(marker(log, "YE3T_OVERLAY_NVE_STEPS"))
        initial = marker(log, "YE3T_OVERLAY_NVE_INITIAL_ETOTAL")
        final = marker(log, "YE3T_OVERLAY_NVE_FINAL_ETOTAL")
        drift = abs(final - initial)
        final_temperature = marker(log, "YE3T_OVERLAY_NVE_FINAL_TEMP")
        if not 0.0 < final_temperature < 2000.0:
            raise AssertionError(f"{name} NVE final temperature is invalid.")
        if drift / atoms > 1.0e-4:
            raise AssertionError(f"{name} NVE energy drift is too large.")
        cases[name] = {
            "atom_count": atoms,
            "timestep_ps": timestep,
            "steps": steps,
            "duration_ps": timestep * steps,
            "initial_total_energy_eV": initial,
            "final_total_energy_eV": final,
            "absolute_drift_eV": drift,
            "absolute_drift_eV_per_atom": drift / atoms,
            "final_temperature_K": final_temperature,
        }
    ratio = (
        cases["1.0_fs"]["absolute_drift_eV"]
        / cases["0.5_fs"]["absolute_drift_eV"]
    )
    if ratio < 2.5:
        raise AssertionError("NVE drift did not improve sufficiently at half timestep.")
    return {"cases": cases, "full_to_half_timestep_drift_ratio": ratio}


def validate_long_nve(output):
    directory = output / "overlay_nve_long"
    if not directory.is_dir():
        return None
    log = directory / "log.lammps"
    atoms = int(marker(log, "YE3T_OVERLAY_NVE_ATOMS"))
    timestep = marker(log, "YE3T_OVERLAY_NVE_DT")
    steps = int(marker(log, "YE3T_OVERLAY_NVE_STEPS"))
    trajectory_interval = int(
        marker(log, "YE3T_OVERLAY_NVE_TRAJECTORY_INTERVAL")
    )
    initial = marker(log, "YE3T_OVERLAY_NVE_INITIAL_ETOTAL")
    final = marker(log, "YE3T_OVERLAY_NVE_FINAL_ETOTAL")
    final_temperature = marker(log, "YE3T_OVERLAY_NVE_FINAL_TEMP")
    drift_per_atom = abs(final - initial) / atoms
    if steps < 10000:
        raise AssertionError("Long NVE qualification requires at least 10000 steps.")
    if not 0.0 < final_temperature < 2000.0:
        raise AssertionError("Long NVE final temperature is invalid.")
    if not math.isfinite(drift_per_atom) or drift_per_atom > 1.0e-3:
        raise AssertionError("Long NVE energy drift is too large.")

    lines = (directory / "trajectory.dump").read_text(encoding="utf-8").splitlines()
    frame_indices = [
        index for index, line in enumerate(lines) if line == "ITEM: TIMESTEP"
    ]
    timesteps = [int(lines[index + 1]) for index in frame_indices]
    frame_count = len(timesteps)
    expected_minimum = steps // trajectory_interval + 1
    if frame_count < expected_minimum:
        raise AssertionError(
            f"Long NVE trajectory has {frame_count} frames; expected at least "
            f"{expected_minimum}."
        )
    if not timesteps or timesteps[0] != 0 or timesteps[-1] != steps:
        raise AssertionError("Long NVE trajectory does not span the full run.")
    return {
        "atom_count": atoms,
        "timestep_ps": timestep,
        "steps": steps,
        "duration_ps": timestep * steps,
        "initial_total_energy_eV": initial,
        "final_total_energy_eV": final,
        "absolute_drift_eV_per_atom": drift_per_atom,
        "final_temperature_K": final_temperature,
        "trajectory_frame_count": frame_count,
        "trajectory_interval_steps": trajectory_interval,
    }


def validate_virial_finite_difference(output):
    base_log = output / "overlay_eos_a3p3" / "log.lammps"
    plus_log = output / "overlay_strain_plus" / "log.lammps"
    minus_log = output / "overlay_strain_minus" / "log.lammps"
    lattice_constant = marker(base_log, "YE3T_OVERLAY_EOS_A")
    pressure = marker(base_log, "YE3T_OVERLAY_EOS_PRESS")
    volume = (3.0 * lattice_constant) ** 3
    analytic_virial_trace = 3.0 * volume * pressure / 1.6021765e6
    atom_count = 54
    strain_delta = 1.0e-6
    plus_energy = atom_count * marker(plus_log, "YE3T_OVERLAY_EOS_EPA")
    minus_energy = atom_count * marker(minus_log, "YE3T_OVERLAY_EOS_EPA")
    numeric_virial_trace = -(plus_energy - minus_energy) / (2.0 * strain_delta)
    error = abs(analytic_virial_trace - numeric_virial_trace)
    if error > 2.0e-5:
        raise AssertionError(
            f"hybrid/overlay virial finite-difference error is {error:.3e} eV"
        )
    return {
        "isotropic_strain_delta": strain_delta,
        "analytic_virial_trace_eV": analytic_virial_trace,
        "numeric_virial_trace_eV": numeric_virial_trace,
        "absolute_error_eV": error,
    }


def validate_static_parity(output):
    anchor = one_frame(
        output / "overlay_static_factorized_rank1_v3" / "overlay.snapshot.dump"
    )
    variants = {
        "direct.rank1": output
        / "overlay_static_direct_rank1_v3"
        / "overlay.snapshot.dump",
        "factorized.rank2": output
        / "overlay_static_factorized_rank2_v3"
        / "overlay.snapshot.dump",
        "factorized.rank4": output
        / "overlay_static_factorized_rank4_v3"
        / "overlay.snapshot.dump",
    }
    parity = {
        name: compare_records(anchor, one_frame(path), name)
        for name, path in variants.items()
    }
    net_force = [
        sum(record[field] for record in anchor.values())
        for field in ("fx", "fy", "fz")
    ]
    net_force_l2 = math.sqrt(sum(value * value for value in net_force))
    if net_force_l2 > 2.0e-8:
        raise AssertionError(f"hybrid/overlay net force is {net_force_l2:.3e} eV/A")
    return {
        "source_and_mpi_parity_max_abs": parity,
        "net_force_l2_eV_per_A": net_force_l2,
    }


def validate_force_finite_difference(output, delta):
    log = output / "overlay_numdiff_rank1_v2" / "log.lammps"
    analytic_force = marker(log, "YE3T_OVERLAY_INITIAL_FX")
    plus_energy = marker(log, "YE3T_OVERLAY_PLUS_PE")
    minus_energy = marker(log, "YE3T_OVERLAY_MINUS_PE")
    numeric_force = -(plus_energy - minus_energy) / (2.0 * delta)
    force_error = abs(analytic_force - numeric_force)
    if force_error > 2.0e-6:
        raise AssertionError(
            f"hybrid/overlay force finite-difference error is {force_error:.3e} eV/A"
        )
    return {
        "delta_A": delta,
        "analytic_force_eV_per_A": analytic_force,
        "numeric_force_eV_per_A": numeric_force,
        "absolute_error_eV_per_A": force_error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--delta", type=float, default=1.0e-5)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    report = {
        "schema": "ye3t_fitted_joint_overlay_cpu_validation_v2",
        "status": "pending",
        "failures": [],
    }
    gates = (
        ("static_parity", lambda: validate_static_parity(args.output)),
        (
            "force_finite_difference",
            lambda: validate_force_finite_difference(args.output, args.delta),
        ),
        ("bcc_equation_of_state", lambda: validate_eos(args.output)),
        ("close_range_pair_scan", lambda: validate_pair_scan(args.output)),
        (
            "virial_finite_difference",
            lambda: validate_virial_finite_difference(args.output),
        ),
        ("nve_300K_smoke", lambda: validate_nve(args.output)),
        ("nve_300K_long", lambda: validate_long_nve(args.output)),
    )
    for name, function in gates:
        try:
            value = function()
        except Exception as error:
            report[name] = getattr(error, "evidence", None)
            report["failures"].append(
                {
                    "gate": name,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        else:
            if name == "static_parity":
                report.update(value)
            else:
                report[name] = value
    report["status"] = "passed" if not report["failures"] else "failed"
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json is not None:
        args.json.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if report["failures"]:
        names = ", ".join(row["gate"] for row in report["failures"])
        raise AssertionError(f"Fitted-overlay validation failed gates: {names}")


if __name__ == "__main__":
    main()
