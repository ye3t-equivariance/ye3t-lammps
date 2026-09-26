#!/usr/bin/env python3
"""Validate the tagged-Cauchy Kokkos device path against the CPU
evaluator and against the same Python `exact_moment_reduction`
oracle `tests/validate_fitted_tagged_cauchy.py` uses.

Reuses that script's deck renderers (`render_single_point_deck`,
`render_bcc_mpi_deck`, `render_nve_deck`), its `run_lmp` helper (which
already accepts `extra_args`, the hook used here to append the Kokkos
`-k on g 1 -pk kokkos neigh half -sf kk` flags), and its Python-reference
helpers (`cross_check_python`, `per_atom_energies_python`, `_make_atoms`),
instead of duplicating them. Everything CPU/KK-parity-specific (running
both backends on the same deck, diffing at 1e-10, the multi-species
fixture-driven deck, the sized-cell timing decks, and the optional
pre-GPU-test free-memory guard) is new here.

No LAMMPS runs happen at import time; every subprocess call goes through
either `run_lmp` (CPU legs) or `run_kk_lmp` (Kokkos legs, which wait for
free memory first -- see `wait_for_memory_gate`).
"""

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from test_lifted_cauchy_lammps import marker, read_dump  # noqa: E402
from test_lifted_cauchy_mpi import one_frame  # noqa: E402
from validate_fitted_tagged_cauchy import (  # noqa: E402
    ENERGY_TOL,
    FORCE_TOL,
    MASS_TA,
    NUMDIFF_FORCE_TOL,
    NUMDIFF_VIRIAL_TOL_EV,
    PER_ATOM_ENERGY_TOL,
    _make_atoms,
    cross_check_python,
    fnum,
    per_atom_energies_python,
    render_bcc_mpi_deck,
    render_nve_deck,
    render_single_point_deck,
    require_close,
    run_lmp,
)

# Tolerance for CPU/Kokkos-device numeric parity (1e-10). This is
# deliberately far tighter than validate_fitted_tagged_cauchy.py's
# FORCE_TOL/ENERGY_TOL (1e-8), which compare a native evaluator against a
# precomputed reference file; here both sides are the same native evaluator
# running on the same model and the same geometry, just CPU vs. GPU
# floating point, so 1e-10 is the right bar.
DEVICE_PARITY_TOL = 1.0e-10

# ---------------------------------------------------------------------
# Optional free-memory guard, controlled by --min-free-gb,
# --memory-poll-seconds and --memory-max-wait-seconds (the shell driver
# forwards YE3T_MIN_FREE_GB, YE3T_MEMORY_POLL_SECONDS and
# YE3T_MEMORY_MAX_WAIT_SECONDS). Every GPU-dispatching LAMMPS invocation
# (every run_kk_lmp call) checks `free -g`/`free -m` first and, if fewer
# than --min-free-gb GB are available, sleeps in --memory-poll-seconds
# steps for up to --memory-max-wait-seconds before proceeding anyway.
# Every reading taken is appended to MEMORY_READINGS and dumped into the
# JSON report's "memory_readings" field.
# ---------------------------------------------------------------------

MEMORY_READINGS = []


def read_free_gb():
    import subprocess

    result_g = subprocess.run(["free", "-g"], capture_output=True, text=True, check=True)
    result_m = subprocess.run(["free", "-m"], capture_output=True, text=True, check=True)
    mem_line_g = next(line for line in result_g.stdout.splitlines() if line.startswith("Mem:"))
    mem_line_m = next(line for line in result_m.stdout.splitlines() if line.startswith("Mem:"))
    return {
        "free_g_line": mem_line_g,
        "free_m_line": mem_line_m,
        "available_gb": int(mem_line_g.split()[-1]),
        "available_mb": int(mem_line_m.split()[-1]),
    }


def wait_for_memory_gate(context, min_gb=3, poll_seconds=60, max_wait_seconds=1800):
    waited = 0
    reading = read_free_gb()
    while reading["available_gb"] < min_gb and waited < max_wait_seconds:
        print(
            f"[memory-gate] {context}: only {reading['available_gb']} GB available "
            f"(< {min_gb} GB); waiting {poll_seconds}s ({waited}s/{max_wait_seconds}s so far)",
            file=sys.stderr,
        )
        time.sleep(poll_seconds)
        waited += poll_seconds
        reading = read_free_gb()
    record = {
        "context": context,
        "waited_seconds": waited,
        "proceeded_below_threshold": reading["available_gb"] < min_gb,
        **reading,
    }
    MEMORY_READINGS.append(record)
    print(
        f"[memory-gate] {context}: proceeding, available_gb={reading['available_gb']} "
        f"(waited {waited}s)",
        file=sys.stderr,
    )
    return record


def kokkos_extra_args():
    return ["-k", "on", "g", "1", "-pk", "kokkos", "neigh", "half", "-sf", "kk"]


def run_kk_lmp(kk_lmp, deck_path, log_path, screen_path, *, context, mpiexec=None, ranks=1,
               timeout=180, memory_gate_kwargs=None):
    wait_for_memory_gate(context, **(memory_gate_kwargs or {}))
    result = run_lmp(
        str(kk_lmp), deck_path, log_path, screen_path, mpiexec=mpiexec, ranks=ranks,
        timeout=timeout, extra_args=kokkos_extra_args(),
    )
    log_text = Path(log_path).read_text(encoding="utf-8")
    if "YE3T tagged Kokkos dispatch:" not in log_text or "YE3T tagged-Cauchy CPU dispatch:" in log_text:
        raise AssertionError(
            f"{context}: the tagged Kokkos device path was not dispatched (see {log_path})"
        )
    return result


# ---------------------------------------------------------------------
# Tight CPU/KK numeric parity
# ---------------------------------------------------------------------


def max_abs_field_diff(left, right, fields):
    worst = 0.0
    worst_where = None
    for atom_id in left:
        for field in fields:
            diff = abs(left[atom_id][field] - right[atom_id][field])
            if diff > worst:
                worst = diff
                worst_where = (atom_id, field)
    return worst, worst_where


def require_device_parity(left, right, label, fields=("fx", "fy", "fz", "c_atom_energy"), tol=DEVICE_PARITY_TOL):
    if left.keys() != right.keys():
        raise AssertionError(f"{label}: atom ID sets differ")
    worst, where = max_abs_field_diff(left, right, fields)
    if worst > tol:
        raise AssertionError(f"{label}: parity {worst:.3e} at {where} exceeds {tol:.3e}")
    return worst


# ---------------------------------------------------------------------
# Check 1: CPU/Python parity k1/k2 + numdiff on frame0, through the device
# ---------------------------------------------------------------------


def device_structure_check(*, cpu_lmp, kk_lmp, model_path, element, cutoff, structure, reference,
                           per_atom_python, output_dir, label, numdiff, memory_gate_kwargs):
    positions = structure["positions_angstrom"]
    cell = structure.get("cell_angstrom")
    pbc = structure.get("pbc", (False, False, False))
    n_atoms = structure["n_atoms"]

    cpu_dump = output_dir / f"{label}.cpu.dump"
    kk_dump = output_dir / f"{label}.kk.dump"
    cpu_deck_path = output_dir / f"in.{label}.cpu"
    kk_deck_path = output_dir / f"in.{label}.kk"
    cpu_log = output_dir / f"log.{label}.cpu"
    kk_log = output_dir / f"log.{label}.kk"
    cpu_screen = output_dir / f"screen.{label}.cpu"
    kk_screen = output_dir / f"screen.{label}.kk"

    cpu_deck_path.write_text(
        render_single_point_deck(
            model_path=model_path, element=element, cutoff=cutoff, positions=positions,
            cell=cell, pbc=pbc, snapshot_path=cpu_dump, numdiff=numdiff, marker_prefix="YE3T_TAGGED",
        ),
        encoding="utf-8",
    )
    kk_deck_path.write_text(
        render_single_point_deck(
            model_path=model_path, element=element, cutoff=cutoff, positions=positions,
            cell=cell, pbc=pbc, snapshot_path=kk_dump, numdiff=numdiff, marker_prefix="YE3T_TAGGED",
        ),
        encoding="utf-8",
    )
    run_lmp(str(cpu_lmp), cpu_deck_path, cpu_log, cpu_screen)
    run_kk_lmp(kk_lmp, kk_deck_path, kk_log, kk_screen, context=label, memory_gate_kwargs=memory_gate_kwargs)

    cpu_frames = read_dump(cpu_dump)
    kk_frames = read_dump(kk_dump)
    if len(cpu_frames) != 1 or len(kk_frames) != 1:
        raise AssertionError(f"{label}: expected one dump frame per backend")
    _, cpu_records = cpu_frames[0]
    _, kk_records = kk_frames[0]
    if sorted(kk_records) != list(range(1, n_atoms + 1)):
        raise AssertionError(f"{label}: unexpected atom IDs in the Kokkos dump")

    cpu_kk_parity = require_device_parity(cpu_records, kk_records, f"{label} CPU/KK")

    errors = []
    energy_sum = 0.0
    max_force_error = 0.0
    max_per_atom_energy_error = 0.0
    for atom_id in range(1, n_atoms + 1):
        record = kk_records[atom_id]
        expected_force = reference["forces_ev_per_angstrom"][atom_id - 1]
        for component, field in enumerate(("fx", "fy", "fz")):
            error = require_close(
                record[field], expected_force[component],
                f"{label} kk atom {atom_id} {field}", FORCE_TOL, errors,
            )
            max_force_error = max(max_force_error, error)
        energy_sum += record["c_atom_energy"]
        error = require_close(
            record["c_atom_energy"], float(per_atom_python[atom_id - 1]),
            f"{label} kk atom {atom_id} per-atom energy vs Python exact_moment_reduction",
            PER_ATOM_ENERGY_TOL, errors,
        )
        max_per_atom_energy_error = max(max_per_atom_energy_error, error)

    total_energy_kk = marker(kk_log, "YE3T_TAGGED_PE")
    require_close(total_energy_kk, reference["energy_ev"], f"{label} kk total energy (log)", ENERGY_TOL, errors)
    require_close(energy_sum, reference["energy_ev"], f"{label} kk total energy (sum of per-atom)", ENERGY_TOL, errors)

    result = {
        "n_atoms": n_atoms,
        "cpu_kk_parity_max_abs": cpu_kk_parity,
        "kk_total_energy_error_eV": abs(total_energy_kk - reference["energy_ev"]),
        "kk_force_max_abs_error_eV_per_A": max_force_error,
        "kk_per_atom_energy_max_abs_error_eV": max_per_atom_energy_error,
    }
    if numdiff:
        force_fd = marker(kk_log, "YE3T_TAGGED_NUMDIFF_FORCE_MAX_ABS")
        virial_fd = marker(kk_log, "YE3T_TAGGED_NUMDIFF_VIRIAL_L2_EV")
        if force_fd > NUMDIFF_FORCE_TOL:
            errors.append(f"{label} kk numdiff force error {force_fd:.3e} > {NUMDIFF_FORCE_TOL:.3e}")
        if virial_fd > NUMDIFF_VIRIAL_TOL_EV:
            errors.append(f"{label} kk numdiff virial error {virial_fd:.3e} eV > {NUMDIFF_VIRIAL_TOL_EV:.3e}")
        result["kk_numdiff_force_max_abs_eV_per_A"] = force_fd
        result["kk_numdiff_virial_l2_eV"] = virial_fd

    if errors:
        raise AssertionError(f"{label}: " + "; ".join(errors[:20]) + (" ..." if len(errors) > 20 else ""))
    return result


# ---------------------------------------------------------------------
# Check 2: MPI 1/2/4 ranks on one (oversubscribed) GPU
# ---------------------------------------------------------------------


def validate_mpi_kokkos(*, cpu_lmp, kk_lmp, mpiexec, model_path, element, output_dir, memory_gate_kwargs):
    # Same BCC a=3.300, region 0..4 lattice units box (edge 13.2 A) as
    # validate_fitted_tagged_cauchy.render_bcc_mpi_deck's own MPI check;
    # (6.6, 3.3, 3.3) is simultaneously the rank-2 midpoint boundary and a
    # rank-4 quarter boundary, and a BCC corner lattice site there.
    mover_point = (6.6, 3.3, 3.3)
    frames = {}
    per_rank = []

    cpu_deck = output_dir / "in.kk_mpi_anchor.cpu"
    cpu_initial = output_dir / "kk_mpi_anchor.cpu.initial.dump"
    cpu_final = output_dir / "kk_mpi_anchor.cpu.final.dump"
    cpu_deck.write_text(
        render_bcc_mpi_deck(model_path=model_path, element=element, ranks=1,
                            mover_seed_point=mover_point, dump_initial=cpu_initial, dump_final=cpu_final),
        encoding="utf-8",
    )
    run_lmp(str(cpu_lmp), cpu_deck, output_dir / "log.kk_mpi_anchor.cpu",
           output_dir / "screen.kk_mpi_anchor.cpu")
    cpu_anchor_initial = one_frame(cpu_initial)
    cpu_anchor_final = one_frame(cpu_final)

    for ranks in (1, 2, 4):
        label = f"kk.mpi.rank{ranks}"
        deck_path = output_dir / f"in.{label}"
        initial = output_dir / f"{label}.initial.dump"
        final = output_dir / f"{label}.final.dump"
        deck_path.write_text(
            render_bcc_mpi_deck(model_path=model_path, element=element, ranks=ranks,
                                mover_seed_point=mover_point, dump_initial=initial, dump_final=final),
            encoding="utf-8",
        )
        run_kk_lmp(kk_lmp, deck_path, output_dir / f"log.{label}", output_dir / f"screen.{label}",
                  context=label, mpiexec=mpiexec, ranks=ranks, memory_gate_kwargs=memory_gate_kwargs)

        initial_records = one_frame(initial)
        final_records = one_frame(final)
        owners = {int(record["proc"]) for record in initial_records.values()}
        if len(owners) != ranks:
            raise AssertionError(
                f"{label}: {len(owners)} MPI ranks own atoms, expected {ranks} "
                "(oversubscribed on one GPU)"
            )
        net_force = math.sqrt(
            sum(sum(record[f] for record in final_records.values()) ** 2 for f in ("fx", "fy", "fz"))
        )
        frames[ranks] = (initial_records, final_records)
        per_rank.append({"ranks": ranks, "owning_ranks": len(owners), "net_force_norm_eV_per_A": net_force})

    anchor_initial, anchor_final = frames[1]
    parity = {
        "kk_rank1_vs_cpu_rank1_initial": require_device_parity(
            cpu_anchor_initial, anchor_initial, "kk rank1 vs cpu rank1 initial",
            fields=("fx", "fy", "fz", "c_atom_energy"),
        ),
        "kk_rank1_vs_cpu_rank1_final": require_device_parity(
            cpu_anchor_final, anchor_final, "kk rank1 vs cpu rank1 final",
            fields=("fx", "fy", "fz", "c_atom_energy"),
        ),
    }
    for ranks in (2, 4):
        initial, final = frames[ranks]
        parity[f"kk_rank1_vs_rank{ranks}_initial"] = require_device_parity(
            anchor_initial, initial, f"kk rank1 vs rank{ranks} initial",
            fields=("fx", "fy", "fz", "c_atom_energy"),
        )
        parity[f"kk_rank1_vs_rank{ranks}_final"] = require_device_parity(
            anchor_final, final, f"kk rank1 vs rank{ranks} final",
            fields=("fx", "fy", "fz", "c_atom_energy"),
        )
    return {"per_rank": per_rank, "parity_max_abs": parity}


# ---------------------------------------------------------------------
# Check 3: multi-species fixtures through the device path
# ---------------------------------------------------------------------


def validate_multispecies_device(*, cpu_lmp, kk_lmp, model_path, ta_fixture_path, w_fixture_path,
                                  output_dir, memory_gate_kwargs):
    model = json.loads(Path(model_path).read_text(encoding="utf-8"))
    species_order = model["species_order"]
    cutoff = float(model["cutoff"])
    results = {}
    for fixture_path in (ta_fixture_path, w_fixture_path):
        fixture = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
        label = fixture["label"]
        central_species = fixture["central_species"]
        edges = fixture["edges"]

        positions = [[0.0, 0.0, 0.0]]
        types = [species_order.index(central_species) + 1]
        for edge in edges:
            d = edge["displacement_A"]
            positions.append([float(d[0]), float(d[1]), float(d[2])])
            types.append(species_order.index(edge["neighbor_species"]) + 1)
        positions_array = np.asarray(positions, dtype=np.float64)
        lo = positions_array.min(axis=0) - (cutoff + 2.0)
        hi = positions_array.max(axis=0) + (cutoff + 2.0)

        lines = [
            "units metal", "atom_style atomic", "boundary f f f",
            "atom_modify map yes sort 0 0.0", "newton on", "",
            "region cell block "
            f"{fnum(lo[0])} {fnum(hi[0])} {fnum(lo[1])} {fnum(hi[1])} {fnum(lo[2])} {fnum(hi[2])} units box",
            f"create_box {len(species_order)} cell",
        ]
        for (x, y, z), atom_type in zip(positions_array, types):
            lines.append(f"create_atoms {atom_type} single {fnum(x)} {fnum(y)} {fnum(z)} units box")
        for species_index in range(1, len(species_order) + 1):
            lines.append(f"mass {species_index} 1.0")
        lines += ["", "neighbor 0.3 bin", "neigh_modify every 1 delay 0 check yes", ""]
        lines += [
            "pair_style ye3t model_family tagged_cauchy chunksize 8",
            f"pair_coeff * * {model_path} " + " ".join(species_order),
            "",
            "compute atom_energy all pe/atom",
            "compute force_sum all reduce sum fx fy fz",
            "",
            "thermo 1",
            "thermo_style custom step pe c_force_sum[1] c_force_sum[2] c_force_sum[3]",
            "thermo_modify format float %.17g",
            "",
        ]
        deck_template = "\n".join(lines) + "\n"

        cpu_dump = output_dir / f"multispecies.{label}.cpu.dump"
        kk_dump = output_dir / f"multispecies.{label}.kk.dump"
        dump_tail = (
            "dump snapshot all custom 1 {dump} id type x y z c_atom_energy fx fy fz\n"
            "dump_modify snapshot sort id format float %.17g\n"
            "run 0 post no\n"
        )
        cpu_deck_path = output_dir / f"in.multispecies.{label}.cpu"
        kk_deck_path = output_dir / f"in.multispecies.{label}.kk"
        cpu_deck_path.write_text(deck_template + dump_tail.format(dump=cpu_dump), encoding="utf-8")
        kk_deck_path.write_text(deck_template + dump_tail.format(dump=kk_dump), encoding="utf-8")

        run_lmp(str(cpu_lmp), cpu_deck_path, output_dir / f"log.multispecies.{label}.cpu",
               output_dir / f"screen.multispecies.{label}.cpu")
        run_kk_lmp(kk_lmp, kk_deck_path, output_dir / f"log.multispecies.{label}.kk",
                  output_dir / f"screen.multispecies.{label}.kk",
                  context=f"multispecies.{label}", memory_gate_kwargs=memory_gate_kwargs)

        cpu_records = one_frame(cpu_dump)
        kk_records = one_frame(kk_dump)
        parity = require_device_parity(cpu_records, kk_records, f"multispecies {label} CPU/KK")

        central_cpu_energy = cpu_records[1]["c_atom_energy"]
        central_kk_energy = kk_records[1]["c_atom_energy"]
        fixture_energy = float(fixture["energy_eV"])
        central_vs_fixture_cpu = abs(central_cpu_energy - fixture_energy)
        central_vs_fixture_kk = abs(central_kk_energy - fixture_energy)
        # The central atom's own neighbor list in this deck is exactly the
        # fixture's `edges` (no other atom is within cutoff of it), so its
        # per-atom energy is independently comparable to the fixture's
        # energy_eV; atoms 2..N may see each other too and are not checked
        # against the fixture (their environments were never encoded by
        # this single-center fixture), only against each other (CPU vs KK).
        if central_vs_fixture_cpu > 1.0e-8 or central_vs_fixture_kk > 1.0e-8:
            raise AssertionError(
                f"multispecies {label}: central-atom energy vs fixture energy_eV differs "
                f"(cpu={central_vs_fixture_cpu:.3e}, kk={central_vs_fixture_kk:.3e})"
            )
        results[label] = {
            "central_species": central_species,
            "neighbor_count": len(edges),
            "cpu_kk_parity_max_abs": parity,
            "central_atom_energy_vs_fixture_error_eV": {
                "cpu": central_vs_fixture_cpu, "kk": central_vs_fixture_kk,
            },
        }
    return results


# ---------------------------------------------------------------------
# Check 5: NVE drift on the device
# ---------------------------------------------------------------------


def validate_nve_kokkos(*, kk_lmp, model_path, element, output_dir, memory_gate_kwargs,
                        steps=500, timestep=0.001, seed=934817):
    label = "kk.nve"
    deck_path = output_dir / f"in.{label}"
    dump_path = output_dir / f"{label}.dump"
    log_path = output_dir / f"log.{label}"
    screen_path = output_dir / f"screen.{label}"
    deck_path.write_text(
        render_nve_deck(model_path=model_path, element=element, dump_path=dump_path,
                        steps=steps, timestep=timestep, seed=seed),
        encoding="utf-8",
    )
    run_kk_lmp(kk_lmp, deck_path, log_path, screen_path, context=label, timeout=600,
              memory_gate_kwargs=memory_gate_kwargs)

    initial = marker(log_path, "YE3T_TAGGED_NVE_INITIAL_ETOTAL")
    final = marker(log_path, "YE3T_TAGGED_NVE_FINAL_ETOTAL")
    text = Path(log_path).read_text(encoding="utf-8")
    etotal_values = []
    in_thermo = False
    header = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Step") and stripped.split()[0] == "Step":
            in_thermo = True
            header = stripped.split()
            etotal_col = header.index("TotEng")
            continue
        if in_thermo:
            parts = stripped.split()
            if header is not None and len(parts) == len(header):
                try:
                    etotal_values.append(float(parts[etotal_col]))
                    continue
                except ValueError:
                    pass
            in_thermo = False
    drift_abs = max(abs(v - initial) for v in etotal_values) if etotal_values else abs(final - initial)
    n_atoms = 128
    return {
        "steps": steps,
        "n_atoms": n_atoms,
        "initial_etotal_eV": initial,
        "final_etotal_eV": final,
        "max_abs_drift_eV": drift_abs,
        "max_abs_drift_eV_per_atom": drift_abs / n_atoms,
        "thermo_sample_count": len(etotal_values),
    }


# ---------------------------------------------------------------------
# Check 6: timing, 128-atom and 1024-atom cells, device vs CPU-tagged vs
# pace/kk product, GPU memory reported
# ---------------------------------------------------------------------


def render_timing_deck_sized(*, pair_style_line, pair_coeff_line, steps, replication):
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


def measure_timing_sized(*, lmp, label, pair_style_line, pair_coeff_line, output_dir,
                         replication, steps=200, extra_args=()):
    n_atoms = 2 * replication ** 3
    deck_path = output_dir / f"in.timing.{label}"
    log_path = output_dir / f"log.timing.{label}"
    screen_path = output_dir / f"screen.timing.{label}"
    deck_path.write_text(
        render_timing_deck_sized(pair_style_line=pair_style_line, pair_coeff_line=pair_coeff_line,
                                 steps=steps, replication=replication),
        encoding="utf-8",
    )
    start = time.perf_counter()
    run_lmp(str(lmp), deck_path, log_path, screen_path, timeout=300, extra_args=extra_args)
    wall_seconds = time.perf_counter() - start

    text = Path(log_path).read_text(encoding="utf-8")
    loop_seconds = None
    for line in text.splitlines():
        if line.startswith("Loop time of"):
            loop_seconds = float(line.split()[3])
            break
    free_bytes = total_bytes = None
    memory_match = re.search(r"free_device_bytes (\d+), total_device_bytes (\d+)", text)
    if memory_match:
        free_bytes, total_bytes = int(memory_match.group(1)), int(memory_match.group(2))
    seconds = loop_seconds if loop_seconds is not None else wall_seconds
    return {
        "label": label,
        "replication": replication,
        "n_atoms": n_atoms,
        "steps": steps,
        "loop_seconds": loop_seconds,
        "wall_seconds": wall_seconds,
        "microseconds_per_atom_step": seconds * 1.0e6 / (n_atoms * steps),
        "gpu_free_device_bytes": free_bytes,
        "gpu_total_device_bytes": total_bytes,
    }


CELLS = ((4, "128atom"), (8, "1024atom"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--cpu-lmp", type=Path, required=True)
    parser.add_argument("--kk-lmp", type=Path, required=True)
    parser.add_argument("--model-k1", type=Path, required=True)
    parser.add_argument("--reference-k1", type=Path, required=True)
    parser.add_argument("--model-k2", type=Path)
    parser.add_argument("--reference-k2", type=Path)
    parser.add_argument("--multispecies-model", type=Path)
    parser.add_argument("--multispecies-ta-fixture", type=Path)
    parser.add_argument("--multispecies-w-fixture", type=Path)
    parser.add_argument("--yace-model", type=Path)
    parser.add_argument("--pace-kk-lmp", type=Path)
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--min-free-gb", type=int, default=3)
    parser.add_argument("--memory-poll-seconds", type=int, default=60)
    parser.add_argument("--memory-max-wait-seconds", type=int, default=1800)
    parser.add_argument("--skip-mpi", action="store_true")
    parser.add_argument("--skip-multispecies", action="store_true")
    parser.add_argument("--skip-nve", action="store_true")
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    memory_gate_kwargs = {
        "min_gb": args.min_free_gb,
        "poll_seconds": args.memory_poll_seconds,
        "max_wait_seconds": args.memory_max_wait_seconds,
    }
    report = {"schema": "ye3t_fitted_tagged_cauchy_kokkos_validation_v1", "artifacts": {}}

    wait_for_memory_gate("startup", **memory_gate_kwargs)

    models = [("k1", args.model_k1, args.reference_k1)]
    if args.model_k2 and args.reference_k2:
        models.append(("k2", args.model_k2, args.reference_k2))

    for tag, model_path, reference_path in models:
        artifact_report = {"model_path": str(model_path)}
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        structures = reference["structures"]
        model, cross_check = cross_check_python(model_path, structures)
        artifact_report["python_cross_check_vs_reference"] = cross_check
        element = model.species_order[0]
        cutoff = model.cutoff

        structure_results = {}
        for label, structure in structures.items():
            atoms = _make_atoms(structure)
            per_atom_python = per_atom_energies_python(model, atoms)
            is_frame0 = label == "ta_frame0"
            structure_results[label] = device_structure_check(
                cpu_lmp=args.cpu_lmp, kk_lmp=args.kk_lmp, model_path=model_path, element=element,
                cutoff=cutoff, structure=structure, reference=structure,
                per_atom_python=per_atom_python, output_dir=args.output,
                label=f"{tag}.{label}", numdiff=is_frame0, memory_gate_kwargs=memory_gate_kwargs,
            )
        artifact_report["structures"] = structure_results

        if not args.skip_mpi:
            artifact_report["mpi_device"] = validate_mpi_kokkos(
                cpu_lmp=args.cpu_lmp, kk_lmp=args.kk_lmp, mpiexec=args.mpiexec,
                model_path=model_path, element=element, output_dir=args.output,
                memory_gate_kwargs=memory_gate_kwargs,
            )

        if not args.skip_nve and tag == "k2":
            artifact_report["nve_500step_300K_128atom_device"] = validate_nve_kokkos(
                kk_lmp=args.kk_lmp, model_path=model_path, element=element,
                output_dir=args.output, memory_gate_kwargs=memory_gate_kwargs,
            )

        if not args.skip_timing:
            timing = {}
            for replication, cell_label in CELLS:
                wait_for_memory_gate(f"timing.{tag}.{cell_label}.kk", **memory_gate_kwargs)
                timing[f"{cell_label}_device"] = measure_timing_sized(
                    lmp=args.kk_lmp, label=f"tagged_{tag}_{cell_label}_kk",
                    pair_style_line="pair_style ye3t model_family tagged_cauchy chunksize 32",
                    pair_coeff_line=f"pair_coeff * * {model_path} {element}",
                    output_dir=args.output, replication=replication, extra_args=kokkos_extra_args(),
                )
                timing[f"{cell_label}_cpu_tagged"] = measure_timing_sized(
                    lmp=args.cpu_lmp, label=f"tagged_{tag}_{cell_label}_cpu",
                    pair_style_line="pair_style ye3t model_family tagged_cauchy chunksize 32",
                    pair_coeff_line=f"pair_coeff * * {model_path} {element}",
                    output_dir=args.output, replication=replication,
                )
            artifact_report["timing"] = timing

        report["artifacts"][tag] = artifact_report

    if not args.skip_timing and args.yace_model:
        pace_lmp = args.pace_kk_lmp or args.kk_lmp
        pace_timing = {}
        for replication, cell_label in CELLS:
            wait_for_memory_gate(f"timing.pace.{cell_label}.kk", **memory_gate_kwargs)
            pace_timing[f"{cell_label}_pace_kk_product"] = measure_timing_sized(
                lmp=pace_lmp, label=f"pace_{cell_label}_kk",
                pair_style_line="pair_style pace product",
                pair_coeff_line=f"pair_coeff * * {args.yace_model} Ta",
                output_dir=args.output, replication=replication, extra_args=kokkos_extra_args(),
            )
        report["timing_pace_kk_product_reference"] = pace_timing

    if (not args.skip_multispecies and args.multispecies_model
            and args.multispecies_ta_fixture and args.multispecies_w_fixture):
        report["multispecies_device"] = validate_multispecies_device(
            cpu_lmp=args.cpu_lmp, kk_lmp=args.kk_lmp, model_path=args.multispecies_model,
            ta_fixture_path=args.multispecies_ta_fixture, w_fixture_path=args.multispecies_w_fixture,
            output_dir=args.output, memory_gate_kwargs=memory_gate_kwargs,
        )
    else:
        report["multispecies_device"] = (
            "NOT_RUN: --multispecies-model/--multispecies-ta-fixture/--multispecies-w-fixture "
            "not all specified, or --skip-multispecies was passed"
        )

    report["memory_readings"] = MEMORY_READINGS
    report["memory_gate_policy"] = {
        "min_free_gb": args.min_free_gb,
        "poll_seconds": args.memory_poll_seconds,
        "max_wait_seconds": args.memory_max_wait_seconds,
        "note": (
            "Free-memory guard: checked before every GPU-dispatching LAMMPS "
            "invocation, sleeping in poll_seconds steps up to max_wait_seconds "
            "if fewer than min_free_gb GB are available."
        ),
    }

    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
