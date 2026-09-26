#!/usr/bin/env python3
"""Validate a tagged-Cauchy V3 CPU or experimental Kokkos runtime.

The checked-in V3 artifact is a deterministic software fixture produced by a
small synthetic ridge solve.  It is not a fitted tantalum potential and is not
evidence of physical accuracy or molecular-dynamics stability.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from test_lifted_cauchy_lammps import marker, read_dump  # noqa: E402

from ye3t_ace.tagged_cauchy_image import load_tagged_cauchy_image_model


MASS_TA = 180.94788
METAL_NKTV2P = 1.6021765e6
ENERGY_TOL = 2.0e-8
PER_ATOM_ENERGY_TOL = 2.0e-8
FORCE_TOL = 2.0e-8
VIRIAL_TOL = 2.0e-7
MPI_PARITY_TOL = 2.0e-8
NUMDIFF_FORCE_TOL = 2.0e-5
NUMDIFF_VIRIAL_TOL = 2.0e-4


def fnum(value):
    return f"{float(value):.17g}"


def bcc_fixture(repetitions=3, lattice_constant=3.3):
    positions = []
    for i in range(repetitions):
        for j in range(repetitions):
            for k in range(repetitions):
                origin = np.asarray((i, j, k), dtype=np.float64) * lattice_constant
                positions.append(origin)
                positions.append(origin + 0.5 * lattice_constant)
    positions = np.asarray(positions, dtype=np.float64)
    box_length = repetitions * lattice_constant
    boundary = 0.5 * box_length
    candidates = np.flatnonzero(np.isclose(positions[:, 0], boundary))
    if not len(candidates):
        raise AssertionError("BCC fixture has no atom on the two-rank boundary.")
    mover = int(candidates[0])
    positions[mover, 0] -= 0.05
    return positions, np.eye(3) * box_length, mover


def write_data(path, positions, cell):
    lengths = np.diag(cell)
    lines = [
        "Tagged-Cauchy V3 deterministic validation fixture",
        "",
        f"{len(positions)} atoms",
        "1 atom types",
        "",
        f"0.0 {fnum(lengths[0])} xlo xhi",
        f"0.0 {fnum(lengths[1])} ylo yhi",
        f"0.0 {fnum(lengths[2])} zlo zhi",
        "",
        "Masses",
        "",
        f"1 {MASS_TA}",
        "",
        "Atoms # atomic",
        "",
    ]
    for atom_id, row in enumerate(positions, 1):
        lines.append(
            f"{atom_id} 1 {fnum(row[0])} {fnum(row[1])} {fnum(row[2])}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_deck(data_path, model_path, ranks, mover_id, initial_dump, final_dump,
                numdiff):
    lines = [
        "units metal",
        "atom_style atomic",
        "boundary p p p",
        "atom_modify map yes sort 0 0.0",
        "newton on",
        f"processors {ranks} 1 1",
        f"read_data {data_path}",
        "",
        "neighbor 0.3 bin",
        "neigh_modify every 1 delay 0 check yes",
        "",
        "pair_style ye3t model_family tagged_cauchy chunksize 32",
        f"pair_coeff * * {model_path} Ta",
        "",
        "compute atom_energy all pe/atom",
        "compute pair_pressure all pressure NULL pair",
        "compute force_sum all reduce sum fx fy fz",
    ]
    if numdiff:
        lines += [
            "fix numerical_force all numdiff 1 1.0e-5",
            "fix numerical_virial all numdiff/virial 1 1.0e-6",
            "variable dfx atom f_numerical_force[1]-fx",
            "variable dfy atom f_numerical_force[2]-fy",
            "variable dfz atom f_numerical_force[3]-fz",
            "variable dfmag atom sqrt(v_dfx*v_dfx+v_dfy*v_dfy+v_dfz*v_dfz)",
            "compute force_error all reduce max v_dfmag",
            "variable dv1 equal f_numerical_virial[1]-c_pair_pressure[1]",
            "variable dv2 equal f_numerical_virial[2]-c_pair_pressure[2]",
            "variable dv3 equal f_numerical_virial[3]-c_pair_pressure[3]",
            "variable dv4 equal f_numerical_virial[4]-c_pair_pressure[6]",
            "variable dv5 equal f_numerical_virial[5]-c_pair_pressure[5]",
            "variable dv6 equal f_numerical_virial[6]-c_pair_pressure[4]",
            "variable vdiff equal sqrt(v_dv1*v_dv1+v_dv2*v_dv2+v_dv3*v_dv3+"
            "v_dv4*v_dv4+v_dv5*v_dv5+v_dv6*v_dv6)*vol/1.6021765e6",
        ]
    lines += [
        "thermo 1",
        "thermo_style custom step atoms vol pe c_pair_pressure[1] "
        "c_pair_pressure[2] c_pair_pressure[3] c_pair_pressure[4] "
        "c_pair_pressure[5] c_pair_pressure[6] c_force_sum[1] "
        "c_force_sum[2] c_force_sum[3]",
        "thermo_modify format float %.17g",
        "run 0 post yes",
        f"write_dump all custom {initial_dump} id type proc x y z "
        "c_atom_energy fx fy fz modify sort id format float %.17g",
        'print "YE3T_V3_INITIAL_PE=$(pe:%.17g)"',
    ]
    for index, name in enumerate(("XX", "YY", "ZZ", "XY", "XZ", "YZ"), 1):
        pressure_index = (1, 2, 3, 4, 5, 6)[index - 1]
        lines.append(
            f'print "YE3T_V3_INITIAL_VIRIAL_{name}='
            f'$(c_pair_pressure[{pressure_index}]*vol/1.6021765e6:%.17g)"'
        )
    if numdiff:
        lines += [
            'print "YE3T_V3_NUMDIFF_FORCE_MAX=$(c_force_error:%.17g)"',
            'print "YE3T_V3_NUMDIFF_VIRIAL_L2_EV=$(v_vdiff:%.17g)"',
            "unfix numerical_force",
            "unfix numerical_virial",
        ]
    lines += [
        f"group mover id {mover_id}",
        "timestep 0.001",
        # Move farther than half the 0.3-A neighbor skin so LAMMPS rebuilds
        # the neighbor list and migrates ownership after crossing x=Lx/2.
        "fix cross_domain mover move linear 200.0 0.0 0.0 units box",
        "run 1 post yes",
        f"write_dump all custom {final_dump} id type proc x y z "
        "c_atom_energy fx fy fz modify sort id format float %.17g",
        'print "YE3T_V3_FINAL_PE=$(pe:%.17g)"',
    ]
    for index, name in enumerate(("XX", "YY", "ZZ", "XY", "XZ", "YZ"), 1):
        pressure_index = (1, 2, 3, 4, 5, 6)[index - 1]
        lines.append(
            f'print "YE3T_V3_FINAL_VIRIAL_{name}='
            f'$(c_pair_pressure[{pressure_index}]*vol/1.6021765e6:%.17g)"'
        )
    return "\n".join(lines) + "\n"


def run_lammps(
    lmp,
    mpiexec,
    ranks,
    deck,
    log_path,
    screen_path,
    timeout,
    kokkos,
    kokkos_gpus,
):
    command = [str(lmp)]
    if ranks > 1:
        command = [str(mpiexec), "-n", str(ranks), str(lmp)]
    if kokkos:
        command += [
            "-k",
            "on",
            "g",
            str(kokkos_gpus),
            "-pk",
            "kokkos",
            "neigh",
            "half",
            "-sf",
            "kk",
        ]
    command += ["-in", str(deck), "-log", str(log_path), "-screen", str(screen_path)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(
            f"LAMMPS rank-{ranks} validation failed with {result.returncode}.\n"
            f"stdout tail:\n{result.stdout[-3000:]}\n"
            f"stderr tail:\n{result.stderr[-3000:]}"
        )
    if kokkos:
        text = log_path.read_text(encoding="utf-8")
        required = (
            "YE3T tagged Kokkos dispatch:",
            "evaluator physical_image_v3_direct",
            "qualification experimental_reference_unqualified",
        )
        if any(value not in text for value in required) or (
            "YE3T tagged-Cauchy CPU dispatch:" in text
        ):
            raise AssertionError(
                "The requested experimental tagged-Cauchy Kokkos path did not "
                f"dispatch on rank count {ranks}."
            )


def python_reference(model, records, cell):
    positions = np.asarray(
        [[records[index][axis] for axis in ("x", "y", "z")]
         for index in sorted(records)],
        dtype=np.float64,
    )
    atom_types = torch.zeros(len(positions), dtype=torch.long)
    return model.energy_forces_virial(
        torch.tensor(positions, dtype=torch.float64),
        atom_types,
        cell=torch.tensor(cell, dtype=torch.float64),
        pbc=(True, True, True),
    )


def frame_errors(model, records, cell, log_path, stage):
    energy, forces, virial, atomic_energy = python_reference(model, records, cell)
    errors = {
        "total_energy_eV": abs(marker(log_path, f"YE3T_V3_{stage}_PE") - float(energy)),
        "per_atom_energy_eV": 0.0,
        "force_eV_per_A": 0.0,
        "virial_eV": 0.0,
    }
    for atom_index, atom_id in enumerate(sorted(records)):
        record = records[atom_id]
        errors["per_atom_energy_eV"] = max(
            errors["per_atom_energy_eV"],
            abs(record["c_atom_energy"] - float(atomic_energy[atom_index])),
        )
        for axis, field in enumerate(("fx", "fy", "fz")):
            errors["force_eV_per_A"] = max(
                errors["force_eV_per_A"],
                abs(record[field] - float(forces[atom_index, axis])),
            )
    for index, name in enumerate(("XX", "YY", "ZZ", "XY", "XZ", "YZ")):
        errors["virial_eV"] = max(
            errors["virial_eV"],
            abs(marker(log_path, f"YE3T_V3_{stage}_VIRIAL_{name}") - float(virial[index])),
        )
    if errors["total_energy_eV"] > ENERGY_TOL:
        raise AssertionError(f"{stage} total-energy mismatch: {errors}")
    if errors["per_atom_energy_eV"] > PER_ATOM_ENERGY_TOL:
        raise AssertionError(f"{stage} per-atom-energy mismatch: {errors}")
    if errors["force_eV_per_A"] > FORCE_TOL:
        raise AssertionError(f"{stage} force mismatch: {errors}")
    if errors["virial_eV"] > VIRIAL_TOL:
        raise AssertionError(f"{stage} virial mismatch: {errors}")
    return errors


def compare_rank_records(anchor, candidate, stage):
    maximum = 0.0
    for atom_id in anchor:
        for field in ("x", "y", "z", "c_atom_energy", "fx", "fy", "fz"):
            maximum = max(maximum, abs(anchor[atom_id][field] - candidate[atom_id][field]))
    if maximum > MPI_PARITY_TOL:
        raise AssertionError(
            f"{stage} rank-1/rank-2 maximum mismatch {maximum:.3e} exceeds "
            f"{MPI_PARITY_TOL:.3e}."
        )
    return maximum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmp", required=True, type=Path)
    parser.add_argument("--mpiexec", required=True, type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=THIS_DIR / "fixtures" / "tagged_cauchy_physical_image_v3.json",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--kokkos",
        action="store_true",
        help="validate the experimental, unqualified physical-image V3 /kk path",
    )
    parser.add_argument("--kokkos-gpus", type=int, default=1)
    args = parser.parse_args()

    if args.kokkos_gpus < 1:
        raise ValueError("--kokkos-gpus must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_tagged_cauchy_image_model(args.model)
    positions, cell, mover = bcc_fixture()
    data_path = args.output_dir / "tagged_cauchy_v3_bcc.data"
    write_data(data_path, positions, cell)

    reports = {}
    records_by_rank = {}
    owner_change = None
    for ranks in (1, 2):
        prefix = args.output_dir / f"rank{ranks}"
        deck_path = prefix.with_suffix(".in")
        log_path = prefix.with_suffix(".log")
        screen_path = prefix.with_suffix(".screen")
        initial_dump = args.output_dir / f"rank{ranks}.initial.dump"
        final_dump = args.output_dir / f"rank{ranks}.final.dump"
        deck_path.write_text(
            render_deck(
                data_path.resolve(), args.model.resolve(), ranks, mover + 1,
                initial_dump.resolve(), final_dump.resolve(), ranks == 1,
            ),
            encoding="utf-8",
        )
        run_lammps(
            args.lmp, args.mpiexec, ranks, deck_path, log_path, screen_path,
            args.timeout, args.kokkos, args.kokkos_gpus,
        )
        initial = read_dump(initial_dump)[0][1]
        final = read_dump(final_dump)[0][1]
        records_by_rank[ranks] = (initial, final)
        reports[f"rank{ranks}"] = {
            "initial": frame_errors(model, initial, cell, log_path, "INITIAL"),
            "final": frame_errors(model, final, cell, log_path, "FINAL"),
        }
        if ranks == 1:
            force_fd = marker(log_path, "YE3T_V3_NUMDIFF_FORCE_MAX")
            virial_fd = marker(log_path, "YE3T_V3_NUMDIFF_VIRIAL_L2_EV")
            if force_fd > NUMDIFF_FORCE_TOL or virial_fd > NUMDIFF_VIRIAL_TOL:
                raise AssertionError(
                    "LAMMPS numerical differentiation failed: "
                    f"force={force_fd:.3e}, virial={virial_fd:.3e}."
                )
            reports["rank1"]["numdiff_force_eV_per_A"] = force_fd
            reports["rank1"]["numdiff_virial_l2_eV"] = virial_fd
        else:
            owner_change = (
                int(initial[mover + 1]["proc"]), int(final[mover + 1]["proc"])
            )
            if owner_change[0] == owner_change[1]:
                raise AssertionError("The selected atom did not cross an MPI domain.")

    anchor_initial, anchor_final = records_by_rank[1]
    rank2_initial, rank2_final = records_by_rank[2]
    reports["mpi"] = {
        "initial_rank_parity_max_abs": compare_rank_records(
            anchor_initial, rank2_initial, "initial"
        ),
        "final_rank_parity_max_abs": compare_rank_records(
            anchor_final, rank2_final, "final"
        ),
        "mover_owner_before_after": owner_change,
    }
    reports["runtime"] = {
        "backend": "kokkos" if args.kokkos else "cpu",
        "qualification": (
            "experimental_reference_unqualified" if args.kokkos else "cpu_reference"
        ),
        "kokkos_gpus_per_node": args.kokkos_gpus if args.kokkos else None,
        "two_rank_gpu_run": (
            "oversubscribed_correctness_only" if args.kokkos else None
        ),
    }
    report_path = args.output_dir / "tagged_cauchy_v3_lammps_report.json"
    report_path.write_text(
        json.dumps(reports, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(reports, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
