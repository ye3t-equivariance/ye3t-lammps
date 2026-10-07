#!/usr/bin/env python3
"""Validate fitted tagged-Cauchy artifacts against LAMMPS.

Exercises `pair_style ye3t model_family tagged_cauchy` (the native CPU
evaluator; see `src/ye3t_tagged_cauchy_model.*` / `src/ye3t_tagged_cauchy_
cpu.*`) the way `tests/validate_fitted_lifted_cauchy.py` /
`tests/test_lifted_cauchy_lammps.py` exercise the lifted-Cauchy family,
reusing their dump-parsing and LAMMPS-invocation conventions. Structure
data (positions/cell/pbc) and reference energies/forces come from
the reference JSON files (built from
`ye3t_methods.atomistic.tagged_cauchy_linear.energy_and_forces`, execution_strategy
`exact_moment_reduction`). Per-atom energies have no such precomputed
reference, so this script also computes them itself from the same
`exact_moment_reduction` evaluator (`TaggedCauchyModel.moment_evaluator.
descriptors(...) @ beta + offsets`, i.e. `energy_and_forces`'s own
per-atom vector before the final sum). The virial finite-strain check is
likewise computed here directly (central difference of `energy_and_forces`
under a small symmetric-strain deformation of the cell and positions),
independent of the reference file.

LAMMPS input decks are generated in-memory (not checked in as fixtures)
and run against the built `lmp` binary.
"""

import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from test_lifted_cauchy_lammps import marker, read_dump  # noqa: E402

METAL_NKTV2P = 1.6021765e6
ENERGY_TOL = 1.0e-8
PER_ATOM_ENERGY_TOL = 1.0e-8
FORCE_TOL = 1.0e-8
VIRIAL_FD_TOL_EV = 1.0e-6
NUMDIFF_FORCE_TOL = 5.0e-6
NUMDIFF_VIRIAL_TOL_EV = 5.0e-5

MASS_TA = 180.94788


def fnum(value):
    return f"{float(value):.17g}"


def require_close(actual, expected, label, tol, errors):
    error = abs(actual - expected)
    ok = math.isfinite(actual) and error <= tol
    if not ok:
        errors.append(f"{label}: {actual!r} != {expected!r} (|error|={error:.6e} > tol {tol:.3e})")
    return error


def run_lmp(lmp, deck_path, log_path, screen_path, mpiexec=None, ranks=1, timeout=180, extra_args=()):
    if mpiexec and ranks > 1:
        cmd = [mpiexec, "-n", str(ranks), lmp]
    else:
        cmd = [lmp]
    cmd += ["-screen", str(screen_path), "-log", str(log_path), "-in", str(deck_path), *extra_args]
    result = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"lmp failed (exit {result.returncode}): {' '.join(cmd)}\n"
            f"--- stdout tail ---\n{result.stdout[-4000:]}\n--- stderr tail ---\n{result.stderr[-4000:]}"
        )
    return result


def render_single_point_deck(
    *, model_path, element, cutoff, positions, cell, pbc, snapshot_path,
    chunksize=8, numdiff=False, force_delta=1.0e-5, virial_delta=1.0e-6,
    marker_prefix="YE3T_TAGGED",
):
    positions = np.asarray(positions, dtype=np.float64)
    lines = [
        "units metal",
        "atom_style atomic",
    ]
    if all(pbc):
        lines.append("boundary p p p")
    else:
        lines.append("boundary f f f")
    lines += ["atom_modify map yes sort 1 0.0", "newton on", ""]

    if all(pbc):
        cell = np.asarray(cell, dtype=np.float64)
        off_diagonal = [cell[0, 1], cell[0, 2], cell[1, 0], cell[1, 2], cell[2, 0], cell[2, 1]]
        if max(abs(v) for v in off_diagonal) > 1.0e-8:
            raise ValueError("render_single_point_deck only supports an orthogonal cell")
        lx, ly, lz = cell[0, 0], cell[1, 1], cell[2, 2]
        lines.append(f"region cell block 0 {fnum(lx)} 0 {fnum(ly)} 0 {fnum(lz)} units box")
    else:
        lo = positions.min(axis=0) - (cutoff + 2.0)
        hi = positions.max(axis=0) + (cutoff + 2.0)
        lines.append(
            "region cell block "
            f"{fnum(lo[0])} {fnum(hi[0])} {fnum(lo[1])} {fnum(hi[1])} {fnum(lo[2])} {fnum(hi[2])} units box"
        )
    lines.append("create_box 1 cell")
    for row in positions:
        lines.append(f"create_atoms 1 single {fnum(row[0])} {fnum(row[1])} {fnum(row[2])} units box")
    # No `reset_atoms id sort yes` here: create_atoms 1 single already assigns
    # ids 1..n in call order on a fresh single-rank box, and this script's
    # dump-to-reference-array mapping (atom id i <-> positions[i-1]) depends
    # on that order being preserved, not resorted spatially.
    lines += [f"mass 1 {MASS_TA}", ""]
    lines += ["neighbor 0.3 bin", "neigh_modify every 1 delay 0 check yes", ""]
    lines += [
        f"pair_style ye3t model_family tagged_cauchy chunksize {chunksize}",
        f"pair_coeff * * {model_path} {element}",
        "",
        "compute atom_energy all pe/atom",
        "compute atom_stress all stress/atom NULL pair",
        "compute pair_pressure all pressure NULL pair",
        "compute force_sum all reduce sum fx fy fz",
        "",
    ]
    if numdiff:
        lines += [
            f"fix numerical_force all numdiff 1 {fnum(force_delta)}",
            f"fix numerical_virial all numdiff/virial 1 {fnum(virial_delta)}",
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
            "variable virial_error_bar equal sqrt(v_dv1*v_dv1+v_dv2*v_dv2+"
            "v_dv3*v_dv3+v_dv4*v_dv4+v_dv5*v_dv5+v_dv6*v_dv6)",
            "variable virial_error_eV equal v_virial_error_bar*vol/1.6021765e6",
        ]
    lines += [
        "variable virial_xx equal c_pair_pressure[1]*vol/1.6021765e6",
        "variable virial_yy equal c_pair_pressure[2]*vol/1.6021765e6",
        "variable virial_zz equal c_pair_pressure[3]*vol/1.6021765e6",
        "variable virial_xy equal c_pair_pressure[4]*vol/1.6021765e6",
        "variable virial_xz equal c_pair_pressure[5]*vol/1.6021765e6",
        "variable virial_yz equal c_pair_pressure[6]*vol/1.6021765e6",
        "",
        "thermo 1",
        "thermo_style custom step atoms vol pe c_pair_pressure[1] c_pair_pressure[2] "
        "c_pair_pressure[3] c_pair_pressure[4] c_pair_pressure[5] c_pair_pressure[6] "
        "c_force_sum[1] c_force_sum[2] c_force_sum[3]",
        "thermo_modify format float %.17g",
        "",
        f"dump snapshot all custom 1 {snapshot_path} id type x y z c_atom_energy "
        "c_atom_stress[1] c_atom_stress[2] c_atom_stress[3] c_atom_stress[4] "
        "c_atom_stress[5] c_atom_stress[6] fx fy fz",
        "dump_modify snapshot sort id format float %.17g",
        "run 0 post no",
        "",
        f'print "{marker_prefix}_PE=$(pe:%.17g)"',
    ]
    if numdiff:
        lines += [
            f'print "{marker_prefix}_NUMDIFF_FORCE_MAX_ABS=$(c_force_error:%.17g)"',
            f'print "{marker_prefix}_NUMDIFF_VIRIAL_L2_EV=$(v_virial_error_eV:%.17g)"',
        ]
    for name in ("XX", "YY", "ZZ", "XY", "XZ", "YZ"):
        lines.append(f'print "{marker_prefix}_VIRIAL_{name}=$(v_virial_{name.lower()}:%.17g)"')
    return "\n".join(lines) + "\n"


def validate_structure(*, lmp, model_path, element, cutoff, structure, reference,
                       per_atom_python, output_dir, label, numdiff):
    positions = structure["positions_angstrom"]
    cell = structure.get("cell_angstrom")
    pbc = structure.get("pbc", (False, False, False))
    n_atoms = structure["n_atoms"]

    deck_path = output_dir / f"in.{label}"
    dump_path = output_dir / f"{label}.snapshot.dump"
    log_path = output_dir / f"log.{label}"
    screen_path = output_dir / f"screen.{label}"
    deck = render_single_point_deck(
        model_path=model_path, element=element, cutoff=cutoff, positions=positions,
        cell=cell, pbc=pbc, snapshot_path=dump_path, numdiff=numdiff,
        marker_prefix="YE3T_TAGGED",
    )
    deck_path.write_text(deck, encoding="utf-8")
    run_lmp(lmp, deck_path, log_path, screen_path)

    frames = read_dump(dump_path)
    if len(frames) != 1:
        raise AssertionError(f"{label}: expected one dump frame")
    _, records = frames[0]
    if sorted(records) != list(range(1, n_atoms + 1)):
        raise AssertionError(f"{label}: unexpected atom IDs in dump")

    errors = []
    energy_sum = 0.0
    max_energy_error = 0.0
    max_force_error = 0.0
    max_per_atom_energy_error = 0.0
    for atom_id in range(1, n_atoms + 1):
        record = records[atom_id]
        expected_force = reference["forces_ev_per_angstrom"][atom_id - 1]
        for component, field in enumerate(("fx", "fy", "fz")):
            error = require_close(
                record[field], expected_force[component],
                f"{label} atom {atom_id} {field}", FORCE_TOL, errors,
            )
            max_force_error = max(max_force_error, error)
        energy_sum += record["c_atom_energy"]
        if per_atom_python is not None:
            error = require_close(
                record["c_atom_energy"], float(per_atom_python[atom_id - 1]),
                f"{label} atom {atom_id} per-atom energy vs Python exact_moment_reduction",
                PER_ATOM_ENERGY_TOL, errors,
            )
            max_per_atom_energy_error = max(max_per_atom_energy_error, error)

    total_energy_lmp = marker(log_path, "YE3T_TAGGED_PE")
    require_close(total_energy_lmp, reference["energy_ev"], f"{label} total energy (log)", ENERGY_TOL, errors)
    require_close(energy_sum, reference["energy_ev"], f"{label} total energy (sum of per-atom)", ENERGY_TOL, errors)
    max_energy_error = abs(total_energy_lmp - reference["energy_ev"])

    result = {
        "n_atoms": n_atoms,
        "total_energy_error_eV": max_energy_error,
        "force_max_abs_error_eV_per_A": max_force_error,
        "per_atom_energy_max_abs_error_eV": max_per_atom_energy_error,
        "lammps_virial_eV": {
            name: marker(log_path, f"YE3T_TAGGED_VIRIAL_{name}")
            for name in ("XX", "YY", "ZZ", "XY", "XZ", "YZ")
        },
    }
    if numdiff:
        force_fd = marker(log_path, "YE3T_TAGGED_NUMDIFF_FORCE_MAX_ABS")
        virial_fd = marker(log_path, "YE3T_TAGGED_NUMDIFF_VIRIAL_L2_EV")
        if force_fd > NUMDIFF_FORCE_TOL:
            errors.append(f"{label} numdiff force error {force_fd:.3e} > {NUMDIFF_FORCE_TOL:.3e}")
        if virial_fd > NUMDIFF_VIRIAL_TOL_EV:
            errors.append(f"{label} numdiff virial error {virial_fd:.3e} eV > {NUMDIFF_VIRIAL_TOL_EV:.3e}")
        result["numdiff_force_max_abs_eV_per_A"] = force_fd
        result["numdiff_virial_l2_eV"] = virial_fd

    if errors:
        raise AssertionError(f"{label}: " + "; ".join(errors[:20]) + (" ..." if len(errors) > 20 else ""))
    return result


# --------------------------------------------------------------------------
# Python-side references independent of the reference JSON:
# per-atom energies and a finite-strain virial, both via the same
# exact_moment_reduction (complex T2) evaluator energy_and_forces uses.
# --------------------------------------------------------------------------

def _make_atoms(structure):
    from ase import Atoms

    positions = np.asarray(structure["positions_angstrom"], dtype=np.float64)
    pbc = tuple(bool(v) for v in structure.get("pbc", (False, False, False)))
    cell = structure.get("cell_angstrom")
    # Prefer the reference's own per-atom `symbols` (present on the
    # arm exports and, now, the slice references too) over the Ta-only
    # hardcode, which stays only as a fallback for any older reference
    # that predates that field.
    symbols = structure.get("symbols") or ["Ta"] * structure["n_atoms"]
    if any(pbc):
        return Atoms(symbols=symbols, positions=positions, cell=np.asarray(cell, dtype=np.float64), pbc=pbc)
    return Atoms(symbols=symbols, positions=positions, pbc=False)


def per_atom_energies_python(model, atoms):
    """`model.compiled`/`model.moment_evaluator` are `None` for a
    multi-content (arm) model (`arm_lammps_model` builds it with
    `compiled=None`) -- use `model.channels`/`model.evaluator_for(
    "real_moment_reduction")` instead, mirroring `cross_check_python`'s
    same `model.multi_content` branch. Per-atom beta selection mirrors
    `ye3t_methods.atomistic.tagged_cauchy_linear._energy_from_per_atom_features` (kept
    per-atom here, not summed, for this function's own per-atom-energy
    contract) so this is correct for both plain and per-species beta.
    """
    import torch

    from ye3t_methods.atomistic.tagged_cauchy_linear import ordinary_edge_primitives

    if model.multi_content:
        evaluator = model.evaluator_for("real_moment_reduction")
        channels = model.channels
    else:
        from ye3t_methods.atomistic.lifted_cauchy_linear import _artifact_channels

        evaluator = model.moment_evaluator
        channels = _artifact_channels(model.compiled)
    positions = torch.tensor(np.asarray(atoms.get_positions(), dtype=np.float64), dtype=torch.float64)
    symbols = atoms.get_chemical_symbols()
    atom_types = torch.tensor([model.species_index[s] for s in symbols], dtype=torch.long)
    periodic = bool(np.any(np.asarray(atoms.pbc, dtype=bool)))
    cell = torch.tensor(np.asarray(atoms.cell.array, dtype=np.float64), dtype=torch.float64) if periodic else None
    pbc = tuple(bool(v) for v in atoms.pbc) if periodic else None
    primitives = ordinary_edge_primitives(positions, atom_types, cell, pbc, model.cutoff, model.radial_config, channels)
    per_atom = evaluator.descriptors(positions, atom_types, cell, pbc, primitives).detach()
    offsets = torch.tensor([model.offsets[s] for s in symbols], dtype=torch.float64)
    if model.per_species_beta:
        beta_matrix = model.beta.to(dtype=per_atom.dtype)
        beta_per_atom = beta_matrix.index_select(0, atom_types)
        atomic_energy = (per_atom * beta_per_atom).sum(dim=-1) + offsets
    else:
        beta = model.beta.to(per_atom.dtype)
        atomic_energy = (per_atom @ beta) + offsets
    return atomic_energy.numpy()


def finite_strain_virial_ev(model, atoms, delta=1.0e-5):
    """Central-difference dE/d(symmetric strain), 6 Voigt components (xx yy zz xy xz yz),
    returned as virial = -dE/dstrain (eV), matching the LAMMPS/lifted-Cauchy
    "lammps_virial = minus_strain_derivative" convention. Requires a periodic cell."""

    from ye3t_methods.atomistic.tagged_cauchy_linear import energy_and_forces

    base_positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    base_cell = np.asarray(atoms.cell.array, dtype=np.float64)
    voigt = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
    virial = []
    for (a, b) in voigt:
        energies = []
        for sign in (+1.0, -1.0):
            strain = np.zeros((3, 3))
            if a == b:
                strain[a, a] = sign * delta
            else:
                strain[a, b] = sign * delta * 0.5
                strain[b, a] = sign * delta * 0.5
            deformation = np.eye(3) + strain
            atoms_perturbed = atoms.copy()
            atoms_perturbed.set_cell(base_cell @ deformation.T, scale_atoms=False)
            atoms_perturbed.set_positions(base_positions @ deformation.T)
            energy, _forces = energy_and_forces(model, atoms_perturbed, execution_strategy="exact_moment_reduction")
            energies.append(float(energy))
        derivative = (energies[0] - energies[1]) / (2.0 * delta)
        virial.append(-derivative)
    return dict(zip(("XX", "YY", "ZZ", "XY", "XZ", "YZ"), virial))


def load_python_model(model_path):
    from ye3t_methods.atomistic.tagged_cauchy_linear import load_tagged_model

    return load_tagged_model(str(model_path))


def cross_check_python(model_path, reference_structures):
    """Independent (of both LAMMPS and the reference JSON) sanity check: does
    ye3t_methods.atomistic.tagged_cauchy_linear.energy_and_forces itself reproduce the
    reference JSON's numbers? Guards against a stale/mismatched reference.

    Multi-content ("arm") models (`model.multi_content`) only expose
    `real_moment_reduction` as a live execution strategy
    (`exact_moment_reduction` is single-content-only); single-content models
    keep the original `exact_moment_reduction`.
    """

    from ye3t_methods.atomistic.tagged_cauchy_linear import energy_and_forces

    model = load_python_model(model_path)
    execution_strategy = "real_moment_reduction" if model.multi_content else "exact_moment_reduction"
    report = {}
    for label, structure in reference_structures.items():
        atoms = _make_atoms(structure)
        energy, forces = energy_and_forces(model, atoms, execution_strategy=execution_strategy)
        energy_error = abs(float(energy) - structure["energy_ev"])
        force_error = float(np.max(np.abs(forces.numpy() - np.asarray(structure["forces_ev_per_angstrom"]))))
        report[label] = {"energy_error_eV": energy_error, "force_max_abs_error_eV_per_A": force_error}
        if energy_error > 1.0e-9 or force_error > 1.0e-9:
            raise AssertionError(
                f"{label}: energy_and_forces does not reproduce the reference JSON "
                f"(energy_error={energy_error:.3e}, force_error={force_error:.3e})"
            )
    return model, report


def render_bcc_mpi_deck(*, model_path, element, ranks, mover_seed_point, dump_initial, dump_final):
    lines = [
        "units metal",
        "atom_style atomic",
        "boundary p p p",
        "atom_modify map yes sort 0 0.0",
        "newton on",
        f"processors {ranks} 1 1",
        "",
        "variable a equal 3.300",
        "lattice bcc ${a}",
        "region cell block 0 4 0 4 0 4 units lattice",
        "create_box 1 cell",
        "create_atoms 1 box",
        f"mass 1 {MASS_TA}",
        "reset_atoms id sort yes",
        "",
        f"region mover_region sphere {fnum(mover_seed_point[0])} {fnum(mover_seed_point[1])} "
        f"{fnum(mover_seed_point[2])} 0.10 units box",
        "group mover region mover_region",
        "displace_atoms mover move -0.10 0.0 0.0 units box",
        "",
        "neighbor 0.3 bin",
        "neigh_modify every 1 delay 0 check yes",
        "",
        "pair_style ye3t model_family tagged_cauchy chunksize 16",
        f"pair_coeff * * {model_path} {element}",
        "",
        "compute atom_energy all pe/atom",
        "compute atom_stress all stress/atom NULL pair",
        "compute pair_pressure all pressure NULL pair",
        "compute force_sum all reduce sum fx fy fz",
        "",
        "thermo 1",
        "thermo_style custom step atoms vol pe c_pair_pressure[1] c_pair_pressure[2] "
        "c_pair_pressure[3] c_force_sum[1] c_force_sum[2] c_force_sum[3]",
        "thermo_modify format float %.17g",
        "",
        "run 0 post yes",
        f"write_dump all custom {dump_initial} id type proc x y z c_atom_energy "
        "c_atom_stress[1] c_atom_stress[2] c_atom_stress[3] c_atom_stress[4] "
        "c_atom_stress[5] c_atom_stress[6] fx fy fz modify sort id format float %.17g",
        "",
        "timestep 0.001",
        "fix migrate mover move linear 200.0 0.0 0.0 units box",
        "run 1 post yes",
        f"write_dump all custom {dump_final} id type proc x y z c_atom_energy "
        "c_atom_stress[1] c_atom_stress[2] c_atom_stress[3] c_atom_stress[4] "
        "c_atom_stress[5] c_atom_stress[6] fx fy fz modify sort id format float %.17g",
    ]
    return "\n".join(lines) + "\n"


def validate_mpi_migration(*, lmp, mpiexec, model_path, element, output_dir):
    results = {}
    frames_by_rank = {}
    # BCC lattice constant 3.300 A, region 0..4 lattice units per axis (see
    # render_bcc_mpi_deck) puts the box edge at 4*3.300 = 13.2 A, so x=6.6 is
    # simultaneously the rank-2 midpoint boundary and one of the rank-4
    # quarter boundaries. (6.6, 3.3, 3.3) is a BCC corner lattice site there.
    mover_point = (6.6, 3.3, 3.3)
    for ranks in (1, 2, 4):
        label = f"mpi_migration.rank{ranks}"
        deck_path = output_dir / f"in.{label}"
        initial = output_dir / f"{label}.initial.dump"
        final = output_dir / f"{label}.final.dump"
        log_path = output_dir / f"log.{label}"
        screen_path = output_dir / f"screen.{label}"
        deck = render_bcc_mpi_deck(
            model_path=model_path, element=element, ranks=ranks,
            mover_seed_point=mover_point, dump_initial=initial, dump_final=final,
        )
        deck_path.write_text(deck, encoding="utf-8")
        run_lmp(lmp, deck_path, log_path, screen_path, mpiexec=mpiexec, ranks=ranks)

        initial_frames = read_dump(initial)
        final_frames = read_dump(final)
        _, initial_records = initial_frames[0]
        _, final_records = final_frames[0]
        owners = {int(record["proc"]) for record in initial_records.values()}
        if len(owners) != ranks:
            raise AssertionError(f"{label}: {len(owners)} ranks own atoms, expected {ranks}")
        changed = [aid for aid in initial_records if initial_records[aid]["proc"] != final_records[aid]["proc"]]
        if ranks > 1 and not changed:
            raise AssertionError(f"{label}: no atom crossed an MPI domain boundary")
        net_force = [sum(record[f] for record in final_records.values()) for f in ("fx", "fy", "fz")]
        net_force_norm = math.sqrt(sum(v * v for v in net_force))
        if net_force_norm > 2.0e-7:
            raise AssertionError(f"{label}: net force {net_force_norm:.3e} exceeds 2e-7 (Newton's 3rd law)")
        frames_by_rank[ranks] = (initial_records, final_records)
        results[label] = {"migrating_atom_count": len(changed), "net_force_norm_eV_per_A": net_force_norm}

    anchor_initial, anchor_final = frames_by_rank[1]
    parity = {}
    for ranks in (2, 4):
        initial, final = frames_by_rank[ranks]
        for tag, records, anchor in (("initial", initial, anchor_initial), ("final", final, anchor_final)):
            max_abs = 0.0
            for atom_id in anchor:
                for field in ("fx", "fy", "fz", "c_atom_energy"):
                    max_abs = max(max_abs, abs(records[atom_id][field] - anchor[atom_id][field]))
            key = f"rank1_vs_rank{ranks}_{tag}_max_abs"
            parity[key] = max_abs
            if max_abs > 1.0e-8:
                raise AssertionError(f"{key}: {max_abs:.3e} exceeds 1e-8")
    results["rank_parity_max_abs"] = parity
    return results


def render_nve_deck(*, model_path, element, dump_path, steps, timestep, seed):
    lines = [
        "units metal",
        "atom_style atomic",
        "boundary p p p",
        "atom_modify map yes sort 0 0.0",
        "newton on",
        "",
        "variable a equal 3.300",
        "lattice bcc ${a}",
        "region cell block 0 4 0 4 0 4 units lattice",
        "create_box 1 cell",
        "create_atoms 1 box",
        f"mass 1 {MASS_TA}",
        "reset_atoms id sort yes",
        f"velocity all create 300.0 {seed} mom yes rot yes dist gaussian loop all",
        "",
        "neighbor 0.3 bin",
        "neigh_modify every 1 delay 0 check yes",
        "",
        "pair_style ye3t model_family tagged_cauchy chunksize 32",
        f"pair_coeff * * {model_path} {element}",
        "",
        f"timestep {fnum(timestep)}",
        "fix integrate all nve",
        "",
        "thermo 1",
        "thermo_style custom step temp pe ke etotal press",
        "thermo_modify format float %.17g",
        f"dump trajectory all custom 50 {dump_path} id type x y z vx vy vz fx fy fz",
        "dump_modify trajectory sort id format float %.17g",
        "",
        "run 0 post no",
        "variable initial_etotal equal $(etotal:%.17g)",
        f"run {steps} post no",
        'print "YE3T_TAGGED_NVE_INITIAL_ETOTAL=$(v_initial_etotal:%.17g)"',
        'print "YE3T_TAGGED_NVE_FINAL_ETOTAL=$(etotal:%.17g)"',
    ]
    return "\n".join(lines) + "\n"


def validate_nve(*, lmp, model_path, element, output_dir, steps=500, timestep=0.001, seed=934817):
    label = "nve"
    deck_path = output_dir / f"in.{label}"
    dump_path = output_dir / f"{label}.dump"
    log_path = output_dir / f"log.{label}"
    screen_path = output_dir / f"screen.{label}"
    deck = render_nve_deck(model_path=model_path, element=element, dump_path=dump_path,
                          steps=steps, timestep=timestep, seed=seed)
    deck_path.write_text(deck, encoding="utf-8")
    run_lmp(lmp, deck_path, log_path, screen_path, timeout=600)

    initial = marker(log_path, "YE3T_TAGGED_NVE_INITIAL_ETOTAL")
    final = marker(log_path, "YE3T_TAGGED_NVE_FINAL_ETOTAL")
    text = log_path.read_text(encoding="utf-8")
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
        "timestep_ps": timestep,
        "n_atoms": n_atoms,
        "initial_etotal_eV": initial,
        "final_etotal_eV": final,
        "max_abs_drift_eV": drift_abs,
        "max_abs_drift_eV_per_atom": drift_abs / n_atoms,
        "thermo_sample_count": len(etotal_values),
    }


def render_timing_deck(*, pair_style_line, pair_coeff_line, steps):
    lines = [
        "units metal",
        "atom_style atomic",
        "boundary p p p",
        "atom_modify map yes sort 0 0.0",
        "newton on",
        "",
        "variable a equal 3.300",
        "lattice bcc ${a}",
        "region cell block 0 4 0 4 0 4 units lattice",
        "create_box 1 cell",
        "create_atoms 1 box",
        f"mass 1 {MASS_TA}",
        "reset_atoms id sort yes",
        "velocity all create 300.0 173285 mom yes rot yes dist gaussian loop all",
        "",
        "neighbor 0.3 bin",
        "neigh_modify every 1 delay 0 check yes",
        "",
        pair_style_line,
        pair_coeff_line,
        "",
        "timestep 0.001",
        "fix integrate all nve",
        "thermo 50",
        f"run {steps} post no",
    ]
    return "\n".join(lines) + "\n"


def measure_timing(*, lmp, label, pair_style_line, pair_coeff_line, output_dir, steps=200):
    deck_path = output_dir / f"in.timing.{label}"
    log_path = output_dir / f"log.timing.{label}"
    screen_path = output_dir / f"screen.timing.{label}"
    deck = render_timing_deck(pair_style_line=pair_style_line, pair_coeff_line=pair_coeff_line, steps=steps)
    deck_path.write_text(deck, encoding="utf-8")
    start = time.perf_counter()
    run_lmp(lmp, deck_path, log_path, screen_path, timeout=300)
    wall_seconds = time.perf_counter() - start

    text = log_path.read_text(encoding="utf-8")
    loop_seconds = None
    for line in text.splitlines():
        if line.startswith("Loop time of"):
            loop_seconds = float(line.split()[3])
            break
    n_atoms = 128
    seconds = loop_seconds if loop_seconds is not None else wall_seconds
    us_per_atom_step = seconds * 1.0e6 / (n_atoms * steps)
    return {
        "label": label,
        "steps": steps,
        "n_atoms": n_atoms,
        "loop_seconds": loop_seconds,
        "wall_seconds": wall_seconds,
        "microseconds_per_atom_step": us_per_atom_step,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--lmp", type=Path, required=True)
    parser.add_argument("--model-k1", type=Path, required=True)
    parser.add_argument("--reference-k1", type=Path, required=True)
    parser.add_argument("--model-k2", type=Path)
    parser.add_argument("--reference-k2", type=Path)
    parser.add_argument("--yace-model", type=Path)
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--skip-mpi", action="store_true")
    parser.add_argument("--skip-nve", action="store_true")
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    report = {"schema": "ye3t_fitted_tagged_cauchy_cpu_validation_v1", "artifacts": {}}

    models = [("k1", args.model_k1, args.reference_k1)]
    if args.model_k2 and args.reference_k2:
        models.append(("k2", args.model_k2, args.reference_k2))

    for tag, model_path, reference_path in models:
        artifact_report = {"model_path": str(model_path)}
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        structures = reference["structures"]
        artifact_report["self_hash_matches_reference"] = (
            json.loads(model_path.read_text(encoding="utf-8"))["self_hash"]
            == reference["source_model_self_hash"]
        )

        model, cross_check = cross_check_python(model_path, structures)
        artifact_report["python_cross_check_vs_reference"] = cross_check
        element = model.species_order[0]
        cutoff = model.cutoff

        structure_results = {}
        for label, structure in structures.items():
            atoms = _make_atoms(structure)
            per_atom_python = per_atom_energies_python(model, atoms)
            per_atom_sum_error = abs(float(per_atom_python.sum()) - structure["energy_ev"])
            if per_atom_sum_error > 1.0e-9:
                raise AssertionError(
                    f"{tag} {label}: python per-atom energies do not sum to the reference total "
                    f"(|error|={per_atom_sum_error:.3e})"
                )
            is_frame0 = label == "ta_frame0"
            result = validate_structure(
                lmp=str(args.lmp), model_path=str(model_path), element=element, cutoff=cutoff,
                structure=structure, reference=structure, per_atom_python=per_atom_python,
                output_dir=args.output, label=f"{tag}.{label}", numdiff=is_frame0,
            )
            if all(structure.get("pbc", (False, False, False))):
                fd_virial = finite_strain_virial_ev(model, atoms)
                max_virial_fd_error = 0.0
                for name in ("XX", "YY", "ZZ", "XY", "XZ", "YZ"):
                    error = abs(result["lammps_virial_eV"][name] - fd_virial[name])
                    max_virial_fd_error = max(max_virial_fd_error, error)
                result["python_finite_strain_virial_eV"] = fd_virial
                result["virial_vs_python_finite_strain_max_abs_error_eV"] = max_virial_fd_error
                if max_virial_fd_error > VIRIAL_FD_TOL_EV:
                    raise AssertionError(
                        f"{tag} {label}: virial vs Python finite-strain derivative error "
                        f"{max_virial_fd_error:.3e} eV exceeds {VIRIAL_FD_TOL_EV:.3e}"
                    )
            structure_results[label] = result
        artifact_report["structures"] = structure_results

        if not args.skip_mpi:
            artifact_report["mpi_migration"] = validate_mpi_migration(
                lmp=str(args.lmp), mpiexec=args.mpiexec, model_path=str(model_path),
                element=element, output_dir=args.output,
            )

        if not args.skip_nve and tag == "k2":
            artifact_report["nve_500step_300K_128atom"] = validate_nve(
                lmp=str(args.lmp), model_path=str(model_path), element=element, output_dir=args.output,
            )

        if not args.skip_timing:
            artifact_report["timing"] = measure_timing(
                lmp=str(args.lmp), label=f"tagged_cauchy_{tag}",
                pair_style_line="pair_style ye3t model_family tagged_cauchy chunksize 32",
                pair_coeff_line=f"pair_coeff * * {model_path} {element}",
                output_dir=args.output,
            )

        report["artifacts"][tag] = artifact_report

    if not args.skip_timing and args.yace_model:
        report["timing_yace_reference"] = measure_timing(
            lmp=str(args.lmp), label="yace",
            pair_style_line="pair_style ye3t model_family yace chunksize 32",
            pair_coeff_line=f"pair_coeff * * {args.yace_model} Ta",
            output_dir=args.output,
        )

    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
