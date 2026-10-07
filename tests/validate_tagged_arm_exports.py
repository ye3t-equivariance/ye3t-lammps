#!/usr/bin/env python3
"""Validate multi-content "arm" tagged-Cauchy exports (K2SO4 k=0,
Ta arms A0/A1/A2) against their position-bearing reference JSONs,
through both the CPU and the Kokkos device binaries.

Structures schema: top-level
`structures` mapping, each entry `{positions_angstrom, symbols, n_atoms,
energy_ev, forces_ev_per_angstrom, pbc, [cell_angstrom], dataset_index}` --
the same shape validate_fitted_tagged_cauchy.py already consumes for the
single-content k1/k2 slices, except `symbols` is now per-atom (multi-
species) instead of implicitly one element for every atom, so the deck
renderer here supports an arbitrary number of species instead of
`render_single_point_deck`'s single-`element` restriction.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from test_lifted_cauchy_lammps import marker, read_dump  # noqa: E402
from validate_fitted_tagged_cauchy import _make_atoms, fnum, load_python_model, run_lmp  # noqa: E402

ENERGY_TOL = 1.0e-8
FORCE_TOL = 1.0e-8
NUMDIFF_FORCE_TOL = 5.0e-6
NUMDIFF_VIRIAL_TOL_EV = 5.0e-5
CPU_KK_PARITY_TOL = 1.0e-10

# --fitted-model relative coefficients. The fitted 923-feature models carry
# pooled beta up to 1e12 (numerically singular data design at small ridge);
# the absolute tolerances above were sized for the slice fixtures and are far
# too tight for that scale, even though CPU/Kokkos/Python still agree to
# ~3e-10 relative. Bound = REL * reference_scale + ABS_FLOOR; reference_scale
# is |E_ref| for energy, max|F_ref| (one scalar per structure, over every
# atom and component) for force, and max_ab|W_ab| (over the Python
# strain-derivative matrix) for virial.
FITTED_ABS_FLOOR = 1.0e-12
FITTED_ENERGY_REL = 1.0e-9
FITTED_FORCE_REL = 1.0e-8
FITTED_PARITY_ENERGY_REL = 1.0e-9
FITTED_PARITY_FORCE_REL = 1.0e-8
FITTED_VIRIAL_REL = 1.0e-8

ELEMENT_MASS = {
    "H": 1.008, "K": 39.0983, "O": 15.999, "S": 32.06, "Ta": 180.94788,
}


def kokkos_extra_args():
    return ["-k", "on", "g", "1", "-pk", "kokkos", "neigh", "half", "-sf", "kk"]


def render_multi_species_deck(*, model_path, species_order, positions, symbols, cell, pbc,
                              cutoff, snapshot_path, numdiff=False, marker_prefix="YE3T_ARM"):
    positions = np.asarray(positions, dtype=np.float64)
    types = [species_order.index(symbol) + 1 for symbol in symbols]
    lines = ["units metal", "atom_style atomic"]
    periodic = all(pbc)
    lines.append("boundary p p p" if periodic else "boundary f f f")
    lines += ["atom_modify map yes sort 1 0.0", "newton on", ""]
    if periodic:
        cell_array = np.asarray(cell, dtype=np.float64)
        off_diagonal = [cell_array[0, 1], cell_array[0, 2], cell_array[1, 0],
                        cell_array[1, 2], cell_array[2, 0], cell_array[2, 1]]
        if max(abs(v) for v in off_diagonal) > 1.0e-8:
            raise ValueError("render_multi_species_deck only supports an orthogonal cell")
        lx, ly, lz = cell_array[0, 0], cell_array[1, 1], cell_array[2, 2]
        lines.append(f"region cell block 0 {fnum(lx)} 0 {fnum(ly)} 0 {fnum(lz)} units box")
    else:
        lo = positions.min(axis=0) - (cutoff + 2.0)
        hi = positions.max(axis=0) + (cutoff + 2.0)
        lines.append(
            "region cell block "
            f"{fnum(lo[0])} {fnum(hi[0])} {fnum(lo[1])} {fnum(hi[1])} {fnum(lo[2])} {fnum(hi[2])} units box"
        )
    lines.append(f"create_box {len(species_order)} cell")
    for (x, y, z), atom_type in zip(positions, types):
        lines.append(f"create_atoms {atom_type} single {fnum(x)} {fnum(y)} {fnum(z)} units box")
    for species_index, species_name in enumerate(species_order, start=1):
        lines.append(f"mass {species_index} {ELEMENT_MASS.get(species_name, 1.0)}")
    lines += ["", "neighbor 0.3 bin", "neigh_modify every 1 delay 0 check yes", ""]
    lines += [
        "pair_style ye3t model_family tagged_cauchy chunksize 32",
        "pair_coeff * * " + str(model_path) + " " + " ".join(species_order),
        "",
        "compute atom_energy all pe/atom",
        "compute atom_stress all stress/atom NULL pair",
        "compute pair_pressure all pressure NULL pair",
        "compute force_sum all reduce sum fx fy fz",
        "",
    ]
    if numdiff:
        lines += [
            "fix numerical_force all numdiff 1 1.0e-5",
            "fix numerical_virial all numdiff/virial 1 1.0e-5",
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
        "variable virial_c4 equal c_pair_pressure[4]*vol/1.6021765e6",
        "variable virial_c5 equal c_pair_pressure[5]*vol/1.6021765e6",
        "variable virial_c6 equal c_pair_pressure[6]*vol/1.6021765e6",
        "",
        "thermo 1",
        "thermo_style custom step pe c_force_sum[1] c_force_sum[2] c_force_sum[3]",
        "thermo_modify format float %.17g",
        "",
        f"dump snapshot all custom 1 {snapshot_path} id type x y z c_atom_energy fx fy fz",
        "dump_modify snapshot sort id format float %.17g",
        "run 0 post no",
        "",
        f'print "{marker_prefix}_PE=$(pe:%.17g)"',
        f'print "{marker_prefix}_VIRIAL_XX=$(v_virial_xx:%.17g)"',
        f'print "{marker_prefix}_VIRIAL_YY=$(v_virial_yy:%.17g)"',
        f'print "{marker_prefix}_VIRIAL_ZZ=$(v_virial_zz:%.17g)"',
        f'print "{marker_prefix}_VIRIAL_C4=$(v_virial_c4:%.17g)"',
        f'print "{marker_prefix}_VIRIAL_C5=$(v_virial_c5:%.17g)"',
        f'print "{marker_prefix}_VIRIAL_C6=$(v_virial_c6:%.17g)"',
    ]
    if numdiff:
        lines += [
            f'print "{marker_prefix}_NUMDIFF_FORCE_MAX_ABS=$(c_force_error:%.17g)"',
            f'print "{marker_prefix}_NUMDIFF_VIRIAL_L2_EV=$(v_virial_error_eV:%.17g)"',
        ]
    return "\n".join(lines) + "\n"


def run_one(*, lmp, extra_args, label, model_path, species_order, structure, output_dir,
           numdiff, cutoff):
    deck_path = output_dir / f"in.{label}"
    dump_path = output_dir / f"{label}.dump"
    log_path = output_dir / f"log.{label}"
    screen_path = output_dir / f"screen.{label}"
    deck_path.write_text(
        render_multi_species_deck(
            model_path=model_path, species_order=species_order,
            positions=structure["positions_angstrom"], symbols=structure["symbols"],
            cell=structure.get("cell_angstrom"), pbc=structure["pbc"], cutoff=cutoff,
            snapshot_path=dump_path, numdiff=numdiff,
        ),
        encoding="utf-8",
    )
    run_lmp(str(lmp), deck_path, log_path, screen_path, timeout=180, extra_args=extra_args)
    frames = read_dump(dump_path)
    _, records = frames[0]
    total_energy = marker(log_path, "YE3T_ARM_PE")
    result = {
        "total_energy_eV": total_energy, "records": records,
        # LAMMPS-native [xx, yy, zz] plus the deck's own numdiff/virial
        # component order for the shear terms (c4/c5/c6, i.e. whatever
        # fix numdiff/virial's own components 4/5/6 are, NOT necessarily
        # LAMMPS's [xy, xz, yz] compute-pressure labels -- see
        # compare_virial's docstring for the exact correspondence).
        "virial_xx_eV": marker(log_path, "YE3T_ARM_VIRIAL_XX"),
        "virial_yy_eV": marker(log_path, "YE3T_ARM_VIRIAL_YY"),
        "virial_zz_eV": marker(log_path, "YE3T_ARM_VIRIAL_ZZ"),
        "virial_c4_eV": marker(log_path, "YE3T_ARM_VIRIAL_C4"),
        "virial_c5_eV": marker(log_path, "YE3T_ARM_VIRIAL_C5"),
        "virial_c6_eV": marker(log_path, "YE3T_ARM_VIRIAL_C6"),
    }
    if numdiff:
        result["numdiff_force_max_abs"] = marker(log_path, "YE3T_ARM_NUMDIFF_FORCE_MAX_ABS")
        result["numdiff_virial_l2_eV"] = marker(log_path, "YE3T_ARM_NUMDIFF_VIRIAL_L2_EV")
    return result


VIRIAL_TOL_EV = 1.0e-6


def python_strain_virial(model, atoms, execution_strategy):
    """Exact strain-derivative virial: `eps` a 3x3 torch tensor of zeros
    with `requires_grad`; map positions `r -> r @ (I + eps)^T` and, if
    periodic, `cell -> cell @ (I + eps)^T`; evaluate the model energy E
    with the evaluator for `execution_strategy` on the mapped structure;
    `W_ab = -dE/deps_ab` via `torch.autograd.grad` (exact, no step size).

    Returns a 3x3 numpy array (eV), `W[i,j] = -dE/deps[i,j]`, i,j in
    {0,1,2} = {x,y,z}, not forcibly symmetrized (a rotation-invariant
    model's dE/deps comes out symmetric on its own to autograd precision --
    this is itself an implicit self-consistency check, not asserted here).
    `model.channels`/`model.offsets`/`model.species_index` are populated
    identically for single- and multi-content models (confirmed by reading
    `ye3t_methods.atomistic.tagged_cauchy_linear`'s `TaggedCauchyModel.__init__` and
    `energy_and_forces`, which already uses `model.channels` directly, not
    `_artifact_channels(model.compiled)`), so this function needs no
    `model.multi_content` branch of its own.
    """
    import torch

    from ye3t_methods.atomistic.tagged_cauchy_linear import _energy_from_per_atom_features, ordinary_edge_primitives

    evaluator = model.evaluator_for(execution_strategy)
    base_positions = torch.tensor(np.asarray(atoms.get_positions(), dtype=np.float64), dtype=torch.float64)
    symbols = atoms.get_chemical_symbols()
    atom_types = torch.tensor([model.species_index[s] for s in symbols], dtype=torch.long)
    periodic = bool(np.any(np.asarray(atoms.pbc, dtype=bool)))
    base_cell = (
        torch.tensor(np.asarray(atoms.cell.array, dtype=np.float64), dtype=torch.float64)
        if periodic else None
    )
    pbc = tuple(bool(v) for v in atoms.pbc) if periodic else None

    eps = torch.zeros((3, 3), dtype=torch.float64, requires_grad=True)
    deformation = torch.eye(3, dtype=torch.float64) + eps
    positions = base_positions @ deformation.T
    cell = (base_cell @ deformation.T) if periodic else None

    primitives = ordinary_edge_primitives(
        positions, atom_types, cell, pbc, model.cutoff, model.radial_config, model.channels
    )
    per_atom = evaluator.descriptors(positions, atom_types, cell, pbc, primitives)
    offsets = torch.tensor([model.offsets[s] for s in symbols], dtype=torch.float64)
    energy = _energy_from_per_atom_features(model, per_atom, atom_types) + offsets.sum()
    (grad_eps,) = torch.autograd.grad(energy, eps)
    return (-grad_eps).detach().numpy()


def compare_virial(*, w_python, backend_result, label, fitted_model):
    """W_lammps read off `c_pair_pressure[1..6]*vol/nktv2p` directly
    (`virial_xx`/`yy`/`zz`/`c4`/`c5`/`c6` in the deck, `c_pair_pressure`'s
    own standard LAMMPS ordering: 1=xx,2=yy,3=zz,4=xy,5=xz,6=yz -- no
    numdiff-specific component swap here; that swap (see the deck's
    `dv4`/`dv5`/`dv6` variables) is specific to `fix numdiff/virial`'s own
    internal component ordering versus `compute pressure`'s, established
    empirically while debugging this exact function: printing both sides
    for A1_frame0 showed `c_pair_pressure[4]` matches `w_python[0,1]`
    (xy) directly, `[5]` matches `w_python[0,2]` (xz), `[6]` matches
    `w_python[1,2]` (yz) -- the direct, unswapped correspondence, not the
    numdiff one. `w_python`'s off-diagonal entries: `w_python[0,1]`=xy,
    `w_python[0,2]`=xz, `w_python[1,2]`=yz.

    Always computes and reports both the absolute (`VIRIAL_TOL_EV`)
    and relative (`FITTED_VIRIAL_REL * max_ab|W_python_ab| + FITTED_ABS_
    FLOOR`) bounds; gates on whichever `fitted_model` selects.
    """
    pairs = {
        "xx": (w_python[0, 0], backend_result["virial_xx_eV"]),
        "yy": (w_python[1, 1], backend_result["virial_yy_eV"]),
        "zz": (w_python[2, 2], backend_result["virial_zz_eV"]),
        "xy": (w_python[0, 1], backend_result["virial_c4_eV"]),
        "xz": (w_python[0, 2], backend_result["virial_c5_eV"]),
        "yz": (w_python[1, 2], backend_result["virial_c6_eV"]),
    }
    reference_scale = float(np.max(np.abs(w_python)))
    result = {}
    max_abs = 0.0
    for component, (py, lmp) in pairs.items():
        error = abs(float(py) - float(lmp))
        result[component] = {"python_eV": float(py), "lammps_eV": float(lmp), "abs_error_eV": error}
        max_abs = max(max_abs, error)
    bound_abs = VIRIAL_TOL_EV
    bound_rel = FITTED_VIRIAL_REL * reference_scale + FITTED_ABS_FLOOR
    result.update({
        "max_abs_error_eV": max_abs,
        "reference_scale_eV": reference_scale,
        "abs_tolerance_eV": bound_abs,
        "relative_tolerance_eV": bound_rel,
    })
    bound = bound_rel if fitted_model else bound_abs
    if max_abs > bound:
        mode = "relative" if fitted_model else "absolute"
        raise AssertionError(f"{label}: strain-derivative virial max abs error {max_abs:.3e} > {bound:.3e} ({mode})")
    return result


def validate_model(*, tag, model_path, reference_path, cpu_lmp, kk_lmp, output_dir, fitted_model=False):
    """`fitted_model=False` (default): every check uses the fixed absolute
    tolerances (`ENERGY_TOL`, `FORCE_TOL`, `CPU_KK_PARITY_TOL`,
    `VIRIAL_TOL_EV`, `NUMDIFF_FORCE_TOL`). `fitted_model=True`
    (`--fitted-model`) switches every one of those to the relative bound
    (`REL * reference_scale + FITTED_ABS_FLOOR`) instead, and numdiff force
    (like numdiff virial, which is informational in both modes) stops
    gating. Both the absolute and relative numbers are always computed and
    recorded in the report, regardless of which one is the active check.
    """
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    model = json.loads(Path(model_path).read_text(encoding="utf-8"))
    species_order = model["species_order"]
    cutoff = float(model["cutoff"])
    report = {
        "model_path": str(model_path),
        "self_hash_matches_reference": model["self_hash"] == reference.get("source_model_self_hash"),
        "fitted_model_relative_mode": fitted_model,
        "structures": {},
    }
    errors = []
    frame0 = True
    for label, structure in reference["structures"].items():
        n_atoms = structure["n_atoms"]
        ref_energy = float(structure["energy_ev"])
        ref_force_scale = float(np.max(np.abs(np.asarray(structure["forces_ev_per_angstrom"], dtype=np.float64))))
        energy_bound_rel = FITTED_ENERGY_REL * abs(ref_energy) + FITTED_ABS_FLOOR
        force_bound_rel = FITTED_FORCE_REL * ref_force_scale + FITTED_ABS_FLOOR
        energy_bound = energy_bound_rel if fitted_model else ENERGY_TOL
        force_bound = force_bound_rel if fitted_model else FORCE_TOL

        cpu = run_one(lmp=cpu_lmp, extra_args=(), label=f"{tag}.{label}.cpu",
                     model_path=model_path, species_order=species_order, structure=structure,
                     output_dir=output_dir, numdiff=frame0, cutoff=cutoff)
        kk = run_one(lmp=kk_lmp, extra_args=kokkos_extra_args(), label=f"{tag}.{label}.kk",
                    model_path=model_path, species_order=species_order, structure=structure,
                    output_dir=output_dir, numdiff=frame0, cutoff=cutoff)

        entry = {
            "n_atoms": n_atoms,
            "reference_energy_eV": ref_energy,
            "reference_force_scale_eV_per_A": ref_force_scale,
        }
        for backend_name, backend in (("cpu", cpu), ("kk", kk)):
            energy_error = abs(backend["total_energy_eV"] - ref_energy)
            if energy_error > energy_bound:
                errors.append(f"{tag} {label} {backend_name} energy error {energy_error:.3e} > {energy_bound:.3e}")
            max_force_error = 0.0
            energy_sum = 0.0
            for atom_id in range(1, n_atoms + 1):
                record = backend["records"][atom_id]
                expected_force = structure["forces_ev_per_angstrom"][atom_id - 1]
                for component, field in enumerate(("fx", "fy", "fz")):
                    error = abs(record[field] - expected_force[component])
                    max_force_error = max(max_force_error, error)
                energy_sum += record["c_atom_energy"]
            if max_force_error > force_bound:
                errors.append(f"{tag} {label} {backend_name} force max abs error {max_force_error:.3e} > {force_bound:.3e}")
            sum_error = abs(energy_sum - backend["total_energy_eV"])
            sum_bound = energy_bound_rel if fitted_model else 1.0e-8
            if sum_error > sum_bound:
                errors.append(f"{tag} {label} {backend_name} sum(per-atom) vs total energy error {sum_error:.3e} > {sum_bound:.3e}")
            entry[f"{backend_name}_total_energy_error_eV"] = energy_error
            entry[f"{backend_name}_total_energy_abs_tolerance_eV"] = ENERGY_TOL
            entry[f"{backend_name}_total_energy_relative_tolerance_eV"] = energy_bound_rel
            entry[f"{backend_name}_force_max_abs_error_eV_per_A"] = max_force_error
            entry[f"{backend_name}_force_abs_tolerance_eV_per_A"] = FORCE_TOL
            entry[f"{backend_name}_force_relative_tolerance_eV_per_A"] = force_bound_rel
            entry[f"{backend_name}_sum_per_atom_vs_total_error_eV"] = sum_error
            if frame0:
                # numdiff virial is informational only (unconditional -- the
                # exact Python strain-derivative check is authoritative in
                # both modes). numdiff force is also informational-only in
                # --fitted-model mode; default mode gates on it.
                if not fitted_model and backend["numdiff_force_max_abs"] > NUMDIFF_FORCE_TOL:
                    errors.append(f"{tag} {label} {backend_name} numdiff force {backend['numdiff_force_max_abs']:.3e} > {NUMDIFF_FORCE_TOL:.3e}")
                entry[f"{backend_name}_numdiff_force_max_abs_eV_per_A"] = backend["numdiff_force_max_abs"]
                entry[f"{backend_name}_numdiff_virial_l2_eV"] = backend["numdiff_virial_l2_eV"]

        if frame0:
            python_model = load_python_model(model_path)
            execution_strategy = "real_moment_reduction" if python_model.multi_content else "exact_moment_reduction"
            atoms = _make_atoms(structure)
            w_python = python_strain_virial(python_model, atoms, execution_strategy)
            virial_entry = {}
            for backend_name, backend in (("cpu", cpu), ("kk", kk)):
                try:
                    virial_entry[backend_name] = compare_virial(
                        w_python=w_python, backend_result=backend, label=f"{tag} {label} {backend_name}",
                        fitted_model=fitted_model,
                    )
                except AssertionError as failure:
                    errors.append(str(failure))
                    virial_entry[backend_name] = {"error": str(failure)}
            entry["strain_virial_eV"] = virial_entry

        # cpu/kk parity: energy (c_atom_energy) and force tracked separately
        # so the relative bound can apply FITTED_PARITY_ENERGY_REL/
        # FITTED_PARITY_FORCE_REL to each appropriately; their combined max
        # (cpu_kk_parity_max_abs) is the single number default mode reports
        # and gates on (max is associative over the flattened
        # {fx,fy,fz,c_atom_energy} set either way).
        cpu_kk_parity_force = 0.0
        cpu_kk_parity_energy = 0.0
        for atom_id in range(1, n_atoms + 1):
            for field in ("fx", "fy", "fz"):
                cpu_kk_parity_force = max(cpu_kk_parity_force, abs(cpu["records"][atom_id][field] - kk["records"][atom_id][field]))
            cpu_kk_parity_energy = max(cpu_kk_parity_energy, abs(cpu["records"][atom_id]["c_atom_energy"] - kk["records"][atom_id]["c_atom_energy"]))
        cpu_kk_parity = max(cpu_kk_parity_force, cpu_kk_parity_energy)
        parity_energy_bound_rel = FITTED_PARITY_ENERGY_REL * abs(ref_energy) + FITTED_ABS_FLOOR
        parity_force_bound_rel = FITTED_PARITY_FORCE_REL * ref_force_scale + FITTED_ABS_FLOOR
        if fitted_model:
            # Energy and force parity are checked (and reported) against
            # their OWN separate bounds, not a combined max vs a combined
            # bound.
            if cpu_kk_parity_energy > parity_energy_bound_rel:
                errors.append(f"{tag} {label} cpu/kk parity (energy) {cpu_kk_parity_energy:.3e} > {parity_energy_bound_rel:.3e}")
            if cpu_kk_parity_force > parity_force_bound_rel:
                errors.append(f"{tag} {label} cpu/kk parity (force) {cpu_kk_parity_force:.3e} > {parity_force_bound_rel:.3e}")
        else:
            if cpu_kk_parity > CPU_KK_PARITY_TOL:
                errors.append(f"{tag} {label} cpu/kk parity {cpu_kk_parity:.3e} > {CPU_KK_PARITY_TOL:.3e}")
        entry["cpu_kk_parity_max_abs"] = cpu_kk_parity
        entry["cpu_kk_parity_energy_max_abs"] = cpu_kk_parity_energy
        entry["cpu_kk_parity_force_max_abs"] = cpu_kk_parity_force
        entry["cpu_kk_parity_abs_tolerance"] = CPU_KK_PARITY_TOL
        entry["cpu_kk_parity_energy_relative_tolerance"] = parity_energy_bound_rel
        entry["cpu_kk_parity_force_relative_tolerance"] = parity_force_bound_rel
        report["structures"][label] = entry
        frame0 = False

    report["errors"] = errors
    report["status"] = "PASS" if not errors else "FAIL"
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--cpu-lmp", type=Path, required=True)
    parser.add_argument("--kk-lmp", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--fitted-model", action="store_true",
        help="use relative tolerances (energy/force/parity/virial all "
             "REL*reference_scale+1e-12) instead of the fixture/slice absolute "
             "tolerances; numdiff force becomes informational-only in this mode too.",
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    report = validate_model(
        tag=args.tag, model_path=args.model, reference_path=args.reference,
        cpu_lmp=args.cpu_lmp, kk_lmp=args.kk_lmp, output_dir=args.output,
        fitted_model=args.fitted_model,
    )
    payload = json.dumps(report, default=str, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.write_text(payload, encoding="utf-8")
    print(payload)
    sys.exit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
