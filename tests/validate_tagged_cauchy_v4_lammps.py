"""Native V4 pair-style parity: chemistry, cutoffs, references, MPI and migration.

This is software validation of a supplied model, not physical MD qualification.
All generated files go to the explicit --output directory.
"""

import argparse
import itertools
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from ase.data import atomic_masses, atomic_numbers

from ye3t_ace.tagged_cauchy_image import load_tagged_cauchy_image_model


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--lammps", type=Path, required=True)
parser.add_argument("--plugin", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--ranks", type=int, nargs="+", default=[1, 4])
parser.add_argument("--compiler-validation", choices=["full", "certificate"], default="full")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
if args.plugin:
    # A missing PLUGIN package or incompatible MPI/header ABI must fail before
    # loading the large proof or generating the independent reference cases.
    probe = args.output/"in.plugin_probe"
    probe.write_text(f'plugin load "{args.plugin.resolve()}"\nunits metal\natom_style atomic\n'
                     'pair_style ye3t model_family tagged_cauchy block_policy direct\n')
    check = subprocess.run([str(args.lammps.resolve()), "-in", probe.name, "-log", "plugin_probe.log",
                            "-screen", "plugin_probe.screen"], cwd=args.output,
                           capture_output=True, text=True, timeout=30)
    assert check.returncode == 0, (args.output/"plugin_probe.screen").read_text()+check.stderr
started = time.perf_counter()
model = load_tagged_cauchy_image_model(args.model, compiler_validation=args.compiler_validation)
loaded = time.perf_counter()-started
print(f"Python model loaded in {loaded:.2f}s", flush=True)
species = model.species_order
assert len(species) >= 2
assert model.reference_terms.get("zbl"), "This gate must exercise bound ZBL."
pairs = model.reference_terms["zbl"].get("pair_cutoffs_A")
assert pairs, "This gate must exercise pair-specific switches."
cutoffs = model.evaluator.source_plan["pair_cutoffs_A"]
box = max(model.evaluator.source_plan["cutoff"]*4, 24.0)
cell = np.eye(3)*box
cases = []
for left, right in itertools.combinations_with_replacement(species, 2):
    inner, outer = pairs.get(left+"-"+right, pairs.get(right+"-"+left))
    cut = cutoffs[left+"-"+right]
    for radius in (inner*0.8, inner, (inner+outer)/2, outer, outer+1e-5, cut-1e-5, cut, cut+1e-5):
        positions = np.array([[box/3]*3, [box/3+radius, box/3, box/3]])
        cases.append((left+"_"+right+"_"+format(radius, ".7g"), [left, right], positions))
# Cross a periodic boundary and repeat after reversing atom order.
positions = np.array([[0.2, 2.0, 2.0], [box-0.4, 2.1, 2.0],
                      [0.6, 3.2, 2.4], [1.6, 2.4, 3.0]])
symbols = [species[index % len(species)] for index in range(4)]
cases.extend([("periodic", symbols, positions), ("reordered", symbols[::-1], positions[::-1].copy())])
positions = np.array([[box/2-0.05, 10, 10], [box/2+1.3, 10.3, 10],
                      [box/2+0.7, 11.4, 10.4], [box/2-0.7, 10.4, 11.2]])
cases.append(("before_migration", symbols, positions))
moved = positions.copy()
moved[0, 0] += 0.2
cases.append(("after_migration", symbols, moved))
cases.extend([("repeated_run", symbols, moved), ("restart_reinitialized", symbols, moved)])

expected = []
for name, symbols, positions in cases:
    types = [model.evaluator.type_map[symbol] for symbol in symbols]
    result = model.energy_forces_virial(positions, types, cell=cell, pbc=True)
    expected.append({"name": name, "energy": float(result[0]), "forces": result[1].numpy(),
                     "virial": result[2].numpy(), "atomic": result[3].numpy()})
print(f"Computed {len(expected)} Python reference cases", flush=True)

records = []
for ranks in args.ranks:
    run = args.output/("mpi_"+str(ranks))
    run.mkdir(exist_ok=True)
    lines = ([f'plugin load "{args.plugin.resolve()}"'] if args.plugin else []) + [
        "units metal", "atom_style atomic", "boundary p p p", "atom_modify map array sort 0 0",
        f"processors {ranks} 1 1", f"region domain block 0 {box} 0 {box} 0 {box}",
        f"create_box {len(species)} domain"]
    lines += [f"mass {index+1} {atomic_masses[atomic_numbers[symbol]]:.17g}"
              for index, symbol in enumerate(species)]
    model_setup = ["pair_style ye3t model_family tagged_cauchy block_policy direct",
              f'pair_coeff * * "{args.model.resolve()}" '+" ".join(species),
              "newton on", "neighbor 0.3 bin", "neigh_modify delay 0 every 1 check yes",
              "compute atomic all pe/atom", "compute pressure all pressure NULL pair",
              "compute atomic_sum all reduce sum c_atomic",
              "thermo_style custom step atoms pe c_atomic_sum c_pressure[1] c_pressure[2] c_pressure[3] c_pressure[4] c_pressure[5] c_pressure[6]",
              "thermo_modify format float %.17g"]
    lines += model_setup
    for index, (name, symbols, positions) in enumerate(cases):
        if name == "after_migration":
            lines += ["group mover id 1", "timestep 0.001",
                      "fix migration mover move linear 200.0 0.0 0.0 units box", "run 1", "unfix migration"]
        elif name == "repeated_run":
            lines += ["run 0"]
        elif name == "restart_reinitialized":
            # Pair coefficients are external model data (restartinfo=0): reload
            # the same hash-bound artifact after reading the atomic state.
            lines += ["write_restart state.restart", "clear", "units metal", "atom_style atomic",
                      "atom_modify map array sort 0 0", f"processors {ranks} 1 1", "read_restart state.restart"]
            lines += model_setup + ["run 0"]
        else:
            lines += ["delete_atoms group all"]
            lines += [f"create_atoms {species.index(symbol)+1} single "+" ".join(f"{x:.17g}" for x in position)+" units box"
                      for symbol, position in zip(symbols, positions)]
            lines += ["run 0"]
        # LAMMPS metal nktv2p, intentionally its rounded units constant rather
        # than ASE's newer physical conversion (which would spoil parity).
        values = "$(pe:%.17g) " + " ".join(f"$(c_pressure[{component}]*vol/1.6021765e6:%.17g)" for component in range(1, 7))
        lines += [f'print "{values}" file {index}.energy screen no',
                  f"write_dump all custom {index}.dump id proc fx fy fz c_atomic modify sort id format float %.17g"]
    deck = run/"in.validation"
    deck.write_text("\n".join(lines)+"\n")
    command = (["mpirun", "-n", str(ranks)] if ranks > 1 else []) + [str(args.lammps.resolve()), "-in", deck.name,
                                                                              "-log", "lammps.log", "-screen", "screen.log"]
    begin = time.perf_counter()
    result = subprocess.run(command, cwd=run, capture_output=True, text=True, timeout=900)
    (run/"launcher.log").write_text(result.stdout+result.stderr)
    assert result.returncode == 0, (run/"screen.log").read_text()[-6000:]+result.stderr
    maxima = dict(energy=0.0, forces=0.0, virial=0.0, atomic=0.0)
    owners = []
    for index, oracle in enumerate(expected):
        values = np.loadtxt(run/f"{index}.energy")
        lines = (run/f"{index}.dump").read_text().splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith("ITEM: ATOMS"))+1
        rows = np.array([[float(value) for value in line.split()] for line in lines[start:]])
        actual = {"energy": values[0], "virial": values[1:], "forces": rows[:, 2:5], "atomic": rows[:, 5]}
        owners.append(rows[:, 1].astype(int))
        for field, value in actual.items():
            error = float(np.max(np.abs(value-oracle[field])))
            maxima[field] = max(maxima[field], error)
            np.testing.assert_allclose(value, oracle[field], rtol=2e-9, atol=2e-7,
                                       err_msg=f"ranks={ranks} case={oracle['name']} {field}")
    if ranks > 1:
        before = next(i for i, case in enumerate(cases) if case[0] == "before_migration")
        assert owners[before][0] != owners[before+1][0], "MPI case did not migrate atom 1."
    records.append({"ranks": ranks, "passed": True, "cases": len(cases), "seconds": time.perf_counter()-begin,
                    "maximum_absolute_errors": maxima, "command": command, "domain_migration": ranks > 1})
    print(json.dumps(records[-1]), flush=True)

# Independent energy finite differences on the migrated periodic geometry.
types = [model.evaluator.type_map[symbol] for symbol in symbols]
reference = expected[-1]
step = 2e-6
force_errors, virial_errors = [], []
for axis in range(3):
    plus, minus = moved.copy(), moved.copy()
    plus[0, axis] += step
    minus[0, axis] -= step
    derivative = -(float(model.energy_forces_virial(plus, types, cell=cell, pbc=True)[0])-
                   float(model.energy_forces_virial(minus, types, cell=cell, pbc=True)[0]))/(2*step)
    force_errors.append(abs(derivative-reference["forces"][0, axis]))
for component, (row, column) in enumerate(((0, 0), (1, 1), (2, 2), (1, 0), (2, 0), (2, 1))):
    strain = np.zeros((3, 3)); strain[row, column] = step
    energies = [float(model.energy_forces_virial(moved @ transform.T, types,
        cell=cell @ transform.T, pbc=True)[0]) for transform in (np.eye(3)+strain, np.eye(3)-strain)]
    virial_errors.append(abs(-(energies[0]-energies[1])/(2*step)-reference["virial"][component]))
assert max(force_errors+virial_errors) < 3e-4
summary = {"passed": True, "model": str(args.model.resolve()), "python_load_seconds": loaded,
           "records": records, "force_fd_max": max(force_errors), "virial_fd_max": max(virial_errors),
           "total_seconds": time.perf_counter()-started, "scope": "CPU software parity; physical MD stability unqualified"}
(args.output/"validation.json").write_text(json.dumps(summary, indent=2)+"\n")
print(json.dumps(summary, indent=2), flush=True)
