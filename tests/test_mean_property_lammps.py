"""Optional pinned-host integration check for the per-atom property compute."""

import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


binary = Path(sys.argv[1]).resolve()
plugin = None if sys.argv[2] == "-" else Path(sys.argv[2]).resolve()
fixtures = Path(sys.argv[3]).resolve()
mode = sys.argv[4]
launcher = sys.argv[5] if mode == "mpi2" else None
compute_style = os.environ.get("YE3T_PROPERTY_COMPUTE_STYLE", "ye3t/property/atom")
model_family = os.environ.get("YE3T_PROPERTY_MODEL_FAMILY", "all")
lammps_args = shlex.split(os.environ.get("YE3T_PROPERTY_LAMMPS_ARGS", ""))
save_rows = os.environ.get("YE3T_PROPERTY_SAVE_ROWS")
if mode not in {"single", "mpi2"}:
    raise SystemExit("expected single or mpi2 mode")
if model_family not in {"all", "tagged", "density"}:
    raise SystemExit("expected all, tagged, or density model family")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


binary_sha256 = sha256_file(binary) if save_rows else None
if save_rows:
    Path(save_rows).mkdir(parents=True, exist_ok=True)
    if any(Path(save_rows).glob("*.json")):
        raise SystemExit("saved property row destination already contains JSON cases")


def run_case(directory, name, group_null=False, newton_off=False,
             transformation=None, isolated=False, skin_only=False):
    record = json.loads((fixtures / (name + ".fixture.json")).read_text())
    width = len(record["expected"][0])
    species = record["case"]["species"]
    periodic = name in {"tagged_l2_tag2_periodic", "density_l2_periodic"}
    lines = ["units metal", "atom_style atomic"]
    if newton_off:
        lines.append("newton off")
    lines += ["boundary p f f" if periodic else "boundary f f f"]
    if mode == "mpi2":
        lines.append("processors 2 1 1")
    lines += [
        "region box block 0 2.1 -3 3 -3 3 units box" if periodic
        else "region box block -5 8 -5 8 -5 8 units box",
        "create_box {} box".format(len(species) + int(group_null)),
    ]
    positions = ([[0.0, 0.0, 0.0], [1.2, 0.2, 0.1]] if periodic else
                 [[0.0, 0.0, 0.0], [1.0, 0.2, 0.1],
                  [0.1, 1.2, 0.2], [0.2, 0.1, 1.3]])
    if name == "tagged_l1_uneven":
        positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    if name == "tagged_l2_tag2_one_neighbor":
        positions = [[0.0, 0.0, 0.0], [1.2, 0.2, 0.1]]
    if name == "tagged_l1_rank3_mixed":
        positions = [[0.0, 0.0, 0.0], [1.1, 0.2, 0.1],
                     [0.1, 1.3, 0.2], [0.2, 0.1, 1.4]]
    if name == "density_l1_two_species":
        positions = [[0.0, 0.0, 0.0], [0.8, 0.2, 0.1], [0.1, 0.9, 0.3]]
    if name == "density_l2_angular":
        positions = [[0.0, 0.0, 0.0], [0.8, 0.2, 0.1],
                     [0.1, 0.9, 0.3], [0.2, 0.1, 1.0]]
    if name == "density_l2_rank3_multipath":
        positions = [[0.0, 0.0, 0.0], [0.6, 0.2, 0.1],
                     [0.1, 0.7, 0.2], [0.2, 0.1, 0.8]]
    positions = positions[:len(record["central_species"])]
    atom_species = record["central_species"]
    if isolated:
        positions = positions[:1]
        atom_species = atom_species[:1]
    if skin_only:
        positions = [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]
        atom_species = atom_species[:2]
    if transformation == "rotation":
        matrix = record["rotation_matrix"]
        positions = [[sum(matrix[row][column] * position[column]
                          for column in range(3)) for row in range(3)]
                     for position in positions]
    elif transformation == "inversion":
        positions = [[-value for value in position] for position in positions]
    for atom_type, position in zip(atom_species, positions):
        lines.append("create_atoms {} single {} {} {} units box".format(
            atom_type + 1, *position))
    if group_null:
        lines += ["create_atoms {} single 0.5 0.5 0.5 units box".format(
                      len(species) + 1), "group selected id 1 2"]
    for atom_type in range(1, len(species) + 1 + int(group_null)):
        lines.append("mass {} 1.0".format(atom_type))
    lines += [
        "neighbor 0.3 bin",
        "pair_style zero 2.4",
        "pair_coeff * *",
        "compute p {} {} {} {}{}".format(
            "selected" if group_null else "all",
            compute_style,
            (fixtures / (name + ".ye3t.json")).as_posix(),
            " ".join(species),
            " NULL" if group_null else ""),
    ]
    if plugin is not None:
        lines.insert(-1, "plugin load " + plugin.as_posix())
    dump_path = directory / "property.dump"
    lines += [
        "dump d all custom 1 {} id type {}".format(
            dump_path.as_posix(), " ".join("c_p[{}]".format(index + 1)
                                            for index in range(width))),
        "dump_modify d sort id",
        "dump_modify d format float %.17g",
        "run 0",
        "run 1",
    ]
    script = directory / "in.property"
    script.write_text("\n".join(lines) + "\n")
    command = ([launcher, "-n", "2"] if launcher else []) + [
        binary.as_posix(), *lammps_args,
        "-in", script.as_posix(), "-log", "none"]
    completed = subprocess.run(command, cwd=directory, capture_output=True,
                               text=True, timeout=120)
    if completed.returncode:
        raise AssertionError(completed.stdout + completed.stderr)
    dump = dump_path.read_text().splitlines()
    header = "ITEM: ATOMS id type " + " ".join("c_p[{}]".format(index + 1)
                                                for index in range(width))
    heads = [index for index, line in enumerate(dump) if line == header]
    if len(heads) < 2:
        raise AssertionError("both run 0 and run 1 must write property rows")
    expected = record["expected"]
    if transformation == "rotation":
        expected = record["rotated_expected"]
    elif transformation == "inversion":
        expected = [[(-1) ** record["case"]["L"] * value for value in row]
                    for row in expected]
    if group_null:
        expected = expected[:2] + [[0.0] * width for _ in range(len(expected) - 1)]
    if isolated:
        expected = [[0.0] * width]
    if skin_only:
        expected = [[0.0] * width for _ in range(2)]
    rows = [list(map(float, dump[heads[-1] + 1 + index].split()))
            for index in range(len(expected))]
    actual = [row[2:] for row in rows]
    if [int(row[0]) for row in rows] != list(range(1, len(rows) + 1)):
        raise AssertionError("LAMMPS dump atom IDs changed")
    maximum = max(abs(value - reference)
                  for values, references in zip(actual, expected)
                  for value, reference in zip(values, references))
    if not math.isfinite(maximum) or maximum > record["tolerance"]:
        raise AssertionError("{}: max error {} exceeds {}".format(
            name, maximum, record["tolerance"]))
    if save_rows:
        destination = Path(save_rows)
        model_sha256 = sha256_file(fixtures / (name + ".ye3t.json"))
        (destination / (directory.name + "_" + name + ".json")).write_text(
            json.dumps({"model": name, "mode": mode, "newton_off": newton_off,
                        "group_null": group_null, "isolated": isolated,
                        "skin_only": skin_only,
                        "transformation": transformation,
                        "compute_style": compute_style,
                        "binary_sha256": binary_sha256,
                        "model_sha256": model_sha256,
                        "rows": actual},
                       sort_keys=True))
    print(name, "mpi2" if launcher else "single", "newton_off", newton_off,
          "group_null", group_null, "isolated", isolated,
          "skin_only", skin_only, "transformation", transformation,
          "max_abs_error", maximum)


with tempfile.TemporaryDirectory(prefix="ye3t-property-", dir=Path.cwd()) as root:
    root = Path(root)
    if mode == "single":
        cases = [("tagged_l1_ni", False, None),
                 ("tagged_l1_tag1", False, None),
                 ("tagged_l1_rank3_mixed", False, None),
                 ("tagged_l1_rank3_mixed", False, "rotation"),
                 ("tagged_l1_two_species", False, None),
                 ("tagged_l1_uneven", False, None),
                 ("tagged_l2_l2_ni", False, None),
                 ("tagged_l2_l2_ni", False, "rotation"),
                 ("tagged_l1_tag1", True, None),
                 ("tagged_l1_tag1", False, "rotation"),
                 ("tagged_l1_tag1", False, "inversion"),
                 ("tagged_l2_tag2_periodic", False, None),
                 ("density_l1_two_species", False, None),
                 ("density_l1_two_species", True, None),
                 ("density_l2_angular", False, None),
                 ("density_l2_angular", False, "rotation"),
                 ("density_l2_periodic", False, None),
                 ("density_l2_rank3_multipath", False, None),
                 ("density_l2_rank3_multipath", False, "rotation")]
        cases.append(("tagged_l2_tag2_one_neighbor", False, None))
        if model_family != "all":
            cases = [case for case in cases if case[0].startswith(model_family)]
        for number, (name, group_null, transformation) in enumerate(cases):
            directory = root / str(number)
            directory.mkdir()
            run_case(directory, name, group_null=group_null,
                     transformation=transformation)
        if model_family != "density":
            directory = root / str(len(cases))
            directory.mkdir()
            run_case(directory, "tagged_l1_ni", isolated=True)
            directory = root / str(len(cases) + 1)
            directory.mkdir()
            run_case(directory, "tagged_l1_ni", skin_only=True)
    else:
        number = 0
        for name in ("tagged_l2_tag2_periodic", "density_l2_periodic"):
            if model_family != "all" and not name.startswith(model_family):
                continue
            for off in (False, True):
                directory = root / str(number)
                directory.mkdir()
                run_case(directory, name, newton_off=off)
                number += 1
