"""Emit native CPU parity fixtures from the public fitted full-M workflow."""

import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from ye3t import YE3TRepresentation
from ye3t_methods import Basis, LinearModel
from ye3t_methods.tesseral_targets import (cartesian_to_real_tesseral,
                                           real_tesseral_to_cartesian)


def make_case(directory, name, rank, L, species, atoms, tag_counts, lmax=1,
              fit_atoms=None, nmax=2):
    parity = "odd" if L == 1 else "even"
    representation = YE3TRepresentation.from_config({
        "group": "O3", "ranks": [rank],
        "parent": {"young_lambda": "(N)", "L": L, "parity": parity},
        "factorization": "cauchy", "subspace": "full",
        "uncoupled_factor_inputs": {
            "eta_count_per_rank": {rank: nmax}, "l_max_per_rank": {rank: lmax}},
        "intermediates": {"young_kappa": "all_valid",
                          "block_rotation": {"policy": "all_valid"}},
    })
    basis_config = {
        "single_factors": {"species": list(species),
                           "radial": {"family": "shifted_jacobi", "cutoff_A": 2.4},
                           "chemical": {"kind": "explicit"}},
        "tensor_product": {"kind": "tagged",
                           "tag_counts_per_rank": {rank: tag_counts}},
        "catalogue": {"ranks": [rank], "nmax_per_rank": {rank: nmax},
                      "lmax_per_rank": {rank: lmax},
                      "source_block_partitions_by_rank": {rank: [[rank]]}},
    }
    runtime = {"evaluator": "reference", "neighbors": "ase",
               "cache": {"mode": "off"}, "dtype": "float64", "device": "cpu"}
    basis = Basis.from_config(basis_config, representation=representation,
                              runtime=runtime)
    training_atoms = atoms if fit_atoms is None else fit_atoms
    rows = basis.create(training_atoms)
    ordered_species = basis.elements
    coefficients = np.linspace(.15, .8, len(ordered_species) * len(basis.labels)).reshape(
        len(ordered_species), len(basis.labels))
    training_central = np.array([ordered_species.index(symbol)
                                 for symbol in training_atoms.get_chemical_symbols()])
    training_atoms.new_array("target", np.einsum(
        "nfm,nf->nm", rows, coefficients[training_central]))
    config = {
        "metadata": {"schema": "ye3t_config_v1", "name": name,
                     "status": "experimental"},
        "representation": representation.to_dict(), "basis": basis_config,
        "runtime": runtime,
        "model": {"kind": "linear", "output": {"scope": "per_atom"},
                  "fit": {"solver": "ridge", "alpha": 1e-10}},
        "targets": {"per_atom": {"key": "target", "input": "real_tesseral",
                                  "units": "arbitrary"}},
        "validation": {"checks": []},
    }
    model = LinearModel(basis).fit([training_atoms], config=config)
    model_path = model.write(directory / (name + ".ye3t.json"))
    if name == "tagged_l1_rank3_mixed":
        plan = json.loads(model_path.read_text())["native_property_plan"]
        mixed_ids = {row["label"]["coordinate_id"]
                     for schedule in plan["schedules"]
                     for row in schedule["inventory"]
                     if row["label"]["block_kappas"] == [[2, 1]]}
        assert any(row["coordinate_id"] in mixed_ids and
                   abs(plan["readout"]["coefficients_by_species"][0][index]) > 1e-8
                   for index, row in enumerate(plan["feature_slices"]))
        assert any(schedule["tag_count"] == 2 and
                   schedule["support_tag_count"] == 1
                   for schedule in plan["schedules"])
    centers, neighbors, shifts = neighbor_list("ijS", atoms, basis.cutoff)
    order = np.argsort(centers, kind="stable")
    centers, neighbors, shifts = centers[order], neighbors[order], shifts[order]
    counts = np.bincount(centers, minlength=len(atoms))
    if name == "tagged_l1_uneven":
        assert counts.tolist() == [1, 2, 1]
    offsets = np.concatenate(([0], np.cumsum(counts)))
    vectors = (atoms.positions[neighbors] - atoms.positions[centers] +
               shifts @ atoms.cell.array)
    angle_z, angle_y = np.deg2rad([37.0, 23.0])
    cz, sz = np.cos(angle_z), np.sin(angle_z)
    cy, sy = np.cos(angle_y), np.sin(angle_y)
    rotation = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array(
        [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rotated = atoms.copy()
    rotated.positions = atoms.positions @ rotation.T
    rotated.cell = atoms.cell.array @ rotation.T
    expected = model.predict(atoms)["mean_real_tesseral"]
    if name == "tagged_l2_tag2_one_neighbor":
        np.testing.assert_allclose(expected, 0.0, atol=1e-12, rtol=0.0)
    cartesian = real_tesseral_to_cartesian(expected, L, parity)
    if L == 1:
        expected_rotated = cartesian_to_real_tesseral(
            cartesian @ rotation.T, L, parity)
    else:
        expected_rotated = cartesian_to_real_tesseral(
            np.einsum("ab,nbc,dc->nad", rotation, cartesian, rotation),
            L, parity)
    np.testing.assert_allclose(
        model.predict(rotated)["mean_real_tesseral"], expected_rotated,
        atol=2e-9, rtol=2e-9)
    inverted = atoms.copy()
    inverted.positions = -atoms.positions
    inverted.cell = -atoms.cell.array
    np.testing.assert_allclose(
        model.predict(inverted)["mean_real_tesseral"], (-1) ** L * expected,
        atol=2e-9, rtol=2e-9)
    if name == "tagged_l2_periodic":
        assert len({(int(i), int(j)) for i, j in zip(centers, neighbors)}) < len(centers)
    record = {
        "schema": "ye3t_mean_property_cpu_fixture_v1",
        "model": model_path.name,
        "central_species": [ordered_species.index(symbol)
                            for symbol in atoms.get_chemical_symbols()],
        "edge_offsets": offsets.tolist(),
        "neighbor_species": [ordered_species.index(atoms[j].symbol)
                             for j in neighbors],
        "edge_vectors": vectors.reshape(-1).tolist(),
        "expected": expected.tolist(),
        "rotated_edge_vectors": (vectors @ rotation.T).reshape(-1).tolist(),
        "rotated_expected": expected_rotated.tolist(),
        "rotation_matrix": rotation.tolist(),
        "tolerance": 2e-9,
        "case": {"L": L, "rank": rank, "species": list(ordered_species),
                 "periodic_occurrences": len(centers),
                 "selected_features": len(basis.labels)},
    }
    (directory / (name + ".fixture.json")).write_text(json.dumps(
        record, sort_keys=True, separators=(",", ":")))


output = Path(sys.argv[1]).resolve()
output.mkdir(parents=True, exist_ok=True)
make_case(output, "tagged_l1_ni", 1, 1, ("Ni",), Atoms(
    "Ni4", positions=[[0, 0, 0], [1.0, .2, .1], [.1, 1.2, .2],
                      [.2, .1, 1.3]]), [0, 1])
make_case(output, "tagged_l1_tag1", 1, 1, ("Ni",), Atoms(
    "Ni4", positions=[[0, 0, 0], [1.0, .2, .1], [.1, 1.2, .2],
                      [.2, .1, 1.3]]), [1])
make_case(output, "tagged_l2_l2_ni", 1, 2, ("Ni",), Atoms(
    "Ni4", positions=[[0, 0, 0], [1.0, .2, .1], [.1, 1.2, .2],
                      [.2, .1, 1.3]]), [0, 1], lmax=2)
make_case(output, "tagged_l2_periodic", 2, 2, ("Ni",), Atoms(
    "Ni2", positions=[[0, 0, 0], [1.0, .2, .1]], cell=[2.1, 6, 6],
    pbc=[True, False, False]), [0, 1, 2])
make_case(output, "tagged_l2_periodic_split", 2, 2, ("Ni",), Atoms(
    "Ni2", positions=[[0, 0, 0], [1.2, .2, .1]], cell=[2.1, 6, 6],
    pbc=[True, False, False]), [0, 1, 2])
make_case(output, "tagged_l2_tag2_periodic", 2, 2, ("Ni",), Atoms(
    "Ni2", positions=[[0, 0, 0], [1.2, .2, .1]], cell=[2.1, 6, 6],
    pbc=[True, False, False]), [2])
make_case(output, "tagged_l1_two_species", 1, 1, ("Ni", "Cu"), Atoms(
    "NiCuNi", positions=[[0, 0, 0], [1.0, .2, .1], [.1, 1.2, .2]]),
    [0, 1])
make_case(output, "tagged_l1_uneven", 1, 1, ("Ni",), Atoms(
    "Ni3", positions=[[0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]]), [0, 1])
make_case(output, "tagged_l2_tag2_one_neighbor", 2, 2, ("Ni",), Atoms(
    "Ni2", positions=[[0, 0, 0], [1.2, .2, .1]]), [2],
    fit_atoms=Atoms("Ni2", positions=[[0, 0, 0], [1.2, .2, .1]],
                    cell=[2.1, 6, 6], pbc=[True, False, False]))
make_case(output, "tagged_l1_rank3_mixed", 3, 1, ("Ni",), Atoms(
    "Ni4", positions=[[0, 0, 0], [1.1, .2, .1],
                     [.1, 1.3, .2], [.2, .1, 1.4]]),
    [2], nmax=1)
