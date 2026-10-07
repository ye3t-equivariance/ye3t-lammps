"""Emit ordinary-density full-M models and independent geometry rows."""

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


def make_case(directory, name, rank, L, lmax, atoms, nmax=1, partition=None):
    parity = "odd" if L == 1 else "even"
    species = list(dict.fromkeys(atoms.get_chemical_symbols()))
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
        "single_factors": {
            "species": species,
            "radial": {"family": "pace_chebexp_cos", "cutoff_A": 1.4,
                       "cutoff_width_A": .01, "lambda": .79},
            "chemical": {"kind": "explicit"}},
        "tensor_product": {"kind": "density"},
        "catalogue": {"ranks": [rank], "nmax_per_rank": {rank: nmax},
                      "lmax_per_rank": {rank: lmax},
                      "source_block_partitions_by_rank": {
                          rank: [partition if partition is not None else [rank]]}},
    }
    runtime = {"evaluator": "torch", "neighbors": "ase",
               "cache": {"mode": "off"}, "dtype": "float64", "device": "cpu"}
    basis = Basis.from_config(basis_config, representation=representation,
                              runtime=runtime)
    rows = basis.create(atoms)
    assert len(basis.labels) > 0
    beta = np.linspace(.2, .7, len(basis.labels))
    atoms.new_array("target", np.einsum("nfm,f->nm", rows, beta))
    config = {
        "metadata": {"schema": "ye3t_config_v1", "name": name,
                     "status": "experimental"},
        "representation": representation.to_dict(), "basis": basis_config,
        "runtime": runtime,
        "model": {"kind": "linear", "output": {"scope": "per_atom"},
                  "fit": {"solver": "ridge", "alpha": 0.0}},
        "targets": {"per_atom": {"key": "target", "input": "real_tesseral",
                                  "units": "arbitrary"}},
        "validation": {"checks": ["round_trip"]},
    }
    model = LinearModel(basis).fit([atoms], config=config)
    model_path = model.write(directory / (name + ".ye3t.json"))
    payload = json.loads(model_path.read_text())
    assert payload["schema"] == "ye3t_methods_density_full_m_per_atom_v2"
    assert payload["native_property_plan"]["selected_coordinate_ids"] == [
        label.identity for label in basis.labels]

    centers, neighbors, shifts = neighbor_list("ijS", atoms, basis.cutoff)
    order = np.argsort(centers, kind="stable")
    centers, neighbors, shifts = centers[order], neighbors[order], shifts[order]
    if atoms.pbc.any():
        assert len(set(zip(centers.tolist(), neighbors.tolist()))) < len(centers)
    counts = np.bincount(centers, minlength=len(atoms))
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
    assert np.max(np.abs(expected)) > 1e-8
    np.testing.assert_allclose(expected, atoms.arrays["target"],
                               atol=2e-9, rtol=2e-9)
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

    record = {
        "schema": "ye3t_mean_property_cpu_fixture_v1",
        "model": model_path.name,
        "central_species": [species.index(symbol)
                            for symbol in atoms.get_chemical_symbols()],
        "edge_offsets": offsets.tolist(),
        "neighbor_species": [species.index(atoms[j].symbol) for j in neighbors],
        "edge_vectors": vectors.reshape(-1).tolist(),
        "expected": expected.tolist(),
        "rotated_edge_vectors": (vectors @ rotation.T).reshape(-1).tolist(),
        "rotated_expected": expected_rotated.tolist(),
        "rotation_matrix": rotation.tolist(),
        "tolerance": 2e-9,
        "case": {"L": L, "rank": rank, "species": species,
                 "periodic_occurrences": len(centers),
                 "selected_features": len(basis.labels)},
    }
    (directory / (name + ".fixture.json")).write_text(json.dumps(
        record, sort_keys=True, separators=(",", ":")))
    print(name, "features", len(basis.labels), "directed_occurrences", len(centers))


output = Path(sys.argv[1]).resolve()
output.mkdir(parents=True, exist_ok=True)
make_case(output, "density_l1_two_species", 1, 1, 1, Atoms(
    "NiCuNi", positions=[[0, 0, 0], [.8, .2, .1], [.1, .9, .3]]))
make_case(output, "density_l2_angular", 1, 2, 2, Atoms(
    "Ni4", positions=[[0, 0, 0], [.8, .2, .1], [.1, .9, .3],
                     [.2, .1, 1.0]]))
make_case(output, "density_l2_periodic", 2, 2, 1, Atoms(
    "Ni2", positions=[[0, 0, 0], [1.2, .2, .1]], cell=[2.1, 6, 6],
    pbc=[True, False, False]))
make_case(output, "density_l2_rank3_multipath", 3, 2, 2, Atoms(
    "Ni4", positions=[[0, 0, 0], [.6, .2, .1], [.1, .7, .2],
                     [.2, .1, .8]]), nmax=2, partition=[2, 1])
