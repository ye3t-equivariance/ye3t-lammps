#!/usr/bin/env python3
"""Focused tests for publication-overlay validation evidence."""

from pathlib import Path
import json
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_fitted_joint_overlay as validator
from validate_fitted_joint_overlay import GateValidationError, validate_pair_scan


def write_pair_log(root, token, separation, energy, force):
    directory = root / f"overlay_pair_r{token}"
    directory.mkdir()
    (directory / "log.lammps").write_text(
        "\n".join(
            (
                f"YE3T_OVERLAY_PAIR_R {separation}",
                f"YE3T_OVERLAY_PAIR_PE {energy}",
                f"YE3T_OVERLAY_PAIR_FX1 {force}",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def make_pair_scan(root, first_force):
    rows = (
        ("1p5", 1.5, 6.0, first_force),
        ("1p8", 1.8, 5.0, -4.0),
        ("2p2", 2.2, 4.0, -3.0),
        ("2p5", 2.5, 3.0, -2.0),
        ("2p86", 2.86, 2.0, -1.0),
        ("3p2", 3.2, 1.0, 0.5),
    )
    for token, separation, energy, force in rows:
        write_pair_log(root, token, separation, energy, force)


def test_pair_scan_returns_pass_evidence(tmp_path):
    make_pair_scan(tmp_path, -5.0)
    evidence = validate_pair_scan(tmp_path)
    assert evidence["repulsive_force_passed"] is True
    assert evidence["monotonic_energy_passed"] is True
    assert len(evidence["points"]) == 6


def test_pair_scan_failure_retains_all_points(tmp_path):
    make_pair_scan(tmp_path, 5.0)
    with pytest.raises(GateValidationError) as captured:
        validate_pair_scan(tmp_path)
    evidence = captured.value.evidence
    assert evidence["repulsive_force_passed"] is False
    assert evidence["monotonic_energy_passed"] is True
    assert evidence["points"][0]["separation_A"] == 1.5
    assert evidence["points"][0]["force_x_atom_1_eV_per_A"] == 5.0


def test_main_writes_failed_pair_scan_evidence(tmp_path, monkeypatch):
    make_pair_scan(tmp_path, 5.0)
    monkeypatch.setattr(validator, "validate_static_parity", lambda output: {})
    monkeypatch.setattr(
        validator, "validate_force_finite_difference", lambda output, delta: {}
    )
    monkeypatch.setattr(validator, "validate_eos", lambda output: {})
    monkeypatch.setattr(
        validator, "validate_virial_finite_difference", lambda output: {}
    )
    monkeypatch.setattr(validator, "validate_nve", lambda output: {})
    monkeypatch.setattr(validator, "validate_long_nve", lambda output: {})
    destination = tmp_path / "validation.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_fitted_joint_overlay.py",
            str(tmp_path),
            "--json",
            str(destination),
        ],
    )
    with pytest.raises(AssertionError, match="close_range_pair_scan"):
        validator.main()
    report = json.loads(destination.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert [row["gate"] for row in report["failures"]] == [
        "close_range_pair_scan"
    ]
    assert len(report["close_range_pair_scan"]["points"]) == 6
