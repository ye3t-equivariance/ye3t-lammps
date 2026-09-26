import json
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
VERIFIER = REPOSITORY / "examples" / "PACKAGES" / "ye3t" / "verify_examples.py"


def dump(force="0.0", stress="0.0"):
    return f"""ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z c_atom_energy c_atom_stress[1] c_atom_stress[2] c_atom_stress[3] c_atom_stress[4] c_atom_stress[5] c_atom_stress[6] fx fy fz
1 1 0 0 0 -1 {stress} 0 0 0 0 0 {force} 0 0
"""


def run_verifier(directory, config):
    config_path = directory / "verify.config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    environment = dict(os.environ)
    environment["CONFIG_PATH"] = str(config_path)
    return subprocess.run(
        [sys.executable, str(VERIFIER)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_dump_comparison_rejects_nonfinite_values(tmp_path):
    reference = tmp_path / "reference.dump"
    candidate = tmp_path / "candidate.dump"
    reference.write_text(dump(), encoding="utf-8")
    candidate.write_text(dump(force="nan"), encoding="utf-8")

    result = run_verifier(
        tmp_path,
        {"check": "dumps", "reference": str(reference), "candidates": [str(candidate)]},
    )

    assert result.returncode == 2
    assert "non-finite fx" in result.stderr


def test_dump_comparison_converts_metal_stress_volume_to_ev(tmp_path):
    reference = tmp_path / "reference.dump"
    candidate = tmp_path / "candidate.dump"
    reference.write_text(dump(), encoding="utf-8")
    candidate.write_text(dump(stress="1.6021765e-3"), encoding="utf-8")

    result = run_verifier(
        tmp_path,
        {
            "check": "dumps",
            "reference": str(reference),
            "candidates": [str(candidate)],
            "virial_atol": 2e-9,
            "total_virial_atol": 2e-9,
        },
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    comparison = report["comparisons"][0]
    assert abs(comparison["maximum_absolute_virial_component_eV"] - 1.0e-9) < 1.0e-20


def test_md_check_requires_finite_bounded_energy_drift(tmp_path):
    log = tmp_path / "log.md"
    log.write_text(
        """LAMMPS test log
   Step Atoms Temp PotEng KinEng TotEng Press
      0    10  300   -11.0    1.0  -10.0000 0.0
     20    10  299   -10.9    0.9   -9.9995 0.0
Loop time of 1 on 1 procs for 20 steps with 10 atoms
""",
        encoding="utf-8",
    )

    result = run_verifier(tmp_path, {"check": "md", "log": str(log)})

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["elapsed_steps"] == 20
    assert abs(report["maximum_absolute_total_energy_drift_eV_per_atom"] - 5.0e-5) < 1.0e-14


def test_unknown_check_and_missing_config_are_rejected(tmp_path):
    result = run_verifier(tmp_path, {"check": "unknown"})
    assert result.returncode == 2
    assert "'check' must be one of" in result.stderr

    environment = dict(os.environ)
    environment.pop("CONFIG_PATH", None)
    result = subprocess.run(
        [sys.executable, str(VERIFIER)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 2
    assert "CONFIG_PATH" in result.stderr
