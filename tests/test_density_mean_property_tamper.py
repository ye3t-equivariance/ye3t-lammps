"""Reject self-consistent changes outside the density native contract."""

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


executable = Path(sys.argv[1]).resolve()
source = Path(sys.argv[2]).resolve()
original = json.loads(source.read_text())


def seal(record):
    record.pop("self_hash", None)
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode()
    record["self_hash"] = hashlib.sha256(encoded).hexdigest()


def check(directory, name, change, accepted=False, rebind_compiler=False,
          rebind_fit=False, diagnostic=None):
    model = copy.deepcopy(original)
    change(model)
    if rebind_compiler:
        encoded = json.dumps(model["compiled_density"], sort_keys=True,
                             separators=(",", ":"), ensure_ascii=False,
                             allow_nan=False).encode()
        compiler_hash = hashlib.sha256(encoded).hexdigest()
        model["compiler_hash"] = compiler_hash
        model["native_property_plan"]["compiler_hash"] = compiler_hash
        if rebind_fit:
            model["fit"]["fit_metadata"]["coordinate_penalty_metric"][
                "physical_image_plan_hash"] = compiler_hash
            fit_bytes = json.dumps(model["fit"], sort_keys=True,
                                   separators=(",", ":"), ensure_ascii=False,
                                   allow_nan=False).encode()
            model["native_property_plan"]["readout"]["fit_sha256"] = (
                hashlib.sha256(fit_bytes).hexdigest())
    seal(model["native_property_plan"])
    seal(model)
    path = directory / (name + ".ye3t.json")
    path.write_text(json.dumps(model, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False))
    result = subprocess.run([executable.as_posix(),
                             "--accept-model" if accepted else "--reject-model",
                             path.as_posix()], capture_output=True, text=True)
    if result.returncode:
        raise AssertionError(name + ": " + result.stdout + result.stderr)
    if diagnostic and diagnostic not in result.stdout + result.stderr:
        raise AssertionError(name + ": wrong rejection: " + result.stdout + result.stderr)
    print(name, "accepted" if accepted else "rejected")


with tempfile.TemporaryDirectory(prefix="ye3t-density-property-tamper-") as root:
    directory = Path(root)
    check(directory, "unchanged", lambda model: None, accepted=True)
    check(directory, "pair_cutoff", lambda model:
          model["native_property_plan"]["source"]["pair_cutoffs_A"].update(
              {"Ni-Ni": 1.3}))
    check(directory, "readout", lambda model:
          model["native_property_plan"]["readout"]["coefficients"].__setitem__(
              0, 1.0))
    check(directory, "real_form", lambda model:
          model["native_property_plan"]["target"]["real_to_complex_matrix"][0][0]
          .__setitem__(0, 0.5))
    check(directory, "compiler_coefficient", lambda model:
          model["compiled_density"]["specs_by_M"][0][0]["coeffs"][0]
          .__setitem__(0, 0.5))
    check(directory, "physical_eta", lambda model:
          model["native_property_plan"]["source"]["physical_eta_by_center"]
          ["Ni"][0]["chemical"].__setitem__("neighbor_species", "Cu"))
    check(directory, "magnetic_path", lambda model:
          model["compiled_density"]["specs_by_M"][0][0]["ms_combinations"][0]
          .__setitem__(0, 0))
    check(directory, "magnetic_path_rebound", lambda model:
          model["compiled_density"]["specs_by_M"][0][0]["ms_combinations"][0]
          .__setitem__(0, 0), rebind_compiler=True, rebind_fit=True,
          diagnostic="wrong total magnetic index")

    def changed_site_field(model, key, value):
        if isinstance(value, (int, float)) and isinstance(
                model["compiled_density"]["site_basis"][key], list):
            value = [value] * len(model["compiled_density"]["site_basis"][key])
        model["compiled_density"]["site_basis"][key] = value
        model["native_property_plan"]["source"]["site_basis"][key] = value

    check(directory, "lambda_zero", lambda model:
          changed_site_field(model, "lmbda", 0.0),
          rebind_compiler=True, rebind_fit=True,
          diagnostic="unsupported density directed bond parameters")
    check(directory, "spline_overflow", lambda model:
          changed_site_field(model, "pace_spline_spacing", 1e-20),
          rebind_compiler=True, rebind_fit=True,
          diagnostic="unsupported density directed bond parameters")
    check(directory, "dtype", lambda model:
          changed_site_field(model, "dtype", "float32"),
          rebind_compiler=True, rebind_fit=True,
          diagnostic="unsupported density physical source convention")
    check(directory, "rebound_source_with_old_fit", lambda model:
          changed_site_field(model, "lmbda", 0.8),
          rebind_compiler=True, diagnostic="density readout fit binding mismatch")

    def changed_source(model):
        model["compiled_density"]["site_basis"]["chemical_basis"] = "embedding"
        model["native_property_plan"]["source"]["site_basis"][
            "chemical_basis"] = "embedding"

    check(directory, "unsupported_chemistry", changed_source)
