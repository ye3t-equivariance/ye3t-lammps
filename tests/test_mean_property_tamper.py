"""Reader rejects self-consistent unsupported conventions and broken bindings."""

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
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode()
    record["self_hash"] = hashlib.sha256(payload).hexdigest()


def check(directory, name, change, reseal_schedule=False, accepted=False,
          error_contains=None):
    model = copy.deepcopy(original)
    change(model)
    if reseal_schedule:
        for schedule in model["native_property_plan"]["schedules"]:
            seal(schedule)
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
    if error_contains and error_contains not in result.stdout + result.stderr:
        raise AssertionError(name + ": expected rejection reason " + error_contains)
    print(name, "accepted" if accepted else "rejected")


with tempfile.TemporaryDirectory(prefix="ye3t-property-tamper-") as temporary:
    directory = Path(temporary)
    check(directory, "target_group", lambda model: (
        model["coordinate_convention"].update(group="SO3"),
        model["native_property_plan"]["target"].update(group="SO3")))
    check(directory, "unsupported_target_parity", lambda model: (
        model["coordinate_convention"].update(parity="odd"),
        model["native_property_plan"]["target"].update(parity="odd")),
        error_contains="target natural parity changed")
    check(directory, "radial_units", lambda model:
          model["native_property_plan"]["source"]["radial"].update(units="Bohr"))
    check(directory, "unbound_schedule", lambda model:
          model["native_property_plan"]["schedules"][0]["coefficient_values"].__setitem__(
              0, 2.0))
    check(directory, "false_marginal", lambda model:
          model["native_property_plan"]["schedules"][0][
              "marginal_image_certificate"].update(identity="unverified"),
          reseal_schedule=True)

    def density_only_support(model):
        schedule = model["native_property_plan"]["schedules"][0]
        density_input = next(index for index, coordinate in enumerate(
            schedule["input_coordinates"]) if coordinate[1] == 1)
        first, last = schedule["term_offsets"][:2]
        assert last > first
        schedule["term_components"][first:last] = [density_input] * (last - first)

    check(directory, "density_only_support", density_only_support,
          reseal_schedule=True, error_contains="no explicit edge factor")

    def large_ratio(model):
        squared = model["native_property_plan"]["source"]["channels"][0][
            "normalization_squared"]
        squared["numerator"] *= 1 << 40
        squared["denominator"] *= 1 << 40

    check(directory, "large_exact_ratio", large_ratio, accepted=True)
