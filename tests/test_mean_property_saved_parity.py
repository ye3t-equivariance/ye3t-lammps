"""Compare retained CPU and device rows from the same integration cases."""

import json
import math
import sys
from pathlib import Path


cpu = Path(sys.argv[1])
device = Path(sys.argv[2])
limit = float(sys.argv[3]) if len(sys.argv) > 3 else 2e-9
cpu_files = {path.name: path for path in cpu.glob("*.json")}
device_files = {path.name: path for path in device.glob("*.json")}
if not cpu_files or cpu_files.keys() != device_files.keys():
    raise AssertionError("CPU and device case inventories differ")

maximum = 0.0
for name, path in sorted(cpu_files.items()):
    baseline = json.loads(path.read_text())
    candidate = json.loads(device_files[name].read_text())
    if (baseline["compute_style"] != "ye3t/property/atom" or
            candidate["compute_style"] != "ye3t/property/atom/kk/device"):
        raise AssertionError(name + ": CPU/device compute styles not identified")
    omitted = {"rows", "compute_style"}
    if {key: value for key, value in baseline.items() if key not in omitted} != {
            key: value for key, value in candidate.items() if key not in omitted}:
        raise AssertionError(name + ": case metadata differs")
    if len(baseline["rows"]) != len(candidate["rows"]):
        raise AssertionError(name + ": atom count differs")
    for left, right in zip(baseline["rows"], candidate["rows"]):
        if len(left) != len(right):
            raise AssertionError(name + ": component count differs")
        for value, reference in zip(right, left):
            difference = abs(value - reference)
            if not math.isfinite(difference) or difference > limit * max(1.0, abs(reference)):
                raise AssertionError(name + ": CPU/device row differs")
            maximum = max(maximum, difference)
print(len(cpu_files), "cases; maximum CPU/device absolute difference", maximum)
