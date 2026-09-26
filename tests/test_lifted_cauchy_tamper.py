#!/usr/bin/env python3
"""Reject coherently re-hashed lifted-Cauchy deployment-table tampering."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def canonical_hash(payload):
    body = {key: value for key, value in payload.items() if key != "self_hash"}
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_manifest(root):
    names = (
        "compiled_lifted_cauchy.json",
        "model.ye3t.json",
        "native_runtime.ye3t.json",
    )
    (root / "YE3T_LIFTED_BUNDLE_MANIFEST.sha256").write_text(
        "".join(f"{file_hash(root / name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


def require_rejected(executable, root, expected):
    result = subprocess.run(
        (str(executable), str(root)),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = result.stdout + result.stderr
    if result.returncode == 0:
        raise AssertionError("native loader accepted a coherently re-hashed tamper")
    if expected not in output:
        raise AssertionError("native loader rejected the wrong condition:\n" + output)


def require_accepted(executable, root):
    result = subprocess.run(
        (str(executable), str(root)),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise AssertionError(
            "native loader rejected a coherently bound certificate:\n"
            + result.stdout
            + result.stderr
        )


def tamper_readout(executable, fixture, destination):
    shutil.copytree(fixture, destination)
    model_path = destination / "model.ye3t.json"
    model = read_json(model_path)
    model["readout"]["coefficients"][0][0] += 0.125
    model["self_hash"] = canonical_hash(model)
    write_json(model_path, model)
    write_manifest(destination)
    require_rejected(executable, destination, "deployment identity mismatch")


def tamper_source_transform(executable, fixture, destination):
    shutil.copytree(fixture, destination)
    native_path = destination / "native_runtime.ye3t.json"
    native = read_json(native_path)
    native["source_groups"][0]["transform_q_from_f"][0][0] += 0.125
    native["self_hash"] = canonical_hash(native)
    write_json(native_path, native)

    model_path = destination / "model.ye3t.json"
    model = read_json(model_path)
    model["native_runtime_reference"]["plan_self_hash"] = native["self_hash"]
    model["native_runtime_reference"]["file_sha256"] = file_hash(native_path)
    model["self_hash"] = canonical_hash(model)
    write_json(model_path, model)
    write_manifest(destination)
    require_rejected(
        executable,
        destination,
        "native transform disagrees with the source certificate",
    )


def tamper_both_float_transforms(executable, fixture, destination):
    shutil.copytree(fixture, destination)
    native_path = destination / "native_runtime.ye3t.json"
    native = read_json(native_path)
    native["source_groups"][0]["transform_q_from_f"][0][0] += 0.125
    native["self_hash"] = canonical_hash(native)
    write_json(native_path, native)

    model_path = destination / "model.ye3t.json"
    model = read_json(model_path)
    model["source"]["groups"][0]["factorized_lowering"]["binary64_matrix"][0][
        0
    ] += 0.125
    model["native_runtime_reference"]["plan_self_hash"] = native["self_hash"]
    model["native_runtime_reference"]["file_sha256"] = file_hash(native_path)
    model["self_hash"] = canonical_hash(model)
    write_json(model_path, model)
    write_manifest(destination)
    require_rejected(
        executable,
        destination,
        "serialized transform disagrees with the exact certificate",
    )


def bind_rewritten_native(destination, native):
    native_path = destination / "native_runtime.ye3t.json"
    native["self_hash"] = canonical_hash(native)
    write_json(native_path, native)
    model_path = destination / "model.ye3t.json"
    model = read_json(model_path)
    model["native_runtime_reference"]["plan_self_hash"] = native["self_hash"]
    model["native_runtime_reference"]["file_sha256"] = file_hash(native_path)
    model["self_hash"] = canonical_hash(model)
    write_json(model_path, model)
    write_manifest(destination)


def tamper_native_polynomial(executable, fixture, destination, field):
    shutil.copytree(fixture, destination)
    native = read_json(destination / "native_runtime.ye3t.json")
    polynomial = native["heads"][0]["polynomial"]
    if field == "coefficient":
        polynomial["monomial_coefficients"][0] += 0.125
    elif field == "offset":
        polynomial["offset"] += 0.125
    else:
        raise AssertionError("unknown polynomial tamper")
    bind_rewritten_native(destination, native)
    require_rejected(
        executable,
        destination,
        "native polynomial disagrees with compiler/readout lowering",
    )


def tamper_physical_reality_hash(executable, fixture, destination):
    shutil.copytree(fixture, destination)
    native_path = destination / "native_runtime.ye3t.json"
    native = read_json(native_path)
    if native["schema"] != "ye3t_lifted_cauchy_native_runtime_v2":
        return
    compiler = read_json(destination / "compiled_lifted_cauchy.json")
    reality_hash = compiler["payload"]["equivalence_certificate"][
        "physical_scalar_reality_report_hash"
    ]
    native["certificates"]["physical_scalar_reality_report_hash"] = reality_hash
    bind_rewritten_native(destination, native)
    require_accepted(executable, destination)
    native = read_json(native_path)
    native["certificates"]["physical_scalar_reality_report_hash"] = "0" * 64
    bind_rewritten_native(destination, native)
    require_rejected(
        executable,
        destination,
        "compiler physical-reality certificate mismatch",
    )


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: test_lifted_cauchy_tamper.py EXECUTABLE FIXTURE")
    executable = Path(sys.argv[1]).resolve()
    fixture = Path(sys.argv[2]).resolve()
    with tempfile.TemporaryDirectory(prefix="ye3t_lifted_tamper_") as temporary:
        root = Path(temporary)
        tamper_readout(executable, fixture, root / "readout")
        tamper_source_transform(executable, fixture, root / "source_transform")
        tamper_both_float_transforms(
            executable, fixture, root / "both_float_transforms"
        )
        tamper_native_polynomial(
            executable, fixture, root / "polynomial_coefficient", "coefficient"
        )
        tamper_native_polynomial(
            executable, fixture, root / "polynomial_offset", "offset"
        )
        tamper_physical_reality_hash(
            executable, fixture, root / "physical_reality_hash"
        )


if __name__ == "__main__":
    main()
