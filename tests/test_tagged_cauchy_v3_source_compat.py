#!/usr/bin/env python3
"""Replay a V3 model through the native CPU legacy-V1 source realization."""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def payload_hash(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path, payload):
    path.write_text(
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


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: test_tagged_cauchy_v3_source_compat.py EXECUTABLE FIXTURE"
        )
    executable = Path(sys.argv[1]).resolve()
    fixture_path = Path(sys.argv[2]).resolve()
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    model_path = fixture_path.parent / fixture["model_path"]
    model = json.loads(model_path.read_text(encoding="utf-8"))

    source = model["source_binding"]["payload"]
    if source["schema"] != "ye3t_tagged_cauchy_direct_source_v2":
        raise AssertionError("compatibility input is not the stable V2 fixture")
    source["schema"] = "ye3t_tagged_cauchy_direct_source_v1"
    source.pop("numerical_evaluation")
    source_body = {
        key: value for key, value in source.items() if key != "source_plan_hash"
    }
    source["source_plan_hash"] = payload_hash(source_body)
    model["source_binding"]["hash"] = payload_hash(source)
    identity = {
        "compiler_artifact_hash": model["compiler_artifact_hash"],
        "conventions": model["conventions"],
        "readout_hash": model["readout_binding"]["hash"],
        "schedule_hash": model["schedule_binding"]["hash"],
        "source_plan_hash": model["source_binding"]["hash"],
    }
    model["deployment_identity_hash"] = payload_hash(identity)
    model_body = {key: value for key, value in model.items() if key != "self_hash"}
    model["self_hash"] = payload_hash(model_body)

    with tempfile.TemporaryDirectory(prefix="ye3t_tagged_v3_source_v1_") as tmp:
        root = Path(tmp)
        generated_model = root / "legacy_source_v1.model.json"
        generated_fixture = root / "legacy_source_v1.fixture.json"
        write_json(generated_model, model)
        fixture["model_path"] = generated_model.name
        fixture["model_self_hash"] = model["self_hash"]
        write_json(generated_fixture, fixture)
        result = subprocess.run(
            (str(executable), str(generated_fixture)),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise AssertionError(
                "native CPU legacy-V1 source replay failed:\n"
                + result.stdout
                + result.stderr
            )


if __name__ == "__main__":
    main()
