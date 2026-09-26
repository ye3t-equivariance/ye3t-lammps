#!/usr/bin/env python3
"""Require native rejection of coherently root-rehashed V3 tampering."""

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


def run_tamper(executable, fixture_payload, model_payload, root, name, expected):
    model_body = {
        key: value for key, value in model_payload.items() if key != "self_hash"
    }
    model_payload["self_hash"] = payload_hash(model_body)
    model_path = root / f"{name}.model.json"
    fixture_path = root / f"{name}.fixture.json"
    write_json(model_path, model_payload)
    rewritten_fixture = dict(fixture_payload)
    rewritten_fixture["model_path"] = model_path.name
    rewritten_fixture["model_self_hash"] = model_payload["self_hash"]
    write_json(fixture_path, rewritten_fixture)
    result = subprocess.run(
        (str(executable), str(fixture_path)),
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = result.stdout + result.stderr
    if result.returncode == 0:
        raise AssertionError(f"native loader accepted {name} tampering")
    if expected not in output:
        raise AssertionError(
            f"native loader rejected {name} for the wrong reason:\n{output}"
        )


def rehash_binding(model_payload, name, self_hash_name):
    nested = model_payload[name]["payload"]
    nested_body = {
        key: value for key, value in nested.items() if key != self_hash_name
    }
    nested[self_hash_name] = payload_hash(nested_body)
    model_payload[name]["hash"] = payload_hash(nested)
    identity = {
        "compiler_artifact_hash": model_payload["compiler_artifact_hash"],
        "conventions": model_payload["conventions"],
        "readout_hash": model_payload["readout_binding"]["hash"],
        "schedule_hash": model_payload["schedule_binding"]["hash"],
        "source_plan_hash": model_payload["source_binding"]["hash"],
    }
    model_payload["deployment_identity_hash"] = payload_hash(identity)


def rehash_root(model_payload):
    identity = {
        "compiler_artifact_hash": model_payload["compiler_artifact_hash"],
        "conventions": model_payload["conventions"],
        "readout_hash": model_payload["readout_binding"]["hash"],
        "schedule_hash": model_payload["schedule_binding"]["hash"],
        "source_plan_hash": model_payload["source_binding"]["hash"],
    }
    model_payload["deployment_identity_hash"] = payload_hash(identity)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: test_tagged_cauchy_v3_tamper.py EXECUTABLE FIXTURE")
    executable = Path(sys.argv[1]).resolve()
    fixture_path = Path(sys.argv[2]).resolve()
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    model_path = fixture_path.parent / fixture["model_path"]
    original = json.loads(model_path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="ye3t_tagged_v3_tamper_") as temporary:
        root = Path(temporary)
        source = json.loads(json.dumps(original))
        source["source_binding"]["payload"]["cutoff"] += 0.125
        run_tamper(
            executable, fixture, source, root, "source", "payload hash mismatch"
        )

        identity = json.loads(json.dumps(original))
        identity["deployment_identity_hash"] = "0" * 64
        run_tamper(
            executable,
            fixture,
            identity,
            root,
            "deployment_identity",
            "deployment identity mismatch",
        )

        schedule_channel = json.loads(json.dumps(original))
        schedule_channel["schedule_binding"]["payload"]["channels"][0][
            "q"
        ] += 1
        rehash_binding(schedule_channel, "schedule_binding", "program_hash")
        run_tamper(
            executable,
            fixture,
            schedule_channel,
            root,
            "schedule_channel",
            "schedule differs from its compiler commitment",
        )

        forward = json.loads(json.dumps(original))
        forward["schedule_binding"]["payload"]["terms"][0][
            "coefficient"
        ] += 0.125
        rehash_binding(forward, "schedule_binding", "program_hash")
        run_tamper(
            executable,
            fixture,
            forward,
            root,
            "forward_adjoint",
            "schedule differs from its compiler commitment",
        )

        coherent = json.loads(json.dumps(original))
        for term in coherent["schedule_binding"]["payload"]["terms"]:
            term["coefficient"] *= 2.0
        for term in coherent["schedule_binding"]["payload"]["adjoint_terms"]:
            term["coefficient"] *= 2.0
        rehash_binding(coherent, "schedule_binding", "program_hash")
        run_tamper(
            executable,
            fixture,
            coherent,
            root,
            "coherent_forward_adjoint",
            "schedule differs from its compiler commitment",
        )

        radial = json.loads(json.dumps(original))
        radial["source_binding"]["payload"]["channels"][0][
            "binary64_power_coefficients"
        ][0] += 0.125
        rehash_binding(radial, "source_binding", "source_plan_hash")
        run_tamper(
            executable,
            fixture,
            radial,
            root,
            "binary64_radial",
            "binary64 radial coefficients differ from exact payload",
        )

        source_semantics = json.loads(json.dumps(original))
        source_semantics["source_binding"]["payload"]["radial_coordinate"] = "r"
        rehash_binding(
            source_semantics, "source_binding", "source_plan_hash"
        )
        run_tamper(
            executable,
            fixture,
            source_semantics,
            root,
            "source_semantics",
            "unsupported radial coordinate, measure, or envelope",
        )

        source_certificate = json.loads(json.dumps(original))
        source_certificate["source_binding"]["payload"]["certificate"][
            "runtime_gram_solve"
        ] = True
        rehash_binding(
            source_certificate, "source_binding", "source_plan_hash"
        )
        run_tamper(
            executable,
            fixture,
            source_certificate,
            root,
            "source_certificate",
            "V3 source plan is uncertified",
        )

        conventions = json.loads(json.dumps(original))
        conventions["conventions"]["lammps_virial"] = "plus_strain_derivative"
        rehash_root(conventions)
        run_tamper(
            executable,
            fixture,
            conventions,
            root,
            "conventions",
            "unsupported native V3 convention",
        )


if __name__ == "__main__":
    main()
