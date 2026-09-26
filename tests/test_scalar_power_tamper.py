#!/usr/bin/env python3
"""Require native rejection of a fully rehashed scalar-certificate attack."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def canonical_bytes(payload):
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def stable_hash(payload):
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    Path(path).write_bytes(canonical_bytes(payload))


def alternative_identity(alternative):
    semantic = deepcopy(alternative)
    semantic.pop("alternative_id", None)
    semantic.pop("availability", None)
    return stable_hash(semantic)


def lammps_command(args, manifest, output_root, label):
    return [
        str(Path(args.lmp).resolve()),
        "-in",
        str(Path(args.input).resolve()),
        "-var",
        "model",
        str(Path(args.model).resolve()),
        "-var",
        "plan",
        str(Path(manifest).resolve()),
        "-var",
        "policy",
        "scalar_power",
        "-var",
        "ye3t_chunksize",
        "17",
        "-var",
        "dump_path",
        str(output_root / (label + ".dump")),
        "-log",
        str(output_root / (label + ".log")),
        "-screen",
        "none",
    ]


def run_lammps(args, manifest, output_root, label):
    return subprocess.run(
        lammps_command(args, manifest, output_root, label),
        cwd=output_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmp", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="ye3t-scalar-tamper-") as temporary:
        root = Path(temporary)
        baseline = run_lammps(args, args.manifest, root, "baseline")
        if baseline.returncode != 0:
            raise AssertionError(
                "baseline scalar bundle failed before tampering:\n"
                + baseline.stdout
                + baseline.stderr
            )

        source_manifest = Path(args.manifest).resolve()
        manifest = load_json(source_manifest)
        hostile_root = root / "hostile"
        hostile_root.mkdir()
        for record in manifest["payloads"].values():
            source = source_manifest.parent / record["path"]
            destination = hostile_root / record["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        hostile_manifest = hostile_root / "manifest.json"

        plan_path = hostile_root / manifest["payloads"]["execution_plan"]["path"]
        mapping_path = hostile_root / manifest["payloads"]["yace_function_map"]["path"]
        plan = load_json(plan_path)
        mapping = load_json(mapping_path)

        scalar_alternatives = [
            alternative
            for entry in mapping["entries"]
            for alternative in entry["alternatives"]
            if alternative["evaluator"] == "scalar_invariant_power"
        ]
        if not scalar_alternatives:
            raise AssertionError("scalar tamper fixture has no scalar alternative")
        old_factorization = scalar_alternatives[0]["factorization_id"]
        certificates = [
            certificate
            for instruction in plan["instructions"]
            for certificate in instruction["metadata"].get(
                "fast_route_certificates", []
            )
            if certificate["certificate_sha256"] == old_factorization
        ]
        if len(certificates) != 1:
            raise AssertionError("scalar factorization certificate is not unique")
        certificate = certificates[0]
        coefficient = certificate["quadratic_terms"][0]["coefficient"]
        coefficient[0] = float(coefficient[0]) + 0.125
        certificate["certificate_sha256"] = stable_hash(
            {
                key: value
                for key, value in certificate.items()
                if key != "certificate_sha256"
            }
        )
        new_factorization = certificate["certificate_sha256"]

        plan_without_hash = {
            key: value for key, value in plan.items() if key != "plan_hash"
        }
        plan["plan_hash"] = stable_hash(plan_without_hash)
        old_plan_hash = mapping["plan_hash"]
        mapping["plan_hash"] = plan["plan_hash"]
        for entry in mapping["entries"]:
            for alternative in entry["alternatives"]:
                if alternative.get("compiler_plan_hash") == old_plan_hash:
                    alternative["compiler_plan_hash"] = plan["plan_hash"]
                if alternative.get("factorization_id") == old_factorization:
                    alternative["factorization_id"] = new_factorization
                alternative["alternative_id"] = alternative_identity(alternative)

        write_json(plan_path, plan)
        write_json(mapping_path, mapping)
        manifest["compiler"]["plan_hash"] = plan["plan_hash"]
        manifest["payloads"]["execution_plan"]["sha256"] = sha256_file(plan_path)
        manifest["payloads"]["yace_function_map"]["sha256"] = sha256_file(
            mapping_path
        )
        write_json(hostile_manifest, manifest)

        hostile = run_lammps(args, hostile_manifest, root, "hostile")
        hostile_log = root / "hostile.log"
        log_output = (
            hostile_log.read_text(encoding="utf-8", errors="replace")
            if hostile_log.exists()
            else ""
        )
        output = hostile.stdout + hostile.stderr + log_output
        if hostile.returncode == 0:
            raise AssertionError(
                "native loader accepted a rehashed scalar coefficient attack"
            )
        if "scalar-invariant-power coefficient identity failed" not in output:
            raise AssertionError("native loader rejected the wrong condition:\n" + output)

    print("scalar-power rehashed coefficient tamper validation passed")


if __name__ == "__main__":
    main()
