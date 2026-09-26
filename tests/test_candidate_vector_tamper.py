#!/usr/bin/env python3
"""Exercise fail-closed v3 candidate-sidecar validation."""

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
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


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    Path(path).write_bytes(canonical_bytes(payload))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_validator(path):
    path = Path(path).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("ye3t_sidecar_v3_validator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the strict v3 validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_bundle(source_manifest, destination):
    source_manifest = Path(source_manifest).resolve()
    manifest = load_json(source_manifest)
    destination.mkdir()
    target_manifest = destination / "manifest.json"
    for record in manifest["payloads"].values():
        relative = Path(record["path"])
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_manifest.parent / relative, target)
    write_json(target_manifest, manifest)
    return target_manifest


def payload_file(manifest_path, name):
    manifest = load_json(manifest_path)
    return manifest_path.parent / manifest["payloads"][name]["path"]


def refresh_payload_hash(manifest_path, name):
    manifest = load_json(manifest_path)
    payload_path = manifest_path.parent / manifest["payloads"][name]["path"]
    manifest["payloads"][name]["sha256"] = sha256_file(payload_path)
    write_json(manifest_path, manifest)


def refresh_plan_hashes(validator, manifest_path, plan, mapping):
    plan["plan_hash"] = ""
    plan = validator.YE3TExecutionPlan.from_dict(plan).to_dict()
    plan_hash = plan["plan_hash"]
    mapping["plan_hash"] = plan_hash
    for entry in mapping["entries"]:
        for alternative in entry["alternatives"]:
            if alternative["evaluator"] != "execution_plan_readout":
                continue
            alternative["compiler_plan_hash"] = plan_hash
            semantic_candidate = {
                key: value
                for key, value in alternative.items()
                if key != "alternative_id"
            }
            alternative["alternative_id"] = validator.candidate_identity(
                semantic_candidate
            )

    plan_path = payload_file(manifest_path, "execution_plan")
    mapping_path = payload_file(manifest_path, "yace_function_map")
    manifest = load_json(manifest_path)
    manifest["compiler"]["plan_hash"] = plan_hash
    write_json(plan_path, plan)
    write_json(mapping_path, mapping)
    write_json(manifest_path, manifest)
    refresh_payload_hash(manifest_path, "execution_plan")
    refresh_payload_hash(manifest_path, "yace_function_map")


def native_command(args, manifest):
    return [
        str(Path(args.evaluator).resolve()),
        str(Path(args.model).resolve()),
        str(Path(args.environment).resolve()),
        str(Path(manifest).resolve()),
        str(args.expected_routes),
        str(args.expected_plans),
        args.mode,
        str(args.expected_candidates),
    ]


def run_native(args, manifest):
    return subprocess.run(
        native_command(args, manifest),
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def require_native_success(args, manifest):
    result = run_native(args, manifest)
    if result.returncode != 0:
        raise AssertionError(
            "baseline native candidate fixture failed:\n"
            + result.stdout
            + result.stderr
        )


def require_rejected(validator, args, manifest, python_fragment, native_fragment):
    try:
        validator.validate_bundle(manifest, args.model)
    except Exception as error:
        if python_fragment not in str(error):
            raise AssertionError(
                "Python validator rejected the wrong condition: " + str(error)
            ) from error
    else:
        raise AssertionError("Python validator accepted a tampered sidecar")

    result = run_native(args, manifest)
    output = result.stdout + result.stderr
    if result.returncode == 0:
        raise AssertionError("native loader accepted a tampered sidecar")
    if native_fragment not in output:
        raise AssertionError("native loader rejected the wrong condition: " + output)


def first_readout_alternative(mapping):
    for entry in mapping["entries"]:
        for alternative in entry["alternatives"]:
            if alternative["evaluator"] == "execution_plan_readout":
                return alternative
    raise AssertionError("fixture has no execution-plan readout candidate")


def first_direct_alternative(mapping):
    for entry in mapping["entries"]:
        for alternative in entry["alternatives"]:
            if alternative["evaluator"] == "explicit_ctilde":
                return alternative
    raise AssertionError("fixture has no direct candidate")


def replace_readout_identity(validator, record, mapping):
    old_readout_id = record["readout_id"]
    semantic_record = {
        key: value for key, value in record.items() if key != "readout_id"
    }
    record["readout_id"] = validator.readout_identity(semantic_record)
    matches = 0
    for entry in mapping["entries"]:
        for alternative in entry["alternatives"]:
            if alternative.get("readout_id") == old_readout_id:
                alternative["readout_id"] = record["readout_id"]
                matches += 1
    if matches != 1:
        raise AssertionError("readout certificate is not referenced exactly once")


def first_mixed_readout(plan):
    return next(
        record
        for record in plan["certificate"]["yace_candidate_readouts"]["records"]
        if record["equivalence"]["method"]
        == "coefficientwise_sparse_polynomial_mixed_v1"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--validator", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-routes", required=True, type=int)
    parser.add_argument("--expected-plans", required=True, type=int)
    parser.add_argument("--expected-candidates", required=True, type=int)
    parser.add_argument(
        "--mode",
        choices=("candidate_vector", "gpu_auto_candidate_vector"),
        default="candidate_vector",
    )
    args = parser.parse_args()
    validator = load_validator(args.validator)

    baseline = validator.validate_bundle(args.manifest, args.model)
    if baseline["readout_count"] == 0:
        raise AssertionError("baseline fixture has no certified candidate readout")
    if baseline["binding_occurrence_count"] <= baseline["binding_count"]:
        raise AssertionError("baseline fixture has no shared candidate binding")
    require_native_success(args, args.manifest)

    with tempfile.TemporaryDirectory(prefix="ye3t-candidate-tamper-") as temporary:
        root = Path(temporary)

        mixed_manifest = copy_bundle(args.manifest, root / "mixed_certificate")
        plan_path = payload_file(mixed_manifest, "execution_plan")
        mapping_path = payload_file(mixed_manifest, "yace_function_map")
        plan = load_json(plan_path)
        mapping = load_json(mapping_path)
        record = plan["certificate"]["yace_candidate_readouts"]["records"][0]
        equivalence = record["equivalence"]
        equivalence.update(
            {
                "absolute_tolerance": 5.0e-11,
                "maximum_mixed_tolerance_ratio": 0.0,
                "maximum_reference_coefficient_magnitude": 1.0,
                "method": "coefficientwise_sparse_polynomial_mixed_v1",
                "relative_tolerance": 1.0e-12,
                "tolerance_rule": "absolute_plus_relative_reference",
            }
        )
        equivalence["maximum_mixed_tolerance_ratio"] = equivalence[
            "maximum_absolute_coefficient_residual"
        ] / (
            equivalence["absolute_tolerance"]
            + equivalence["relative_tolerance"]
            * equivalence["maximum_reference_coefficient_magnitude"]
        )
        replace_readout_identity(validator, record, mapping)
        refresh_plan_hashes(validator, mixed_manifest, plan, mapping)
        validator.validate_bundle(mixed_manifest, args.model)
        require_native_success(args, mixed_manifest)

        manifest = copy_bundle(mixed_manifest, root / "mixed_ratio")
        plan_path = payload_file(manifest, "execution_plan")
        mapping_path = payload_file(manifest, "yace_function_map")
        plan = load_json(plan_path)
        mapping = load_json(mapping_path)
        record = first_mixed_readout(plan)
        record["equivalence"]["maximum_mixed_tolerance_ratio"] = 1.0001
        replace_readout_identity(validator, record, mapping)
        refresh_plan_hashes(validator, manifest, plan, mapping)
        require_rejected(
            validator,
            args,
            manifest,
            "residual exceeds its declared tolerance",
            "mixed coefficient certificate failed",
        )

        manifest = copy_bundle(mixed_manifest, root / "mixed_summary")
        plan_path = payload_file(manifest, "execution_plan")
        mapping_path = payload_file(manifest, "yace_function_map")
        plan = load_json(plan_path)
        mapping = load_json(mapping_path)
        record = first_mixed_readout(plan)
        equivalence = record["equivalence"]
        equivalence["maximum_absolute_coefficient_residual"] = 2.0 * (
            equivalence["absolute_tolerance"]
            + equivalence["relative_tolerance"]
            * equivalence["maximum_reference_coefficient_magnitude"]
        )
        replace_readout_identity(validator, record, mapping)
        refresh_plan_hashes(validator, manifest, plan, mapping)
        require_rejected(
            validator,
            args,
            manifest,
            "residual exceeds its declared tolerance",
            "mixed coefficient certificate failed",
        )

        manifest = copy_bundle(mixed_manifest, root / "mixed_rule")
        plan_path = payload_file(manifest, "execution_plan")
        mapping_path = payload_file(manifest, "yace_function_map")
        plan = load_json(plan_path)
        mapping = load_json(mapping_path)
        record = first_mixed_readout(plan)
        record["equivalence"]["tolerance_rule"] = "unsupported_rule"
        replace_readout_identity(validator, record, mapping)
        refresh_plan_hashes(validator, manifest, plan, mapping)
        require_rejected(
            validator,
            args,
            manifest,
            "residual exceeds its declared tolerance",
            "mixed coefficient certificate failed",
        )

        manifest = copy_bundle(args.manifest, root / "candidate_id")
        mapping_path = payload_file(manifest, "yace_function_map")
        mapping = load_json(mapping_path)
        mapping["entries"][0]["alternatives"][0]["alternative_id"] = "0" * 64
        write_json(mapping_path, mapping)
        refresh_payload_hash(manifest, "yace_function_map")
        require_rejected(
            validator,
            args,
            manifest,
            "candidate ID is nonsemantic or duplicate",
            "direct candidate is not bound to this YACE function",
        )

        manifest = copy_bundle(args.manifest, root / "missing_readout")
        plan_path = payload_file(manifest, "execution_plan")
        mapping_path = payload_file(manifest, "yace_function_map")
        plan = load_json(plan_path)
        mapping = load_json(mapping_path)
        alternative = first_readout_alternative(mapping)
        missing_id = alternative["readout_id"]
        records = plan["certificate"]["yace_candidate_readouts"]["records"]
        plan["certificate"]["yace_candidate_readouts"]["records"] = [
            record for record in records if record["readout_id"] != missing_id
        ]
        write_json(plan_path, plan)
        refresh_payload_hash(manifest, "execution_plan")
        require_rejected(
            validator,
            args,
            manifest,
            "execution-plan hash does not match its payload",
            "native reconstruction does not match the plan payload",
        )

        manifest = copy_bundle(args.manifest, root / "binding_scale")
        plan_path = payload_file(manifest, "execution_plan")
        mapping_path = payload_file(manifest, "yace_function_map")
        plan = load_json(plan_path)
        mapping = load_json(mapping_path)
        record = plan["certificate"]["yace_candidate_readouts"]["records"][0]
        old_readout_id = record["readout_id"]
        record["terms"][0]["scale"][0] += 0.125
        semantic_record = {
            key: value for key, value in record.items() if key != "readout_id"
        }
        record["readout_id"] = validator.readout_identity(semantic_record)
        alternative = first_readout_alternative(mapping)
        if alternative["readout_id"] != old_readout_id:
            raise AssertionError("first candidate and first readout are not paired")
        alternative["readout_id"] = record["readout_id"]
        semantic_candidate = {
            key: value for key, value in alternative.items() if key != "alternative_id"
        }
        alternative["alternative_id"] = validator.candidate_identity(
            semantic_candidate
        )
        refresh_plan_hashes(validator, manifest, plan, mapping)
        require_rejected(
            validator,
            args,
            manifest,
            "candidate binding ID does not hash its semantics",
            "candidate readout binding ID is invalid or duplicate",
        )

        manifest = copy_bundle(args.manifest, root / "synthesis_coefficient")
        plan_path = payload_file(manifest, "execution_plan")
        plan = load_json(plan_path)
        plan["synthesis_tables"][0]["values"][0][0] += 0.125
        write_json(plan_path, plan)
        refresh_payload_hash(manifest, "execution_plan")
        require_rejected(
            validator,
            args,
            manifest,
            "synthesis-table coefficient hash does not match its payload",
            "native reconstruction does not match the synthesis tables",
        )

        manifest = copy_bundle(args.manifest, root / "stale_plan_hash")
        plan_path = payload_file(manifest, "execution_plan")
        plan = load_json(plan_path)
        table = plan["synthesis_tables"][0]
        table["values"][0][0] += 0.125
        table["coefficient_hash"] = validator.stable_hash(
            {
                "column_indices": table["column_indices"],
                "convention_id": table["convention_id"],
                "input_dimension": table["input_dimension"],
                "orientation": table["orientation"],
                "output_dimension": table["output_dimension"],
                "row_indices": table["row_indices"],
                "values": table["values"],
            }
        )
        plan["coefficient_hash"] = validator.stable_hash(
            plan["synthesis_tables"]
        )
        write_json(plan_path, plan)
        refresh_payload_hash(manifest, "execution_plan")
        require_rejected(
            validator,
            args,
            manifest,
            "execution-plan hash does not match its payload",
            "native reconstruction does not match the plan payload",
        )

        manifest = copy_bundle(args.manifest, root / "readout_reference")
        mapping_path = payload_file(manifest, "yace_function_map")
        mapping = load_json(mapping_path)
        alternative = first_readout_alternative(mapping)
        alternative["readout_id"] = "f" * 64
        semantic_candidate = {
            key: value for key, value in alternative.items() if key != "alternative_id"
        }
        alternative["alternative_id"] = validator.candidate_identity(
            semantic_candidate
        )
        write_json(mapping_path, mapping)
        refresh_payload_hash(manifest, "yace_function_map")
        require_rejected(
            validator,
            args,
            manifest,
            "candidate readout is absent or reused",
            "missing identifier",
        )

        manifest = copy_bundle(args.manifest, root / "duplicate_binding")
        plan_path = payload_file(manifest, "execution_plan")
        mapping_path = payload_file(manifest, "yace_function_map")
        plan = load_json(plan_path)
        mapping = load_json(mapping_path)
        record = next(
            item
            for item in plan["certificate"]["yace_candidate_readouts"]["records"]
            if len(item["terms"]) > 1
        )
        old_readout_id = record["readout_id"]
        record["terms"].append(deepcopy(record["terms"][0]))
        semantic_record = {
            key: value for key, value in record.items() if key != "readout_id"
        }
        record["readout_id"] = validator.readout_identity(semantic_record)
        for entry in mapping["entries"]:
            for alternative in entry["alternatives"]:
                if alternative.get("readout_id") == old_readout_id:
                    alternative["readout_id"] = record["readout_id"]
        refresh_plan_hashes(validator, manifest, plan, mapping)
        require_rejected(
            validator,
            args,
            manifest,
            "duplicate candidate binding ID within one readout",
            "candidate readout binding ID is invalid or duplicate",
        )

        manifest = copy_bundle(args.manifest, root / "candidate_bound")
        mapping_path = payload_file(manifest, "yace_function_map")
        mapping = load_json(mapping_path)
        candidate = mapping["entries"][0]["alternatives"][0]
        mapping["entries"][0]["alternatives"] = [
            deepcopy(candidate) for _ in range(65)
        ]
        write_json(mapping_path, mapping)
        refresh_payload_hash(manifest, "yace_function_map")
        require_rejected(
            validator,
            args,
            manifest,
            "candidate count is outside the supported bound",
            "v3 candidate count is outside the supported bound",
        )

        manifest = copy_bundle(args.manifest, root / "candidate_capabilities")
        mapping_path = payload_file(manifest, "yace_function_map")
        mapping = load_json(mapping_path)
        alternative = first_direct_alternative(mapping)
        alternative["required_capabilities"].append(
            "block_symmetric_power_forward_v1"
        )
        alternative["required_capabilities"].sort()
        semantic_candidate = {
            key: value for key, value in alternative.items() if key != "alternative_id"
        }
        alternative["alternative_id"] = validator.candidate_identity(
            semantic_candidate
        )
        write_json(mapping_path, mapping)
        refresh_payload_hash(manifest, "yace_function_map")
        require_rejected(
            validator,
            args,
            manifest,
            "direct candidate capability set changed",
            "each v3 row requires one identity direct candidate",
        )

        manifest = copy_bundle(args.manifest, root / "availability_reason")
        mapping_path = payload_file(manifest, "yace_function_map")
        mapping = load_json(mapping_path)
        alternative = first_direct_alternative(mapping)
        alternative["availability"]["reason"] = "unexpected"
        semantic_candidate = {
            key: value for key, value in alternative.items() if key != "alternative_id"
        }
        alternative["alternative_id"] = validator.candidate_identity(
            semantic_candidate
        )
        write_json(mapping_path, mapping)
        refresh_payload_hash(manifest, "yace_function_map")
        require_rejected(
            validator,
            args,
            manifest,
            "fixture contains an unavailable candidate",
            "available candidates with an empty reason only",
        )

        manifest = copy_bundle(args.manifest, root / "unknown_capability")
        payload = load_json(manifest)
        payload["capabilities"]["required"].append(
            "unsupported_candidate_backend_v1"
        )
        write_json(manifest, payload)
        require_rejected(
            validator,
            args,
            manifest,
            "manifest capabilities are unsupported, incomplete, or duplicated",
            "unsupported capability",
        )

    print("candidate-vector tamper validation passed")


if __name__ == "__main__":
    main()
