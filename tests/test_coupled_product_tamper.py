#!/usr/bin/env python3
"""Reject a rehashed coupled-product plan whose exact image is false."""

import argparse
from pathlib import Path
import subprocess
import tempfile

from test_candidate_vector_tamper import copy_bundle
from test_candidate_vector_tamper import load_json
from test_candidate_vector_tamper import load_validator
from test_candidate_vector_tamper import payload_file
from test_candidate_vector_tamper import refresh_payload_hash
from test_candidate_vector_tamper import replace_readout_identity
from test_candidate_vector_tamper import write_json


def native_command(args, manifest):
    return [
        str(Path(args.evaluator).resolve()),
        str(Path(args.model).resolve()),
        str(Path(args.environment).resolve()),
        str(Path(manifest).resolve()),
        "1",
        "0",
        "coupled_product",
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
            "baseline native coupled-product fixture failed:\n"
            + result.stdout
            + result.stderr
        )


def rehash_false_exact_solution(validator, manifest_path):
    plan_path = payload_file(manifest_path, "execution_plan")
    mapping_path = payload_file(manifest_path, "yace_function_map")
    plan = load_json(plan_path)
    mapping = load_json(mapping_path)

    metadata = plan["instructions"][0]["metadata"]
    certificate = metadata["exact_image_certificate"]
    term = certificate["readout_solution"]["entries"][0]["value"]["real"][
        "terms"
    ][0]
    term["coefficient"]["numerator"] = 2
    certificate["certificate_sha256"] = validator.stable_hash(
        {
            key: value
            for key, value in certificate.items()
            if key != "certificate_sha256"
        }
    )
    metadata["semantic_sha256"] = validator.stable_hash(
        {
            key: value
            for key, value in metadata.items()
            if key != "semantic_sha256"
        }
    )
    plan["certificate"]["ace_coupled_product_dag"]["semantic_sha256"] = (
        metadata["semantic_sha256"]
    )
    plan["plan_hash"] = validator.stable_hash(
        {key: value for key, value in plan.items() if key != "plan_hash"}
    )

    mapping["plan_hash"] = plan["plan_hash"]
    readout_candidates = []
    for entry in mapping["entries"]:
        for alternative in entry["alternatives"]:
            if alternative["evaluator"] == "execution_plan_readout":
                alternative["compiler_plan_hash"] = plan["plan_hash"]
                alternative["alternative_id"] = validator.candidate_identity(
                    {
                        key: value
                        for key, value in alternative.items()
                        if key != "alternative_id"
                    }
                )
                readout_candidates.append(alternative)
    if len(readout_candidates) != 1:
        raise AssertionError("fixture does not contain one compiler candidate")

    manifest = load_json(manifest_path)
    manifest["compiler"]["plan_hash"] = plan["plan_hash"]
    write_json(plan_path, plan)
    write_json(mapping_path, mapping)
    write_json(manifest_path, manifest)
    refresh_payload_hash(manifest_path, "execution_plan")
    refresh_payload_hash(manifest_path, "yace_function_map")


def rehash_mutated_cg_table(validator, manifest_path):
    plan_path = payload_file(manifest_path, "execution_plan")
    mapping_path = payload_file(manifest_path, "yace_function_map")
    plan = load_json(plan_path)
    mapping = load_json(mapping_path)

    table = plan["synthesis_tables"][0]
    table["values"][0][0] += 0.125
    table["coefficient_hash"] = validator.stable_hash(
        {
            "input_dimension": table["input_dimension"],
            "output_dimension": table["output_dimension"],
            "row_indices": table["row_indices"],
            "column_indices": table["column_indices"],
            "values": table["values"],
            "orientation": table["orientation"],
            "convention_id": table["convention_id"],
        }
    )
    plan["coefficient_hash"] = validator.stable_hash(plan["synthesis_tables"])

    readout = plan["certificate"]["yace_candidate_readouts"]["records"][0]
    readout["equivalence"]["plan_polynomial_sha256"] = validator.stable_hash(
        {
            "mutated_table": table["coefficient_hash"],
            "claim": "self_consistently_rehashed_hostile_fixture",
        }
    )
    replace_readout_identity(validator, readout, mapping)

    plan["plan_hash"] = validator.stable_hash(
        {key: value for key, value in plan.items() if key != "plan_hash"}
    )
    mapping["plan_hash"] = plan["plan_hash"]
    readout_candidates = []
    for entry in mapping["entries"]:
        for alternative in entry["alternatives"]:
            if alternative["evaluator"] != "execution_plan_readout":
                continue
            alternative["compiler_plan_hash"] = plan["plan_hash"]
            alternative["alternative_id"] = validator.candidate_identity(
                {
                    key: value
                    for key, value in alternative.items()
                    if key != "alternative_id"
                }
            )
            readout_candidates.append(alternative)
    if len(readout_candidates) != 1:
        raise AssertionError("fixture does not contain one compiler candidate")

    manifest = load_json(manifest_path)
    manifest["compiler"]["plan_hash"] = plan["plan_hash"]
    manifest["compiler"]["coefficient_hash"] = plan["coefficient_hash"]
    write_json(plan_path, plan)
    write_json(mapping_path, mapping)
    write_json(manifest_path, manifest)
    refresh_payload_hash(manifest_path, "execution_plan")
    refresh_payload_hash(manifest_path, "yace_function_map")


def rehash_readout_overflow(validator, manifest_path):
    plan_path = payload_file(manifest_path, "execution_plan")
    mapping_path = payload_file(manifest_path, "yace_function_map")
    plan = load_json(plan_path)
    mapping = load_json(mapping_path)

    metadata = plan["instructions"][0]["metadata"]
    certificate = metadata["exact_image_certificate"]
    product_coefficient = certificate["product_matrix"]["entries"][0][
        "value"
    ]["real"]["terms"][0]["coefficient"]
    product_coefficient["denominator"] *= 2
    solution_coefficient = certificate["readout_solution"]["entries"][0][
        "value"
    ]["real"]["terms"][0]["coefficient"]
    solution_coefficient["numerator"] = 2
    output = metadata["outputs"][0]["terms"][0]
    output["coefficient_exact"]["real"]["terms"][0]["coefficient"][
        "numerator"
    ] = 2
    output["coefficient_binary64"] = [2.0, 0.0]
    certificate["certificate_sha256"] = validator.stable_hash(
        {
            key: value
            for key, value in certificate.items()
            if key != "certificate_sha256"
        }
    )

    readout = plan["certificate"]["yace_candidate_readouts"]["records"][0]
    term = readout["terms"][0]
    term["scale"] = [1.0e308, 0.0]
    term["binding_id"] = validator.binding_identity(
        {key: value for key, value in term.items() if key != "binding_id"}
    )
    readout["equivalence"]["plan_polynomial_sha256"] = validator.stable_hash(
        {
            "claim": "self_consistently_rehashed_readout_overflow",
            "scale": term["scale"],
        }
    )
    replace_readout_identity(validator, readout, mapping)

    metadata["semantic_sha256"] = validator.stable_hash(
        {
            key: value
            for key, value in metadata.items()
            if key != "semantic_sha256"
        }
    )
    plan["certificate"]["ace_coupled_product_dag"]["semantic_sha256"] = (
        metadata["semantic_sha256"]
    )
    plan["plan_hash"] = validator.stable_hash(
        {key: value for key, value in plan.items() if key != "plan_hash"}
    )
    mapping["plan_hash"] = plan["plan_hash"]
    readout_candidates = []
    for entry in mapping["entries"]:
        for alternative in entry["alternatives"]:
            if alternative["evaluator"] != "execution_plan_readout":
                continue
            alternative["compiler_plan_hash"] = plan["plan_hash"]
            alternative["alternative_id"] = validator.candidate_identity(
                {
                    key: value
                    for key, value in alternative.items()
                    if key != "alternative_id"
                }
            )
            readout_candidates.append(alternative)
    if len(readout_candidates) != 1:
        raise AssertionError("fixture does not contain one compiler candidate")

    manifest = load_json(manifest_path)
    manifest["compiler"]["plan_hash"] = plan["plan_hash"]
    write_json(plan_path, plan)
    write_json(mapping_path, mapping)
    write_json(manifest_path, manifest)
    refresh_payload_hash(manifest_path, "execution_plan")
    refresh_payload_hash(manifest_path, "yace_function_map")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--validator", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    validator = load_validator(args.validator)

    validator.validate_bundle(args.manifest, args.model)
    require_native_success(args, args.manifest)
    with tempfile.TemporaryDirectory(prefix="ye3t-coupled-tamper-") as root:
        manifest = copy_bundle(args.manifest, Path(root) / "false_image")
        rehash_false_exact_solution(validator, manifest)
        try:
            validator.validate_bundle(manifest, args.model)
        except Exception as error:
            if "ACE exact readout does not satisfy P r = c" not in str(error):
                raise AssertionError(
                    "Python validator rejected the wrong condition: " + str(error)
                ) from error
        else:
            raise AssertionError("Python validator accepted a false exact image")

        result = run_native(args, manifest)
        output = result.stdout + result.stderr
        if result.returncode == 0:
            raise AssertionError("native loader accepted a false exact image")
        if "native exact-image validation failed P r = c" not in output:
            raise AssertionError(
                "native loader rejected the wrong condition: " + output
            )

    with tempfile.TemporaryDirectory(prefix="ye3t-coupled-cg-tamper-") as root:
        manifest = copy_bundle(args.manifest, Path(root) / "false_cg")
        rehash_mutated_cg_table(validator, manifest)
        try:
            validator.validate_bundle(manifest, args.model)
        except Exception as error:
            if "ACE product CG coefficients are inconsistent" not in str(error):
                raise AssertionError(
                    "Python validator rejected the wrong CG condition: "
                    + str(error)
                ) from error
        else:
            raise AssertionError("Python validator accepted a mutated CG table")

        result = run_native(args, manifest)
        output = result.stdout + result.stderr
        if result.returncode == 0:
            raise AssertionError("native loader accepted a mutated CG table")
        if (
            "native coupled-product polynomial does not match direct C-tilde"
            not in output
        ):
            raise AssertionError(
                "native loader rejected the wrong CG condition: " + output
            )

    with tempfile.TemporaryDirectory(
        prefix="ye3t-coupled-overflow-tamper-"
    ) as root:
        manifest = copy_bundle(args.manifest, Path(root) / "readout_overflow")
        rehash_readout_overflow(validator, manifest)
        result = run_native(args, manifest)
        output = result.stdout + result.stderr
        if result.returncode == 0:
            raise AssertionError("native loader accepted a readout overflow")
        if "coupled-product scaled readout coefficient overflow" not in output:
            raise AssertionError(
                "native loader rejected the wrong overflow condition: " + output
            )

    print(
        "coupled-product exact-image, runtime-table, and overflow tamper "
        "validation passed"
    )


if __name__ == "__main__":
    main()
