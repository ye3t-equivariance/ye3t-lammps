#!/usr/bin/env python3
"""Freeze target-device PairYE3T/Kokkos calibration as a strict AUTO replay."""

import argparse
import hashlib
import json
from pathlib import Path


def canonical_bytes(payload, pretty=False):
    options = {
        "allow_nan": False,
        "ensure_ascii": True,
        "sort_keys": True,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(payload, **options) + ("\n" if pretty else "")).encode(
        "ascii"
    )


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(name + " is not a lowercase SHA-256 digest")
    return value


def payload_path(manifest_path, manifest, name):
    record = manifest["payloads"][name]
    return Path(manifest_path).resolve().parent / record["path"]


def validate_benchmark_authorization(path, result):
    if (
        result.get("configuration", {}).get("mode") != "performance"
        or result.get("failed_records") != 0
        or result.get("publication_timing_eligible") is not True
    ):
        raise ValueError(
            str(path) + " is not a complete performance-mode calibration"
        )
    certificate = result.get("parity_certificate")
    if not isinstance(certificate, dict) or certificate.get("status") != "passed":
        raise ValueError(
            str(path) + " lacks a passing same-run energy/force/virial certificate"
        )
    supplied_hash = require_digest(
        certificate.get("certificate_sha256"), "parity certificate"
    )
    unsigned = {
        key: value
        for key, value in certificate.items()
        if key != "certificate_sha256"
    }
    if sha256_bytes(canonical_bytes(unsigned)) != supplied_hash:
        raise ValueError(str(path) + " has a corrupted parity certificate")
    artifact_names = ("lmp", "model", "plan", "runner", "verifier")
    try:
        expected_artifacts = {
            name: require_digest(
                result["artifacts"][name]["sha256"], "benchmark " + name
            )
            for name in artifact_names
        }
    except (KeyError, TypeError) as error:
        raise ValueError(str(path) + " has incomplete artifact identities") from error
    if certificate.get("artifacts") != expected_artifacts:
        raise ValueError(str(path) + " parity certificate binds different artifacts")
    comparisons = certificate.get("comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError(str(path) + " parity certificate contains no comparisons")
    configured_engines = result.get("configuration", {}).get("engines")
    records = result.get("records")
    if (
        not isinstance(configured_engines, list)
        or len(configured_engines) < 2
        or len(set(configured_engines)) != len(configured_engines)
        or not isinstance(records, list)
        or not records
    ):
        raise ValueError(str(path) + " has an invalid evaluator record set")
    expected = {}
    for record in records:
        if record.get("status") != "passed":
            raise ValueError(str(path) + " contains a non-passing timing record")
        key = (record.get("ranks"), record.get("repetition"))
        engine = record.get("engine")
        digest = require_digest(
            record.get("parity_dump_sha256"), "timing parity dump"
        )
        if engine not in configured_engines or engine in expected.setdefault(key, {}):
            raise ValueError(str(path) + " has duplicate or unknown evaluator records")
        expected[key][engine] = digest
    configured_set = set(configured_engines)
    if any(set(engines) != configured_set for engines in expected.values()):
        raise ValueError(str(path) + " does not run every configured evaluator")
    covered = set()
    for comparison in comparisons:
        rows = comparison.get("comparison", {}).get("comparisons", [])
        if not rows or not all(row.get("passed") is True for row in rows):
            raise ValueError(str(path) + " parity certificate contains a failed row")
        key = (comparison.get("mpi_ranks"), comparison.get("repetition"))
        if key not in expected or key in covered:
            raise ValueError(str(path) + " parity certificate has an invalid run key")
        reference = comparison.get("reference")
        candidates = comparison.get("candidates")
        if (
            not isinstance(candidates, list)
            or reference in candidates
            or len(set(candidates)) != len(candidates)
            or {reference, *candidates} != configured_set
        ):
            raise ValueError(
                str(path) + " parity certificate does not cover every evaluator"
            )
        if comparison.get("dump_sha256") != expected[key]:
            raise ValueError(str(path) + " parity certificate binds different dumps")
        covered.add(key)
    if covered != set(expected):
        raise ValueError(str(path) + " parity certificate omits a timing run")


def calibration_candidates(path, rank, model_hash, manifest_hash):
    result = load_json(path)
    if result.get("schema") != "ye3t_lammps_gpu_scaling_v1":
        raise ValueError(str(path) + " has the wrong benchmark schema")
    validate_benchmark_authorization(path, result)
    if result["artifacts"]["model"]["sha256"] != model_hash:
        raise ValueError(str(path) + " used different model bytes")
    if result["artifacts"]["plan"]["sha256"] != manifest_hash:
        raise ValueError(str(path) + " used a different sidecar manifest")
    summary = result["summary"]["by_rank"].get(str(rank))
    if summary is None:
        raise ValueError(str(path) + " omits the requested calibration rank")
    engines = summary["engines"]
    if engines.get("direct", {}).get("status") != "passed":
        raise ValueError(str(path) + " must contain a passing direct reference")
    identities = result["fixed_ye3t_identities"]
    direct = identities["direct"]
    if direct.get("source_model") != model_hash:
        raise ValueError(str(path) + " dispatch marker used different model bytes")
    candidates = []
    for evaluator in ("block", "scalar_power", "coupled_product"):
        if engines.get(evaluator, {}).get("status") != "passed":
            continue
        candidate_identity = identities[evaluator]
        for field in (
            "device_class",
            "execution_space",
            "kokkos_layout",
            "source_policy",
            "VJP_policy",
        ):
            if direct.get(field) != candidate_identity.get(field):
                raise ValueError(
                    str(path) + " changed " + field + " across engines"
                )
        if candidate_identity.get("source_model") != model_hash:
            raise ValueError(str(path) + " dispatch marker used different model bytes")
        comparison = summary["paired_comparisons"].get(
            "direct_over_" + evaluator
        )
        if comparison is None or comparison["paired_repetitions"] < 8:
            raise ValueError(
                str(path) + " has fewer than eight paired " + evaluator + " repetitions"
            )
        center_counts = [
            int(capacity["local_inum"])
            for record in result["records"]
            if record["ranks"] == rank
            and record["engine"] in ("direct", evaluator)
            and record["status"] == "passed"
            for capacity in record["capacity"]
        ]
        if not center_counts:
            raise ValueError(str(path) + " has no calibrated local-center records")
        selected_block = evaluator == "block"
        candidates.append(
            {
                "benchmark_path": str(Path(path).resolve()),
                "benchmark_sha256": sha256_file(path),
                "candidate_median": engines[evaluator][
                    "loop_microseconds_per_atom_step"
                ]["median"],
                "direct_median": engines["direct"][
                    "loop_microseconds_per_atom_step"
                ]["median"],
                "evaluator": evaluator,
                "lower_speedup": comparison[
                    "bootstrap_95_percent_speedup_interval"
                ][0],
                "median_speedup": comparison["median_speedup_fraction"],
                "device_class": require_digest(
                    direct["device_class"], "device class"
                ),
                "lammps_executable_sha256": require_digest(
                    result["artifacts"]["lmp"]["sha256"],
                    "LAMMPS executable",
                ),
                "execution_space": direct["execution_space"],
                "layout": direct["kokkos_layout"],
                "block_schedule": (
                    candidate_identity["block_schedule"]
                    if selected_block
                    else "not_selected"
                ),
                "block_team_size": (
                    int(candidate_identity["block_team_size"])
                    if selected_block
                    else 0
                ),
                "block_scratch_layout": (
                    candidate_identity["block_scratch_layout"]
                    if selected_block
                    else "not_selected"
                ),
                "block_scratch_bytes": (
                    int(candidate_identity["block_scratch_bytes"])
                    if selected_block
                    else 0
                ),
                "source_policy": candidate_identity["source_policy"],
                "vjp_policy": candidate_identity["VJP_policy"],
                "chunksize": int(result["configuration"]["ye3t_chunksize"]),
                "minimum_observed_centers": min(center_counts),
                "maximum_observed_centers": max(center_counts),
            }
        )
    if not candidates:
        raise ValueError(str(path) + " contains no calibrated non-direct evaluator")
    return candidates


def candidate_group(candidate):
    return (
        candidate["evaluator"],
        candidate["block_schedule"],
        candidate["block_team_size"],
        candidate["block_scratch_layout"],
        candidate["block_scratch_bytes"],
    )


def workload_key(candidate):
    return (
        candidate["minimum_observed_centers"],
        candidate["maximum_observed_centers"],
    )


def select_candidate(candidates, minimum_speedup):
    reference = None
    for candidate in candidates:
        identity = (
            candidate["device_class"],
            candidate["execution_space"],
            candidate["layout"],
            candidate["chunksize"],
            candidate["lammps_executable_sha256"],
        )
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError("calibration inputs use different device/runtime classes")

    workloads = {workload_key(candidate) for candidate in candidates}
    groups = {}
    for candidate in candidates:
        key = candidate_group(candidate)
        group = groups.setdefault(key, {})
        workload = workload_key(candidate)
        if workload in group:
            raise ValueError("duplicate calibration for one evaluator/workload pair")
        group[workload] = candidate
    complete = []
    for key, group in groups.items():
        if set(group) != workloads:
            raise ValueError("each evaluator schedule must cover the full workload grid")
        values = list(group.values())
        worst_lower = min(candidate["lower_speedup"] for candidate in values)
        if worst_lower < minimum_speedup:
            continue
        complete.append(
            (
                worst_lower,
                min(candidate["median_speedup"] for candidate in values),
                -max(candidate["candidate_median"] for candidate in values),
                key,
                values,
            )
        )
    if not complete:
        return "direct", candidates[0], []
    selected = max(complete, key=lambda record: record[:4])
    values = selected[4]
    return values[0]["evaluator"], values[0], values


def selection_records(function_map, elements, evaluator):
    def matches(item):
        if evaluator == "direct":
            return item["evaluator"] == "explicit_ctilde"
        capabilities = set(item.get("required_capabilities", []))
        if evaluator == "block":
            return "block_symmetric_power_forward_v1" in capabilities
        if evaluator == "coupled_product":
            return "ace_coupled_product_dag_forward_v1" in capabilities
        if evaluator == "scalar_power":
            return item["evaluator"] == "scalar_invariant_power"
        raise ValueError("unsupported replay evaluator: " + evaluator)

    records = []
    for entry in function_map["entries"]:
        alternatives = [
            item
            for item in entry["alternatives"]
            if item["availability"] == {"reason": "", "status": "available"}
        ]
        direct = [
            item for item in alternatives if item["evaluator"] == "explicit_ctilde"
        ]
        selected = [item for item in alternatives if matches(item)]
        if len(direct) != 1 or len(selected) > 1:
            raise ValueError("function map contains an ambiguous replay candidate")
        selected = selected[0] if selected else direct[0]
        central = entry["central_type"]
        records.append(
            {
                "candidate_id": selected["alternative_id"],
                "central_type": central,
                "element": elements[central],
                "feature_id": entry["feature_id"],
                "function_index": entry["function_index"],
            }
        )
    records.sort(key=lambda record: (record["central_type"], record["function_index"]))
    return records


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Choose a confidence-qualified Kokkos evaluator and schedule from "
            "exclusive interleaved benchmark results and write a strict replay."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--benchmark", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--minimum-centers-per-rank", type=int, required=True)
    parser.add_argument("--maximum-centers-per-rank", type=int, required=True)
    parser.add_argument(
        "--minimum-lower-confidence-speedup", type=float, default=0.03
    )
    args = parser.parse_args()
    if args.rank < 1:
        parser.error("--rank must be positive")
    if (
        args.minimum_centers_per_rank < 1
        or args.maximum_centers_per_rank < args.minimum_centers_per_rank
    ):
        parser.error("the center-count envelope is invalid")
    if args.minimum_lower_confidence_speedup < 0.0:
        parser.error("the confidence speedup threshold must be nonnegative")

    model_hash = sha256_file(args.model)
    manifest_hash = sha256_file(args.manifest)
    manifest = load_json(args.manifest)
    function_map = load_json(payload_path(args.manifest, manifest, "yace_function_map"))
    candidates = [
        candidate
        for path in args.benchmark
        for candidate in calibration_candidates(
            path, args.rank, model_hash, manifest_hash
        )
    ]
    evaluator, selected, selected_grid = select_candidate(
        candidates, args.minimum_lower_confidence_speedup
    )
    observed_minimum = min(
        candidate["minimum_observed_centers"] for candidate in candidates
    )
    observed_maximum = max(
        candidate["maximum_observed_centers"] for candidate in candidates
    )
    if (args.minimum_centers_per_rank, args.maximum_centers_per_rank) != (
        observed_minimum,
        observed_maximum,
    ):
        parser.error(
            "the replay center-count envelope must equal the calibrated endpoints "
            f"[{observed_minimum}, {observed_maximum}]"
        )
    evidence = {
        "candidates": candidates,
        "minimum_lower_confidence_speedup": args.minimum_lower_confidence_speedup,
        "rank": args.rank,
        "selected_evaluator": evaluator,
        "selected_workload_grid": [
            workload_key(candidate) for candidate in selected_grid
        ],
        "selection_rule": "worst_case_lower_confidence_across_workload_grid_v1",
        "selected_schedule": (
            selected["block_schedule"] if evaluator == "block" else "not_selected"
        ),
    }
    evidence_hash = sha256_bytes(canonical_bytes(evidence))
    if evaluator == "block":
        schedule = selected["block_schedule"]
        team_size = selected["block_team_size"]
        scratch_layout = selected["block_scratch_layout"]
        scratch_bytes = selected["block_scratch_bytes"]
        source_policy = selected["source_policy"]
        vjp_policy = selected["vjp_policy"]
    else:
        schedule = "not_selected"
        team_size = 0
        scratch_layout = "not_selected"
        scratch_bytes = 0
        source_policy = selected["source_policy"]
        vjp_policy = selected["vjp_policy"]

    replay = {
        "calibration": {
            "evidence_sha256": evidence_hash,
            "method": "interleaved_paired_bootstrap_v1",
            "selected_evaluator": evaluator,
            "selection_rule": "worst_case_lower_confidence_across_workload_grid_v1",
        },
        "canonical_encoding": "ye3t_sorted_json_indent2_lf_v1",
        "compiler": {
            "plan_hash": manifest["compiler"]["plan_hash"],
            "sidecar_manifest_sha256": manifest_hash,
        },
        "device": {
            "device_class_sha256": selected["device_class"],
            "execution_space": selected["execution_space"],
        },
        "runtime": {
            "lammps_executable_sha256": selected[
                "lammps_executable_sha256"
            ]
        },
        "replay_sha256": "",
        "schema": "ye3t_kokkos_auto_replay_v2",
        "selections": selection_records(
            function_map, manifest["source_yace"]["ordered_elements"], evaluator
        ),
        "source_yace_sha256": model_hash,
        "workload": {
            "block_schedule": schedule,
            "block_scratch_bytes": scratch_bytes,
            "block_scratch_layout": scratch_layout,
            "block_team_size": team_size,
            "chunksize": selected["chunksize"],
            "kernel_abi": "ye3t_kokkos_candidate_runtime_v2",
            "layout": selected["layout"],
            "maximum_centers_per_rank": args.maximum_centers_per_rank,
            "minimum_centers_per_rank": args.minimum_centers_per_rank,
            "precision": "fp64",
            "source_policy": source_policy,
            "vjp_policy": vjp_policy,
        },
    }
    replay["replay_sha256"] = sha256_bytes(
        canonical_bytes(
            {key: value for key, value in replay.items() if key != "replay_sha256"}
        )
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_bytes(replay, pretty=True))
    print(
        json.dumps(
            {
                "evaluator": evaluator,
                "output": str(output.resolve()),
                "replay_sha256": replay["replay_sha256"],
                "schedule": schedule,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
