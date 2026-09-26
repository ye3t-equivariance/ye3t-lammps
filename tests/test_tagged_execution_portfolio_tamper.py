#!/usr/bin/env python3
"""Exercise tagged execution-portfolio compatibility and fail-closed loading."""

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def canonical_hash(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_case(directory, name, source_fixture, source_model, mutate):
    model = copy.deepcopy(source_model)
    model.pop("self_hash", None)
    mutate(model)
    model["self_hash"] = canonical_hash(model)
    model_path = directory / f"{name}.model.json"
    model_path.write_text(
        json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fixture = copy.deepcopy(source_fixture)
    fixture["model_path"] = model_path.name
    fixture["model_self_hash"] = model["self_hash"]
    fixture_path = directory / f"{name}.fixture.json"
    fixture_path.write_text(
        json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return fixture_path


def run(executable, fixture):
    return subprocess.run(
        [str(executable), str(fixture)],
        check=False,
        capture_output=True,
        text=True,
    )


def require_failure(result, fragment):
    combined = result.stdout + result.stderr
    if result.returncode == 0 or fragment not in combined:
        raise SystemExit(
            f"expected failure containing {fragment!r}; returncode={result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: test_tagged_execution_portfolio_tamper.py "
            "<native-test-executable> <fixture.json>"
        )
    executable = Path(sys.argv[1]).resolve()
    fixture_path = Path(sys.argv[2]).resolve()
    source_fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    model_path = Path(source_fixture["model_path"])
    if not model_path.is_absolute():
        model_path = fixture_path.parent / model_path
    source_model = json.loads(model_path.read_text(encoding="utf-8"))
    if "tagged_execution_portfolio" not in source_model:
        raise SystemExit("fixture model lacks tagged_execution_portfolio")

    with tempfile.TemporaryDirectory(prefix="ye3t-tagged-portfolio-") as tmp:
        directory = Path(tmp)

        legacy = write_case(
            directory,
            "legacy_without_portfolio",
            source_fixture,
            source_model,
            lambda model: model.pop("tagged_execution_portfolio"),
        )
        legacy_result = run(executable, legacy)
        if legacy_result.returncode != 0:
            raise SystemExit(
                "legacy compiled-direct compatibility failed\n"
                f"stdout:\n{legacy_result.stdout}\nstderr:\n{legacy_result.stderr}"
            )

        def stale_hash(model):
            plan = model["tagged_execution_portfolio"]["moment_plan"]
            plan["roots"][0] = 0

        stale = write_case(
            directory,
            "stale_portfolio_hash",
            source_fixture,
            source_model,
            stale_hash,
        )
        require_failure(run(executable, stale), "portfolio self-hash mismatch")

        def wrong_semantics(model):
            portfolio = model["tagged_execution_portfolio"]
            portfolio["moment_plan"]["roots"][0] = 0
            portfolio.pop("portfolio_hash", None)
            portfolio["portfolio_hash"] = canonical_hash(portfolio)

        wrong = write_case(
            directory,
            "wrong_schedule_semantics",
            source_fixture,
            source_model,
            wrong_semantics,
        )
        require_failure(
            run(executable, wrong),
            "binary moment root exponent differs from real_moment_program",
        )

        def stale_program_binding(model):
            model["real_moment_program"]["terms"][0]["coefficient"] += 0.125

        stale_program = write_case(
            directory,
            "stale_program_binding",
            source_fixture,
            source_model,
            stale_program_binding,
        )
        require_failure(
            run(executable, stale_program), "portfolio program binding changed"
        )

        if source_model["tagged_execution_portfolio"]["schema"].endswith("_v2"):
            def wrong_factorization(model):
                portfolio = model["tagged_execution_portfolio"]
                portfolio["outer_plan"]["factorization"] = (
                    "commutative_symmetric_power"
                )
                portfolio.pop("portfolio_hash", None)
                portfolio["portfolio_hash"] = canonical_hash(portfolio)

            factorization = write_case(
                directory,
                "wrong_outer_factorization",
                source_fixture,
                source_model,
                wrong_factorization,
            )
            require_failure(
                run(executable, factorization),
                "binary-plan factorization changed",
            )

            def false_prefix_certificate(model):
                portfolio = model["tagged_execution_portfolio"]
                portfolio["outer_plan"]["certificate"][
                    "preserves_canonical_left_to_right_order"
                ] = False
                portfolio.pop("portfolio_hash", None)
                portfolio["portfolio_hash"] = canonical_hash(portfolio)

            certificate = write_case(
                directory,
                "false_outer_prefix_certificate",
                source_fixture,
                source_model,
                false_prefix_certificate,
            )
            require_failure(
                run(executable, certificate),
                "canonical-prefix plan certificate failed",
            )

            def false_route_order_certificate(model):
                portfolio = model["tagged_execution_portfolio"]
                portfolio["certificate"][
                    "preserves_direct_route_accumulation_order"
                ] = False
                portfolio.pop("portfolio_hash", None)
                portfolio["portfolio_hash"] = canonical_hash(portfolio)

            route_order = write_case(
                directory,
                "false_route_order_certificate",
                source_fixture,
                source_model,
                false_route_order_certificate,
            )
            require_failure(
                run(executable, route_order),
                "portfolio route-order certificate failed",
            )

    print("PASS tagged execution portfolio compatibility and tamper checks")


if __name__ == "__main__":
    main()
