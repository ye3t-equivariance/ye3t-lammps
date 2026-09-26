#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "PACKAGES" / "ye3t"
TOOLS = ROOT / "tools" / "gpu_scaling"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPLAY = load_module(
    "ye3t_kokkos_auto_replay", TOOLS / "make_kokkos_auto_replay.py"
)


def candidate(evaluator, schedule, team, centers, lower, median):
    return {
        "block_schedule": schedule,
        "block_scratch_bytes": 0 if team == 0 else 1024,
        "block_scratch_layout": (
            "not_selected" if team == 0 else "monomial_tile16_lane_fast_v2"
        ),
        "block_team_size": team,
        "candidate_median": 2.0,
        "chunksize": 4096,
        "device_class": "a" * 64,
        "evaluator": evaluator,
        "execution_space": "Cuda",
        "lammps_executable_sha256": "b" * 64,
        "layout": "legacy",
        "lower_speedup": lower,
        "maximum_observed_centers": centers,
        "median_speedup": median,
        "minimum_observed_centers": centers,
        "source_policy": "center_tiled_neighbor_major_v1",
        "vjp_policy": "edge_grouped_harmonic_cached_radial_v1",
    }


def authorized_result():
    artifacts = {
        name: {"sha256": character * 64}
        for name, character in zip(
            ("lmp", "model", "plan", "runner", "verifier"), "abcde"
        )
    }
    certificate = {
        "artifacts": {
            name: record["sha256"] for name, record in artifacts.items()
        },
        "comparisons": [
            {
                "candidates": ["block"],
                "comparison": {"comparisons": [{"passed": True}]},
                "dump_sha256": {"block": "2" * 64, "direct": "1" * 64},
                "mpi_ranks": 1,
                "reference": "direct",
                "repetition": 0,
            }
        ],
        "schema": "ye3t_gpu_pretiming_parity_v1",
        "status": "passed",
    }
    certificate["certificate_sha256"] = REPLAY.sha256_bytes(
        REPLAY.canonical_bytes(certificate)
    )
    return {
        "artifacts": artifacts,
        "configuration": {
            "engines": ["direct", "block"],
            "mode": "performance",
        },
        "failed_records": 0,
        "parity_certificate": certificate,
        "publication_timing_eligible": True,
        "records": [
            {
                "engine": "direct",
                "parity_dump_sha256": "1" * 64,
                "ranks": 1,
                "repetition": 0,
                "status": "passed",
            },
            {
                "engine": "block",
                "parity_dump_sha256": "2" * 64,
                "ranks": 1,
                "repetition": 0,
                "status": "passed",
            },
        ],
    }


class KokkosAutoCalibrationTest(unittest.TestCase):
    def run_dry(self, bundle):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "calibration"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(TOOLS / "calibrate_kokkos_auto.py"),
                    "--lmp",
                    "/bin/true",
                    "--output",
                    str(output),
                    "--bundle",
                    bundle,
                    "--without-pace",
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            return json.loads(
                (output / "calibration_manifest.json").read_text(encoding="utf-8")
            )

    def test_compact_dry_run_covers_workload_and_team_grid(self):
        manifest = self.run_dry("compact")
        self.assertEqual(manifest["status"], "dry_run")
        self.assertEqual(
            manifest["configuration"]["base_cells"], [3, 6, 8, 10]
        )
        self.assertEqual(
            manifest["configuration"]["team_sizes"], [32, 64, 128, 256]
        )
        self.assertEqual(len(manifest["commands"]), 23)
        labels = [record["label"] for record in manifest["commands"]]
        self.assertIn("calibration cells=6 schedule=fused", labels)
        self.assertIn("calibration cells=10 schedule=work_major-256", labels)
        self.assertEqual(
            labels[-3:],
            [
                "freeze AUTO replay",
                "independent timing confirmation cells=9",
                "validate AUTO replay",
            ],
        )
        self.assertIn("--rank-flag=-n", manifest["commands"][0]["command"])

    def test_full_bundle_calibrates_coupled_product_without_block_grid(self):
        manifest = self.run_dry("full")
        self.assertEqual(
            manifest["configuration"]["evaluators"],
            ["direct", "coupled_product"],
        )
        self.assertEqual(len(manifest["commands"]), 7)
        calibration = manifest["commands"][0]["command"]
        self.assertIn("coupled_product", calibration)
        self.assertNotIn("--block-team-size", calibration)

    def test_selection_requires_speedup_across_the_workload_grid(self):
        candidates = [
            candidate("block", "fused_center_lane_v2", 0, 432, 0.08, 0.10),
            candidate("block", "fused_center_lane_v2", 0, 2000, 0.01, 0.03),
            candidate("block", "plan_route_major_v1", 64, 432, 0.05, 0.07),
            candidate("block", "plan_route_major_v1", 64, 2000, 0.04, 0.06),
            candidate("scalar_power", "not_selected", 0, 432, 0.04, 0.08),
            candidate("scalar_power", "not_selected", 0, 2000, 0.035, 0.05),
        ]
        evaluator, selected, grid = REPLAY.select_candidate(candidates, 0.03)
        self.assertEqual(evaluator, "block")
        self.assertEqual(selected["block_team_size"], 64)
        self.assertEqual({REPLAY.workload_key(item) for item in grid}, {(432, 432), (2000, 2000)})

    def test_incomplete_workload_grid_is_rejected(self):
        candidates = [
            candidate("block", "fused_center_lane_v2", 0, 432, 0.05, 0.06),
            candidate("block", "fused_center_lane_v2", 0, 2000, 0.05, 0.06),
            candidate("scalar_power", "not_selected", 0, 432, 0.05, 0.06),
        ]
        with self.assertRaisesRegex(ValueError, "full workload grid"):
            REPLAY.select_candidate(candidates, 0.03)

    def test_compiler_candidates_map_to_each_supported_family(self):
        compact = json.loads(
            (EXAMPLES / "models" / "ta_l8_compact" / "yace_function_map.json").read_text(
                encoding="utf-8"
            )
        )
        full = json.loads(
            (EXAMPLES / "models" / "ta_l8_full" / "yace_function_map.json").read_text(
                encoding="utf-8"
            )
        )
        high = json.loads(
            (EXAMPLES / "models" / "ta_l8_h16" / "yace_function_map.json").read_text(
                encoding="utf-8"
            )
        )
        block = REPLAY.selection_records(compact, ["Ta"], "block")
        coupled = REPLAY.selection_records(full, ["Ta"], "coupled_product")
        scalar = REPLAY.selection_records(high, ["Ta"], "scalar_power")
        direct = REPLAY.selection_records(compact, ["Ta"], "direct")
        self.assertNotEqual(
            {record["candidate_id"] for record in block},
            {record["candidate_id"] for record in direct},
        )
        self.assertEqual(len(coupled), len(full["entries"]))
        self.assertEqual(len(scalar), len(high["entries"]))

    def test_replay_requires_same_run_parity_certificate(self):
        result = authorized_result()
        REPLAY.validate_benchmark_authorization("benchmark.json", result)
        result["parity_certificate"]["comparisons"][0]["comparison"][
            "comparisons"
        ][0]["passed"] = False
        with self.assertRaisesRegex(ValueError, "corrupted parity certificate"):
            REPLAY.validate_benchmark_authorization("benchmark.json", result)

    def test_replay_rejects_correctness_mode_result(self):
        result = authorized_result()
        result["configuration"]["mode"] = "correctness"
        with self.assertRaisesRegex(ValueError, "performance-mode"):
            REPLAY.validate_benchmark_authorization("benchmark.json", result)

    def test_replay_rejects_incomplete_evaluator_parity_coverage(self):
        result = authorized_result()
        comparison = result["parity_certificate"]["comparisons"][0]
        comparison["candidates"] = []
        comparison["dump_sha256"].pop("block")
        unsigned = {
            key: value
            for key, value in result["parity_certificate"].items()
            if key != "certificate_sha256"
        }
        result["parity_certificate"]["certificate_sha256"] = (
            REPLAY.sha256_bytes(REPLAY.canonical_bytes(unsigned))
        )
        with self.assertRaisesRegex(ValueError, "cover every evaluator"):
            REPLAY.validate_benchmark_authorization("benchmark.json", result)


if __name__ == "__main__":
    unittest.main()
