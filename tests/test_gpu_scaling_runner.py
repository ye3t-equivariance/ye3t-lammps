#!/usr/bin/env python3

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load_module(
    "ye3t_gpu_scaling_runner",
    ROOT / "tools" / "gpu_scaling" / "run_gpu_scaling.py",
)
WRAPPER = load_module(
    "ye3t_gpu_rank_wrapper",
    ROOT / "tools" / "gpu_scaling" / "gpu_rank_wrapper.py",
)


class GPUScalingRunnerTest(unittest.TestCase):
    def test_weak_scaling_factorization(self):
        self.assertEqual(RUNNER.factor_3d(1), (1, 1, 1))
        self.assertEqual(RUNNER.factor_3d(2), (2, 1, 1))
        self.assertEqual(RUNNER.factor_3d(4), (2, 2, 1))
        self.assertEqual(RUNNER.factor_3d(8), (2, 2, 2))
        self.assertEqual(RUNNER.cell_shape("weak", 5, 4), (10, 10, 5))
        self.assertEqual(RUNNER.cell_shape("strong", 5, 4), (5, 5, 5))

    def test_rank_environment_and_explicit_mapping(self):
        environment = {
            "OMPI_COMM_WORLD_RANK": "3",
            "OMPI_COMM_WORLD_LOCAL_RANK": "1",
            "OMPI_COMM_WORLD_SIZE": "4",
            "PMI_RANK": "3",
            "PMI_SIZE": "4",
        }
        self.assertEqual(
            WRAPPER.rank_context(environment),
            {"global_rank": 3, "local_rank": 1, "world_size": 4},
        )
        mapping = WRAPPER.select_device("explicit", "2,5", 3, environment)
        self.assertEqual(mapping["device"], "5")
        self.assertTrue(mapping["mapping_wrapped"])

    def test_launcher_mapping_requires_one_visible_device(self):
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            WRAPPER.select_device(
                "launcher", "", 0, {"CUDA_VISIBLE_DEVICES": "0,1"}
            )
        mapping = WRAPPER.select_device(
            "launcher", "", 7, {"CUDA_VISIBLE_DEVICES": "GPU-abc"}
        )
        self.assertEqual(mapping["device"], "GPU-abc")
        self.assertFalse(mapping["mapping_wrapped"])

    @staticmethod
    def rank_record(
        global_rank,
        local_rank,
        uuid,
        wrapped=False,
        processes=None,
        pci_bus_id=None,
    ):
        return {
            "schema": "ye3t_gpu_rank_map_v1",
            "hostname": "node0",
            "global_rank": global_rank,
            "local_rank": local_rank,
            "world_size": 2,
            "mapping_wrapped": wrapped,
            "gpu": {
                "uuid": uuid,
                "pci_bus_id": pci_bus_id or "00000000:01:00.0",
                "memory_total_mib": 1000.0,
                "memory_free_mib": 900.0,
            },
            "compute_processes": [] if processes is None else processes,
        }

    def test_performance_rank_map_rejects_duplicate_gpu(self):
        args = SimpleNamespace(
            mode="performance", memory_headroom=0.15, allow_oversubscription=False
        )
        records = [
            self.rank_record(0, 0, "GPU-a"),
            self.rank_record(1, 1, "GPU-a", wrapped=True),
        ]
        with self.assertRaisesRegex(RuntimeError, "unique physical GPU"):
            RUNNER.validate_rank_map(args, records, 2)

    def test_correctness_rank_map_requires_explicit_oversubscription(self):
        records = [
            self.rank_record(0, 0, "GPU-a"),
            self.rank_record(1, 1, "GPU-a", wrapped=True),
        ]
        args = SimpleNamespace(
            mode="correctness", memory_headroom=0.15, allow_oversubscription=False
        )
        with self.assertRaisesRegex(RuntimeError, "--allow-oversubscription"):
            RUNNER.validate_rank_map(args, records, 2)
        args.allow_oversubscription = True
        RUNNER.validate_rank_map(args, records, 2)

    def test_performance_rank_map_rejects_same_physical_pci_device(self):
        args = SimpleNamespace(
            mode="performance", memory_headroom=0.15, allow_oversubscription=False
        )
        records = [
            self.rank_record(0, 0, "MIG-a", pci_bus_id="00000000:01:00.0"),
            self.rank_record(1, 1, "MIG-b", pci_bus_id="00000000:01:00.0"),
        ]
        with self.assertRaisesRegex(RuntimeError, "unique physical GPU"):
            RUNNER.validate_rank_map(args, records, 2)

    def test_timing_and_dispatch_parsers(self):
        digest = "a" * 64
        text = """
YE3T Kokkos device-plan probe: execution_space Cuda, source_model {digest}, direct_logical_plan {digest}, requested_policy auto, evaluator auto[direct], active_families direct, flat_plan {digest}, device_schedule {digest}, semantic_selection {digest}, selection_rank_consensus passed, source_policy center_tiled_neighbor_major_v1, source_serial_terms 78, free_device_bytes 900, total_device_bytes 1000, VJP_schedule {digest}, copy_sentinel_maximum_error 0, scalar_math_probe not_run, coupled_math_probe not_run
(1) pair ye3t/kk, perpetual
    attributes: full, newton on, kokkos_device
    pair build: full/bin/kk/device
Loop time of 0.1 on 2 procs for 1 steps with 16 atoms
Loop time of 2.0 on 2 procs for 10 steps with 16 atoms
MPI task timing breakdown:
Section | min time | avg time | max time |%varavg| %CPU | %total
Pair | 1.0 | 1.1 | 1.2 | 0.0 | 100 | 55
Comm | 0.2 | 0.3 | 0.4 | 0.0 | 100 | 15
Per MPI rank memory allocation (min/avg/max) = 10 | 11 | 12 Mbytes
Nlocal: 8 ave 8 max 8 min
Nghost: 4 ave 4 max 4 min
FullNghs: 208 ave 208 max 208 min
""".format(digest=digest).replace(
            "selection_rank_consensus passed,",
            "selection_rank_consensus passed, active_direct yes, active_block no, "
            "active_scalar_power no, active_coupled_product no, "
            "direct_monomial_storage 10, block_routes 0, scalar_routes 0, "
            "coupled_plans 0, planner_profile test, planner_algorithm test, "
            "planner_status test, calibration_hash not_applicable, "
            "decision_reason test,",
        )
        timing = RUNNER.parse_loop_timing(text, 10, 16, 2)
        self.assertEqual(timing["loop_seconds"], 2.0)
        self.assertEqual(timing["pair_seconds"], 1.1)
        self.assertEqual(timing["comm_seconds"], 0.3)
        self.assertEqual(timing["memory_per_rank"]["maximum_mib"], 12.0)
        marker = RUNNER.parse_ye3t_dispatch(
            text, "auto", {"selected_evaluator": "direct"}
        )
        self.assertEqual(marker["evaluator"], "auto[direct]")
        self.assertEqual(marker["source_policy"], "center_tiled_neighbor_major_v1")
        self.assertEqual(marker["source_serial_terms"], "78")
        with self.assertRaisesRegex(RuntimeError, "replay selection"):
            RUNNER.parse_ye3t_dispatch(
                text, "auto", {"selected_evaluator": "block"}
            )

    def test_checked_case_matrix_covers_all_gpu_evaluator_families(self):
        matrix = RUNNER.load_case_matrix(
            ROOT / "tools" / "gpu_scaling" / "gpu_scaling_cases.json"
        )
        covered = set()
        for record in matrix["cases"].values():
            covered.update(record["legal_evaluators"])
        self.assertEqual(covered, set(RUNNER.EVALUATORS))
        self.assertFalse(matrix["cases"]["h32"]["pace_product"]["supported"])

    def test_forced_optimized_dispatch_requires_a_route(self):
        digest = "b" * 64
        text = (
            "YE3T Kokkos device-plan probe: execution_space Cuda, "
            "source_model {0}, direct_logical_plan {0}, requested_policy block, "
            "evaluator block, active_families direct, flat_plan {0}, "
            "device_schedule {0}, semantic_selection {0}, "
            "selection_rank_consensus passed, active_direct yes, active_block no, "
            "active_scalar_power no, active_coupled_product no, block_routes 0, "
            "direct_monomial_storage 10, scalar_routes 0, coupled_plans 0, "
            "free_device_bytes 900, "
            "total_device_bytes 1000, VJP_schedule {0}, "
            "copy_sentinel_maximum_error 0, scalar_math_probe not_run, "
            "coupled_math_probe not_run\n"
            "(1) pair ye3t/kk, perpetual\n"
            "    attributes: full, newton on, kokkos_device\n"
            "    pair build: full/bin/kk/device\n"
        ).format(digest)
        with self.assertRaisesRegex(RuntimeError, "no optimized routes"):
            RUNNER.parse_ye3t_dispatch(text, "block")

    def test_pace_dispatch_rejects_recursive_marker(self):
        text = (
            "KOKKOS mode with Kokkos version 5.2.1 is enabled\n"
            "Product evaluator is used\nRecursive evaluator is used\n"
            "(1) pair pace/kk, perpetual\n"
            "    attributes: full, newton on, kokkos_device\n"
            "    pair build: full/bin/kk/device\n"
        )
        with self.assertRaisesRegex(RuntimeError, "recursive"):
            RUNNER.parse_pace_dispatch(text)

    def test_correctness_summary_suppresses_performance_statistics(self):
        records = [
            {
                "ranks": 1,
                "repetition": 0,
                "engine": "direct",
                "status": "passed",
                "timing": {
                    "loop_seconds": 1.0,
                    "loop_microseconds_per_atom_step": 2.0,
                },
            }
        ]
        summary = RUNNER.summarize(
            records, [1], ["direct"], "strong", performance_statistics=False
        )
        self.assertEqual(summary["scaling"], {})
        self.assertIsNone(summary["by_rank"]["1"]["fastest_ye3t_evaluator"])
        self.assertEqual(summary["by_rank"]["1"]["paired_comparisons"], {})

    def test_runtime_device_and_capacity_records(self):
        digest = "c" * 64
        text = (
            "YE3T Kokkos rank-device: world_rank 0, world_size 1, local_rank 0, "
            "local_size 1, launcher_local_rank 0, hostname node0, "
            "execution_space Cuda, device_ordinal 0, uuid GPU-a, "
            "uuid_status available, pci_bus_id 0000:01:00.0, device_class {0}, "
            "visible_mask {0}, gpu_aware_mpi 1, runtime_version 13000, "
            "driver_version 13000, total_device_bytes 1000\n"
            "YE3T Kokkos rank-device summary: world_size 1, node_count 1, "
            "rank_map {0}, uuid_status complete, device_class_consensus passed\n"
            "YE3T Kokkos capacity: world_rank 0, configured_chunk 64, "
            "local_inum 54, requested_chunk 54, effective_chunk 54, "
            "center_capacity 64, edge_capacity 100, dynamic_bytes 400, "
            "resident_bytes 500, reallocations_before 0, reallocations_after 2, "
            "center_reduced 0, edge_reductions 0, free_device_bytes 900, "
            "total_device_bytes 1000, exact_chunk_required 1\n"
        ).format(digest)
        runtime, summary = RUNNER.parse_runtime_device_records(text, 1)
        self.assertEqual(summary["rank_map"], digest)
        probes = [
            {
                "global_rank": 0,
                "hostname": "node0",
                "gpu": {"uuid": "GPU-a", "pci_bus_id": "00000000:01:00.0"},
            }
        ]
        RUNNER.validate_runtime_device_records(runtime, probes)
        capacity = RUNNER.parse_capacity_records(text, 1, True, 0.15)
        self.assertEqual(capacity[0]["effective_chunk"], "54")

    def test_pretiming_parity_certificate_binds_dumps_and_artifacts(self):
        dump = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS pp pp pp
0 1
0 1
0 1
ITEM: ATOMS id type x y z c_atom_energy c_atom_stress[1] c_atom_stress[2] c_atom_stress[3] c_atom_stress[4] c_atom_stress[5] c_atom_stress[6] fx fy fz
1 1 0 0 0 -1 0 0 0 0 0 0 0.1 -0.2 0.3
"""
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "direct.dump"
            second = Path(temporary) / "block.dump"
            first.write_text(dump, encoding="utf-8")
            second.write_text(dump, encoding="utf-8")
            records = []
            for engine, path in (("direct", first), ("block", second)):
                records.append(
                    {
                        "engine": engine,
                        "parity_dump_sha256": RUNNER.sha256(path),
                        "paths": {"parity_dump": str(path)},
                        "ranks": 1,
                        "repetition": 0,
                        "status": "passed",
                    }
                )
            model = ROOT / "examples" / "PACKAGES" / "ye3t" / "models" / "ta_l8_compact" / "model.yace"
            paths = {
                "lmp": Path("/bin/true"),
                "model": model,
                "plan": ROOT / "examples" / "PACKAGES" / "ye3t" / "models" / "ta_l8_compact" / "manifest.json",
                "runner": ROOT / "tools" / "gpu_scaling" / "run_gpu_scaling.py",
                "verifier": ROOT / "examples" / "PACKAGES" / "ye3t" / "verify_examples.py",
            }
            certificate = RUNNER.build_parity_certificate(
                records, [1], ["direct", "block"], paths
            )
            self.assertEqual(certificate["status"], "passed")
            self.assertEqual(len(certificate["certificate_sha256"]), 64)
            self.assertTrue(
                certificate["comparisons"][0]["comparison"]["comparisons"][0][
                    "passed"
                ]
            )


if __name__ == "__main__":
    unittest.main()
