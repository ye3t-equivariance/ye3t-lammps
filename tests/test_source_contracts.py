#!/usr/bin/env python3
"""Source-level layout checks for the evaluators and the installer inventory.

These checks read the C++ sources as text and are independent of
formatting: every comparison is made on a whitespace-normalized copy of the
file. They are not a Kokkos compilation or runtime test.
"""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


def squash(text):
    """Collapse every run of whitespace so formatting does not matter."""
    return re.sub(r"\s+", " ", text).strip()


class Source:
    def __init__(self, relative_path):
        self.text = squash((ROOT / relative_path).read_text(encoding="utf-8"))

    def count(self, needle):
        return self.text.count(squash(needle))

    def index(self, needle):
        return self.text.index(squash(needle))

    def has(self, needle):
        return squash(needle) in self.text

    def between(self, start, end):
        start_index = self.index(start)
        end_index = self.text.index(squash(end), start_index)
        return self.text[start_index:end_index]


class SourceContracts(unittest.TestCase):
    def test_all_model_families_have_force_only_launch(self):
        source = Source("src/KOKKOS/pair_ye3t_kokkos.cpp")
        for family in ("", "Lifted", "Tagged"):
            label = f'"YE3TScatter{family}ForceOnly"'
            self.assertTrue(source.has(label))
            start = source.index(label)
            self.assertIn("Kokkos::parallel_for(", source.text[start - 70:start])
        self.assertEqual(source.count("if (eflag_global || vflag_global) {"), 3)
        self.assertEqual(source.count("k_eatom_.sync_host();"), 3)
        self.assertEqual(source.count("k_vatom_.sync_host();"), 3)
        self.assertEqual(source.count("pair_virial_fdotr_compute(this);"), 3)

    def test_unused_reductions_removed_not_status_checks(self):
        source = Source("src/KOKKOS/pair_ye3t_kokkos.cpp")
        for family in ("Lifted", "Tagged"):
            label = f'"YE3TCount{family}Edges"'
            start = source.index(label)
            self.assertIn("Kokkos::parallel_for(", source.text[start - 70:start])
        self.assertFalse(source.has("residual_imaginary_unused"))
        self.assertTrue(source.has('"YE3TFinalizeTotalDensity"'))
        # One deferred status read per step on the tally fence; the edge-summary
        # read remains per chunk.
        self.assertFalse(source.has("report_device_status(copy_device_status());"))
        self.assertEqual(source.count("report_device_status(deferred_status);"), 3)
        self.assertEqual(source.count("step_state_.read_totals(device_status_, deferred_status)"), 3)
        self.assertTrue(source.has("if (use_neighbor_major_source_) {"))
        self.assertEqual(source.count("if (d_eatom_.extent(0) <"), 3)
        self.assertEqual(source.count("if (d_vatom_.extent(0) <"), 3)

    def test_gpu_direct_prefix_lowered_before_upload(self):
        plan = Source("src/KOKKOS/ye3t_kokkos_plan.h")
        start = plan.index("const auto binary = lower_gpu_binary_dag(polynomial);")
        self.assertLess(start, plan.index('checked_size(binary.left.size(), "binary DAG")'))
        self.assertLess(start, plan.index("const auto reverse = make_gpu_dag_schedule("))
        self.assertFalse(plan.has("requires a binary product DAG"))
        self.assertFalse(plan.has("polynomial.binary_node_left"))
        self.assertFalse(plan.has("polynomial.binary_node_right"))
        self.assertFalse(plan.has("polynomial.monomial_nodes"))
        # The CPU compiler keeps its retained-prefix cost comparison.
        loader = Source("src/ye3t_yace_model.cpp")
        self.assertTrue(loader.has("binary.node_left.size()) < prefix_product_count"))

    def test_installer_carries_every_source(self):
        script = (ROOT / "tools/patch_lammps.sh").read_text(encoding="utf-8")
        for file in ("ye3t_cpu_batching.h", "ye3t_lifted_cauchy_source.cpp",
                     "ye3t_cpu_source_tiling.h", "ye3t_cpu_source_tiling.cpp",
                     "ye3t_gpu_dag_schedule.h", "ye3t_gpu_tagged_source.h",
                     "ye3t_kokkos_step_state.h", "ye3t_kokkos_types.h",
                     "ye3t_gpu_block_schedule.h", "ye3t_gpu_harmonic_stream.h"):
            self.assertIn(file, script)
        for path in sorted((ROOT / "src").glob("*.cpp")) + sorted((ROOT / "src").glob("*.h")):
            if path.name != "ye3t_plugin.cpp":
                self.assertIn(path.name, script, path.name)
        for path in sorted((ROOT / "src" / "KOKKOS").iterdir()):
            self.assertIn(path.name, script, path.name)
        self.assertIn("doc/src/pair_ye3t.rst", script)
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("src/ye3t_lifted_cauchy_source.cpp", cmake)

    def test_gpu_scheduling_contracts(self):
        source = Source("src/KOKKOS/pair_ye3t_kokkos.cpp")
        team = source.between(
            "struct DirectTeamForwardAdjoint",
            "template <class DeviceType> KOKKOS_INLINE_FUNCTION void add_full_channel_adjoint",
        )
        self.assertNotIn("Kokkos::atomic_add", team)
        self.assertIn("gather_gpu_dag_adjoint", team)
        self.assertIn("dag_source_power_offsets", team)
        self.assertEqual(source.count("step_state_.read_edge_summary();"), 3)
        # Three count-stage checks consume the combined snapshot; all other
        # status flags are collected once per step on the tally fence.
        self.assertEqual(source.count("report_device_status(edge_summary.status);"), 3)
        self.assertEqual(source.count("Kokkos::deep_copy(DeviceType{}, device_status_, 0);"), 3)
        self.assertEqual(source.count("step_state_.reset();"), 3)
        self.assertEqual(source.count("step_state_.read_totals("), 3)
        self.assertTrue(source.has("if (vflag_global || TALLY_ATOM_VIRIAL)"))
        for removed in ("chunk_ev", "chunk_imaginary"):
            self.assertFalse(source.has(removed))

    def test_structured_gpu_execution_contracts(self):
        source = Source("src/KOKKOS/pair_ye3t_kokkos.cpp")
        uploader = Source("src/KOKKOS/ye3t_kokkos_plan.h")
        self.assertTrue(source.has("gpu_block_adjoint_transpose"))
        self.assertTrue(source.has("gpu_block_forward_tiles"))
        self.assertTrue(source.has("direct_resident_values_"))
        self.assertTrue(source.has("GpuHarmonicStream<false"))
        self.assertTrue(source.has("GpuHarmonicStream<true"))
        self.assertTrue(uploader.has("make_gpu_block_execution_schedule"))
        self.assertTrue(source.has("ye3t_kokkos_candidate_runtime_v2"))
        # Three complete compute paths each annotate all four physical stages.
        for stage in ("source", "readout_forward_adjoint", "source_pullback", "force_virial_scatter"):
            self.assertEqual(source.count('phase.next("YE3T::' + stage + '")'), 3)
        profile = source.between("class YE3TProfilePhase", "};")
        self.assertNotIn("fence(", profile)

    def test_cpu_source_tiling_is_internal_and_automatic(self):
        header = Source("src/pair_ye3t.h")
        source = Source("src/pair_ye3t.cpp")
        evaluator = Source("src/ye3t_cpu_evaluator.cpp")
        self.assertFalse(header.has("cpu_source_tile_bytes"))
        self.assertFalse(source.has("cpu_source_tile_bytes"))
        self.assertFalse(source.has("cpu_source_tile_end("))
        self.assertTrue(evaluator.has("make_cpu_source_tile_policy("))
        self.assertTrue(evaluator.has("cpu_source_tiled_passes("))
        self.assertTrue(evaluator.has("prepare_source_tile("))
        self.assertTrue(evaluator.has("evaluate_complete_batch("))
        self.assertTrue(evaluator.has("reference_edge_tiling_ || atom_count == 0"))
        self.assertTrue(evaluator.has("begin, std::max(1, count)"))
        self.assertTrue(evaluator.has("complex_spherical_harmonics_nonnegative_unit_recurrence"))
        self.assertTrue(evaluator.has("cpu_accumulate_ace_source_edge("))
        self.assertTrue(evaluator.has("cpu_pullback_ace_source_edge("))
        self.assertTrue(evaluator.has("evaluate_readout_batch(model_->species(batch_species), atom_count,"))
        self.assertTrue(source.has("neighbors[jj] & NEIGHMASK"))
        self.assertTrue(source.has("x * x + y * y + z * z >= cutoff * cutoff"))
        for axis in range(3):
            self.assertTrue(source.has(f"forces[atom_j][{axis}] -="))

    def test_lammps_file_conventions(self):
        banner = "LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator"
        for directory in (ROOT / "src", ROOT / "src" / "KOKKOS"):
            for path in sorted(directory.glob("*.cpp")) + sorted(directory.glob("*.h")):
                text = path.read_text(encoding="utf-8")
                self.assertIn(banner, text[:400], path.name)
                self.assertIn("Contributing author", text[:1200], path.name)
                if path.suffix == ".h":
                    self.assertNotIn("#pragma once", text, path.name)
                    self.assertIn("#ifndef LMP_", text, path.name)
        self.assertTrue((ROOT / ".clang-format").is_file())
        self.assertTrue((ROOT / "doc" / "src" / "pair_ye3t.rst").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
