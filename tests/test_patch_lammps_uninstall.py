#!/usr/bin/env python3
"""Installer filesystem regressions; no runtime, compiler, MPI, or network needed.

Run: python3 tests/test_patch_lammps_uninstall.py [-v]
Optional environment: YE3T_LAMMPS_ROOT, YE3T_TEST_SHELL (sh, dash, or bash).
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import unittest

ROOT = Path(os.environ.get("YE3T_LAMMPS_ROOT", Path(__file__).resolve().parents[1])).resolve()
SHELL = os.environ.get("YE3T_TEST_SHELL", "sh")
CMAKE = """# installer fixture; unrelated CMake content must survive
set(STANDARD_PACKAGES
  ML-IAP
  ML-PACE
  ML-SNAP)

foreach(PKG_WITH_INCL GRAPHICS ML-IAP COMPRESS ML-PACE LEPTON FENIX)
  if(PKG_${PKG_WITH_INCL})
    include(Packages/${PKG_WITH_INCL})
  endif()
endforeach()
"""
PACKAGE = Path("src/ML-YE3T")
EXAMPLES = Path("examples/PACKAGES/ye3t")
PMANIFEST = PACKAGE / "YE3T_LAMMPS_MANIFEST.sha256"
KMANIFEST = PACKAGE / "YE3T_LAMMPS_KOKKOS_MANIFEST.sha256"
EMANIFEST = EXAMPLES / "YE3T_EXAMPLE_MANIFEST.sha256"
CPU = PACKAGE / "pair_ye3t.cpp"
GPU = Path("src/KOKKOS/pair_ye3t_kokkos.cpp")


def run_command(*args, env=None):
    return subprocess.run([str(a) for a in args], text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, env=env, timeout=45)


def snapshot(root):
    """Bytes, modes, symlinks and directories; never follow directory symlinks."""
    result = {}
    for parent, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(parent) / name
            rel = str(path.relative_to(root))
            info = path.lstat()
            mode = stat.S_IMODE(info.st_mode)
            if path.is_symlink():
                result[rel] = ("link", mode, os.readlink(path))
            elif path.is_dir():
                result[rel] = ("dir", mode)
            elif path.is_file():
                result[rel] = ("file", mode, path.read_bytes())
            else:
                result[rel] = ("special", mode)
    return result


def make_tree(root):
    for rel in ("src/KOKKOS", "cmake/Modules/Packages", "doc/src", "examples/PACKAGES"):
        (root / rel).mkdir(parents=True)
    (root / "src/lammps.cpp").write_text("// root sentinel\n")
    (root / "cmake/CMakeLists.txt").write_text(CMAKE)


class InstallerUninstallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_tmp = tempfile.TemporaryDirectory(prefix="ye3t-uninstall-template.")
        cls.template = Path(cls.base_tmp.name) / "lammps"
        make_tree(cls.template)
        proc = run_command(SHELL, ROOT / "tools/patch_lammps.sh", "--apply",
                           "--lammps-source", cls.template)
        if proc.returncode:
            raise RuntimeError(proc.stdout + proc.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.base_tmp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ye3t uninstall test ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "LAMMPS source with spaces"
        shutil.copytree(self.template, self.root)
        self.patcher = ROOT / "tools/patch_lammps.sh"

    def patch(self, *args, success=True, env=None):
        proc = run_command(SHELL, self.patcher, *args, "--lammps-source", self.root, env=env)
        if success:
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        else:
            self.assertNotEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        return proc

    def backups(self):
        return list((self.root / ".ye3t-uninstall-backups").glob("uninstall.*"))

    def assert_removed(self):
        self.assertFalse((self.root / CPU).exists())
        self.assertFalse((self.root / GPU).exists())
        for rel in (PACKAGE / "ye3t_gpu_dag_schedule.h",
                    PACKAGE / "ye3t_gpu_block_schedule.h",
                    PACKAGE / "ye3t_gpu_harmonic_stream.h",
                    PACKAGE / "ye3t_gpu_tagged_source.h",
                    PACKAGE / "ye3t_tagged_cauchy_readout_plan.h",
                    Path("src/KOKKOS/ye3t_kokkos_step_state.h"),
                    Path("src/KOKKOS/ye3t_kokkos_types.h")):
            self.assertFalse((self.root / rel).exists())
        self.assertFalse((self.root / "cmake/Modules/Packages/ML-YE3T.cmake").exists())
        self.assertFalse((self.root / "doc/src/pair_ye3t.rst").exists())
        self.assertTrue((self.root / "doc/src").is_dir())
        self.assertNotIn("ML-YE3T", (self.root / "cmake/CMakeLists.txt").read_text())

    def assert_refuses_without_changes(self, *args):
        before = snapshot(self.root)
        proc = self.patch(*args, success=False)
        self.assertEqual(snapshot(self.root), before)
        return proc

    def test_round_trip_and_complete_backup(self):
        before = snapshot(self.root)
        self.patch("--uninstall")
        self.assert_removed()
        self.assertEqual((self.root / "cmake/CMakeLists.txt").read_text(), CMAKE)
        self.assertFalse((self.root / PACKAGE).exists())
        self.assertFalse((self.root / EXAMPLES).exists())
        self.assertTrue((self.root / "src/KOKKOS").is_dir())
        backup = self.backups()[0]
        removed = (backup / "removed-files.txt").read_text().splitlines()
        self.assertGreater(len(removed), 100)
        saved = snapshot(backup / "files")
        for rel in removed + ["cmake/CMakeLists.txt"]:
            self.assertEqual(saved[rel], before[rel], rel)
        self.patch("--check")
        self.patch("--apply")
        self.patch("--check")
        self.assertTrue((self.root / CPU).exists())

    def test_dry_run_is_read_only_and_previews_paths(self):
        before = snapshot(self.root)
        proc = self.patch("--uninstall", "--dry-run")
        self.assertIn(str(CPU), proc.stdout)
        self.assertIn(str(GPU), proc.stdout)
        self.assertIn("---", proc.stdout)
        self.assertEqual(snapshot(self.root), before)

    def test_repeated_uninstall_is_noop(self):
        self.patch("--uninstall")
        before = snapshot(self.root)
        proc = self.patch("--uninstall")
        self.assertIn("nothing to uninstall", proc.stdout)
        self.assertEqual(snapshot(self.root), before)
        self.assertEqual(len(self.backups()), 1)

    def test_modified_files_do_not_need_force_or_rehashing(self):
        modified = [CPU, GPU, EXAMPLES / "README.md",
                    Path("cmake/Modules/Packages/ML-YE3T.cmake")]
        for rel in modified:
            with (self.root / rel).open("a") as stream:
                stream.write("\nlocal modification\n")
        self.patch("--check", success=False)
        self.patch("--uninstall")
        backup = self.backups()[0]
        for rel in modified:
            self.assertFalse((self.root / rel).exists())
            self.assertIn("local modification", (backup / "files" / rel).read_text())
        self.patch("--apply")
        self.patch("--check")

    def test_preserves_unlisted_files_and_allows_reinstall(self):
        keep = [PACKAGE / "local-notes.txt", Path("src/KOKKOS/pair_other.cpp"),
                EXAMPLES / "log.lammps", EXAMPLES / "my-results" / "forces.dat",
                EXAMPLES / "cost_comparison" / "Ni" / "user-result.txt",
                Path("cmake/Modules/Packages/LOCAL.cmake")]
        for rel in keep:
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.root / rel).write_text(str(rel))
        self.patch("--uninstall")
        for rel in keep:
            self.assertEqual((self.root / rel).read_text(), str(rel))
        self.patch("--check")
        self.patch("--apply")
        self.patch("--check")
        for rel in keep:
            self.assertEqual((self.root / rel).read_text(), str(rel))

    def test_installed_inventory_removes_retired_filenames(self):
        retired = [(PACKAGE / "ye3t_retired.cpp", PMANIFEST, "ye3t_retired.cpp"),
                   (Path("src/KOKKOS/ye3t_retired_kokkos.h"), KMANIFEST,
                    "../KOKKOS/ye3t_retired_kokkos.h"),
                   (EXAMPLES / "old" / "retired.json", EMANIFEST, "old/retired.json")]
        for rel, manifest, name in retired:
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.root / rel).write_text("old version\n")
            # File hashes are intentionally stale: inventory, not integrity, is used.
            with (self.root / manifest).open("a") as stream:
                stream.write("0" * 64 + "  " + name + "\n")
        self.patch("--uninstall")
        for rel, _, _ in retired:
            self.assertFalse((self.root / rel).exists())
            self.assertTrue((self.backups()[0] / "files" / rel).is_file())

    def test_uninstall_does_not_require_package_sources(self):
        self.patcher = Path(self.temp.name) / "standalone patcher.sh"
        shutil.copy2(ROOT / "tools/patch_lammps.sh", self.patcher)
        self.patch("--uninstall")
        self.assert_removed()
        self.assertFalse((self.root / EXAMPLES).exists())

    def test_missing_sources_manifests_and_record(self):
        for rel in (CPU, GPU, PMANIFEST, KMANIFEST, EMANIFEST,
                    PACKAGE / "YE3T_LAMMPS_INSTALL.txt"):
            (self.root / rel).unlink()
        self.patch("--uninstall")
        self.assert_removed()
        self.assertFalse((self.root / PACKAGE).exists())
        self.assertFalse((self.root / EXAMPLES).exists())
        self.patch("--apply")
        self.patch("--check")

    def test_uninstall_with_missing_shared_directories(self):
        shutil.rmtree(self.root / "src/KOKKOS")
        shutil.rmtree(self.root / "cmake/Modules/Packages")
        shutil.rmtree(self.root / "examples/PACKAGES")
        self.patch("--uninstall")
        self.assert_removed()
        self.assertFalse((self.root / PACKAGE).exists())

    def test_partial_and_duplicate_cmake_registrations(self):
        cmake = self.root / "cmake/CMakeLists.txt"
        cmake.write_text(cmake.read_text().replace("  ML-YE3T\n", "  ML-YE3T\n  ML-YE3T\n")
                         .replace("ML-PACE ML-YE3T LEPTON", "ML-PACE LEPTON"))
        self.patch("--check", success=False)
        self.patch("--uninstall")
        self.assertEqual(cmake.read_text(), CMAKE)
        self.patch("--apply")

    def test_absent_cmake_registration_with_existing_sources(self):
        cmake = self.root / "cmake/CMakeLists.txt"
        cmake.write_text(CMAKE)
        self.patch("--uninstall")
        self.assert_removed()
        self.assertEqual(cmake.read_text(), CMAKE)

    def test_cmake_only_partial_installation(self):
        shutil.rmtree(self.root / PACKAGE)
        shutil.rmtree(self.root / "src/KOKKOS")
        shutil.rmtree(self.root / EXAMPLES)
        (self.root / "cmake/Modules/Packages/ML-YE3T.cmake").unlink()
        self.patch("--uninstall")
        self.assertEqual((self.root / "cmake/CMakeLists.txt").read_text(), CMAKE)
        self.assertTrue((self.backups()[0] / "files/cmake/CMakeLists.txt").is_file())

    def test_preserves_unrelated_cmake_edits_comments_and_substrings(self):
        cmake = self.root / "cmake/CMakeLists.txt"
        suffix = "\n# retain this ML-YE3T comment\nset(MY_LOCAL_OPTION ON)\n"
        text = cmake.read_text().replace("  ML-SNAP)", "  ML-YE3T-OTHER\n  ML-SNAP)")
        text = text.replace("  ML-YE3T\n", "  ML-YE3T # local comment\n") + suffix
        cmake.write_text(text)
        expected = CMAKE.replace("  ML-SNAP)", "  ML-YE3T-OTHER\n  ML-SNAP)")
        expected = expected.replace("  ML-PACE\n", "  ML-PACE\n  # local comment\n") + suffix
        self.patch("--uninstall")
        self.assertEqual(cmake.read_text(), expected)

    def test_cmake_multiline_lists(self):
        cmake = self.root / "cmake/CMakeLists.txt"
        text = ("set(STANDARD_PACKAGES ML-PACE ML-YE3T ML-SNAP)\n"
                "foreach(PKG_WITH_INCL\n  ML-PACE\n  ML-YE3T\n  LEPTON)\nendforeach()\n")
        cmake.write_text(text)
        self.patch("--uninstall")
        self.assertEqual(cmake.read_text(), text.replace(" ML-YE3T", "", 1)
                         .replace("  ML-YE3T\n", ""))

    def test_clean_no_final_newline_does_not_create_backup(self):
        self.patch("--uninstall")
        cmake = self.root / "cmake/CMakeLists.txt"
        cmake.write_text(CMAKE.rstrip("\n"))
        before = snapshot(self.root)
        self.patch("--uninstall")
        self.assertEqual(snapshot(self.root), before)
        self.assertEqual(len(self.backups()), 1)

    def test_uninstall_preserves_cmake_permissions(self):
        cmake = self.root / "cmake/CMakeLists.txt"
        cmake.chmod(0o640)
        self.patch("--uninstall")
        self.assertEqual(stat.S_IMODE(cmake.stat().st_mode), 0o640)

    def test_unsafe_manifests_refused_without_deletion(self):
        cases = [(PMANIFEST, "../../lammps.cpp"), (PMANIFEST, "/tmp/target"),
                 (PMANIFEST, "../KOKKOS/pair_other.cpp"),
                 (KMANIFEST, "../KOKKOS/pair_pace_kokkos.cpp"),
                 (KMANIFEST, "../KOKKOS/../../other.cpp"),
                 (EMANIFEST, "../../../src/lammps.cpp"),
                 (EMANIFEST, "nested//bad"), (EMANIFEST, "./nested/../bad"),
                 (EMANIFEST, "*.txt"), (EMANIFEST, "has space"),
                 (EMANIFEST, "nested/./bad")]
        for manifest, path in cases:
            with self.subTest(path=path):
                original = (self.root / manifest).read_bytes()
                (self.root / manifest).write_text("0" * 64 + "  " + path + "\n")
                proc = self.assert_refuses_without_changes("--uninstall")
                self.assertIn("inventory", proc.stderr)
                (self.root / manifest).write_bytes(original)

    def test_duplicate_and_malformed_manifest_refused(self):
        manifest = self.root / PMANIFEST
        original = manifest.read_text()
        for text in (original + original.splitlines()[0] + "\n", "bogus checksum\n"):
            manifest.write_text(text)
            self.assert_refuses_without_changes("--uninstall")
        manifest.write_text(original)

    def test_optional_dot_prefix_and_binary_checksum_marker(self):
        manifest = self.root / PMANIFEST
        lines = []
        for line in manifest.read_text().splitlines():
            lines.append(line[:64] + " *./" + line[66:])
        manifest.write_text("\n".join(lines) + "\n")
        self.patch("--uninstall")
        self.assert_removed()

    def test_directory_symlink_refused_and_external_tree_untouched(self):
        for rel in (Path("src/KOKKOS"), PACKAGE, Path("cmake/Modules"), EXAMPLES):
            with self.subTest(path=rel):
                outside = Path(self.temp.name) / ("outside-" + rel.name)
                original = self.root / rel
                original.rename(outside)
                original.symlink_to(outside, target_is_directory=True)
                outside_before = snapshot(outside)
                self.assert_refuses_without_changes("--uninstall")
                self.assertEqual(snapshot(outside), outside_before)
                original.unlink()
                outside.rename(original)

    def test_nested_example_directory_symlink_refused(self):
        rel = EXAMPLES / "cost_comparison" / "Ni"
        outside = Path(self.temp.name) / "outside-nested"
        (self.root / rel).rename(outside)
        (self.root / rel).symlink_to(outside, target_is_directory=True)
        before = snapshot(outside)
        self.assert_refuses_without_changes("--uninstall")
        self.assertEqual(snapshot(outside), before)

    def test_manifest_symlink_refused(self):
        manifest = self.root / PMANIFEST
        saved = Path(self.temp.name) / "saved-manifest"
        manifest.rename(saved)
        manifest.symlink_to(saved)
        self.assert_refuses_without_changes("--uninstall")

    def test_cmake_symlink_refused(self):
        cmake = self.root / "cmake/CMakeLists.txt"
        saved = Path(self.temp.name) / "saved-cmake"
        cmake.rename(saved)
        cmake.symlink_to(saved)
        self.assert_refuses_without_changes("--uninstall")

    def test_leaf_symlink_is_backed_up_and_unlinked_not_followed(self):
        outside = Path(self.temp.name) / "external-source"
        outside.write_text("do not change me\n")
        (self.root / CPU).unlink()
        (self.root / CPU).symlink_to(outside)
        (self.root / GPU).unlink()
        (self.root / GPU).symlink_to(Path(self.temp.name) / "missing-target")
        self.patch("--uninstall")
        self.assertEqual(outside.read_text(), "do not change me\n")
        self.assertFalse((self.root / CPU).is_symlink())
        self.assertFalse((self.root / GPU).is_symlink())
        saved = self.backups()[0] / "files"
        self.assertTrue((saved / CPU).is_symlink())
        self.assertTrue((saved / GPU).is_symlink())
        self.assertEqual(os.readlink(saved / CPU), str(outside))

    def test_special_file_or_directory_at_owned_path_refused(self):
        path = self.root / CPU
        path.unlink()
        path.mkdir()
        self.assert_refuses_without_changes("--uninstall")
        path.rmdir()
        os.mkfifo(path)
        self.assert_refuses_without_changes("--uninstall")

    def test_backup_directory_symlink_refused(self):
        outside = Path(self.temp.name) / "outside-backup"
        outside.mkdir()
        (self.root / ".ye3t-uninstall-backups").symlink_to(outside, target_is_directory=True)
        self.assert_refuses_without_changes("--uninstall")
        self.assertEqual(list(outside.iterdir()), [])

    def test_backup_copy_failure_does_not_remove_sources(self):
        wrappers = Path(self.temp.name) / "fake-bin"
        wrappers.mkdir()
        real_cp = shutil.which("cp")
        self.assertIsNotNone(real_cp)
        fake_cp = wrappers / "cp"
        fake_cp.write_text("#!/bin/sh\ncase \"$*\" in *pair_ye3t.cpp*) exit 77;; esac\n"
                           f"exec '{real_cp}' \"$@\"\n")
        fake_cp.chmod(0o755)
        env = dict(os.environ, PATH=str(wrappers) + os.pathsep + os.environ["PATH"])
        before = snapshot(self.root)
        self.patch("--uninstall", success=False, env=env)
        after = {k: v for k, v in snapshot(self.root).items()
                 if not k.startswith(".ye3t-uninstall-backups")}
        self.assertEqual(after, before)

    def test_failed_removal_keeps_inventory_for_retry(self):
        retired = PACKAGE / "ye3t_zzz_retired.cpp"
        (self.root / retired).write_text("retired file\n")
        with (self.root / PMANIFEST).open("a") as stream:
            stream.write("0" * 64 + "  ye3t_zzz_retired.cpp\n")
        wrappers = Path(self.temp.name) / "fake-rm-bin"
        wrappers.mkdir()
        real_rm = shutil.which("rm")
        self.assertIsNotNone(real_rm)
        fake_rm = wrappers / "rm"
        fake_rm.write_text("#!/bin/sh\ncase \"$*\" in *pair_ye3t.cpp*) exit 77;; esac\n"
                           f"exec '{real_rm}' \"$@\"\n")
        fake_rm.chmod(0o755)
        env = dict(os.environ, PATH=str(wrappers) + os.pathsep + os.environ["PATH"])
        self.patch("--uninstall", success=False, env=env)
        for rel in (PMANIFEST, KMANIFEST, EMANIFEST):
            self.assertTrue((self.root / rel).is_file())
        self.patch("--uninstall")
        self.assertFalse((self.root / retired).exists())
        self.assert_removed()
        self.patch("--apply")
        self.patch("--check")

    def test_develop_multiline_include_and_separate_suffix_list(self):
        # Reduced structural fixture for the two PKG_WITH_INCL groups in
        # public LAMMPS develop; not a complete LAMMPS configure/build test.
        self.patch("--uninstall")
        cmake = self.root / "cmake/CMakeLists.txt"
        text = CMAKE.replace(
            "foreach(PKG_WITH_INCL GRAPHICS ML-IAP COMPRESS ML-PACE LEPTON FENIX)",
            "foreach(PKG_WITH_INCL GRAPHICS ML-IAP\n"
            "        COMPRESS ML-PACE LEPTON FENIX)")
        text += "\nforeach(PKG_WITH_INCL CORESHELL OPENMP KOKKOS GPU)\nendforeach()\n"
        cmake.write_text(text)
        self.patch("--apply")
        self.patch("--check")
        self.patch("--uninstall")
        self.assertEqual(cmake.read_text(), text)

    def test_apply_refuses_colliding_output_after_uninstall(self):
        self.patch("--uninstall")
        collision = self.root / EXAMPLES / "README.md"
        collision.parent.mkdir(parents=True)
        collision.write_text("user file with an installed filename\n")
        self.assert_refuses_without_changes("--apply")

    def test_apply_refuses_dangling_destination_link(self):
        self.patch("--uninstall")
        path = self.root / CPU
        path.parent.mkdir(parents=True)
        path.symlink_to(Path(self.temp.name) / "nonexistent")
        self.assert_refuses_without_changes("--apply")

    def test_no_git_needed_during_uninstall(self):
        wrappers = Path(self.temp.name) / "no-git"
        wrappers.mkdir()
        marker = Path(self.temp.name) / "git-invoked"
        fake_git = wrappers / "git"
        fake_git.write_text(f"#!/bin/sh\ntouch '{marker}'\nexit 99\n")
        fake_git.chmod(0o755)
        env = dict(os.environ, PATH=str(wrappers) + os.pathsep + os.environ["PATH"])
        self.patch("--uninstall", env=env)
        self.assertFalse(marker.exists())
        self.assert_removed()

    @unittest.skipUnless(shutil.which("git"), "git required for tracked-file/index regression")
    def test_tracked_sources_deleted_without_changing_git_index(self):
        for args in (("init", "-q"), ("config", "user.email", "test@example.invalid"),
                     ("config", "user.name", "Installer test"), ("add", "."),
                     ("commit", "-qm", "installed fixture")):
            proc = run_command("git", "-C", self.root, *args)
            self.assertEqual(proc.returncode, 0, proc.stderr)
        cmake = self.root / "cmake/CMakeLists.txt"
        suffix = "\nset(USER_STAGED_OPTION ON)\n"
        cmake.write_text(cmake.read_text() + suffix)
        self.assertEqual(run_command("git", "-C", self.root, "add", "cmake/CMakeLists.txt").returncode, 0)
        indexed = run_command("git", "-C", self.root, "show", ":cmake/CMakeLists.txt").stdout
        self.patch("--uninstall")
        self.assert_removed()
        self.assertEqual(cmake.read_text(), CMAKE + suffix)
        self.assertEqual(run_command("git", "-C", self.root, "show",
                                     ":cmake/CMakeLists.txt").stdout, indexed)
        deleted = run_command("git", "-C", self.root, "ls-files", "--deleted").stdout.splitlines()
        self.assertIn(str(CPU), deleted)
        self.assertIn(str(GPU), deleted)
        self.patch("--apply")
        self.patch("--check")

    def test_cli_rejects_conflicting_modes_and_unsupported_dry_run(self):
        for args in (("--apply", "--uninstall"), ("--uninstall", "--check"),
                     ("--apply", "--dry-run"), ("--check", "--dry-run")):
            with self.subTest(args=args):
                self.assert_refuses_without_changes(*args)


if __name__ == "__main__":
    unittest.main()
