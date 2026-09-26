# Tools

## `patch_lammps.sh`: install the package into a LAMMPS source tree

```text
tools/patch_lammps.sh --check     --lammps-source /path/to/lammps
tools/patch_lammps.sh --apply     --lammps-source /path/to/lammps
tools/patch_lammps.sh --uninstall [--dry-run] --lammps-source /path/to/lammps
```

`--check` validates the target tree without changing it. `--apply` installs
the CPU sources in `src/ML-YE3T` (including the package `README`), the Kokkos
sources in `src/KOKKOS`, the CMake package module, the documentation page
`doc/src/pair_ye3t.rst`, and the examples in `examples/PACKAGES/ye3t` with
their checksum manifests, and writes an install record. It is idempotent for
an identical installation and refuses partial, modified, or conflicting
files. The installer verifies the example checksum manifest, so after editing
anything under `examples/PACKAGES/ye3t` run `tools/refresh_example_manifest.py`.

`--uninstall` removes the installed CPU files, the YE3T-only Kokkos files,
the CMake module, the documentation page, the integrity records, and the
manifest-listed example files by explicit filename. It uses the installed
manifests, so retired filenames are removed even when the package revision
has changed, and file contents need not match their old checksums: locally
edited files are backed up and removed. Missing files are tolerated, and the
current filenames provide recovery when an installed manifest is missing.

Before deleting anything, the script copies every file it will remove and
the original `cmake/CMakeLists.txt` to a unique directory below
`.ye3t-uninstall-backups/` in the LAMMPS tree and prints that location.
The CMake edit removes only the `ML-YE3T` package-list registrations and
preserves unrelated edits; no git operation is performed, and non-git trees
work. Unlisted files such as custom inputs, logs, and notes are preserved,
and only empty package and example directories are removed. Unsafe manifest
paths, directory symlinks, symlinked inventories, and directories or special
files in place of expected payload files are refused before deletion.

Uninstall is repeatable and needs neither the YE3T runtime nor the current
package sources. It changes the source integration only: build directories,
installed executables, and runtime dependencies are untouched, so reconfigure
and rebuild afterwards. Do not run it concurrently with a build or another
installer. Deletion is not atomic, but inventories remain until payload
removal and the CMake edit succeed, so a partial failure can be retried.

## Standalone CMake project

The repository's own `CMakeLists.txt` is a developer project for the CPU
model evaluator, its tests, and an optional loadable plugin; the LAMMPS
source-package route above does not use it.

| Option | Default | Purpose |
| --- | --- | --- |
| `ML_YE3T_RUNTIME_SOURCE` | | path to a `ye3t` checkout providing `ye3t/runtime/csrc` (supported route) |
| `ML_YE3T_RUNTIME_ROOT` | | installed `ye3t` runtime prefix, an alternative to the source route |
| `ML_YE3T_BUILD_CPU_RUNTIME` | `ON` | build the standalone CPU model evaluator |
| `ML_YE3T_BUILD_NATIVE_CPU_TESTS` | `OFF` | build the native-CPU evaluator tests (see `tests/native_cpu/README.md`) |
| `ML_YE3T_BUILD_LAMMPS_PLUGIN` | `OFF` | build the loadable PairYE3T CPU plugin |
| `ML_YE3T_LAMMPS_SOURCE` | | LAMMPS `src` directory whose headers the plugin compiles against |
| `ML_YE3T_LAMMPS_SIZES` | | `smallbig` or `bigbig`, matching the host LAMMPS build |
| `ML_YE3T_LAMMPS_MPI` | `OFF` | build the plugin for an MPI LAMMPS host instead of the serial stubs |
| `ML_YE3T_ENABLE_NATIVE_CPU` | `OFF` | compile for the build machine's CPU |
| `ML_YE3T_ENABLE_IPO` | `OFF` | enable interprocedural optimization |
| `ML_YE3T_BUILD_PACE_ORACLE` | `OFF` | build the PACE oracle used by the comparison tests |

The plugin host must include the LAMMPS PLUGIN package, and an MPI plugin
must use the same MPI implementation and toolchain as the host.

## Kokkos type names

`src/KOKKOS/ye3t_kokkos_types.h` maps both the 22 Jul 2025 stable Kokkos API
(`X_FLOAT`/`F_FLOAT`/`E_FLOAT` and `t_x_array`/`t_f_array`/`t_efloat_1d`/
`t_virial_array`) and the renamed post-refactor API (`KK_FLOAT`/`KK_ACC_FLOAT`
and `t_kkfloat_*`/`t_kkacc_*`) onto one alias set, selected by the LAMMPS
precision macros. Double precision is required and checked at compile time.
Any new Kokkos source file must be added to the installer's Kokkos file list
and to `tests/test_patch_lammps.sh`.

## Other tools

- `refresh_example_manifest.py`: regenerate
  `examples/PACKAGES/ye3t/YE3T_EXAMPLE_MANIFEST.sha256`.
- `verify_cost_comparison_bundle.py`: verify every model byte of the
  cost-comparison examples against the promoted results bundle; run from the
  repository root as shown in `examples/PACKAGES/ye3t/cost_comparison/README.md`.
- `gpu_scaling/`: GPU calibration, replay generation, and 1/2/4-GPU scaling
  runs (`run_gpu_scaling.py`, `gpu_rank_wrapper.py`, `calibrate_kokkos_auto.py`,
  `make_kokkos_auto_replay.py`, `gpu_scaling_cases.json`, and their input
  decks). `run_gpu_scaling.py` requires `nvidia-smi`, maps one GPU per MPI
  rank, and refuses oversubscription in performance mode. A replay is bound
  to the executable, model, plan, GPU class, driver and runtime versions,
  Kokkos layout, and measured center-count envelope, and is rejected on a
  different device class.
- `pace_oracle/`: the PACE reference oracle used by the comparison tests.
