# ye3t-lammps

`ye3t-lammps` provides the `ML-YE3T` source package for LAMMPS: the CPU
`pair_style ye3t` and the FP64 Kokkos `pair_style ye3t/kk`. Models are
trained and exported outside LAMMPS with `ye3t` and `ye3t-ace`; LAMMPS only
loads the resulting standard `.yace` potential, a native YE3T bundle, or a
hash-bound composite that joins an ordinary backbone and a tagged correction.
`ye3t-ace` (fitting and export) is not yet publicly released; every shipped
example runs from its pre-exported `.yace`, `.ye3t.json`, or plan bundle and
does not need it.

The package supports standard linear PACE-compatible `.yace` models,
native tagged-Cauchy and lifted-Cauchy linear YE3T bundles, multi-element
type maps, MPI domain decomposition, and per-atom energy and virial.
Ordinary `.yace` models expose the following evaluators.

| policy | input | purpose |
|---|---|---|
| `direct` | standard `.yace` | exact shared product-DAG compatibility path |
| `block` | `.yace` plus plan bundle | force compiler-certified block descriptors |
| `auto` | `.yace` plus plan bundle | choose an exact catalogue-level plan for the active backend |
| `scalar_power` | supported plan bundle | force certified homogeneous scalar powers |
| `coupled_product` | supported plan bundle | force certified coupled-product DAGs |

The standard `.yace` path is standalone and is the backwards-compatible route
for reproducing `pair_style pace product`. Execution plans are validated and
lowered once during `pair_coeff`; they do not add JSON parsing or hashing to
the timestep hot path. The CPU pair style evaluates complete environments in
8/16-center batches with a shared harmonic recurrence; the Kokkos path uses
owner-gather direct adjoints, tiled block execution, and the same recurrence.
Both require the `ye3t` runtime that provides
`complex_spherical_harmonics_nonnegative_unit_recurrence_with_derivative_prevalidated`.

The full keyword syntax, restrictions, and defaults are documented in
[doc/src/pair_ye3t.rst](doc/src/pair_ye3t.rst), which the installer copies
into the LAMMPS documentation tree.

## Performance status

Performance is hardware- and workload-specific and the retained measurements
are engineering results, not portable claims. The compact evidence bundles
under [docs/results](docs/results) record the conditions of every retained
timing. The current six-element CPU comparison of PACE `product`,
YE3T-symmetric AUTO on byte-identical ordinary `.yace` files, and
YE3T-mixed tagged models is in
[docs/results/cost_comparison_three_way_auto_v3_20260921](docs/results/cost_comparison_three_way_auto_v3_20260921/README.md).
Every ordinary AUTO timing there loads a compiler-produced nonempty candidate
portfolio and rejects fallback states; the retained portfolios resolve to
all-direct schedules on that CPU after candidate scoring, which is a measured
planner choice rather than a fallback. Every YE3T timing uses
`pair_style ye3t` without PairPACE. The tagged Kokkos path is a correctness
reference and carries no performance claim.

YE3T chooses a memory-safe internal center capacity with an initial target of
4096 centers and reduces it when the model workspace or device headroom
requires that. Normal users do not need to set `chunksize`; the explicit option
is retained only as an expert override and AUTO replays bind the resolved
value used during calibration.

## Requirements

- Linux with a C++17 compiler
- CMake 3.20 or newer
- MPI for multi-process LAMMPS runs
- `yaml-cpp` with CMake package metadata
- a `ye3t` source checkout containing `ye3t/runtime/csrc`
- Ninja and ccache are optional but recommended
- a Kokkos-supported accelerator toolchain for `ye3t/kk`; CUDA is the
  currently validated device backend

For the optional CMake plugin build (`ML_YE3T_BUILD_LAMMPS_PLUGIN=ON`),
set `ML_YE3T_LAMMPS_SOURCE` to the exact host LAMMPS headers and match
`ML_YE3T_LAMMPS_SIZES`. An MPI host additionally requires
`ML_YE3T_LAMMPS_MPI=ON` and the same MPI implementation/toolchain as the host.
The default `OFF` uses LAMMPS serial stubs; selecting `mpicxx` alone does not
change that mode. The host must include the LAMMPS PLUGIN package. These
options affect only the standalone plugin; the source-package installation
inherits the host's build settings.

For Debian or Ubuntu, a typical dependency installation is:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build ccache \
  libyaml-cpp-dev libopenmpi-dev openmpi-bin python3
```

## Install into LAMMPS

The tested reference is the LAMMPS stable release `stable_22Jul2025_update4`
(revision `611ca3b7f8525ba802373b04f9e3d632d515f7e3`). The patcher checks
semantic CMake anchors and fails without modifying unsupported layouts.

`pair_style ye3t/kk` selects the LAMMPS Kokkos precision and array type API
at compile time through `src/KOKKOS/ye3t_kokkos_types.h`. On the 22 Jul 2025
stable series it uses the `X_FLOAT`/`F_FLOAT`/`E_FLOAT` scalars and
`t_x_array`/`t_f_array`/`t_efloat_1d`/`t_virial_array` views; on development
revisions after the Kokkos accumulator-type refactor, for example
`3214c5b1a0e8ad562e436e9a81ec5e8e4883f61a` (2 Sep 2026), it uses
`KK_FLOAT`/`KK_ACC_FLOAT` and the `t_kkfloat_*`/`t_kkacc_*` views. Both
trees have been built with `PKG_KOKKOS` and CUDA and pass the device smoke
test; double precision (`LMP_PRECISION=2` or `LMP_KOKKOS_DOUBLE_DOUBLE`) is
required and enforced at compile time.

Every example that exercises `pair_style ye3t` runs without ML-PACE: tagged
models load a self-contained composite manifest that binds the ordinary
backbone and the tagged correction. `pair_style pace` appears only in the
explicit PACE product/recursive comparison inputs.

```bash
git clone https://github.com/lammps/lammps.git
git -C lammps checkout stable_22Jul2025_update4

git clone https://github.com/ye3t-equivariance/ye3t.git
git clone https://github.com/ye3t-equivariance/ye3t-lammps.git

./ye3t-lammps/tools/patch_lammps.sh \
  --check --lammps-source "$PWD/lammps"
./ye3t-lammps/tools/patch_lammps.sh \
  --apply --lammps-source "$PWD/lammps"
```

`--apply` installs the package sources in `lammps/src/ML-YE3T`, the CMake
module, the documentation page `lammps/doc/src/pair_ye3t.rst`, and the
runnable examples in `lammps/examples/PACKAGES/ye3t`. It is idempotent for an
identical installation and refuses partial, modified, or conflicting files.
After editing or updating `ye3t-lammps`, uninstall the old integration and
apply the new one to the **same LAMMPS tree**:

```bash
# Optional read-only preview.
./ye3t-lammps/tools/patch_lammps.sh \
  --uninstall --dry-run --lammps-source "$PWD/lammps"

./ye3t-lammps/tools/patch_lammps.sh \
  --uninstall --lammps-source "$PWD/lammps" &&
./ye3t-lammps/tools/patch_lammps.sh \
  --apply --lammps-source "$PWD/lammps"
```

`--uninstall` removes the installed CPU files, YE3T-only Kokkos files, CMake
package module, documentation page, integrity records, and manifest-listed
example files by explicit filename. It uses the **installed** manifests, so
retired filenames are removed even when the incoming package revision has
changed. File contents need not match their old checksums: locally edited
files are backed up and removed, without a `--force` option or manifest
regeneration. Missing files are tolerated, and the known current filenames
provide recovery when an installed manifest is missing. If both an installed
example manifest and the current example manifest are unavailable,
unidentified examples are left in place.

Before deleting anything, the script copies all files being removed and the
original `cmake/CMakeLists.txt` to a unique directory below
`lammps/.ye3t-uninstall-backups/`, and prints that location. Backups are retained
until you remove them. The CMake edit removes only the `ML-YE3T` package-list
registrations, preserving unrelated edits; **no Git restoration, cleaning,
staging, or index changes are performed**. Both tracked and untracked installed
source files are removed. This also works with non-Git LAMMPS source trees.

Unlisted files such as custom inputs, logs, and notes are preserved. Only empty
package/example directories are removed; a later `--apply` accepts preserved
files but still refuses to overwrite a conflicting installation destination.
Unsafe manifest paths, directory symlinks, symlinked inventories/CMake input,
and directories or special files in place of expected payload files are refused
before deletion. A source-file symlink itself can be backed up and unlinked
without following its target.

Uninstall is repeatable and needs neither the YE3T runtime nor the current
package source contents. It changes the **source integration only**: it does
not remove build directories, installed executables, runtime dependencies, or
the Git index. Reconfigure/rebuild after installing the new sources. Do not run
an uninstall concurrently with a build or another installer. Backups finish
before source changes; deletion is not an atomic transaction, but inventories
remain until payload removal and the CMake edit succeed, permitting a retry
after a partial failure.

Configure and build the supported source-runtime route:

```bash
cmake -S "$PWD/lammps/cmake" -B "$PWD/lammps-build" -G Ninja \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -D BUILD_MPI=ON \
  -D PKG_ML-YE3T=ON \
  -D PKG_EXTRA-FIX=ON \
  -D ML_YE3T_RUNTIME_SOURCE="$PWD/ye3t"
cmake --build "$PWD/lammps-build" --parallel 4
```

`PKG_EXTRA-FIX` is only needed for the numerical-differentiation examples.
Omit `-G Ninja` and the ccache launcher if those tools are unavailable.

### CUDA/Kokkos build

The GPU evaluators require FP64 Kokkos and a device architecture selected for
the target machine. The validated layout is the Kokkos legacy layout; the
selected layout is recorded in and enforced by calibrated AUTO replays. For
example, an NVIDIA Ada 8.9 build uses:

```bash
cmake -S "$PWD/lammps/cmake" -B "$PWD/lammps-build-cuda" -G Ninja \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_COMPILER="$PWD/lammps/lib/kokkos/bin/nvcc_wrapper" \
  -D CMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -D BUILD_MPI=ON \
  -D PKG_KOKKOS=ON \
  -D PKG_ML-YE3T=ON \
  -D PKG_EXTRA-FIX=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ENABLE_SERIAL=ON \
  -D Kokkos_ARCH_ADA89=ON \
  -D KOKKOS_PREC=double \
  -D KOKKOS_LAYOUT=legacy \
  -D ML_YE3T_RUNTIME_SOURCE="$PWD/ye3t"
cmake --build "$PWD/lammps-build-cuda" --parallel 4
```

Replace `Kokkos_ARCH_ADA89` with the architecture documented by Kokkos for
the destination GPU. The pinned Kokkos source identifies NVIDIA H100 as
`-D Kokkos_ARCH_HOPPER90=ON`. Add `PKG_ML-PACE=ON` and `LOCAL_ML-PACE` from the
comparison build below to put `pace/kk product` and `ye3t/kk` in one
binary. Confirm both registrations:

```bash
./lammps-build-cuda/lmp -help | grep -E 'pace/kk|ye3t/kk'
```

`pace/kk` does not require `KOKKOS_LAYOUT=legacy`; the pinned LAMMPS source
supports both `legacy` and `default` for ML-PACE and ML-YE3T. ML-IAP currently
has a separate default-layout restriction. A default-layout comparison should
therefore use one binary with ML-IAP disabled, and it requires a fresh YE3T
AUTO calibration because layout is part of the replay identity. PairYE3T's
large private workspaces are manually flattened rank-one Views, so changing
this global option alone does not change their logical index order.

Confirm that LAMMPS registered the style:

```bash
./lammps-build/lmp -help | grep -w ye3t
```

### Optional ML-PACE comparison build

The side-by-side examples use the same `.yace` bytes with `pace product` and
`ye3t`. Enable ML-PACE in the same LAMMPS binary:

```bash
git clone --branch v.2025.12.4.p1 \
  https://github.com/ICAMS/lammps-user-pace.git lammps-user-pace

cmake -S "$PWD/lammps/cmake" -B "$PWD/lammps-build" -G Ninja \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -D BUILD_MPI=ON \
  -D PKG_ML-YE3T=ON \
  -D PKG_ML-PACE=ON \
  -D PKG_EXTRA-FIX=ON \
  -D LOCAL_ML-PACE="$PWD/lammps-user-pace" \
  -D ML_YE3T_RUNTIME_SOURCE="$PWD/ye3t"
cmake --build "$PWD/lammps-build" --parallel 4
```

If `LOCAL_ML-PACE` is omitted, LAMMPS may download its pinned PACE release at
configure time.

## Pair-style syntax

A standard `.yace` model needs no YE3T sidecar:

```lammps
units metal
newton on
pair_style ye3t
pair_coeff * * model.yace Ta
```

For an exact compiled plan:

```lammps
pair_style ye3t plan /path/to/manifest.json block_policy auto
pair_coeff * * /path/to/model.yace Ta
```

CPU `auto` selects among the plan's candidate evaluators at `pair_coeff`. Small
portfolios are scored exhaustively; wide catalogues (hundreds of candidate
functions) use a coordinate descent that ranks single-function moves with
precomputed per-function marginal costs and recompiles the exact residual
only for the move it accepts, so load time grows with the number of
accepted moves rather than with the number of candidates. The `pair_coeff`
log line reports the planner algorithm and its exact score count.

GPU `auto` is conservative until it is given a replay generated on the target
GPU class. A replay is parsed and hash-checked at `pair_coeff`; no calibration,
JSON parsing, or hashing occurs in the timestep hot path:

```lammps
pair_style ye3t plan /path/to/manifest.json block_policy auto \
  auto_replay /path/to/kokkos_auto_replay.json chunksize 4096
pair_coeff * * /path/to/model.yace Ta
```

The element names after the model map LAMMPS atom types in order, exactly as
for `pair_style pace`. The CPU implementation currently requires `newton on`
and a full centered neighbor list.

### Tagged-Cauchy models

Tagged-Cauchy bundles retain compiler-owned tag-placement, intermediate
permutation/rotation, physical-image, source, and transpose-adjoint metadata
that cannot be represented faithfully by a standard `.yace` file:

```lammps
units metal
newton on
pair_style ye3t model_family tagged_cauchy block_policy direct
pair_coeff * * /path/to/model.ye3t.json Ta
```

`ye3t_tagged_cauchy_composite_v1` manifests bind an ordinary `.yace` backbone
and a tagged correction by SHA-256. Both are evaluated inside
`pair_style ye3t`; no `pair_style pace` instance is required. For legacy bundles
without bound references, restore the subtracted ZBL reference with
`pair_style hybrid/overlay ye3t ... zbl ...`; the fitted element models in
`examples/PACKAGES/ye3t/cost_comparison` show the exact decks. Composite
tagged execution is currently CPU-only; `ye3t/kk` rejects it explicitly
instead of silently omitting the ordinary component. The tagged Kokkos path
for native bundles is a direct correctness reference with fixed `l<=8`,
128-component, and 32-factors-per-term limits and has no block, AUTO,
performance, or multi-GPU scaling claim.

General `ye3t_tagged_cauchy_slice_v4` models support pair-specific descriptor
cutoffs and artifact-bound atomic references/ZBL on the CPU. Use one
`pair_style ye3t model_family tagged_cauchy block_policy direct`: **do not add
a ZBL overlay** when `reference_terms.zbl` is present. The native V4 path
includes its energy, forces, and virial. V4 is explicitly rejected by Kokkos
until the pair-specific source/reference implementation is qualified there.

The portable workflow that counts, fits, exports, and replays a tagged model
lives in the `ye3t-ace` fitting package (not yet public). Models fitted to
ZBL-subtracted targets must be deployed with the identical ZBL component using
`pair_style hybrid/overlay`; adding ZBL after fitting a total-target model
double counts the repulsion.

### Lifted-Cauchy models

`model_family lifted_cauchy` loads the native lifted-Cauchy bundle format
exported by `ye3t-ace` (`model.ye3t.json` plus its checksummed components).
The runtime, its tests, and its fixtures are part of the package, but no
lifted-density model ships as a public example: the linear YE3T examples use
the tagged basis or ordinary density-projected features.

## Runnable examples

After patching, change to the installed example directory:

```bash
cd lammps/examples/PACKAGES/ye3t
```

The example set contains:

- a physically qualified 58-function Ta rank-through-8 model with
  `direct`, `block`, and `auto` execution;
- a physically qualified 61-function Ta rank-through-8 model using `direct`
  and a certified coupled-product candidate with exact direct residual;
- identical-model `pace product`, bounded `pace recursive`, and `ye3t`
  energy/force/virial checks;
- one- and multi-rank MPI checks, including an atom crossing a domain;
- LAMMPS `fix numdiff` and `fix numdiff/virial` checks;
- a six-evaluation fixed-position replay across CPU, Kokkos, and PACE;
- a short deterministic NVE run;
- a rank-16 execution fixture; and
- fitted Li, Mo, Cu, Ni, Si, and Ge linear ACE controls and tagged YE3T
  models with PACE/YE3T same-model, tagged, finite-difference, and NVE inputs.

`run_examples.sh` takes no command-line arguments; every switch is an
environment variable documented at the top of the script. It runs the Ta
decks and the rank-16 fixture; the fitted Li/Mo/Cu/Ni/Si/Ge decks are run
individually from each `cost_comparison/<element>` directory as described in
`cost_comparison/README.md`. Run the fitted Ta examples on one process:

```bash
YE3T_LMP=/absolute/path/to/lammps-build/lmp ./run_examples.sh
```

Run the same matrix on four MPI processes and compare against ML-PACE:

```bash
YE3T_LMP=/absolute/path/to/lammps-build/lmp YE3T_MPI_RANKS=4 \
  YE3T_WITH_PACE=yes ./run_examples.sh
```

Add finite differences and the rank-16 execution check:

```bash
YE3T_LMP=/absolute/path/to/lammps-build/lmp YE3T_WITH_PACE=yes \
  YE3T_WITH_NUMDIFF=yes YE3T_WITH_HIGH_RANK=yes ./run_examples.sh
```

Run the CUDA/Kokkos direct, block, coupled-product, and catalogue-global AUTO
correctness lanes and compare against `pace/kk product` in the same binary:

```bash
YE3T_LMP=/absolute/path/to/lammps-build-cuda/lmp YE3T_WITH_KOKKOS=yes \
  YE3T_WITH_PACE=yes YE3T_WITH_NUMDIFF=yes YE3T_WITH_HIGH_RANK=yes \
  ./run_examples.sh
```

For a single-node one-rank-per-GPU correctness run, set matching values:

```bash
YE3T_LMP=/absolute/path/to/lmp YE3T_WITH_KOKKOS=yes \
  YE3T_MPI_RANKS=4 YE3T_KOKKOS_GPUS=4 ./run_examples.sh
```

This example runner validates device dispatch and numerical results; it is not
a timing harness. If ranks exceed `YE3T_KOKKOS_GPUS`, the runner labels the run
as oversubscribed correctness only. The runner writes into a new
`generated/run-PID` directory, applies a bounded timeout to every LAMMPS
process, and reports measured maximum residuals. See
[`examples/PACKAGES/ye3t/README.md`](examples/PACKAGES/ye3t/README.md) for
individual commands and model status. Reference one- and four-rank logs and
their measured residuals are recorded in
[`REFERENCE_RESULTS.md`](examples/PACKAGES/ye3t/REFERENCE_RESULTS.md).

Dump comparison keeps the atomic-energy, force, and per-atom virial tolerances
fixed. Only the explicitly summed extensive virial uses the transparent bound
`1e-8 + number_of_atoms * 5e-11` eV, which accounts for bounded accumulation
of already accepted per-atom roundoff and is printed in every comparison.

## GPU calibration and scaling tools

`tools/gpu_scaling/` holds the developer tools for evaluator timing and real
1/2/4-GPU scaling. They are not examples and keep their command-line
interfaces. `run_gpu_scaling.py` maps one physical GPU UUID per MPI rank
through `gpu_rank_wrapper.py`, rejects oversubscription in performance mode,
interleaves the eligible YE3T evaluator portfolio with `pace/kk product`, and
emits a machine-readable preflight and scaling result. `calibrate_kokkos_auto.py`
measures every evaluator family certified for a bundle over four workload
points, retains same-run energy, force, and virial parity certificates, and
freezes the catalogue plan with the best worst-point paired 95% confidence
bound over direct into a replay that is bound to the exact executable, model,
plan, GPU class, driver/runtime versions, Kokkos layout, and measured
center-count envelope. Run the calibration on the target GPU; a replay from a
different device class is rejected rather than silently retuned. The checked
`gpu_scaling_cases.json` prevents a preset from silently requesting an
evaluator its plan does not contain.

## Model and runtime boundary

`ye3t-lammps` does not fit models. The portable workflow is:

```text
ye3t representation/coupling plan
        -> ye3t-ace descriptor construction and linear fit
        -> standard .yace plus optional compiled plan bundle
        -> pair_style ye3t
```

A plan bundle contains a manifest, execution plan, and `.yace` function map.
All payloads are content-hashed and use relative paths, so the bundle can be
moved as a directory. A plan is optional; the original `.yace` remains usable
by both ML-PACE and YE3T.

The matching installed-package example is
`ye3t/examples/compile_scalar_ace_lammps_plans.py` in the public `ye3t`
repository; the Ta fit and strict `.yace` export workflow lives in `ye3t-ace`
(not yet public). Symmetry compilation stays in `ye3t`, atomistic
fitting/export in `ye3t-ace`, and inference in this repository.

## Current limitations

- `pair_style ye3t/kk` implements ordinary symmetric-ACE direct,
  block/symmetric-power, homogeneous scalar-power, and coupled-product FP64
  evaluators. GPU `auto` validates the complete candidate portfolio and makes
  one deterministic catalogue-level decision at model load. Until an exact
  catalogue-and-hardware replay is supplied, it records a conservative
  decision and selects direct. Forced modes remain available for validation
  and calibration.
- CUDA has been validated locally; other Kokkos device backends are not yet
  release-qualified. The GPU scaling tools currently require NVIDIA
  `nvidia-smi`; a HIP identity backend is not yet implemented.
- The CPU backend requires `newton on`.
- The rank-16 fixture is an exact execution fixture, not a fitted high-rank
  material potential. Do not use it for production MD.
- The shipped Ta controls passed bounded EOS, elasticity/Born, and NVE
  qualification but are not close-collision potentials. ZBL-reference models
  must be trained against `E-E_ZBL` and `F-F_ZBL`, then run with the exact
  exported reference definition. Legacy residual-only models use
  `hybrid/overlay ... zbl`; V4 models with bound references already restore
  ZBL internally and must not use an additional overlay.
- Pair coefficients are not serialized to binary restart files. Restate
  `pair_style` and `pair_coeff` after `read_restart`.
- The installed-runtime-prefix CMake route is experimental. The documented
  `ML_YE3T_RUNTIME_SOURCE` route is the supported portable route.
- Multi-element parsing and execution are implemented and covered by the
  two-species tagged-Cauchy test fixtures, but the shipped fitted examples are
  single-element.

## Developer checks

The inexpensive installer test is:

```bash
tests/test_patch_lammps.sh "$PWD"
python3 tests/test_patch_lammps_uninstall.py -v
```

The GPU-runner parser, rank mapping, oversubscription refusal, timing, and
dispatch checks have a CPU-only unit test:

```bash
python3 tests/test_gpu_scaling_runner.py
```

The full source-package test creates a disposable local LAMMPS checkout,
builds `lmp`, and runs the public examples. It requires `git`, Ninja,
`ccache`, an MPI launcher able to start four ranks, GNU `timeout`,
`python3`, and `sha256sum`:

```bash
tests/test_source_package_lammps.sh \
  "$PWD" /path/to/lammps /path/to/ye3t /path/to/lammps-user-pace
```

With a CUDA build, run the focused device tests:

```bash
tests/test_kokkos_b0_cuda.sh "$PWD" /path/to/lmp \
  examples/PACKAGES/ye3t/models/ta_l8_compact/model.yace
tests/test_kokkos_direct_cuda.sh "$PWD" /path/to/ye3t-enabled-lmp \
  examples/PACKAGES/ye3t/models/ta_l8_compact/model.yace \
  /path/to/pace-enabled-lmp
tests/test_kokkos_block_cuda.sh "$PWD" /path/to/ye3t-enabled-lmp \
  examples/PACKAGES/ye3t/models/ta_l8_compact/model.yace \
  examples/PACKAGES/ye3t/models/ta_l8_compact/manifest.json \
  /path/to/pace-enabled-lmp
tests/test_kokkos_auto_cuda.sh "$PWD" /path/to/ye3t-enabled-lmp \
  examples/PACKAGES/ye3t/models/ta_l8_compact/model.yace \
  examples/PACKAGES/ye3t/models/ta_l8_compact/manifest.json \
  /path/to/pace-enabled-lmp
tests/test_kokkos_auto_replay_cuda.sh "$PWD" /path/to/ye3t-enabled-lmp \
  examples/PACKAGES/ye3t/models/ta_l8_compact/model.yace \
  examples/PACKAGES/ye3t/models/ta_l8_compact/manifest.json \
  /path/to/device-matched/kokkos_auto_replay.json \
  /path/to/pace-enabled-lmp
tests/test_kokkos_scalar_power_cuda.sh "$PWD" /path/to/ye3t-enabled-lmp \
  examples/PACKAGES/ye3t/models/ta_l8_h16/model.yace \
  examples/PACKAGES/ye3t/models/ta_l8_h16/manifest.json \
  /path/to/pace-enabled-lmp
tests/test_kokkos_coupled_product_cuda.sh "$PWD" \
  /path/to/ye3t-enabled-lmp \
  examples/PACKAGES/ye3t/models/ta_l8_full/model.yace \
  examples/PACKAGES/ye3t/models/ta_l8_full/manifest.json \
  /path/to/pace-enabled-lmp
tests/test_kokkos_ownership_negative_cuda.sh "$PWD" \
  /path/to/ye3t-enabled-lmp \
  examples/PACKAGES/ye3t/models/ta_l8_compact/model.yace
```

The C++ sources follow the LAMMPS coding conventions: the repository
`.clang-format` is the LAMMPS configuration, every file carries the LAMMPS
banner, and headers use `LMP_` include guards. Build trees and raw benchmark
output should remain outside this repository.

## License and authors

`ye3t-lammps` is distributed under the GNU General Public License, version 2
or (at your option) any later version (SPDX: `GPL-2.0-or-later`); the full
license text is in `LICENSE`. Copyright (c) 2026 James M. Goff; the authors are
listed in `AUTHORS.md`. The separately distributed YE3T runtime is
BSD-3-Clause.
