# ye3t-lammps

`ye3t-lammps` adds the `ML-YE3T` package to LAMMPS. The CPU
`pair_style ye3t` and FP64 Kokkos `pair_style ye3t/kk` evaluate linear
Young-E(3)-tensor (YE3T) potentials. Both accept PACE-compatible `.yace`
models, with an optional validated compiler execution plan. They also accept
native tagged-Cauchy and lifted-Cauchy bundles with nontrivial permutation
intermediates; those models have no `.yace` representation. Both styles
support multi-element type maps, MPI domain decomposition, and per-atom
energy and virial.

The CPU `compute ye3t/property/atom` evaluates real-tesseral L=1 and L=2
per-atom mean properties from compiler-bound tagged or ordinary-density full-M
v2 models exported by `ye3t-methods`. It supports explicit element/`NULL` type maps and an
independent compute group. Its syntax and limits are in
[doc/src/compute_ye3t_property_atom.rst](doc/src/compute_ye3t_property_atom.rst).
An experimental tagged full-M Kokkos device compute has matched CPU output
on single-rank and two-rank cases; ordinary-density full-M currently uses
the CPU compute.

Models are fitted and exported outside LAMMPS with `ye3t` and
`ye3t-methods`, which loads older saved models through a `ye3t_ace` import shim. LAMMPS
performs inference only; shipped examples run from pre-exported model files.
For a new checkout, follow [Install into LAMMPS](#install-into-lammps) before
running the examples below.

## First calculation

From the directory containing the `ye3t-lammps` checkout, run the bundled
Ta model after building LAMMPS:

```bash
cd ye3t-lammps/examples/PACKAGES/ye3t
/absolute/path/to/lmp -in in.ye3t.direct
```

The input constructs a bcc Ta cell, displaces one atom, loads the bundled
`.yace` model with `pair_style ye3t`, and writes per-atom energy, stress,
and forces. Change `variable model` and the element in `pair_coeff` to use a
compatible exported model. For a model fitted in Python, start with the
[`ye3t-methods` representation → basis → model examples](https://github.com/ye3t-equivariance/ye3t-methods/tree/main/examples/quickstart)
and its [deployment guide](https://github.com/ye3t-equivariance/ye3t-methods/blob/main/docs/deployment.rst).
The [example README](examples/PACKAGES/ye3t/README.md) lists the CPU, GPU,
MPI, finite-difference, and paper comparison runs.

## Per-atom property calculation from ASE

The `ye3t-methods` [vector quickstart](https://github.com/ye3t-equivariance/ye3t-methods/blob/main/examples/quickstart/per_atom_vector_to_lammps.py)
constructs an ASE Cu cell, fits an ordinary-density `L=1` per-atom model,
and writes `model.ye3t.json`, `atoms.data`, `in.property`, and Python
reference values. From the `ye3t-methods` checkout with both Python packages
installed, run:

```bash
python examples/quickstart/per_atom_vector_to_lammps.py
```

The script prints `lammps_input`, whose parent is the output directory set by
`metadata.output_path` in the example config. Run LAMMPS from that directory:

```bash
cd /path/to/output-directory
/absolute/path/to/lmp -in in.property
```

The LAMMPS input uses `pair_style zero` for neighbor lists and
`compute ye3t/property/atom` for the fitted property. The dump columns are
in the model's saved real-tesseral order. The analytic vector target is a
workflow demonstration, not a measured Cu property. This CPU route supports
natural-parity `L=1` and `L=2` density models under the input and normalization
limits in [the compute documentation](doc/src/compute_ye3t_property_atom.rst).
Python supports higher `L` subject to its configured source and compiler limits.

The pair-style keywords (`model_family`, `plan`, `block_policy`,
`auto_replay`, `chunksize`, `source_realization`), their defaults, and the
restrictions are documented in [doc/src/pair_ye3t.rst](doc/src/pair_ye3t.rst),
which the installer copies into the LAMMPS documentation tree. The five
`block_policy` values are `direct`, `block`, `auto`, `scalar_power`, and
`coupled_product`.

## Requirements

- Linux with a C++17 compiler
- CMake 3.20 or newer
- MPI for multi-process LAMMPS runs
- `yaml-cpp` with CMake package metadata
- a [ye3t](https://github.com/ye3t-equivariance/ye3t) source checkout, whose
  `ye3t/runtime/csrc` provides the native runtime both styles link
- Ninja and ccache, optional but recommended
- for `ye3t/kk`: a Kokkos-supported accelerator toolchain; CUDA is the
  validated device backend

On Debian or Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build ccache \
  libyaml-cpp-dev libopenmpi-dev openmpi-bin python3
```

## Install into LAMMPS

The package is tested against the LAMMPS stable release
`stable_22Jul2025_update4`. The installer changes only recognized layouts.

```bash
git clone https://github.com/lammps/lammps.git
git -C lammps checkout stable_22Jul2025_update4

git clone https://github.com/ye3t-equivariance/ye3t.git
git clone https://github.com/ye3t-equivariance/ye3t-lammps.git

./ye3t-lammps/tools/patch_lammps.sh --check --lammps-source "$PWD/lammps"
./ye3t-lammps/tools/patch_lammps.sh --apply --lammps-source "$PWD/lammps"
```

`--apply` installs the package sources in `lammps/src/ML-YE3T`, the CMake
module, the documentation pages, and the runnable examples in
`lammps/examples/PACKAGES/ye3t`, together with their checksum manifests. It
is idempotent for an identical installation and refuses partial, modified,
or conflicting files. To update the package, uninstall and apply again on the
same LAMMPS tree:

```bash
./ye3t-lammps/tools/patch_lammps.sh --uninstall --lammps-source "$PWD/lammps"
./ye3t-lammps/tools/patch_lammps.sh --apply --lammps-source "$PWD/lammps"
```

`--uninstall` backs up every file it removes, together with the original
`cmake/CMakeLists.txt`, under `lammps/.ye3t-uninstall-backups/`, removes only
YE3T files and CMake registrations, and never runs git. Reconfigure and
rebuild afterwards. [tools/README.md](tools/README.md) documents the
installer in detail.

Configure and build:

```bash
cmake -S "$PWD/lammps/cmake" -B "$PWD/lammps-build" -G Ninja \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -D BUILD_MPI=ON \
  -D PKG_ML-YE3T=ON \
  -D PKG_EXTRA-FIX=ON \
  -D ML_YE3T_RUNTIME_SOURCE="$PWD/ye3t"
cmake --build "$PWD/lammps-build" --parallel 4
./lammps-build/lmp -help | grep -w ye3t
```

`PKG_EXTRA-FIX` is needed only for the finite-difference examples. Omit
`-G Ninja` and the ccache launcher if those tools are unavailable.

### CUDA/Kokkos build

`ye3t/kk` requires an FP64 Kokkos build with the device architecture of the
target machine. It builds against both the 22 Jul 2025 stable Kokkos type
names and the renamed post-refactor names, and double precision is checked at
compile time. For an NVIDIA Ada (8.9) GPU:

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
./lammps-build-cuda/lmp -help | grep -E 'ye3t/kk'
```

Replace `Kokkos_ARCH_ADA89` with the `Kokkos_ARCH_*` option for the
destination GPU. Run the device style with the usual Kokkos switches, for
example `lmp -k on g 1 -pk kokkos neigh half -sf kk -in in.ye3t.direct`.
Both the `legacy` and `default` Kokkos layouts build; `legacy` is the
validated one. `KOKKOS_LAYOUT` is part of an AUTO replay's identity, so
recalibrate after changing it.

### Optional ML-PACE comparison build

The side-by-side examples run the same `.yace` bytes through `pace product`
and `ye3t`. To put both styles in one binary, add ML-PACE:

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

Without `LOCAL_ML-PACE`, LAMMPS may download its pinned PACE release at
configure time. The same additions apply to the CUDA build (`pace/kk`).

## Using the pair style

A standard `.yace` model needs no sidecar:

```lammps
units metal
newton on
pair_style ye3t
pair_coeff * * model.yace Ta
```

With a compiled plan bundle, `auto` scores the plan's candidate evaluators
once at `pair_coeff` and logs the selected schedule:

```lammps
pair_style ye3t plan /path/to/manifest.json block_policy auto
pair_coeff * * /path/to/model.yace Ta
```

On the GPU, `auto` selects the direct route unless a replay calibrated on the
same GPU class, executable, model, and plan is supplied:

```lammps
pair_style ye3t plan /path/to/manifest.json block_policy auto \
  auto_replay /path/to/kokkos_auto_replay.json
pair_coeff * * /path/to/model.yace Ta
```

The element names after the model map LAMMPS atom types in order, exactly as
for `pair_style pace`.

### Tagged-Cauchy models

Tagged-Cauchy bundles carry basis functions with nontrivial permutation
intermediates that a `.yace` file cannot represent:

```lammps
units metal
newton on
pair_style ye3t model_family tagged_cauchy block_policy direct
pair_coeff * * /path/to/model.ye3t.json Ta
```

A `ye3t_tagged_cauchy_composite_v1` manifest binds an ordinary `.yace`
backbone and a tagged correction by SHA-256, and both are evaluated inside
`pair_style ye3t`. Models fitted to ZBL-subtracted targets must be deployed
with the identical ZBL component: legacy residual-only bundles add it with
`pair_style hybrid/overlay ye3t ... zbl ...` (the element decks in
`examples/PACKAGES/ye3t/cost_comparison` show the exact commands), while
`ye3t_tagged_cauchy_slice_v4` bundles with bound references include it
themselves and must not be overlaid. Composite and V4 execution is CPU-only;
the doc page lists the `ye3t/kk` limits for native tagged bundles.

### Lifted-Cauchy models

`model_family lifted_cauchy` loads the native lifted-Cauchy bundle format
(`model.ye3t.json` plus its checksummed components); the `source_realization`
keyword is described on the doc page. No lifted example ships; the public
linear examples use the tagged basis or ordinary density-projected features.

## Examples

After patching, the examples are in `lammps/examples/PACKAGES/ye3t`
([README](examples/PACKAGES/ye3t/README.md)):

- two fitted Ta rank-through-8 models (58 and 61 functions)
  with `direct`, `block`, `auto`, and coupled-product execution, same-model
  `pace product` and `pace recursive` comparisons, one- and multi-rank MPI
  checks, finite-difference force and virial checks, a fixed-position replay
  across CPU, Kokkos, and PACE, a short NVE run, and a rank-16 execution
  fixture;
- fitted Li, Mo, Cu, Ni, Si, and Ge linear ACE controls and tagged YE3T
  models under `cost_comparison/`, each with PACE/YE3T same-model, tagged,
  finite-difference, and NVE inputs and reference logs.

`run_examples.sh` runs the Ta decks (and, with `YE3T_WITH_HIGH_RANK=yes`,
the rank-16 fixture); it takes no command-line arguments and is configured
through environment variables:
`YE3T_LMP` (required), `YE3T_MPI_RANKS`, `YE3T_MPIEXEC`,
`YE3T_EXAMPLE_OUTPUT`, `YE3T_EXAMPLE_TIMEOUT`, `YE3T_WITH_PACE`,
`YE3T_WITH_NUMDIFF`, `YE3T_WITH_HIGH_RANK`, `YE3T_WITH_KOKKOS`, and
`YE3T_KOKKOS_GPUS`. For example:

```bash
cd lammps/examples/PACKAGES/ye3t
YE3T_LMP=/absolute/path/to/lammps-build/lmp YE3T_MPI_RANKS=4 \
  YE3T_WITH_PACE=yes YE3T_WITH_NUMDIFF=yes ./run_examples.sh
```

The runner writes every log, dump, and JSON verification report to a new
output directory, applies a timeout to every LAMMPS run, and exits nonzero on
the first failed check. The fitted element decks are run individually from
their `cost_comparison/<element>` directories. Reference logs and measured
residuals are recorded in
[REFERENCE_RESULTS.md](examples/PACKAGES/ye3t/REFERENCE_RESULTS.md).

## Performance evidence

Performance is hardware- and model-specific. The six-element CPU
comparison of PACE `product`, YE3T-symmetric AUTO on byte-identical `.yace`
files, and YE3T-mixed tagged models, with the conditions of every timing, is
in
[docs/results/cost_comparison_three_way_auto_v3_20260921](docs/results/cost_comparison_three_way_auto_v3_20260921/README.md).

## GPU calibration and scaling tools

`tools/gpu_scaling/` holds the developer tools for evaluator timing and
1/2/4-GPU scaling: `run_gpu_scaling.py` (one GPU per MPI rank through
`gpu_rank_wrapper.py`), and `calibrate_kokkos_auto.py` with
`make_kokkos_auto_replay.py`, which calibrate a plan on the target GPU and
freeze the result as the `auto_replay` file bound to that executable, model,
plan, GPU class, and Kokkos layout. Each script documents its command line.

## Model and runtime boundary

```text
ye3t representation/coupling plan
        -> ye3t-methods descriptor construction and linear fit
        -> standard .yace plus optional compiled plan bundle
        -> pair_style ye3t
```

A plan bundle contains a manifest, an execution plan, and a `.yace` function
map. All payloads are content-hashed and use relative paths, so a bundle can
be moved as a directory. The plan is optional; the `.yace` file remains
usable by both ML-PACE and YE3T. `ye3t/examples/compile_scalar_ace_lammps_plans.py`
in the `ye3t` repository compiles representative plans.

## Developer checks

```bash
tests/test_patch_lammps.sh "$PWD"
python3 tests/test_patch_lammps_uninstall.py -v
python3 tests/test_source_contracts.py
python -m pytest tests
```

The pytest suite excludes command-line integration drivers invoked by CTest
and a separate manual CPU/device comparison. [tests/README.md](tests/README.md)
lists those routes, the full source-package test, the native-CPU CTest route,
and the CUDA/Kokkos test scripts;
[CONTRIBUTING.md](CONTRIBUTING.md) states the source conventions.

## License and authors

`ye3t-lammps` is distributed under the GNU General Public License, version 2
or (at your option) any later version (SPDX: `GPL-2.0-or-later`); the full
license text is in `LICENSE`. Copyright (c) 2026 James M. Goff; the authors are
listed in `AUTHORS.md`. The separately distributed YE3T runtime is
BSD-3-Clause.
