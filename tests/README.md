# Tests

Inexpensive checks that need no LAMMPS build:

```bash
tests/test_patch_lammps.sh "$PWD"
python3 tests/test_patch_lammps_uninstall.py -v
python3 tests/test_gpu_scaling_runner.py
python3 tests/test_source_contracts.py
pytest tests/test_public_example_verifier.py
```

`pytest tests/` runs the Python suite; tests that need the `ye3t` or
`ye3t-ace` packages skip when they are not installed.

## Full source-package test

Creates a disposable LAMMPS checkout, builds `lmp`, and runs the public
examples on one and four ranks. It requires `git`, Ninja, `ccache`, an MPI
launcher able to start four ranks, GNU `timeout`, `python3`, and
`sha256sum`:

```bash
tests/test_source_package_lammps.sh \
  "$PWD" /path/to/lammps /path/to/ye3t /path/to/lammps-user-pace
```

## Native-CPU evaluator tests

The evaluator and the production spline/DAG compilation helpers are tested
without LAMMPS through the standalone CMake project; see
`tests/native_cpu/README.md` for the configure, build, and `ctest` commands.

## CUDA/Kokkos tests

With a CUDA build of LAMMPS, run the focused device tests:

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

The tagged-Cauchy device test compares a CPU build and a Kokkos build on the
tagged fixtures:

```bash
tests/test_kokkos_tagged_cauchy_cuda.sh "$PWD" /path/to/cpu-lmp /path/to/kk-lmp \
  /path/to/new-output-dir --model-k1 PATH --reference-k1 PATH
```

Run it without arguments to see the optional multispecies, timing, and skip
switches. `tests/kokkos_followup/README.md` describes the opt-in Kokkos
device checks that build against the LAMMPS Kokkos sources.
