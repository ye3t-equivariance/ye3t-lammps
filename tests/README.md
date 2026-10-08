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
`ye3t-methods` packages skip when they are not installed.
CTest runs four command-line integration drivers;
`test_mean_property_saved_parity.py` is a manual comparison of saved CPU and
device rows after the corresponding runs.

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

The full-M tagged and ordinary-density per-atom property tests use the bundled
models in `tests/fixtures/mean_property/`. After building the standalone native test
target, run `ctest -R ml_ye3t_mean_property --output-on-failure`. Supplying
`ML_YE3T_LAMMPS_TEST_EXECUTABLE` when configuring the plugin build also
enables the single-rank and two-rank LAMMPS integration cases. The optional
in-tree Kokkos route uses the same integration driver with
`YE3T_PROPERTY_COMPUTE_STYLE=ye3t/property/atom/kk/device` and
`YE3T_PROPERTY_LAMMPS_ARGS='-k on g 1 -sf kk'`. Set
`YE3T_PROPERTY_SAVE_ROWS` to a new empty directory for each CPU/device run,
then pass both directories to `tests/test_mean_property_saved_parity.py` for
a direct per-atom comparison. Set `YE3T_PROPERTY_MODEL_FAMILY=tagged` for
the CPU/device comparison because the density device plan currently rejects
density models. The density single-rank cases include a rank-3 repeated-content
multipath model. The tagged single-rank cases include a rank-3 L1 model with
an internal ``κ=(2,1)`` path and an isolated atom
and a neighbor inside the skin but outside the model cutoff. A pure two-tag
case also verifies a zero result with only one distinct neighbor.

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
