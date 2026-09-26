# Native CPU evaluator tests and benchmark

These tests use the actual YE3T runtime, the pair style's evaluator, and the
production spline/DAG compilation helpers. They do **not** load deployment YAML
or sidecars and they do **not** execute LAMMPS, MPI, Kokkos, or PACE.

`generate_support.py` extracts the unchanged production `build_splines` and
monomial-DAG compiler functions from `src/ye3t_yace_model.cpp` into a test-only
translation unit. Only fixture decoding/model-field construction is test-specific.
Both the original YACE and generated fixture hashes must match `provenance.json`;
for the two retired Si fixtures whose source `.yace` is not shipped
(`source_shipped: false`), only the fixture hash is checked and the source
SHA-256 is recorded for identification.
This avoids a mock runtime and avoids requiring yaml-cpp for these narrow tests;
it is not a substitute for the deployment-loader test suite.

## Build and run

Point `ML_YE3T_RUNTIME_SOURCE` at the ye3t checkout matching this package before configuring:

```sh
cmake -S . -B build-native-tests -DCMAKE_BUILD_TYPE=Release \
  -DML_YE3T_BUILD_CPU_RUNTIME=OFF \
  -DML_YE3T_BUILD_NATIVE_CPU_TESTS=ON \
  -DML_YE3T_RUNTIME_SOURCE=/absolute/path/to/ye3t
cmake --build build-native-tests --parallel 2
ctest --test-dir build-native-tests --output-on-failure
```

`ML_YE3T_BUILD_CPU_RUNTIME=OFF` disables the deployment-loader library for this
**test configuration only**. It is not the setting to use when building LAMMPS.

```sh
# 4096 local environments, outer evaluator batch 256, production schedule.
build-native-tests/ml_ye3t_cpu_native_benchmark \
  tests/native_cpu/full.fixture 4096 256 auto
```

Benchmark-only schedules `replay` and `whole` access the private test friend.
`replay` selects the former cache-budget edge-recomputation schedule; `whole`
retains a complete outer batch's tables. They both use the current angular
and readout kernels, so neither alone reconstructs an earlier source revision.
There is no production `pair_style` input for these controls.

The native benchmark uses repeated deterministically distorted BCC environments
at lattice parameter 3.3161146998079496, including for the silicon-model kernel
fixtures. This is a fixed synthetic computational workload, not a silicon
physical-validation dataset. It has 14 accepted neighbors for the Ta fixtures and
26 for the Si fixtures. Model initialization is excluded; two evaluations warm up
the evaluator. Each reported median is over three timing samples of at least
0.12 seconds. An optional output path writes all energies then Cartesian edge
gradients; a final `--verify-only` argument skips the timing loop.

Tests compare automatic, whole-batch, and tiny-edge replay schedules for 0, 1, 7,
8, 9, 16, 17, 31, 129, and 256 centers. They exercise shuffled edges with scatter,
isolated centers, cutoff edges, changing geometry, invalid data, and coordinate
finite differences. They assert one source evaluation per edge in production.
An explicitly synthetic two-species variant tests central-species grouping; it
is not a fitted alloy potential. The five imported models exercise the DIRECT
readout, not every block/scalar/coupled-product sidecar route.

## Fixture regeneration

`emit_native_fixture.py MODEL.yace OUTPUT.fixture` requires PyYAML and supports
only the source conventions used by these fixtures. Regeneration must update
both source and output hashes in `provenance.json` after reviewing any model
change. Ordinary builds require only Python's standard library.
