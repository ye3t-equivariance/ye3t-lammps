# Shared native loader — 2026-11-13

The native loader and JSON reader are synchronized from `ye3t-methods` commit
`5796e546e604725a87f1194265eeb7b18ec1f39c`, files
`native/src/ye3t_tagged_cauchy_model.cpp` and
`native/src/ye3t_canonical_json_hash.{cpp,h}`. Their existing GPL notices are
preserved. No third-party implementation or new dependency is introduced.
The existing `canonical_json_value_array_items` entry point remains a thin
forwarder to the shared array reader for the mean-property loader.

`CanonicalJSONDocument` retains an immutable input snapshot and lazily builds
its compact member index once. The tagged loader reuses that document for
field lookup and integrity checks. V4 forward/adjoint records are converted
individually, and only consumed fields of offline moment records become YAML
nodes. Complete original content remains covered by the integrity hashes.
This preserves V2/V3/V4 and composite model formats. It does not reconstruct
or revalidate coupling coefficients.

The evaluator and its pre-existing changes are outside this migration. The
existing Kokkos V4 limitations also remain unchanged.

## Regression coverage

`ml_ye3t_canonical_json` checks immutable-file reuse, repeated lookup, canonical
hashes, streamed records, numeric spelling and rejection of malformed or
duplicate fields, including unused subtrees. Existing tagged/lifted/property
tests cover other consumers of the JSON helper.

To exercise the optional V4 test with an existing compiler-exported artifact,
configure the normal standalone build with:

```sh
cmake -S . -B build -DBUILD_TESTING=ON \
  -DML_YE3T_BUILD_TAGGED_C_API=ON \
  -DML_YE3T_TAGGED_V4_TEST_MODEL=/absolute/path/model.lammps.ye3t.json
cmake --build build --target ye3t_tagged_c_api ml_ye3t_canonical_json_test \
  ml_ye3t_tagged_cauchy_cpu_test
ctest --test-dir build --output-on-failure -R 'ml_ye3t_(canonical_json|tagged_cauchy)'
```

This V4 test requires only Python's standard library and the built native
library. It checks compact/pretty encodings, modified unused proof content,
forward/adjoint corruption, a rehashed inconsistent schedule, and replacement
of a previously loaded pathname.

Matched baseline/candidate builds, timing samples, memory and new-plugin
energy/force comparisons are retained in the workspace study
`ye3t-workflows/MLIP/linear_standardization_20261113/lammps_loader_migration/`.
That study is development evidence and is not an installation dependency.
