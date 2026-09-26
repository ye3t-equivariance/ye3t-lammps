# Opt-in Kokkos device checks

These targets exercise actual Kokkos views, kernels, team barriers, reductions,
scan metadata, and the tagged source-plan upload. They do not emulate Kokkos.
They require a Kokkos-enabled build (CUDA/HIP or a host backend) and are not
part of the default ctest run. The main project also includes independently
runnable *host arithmetic* tests; those do not validate device execution.

## Configure with the same Kokkos and compiler as LAMMPS

These checks require CMake 3.22 or newer and a C++20 compiler, in addition
to the Kokkos toolchain. From the repository root, use the Kokkos source
bundled with your LAMMPS checkout:

```sh
cmake -S tests/kokkos_followup -B build-kokkos-followup \
  -DYE3T_KOKKOS_SOURCE=/absolute/path/to/lammps/lib/kokkos \
  -DYE3T_RUNTIME_SOURCE=/absolute/path/to/ye3t \
  -DCMAKE_BUILD_TYPE=Release \
  <the same CMAKE_CXX_COMPILER, Kokkos_ENABLE_CUDA/HIP, and Kokkos_ARCH_* flags as your LAMMPS build>
cmake --build build-kokkos-followup --parallel 2
ctest --test-dir build-kokkos-followup --output-on-failure
```

The bracketed compiler/backend flags are placeholders to replace, not shell
syntax. For an installed Kokkos, omit `YE3T_KOKKOS_SOURCE` and provide `Kokkos_DIR`
or the appropriate `CMAKE_PREFIX_PATH`. Do not select a different Kokkos version
or host backend merely to obtain a passing build. The executable prints the
execution-space name; a Serial/OpenMP pass is not a CUDA/HIP pass.

`YE3T_RUNTIME_SOURCE` is optional for the executable and enables an additional
object target explicitly instantiating the actual ordinary device-plan template.
Use the same ye3t runtime revision that the pair style is built against.

## Covered here

- Combined edge count / maximum / status from a real exclusive scan, including
  empty chunks and status values; device-valued EV and diagnostic accumulation
  across chunks and reset across timesteps.
- Owner-gather reverse differentiation of a shared/repeated-child DAG using a
  real team kernel and team barriers, with untouched trailing lanes.
- Actual tagged plan upload and grouped source kernels at degrees 0, 1, 4, and 8;
  legacy and physical-image V3 source conventions; absolute values, derivatives,
  values-only specialization, and the V3 streamed density-only pullback.

The test EV type is a small test tally, not LAMMPS `EV_FLOAT`. The executable does
not instantiate `PairYE3TKokkos` or perform LAMMPS force/virial assembly. Full
pair-style compilation and the existing LAMMPS device tests are still required.

## Deployment validation

Build LAMMPS normally against this package and the matching
runtime. Run the existing direct, block, scalar-power, coupled-product, lifted,
and tagged validation scripts for the model families actually deployed. Include
force-only, global-tally and per-atom-tally steps, an empty MPI rank, migration,
sorting, cutoff crossings and periodic/triclinic boxes. Compare forces and all
virial modes against the unchanged plain CPU implementation and, for ordinary
ACE, PACE. Use compute-sanitizer on CUDA when available.

The existing direct script accepts:

```sh
sh tests/test_kokkos_direct_cuda.sh "$PWD" /path/to/ye3t-lmp \
  /path/to/model.yace /path/to/pace-lmp
```

It runs both ordinary CPU and actual CUDA/HIP dispatch and checks the execution
space. Regenerate any GPU AUTO calibration replay after changing the Kokkos pair
style, because the device schedule and plan identity change with it. No new
pair-style input keyword is required.

## Structured-execution pass

With `YE3T_RUNTIME_SOURCE` specified, the second executable tests the actual
production block view record and tile/transpose arithmetic in a real Kokkos
team kernel, including partial center tiles and zero-valued inputs. A separate
team test uses shared-resident DIRECT graph values and adjoints with repeated
children. A pole-axis test instantiates both values-only and Cartesian-gradient
harmonic streams on the selected execution space. The object target instantiates the updated production device-plan
upload, including the execution schedules and harmonic recurrence tables.
These additions also require a Kokkos-enabled build and are not part of the
default ctest run. Neither executable instantiates the complete pair style; the
normal LAMMPS build remains essential.

The current replay ABI is `ye3t_kokkos_candidate_runtime_v2`, and the block scratch
layout is `monomial_tile16_lane_fast_v2`. Regenerate replay files rather than
editing their ABI/hash fields. Host duration of a `YE3T::*` profiling region is
not GPU elapsed time; regions deliberately contain no extra fences.
