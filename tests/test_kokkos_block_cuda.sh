#!/bin/sh

set -eu

if test "$#" -lt 4 || test "$#" -gt 5; then
  echo "Usage: test_kokkos_block_cuda.sh YE3T_LAMMPS_ROOT YE3T_LMP MODEL_YACE PLAN_MANIFEST [PACE_LMP]" >&2
  exit 2
fi

repo_root=$(CDPATH= cd -- "$1" && pwd -P)
ye3t_lmp=$(CDPATH= cd -- "$(dirname -- "$2")" && pwd -P)/$(basename -- "$2")
model=$(CDPATH= cd -- "$(dirname -- "$3")" && pwd -P)/$(basename -- "$3")
manifest=$(CDPATH= cd -- "$(dirname -- "$4")" && pwd -P)/$(basename -- "$4")
pace_lmp=
if test "$#" -eq 5; then
  pace_lmp=$(CDPATH= cd -- "$(dirname -- "$5")" && pwd -P)/$(basename -- "$5")
fi

if test ! -x "$ye3t_lmp" || test ! -f "$model" || test ! -f "$manifest"; then
  echo "The PairYE3T executable, model, or plan manifest does not exist" >&2
  exit 2
fi
if test -n "$pace_lmp" && test ! -x "$pace_lmp"; then
  echo "The PACE LAMMPS executable does not exist" >&2
  exit 2
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required" >&2
  exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required" >&2
  exit 2
fi

tmp_base=${TMPDIR:-/tmp}
fixture_root=$(mktemp -d "$tmp_base/ye3t-kokkos-block-test.XXXXXX")
cleanup()
{
  case "$fixture_root" in
    "$tmp_base"/ye3t-kokkos-block-test.*) rm -rf -- "$fixture_root" ;;
    *) echo "Refusing to remove unexpected fixture path: $fixture_root" >&2 ;;
  esac
}
trap cleanup EXIT HUP INT TERM

direct_input=$repo_root/examples/PACKAGES/ye3t/in.ye3t.direct
block_input=$repo_root/examples/PACKAGES/ye3t/in.ye3t.block
numdiff_input=$repo_root/examples/PACKAGES/ye3t/in.ye3t.numdiff
verifier=$repo_root/examples/PACKAGES/ye3t/verify_examples.py
. "$repo_root/tests/example_verifier.sh"

timeout 120 "$ye3t_lmp" -in "$direct_input" -var model "$model" \
  -var dump_path "$fixture_root/cpu-direct.dump" \
  -log "$fixture_root/cpu-direct.log" -screen none
timeout 120 "$ye3t_lmp" -in "$block_input" -var model "$model" \
  -var plan "$manifest" -var dump_path "$fixture_root/cpu-block.dump" \
  -log "$fixture_root/cpu-block.log" -screen none
timeout 120 "$ye3t_lmp" -k on g 1 -pk kokkos neigh half -sf kk \
  -in "$direct_input" -var model "$model" \
  -var dump_path "$fixture_root/gpu-direct.dump" \
  -log "$fixture_root/gpu-direct.log" -screen none
timeout 120 "$ye3t_lmp" -k on g 1 -pk kokkos neigh half -sf kk \
  -in "$block_input" -var model "$model" -var plan "$manifest" \
  -var dump_path "$fixture_root/gpu-block.dump" \
  -log "$fixture_root/gpu-block.log" -screen none
timeout 180 "$ye3t_lmp" -k on g 1 -pk kokkos neigh half -sf kk \
  -in "$numdiff_input" -var model "$model" -var plan "$manifest" \
  -var policy block -log "$fixture_root/gpu-block-numdiff.log" -screen none

grep -Eq 'YE3T Kokkos device-plan probe: execution_space (Cuda|HIP)' \
  "$fixture_root/gpu-block.log"
grep -q 'evaluator block' "$fixture_root/gpu-block.log"
grep -Eq 'block_routes [1-9][0-9]*' "$fixture_root/gpu-block.log"
grep -Eq 'block_symmetric_power_child_plans [1-9][0-9]*' \
  "$fixture_root/gpu-block.log"
grep -Eq 'block_schedule (fused_center_lane_v2|plan_route_major_v1)' \
  "$fixture_root/gpu-block.log"
grep -q 'maximum_error 0' "$fixture_root/gpu-block.log"
grep -q 'pair ye3t/kk' "$fixture_root/gpu-block.log"
grep -q 'evaluator direct' "$fixture_root/gpu-direct.log"
if grep -q 'YE3T CPU dispatch:' "$fixture_root/gpu-block.log"; then
  echo "The Kokkos block test entered the CPU evaluator" >&2
  exit 1
fi
if test "${YE3T_REQUIRE_ZERO_DIRECT:-0}" = 1; then
  grep -q 'direct_monomial_storage 0' "$fixture_root/gpu-block.log"
fi

cpu_portfolio=$(sed -n 's/.*evaluator portfolio \([0-9a-f]*\).*/\1/p' \
  "$fixture_root/cpu-block.log" | tail -n 1)
gpu_portfolio=$(sed -n 's/.*evaluator portfolio \([0-9a-f]*\).*/\1/p' \
  "$fixture_root/gpu-block.log" | tail -n 1)
if test -z "$cpu_portfolio" || test "$cpu_portfolio" != "$gpu_portfolio"; then
  echo "CPU/GPU selected block-portfolio identity mismatch" >&2
  exit 1
fi

direct_flat=$(sed -n '/YE3T Kokkos device-plan probe:.*evaluator direct,/s/.*flat_plan \([0-9a-f]*\).*/\1/p' \
  "$fixture_root/gpu-direct.log" | tail -n 1)
block_flat=$(sed -n '/YE3T Kokkos device-plan probe:.*evaluator block,/s/.*flat_plan \([0-9a-f]*\).*/\1/p' \
  "$fixture_root/gpu-block.log" | tail -n 1)
if test -z "$direct_flat" || test -z "$block_flat" || \
   test "$direct_flat" = "$block_flat"; then
  echo "Direct and block GPU-layout identities were not distinguished" >&2
  exit 1
fi

verify_dumps "$fixture_root/ye3t-parity.json" "$fixture_root/cpu-direct.dump" \
  "$fixture_root/cpu-block.dump" "$fixture_root/gpu-direct.dump" \
  "$fixture_root/gpu-block.dump"
verify_numdiff "$fixture_root/gpu-block-numdiff.json" "$fixture_root/gpu-block-numdiff.log" \
  force_atol=1.0e-6 virial_atol=1.0e-2

if test -n "$pace_lmp"; then
  pace_input=$repo_root/examples/PACKAGES/ye3t/in.ye3t.pace-product
  timeout 120 "$pace_lmp" -k on g 1 -pk kokkos neigh half -sf kk \
    -in "$pace_input" -var model "$model" \
    -var dump_path "$fixture_root/pace-kokkos.dump" \
    -log "$fixture_root/pace-kokkos.log" -screen none
  grep -q 'KOKKOS mode with Kokkos version' "$fixture_root/pace-kokkos.log"
  grep -q 'Product evaluator is used' "$fixture_root/pace-kokkos.log"
  grep -q 'pair pace/kk' "$fixture_root/pace-kokkos.log"
  verify_dumps "$fixture_root/pace-block-parity.json" "$fixture_root/pace-kokkos.dump" \
    "$fixture_root/gpu-block.dump"
fi

echo "PairYE3T forced-block CPU/direct/Kokkos parity and finite differences passed."
if test -n "$pace_lmp"; then
  echo "PACE/Kokkos product versus PairYE3T/Kokkos block parity passed."
fi
