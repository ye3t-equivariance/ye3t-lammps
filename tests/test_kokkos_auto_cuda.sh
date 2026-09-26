#!/bin/sh

set -eu

if test "$#" -lt 4 || test "$#" -gt 5; then
  echo "Usage: test_kokkos_auto_cuda.sh YE3T_LAMMPS_ROOT YE3T_LMP MODEL_YACE PLAN_MANIFEST [PACE_LMP]" >&2
  exit 2
fi

repo_root=$(CDPATH= cd -- "$1" && pwd -P)
ye3t_lmp=$(CDPATH= cd -- "$(dirname -- "$2")" && pwd -P)/$(basename -- "$2")
model=$(CDPATH= cd -- "$(dirname -- "$3")" && pwd -P)/$(basename -- "$3")
plan=$(CDPATH= cd -- "$(dirname -- "$4")" && pwd -P)/$(basename -- "$4")
pace_lmp=
if test "$#" -eq 5; then
  pace_lmp=$(CDPATH= cd -- "$(dirname -- "$5")" && pwd -P)/$(basename -- "$5")
fi

if test ! -x "$ye3t_lmp" || test ! -f "$model" || test ! -f "$plan"; then
  echo "The PairYE3T executable, model, or evaluator plan does not exist" >&2
  exit 2
fi
if test -n "$pace_lmp" && test ! -x "$pace_lmp"; then
  echo "The PACE LAMMPS executable does not exist" >&2
  exit 2
fi
for command in timeout python3 mpiexec; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "$command is required" >&2
    exit 2
  fi
done

tmp_base=${TMPDIR:-/tmp}
fixture_root=$(mktemp -d "$tmp_base/ye3t-kokkos-auto-test.XXXXXX")
cleanup()
{
  case "$fixture_root" in
    "$tmp_base"/ye3t-kokkos-auto-test.*) rm -rf -- "$fixture_root" ;;
    *) echo "Refusing to remove unexpected fixture path: $fixture_root" >&2 ;;
  esac
}
trap cleanup EXIT HUP INT TERM

direct_input=$repo_root/examples/PACKAGES/ye3t/in.ye3t.direct
auto_input=$repo_root/examples/PACKAGES/ye3t/in.ye3t.auto
numdiff_input=$repo_root/examples/PACKAGES/ye3t/in.ye3t.numdiff
verifier=$repo_root/examples/PACKAGES/ye3t/verify_examples.py
. "$repo_root/tests/example_verifier.sh"

timeout 120 "$ye3t_lmp" -k on g 1 -pk kokkos neigh half -sf kk \
  -in "$direct_input" -var model "$model" \
  -var dump_path "$fixture_root/direct.dump" \
  -log "$fixture_root/direct.log" -screen none
timeout 120 "$ye3t_lmp" -k on g 1 -pk kokkos neigh half -sf kk \
  -in "$auto_input" -var model "$model" -var plan "$plan" \
  -var policy auto -var dump_path "$fixture_root/auto.dump" \
  -log "$fixture_root/auto.log" -screen none

grep -Eq 'YE3T Kokkos device-plan probe: execution_space (Cuda|HIP)' \
  "$fixture_root/auto.log"
grep -Fq 'evaluator auto[direct]' "$fixture_root/auto.log"
grep -Fq 'planner_profile kokkos_gpu_conservative_direct_v1' \
  "$fixture_root/auto.log"
grep -Fq 'calibration_hash not_applicable' "$fixture_root/auto.log"
grep -Fq 'decision_reason no_authorized_non_direct_profile_match' \
  "$fixture_root/auto.log"
grep -Fq 'active_direct yes, active_block no, active_scalar_power no, active_coupled_product no' \
  "$fixture_root/auto.log"
if grep -q 'YE3T CPU dispatch:' "$fixture_root/auto.log"; then
  echo "The Kokkos AUTO test entered the CPU evaluator" >&2
  exit 1
fi

direct_flat=$(sed -n 's/.*evaluator direct,.*flat_plan \([0-9a-f]*\).*/\1/p' \
  "$fixture_root/direct.log" | tail -n 1)
auto_flat=$(sed -n 's/.*evaluator auto\[direct\],.*flat_plan \([0-9a-f]*\).*/\1/p' \
  "$fixture_root/auto.log" | tail -n 1)
direct_semantic=$(sed -n 's/.*semantic_selection \([0-9a-f]*\).*/\1/p' \
  "$fixture_root/direct.log" | tail -n 1)
auto_semantic=$(sed -n 's/.*semantic_selection \([0-9a-f]*\).*/\1/p' \
  "$fixture_root/auto.log" | tail -n 1)
if test -z "$direct_flat" || test "$direct_flat" != "$auto_flat"; then
  echo "GPU AUTO did not retain the selected direct numeric plan" >&2
  exit 1
fi
if test -z "$direct_semantic" || test -z "$auto_semantic" ||
   test "$direct_semantic" = "$auto_semantic"; then
  echo "GPU AUTO did not retain a distinct semantic selection identity" >&2
  exit 1
fi
verify_dumps "$fixture_root/direct-auto-parity.json" "$fixture_root/direct.dump" \
  "$fixture_root/auto.dump"

timeout 120 mpiexec -n 2 "$ye3t_lmp" -k on g 1 \
  -pk kokkos neigh half -sf kk -in "$auto_input" -var model "$model" \
  -var plan "$plan" -var policy auto \
  -var dump_path "$fixture_root/auto-rank2.dump" \
  -log "$fixture_root/auto-rank2.log" -screen none
grep -Fq 'evaluator auto[direct]' "$fixture_root/auto-rank2.log"
auto_device=$(sed -n 's/.*device_schedule \([0-9a-f]*\).*/\1/p' \
  "$fixture_root/auto.log" | tail -n 1)
auto_rank2_device=$(sed -n 's/.*device_schedule \([0-9a-f]*\).*/\1/p' \
  "$fixture_root/auto-rank2.log" | tail -n 1)
if test -z "$auto_device" || test "$auto_device" != "$auto_rank2_device"; then
  echo "GPU AUTO device identity changed with the MPI rank count" >&2
  exit 1
fi
verify_dumps "$fixture_root/auto-rank-parity.json" "$fixture_root/auto.dump" \
  "$fixture_root/auto-rank2.dump"

timeout 120 "$ye3t_lmp" -k on g 1 -pk kokkos neigh half -sf kk \
  -in "$numdiff_input" -var model "$model" -var plan "$plan" \
  -var policy auto -var force_delta 1e-4 -var virial_delta 1e-6 \
  -log "$fixture_root/numdiff-auto.log" -screen none
grep -Fq 'evaluator auto[direct]' "$fixture_root/numdiff-auto.log"
verify_numdiff "$fixture_root/numdiff-auto.json" "$fixture_root/numdiff-auto.log"

if test -n "$pace_lmp"; then
  pace_input=$repo_root/examples/PACKAGES/ye3t/in.ye3t.pace-product
  timeout 120 "$pace_lmp" -k on g 1 -pk kokkos neigh half -sf kk \
    -in "$pace_input" -var model "$model" \
    -var dump_path "$fixture_root/pace.dump" \
    -log "$fixture_root/pace.log" -screen none
  grep -q 'Product evaluator is used' "$fixture_root/pace.log"
  grep -q 'pair pace/kk' "$fixture_root/pace.log"
  verify_dumps "$fixture_root/pace-auto-parity.json" "$fixture_root/pace.dump" \
    "$fixture_root/auto.dump"
fi

echo "PairYE3T catalogue-global GPU AUTO parity, MPI identity, and finite differences passed."
