#!/bin/sh

set -eu

if test "$#" -lt 4 || test "$#" -gt 5; then
  echo "Usage: test_kokkos_scalar_power_cuda.sh YE3T_LAMMPS_ROOT YE3T_LMP MODEL_YACE PLAN_MANIFEST [PACE_LMP|--pure-scalar]" >&2
  exit 2
fi

repo_root=$(CDPATH= cd -- "$1" && pwd -P)
ye3t_lmp=$(CDPATH= cd -- "$(dirname -- "$2")" && pwd -P)/$(basename -- "$2")
model=$(CDPATH= cd -- "$(dirname -- "$3")" && pwd -P)/$(basename -- "$3")
manifest=$(CDPATH= cd -- "$(dirname -- "$4")" && pwd -P)/$(basename -- "$4")
pace_lmp=
pure_scalar=0
if test "$#" -eq 5; then
  if test "$5" = "--pure-scalar"; then
    pure_scalar=1
  else
    pace_lmp=$(CDPATH= cd -- "$(dirname -- "$5")" && pwd -P)/$(basename -- "$5")
  fi
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
keep_evidence=0
if test -n "${YE3T_TEST_EVIDENCE_DIR:-}"; then
  if test -e "$YE3T_TEST_EVIDENCE_DIR"; then
    echo "YE3T_TEST_EVIDENCE_DIR already exists: $YE3T_TEST_EVIDENCE_DIR" >&2
    exit 2
  fi
  mkdir -p -- "$YE3T_TEST_EVIDENCE_DIR"
  fixture_root=$(CDPATH= cd -- "$YE3T_TEST_EVIDENCE_DIR" && pwd -P)
  keep_evidence=1
else
  fixture_root=$(mktemp -d "$tmp_base/ye3t-kokkos-scalar-power-test.XXXXXX")
fi
cleanup()
{
  if test "$keep_evidence" -eq 1; then
    return
  fi
  case "$fixture_root" in
    "$tmp_base"/ye3t-kokkos-scalar-power-test.*) rm -rf -- "$fixture_root" ;;
    *) echo "Refusing to remove unexpected fixture path: $fixture_root" >&2 ;;
  esac
}
trap cleanup EXIT HUP INT TERM

input=$repo_root/tests/fixtures/high_rank/in.ye3t.high-rank-scalar-power
numdiff_input=$repo_root/examples/PACKAGES/ye3t/in.ye3t.numdiff
verifier=$repo_root/examples/PACKAGES/ye3t/verify_examples.py
. "$repo_root/tests/example_verifier.sh"
test_chunksize=17

timeout 180 "$ye3t_lmp" -in "$input" -var model "$model" \
  -var plan "$manifest" -var policy direct \
  -var ye3t_chunksize "$test_chunksize" \
  -var dump_path "$fixture_root/cpu-direct.dump" \
  -log "$fixture_root/cpu-direct.log" -screen none
timeout 180 "$ye3t_lmp" -in "$input" -var model "$model" \
  -var plan "$manifest" -var policy scalar_power \
  -var ye3t_chunksize "$test_chunksize" \
  -var dump_path "$fixture_root/cpu-scalar.dump" \
  -log "$fixture_root/cpu-scalar.log" -screen none
if test "$pure_scalar" -eq 0; then
  timeout 180 "$ye3t_lmp" -k on g 1 -pk kokkos neigh half -sf kk \
    -in "$input" -var model "$model" -var plan "$manifest" \
    -var policy direct -var ye3t_chunksize "$test_chunksize" \
    -var dump_path "$fixture_root/gpu-direct.dump" \
    -log "$fixture_root/gpu-direct.log" -screen none
fi
timeout 180 "$ye3t_lmp" -k on g 1 -pk kokkos neigh half -sf kk \
  -in "$input" -var model "$model" -var plan "$manifest" \
  -var policy scalar_power -var ye3t_chunksize "$test_chunksize" \
  -var dump_path "$fixture_root/gpu-scalar.dump" \
  -log "$fixture_root/gpu-scalar.log" -screen none
timeout 240 "$ye3t_lmp" -k on g 1 -pk kokkos neigh half -sf kk \
  -in "$numdiff_input" -var model "$model" -var plan "$manifest" \
  -var policy scalar_power -log "$fixture_root/gpu-scalar-numdiff.log" \
  -screen none

grep -Eq 'YE3T Kokkos device-plan probe: execution_space (Cuda|HIP)' \
  "$fixture_root/gpu-scalar.log"
grep -q 'evaluator scalar_power' "$fixture_root/gpu-scalar.log"
grep -Eq 'scalar_routes [1-9][0-9]*' "$fixture_root/gpu-scalar.log"
grep -Eq 'scalar_value_storage [1-9][0-9]*' "$fixture_root/gpu-scalar.log"
grep -q 'maximum_error 0' "$fixture_root/gpu-scalar.log"
grep -q 'scalar_math_probe passed' "$fixture_root/gpu-scalar.log"
grep -q "chunksize $test_chunksize" "$fixture_root/gpu-scalar.log"
grep -q 'pair ye3t/kk' "$fixture_root/gpu-scalar.log"
if test "$pure_scalar" -eq 0; then
  grep -q 'evaluator direct' "$fixture_root/gpu-direct.log"
else
  grep -q 'direct_monomial_storage 0' "$fixture_root/gpu-scalar.log"
fi
if grep -q 'YE3T CPU dispatch:' "$fixture_root/gpu-scalar.log"; then
  echo "The Kokkos scalar-power test entered the CPU evaluator" >&2
  exit 1
fi

cpu_portfolio=$(sed -n 's/.*evaluator portfolio \([0-9a-f]*\).*/\1/p' \
  "$fixture_root/cpu-scalar.log" | tail -n 1)
gpu_portfolio=$(sed -n 's/.*evaluator portfolio \([0-9a-f]*\).*/\1/p' \
  "$fixture_root/gpu-scalar.log" | tail -n 1)
if test -z "$cpu_portfolio" || test "$cpu_portfolio" != "$gpu_portfolio"; then
  echo "CPU/GPU selected scalar-power portfolio identity mismatch" >&2
  exit 1
fi

scalar_flat=$(sed -n '/evaluator scalar_power,/s/.*flat_plan \([0-9a-f]*\).*/\1/p' \
  "$fixture_root/gpu-scalar.log" | tail -n 1)
if test -z "$scalar_flat"; then
  echo "The scalar-power GPU-layout identity is missing" >&2
  exit 1
fi
if test "$pure_scalar" -eq 0; then
  direct_flat=$(sed -n '/evaluator direct,/s/.*flat_plan \([0-9a-f]*\).*/\1/p' \
    "$fixture_root/gpu-direct.log" | tail -n 1)
  if test -z "$direct_flat" || test "$direct_flat" = "$scalar_flat"; then
    echo "Direct and scalar-power GPU-layout identities were not distinguished" >&2
    exit 1
  fi
fi

if test "$pure_scalar" -eq 0; then
  verify_dumps "$fixture_root/ye3t-parity.json" "$fixture_root/cpu-scalar.dump" \
    "$fixture_root/cpu-direct.dump" "$fixture_root/gpu-direct.dump" \
    "$fixture_root/gpu-scalar.dump"
else
  verify_dumps "$fixture_root/ye3t-parity.json" "$fixture_root/cpu-scalar.dump" \
    "$fixture_root/cpu-direct.dump" "$fixture_root/gpu-scalar.dump"
fi
verify_numdiff "$fixture_root/gpu-scalar-numdiff.json" "$fixture_root/gpu-scalar-numdiff.log" \
  force_atol=1.0e-6 virial_atol=1.0e-2

if test -n "$pace_lmp"; then
  pace_input=$repo_root/examples/PACKAGES/ye3t/in.ye3t.high-rank-pace-product
  timeout 180 "$pace_lmp" -k on g 1 -pk kokkos neigh half -sf kk \
    -in "$pace_input" -var model "$model" \
    -var dump_path "$fixture_root/pace-kokkos.dump" \
    -log "$fixture_root/pace-kokkos.log" -screen none
  grep -q 'KOKKOS mode with Kokkos version' "$fixture_root/pace-kokkos.log"
  grep -q 'Product evaluator is used' "$fixture_root/pace-kokkos.log"
  grep -q 'pair pace/kk' "$fixture_root/pace-kokkos.log"
  verify_dumps "$fixture_root/pace-scalar-parity.json" "$fixture_root/pace-kokkos.dump" \
    "$fixture_root/gpu-scalar.dump"
fi

echo "PairYE3T scalar-power CPU/direct/Kokkos parity and finite differences passed."
if test -n "$pace_lmp"; then
  echo "PACE/Kokkos product versus PairYE3T/Kokkos scalar-power parity passed."
fi
if test "$keep_evidence" -eq 1; then
  echo "Retained scalar-power test evidence in $fixture_root"
fi
