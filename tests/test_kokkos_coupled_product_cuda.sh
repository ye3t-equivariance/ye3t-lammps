#!/bin/sh

set -eu

if test "$#" -lt 4 || test "$#" -gt 5; then
  echo "Usage: test_kokkos_coupled_product_cuda.sh YE3T_LAMMPS_ROOT YE3T_LMP MODEL_YACE PLAN_MANIFEST [PACE_LMP|--pure-coupled]" >&2
  exit 2
fi

repo_root=$(CDPATH= cd -- "$1" && pwd -P)
ye3t_lmp=$(CDPATH= cd -- "$(dirname -- "$2")" && pwd -P)/$(basename -- "$2")
model=$(CDPATH= cd -- "$(dirname -- "$3")" && pwd -P)/$(basename -- "$3")
manifest=$(CDPATH= cd -- "$(dirname -- "$4")" && pwd -P)/$(basename -- "$4")
pace_lmp=
pure_coupled=0
if test "$#" -eq 5; then
  if test "$5" = "--pure-coupled"; then
    pure_coupled=1
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
  fixture_root=$(mktemp -d "$tmp_base/ye3t-kokkos-coupled-test.XXXXXX")
fi
cleanup()
{
  if test "$keep_evidence" -eq 1; then
    return
  fi
  case "$fixture_root" in
    "$tmp_base"/ye3t-kokkos-coupled-test.*) rm -rf -- "$fixture_root" ;;
    *) echo "Refusing to remove unexpected fixture path: $fixture_root" >&2 ;;
  esac
}
trap cleanup EXIT HUP INT TERM

input=$repo_root/tests/fixtures/in.ye3t_kokkos_coupled_product
numdiff_input=$repo_root/examples/PACKAGES/ye3t/in.ye3t.numdiff
pace_input=$repo_root/examples/PACKAGES/ye3t/in.ye3t.pace-product
verifier=$repo_root/examples/PACKAGES/ye3t/verify_examples.py
. "$repo_root/tests/example_verifier.sh"
test_chunksize=17

timeout 180 "$ye3t_lmp" -in "$input" -var model "$model" \
  -var plan "$manifest" -var policy direct \
  -var ye3t_chunksize "$test_chunksize" \
  -var dump_path "$fixture_root/cpu-direct.dump" \
  -log "$fixture_root/cpu-direct.log" -screen none
timeout 180 "$ye3t_lmp" -in "$input" -var model "$model" \
  -var plan "$manifest" -var policy coupled_product \
  -var ye3t_chunksize "$test_chunksize" \
  -var dump_path "$fixture_root/cpu-coupled.dump" \
  -log "$fixture_root/cpu-coupled.log" -screen none
timeout 180 "$ye3t_lmp" -k on g 1 -pk kokkos neigh half -sf kk \
  -in "$input" -var model "$model" -var plan "$manifest" \
  -var policy direct -var ye3t_chunksize "$test_chunksize" \
  -var dump_path "$fixture_root/gpu-direct.dump" \
  -log "$fixture_root/gpu-direct.log" -screen none
timeout 180 "$ye3t_lmp" -k on g 1 -pk kokkos neigh half -sf kk \
  -in "$input" -var model "$model" -var plan "$manifest" \
  -var policy coupled_product -var ye3t_chunksize "$test_chunksize" \
  -var dump_path "$fixture_root/gpu-coupled.dump" \
  -log "$fixture_root/gpu-coupled.log" -screen none
timeout 240 "$ye3t_lmp" -k on g 1 -pk kokkos neigh half -sf kk \
  -in "$numdiff_input" -var model "$model" -var plan "$manifest" \
  -var policy coupled_product \
  -log "$fixture_root/gpu-coupled-numdiff.log" -screen none

grep -Eq 'YE3T Kokkos device-plan probe: execution_space (Cuda|HIP)' \
  "$fixture_root/gpu-coupled.log"
grep -q 'evaluator coupled_product' "$fixture_root/gpu-coupled.log"
grep -Eq 'coupled_plans [1-9][0-9]*' "$fixture_root/gpu-coupled.log"
grep -Eq 'coupled_component_storage [1-9][0-9]*' \
  "$fixture_root/gpu-coupled.log"
grep -q 'maximum_error 0' "$fixture_root/gpu-coupled.log"
grep -q 'coupled_math_probe passed' "$fixture_root/gpu-coupled.log"
grep -q "chunksize $test_chunksize" "$fixture_root/gpu-coupled.log"
grep -q 'pair ye3t/kk' "$fixture_root/gpu-coupled.log"
grep -q 'evaluator direct' "$fixture_root/gpu-direct.log"
if test "$pure_coupled" -eq 1; then
  grep -q 'direct_monomial_storage 0' "$fixture_root/gpu-coupled.log"
fi
if grep -q 'YE3T CPU dispatch:' "$fixture_root/gpu-coupled.log"; then
  echo "The Kokkos coupled-product test entered the CPU evaluator" >&2
  exit 1
fi

cpu_portfolio=$(sed -n 's/.*evaluator portfolio \([0-9a-f]*\).*/\1/p' \
  "$fixture_root/cpu-coupled.log" | tail -n 1)
gpu_portfolio=$(sed -n 's/.*evaluator portfolio \([0-9a-f]*\).*/\1/p' \
  "$fixture_root/gpu-coupled.log" | tail -n 1)
if test -z "$cpu_portfolio" || test "$cpu_portfolio" != "$gpu_portfolio"; then
  echo "CPU/GPU selected coupled-product portfolio identity mismatch" >&2
  exit 1
fi

direct_flat=$(sed -n '/evaluator direct,/s/.*flat_plan \([0-9a-f]*\).*/\1/p' \
  "$fixture_root/gpu-direct.log" | tail -n 1)
coupled_flat=$(sed -n '/evaluator coupled_product,/s/.*flat_plan \([0-9a-f]*\).*/\1/p' \
  "$fixture_root/gpu-coupled.log" | tail -n 1)
if test -z "$direct_flat" || test -z "$coupled_flat" || \
   test "$direct_flat" = "$coupled_flat"; then
  echo "Direct and coupled-product GPU-layout identities were not distinguished" >&2
  exit 1
fi

verify_dumps "$fixture_root/ye3t-parity.json" "$fixture_root/cpu-direct.dump" \
  "$fixture_root/cpu-coupled.dump" "$fixture_root/gpu-direct.dump" \
  "$fixture_root/gpu-coupled.dump"
verify_numdiff "$fixture_root/gpu-coupled-numdiff.json" "$fixture_root/gpu-coupled-numdiff.log" \
  force_atol=1.0e-6 virial_atol=1.0e-2

if test -n "$pace_lmp"; then
  timeout 180 "$pace_lmp" -k on g 1 -pk kokkos neigh half -sf kk \
    -in "$pace_input" -var model "$model" \
    -var dump_path "$fixture_root/pace-kokkos.dump" \
    -log "$fixture_root/pace-kokkos.log" -screen none
  grep -q 'KOKKOS mode with Kokkos version' "$fixture_root/pace-kokkos.log"
  grep -q 'Product evaluator is used' "$fixture_root/pace-kokkos.log"
  grep -q 'pair pace/kk' "$fixture_root/pace-kokkos.log"
  verify_dumps "$fixture_root/pace-coupled-parity.json" "$fixture_root/pace-kokkos.dump" \
    "$fixture_root/gpu-coupled.dump"
fi

echo "PairYE3T coupled-product CPU/direct/Kokkos parity and finite differences passed."
if test -n "$pace_lmp"; then
  echo "PACE/Kokkos product versus PairYE3T/Kokkos coupled-product parity passed."
fi
if test "$keep_evidence" -eq 1; then
  echo "Retained coupled-product test evidence in $fixture_root"
fi
