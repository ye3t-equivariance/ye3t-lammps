#!/bin/sh

set -eu

if test "$#" -ne 3; then
  echo "Usage: test_kokkos_b0_cuda.sh YE3T_LAMMPS_ROOT LMP MODEL_YACE" >&2
  exit 2
fi

repo_root=$(CDPATH= cd -- "$1" && pwd -P)
lmp=$(CDPATH= cd -- "$(dirname -- "$2")" && pwd -P)/$(basename -- "$2")
model=$(CDPATH= cd -- "$(dirname -- "$3")" && pwd -P)/$(basename -- "$3")

if test ! -x "$lmp" || test ! -f "$model"; then
  echo "The LAMMPS executable or model does not exist" >&2
  exit 2
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required" >&2
  exit 2
fi
timeout_seconds=${YE3T_TEST_TIMEOUT:-120}
case "$timeout_seconds" in
  *[!0-9]*|'') echo "YE3T_TEST_TIMEOUT must be a positive integer" >&2; exit 2 ;;
esac
if test "$timeout_seconds" -lt 1; then
  echo "YE3T_TEST_TIMEOUT must be a positive integer" >&2
  exit 2
fi

tmp_base=${TMPDIR:-/tmp}
fixture_root=$(mktemp -d "$tmp_base/ye3t-kokkos-b0-test.XXXXXX")
cleanup()
{
  case "$fixture_root" in
    "$tmp_base"/ye3t-kokkos-b0-test.*) rm -rf -- "$fixture_root" ;;
    *) echo "Refusing to remove unexpected fixture path: $fixture_root" >&2 ;;
  esac
}
trap cleanup EXIT HUP INT TERM

set +e
timeout "$timeout_seconds" "$lmp" -k on g 1 -pk kokkos neigh half -sf kk \
  -var model "$model" \
  -in "$repo_root/tests/fixtures/in.ye3t_kokkos_b0_probe" \
  -log "$fixture_root/log.lammps" \
  > "$fixture_root/output.txt" 2>&1
status=$?
set -e

if test "$status" -ne 0; then
  cat "$fixture_root/output.txt" >&2
  if test "$status" -eq 124; then
    echo "The Kokkos device-plan smoke exceeded $timeout_seconds seconds" >&2
    exit 1
  fi
  echo "The Kokkos device-plan and numerical smoke failed" >&2
  exit 1
fi

grep -Eq 'YE3T Kokkos device-plan probe: execution_space (Cuda|HIP)' \
  "$fixture_root/output.txt"
grep -q 'maximum_error 0' "$fixture_root/output.txt"
grep -q 'attributes: full, newton on, kokkos_device' "$fixture_root/output.txt"
grep -q 'pair ye3t/kk' "$fixture_root/output.txt"
if grep -q 'YE3T CPU dispatch:' "$fixture_root/output.txt"; then
  echo "The Kokkos device probe entered the CPU evaluator" >&2
  exit 1
fi

echo "PairYE3T/Kokkos CUDA upload, device dispatch, and run-0 smoke passed."
