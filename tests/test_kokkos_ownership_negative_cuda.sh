#!/bin/sh

set -eu

if test "$#" -ne 3; then
  echo "Usage: test_kokkos_ownership_negative_cuda.sh YE3T_LAMMPS_ROOT LMP MODEL_YACE" >&2
  exit 2
fi

repo_root=$(CDPATH= cd -- "$1" && pwd -P)
lmp=$(CDPATH= cd -- "$(dirname -- "$2")" && pwd -P)/$(basename -- "$2")
model=$(CDPATH= cd -- "$(dirname -- "$3")" && pwd -P)/$(basename -- "$3")

if test ! -x "$lmp" || test ! -f "$model"; then
  echo "The PairYE3T LAMMPS executable or model does not exist" >&2
  exit 2
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required" >&2
  exit 2
fi

tmp_base=${TMPDIR:-/tmp}
fixture_root=$(mktemp -d "$tmp_base/ye3t-kokkos-negative-test.XXXXXX")
cleanup()
{
  case "$fixture_root" in
    "$tmp_base"/ye3t-kokkos-negative-test.*) rm -rf -- "$fixture_root" ;;
    *) echo "Refusing to remove unexpected fixture path: $fixture_root" >&2 ;;
  esac
}
trap cleanup EXIT HUP INT TERM

expect_failure()
{
  name=$1
  expected=$2
  shift 2
  set +e
  timeout 60 "$@" > "$fixture_root/$name.out" 2>&1
  status=$?
  set -e
  if test "$status" -eq 0 || test "$status" -eq 124; then
    cat "$fixture_root/$name.out" >&2
    echo "$name did not fail promptly as required" >&2
    exit 1
  fi
  if ! grep -Fq "$expected" "$fixture_root/$name.out"; then
    cat "$fixture_root/$name.out" >&2
    echo "$name failed for an unexpected reason" >&2
    exit 1
  fi
}

expect_failure newton_off "Pair style ye3t/kk requires newton pair on" \
  "$lmp" -k on g 1 -pk kokkos neigh half -sf kk \
  -var model "$model" \
  -in "$repo_root/tests/fixtures/in.ye3t_kokkos_newton_off" \
  -log "$fixture_root/newton_off.log"

expect_failure full_neighbor \
  "Must use 'newton off' with KOKKOS package option 'neigh full'" \
  "$lmp" -k on g 1 -pk kokkos neigh full -sf kk \
  -var model "$model" \
  -in "$repo_root/tests/fixtures/in.ye3t_kokkos_b0_probe" \
  -log "$fixture_root/full_neighbor.log"

echo "PairYE3T/Kokkos ownership negative-mode checks passed."
