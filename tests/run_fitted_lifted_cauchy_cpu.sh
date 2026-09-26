#!/bin/sh

set -eu

if test "$#" -ne 4 && test "$#" -ne 5; then
  echo "Usage: run_fitted_lifted_cauchy_cpu.sh YE3T_LAMMPS_ROOT LMP MODEL OUTPUT [REFERENCE]" >&2
  exit 2
fi

repo_root=$(CDPATH= cd -- "$1" && pwd -P)
lmp=$(CDPATH= cd -- "$(dirname -- "$2")" && pwd -P)/$(basename -- "$2")
model=$(CDPATH= cd -- "$(dirname -- "$3")" && pwd -P)/$(basename -- "$3")
output=$4
reference=${5:-}
mpi_exec=${YE3T_MPIEXEC:-mpiexec}
timeout_seconds=${YE3T_LIFTED_TEST_TIMEOUT:-120}
fixture=$repo_root/tests/fixtures/lifted_cauchy_ta_v1

if test ! -x "$lmp"; then
  echo "LAMMPS executable not found: $lmp" >&2
  exit 1
fi
if test ! -f "$model"; then
  echo "Fitted YE3T model not found: $model" >&2
  exit 1
fi
if test -n "$reference"; then
  reference=$(CDPATH= cd -- "$(dirname -- "$reference")" && pwd -P)/$(basename -- "$reference")
  if test ! -f "$reference"; then
    echo "Reference JSON not found: $reference" >&2
    exit 1
  fi
fi
if ! command -v "$mpi_exec" >/dev/null 2>&1; then
  echo "MPI launcher not found: $mpi_exec" >&2
  exit 1
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required for bounded validation runs" >&2
  exit 1
fi
case "$timeout_seconds" in
  *[!0-9]*|'') echo "YE3T_LIFTED_TEST_TIMEOUT must be a positive integer" >&2; exit 2 ;;
esac
if test "$timeout_seconds" -lt 1; then
  echo "YE3T_LIFTED_TEST_TIMEOUT must be a positive integer" >&2
  exit 2
fi
if test -e "$output"; then
  echo "Output path already exists: $output" >&2
  exit 1
fi
mkdir -p "$output"
output=$(CDPATH= cd -- "$output" && pwd -P)

run_case()
{
  ranks=$1
  label=$2
  input=$3
  shift 3
  timeout "$timeout_seconds" "$mpi_exec" -n "$ranks" "$lmp" \
    -screen "$output/screen.$label" -log "$output/log.$label" \
    -in "$input" -var model "$model" "$@"
}

for policy in direct factorized; do
  run_case 1 "numdiff.$policy.rank1" "$fixture/in.lifted_cauchy" \
    -var source_realization "$policy" \
    -var snapshot_path "$output/numdiff.$policy.rank1.snapshot.dump" \
    -var replay_path "$output/numdiff.$policy.rank1.replay.dump"
  for ranks in 1 2 4; do
    label=migration.$policy.rank$ranks
    run_case "$ranks" "$label" "$fixture/in.lifted_cauchy_mpi_migration" \
      -var mpi_ranks "$ranks" -var source_realization "$policy" \
      -var initial_dump_path "$output/$label.initial.dump" \
      -var final_dump_path "$output/$label.final.dump"
  done
done

if test -n "$reference"; then
  python3 "$repo_root/tests/validate_fitted_lifted_cauchy.py" "$output" \
    --reference "$reference" --json "$output/validation.json"
else
  python3 "$repo_root/tests/validate_fitted_lifted_cauchy.py" "$output" \
    --json "$output/validation.json"
fi
