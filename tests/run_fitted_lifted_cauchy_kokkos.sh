#!/bin/sh

set -eu

if test "$#" -ne 4; then
  echo "Usage: run_fitted_lifted_cauchy_kokkos.sh YE3T_LAMMPS_ROOT LMP MODEL OUTPUT" >&2
  exit 2
fi

repo_root=$(CDPATH= cd -- "$1" && pwd -P)
lmp=$(CDPATH= cd -- "$(dirname -- "$2")" && pwd -P)/$(basename -- "$2")
model=$(CDPATH= cd -- "$(dirname -- "$3")" && pwd -P)/$(basename -- "$3")
output=$4
mpi_exec=${YE3T_MPIEXEC:-mpiexec}
timeout_seconds=${YE3T_LIFTED_KOKKOS_TEST_TIMEOUT:-180}
fixture=$repo_root/tests/fixtures/lifted_cauchy_ta_v1
multi_fixture=$repo_root/tests/fixtures/lifted_cauchy_ta_w_v1

if test ! -x "$lmp" || test ! -f "$model"; then
  echo "The LAMMPS executable or fitted YE3T model does not exist" >&2
  exit 1
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
  *[!0-9]*|'') echo "YE3T_LIFTED_KOKKOS_TEST_TIMEOUT must be positive" >&2; exit 2 ;;
esac
if test "$timeout_seconds" -lt 1; then
  echo "YE3T_LIFTED_KOKKOS_TEST_TIMEOUT must be positive" >&2
  exit 2
fi
if test -e "$output"; then
  echo "Output path already exists: $output" >&2
  exit 1
fi
mkdir -p "$output"
output=$(CDPATH= cd -- "$output" && pwd -P)

"$lmp" -help > "$output/lammps-help.txt"
if ! grep -Eq '(^|[[:space:]])ye3t/kk([[:space:]]|$)' "$output/lammps-help.txt"; then
  echo "The selected LAMMPS executable does not contain pair_style ye3t/kk" >&2
  exit 1
fi

run_cpu()
{
  label=$1
  input=$2
  case_model=$3
  shift 3
  timeout "$timeout_seconds" "$lmp" -screen "$output/screen.$label" \
    -log "$output/log.$label" -in "$input" -var model "$case_model" "$@"
}

run_kokkos()
{
  ranks=$1
  label=$2
  input=$3
  case_model=$4
  shift 4
  if test "$ranks" -eq 1; then
    timeout "$timeout_seconds" "$lmp" -k on g 1 -pk kokkos neigh half \
      -sf kk -screen "$output/screen.$label" -log "$output/log.$label" \
      -in "$input" -var model "$case_model" "$@"
  else
    timeout "$timeout_seconds" "$mpi_exec" -n "$ranks" "$lmp" \
      -k on g 1 -pk kokkos neigh half -sf kk \
      -screen "$output/screen.$label" -log "$output/log.$label" \
      -in "$input" -var model "$case_model" "$@"
  fi
  if ! grep -q 'YE3T lifted Kokkos dispatch: source direct_q' "$output/log.$label" ||
     grep -q 'YE3T lifted CPU dispatch:' "$output/log.$label"; then
    echo "The requested lifted PairYE3T/Kokkos path was not dispatched: $label" >&2
    exit 1
  fi
}

run_cpu cpu.numdiff "$fixture/in.lifted_cauchy" "$model" \
  -var source_realization direct \
  -var snapshot_path "$output/cpu.numdiff.snapshot.dump" \
  -var replay_path "$output/cpu.numdiff.replay.dump"
run_kokkos 1 kk.numdiff "$fixture/in.lifted_cauchy" "$model" \
  -var source_realization direct \
  -var snapshot_path "$output/kk.numdiff.snapshot.dump" \
  -var replay_path "$output/kk.numdiff.replay.dump"

for ranks in 1 2; do
  label=kk.migration.rank$ranks
  run_kokkos "$ranks" "$label" "$fixture/in.lifted_cauchy_mpi_migration" \
    "$model" \
    -var mpi_ranks "$ranks" -var source_realization direct \
    -var initial_dump_path "$output/$label.initial.dump" \
    -var final_dump_path "$output/$label.final.dump"
done

run_cpu cpu.multi.normal "$multi_fixture/in.lifted_cauchy_multielement" \
  "$multi_fixture/model.ye3t.json" -var mpi_ranks 1 \
  -var source_realization direct \
  -var dump_path "$output/cpu.multi.normal.dump"
for ranks in 1 2; do
  label=kk.multi.normal.rank$ranks
  run_kokkos "$ranks" "$label" \
    "$multi_fixture/in.lifted_cauchy_multielement" \
    "$multi_fixture/model.ye3t.json" -var mpi_ranks "$ranks" \
    -var source_realization direct \
    -var dump_path "$output/$label.dump"
done
run_kokkos 1 kk.multi.reversed.rank1 \
  "$multi_fixture/in.lifted_cauchy_multielement" \
  "$multi_fixture/model.ye3t.json" -var mpi_ranks 1 \
  -var source_realization direct -var ta_type 2 -var w_type 1 \
  -var element1 W -var element2 Ta \
  -var dump_path "$output/kk.multi.reversed.rank1.dump"

if timeout "$timeout_seconds" "$lmp" -k on g 1 -pk kokkos neigh half \
  -sf kk -screen "$output/screen.negative.factorized" \
  -log "$output/log.negative.factorized" -in "$fixture/in.lifted_cauchy" \
  -var model "$model" -var source_realization factorized \
  -var snapshot_path "$output/negative.factorized.snapshot.dump" \
  -var replay_path "$output/negative.factorized.replay.dump" >/dev/null 2>&1; then
  echo "Lifted PairYE3T/Kokkos accepted the unimplemented factorized path" >&2
  exit 1
fi
grep -q 'currently supports source_realization direct' \
  "$output/log.negative.factorized"

if timeout "$timeout_seconds" "$lmp" -k on g 1 -pk kokkos neigh half \
  -sf kk -screen "$output/screen.negative.newton_off" \
  -log "$output/log.negative.newton_off" \
  -in "$fixture/in.lifted_cauchy_mpi_migration" \
  -var model "$model" -var mpi_ranks 1 -var newton_mode off \
  -var source_realization direct \
  -var initial_dump_path "$output/negative.newton.initial.dump" \
  -var final_dump_path "$output/negative.newton.final.dump" >/dev/null 2>&1; then
  echo "Lifted PairYE3T/Kokkos accepted newton off" >&2
  exit 1
fi
grep -q 'requires newton pair on' "$output/log.negative.newton_off"

python3 "$repo_root/tests/validate_fitted_lifted_cauchy_kokkos.py" "$output" \
  --json "$output/validation.json"
