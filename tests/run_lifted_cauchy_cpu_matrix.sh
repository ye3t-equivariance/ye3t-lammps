#!/bin/sh

set -eu

if test "$#" -lt 3 || test "$#" -gt 4; then
  echo "Usage: run_lifted_cauchy_cpu_matrix.sh YE3T_LAMMPS_ROOT LMP OUTPUT [PLUGIN]" >&2
  exit 2
fi

repo_root=$(CDPATH= cd -- "$1" && pwd -P)
lmp=$(CDPATH= cd -- "$(dirname -- "$2")" && pwd -P)/$(basename -- "$2")
output=$3
mpi_exec=${YE3T_MPIEXEC:-mpiexec}
timeout_seconds=${YE3T_LIFTED_TEST_TIMEOUT:-120}
load_plugin=0
plugin_path=none
if test "$#" -eq 4; then
  load_plugin=1
  plugin_path=$4
fi
ta_fixture=$repo_root/tests/fixtures/lifted_cauchy_ta_v1
multi_fixture=$repo_root/tests/fixtures/lifted_cauchy_ta_w_v1
if test ! -x "$lmp"; then
  echo "LAMMPS executable not found: $lmp" >&2
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
    -in "$input" -var load_plugin "$load_plugin" \
    -var plugin_path "$plugin_path" "$@"
}

run_case 1 multi.normal.direct.rank1 \
  "$multi_fixture/in.lifted_cauchy_multielement" \
  -var mpi_ranks 1 -var model "$multi_fixture/model.ye3t.json" \
  -var source_realization direct \
  -var dump_path "$output/multi.normal.direct.rank1.dump"
run_case 2 multi.normal.direct.rank2 \
  "$multi_fixture/in.lifted_cauchy_multielement" \
  -var mpi_ranks 2 -var model "$multi_fixture/model.ye3t.json" \
  -var source_realization direct \
  -var dump_path "$output/multi.normal.direct.rank2.dump"
run_case 1 multi.normal.factorized.rank1 \
  "$multi_fixture/in.lifted_cauchy_multielement" \
  -var mpi_ranks 1 -var model "$multi_fixture/model.ye3t.json" \
  -var source_realization factorized \
  -var dump_path "$output/multi.normal.factorized.rank1.dump"
run_case 2 multi.normal.factorized.rank2 \
  "$multi_fixture/in.lifted_cauchy_multielement" \
  -var mpi_ranks 2 -var model "$multi_fixture/model.ye3t.json" \
  -var source_realization factorized \
  -var dump_path "$output/multi.normal.factorized.rank2.dump"
run_case 1 multi.reversed.direct.rank1 \
  "$multi_fixture/in.lifted_cauchy_multielement" \
  -var mpi_ranks 1 -var model "$multi_fixture/model.ye3t.json" \
  -var source_realization direct -var ta_type 2 -var w_type 1 \
  -var element1 W -var element2 Ta \
  -var dump_path "$output/multi.reversed.direct.rank1.dump"
run_case 1 multi.reversed.factorized.rank1 \
  "$multi_fixture/in.lifted_cauchy_multielement" \
  -var mpi_ranks 1 -var model "$multi_fixture/model.ye3t.json" \
  -var source_realization factorized -var ta_type 2 -var w_type 1 \
  -var element1 W -var element2 Ta \
  -var dump_path "$output/multi.reversed.factorized.rank1.dump"

for policy in direct factorized; do
  for ranks in 1 2 4; do
    label=migration.$policy.rank$ranks
    run_case "$ranks" "$label" \
      "$ta_fixture/in.lifted_cauchy_mpi_migration" \
      -var mpi_ranks "$ranks" -var model "$ta_fixture/model.ye3t.json" \
      -var source_realization "$policy" \
      -var initial_dump_path "$output/$label.initial.dump" \
      -var final_dump_path "$output/$label.final.dump"
  done
done

run_case 1 nve.factorized.rank1.dt "$ta_fixture/in.lifted_cauchy_nve" \
  -var mpi_ranks 1 -var model "$ta_fixture/model.ye3t.json" \
  -var source_realization factorized \
  -var dump_path "$output/nve.factorized.rank1.dt.dump"
run_case 4 nve.factorized.rank4.dt "$ta_fixture/in.lifted_cauchy_nve" \
  -var mpi_ranks 4 -var model "$ta_fixture/model.ye3t.json" \
  -var source_realization factorized \
  -var dump_path "$output/nve.factorized.rank4.dt.dump"
run_case 1 nve.direct.rank1.dt "$ta_fixture/in.lifted_cauchy_nve" \
  -var mpi_ranks 1 -var model "$ta_fixture/model.ye3t.json" \
  -var source_realization direct \
  -var dump_path "$output/nve.direct.rank1.dt.dump"
run_case 1 nve.factorized.rank1.half_dt "$ta_fixture/in.lifted_cauchy_nve" \
  -var mpi_ranks 1 -var model "$ta_fixture/model.ye3t.json" \
  -var source_realization factorized -var integration_timestep 5.0e-6 \
  -var integration_steps 20 \
  -var dump_path "$output/nve.factorized.rank1.half_dt.dump"

for ranks in 1 4; do
  run_case "$ranks" "restart.rank$ranks.write" \
    "$ta_fixture/in.lifted_cauchy_restart_write" \
    -var mpi_ranks "$ranks" -var model "$ta_fixture/model.ye3t.json" \
    -var source_realization factorized \
    -var restart_path "$output/restart.rank$ranks.bin" \
    -var dump_path "$output/restart.rank$ranks.continuous.dump"
  run_case "$ranks" "restart.rank$ranks.read" \
    "$ta_fixture/in.lifted_cauchy_restart_read" \
    -var mpi_ranks "$ranks" -var model "$ta_fixture/model.ye3t.json" \
    -var source_realization factorized \
    -var restart_path "$output/restart.rank$ranks.bin" \
    -var dump_path "$output/restart.rank$ranks.rehydrated.dump"
done

for label in \
  migration.direct.rank1 migration.direct.rank2 migration.direct.rank4 \
  migration.factorized.rank1 migration.factorized.rank2 \
  migration.factorized.rank4; do
  test "$(grep -c 'attributes: full, newton on' "$output/log.$label")" -eq 1
done
grep -q 'source direct_q' "$output/log.multi.normal.direct.rank1"
grep -q 'source factorized_t' "$output/log.multi.normal.factorized.rank1"

if run_case 1 negative.newton_off \
  "$ta_fixture/in.lifted_cauchy_mpi_migration" \
  -var mpi_ranks 1 -var newton_mode off \
  -var model "$ta_fixture/model.ye3t.json" \
  -var source_realization direct \
  -var initial_dump_path "$output/negative.newton.initial.dump" \
  -var final_dump_path "$output/negative.newton.final.dump" >/dev/null 2>&1; then
  echo "Lifted PairYE3T accepted newton off" >&2
  exit 1
fi
grep -q 'requires newton pair on' "$output/log.negative.newton_off"
if run_case 1 negative.unsupported_element \
  "$multi_fixture/in.lifted_cauchy_multielement" \
  -var mpi_ranks 1 -var model "$multi_fixture/model.ye3t.json" \
  -var source_realization direct -var element1 Ta -var element2 H \
  -var dump_path "$output/negative.unsupported.dump" >/dev/null 2>&1; then
  echo "Two-element lifted PairYE3T accepted an unsupported element" >&2
  exit 1
fi
grep -q 'Element H is not present in YE3T potential' \
  "$output/log.negative.unsupported_element"

python3 "$repo_root/tests/test_lifted_cauchy_mpi.py" \
  "$multi_fixture/reference.json" "$output" > "$output/validation.json"
cat "$output/validation.json"
