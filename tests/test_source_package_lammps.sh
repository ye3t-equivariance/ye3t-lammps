#!/bin/sh

set -eu

if test "$#" -lt 3 || test "$#" -gt 4; then
  echo "Usage: test_source_package_lammps.sh YE3T_LAMMPS_ROOT LAMMPS_GIT_ROOT YE3T_ROOT [PACE_ROOT]" >&2
  exit 2
fi

repo_root=$(CDPATH= cd -- "$1" && pwd -P)
lammps_reference=$(CDPATH= cd -- "$2" && pwd -P)
ye3t_root=$(CDPATH= cd -- "$3" && pwd -P)
pace_root=
if test "$#" -eq 4; then
  pace_root=$(CDPATH= cd -- "$4" && pwd -P)
fi
tmp_base=${TMPDIR:-/tmp}
fixture_root=$(mktemp -d "$tmp_base/ye3t-lammps-source-test.XXXXXX")
cleanup()
{
  case "$fixture_root" in
    "$tmp_base"/ye3t-lammps-source-test.*) rm -rf -- "$fixture_root" ;;
    *) echo "Refusing to remove unexpected fixture path: $fixture_root" >&2 ;;
  esac
}
trap cleanup EXIT HUP INT TERM

lammps_checkout=$fixture_root/lammps
build_dir=$fixture_root/build

git clone --quiet --shared "$lammps_reference" "$lammps_checkout"
"$repo_root/tools/patch_lammps.sh" --apply --lammps-source "$lammps_checkout"

set -- -S "$lammps_checkout/cmake" -B "$build_dir" -G Ninja \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -D BUILD_MPI=ON \
  -D PKG_EXTRA-FIX=ON \
  -D PKG_ML-YE3T=ON \
  -D ML_YE3T_RUNTIME_SOURCE="$ye3t_root"
if test -n "$pace_root"; then
  set -- "$@" -D PKG_ML-PACE=ON -D LOCAL_ML-PACE="$pace_root"
fi
cmake "$@"
cmake --build "$build_dir" --target lmp --parallel "${YE3T_BUILD_JOBS:-4}"

grep -q '^PKG_EXTRA-FIX:BOOL=ON$' "$build_dir/CMakeCache.txt"
grep -q '^PKG_ML-YE3T:BOOL=ON$' "$build_dir/CMakeCache.txt"
if test -n "$pace_root"; then
  grep -q '^PKG_ML-PACE:BOOL=ON$' "$build_dir/CMakeCache.txt"
fi
"$build_dir/lmp" -help > "$fixture_root/lammps-help.txt"
grep -Eq '(^|[[:space:]])ye3t([[:space:]]|$)' "$fixture_root/lammps-help.txt"
if test -n "$pace_root"; then
  grep -Eq '(^|[[:space:]])pace([[:space:]]|$)' "$fixture_root/lammps-help.txt"
fi

lifted_fixture=$repo_root/tests/fixtures/lifted_cauchy_ta_v1
lifted_output=$fixture_root/lifted-cauchy
mkdir -p "$lifted_output"
for source_realization in direct factorized; do
  "$build_dir/lmp" -screen "$lifted_output/screen.$source_realization" \
    -log "$lifted_output/log.$source_realization" \
    -in "$lifted_fixture/in.lifted_cauchy" \
    -var model "$lifted_fixture/model.ye3t.json" \
    -var source_realization "$source_realization" \
    -var snapshot_path "$lifted_output/$source_realization.snapshot.dump" \
    -var replay_path "$lifted_output/$source_realization.replay.dump"
  "$build_dir/lmp" -screen "$lifted_output/screen.$source_realization.isolated" \
    -log "$lifted_output/log.$source_realization.isolated" \
    -in "$lifted_fixture/in.lifted_cauchy_isolated" \
    -var model "$lifted_fixture/model.ye3t.json" \
    -var source_realization "$source_realization" \
    -var dump_path "$lifted_output/$source_realization.isolated.dump"
done

grep -q 'source direct_q' "$lifted_output/log.direct"
grep -q 'source factorized_t' "$lifted_output/log.factorized"

"$build_dir/lmp" -screen "$lifted_output/screen.model_default" \
  -log "$lifted_output/log.model_default" \
  -in "$lifted_fixture/in.lifted_cauchy" \
  -var model "$lifted_fixture/model.ye3t.json" \
  -var source_realization model_default \
  -var snapshot_path "$lifted_output/model_default.snapshot.dump" \
  -var replay_path "$lifted_output/model_default.replay.dump"
"$build_dir/lmp" -screen "$lifted_output/screen.direct_unsorted" \
  -log "$lifted_output/log.direct_unsorted" \
  -in "$lifted_fixture/in.lifted_cauchy" \
  -var model "$lifted_fixture/model.ye3t.json" \
  -var source_realization direct -var sort_interval 0 \
  -var snapshot_path "$lifted_output/direct_unsorted.snapshot.dump" \
  -var replay_path "$lifted_output/direct_unsorted.replay.dump"
grep -q 'source factorized_t' "$lifted_output/log.model_default"
grep -q 'source direct_q' "$lifted_output/log.direct_unsorted"

python3 "$repo_root/tests/test_lifted_cauchy_lammps.py" \
  "$lifted_fixture/reference.json" "$lifted_output" \
  > "$lifted_output/validation.json"
cat "$lifted_output/validation.json"

sh "$repo_root/tests/run_lifted_cauchy_cpu_matrix.sh" \
  "$repo_root" "$build_dir/lmp" "$fixture_root/lifted-cauchy-cpu-matrix"

if "$build_dir/lmp" -screen none -log none \
  -in "$lifted_fixture/in.lifted_cauchy" \
  -var model "$lifted_fixture/model.ye3t.json" \
  -var source_realization invalid \
  -var snapshot_path "$lifted_output/invalid.snapshot.dump" \
  -var replay_path "$lifted_output/invalid.replay.dump" >/dev/null 2>&1; then
  echo "Invalid lifted source realization was accepted" >&2
  exit 1
fi
if "$build_dir/lmp" -screen none -log none \
  -in "$lifted_fixture/in.lifted_cauchy" \
  -var model "$lifted_fixture/model.ye3t.json" \
  -var model_family yace -var source_realization model_default \
  -var snapshot_path "$lifted_output/ordinary-source.snapshot.dump" \
  -var replay_path "$lifted_output/ordinary-source.replay.dump" >/dev/null 2>&1; then
  echo "Ordinary YACE accepted an explicitly supplied lifted source realization" >&2
  exit 1
fi
if "$build_dir/lmp" -screen none -log none \
  -in "$lifted_fixture/in.lifted_cauchy" \
  -var model "$lifted_fixture/model.ye3t.json" \
  -var source_realization direct -var block_policy block \
  -var snapshot_path "$lifted_output/block.snapshot.dump" \
  -var replay_path "$lifted_output/block.replay.dump" >/dev/null 2>&1; then
  echo "Lifted source accepted a non-direct descriptor evaluator" >&2
  exit 1
fi
if "$build_dir/lmp" -screen none -log none \
  -in "$lifted_fixture/in.lifted_cauchy" \
  -var model "$lifted_fixture/model.ye3t.json" \
  -var source_realization direct -var element W \
  -var snapshot_path "$lifted_output/wrong-element.snapshot.dump" \
  -var replay_path "$lifted_output/wrong-element.replay.dump" >/dev/null 2>&1; then
  echo "Lifted source accepted an element absent from the model" >&2
  exit 1
fi

installed_examples=$lammps_checkout/examples/PACKAGES/ye3t
cmp "$repo_root/examples/PACKAGES/ye3t/YE3T_EXAMPLE_MANIFEST.sha256" \
  "$installed_examples/YE3T_EXAMPLE_MANIFEST.sha256"
test -x "$installed_examples/verify_examples.py"

run_public_examples()
{
  run_ranks=$1
  run_output=$2
  with_pace=no
  if test -n "$pace_root"; then
    with_pace=yes
  fi
  YE3T_LMP=$build_dir/lmp YE3T_MPI_RANKS=$run_ranks \
    YE3T_EXAMPLE_OUTPUT=$run_output YE3T_WITH_NUMDIFF=yes \
    YE3T_WITH_HIGH_RANK=yes YE3T_WITH_PACE=$with_pace \
    "$installed_examples/run_examples.sh"
}

run_public_examples 1 "$fixture_root/examples-rank1"
run_public_examples 4 "$fixture_root/examples-rank4"

verifier=$installed_examples/verify_examples.py
. "$repo_root/tests/example_verifier.sh"
verify_dumps "$fixture_root/rank1_rank4_parity.json" \
  "$fixture_root/examples-rank1/compact_direct.dump" \
  "$fixture_root/examples-rank4/compact_direct.dump"

grep -q 'Loaded YE3T YACE potential' "$fixture_root/examples-rank1/log.compact_direct"
grep -q 'YE3T CPU dispatch:' "$fixture_root/examples-rank1/log.compact_direct"

python3 "$repo_root/tests/test_scalar_power_tamper.py" \
  --lmp "$build_dir/lmp" \
  --input "$repo_root/tests/fixtures/high_rank/in.ye3t.high-rank-scalar-power" \
  --model "$installed_examples/models/ta_l8_h16/model.yace" \
  --manifest "$installed_examples/models/ta_l8_h16/manifest.json"

echo "ML-YE3T source-package configure/build/registration/run test passed."
