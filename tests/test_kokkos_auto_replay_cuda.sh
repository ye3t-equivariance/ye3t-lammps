#!/bin/sh

set -eu

if test "$#" -lt 5 || test "$#" -gt 6; then
  echo "Usage: test_kokkos_auto_replay_cuda.sh ROOT LMP MODEL MANIFEST REPLAY [PACE_LMP]" >&2
  exit 2
fi

repo_root=$(CDPATH= cd -- "$1" && pwd -P)
lmp=$(CDPATH= cd -- "$(dirname -- "$2")" && pwd -P)/$(basename -- "$2")
model=$(CDPATH= cd -- "$(dirname -- "$3")" && pwd -P)/$(basename -- "$3")
manifest=$(CDPATH= cd -- "$(dirname -- "$4")" && pwd -P)/$(basename -- "$4")
replay=$(CDPATH= cd -- "$(dirname -- "$5")" && pwd -P)/$(basename -- "$5")
pace_lmp=
if test "$#" -eq 6; then
  pace_lmp=$(CDPATH= cd -- "$(dirname -- "$6")" && pwd -P)/$(basename -- "$6")
fi

for path in "$lmp" "$model" "$manifest" "$replay"; do
  if test ! -e "$path"; then
    echo "Missing AUTO replay test input: $path" >&2
    exit 2
  fi
done
for command in timeout python3 mpiexec; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "$command is required" >&2
    exit 2
  fi
done

tmp_base=${TMPDIR:-/tmp}
fixture_root=$(mktemp -d "$tmp_base/ye3t-kokkos-auto-replay-test.XXXXXX")
cleanup()
{
  case "$fixture_root" in
    "$tmp_base"/ye3t-kokkos-auto-replay-test.*) rm -rf -- "$fixture_root" ;;
    *) echo "Refusing to remove unexpected fixture path: $fixture_root" >&2 ;;
  esac
}
trap cleanup EXIT HUP INT TERM

direct_input=$repo_root/examples/PACKAGES/ye3t/in.ye3t.direct
replay_input=$repo_root/tools/gpu_scaling/in.ye3t.auto-replay
numdiff_input=$repo_root/tools/gpu_scaling/in.ye3t.numdiff-auto-replay
verifier=$repo_root/examples/PACKAGES/ye3t/verify_examples.py
. "$repo_root/tests/example_verifier.sh"

timeout 120 "$lmp" -k on g 1 -pk kokkos neigh half -sf kk \
  -in "$direct_input" -var model "$model" \
  -var cells 4 \
  -var dump_path "$fixture_root/direct.dump" \
  -log "$fixture_root/direct.log" -screen none
timeout 120 "$lmp" -k on g 1 -pk kokkos neigh half -sf kk \
  -in "$replay_input" -var model "$model" -var plan "$manifest" \
  -var cells 4 \
  -var auto_replay "$replay" -var dump_path "$fixture_root/replay.dump" \
  -log "$fixture_root/replay.log" -screen none

grep -Eq 'evaluator auto\[(direct|direct\+block|direct\+scalar_power|direct\+coupled_product|block|scalar_power|coupled_product)\]' \
  "$fixture_root/replay.log"
grep -Fq 'planner_profile kokkos_gpu_device_replay_v1' "$fixture_root/replay.log"
grep -Fq 'planner_algorithm device_bound_candidate_replay_v1' \
  "$fixture_root/replay.log"
grep -Fq 'YE3T Kokkos AUTO replay workload:' "$fixture_root/replay.log"
verify_dumps "$fixture_root/direct-replay-parity.json" "$fixture_root/direct.dump" \
  "$fixture_root/replay.dump"

python3 "$repo_root/tests/tamper_kokkos_auto_replay.py" "$replay" \
  "$fixture_root/wrong-executable-replay.json"
if timeout 120 "$lmp" -k on g 1 -pk kokkos neigh half -sf kk \
  -in "$replay_input" -var model "$model" -var plan "$manifest" \
  -var cells 4 \
  -var auto_replay "$fixture_root/wrong-executable-replay.json" \
  -var dump_path "$fixture_root/wrong-executable.dump" \
  -log "$fixture_root/wrong-executable.log" -screen none; then
  echo "PairYE3T/Kokkos accepted an AUTO replay for different executable bytes" >&2
  exit 1
fi
grep -Fq 'AUTO replay was calibrated for a different LAMMPS executable' \
  "$fixture_root/wrong-executable.log"

python3 "$repo_root/tests/tamper_kokkos_auto_replay.py" "$replay" \
  "$fixture_root/wrong-device-replay.json" device
if timeout 120 "$lmp" -k on g 1 -pk kokkos neigh half -sf kk \
  -in "$replay_input" -var model "$model" -var plan "$manifest" \
  -var cells 4 \
  -var auto_replay "$fixture_root/wrong-device-replay.json" \
  -var dump_path "$fixture_root/wrong-device.dump" \
  -log "$fixture_root/wrong-device.log" -screen none; then
  echo "PairYE3T/Kokkos accepted an AUTO replay for a different device class" >&2
  exit 1
fi
grep -Fq 'AUTO replay was calibrated for a different device class' \
  "$fixture_root/wrong-device.log"

timeout 180 "$lmp" -k on g 1 -pk kokkos neigh half -sf kk \
  -in "$numdiff_input" -var model "$model" -var plan "$manifest" \
  -var auto_replay "$replay" -log "$fixture_root/numdiff.log" -screen none
verify_numdiff "$fixture_root/numdiff.json" "$fixture_root/numdiff.log"

timeout 120 mpiexec -n 2 "$lmp" -k on g 1 -pk kokkos neigh half -sf kk \
  -in "$replay_input" -var model "$model" -var plan "$manifest" \
  -var cells 4 \
  -var auto_replay "$replay" -var dump_path "$fixture_root/replay-rank2.dump" \
  -log "$fixture_root/replay-rank2.log" -screen none
verify_dumps "$fixture_root/replay-rank-parity.json" "$fixture_root/replay.dump" \
  "$fixture_root/replay-rank2.dump"

if test -n "$pace_lmp"; then
  timeout 120 "$pace_lmp" -k on g 1 -pk kokkos neigh half -sf kk \
    -in "$repo_root/examples/PACKAGES/ye3t/in.ye3t.pace-product" \
    -var model "$model" -var cells 4 \
    -var dump_path "$fixture_root/pace.dump" \
    -log "$fixture_root/pace.log" -screen none
  verify_dumps "$fixture_root/pace-replay-parity.json" "$fixture_root/pace.dump" \
    "$fixture_root/replay.dump"
fi

echo "PairYE3T/Kokkos calibrated AUTO replay parity, finite differences, and rank-2 domain crossing passed."
