#!/bin/sh

set -eu

if test "$#" -lt 3 || test "$#" -gt 4; then
  echo "Usage: test_kokkos_direct_cuda.sh YE3T_LAMMPS_ROOT YE3T_LMP MODEL_YACE [PACE_LMP]" >&2
  exit 2
fi

repo_root=$(CDPATH= cd -- "$1" && pwd -P)
ye3t_lmp=$(CDPATH= cd -- "$(dirname -- "$2")" && pwd -P)/$(basename -- "$2")
model=$(CDPATH= cd -- "$(dirname -- "$3")" && pwd -P)/$(basename -- "$3")
pace_lmp=
if test "$#" -eq 4; then
  pace_lmp=$(CDPATH= cd -- "$(dirname -- "$4")" && pwd -P)/$(basename -- "$4")
fi

if test ! -x "$ye3t_lmp" || test ! -f "$model"; then
  echo "The PairYE3T LAMMPS executable or model does not exist" >&2
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
fixture_root=$(mktemp -d "$tmp_base/ye3t-kokkos-direct-test.XXXXXX")
cleanup()
{
  case "$fixture_root" in
    "$tmp_base"/ye3t-kokkos-direct-test.*) rm -rf -- "$fixture_root" ;;
    *) echo "Refusing to remove unexpected fixture path: $fixture_root" >&2 ;;
  esac
}
trap cleanup EXIT HUP INT TERM

input=$repo_root/examples/PACKAGES/ye3t/in.ye3t.direct
verifier=$repo_root/examples/PACKAGES/ye3t/verify_examples.py
. "$repo_root/tests/example_verifier.sh"

timeout 120 "$ye3t_lmp" -in "$input" -var model "$model" \
  -var dump_path "$fixture_root/cpu.dump" -log "$fixture_root/cpu.log" \
  -screen none
timeout 120 "$ye3t_lmp" -k on g 1 -pk kokkos neigh half -sf kk \
  -in "$input" -var model "$model" \
  -var dump_path "$fixture_root/kokkos.dump" \
  -log "$fixture_root/kokkos.log" -screen none

grep -Eq 'YE3T Kokkos device-plan probe: execution_space (Cuda|HIP)' \
  "$fixture_root/kokkos.log"
grep -Eq '^YE3T Kokkos rank-device: world_rank 0, world_size 1,.*uuid_status available,.*pci_bus_id [^,]+,.*gpu_aware_mpi [01],' \
  "$fixture_root/kokkos.log"
grep -Eq '^YE3T Kokkos rank-device summary: world_size 1, node_count 1, rank_map [0-9a-f]{64}, uuid_status complete, device_class_consensus passed$' \
  "$fixture_root/kokkos.log"
if test "$(grep -c '^YE3T Kokkos capacity:' "$fixture_root/kokkos.log")" -ne 1; then
  echo "Expected exactly one initial Kokkos capacity record" >&2
  exit 1
fi
grep -Eq '^YE3T Kokkos capacity: world_rank 0, configured_chunk [1-9][0-9]*, local_inum [1-9][0-9]*, requested_chunk [1-9][0-9]*, effective_chunk [1-9][0-9]*,.*center_reduced 0, edge_reductions 0,.*exact_chunk_required 0$' \
  "$fixture_root/kokkos.log"
grep -q 'maximum_error 0' "$fixture_root/kokkos.log"
grep -q 'pair ye3t/kk' "$fixture_root/kokkos.log"
grep -q 'VJP_policy edge_grouped_harmonic_cached_radial_v1' \
  "$fixture_root/kokkos.log"
grep -Eq 'VJP_schedule [0-9a-f]{64}' "$fixture_root/kokkos.log"
grep -q 'VJP_harmonic_groups 15' "$fixture_root/kokkos.log"
grep -q 'VJP_angular_terms 45' "$fixture_root/kokkos.log"
grep -q 'VJP_maximum_group_terms 3' "$fixture_root/kokkos.log"
if grep -q 'YE3T CPU dispatch:' "$fixture_root/kokkos.log"; then
  echo "The Kokkos test entered the CPU evaluator" >&2
  exit 1
fi
cpu_plan=$(sed -n 's/.*direct_logical_plan \([0-9a-f]*\).*/\1/p' \
  "$fixture_root/cpu.log" | tail -n 1)
kokkos_plan=$(sed -n 's/.*direct_logical_plan \([0-9a-f]*\),.*/\1/p' \
  "$fixture_root/kokkos.log" | tail -n 1)
if test -z "$cpu_plan" || test "$cpu_plan" != "$kokkos_plan"; then
  echo "CPU/Kokkos direct logical-plan identity mismatch" >&2
  exit 1
fi
verify_dumps "$fixture_root/cpu-kokkos-parity.json" "$fixture_root/cpu.dump" \
  "$fixture_root/kokkos.dump"

if test -n "$pace_lmp"; then
  pace_input=$repo_root/examples/PACKAGES/ye3t/in.ye3t.pace-product
  timeout 120 "$pace_lmp" -k on g 1 -pk kokkos neigh half -sf kk \
    -in "$pace_input" -var model "$model" \
    -var dump_path "$fixture_root/pace-kokkos.dump" \
    -log "$fixture_root/pace-kokkos.log" -screen none
  grep -q 'KOKKOS mode with Kokkos version' "$fixture_root/pace-kokkos.log"
  grep -q 'Product evaluator is used' "$fixture_root/pace-kokkos.log"
  grep -q 'pair pace/kk' "$fixture_root/pace-kokkos.log"
  verify_dumps "$fixture_root/pace-kokkos-parity.json" "$fixture_root/pace-kokkos.dump" \
    "$fixture_root/kokkos.dump"
fi

echo "PairYE3T direct CPU/Kokkos parity passed."
if test -n "$pace_lmp"; then
  echo "PACE/Kokkos product parity passed."
fi
