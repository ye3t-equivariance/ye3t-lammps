#!/bin/sh
# Bounded correctness matrix for the shipped ML-YE3T examples.
#
# The runner takes no command-line arguments. Configure it with environment
# variables, for example:
#
#   YE3T_LMP=/path/to/lmp YE3T_MPI_RANKS=4 YE3T_WITH_PACE=yes ./run_examples.sh
#
#   YE3T_LMP               LAMMPS executable containing pair_style ye3t (required)
#   YE3T_MPI_RANKS         MPI ranks for the primary matrix (default 1)
#   YE3T_MPIEXEC           MPI launcher (default mpiexec)
#   YE3T_EXAMPLE_OUTPUT    new output directory (default generated/run-PID)
#   YE3T_EXAMPLE_TIMEOUT   timeout in seconds for each LAMMPS run (default 120)
#   YE3T_WITH_PACE         yes: run the same-model ML-PACE comparisons
#   YE3T_WITH_NUMDIFF      yes: run the force/virial finite-difference checks
#   YE3T_WITH_HIGH_RANK    yes: run the rank-16 direct/block execution checks
#   YE3T_WITH_KOKKOS       yes: run the direct, block, AUTO, and coupled-product
#                          Kokkos checks against the CPU results
#   YE3T_KOKKOS_GPUS       GPUs per node passed to Kokkos (default 1)
#
# Every LAMMPS log, dump, and JSON verification report is written to the
# output directory. The script exits nonzero on the first failed check.

set -eu

lmp=${YE3T_LMP:-}
mpi_ranks=${YE3T_MPI_RANKS:-1}
mpi_exec=${YE3T_MPIEXEC:-mpiexec}
output=${YE3T_EXAMPLE_OUTPUT:-}
timeout_seconds=${YE3T_EXAMPLE_TIMEOUT:-120}
with_pace=${YE3T_WITH_PACE:-no}
with_numdiff=${YE3T_WITH_NUMDIFF:-no}
with_high_rank=${YE3T_WITH_HIGH_RANK:-no}
with_kokkos=${YE3T_WITH_KOKKOS:-no}
kokkos_gpus=${YE3T_KOKKOS_GPUS:-1}

if test -z "$lmp"; then
  echo "Set YE3T_LMP to the LAMMPS executable" >&2
  exit 2
fi
case "$mpi_ranks" in *[!0-9]*|'') echo "YE3T_MPI_RANKS must be a positive integer" >&2; exit 2 ;; esac
case "$timeout_seconds" in *[!0-9]*|'') echo "YE3T_EXAMPLE_TIMEOUT must be a positive integer" >&2; exit 2 ;; esac
case "$kokkos_gpus" in *[!0-9]*|'') echo "YE3T_KOKKOS_GPUS must be a positive integer" >&2; exit 2 ;; esac
if test "$mpi_ranks" -lt 1 || test "$timeout_seconds" -lt 1 ||
   test "$kokkos_gpus" -lt 1; then
  echo "MPI ranks, timeout, and Kokkos GPU count must be positive" >&2
  exit 2
fi
for switch in "$with_pace" "$with_numdiff" "$with_high_rank" "$with_kokkos"; do
  case "$switch" in yes|no) ;; *) echo "YE3T_WITH_* switches must be yes or no" >&2; exit 2 ;; esac
done

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
lmp=$(CDPATH= cd -- "$(dirname -- "$lmp")" && pwd -P)/$(basename -- "$lmp")
if test ! -x "$lmp"; then
  echo "LAMMPS executable not found: $lmp" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required for result verification" >&2
  exit 1
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required for bounded example runs" >&2
  exit 1
fi
if test "$mpi_ranks" -gt 1 && ! command -v "$mpi_exec" >/dev/null 2>&1; then
  echo "MPI launcher not found: $mpi_exec" >&2
  exit 1
fi

if test -z "$output"; then
  output=$script_dir/generated/run-$$
fi
if test -e "$output"; then
  echo "Output path already exists: $output" >&2
  exit 1
fi
mkdir -p "$output"
output=$(CDPATH= cd -- "$output" && pwd -P)

"$lmp" -help > "$output/lammps-help.txt"
if ! grep -Eq '(^|[[:space:]])ye3t([[:space:]]|$)' "$output/lammps-help.txt"; then
  echo "The selected LAMMPS binary does not contain pair_style ye3t" >&2
  exit 1
fi
if test "$with_pace" = yes &&
   ! grep -Eq '(^|[[:space:]])pace([[:space:]]|$)' "$output/lammps-help.txt"; then
  echo "YE3T_WITH_PACE requires a LAMMPS binary containing pair_style pace" >&2
  exit 1
fi
if test "$with_kokkos" = yes &&
   ! grep -Eq '(^|[[:space:]])ye3t/kk([[:space:]]|$)' "$output/lammps-help.txt"; then
  echo "YE3T_WITH_KOKKOS requires a LAMMPS binary containing pair_style ye3t/kk" >&2
  exit 1
fi
if test "$with_numdiff" = yes &&
   ! grep -Eq '(^|[[:space:]])numdiff([[:space:]]|$)' "$output/lammps-help.txt"; then
  echo "YE3T_WITH_NUMDIFF requires LAMMPS package EXTRA-FIX" >&2
  exit 1
fi

run_at()
{
  run_ranks=$1
  tag=$2
  input=$3
  shift 3
  log=$output/log.$tag
  screen=$output/screen.$tag
  echo "Running $tag on $run_ranks rank(s)"
  if test "$run_ranks" -eq 1; then
    timeout "$timeout_seconds" "$lmp" -in "$script_dir/$input" \
      -log "$log" -screen "$screen" "$@"
  else
    timeout "$timeout_seconds" "$mpi_exec" -n "$run_ranks" "$lmp" \
      -in "$script_dir/$input" -log "$log" -screen "$screen" "$@"
  fi
}

run_at_kokkos()
{
  run_ranks=$1
  tag=$2
  input=$3
  shift 3
  log=$output/log.$tag
  screen=$output/screen.$tag
  echo "Running $tag on $run_ranks rank(s), $kokkos_gpus Kokkos GPU(s) per node"
  if test "$run_ranks" -eq 1; then
    timeout "$timeout_seconds" "$lmp" -k on g "$kokkos_gpus" \
      -pk kokkos neigh half -sf kk -in "$script_dir/$input" \
      -log "$log" -screen "$screen" "$@"
  else
    timeout "$timeout_seconds" "$mpi_exec" -n "$run_ranks" "$lmp" \
      -k on g "$kokkos_gpus" -pk kokkos neigh half -sf kk \
      -in "$script_dir/$input" -log "$log" -screen "$screen" "$@"
  fi
}

# Write a verify_examples.py configuration and run the named check. The
# first argument is the report name; the remaining arguments are the JSON
# fields, each given as key=value where a value starting with '[' is a JSON
# array and every other value is quoted as a string.
verify()
{
  report=$1
  shift
  config=$output/$report.config.json
  {
    printf '{'
    separator=
    for field in "$@"; do
      key=${field%%=*}
      value=${field#*=}
      case "$value" in
        \[*) printf '%s"%s": %s' "$separator" "$key" "$value" ;;
        *) printf '%s"%s": "%s"' "$separator" "$key" "$value" ;;
      esac
      separator=', '
    done
    printf '}\n'
  } > "$config"
  CONFIG_PATH=$config python3 "$script_dir/verify_examples.py" \
    > "$output/$report.json"
}

# Build a JSON array of quoted paths from the arguments.
json_list()
{
  list=
  for item in "$@"; do
    if test -z "$list"; then
      list="\"$item\""
    else
      list="$list, \"$item\""
    fi
  done
  printf '[%s]' "$list"
}

verify_dumps()
{
  report=$1
  reference=$2
  shift 2
  verify "$report" check=dumps "reference=$reference" \
    "candidates=$(json_list "$@")"
}

verify_numdiff()
{
  report=$1
  shift
  verify "$report" check=numdiff "logs=$(json_list "$@")"
}

verify_replay()
{
  report=$1
  shift
  verify "$report" check=replay "dumps=$(json_list "$@")"
}

verify_md()
{
  verify "$1" check=md "log=$2"
}

verify_virial_consistency()
{
  verify "$1" check=virial_consistency "dump=$2" "log=$3"
}

require_ye3t_kokkos_dispatch()
{
  log=$1
  expected_evaluator=${2:-}
  if ! grep -Eq 'YE3T Kokkos device-plan probe: execution_space (Cuda|HIP)' "$log" ||
     ! grep -q 'maximum_error 0' "$log" ||
     grep -q 'YE3T CPU dispatch:' "$log"; then
    echo "The requested PairYE3T/Kokkos run did not prove device dispatch: $log" >&2
    exit 1
  fi
  if test -n "$expected_evaluator" &&
     ! grep -Fq "evaluator $expected_evaluator" "$log"; then
    echo "The requested Kokkos evaluator was not dispatched: $log" >&2
    exit 1
  fi
  if test "$expected_evaluator" = coupled_product &&
     ! grep -q 'coupled_math_probe passed' "$log"; then
    echo "The coupled-product device math probe did not pass: $log" >&2
    exit 1
  fi
  case "$expected_evaluator" in
    auto*)
      if ! grep -Fq 'planner_profile kokkos_gpu_conservative_direct_v1' "$log" ||
         ! grep -Fq 'decision_reason no_authorized_non_direct_profile_match' "$log"; then
        echo "The GPU AUTO fallback decision was not recorded: $log" >&2
        exit 1
      fi
      ;;
  esac
}

require_pace_kokkos_dispatch()
{
  log=$1
  if ! grep -q 'KOKKOS mode with Kokkos version' "$log" ||
     ! grep -q 'Product evaluator is used' "$log" ||
     ! grep -q 'pair pace/kk' "$log"; then
    echo "The requested PACE/Kokkos product run did not prove device dispatch: $log" >&2
    exit 1
  fi
}

require_matching_direct_plan()
{
  cpu_log=$1
  kokkos_log=$2
  cpu_hash=$(awk '
    /YE3T CPU dispatch:/ {
      for (i = 1; i <= NF; ++i)
        if ($i == "direct_logical_plan") {
          value = $(i + 1)
          gsub(/,/, "", value)
          print value
        }
    }
  ' "$cpu_log" | tail -n 1)
  kokkos_hash=$(awk '
    /YE3T Kokkos device-plan probe:/ {
      for (i = 1; i <= NF; ++i)
        if ($i == "direct_logical_plan") {
          value = $(i + 1)
          gsub(/,/, "", value)
          print value
        }
    }
  ' "$kokkos_log" | tail -n 1)
  if test -z "$cpu_hash" || test "$cpu_hash" != "$kokkos_hash"; then
    echo "CPU/Kokkos direct logical-plan identity mismatch" >&2
    exit 1
  fi
}

compact=$script_dir/models/ta_l8_compact
full=$script_dir/models/ta_l8_full
high_rank=$script_dir/models/ta_l8_h16

# Fitted compact model: direct, block, and AUTO execution agree.
run_at "$mpi_ranks" compact_direct in.ye3t.direct \
  -var model "$compact/model.yace" -var dump_path "$output/compact_direct.dump"
run_at "$mpi_ranks" compact_block in.ye3t.block \
  -var model "$compact/model.yace" -var plan "$compact/manifest.json" \
  -var policy block -var dump_path "$output/compact_block.dump"
run_at "$mpi_ranks" compact_auto in.ye3t.auto \
  -var model "$compact/model.yace" -var plan "$compact/manifest.json" \
  -var policy auto -var dump_path "$output/compact_auto.dump"
verify_dumps compact_ye3t_parity "$output/compact_direct.dump" \
  "$output/compact_block.dump" "$output/compact_auto.dump"
verify_virial_consistency compact_direct_virial_consistency \
  "$output/compact_direct.dump" "$output/log.compact_direct"

# Fitted full model: direct, coupled-product, and AUTO execution agree.
run_at "$mpi_ranks" full_direct in.ye3t.direct \
  -var model "$full/model.yace" -var dump_path "$output/full_direct.dump"
run_at "$mpi_ranks" full_coupled in.ye3t.coupled-product \
  -var model "$full/model.yace" -var plan "$full/manifest.json" \
  -var policy coupled_product -var dump_path "$output/full_coupled.dump"
run_at "$mpi_ranks" full_auto in.ye3t.auto \
  -var model "$full/model.yace" -var plan "$full/manifest.json" \
  -var policy auto -var dump_path "$output/full_auto.dump"
verify_dumps full_ye3t_parity "$output/full_direct.dump" \
  "$output/full_coupled.dump" "$output/full_auto.dump"

# Fixed-position replay and short NVE runs.
run_at "$mpi_ranks" compact_replay_direct in.ye3t.replay-direct \
  -var model "$compact/model.yace" \
  -var dump_path "$output/compact_replay_direct.dump"
verify_replay replay_direct "$output/compact_replay_direct.dump"
run_at "$mpi_ranks" compact_md in.ye3t.md \
  -var model "$compact/model.yace" -var plan "$compact/manifest.json"
verify_md md "$output/log.compact_md"
run_at "$mpi_ranks" compact_md_direct in.ye3t.md-direct \
  -var model "$compact/model.yace"
verify_md md_direct "$output/log.compact_md_direct"

if test "$with_pace" = yes; then
  run_at "$mpi_ranks" compact_pace in.ye3t.pace-product \
    -var model "$compact/model.yace" -var dump_path "$output/compact_pace.dump"
  run_at "$mpi_ranks" compact_pace_recursive in.ye3t.pace-recursive \
    -var model "$compact/model.yace" \
    -var dump_path "$output/compact_pace_recursive.dump"
  run_at "$mpi_ranks" full_pace in.ye3t.pace-product \
    -var model "$full/model.yace" -var dump_path "$output/full_pace.dump"
  run_at "$mpi_ranks" compact_pace_replay in.ye3t.pace-replay \
    -var model "$compact/model.yace" \
    -var dump_path "$output/compact_pace_replay.dump"
  verify_dumps compact_pace_ye3t_parity "$output/compact_pace.dump" \
    "$output/compact_pace_recursive.dump" "$output/compact_direct.dump" \
    "$output/compact_block.dump" "$output/compact_auto.dump"
  verify_virial_consistency compact_pace_virial_consistency \
    "$output/compact_pace.dump" "$output/log.compact_pace"
  verify_dumps full_pace_ye3t_parity "$output/full_pace.dump" \
    "$output/full_direct.dump" "$output/full_coupled.dump"
  verify_replay replay_pace "$output/compact_pace_replay.dump"
  verify_dumps replay_pace_ye3t_parity "$output/compact_replay_direct.dump" \
    "$output/compact_pace_replay.dump"
fi

if test "$mpi_ranks" -gt 1; then
  run_at 1 migration_rank1 in.ye3t.mpi-migration \
    -var mpi_ranks 1 -var model "$compact/model.yace" \
    -var plan "$compact/manifest.json" -var dump_path "$output/migration_rank1.dump"
  run_at "$mpi_ranks" migration_rankN in.ye3t.mpi-migration \
    -var mpi_ranks "$mpi_ranks" -var model "$compact/model.yace" \
    -var plan "$compact/manifest.json" -var dump_path "$output/migration_rankN.dump"
  verify_dumps mpi_migration_parity "$output/migration_rank1.dump" \
    "$output/migration_rankN.dump"
fi

if test "$with_numdiff" = yes; then
  run_at "$mpi_ranks" numdiff_direct in.ye3t.numdiff-direct \
    -var model "$compact/model.yace" -var force_delta 1e-4 \
    -var virial_delta 1e-6
  run_at "$mpi_ranks" numdiff_coupled in.ye3t.numdiff \
    -var model "$full/model.yace" -var plan "$full/manifest.json" \
    -var policy coupled_product -var force_delta 1e-4 \
    -var virial_delta 1e-6
  for spec in coarse:3e-4:3e-6 medium:1e-4:1e-6 fine:3e-5:3e-7; do
    name=${spec%%:*}
    rest=${spec#*:}
    force_delta=${rest%%:*}
    virial_delta=${rest#*:}
    run_at "$mpi_ranks" "numdiff_$name" in.ye3t.numdiff \
      -var model "$compact/model.yace" -var plan "$compact/manifest.json" \
      -var force_delta "$force_delta" -var virial_delta "$virial_delta"
  done
  verify_numdiff numdiff "$output/log.numdiff_direct" \
    "$output/log.numdiff_coupled" "$output/log.numdiff_coarse" \
    "$output/log.numdiff_medium" "$output/log.numdiff_fine"
fi

if test "$with_high_rank" = yes; then
  run_at "$mpi_ranks" high_rank_direct in.ye3t.high-rank-direct \
    -var model "$high_rank/model.yace" \
    -var dump_path "$output/high_rank_direct.dump"
  run_at "$mpi_ranks" high_rank_block in.ye3t.high-rank-block \
    -var model "$high_rank/model.yace" -var plan "$high_rank/manifest.json" \
    -var policy block -var dump_path "$output/high_rank_block.dump"
  verify_dumps high_rank_parity "$output/high_rank_direct.dump" \
    "$output/high_rank_block.dump"
  if test "$with_pace" = yes; then
    run_at "$mpi_ranks" high_rank_pace in.ye3t.high-rank-pace-product \
      -var model "$high_rank/model.yace" \
      -var dump_path "$output/high_rank_pace.dump"
    verify_dumps high_rank_pace_ye3t_parity "$output/high_rank_pace.dump" \
      "$output/high_rank_direct.dump" "$output/high_rank_block.dump"
  fi
fi

if test "$with_kokkos" = yes; then
  if test "$mpi_ranks" -gt "$kokkos_gpus"; then
    echo "Kokkos ranks exceed requested GPUs; this is oversubscribed correctness only."
  fi

  run_at_kokkos "$mpi_ranks" compact_direct_kk in.ye3t.direct \
    -var model "$compact/model.yace" \
    -var dump_path "$output/compact_direct_kk.dump"
  require_ye3t_kokkos_dispatch "$output/log.compact_direct_kk" direct
  run_at_kokkos "$mpi_ranks" compact_block_kk in.ye3t.block \
    -var model "$compact/model.yace" -var plan "$compact/manifest.json" \
    -var policy block -var dump_path "$output/compact_block_kk.dump"
  require_ye3t_kokkos_dispatch "$output/log.compact_block_kk" block
  run_at_kokkos "$mpi_ranks" compact_auto_kk in.ye3t.auto \
    -var model "$compact/model.yace" -var plan "$compact/manifest.json" \
    -var policy auto -var dump_path "$output/compact_auto_kk.dump"
  require_ye3t_kokkos_dispatch \
    "$output/log.compact_auto_kk" 'auto[direct]'
  run_at_kokkos "$mpi_ranks" compact_reorder_kk in.ye3t.direct \
    -var model "$compact/model.yace" -var atom_sort 1 \
    -var dump_path "$output/compact_reorder_kk.dump"
  require_ye3t_kokkos_dispatch "$output/log.compact_reorder_kk" direct
  run_at_kokkos "$mpi_ranks" full_direct_kk in.ye3t.direct \
    -var model "$full/model.yace" -var dump_path "$output/full_direct_kk.dump"
  require_ye3t_kokkos_dispatch "$output/log.full_direct_kk" direct
  run_at_kokkos "$mpi_ranks" full_coupled_kk in.ye3t.coupled-product \
    -var model "$full/model.yace" -var plan "$full/manifest.json" \
    -var policy coupled_product \
    -var dump_path "$output/full_coupled_kk.dump"
  require_ye3t_kokkos_dispatch "$output/log.full_coupled_kk" coupled_product
  run_at_kokkos "$mpi_ranks" full_auto_kk in.ye3t.auto \
    -var model "$full/model.yace" -var plan "$full/manifest.json" \
    -var policy auto -var dump_path "$output/full_auto_kk.dump"
  require_ye3t_kokkos_dispatch "$output/log.full_auto_kk" 'auto[direct]'
  require_matching_direct_plan "$output/log.compact_direct" \
    "$output/log.compact_direct_kk"
  require_matching_direct_plan "$output/log.full_direct" \
    "$output/log.full_direct_kk"
  require_matching_direct_plan "$output/log.full_coupled" \
    "$output/log.full_coupled_kk"
  verify_dumps compact_cpu_kokkos_parity "$output/compact_direct.dump" \
    "$output/compact_direct_kk.dump" "$output/compact_block_kk.dump" \
    "$output/compact_auto_kk.dump" "$output/compact_reorder_kk.dump"
  verify_virial_consistency compact_direct_kk_virial_consistency \
    "$output/compact_direct_kk.dump" "$output/log.compact_direct_kk"
  verify_dumps full_cpu_kokkos_parity "$output/full_direct.dump" \
    "$output/full_coupled.dump" "$output/full_direct_kk.dump" \
    "$output/full_coupled_kk.dump" "$output/full_auto_kk.dump"

  run_at_kokkos "$mpi_ranks" compact_replay_direct_kk \
    in.ye3t.replay-direct -var model "$compact/model.yace" \
    -var dump_path "$output/compact_replay_direct_kk.dump"
  require_ye3t_kokkos_dispatch "$output/log.compact_replay_direct_kk" direct
  verify_replay replay_direct_kk "$output/compact_replay_direct.dump" \
    "$output/compact_replay_direct_kk.dump"
  verify_dumps replay_cpu_kokkos_parity "$output/compact_replay_direct.dump" \
    "$output/compact_replay_direct_kk.dump"

  run_at_kokkos "$mpi_ranks" compact_md_direct_kk in.ye3t.md-direct \
    -var model "$compact/model.yace"
  require_ye3t_kokkos_dispatch "$output/log.compact_md_direct_kk" direct
  verify_md md_direct_kk "$output/log.compact_md_direct_kk"

  if test "$with_numdiff" = yes; then
    run_at_kokkos "$mpi_ranks" numdiff_direct_kk in.ye3t.numdiff-direct \
      -var model "$compact/model.yace" -var force_delta 1e-4 \
      -var virial_delta 1e-6
    require_ye3t_kokkos_dispatch "$output/log.numdiff_direct_kk" direct
    run_at_kokkos "$mpi_ranks" numdiff_block_kk in.ye3t.numdiff \
      -var model "$compact/model.yace" -var plan "$compact/manifest.json" \
      -var policy block -var force_delta 1e-4 -var virial_delta 1e-6
    require_ye3t_kokkos_dispatch "$output/log.numdiff_block_kk" block
    run_at_kokkos "$mpi_ranks" numdiff_coupled_kk in.ye3t.numdiff \
      -var model "$full/model.yace" -var plan "$full/manifest.json" \
      -var policy coupled_product -var force_delta 1e-4 \
      -var virial_delta 1e-6
    require_ye3t_kokkos_dispatch \
      "$output/log.numdiff_coupled_kk" coupled_product
    run_at_kokkos "$mpi_ranks" numdiff_auto_kk in.ye3t.numdiff \
      -var model "$compact/model.yace" -var plan "$compact/manifest.json" \
      -var policy auto -var force_delta 1e-4 -var virial_delta 1e-6
    require_ye3t_kokkos_dispatch \
      "$output/log.numdiff_auto_kk" 'auto[direct]'
    verify_numdiff numdiff_kokkos "$output/log.numdiff_direct_kk" \
      "$output/log.numdiff_block_kk" "$output/log.numdiff_coupled_kk" \
      "$output/log.numdiff_auto_kk"
  fi

  if test "$with_pace" = yes; then
    run_at_kokkos "$mpi_ranks" compact_pace_kk in.ye3t.pace-product \
      -var model "$compact/model.yace" \
      -var dump_path "$output/compact_pace_kk.dump"
    require_pace_kokkos_dispatch "$output/log.compact_pace_kk"
    run_at_kokkos "$mpi_ranks" full_pace_kk in.ye3t.pace-product \
      -var model "$full/model.yace" -var dump_path "$output/full_pace_kk.dump"
    require_pace_kokkos_dispatch "$output/log.full_pace_kk"
    verify_dumps compact_pace_kokkos_ye3t_kokkos_parity \
      "$output/compact_pace_kk.dump" "$output/compact_direct_kk.dump" \
      "$output/compact_auto_kk.dump"
    verify_virial_consistency compact_pace_kk_virial_consistency \
      "$output/compact_pace_kk.dump" "$output/log.compact_pace_kk"
    verify_dumps full_pace_kokkos_ye3t_kokkos_parity \
      "$output/full_pace_kk.dump" "$output/full_direct_kk.dump" \
      "$output/full_coupled_kk.dump" "$output/full_auto_kk.dump"
    run_at_kokkos "$mpi_ranks" compact_pace_replay_kk in.ye3t.pace-replay \
      -var model "$compact/model.yace" \
      -var dump_path "$output/compact_pace_replay_kk.dump"
    require_pace_kokkos_dispatch "$output/log.compact_pace_replay_kk"
    verify_replay replay_pace_kk "$output/compact_pace_replay_kk.dump"
    verify_dumps replay_pace_kokkos_ye3t_kokkos_parity \
      "$output/compact_replay_direct_kk.dump" \
      "$output/compact_pace_replay_kk.dump"
  fi

  if test "$mpi_ranks" -gt 1; then
    run_at_kokkos "$mpi_ranks" migration_rankN_kk in.ye3t.mpi-migration \
      -var mpi_ranks "$mpi_ranks" -var model "$compact/model.yace" \
      -var plan "$compact/manifest.json" -var policy auto \
      -var dump_path "$output/migration_rankN_kk.dump"
    require_ye3t_kokkos_dispatch \
      "$output/log.migration_rankN_kk" 'auto[direct]'
    verify_dumps mpi_migration_kokkos_parity "$output/migration_rank1.dump" \
      "$output/migration_rankN_kk.dump"
  fi

  if test "$with_high_rank" = yes; then
    run_at_kokkos "$mpi_ranks" high_rank_direct_kk \
      in.ye3t.high-rank-direct -var model "$high_rank/model.yace" \
      -var dump_path "$output/high_rank_direct_kk.dump"
    require_ye3t_kokkos_dispatch "$output/log.high_rank_direct_kk" direct
    run_at_kokkos "$mpi_ranks" high_rank_block_kk \
      in.ye3t.high-rank-block -var model "$high_rank/model.yace" \
      -var plan "$high_rank/manifest.json" -var policy block \
      -var dump_path "$output/high_rank_block_kk.dump"
    require_ye3t_kokkos_dispatch "$output/log.high_rank_block_kk" block
    verify_dumps high_rank_cpu_kokkos_parity "$output/high_rank_direct.dump" \
      "$output/high_rank_direct_kk.dump" "$output/high_rank_block_kk.dump"
  fi
fi

echo "All requested examples passed. Results: $output"
