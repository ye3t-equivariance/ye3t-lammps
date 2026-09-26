#!/bin/sh
# Drives the tagged-Cauchy Kokkos/CUDA device path (pair_style ye3t/kk
# model_family tagged_cauchy) through six checks:
#   1. CPU/Python parity k1/k2 + numdiff on frame0 (device binary)
#   2. MPI 1/2/4 ranks on one (oversubscribed) GPU
#   3. multi-species fixtures through the device path
#   4. run_examples.sh --with-kokkos --with-pace regression suite
#   5. 500-step NVE 300K 128-atom BCC k=2 on device, drift reported
#   6. timing k1/k2 device vs CPU-tagged vs pace/kk product, 128- and
#      1024-atom cells, GPU memory reported
#
# Checks 1/2/3/5/6 are driven by validate_fitted_tagged_cauchy_kokkos.py
# (which itself waits for free memory before every GPU-dispatching LAMMPS
# invocation -- see wait_for_memory below for the equivalent done here in
# shell for the two invocations this script makes directly: the Python
# driver's own startup, and run_examples.sh --with-kokkos, both of which
# launch many GPU-touching LAMMPS runs of their own).
#
# Optional free-memory guard, controlled by environment variables: every
# heavy (GPU-test) step here checks `free -g` first and, if fewer than
# YE3T_MIN_FREE_GB (default 3) GB are available, sleeps in
# YE3T_MEMORY_POLL_SECONDS (default 60) steps for up to
# YE3T_MEMORY_MAX_WAIT_SECONDS (default 1800) before proceeding anyway --
# never more than one heavy process runs at a time from this script.

set -eu

usage()
{
  cat <<'EOF' >&2
Usage: test_kokkos_tagged_cauchy_cuda.sh REPO_ROOT CPU_LMP KK_LMP OUTPUT \
    --model-k1 PATH --reference-k1 PATH [options]

Required:
  REPO_ROOT             ye3t-lammps checkout root (for examples/PACKAGES/ye3t)
  CPU_LMP                lmp binary built without Kokkos (lammps-build-tagged)
  KK_LMP                  lmp binary built with Kokkos/CUDA (lammps-build-tagged-cuda)
  OUTPUT                  output directory (must not already exist)
  --model-k1 PATH          k=1 tagged-Cauchy model JSON
  --reference-k1 PATH      k=1 reference JSON

Options:
  --model-k2 PATH --reference-k2 PATH   k=2 model/reference
  --multispecies-model PATH
  --multispecies-ta-fixture PATH
  --multispecies-w-fixture PATH
  --yace-model PATH        baseline .yace model for pace/kk product timing
  --pace-kk-lmp PATH        Kokkos lmp binary with pair_style pace (default: KK_LMP)
  --mpiexec NAME             (default: mpiexec)
  --skip-mpi --skip-multispecies --skip-nve --skip-timing
  --skip-regression          skip the run_examples.sh --with-kokkos regression check
EOF
}

if test "$#" -lt 4; then
  usage
  exit 2
fi

repo_root=$(CDPATH= cd -- "$1" && pwd -P)
cpu_lmp=$(CDPATH= cd -- "$(dirname -- "$2")" && pwd -P)/$(basename -- "$2")
kk_lmp=$(CDPATH= cd -- "$(dirname -- "$3")" && pwd -P)/$(basename -- "$3")
output=$4
shift 4

model_k1= reference_k1= model_k2= reference_k2=
multispecies_model= multispecies_ta= multispecies_w=
yace_model= pace_kk_lmp= mpiexec=mpiexec
extra_python_args=
skip_regression=no

while test "$#" -gt 0; do
  case "$1" in
    --model-k1) model_k1=$2; shift 2 ;;
    --reference-k1) reference_k1=$2; shift 2 ;;
    --model-k2) model_k2=$2; shift 2 ;;
    --reference-k2) reference_k2=$2; shift 2 ;;
    --multispecies-model) multispecies_model=$2; shift 2 ;;
    --multispecies-ta-fixture) multispecies_ta=$2; shift 2 ;;
    --multispecies-w-fixture) multispecies_w=$2; shift 2 ;;
    --yace-model) yace_model=$2; shift 2 ;;
    --pace-kk-lmp) pace_kk_lmp=$2; shift 2 ;;
    --mpiexec) mpiexec=$2; shift 2 ;;
    --skip-mpi|--skip-multispecies|--skip-nve|--skip-timing)
      extra_python_args="$extra_python_args $1"; shift ;;
    --skip-regression) skip_regression=yes; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if test -z "$model_k1" || test -z "$reference_k1"; then
  echo "--model-k1 and --reference-k1 are required" >&2
  exit 2
fi
if test ! -x "$cpu_lmp" || test ! -x "$kk_lmp"; then
  echo "CPU_LMP or KK_LMP does not exist or is not executable" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required" >&2
  exit 1
fi
if ! command -v free >/dev/null 2>&1; then
  echo "free (procps) is required for the free-memory guard" >&2
  exit 1
fi
if test -e "$output"; then
  echo "Output path already exists: $output" >&2
  exit 1
fi
mkdir -p "$output"
output=$(CDPATH= cd -- "$output" && pwd -P)

min_free_gb=${YE3T_MIN_FREE_GB:-3}
poll_seconds=${YE3T_MEMORY_POLL_SECONDS:-60}
max_wait_seconds=${YE3T_MEMORY_MAX_WAIT_SECONDS:-1800}
memory_log=$output/memory-readings.shell.log

wait_for_memory()
{
  context=$1
  waited=0
  while true; do
    reading=$(free -g | awk '/^Mem:/ {print $7}')
    line="[memory-gate] $context: available_gb=$reading waited_s=$waited $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "$line" | tee -a "$memory_log" >&2
    free -g | tee -a "$memory_log" >&2
    if test "$reading" -ge "$min_free_gb"; then
      break
    fi
    if test "$waited" -ge "$max_wait_seconds"; then
      echo "[memory-gate] $context: proceeding below threshold (${reading}GB) after ${waited}s" \
        | tee -a "$memory_log" >&2
      break
    fi
    sleep "$poll_seconds"
    waited=$((waited + poll_seconds))
  done
}

"$cpu_lmp" -help > "$output/lammps-help.cpu.txt"
if ! grep -Eq '(^|[[:space:]])ye3t([[:space:]]|$)' "$output/lammps-help.cpu.txt"; then
  echo "CPU_LMP does not contain pair_style ye3t" >&2
  exit 1
fi
"$kk_lmp" -help > "$output/lammps-help.kk.txt"
if ! grep -Eq '(^|[[:space:]])ye3t/kk([[:space:]]|$)' "$output/lammps-help.kk.txt"; then
  echo "KK_LMP does not contain pair_style ye3t/kk" >&2
  exit 1
fi

wait_for_memory "before-validate_fitted_tagged_cauchy_kokkos.py"

python_args="$output --cpu-lmp $cpu_lmp --kk-lmp $kk_lmp --model-k1 $model_k1 --reference-k1 $reference_k1"
python_args="$python_args --mpiexec $mpiexec --min-free-gb $min_free_gb"
python_args="$python_args --memory-poll-seconds $poll_seconds --memory-max-wait-seconds $max_wait_seconds"
test -n "$model_k2" && python_args="$python_args --model-k2 $model_k2"
test -n "$reference_k2" && python_args="$python_args --reference-k2 $reference_k2"
test -n "$multispecies_model" && python_args="$python_args --multispecies-model $multispecies_model"
test -n "$multispecies_ta" && python_args="$python_args --multispecies-ta-fixture $multispecies_ta"
test -n "$multispecies_w" && python_args="$python_args --multispecies-w-fixture $multispecies_w"
test -n "$yace_model" && python_args="$python_args --yace-model $yace_model"
test -n "$pace_kk_lmp" && python_args="$python_args --pace-kk-lmp $pace_kk_lmp"
python_args="$python_args$extra_python_args --json $output/validation.json"

# shellcheck disable=SC2086
python3 "$repo_root/tests/validate_fitted_tagged_cauchy_kokkos.py" $python_args \
  > "$output/validate_fitted_tagged_cauchy_kokkos.stdout.log"
echo "Checks 1/2/3/5/6 (validate_fitted_tagged_cauchy_kokkos.py) passed."

if test "$skip_regression" = "no"; then
  wait_for_memory "before-run_examples.sh-with-kokkos"
  regression_with_pace=no
  if grep -Eq '(^|[[:space:]])pace([[:space:]]|$)' "$output/lammps-help.kk.txt"; then
    regression_with_pace=yes
  else
    echo "KK_LMP has no pair_style pace: running the Kokkos examples without the PACE comparison" >&2
  fi
  YE3T_LMP=$kk_lmp YE3T_WITH_KOKKOS=yes YE3T_WITH_PACE=$regression_with_pace \
    YE3T_EXAMPLE_OUTPUT=$output/run_examples \
    timeout 1800 "$repo_root/examples/PACKAGES/ye3t/run_examples.sh" \
    > "$output/run_examples.stdout.log" 2>&1
  echo "Check 4 (run_examples.sh with Kokkos, PACE comparison: $regression_with_pace) passed."
else
  echo "Check 4 (run_examples.sh with Kokkos) skipped by request (--skip-regression)."
fi

echo "All requested Kokkos tagged-Cauchy checks passed. Memory readings: $memory_log"
