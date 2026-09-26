# Shell helpers that drive examples/PACKAGES/ye3t/verify_examples.py from the
# test scripts. Source this file after setting `verifier` to the path of
# verify_examples.py. Each helper writes the JSON configuration next to the
# requested report and runs one check:
#
#   verify_dumps   REPORT REFERENCE CANDIDATE... [key=value...]
#   verify_numdiff REPORT LOG... [key=value...]
#   verify_replay  REPORT DUMP... [key=value...]
#   verify_md      REPORT LOG [key=value...]
#   verify_virial_consistency REPORT DUMP LOG [key=value...]
#
# Trailing key=value arguments override the verifier's default tolerances
# (for example force_atol=1.0e-6). Paths must not contain double quotes or
# backslashes.

ye3t_json_list()
{
  ye3t_list=
  for ye3t_item in "$@"; do
    if test -z "$ye3t_list"; then
      ye3t_list="\"$ye3t_item\""
    else
      ye3t_list="$ye3t_list, \"$ye3t_item\""
    fi
  done
  printf '[%s]' "$ye3t_list"
}

ye3t_run_verifier()
{
  ye3t_report=$1
  shift
  ye3t_config=$ye3t_report.config.json
  {
    printf '{'
    ye3t_separator=
    for ye3t_field in "$@"; do
      ye3t_key=${ye3t_field%%=*}
      ye3t_value=${ye3t_field#*=}
      case "$ye3t_value" in
        \[*) printf '%s"%s": %s' "$ye3t_separator" "$ye3t_key" "$ye3t_value" ;;
        [0-9]*|-[0-9]*|.[0-9]*) printf '%s"%s": %s' "$ye3t_separator" "$ye3t_key" "$ye3t_value" ;;
        *) printf '%s"%s": "%s"' "$ye3t_separator" "$ye3t_key" "$ye3t_value" ;;
      esac
      ye3t_separator=', '
    done
    printf '}\n'
  } > "$ye3t_config"
  CONFIG_PATH=$ye3t_config python3 "$verifier" > "$ye3t_report"
}

# Split "positional... key=value..." arguments: positional items are collected
# into ye3t_positional (newline separated), overrides into ye3t_overrides.
ye3t_split_arguments()
{
  ye3t_positional=
  ye3t_overrides=
  for ye3t_argument in "$@"; do
    case "$ye3t_argument" in
      *=*) ye3t_overrides="$ye3t_overrides
$ye3t_argument" ;;
      *) ye3t_positional="$ye3t_positional
$ye3t_argument" ;;
    esac
  done
}

verify_dumps()
{
  ye3t_report=$1
  ye3t_reference=$2
  shift 2
  ye3t_split_arguments "$@"
  ye3t_old_ifs=$IFS
  IFS='
'
  set -- $ye3t_positional
  ye3t_candidates=$(ye3t_json_list "$@")
  set -- $ye3t_overrides
  IFS=$ye3t_old_ifs
  ye3t_run_verifier "$ye3t_report" check=dumps "reference=$ye3t_reference" \
    "candidates=$ye3t_candidates" "$@"
}

verify_numdiff()
{
  ye3t_report=$1
  shift
  ye3t_split_arguments "$@"
  ye3t_old_ifs=$IFS
  IFS='
'
  set -- $ye3t_positional
  ye3t_logs=$(ye3t_json_list "$@")
  set -- $ye3t_overrides
  IFS=$ye3t_old_ifs
  ye3t_run_verifier "$ye3t_report" check=numdiff "logs=$ye3t_logs" "$@"
}

verify_replay()
{
  ye3t_report=$1
  shift
  ye3t_split_arguments "$@"
  ye3t_old_ifs=$IFS
  IFS='
'
  set -- $ye3t_positional
  ye3t_dumps=$(ye3t_json_list "$@")
  set -- $ye3t_overrides
  IFS=$ye3t_old_ifs
  ye3t_run_verifier "$ye3t_report" check=replay "dumps=$ye3t_dumps" "$@"
}

verify_md()
{
  ye3t_report=$1
  ye3t_log=$2
  shift 2
  ye3t_run_verifier "$ye3t_report" check=md "log=$ye3t_log" "$@"
}

verify_virial_consistency()
{
  ye3t_report=$1
  ye3t_dump=$2
  ye3t_log=$3
  shift 3
  ye3t_run_verifier "$ye3t_report" check=virial_consistency \
    "dump=$ye3t_dump" "log=$ye3t_log" "$@"
}
