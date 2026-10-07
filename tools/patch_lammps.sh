#!/bin/sh

set -eu
# All filename lists are data, never shell patterns.
set -f

usage()
{
  cat <<'EOF'
Usage: patch_lammps.sh (--check | --apply | --uninstall) --lammps-source PATH
                       [--dry-run]

Install the ye3t-lammps sources, CMake package metadata, and public examples
into a LAMMPS source tree. --check performs every preflight and prints the
CMake diff without modifying the tree. --apply is idempotent for an identical
installation and refuses partial or locally modified installations.

--uninstall removes installed YE3T files directly (never with git restore),
including modified files and old manifest entries. It first backs up the files
and CMakeLists.txt under PATH/.ye3t-uninstall-backups/. Only YE3T registrations
are removed from CMakeLists.txt; unrelated edits, files, and example outputs
are preserved. Missing files are tolerated. --dry-run previews --uninstall
without changing the LAMMPS tree. To update: --uninstall, then --apply.
EOF
}

mode=
dry_run=false
lammps_source=
while test "$#" -gt 0; do
  case "$1" in
    --check|--apply|--uninstall)
      if test -n "$mode"; then
        echo "Specify exactly one of --check, --apply, or --uninstall" >&2
        exit 2
      fi
      mode=$1
      ;;
    --dry-run)
      dry_run=true
      ;;
    --lammps-source)
      shift
      if test "$#" -eq 0; then
        echo "--lammps-source requires a path" >&2
        exit 2
      fi
      lammps_source=$1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if test -z "$mode" || test -z "$lammps_source"; then
  usage >&2
  exit 2
fi

if test "$dry_run" = true && test "$mode" != --uninstall; then
  echo "--dry-run is only supported with --uninstall; use --check to preview installation" >&2
  exit 2
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd -P)
lammps_root=$(CDPATH= cd -- "$lammps_source" 2>/dev/null && pwd -P) || {
  echo "LAMMPS source path does not exist: $lammps_source" >&2
  exit 1
}

cmake_file=$lammps_root/cmake/CMakeLists.txt
package_dir=$lammps_root/src/ML-YE3T
kokkos_dir=$lammps_root/src/KOKKOS
module_dir=$lammps_root/cmake/Modules/Packages
module_file=$module_dir/ML-YE3T.cmake
manifest_file=$package_dir/YE3T_LAMMPS_MANIFEST.sha256
kokkos_manifest_file=$package_dir/YE3T_LAMMPS_KOKKOS_MANIFEST.sha256
install_record=$package_dir/YE3T_LAMMPS_INSTALL.txt
example_source_dir=$repo_root/examples/PACKAGES/ye3t
example_dir=$lammps_root/examples/PACKAGES/ye3t
example_manifest_name=YE3T_EXAMPLE_MANIFEST.sha256
example_source_manifest=$example_source_dir/$example_manifest_name
example_manifest=$example_dir/$example_manifest_name
doc_source_file=$repo_root/doc/src/pair_ye3t.rst
doc_dir=$lammps_root/doc/src
doc_file=$doc_dir/pair_ye3t.rst
property_doc_source_file=$repo_root/doc/src/compute_ye3t_property_atom.rst
property_doc_file=$doc_dir/compute_ye3t_property_atom.rst

if test ! -f "$lammps_root/src/lammps.cpp" || test ! -f "$cmake_file"; then
  echo "Not a supported LAMMPS source root: $lammps_root" >&2
  exit 1
fi

source_files='pair_ye3t.cpp
pair_ye3t.h
compute_ye3t_property_atom.cpp
compute_ye3t_property_atom.h
ye3t_canonical_json_hash.cpp
ye3t_canonical_json_hash.h
ye3t_lifted_cauchy_cpu.cpp
ye3t_lifted_cauchy_cpu.h
ye3t_lifted_cauchy_source.cpp
ye3t_lifted_cauchy_model.cpp
ye3t_lifted_cauchy_model.h
ye3t_mean_property_cpu.cpp
ye3t_mean_property_cpu.h
ye3t_cpu_batching.h
ye3t_gpu_dag_schedule.h
ye3t_gpu_block_schedule.h
ye3t_gpu_harmonic_stream.h
ye3t_gpu_tagged_source.h
ye3t_cpu_source_tiling.h
ye3t_cpu_source_tiling.cpp
ye3t_cpu_evaluator.cpp
ye3t_cpu_evaluator.h
ye3t_sha256.cpp
ye3t_sha256.h
ye3t_shifted_jacobi.h
ye3t_tagged_cauchy_cpu.cpp
ye3t_tagged_cauchy_cpu.h
ye3t_tagged_cauchy_model.cpp
ye3t_tagged_cauchy_model.h
ye3t_tagged_cauchy_readout_plan.h
ye3t_tagged_c_api.cpp
ye3t_yace_model.cpp
ye3t_yace_model.h'

kokkos_source_files='pair_ye3t_kokkos.cpp
pair_ye3t_kokkos.h
compute_ye3t_property_atom_kokkos.cpp
compute_ye3t_property_atom_kokkos.h
ye3t_mean_property_kokkos_plan.h
ye3t_lifted_cauchy_kokkos_plan.h
ye3t_tagged_cauchy_kokkos_plan.h
ye3t_kokkos_plan.h
ye3t_kokkos_step_state.h
ye3t_kokkos_types.h'

tmp_base=${TMPDIR:-/tmp}
stage_dir=$(mktemp -d "$tmp_base/ye3t-lammps-patch.XXXXXX")
cleanup()
{
  case "$stage_dir" in
    "$tmp_base"/ye3t-lammps-patch.*) rm -rf -- "$stage_dir" ;;
    *) echo "Refusing to remove unexpected staging path: $stage_dir" >&2 ;;
  esac
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM


# Validate every ancestor before traversing a managed path. A source-file
# symlink may be unlinked by --uninstall, but directory symlinks are never
# followed. The LAMMPS root itself was resolved with pwd -P above.
check_managed_path()
{
  checked_path=$lammps_root
  remaining_path=$1
  while :; do
    path_component=${remaining_path%%/*}
    checked_path=$checked_path/$path_component
    if test "$remaining_path" = "$path_component"; then
      break
    fi
    if test -L "$checked_path" ||
       { test -e "$checked_path" && test ! -d "$checked_path"; }; then
      echo "Unsafe managed directory (symlink or non-directory): $checked_path" >&2
      exit 1
    fi
    remaining_path=${remaining_path#*/}
  done
  if test "$2" = directory; then
    if test -L "$checked_path" ||
       { test -e "$checked_path" && test ! -d "$checked_path"; }; then
      echo "Unsafe managed directory (symlink or non-directory): $checked_path" >&2
      exit 1
    fi
  elif test -e "$checked_path" && test ! -f "$checked_path" &&
       test ! -L "$checked_path"; then
    echo "Expected a managed file, not a directory or special file: $checked_path" >&2
    exit 1
  fi
}

# Print root-relative paths from one installed SHA256 inventory. Do not use
# sha256sum -c here: source edits are exactly why uninstallation is needed.
# Paths, however, are strictly checked before any backup/deletion occurs.
manifest_paths()
{
  if test -L "$1" || test ! -f "$1"; then
    echo "Refusing a symlink or non-regular checksum inventory: $1" >&2
    return 1
  fi
  awk -v kind="$2" '
    /^[[:space:]]*$/ { next }
    {
      hash = substr($0, 1, 64)
      separator = substr($0, 65, 2)
      path = substr($0, 67)
      if (length(hash) != 64 || hash ~ /[^0-9a-fA-F]/ ||
          (separator != "  " && separator != " *")) exit 42
      while (substr(path, 1, 2) == "./") path = substr(path, 3)
      if (kind == "kokkos") {
        if (substr(path, 1, 10) != "../KOKKOS/") exit 42
        path = substr(path, 11)
        if (path !~ /^(pair_ye3t|compute_ye3t|ye3t_)[A-Za-z0-9_.+-]*\.(cpp|h|hpp|cc|cxx)$/)
          exit 42
      }
      if (path == "" || path ~ /^\// || path ~ /\/$/ ||
          path ~ /\/\// || path ~ /(^|\/)\.\.?($|\/)/ ||
          path ~ /[^A-Za-z0-9_.+@\/-]/ || seen[path]++) exit 42
      if (kind == "package") print "src/ML-YE3T/" path
      else if (kind == "kokkos") print "src/KOKKOS/" path
      else print "examples/PACKAGES/ye3t/" path
    }
  ' "$1" || {
    echo "Unsafe, duplicate, or malformed path in checksum inventory: $1" >&2
    return 1
  }
}

uninstall_package()
{
  # First stage the inverse CMake edit. This also recovers partial/duplicate
  # registrations without reverting unrelated changes or touching the index.
  # These are package-name lists, not arbitrary CMake expressions.
  awk -v changes="$stage_dir/cmake-removals" '
    {
      line = $0
      comment = ""
      pos = index(line, "#")
      if (pos) { comment = substr(line, pos); line = substr(line, 1, pos - 1) }
      if (line ~ /^[[:space:]]*set[[:space:]]*\([[:space:]]*STANDARD_PACKAGES([[:space:]]|$)/ ||
          line ~ /^[[:space:]]*foreach[[:space:]]*\([[:space:]]*PKG_WITH_INCL([[:space:]]|$)/)
        in_list = 1
      drop = 0
      if (in_list) {
        if (line ~ /^[[:space:]]*ML-YE3T[[:space:]]*$/) {
          removed++
          # Remove the line installed into STANDARD_PACKAGES, but retain any
          # independently added comment on it.
          if (comment == "") drop = 1
          else sub(/ML-YE3T[[:space:]]*$/, "", line)
        } else {
          offset = 1
          while (match(substr(line, offset), /ML-YE3T/)) {
            first = offset + RSTART - 1
            last = first + 6
            before = first == 1 ? "" : substr(line, first - 1, 1)
            after = substr(line, last + 1, 1)
            if ((before == "" || before ~ /[[:space:](]/) &&
                (after == "" || after ~ /[[:space:])]/)) {
              removed++
              if (before ~ /[[:space:]]/) first--
              else if (after ~ /[[:space:]]/) last++
              line = substr(line, 1, first - 1) substr(line, last + 1)
              offset = first
            } else offset = last + 1
          }
        }
        if (line ~ /\)/) in_list = 0
      }
      if (!drop) print line comment
    }
    END { print removed + 0 > changes }
  ' "$cmake_file" > "$stage_dir/CMakeLists.txt"
  if test "$(cat "$stage_dir/cmake-removals")" -eq 0; then
    # Preserve even a missing final newline on an already unregistered tree.
    cp "$cmake_file" "$stage_dir/CMakeLists.txt"
  fi

  # The installed manifests take precedence over the current ye3t-lammps
  # version. Also include the known filenames to recover interrupted/missing
  # manifests.
  : > "$stage_dir/candidates"
  for source_file in $source_files README YE3T_LAMMPS_MANIFEST.sha256 \
      YE3T_LAMMPS_KOKKOS_MANIFEST.sha256 YE3T_LAMMPS_INSTALL.txt; do
    printf '%s\n' "src/ML-YE3T/$source_file" >> "$stage_dir/candidates"
  done
  for source_file in $kokkos_source_files; do
    printf '%s\n' "src/KOKKOS/$source_file" >> "$stage_dir/candidates"
  done
  printf '%s\n' cmake/Modules/Packages/ML-YE3T.cmake doc/src/pair_ye3t.rst \
    doc/src/compute_ye3t_property_atom.rst \
    examples/PACKAGES/ye3t/YE3T_EXAMPLE_MANIFEST.sha256 >> "$stage_dir/candidates"
  if test -e "$manifest_file" || test -L "$manifest_file"; then
    manifest_paths "$manifest_file" package >> "$stage_dir/candidates"
  fi
  if test -e "$kokkos_manifest_file" || test -L "$kokkos_manifest_file"; then
    manifest_paths "$kokkos_manifest_file" kokkos >> "$stage_dir/candidates"
  fi
  if test -e "$example_manifest" || test -L "$example_manifest"; then
    manifest_paths "$example_manifest" examples >> "$stage_dir/candidates"
  elif test -d "$example_dir"; then
    if test -f "$example_source_manifest"; then
      echo "Installed example inventory is missing; using the current ye3t-lammps example filenames." >&2
      manifest_paths "$example_source_manifest" examples >> "$stage_dir/candidates"
    else
      echo "No example inventory is available; preserving unidentified example files." >&2
    fi
  fi
  LC_ALL=C sort -u "$stage_dir/candidates" > "$stage_dir/candidates.sorted"
  : > "$stage_dir/remove-files"
  : > "$stage_dir/remove-dirs"
  while IFS= read -r owned_file; do
    check_managed_path "$owned_file" file
    if test -e "$lammps_root/$owned_file" || test -L "$lammps_root/$owned_file"; then
      printf '%s\n' "$owned_file" >> "$stage_dir/remove-files"
    fi
    # Only prune empty package/example directories, never shared LAMMPS dirs.
    parent=${owned_file%/*}
    while :; do
      case "$parent" in
        src/ML-YE3T|src/ML-YE3T/*|examples/PACKAGES/ye3t|examples/PACKAGES/ye3t/*)
          printf '%s\n' "$parent" >> "$stage_dir/remove-dirs"
          parent=${parent%/*}
          ;;
        *) break ;;
      esac
    done
  done < "$stage_dir/candidates.sorted"
  LC_ALL=C sort -ru "$stage_dir/remove-dirs" > "$stage_dir/remove-dirs.sorted"
  count=$(wc -l < "$stage_dir/remove-files" | tr -d '[:space:]')
  cmake_changed=false
  if ! cmp -s "$cmake_file" "$stage_dir/CMakeLists.txt"; then cmake_changed=true; fi

  if test "$dry_run" = true; then
    echo "Uninstall preview: $count owned files would be backed up and removed:"
    cat "$stage_dir/remove-files"
    if test "$cmake_changed" = true; then
      diff -u "$cmake_file" "$stage_dir/CMakeLists.txt" || true
    fi
    echo "Dry run only; the LAMMPS tree was not changed."
    return
  fi
  if test "$count" -eq 0 && test "$cmake_changed" = false; then
    echo "No installed YE3T files or registrations found; nothing to uninstall."
    return
  fi

  # Save everything being removed, including locally modified files and leaf
  # symlinks themselves. Complete this backup before modifying the source tree.
  check_managed_path .ye3t-uninstall-backups directory
  backup_parent=$lammps_root/.ye3t-uninstall-backups
  mkdir -p "$backup_parent"
  backup_dir=$(mktemp -d "$backup_parent/uninstall.XXXXXX")
  echo "Uninstall backup: $backup_dir"
  mkdir -p "$backup_dir/files/cmake"
  cp -p "$cmake_file" "$backup_dir/files/cmake/CMakeLists.txt"
  cp "$stage_dir/remove-files" "$backup_dir/removed-files.txt"
  while IFS= read -r owned_file; do
    mkdir -p "$backup_dir/files/$(dirname -- "$owned_file")"
    cp -p -P "$lammps_root/$owned_file" "$backup_dir/files/$owned_file"
  done < "$stage_dir/remove-files"
  printf '%s\n' "ye3t-lammps uninstall backup" \
    "lammps_root=$lammps_root" "files=$count" \
    "cmake_registrations_removed=$cmake_changed" > "$backup_dir/README.txt"

  # Deliberately do not invoke git restore, git clean, or rm -rf here. Files
  # tracked by a user's Git checkout must disappear just like untracked files.
  while IFS= read -r owned_file; do
    case "$owned_file" in
      src/ML-YE3T/YE3T_LAMMPS_MANIFEST.sha256|src/ML-YE3T/YE3T_LAMMPS_KOKKOS_MANIFEST.sha256|src/ML-YE3T/YE3T_LAMMPS_INSTALL.txt|examples/PACKAGES/ye3t/YE3T_EXAMPLE_MANIFEST.sha256)
        continue ;; # Keep inventories until all payload deletions have succeeded.
    esac
    rm -f -- "$lammps_root/$owned_file"
  done < "$stage_dir/remove-files"
  if test "$cmake_changed" = true; then
    # Write in place to retain permissions; the full previous file is backed up.
    cat "$stage_dir/CMakeLists.txt" > "$cmake_file"
  fi
  # Retain inventories on a payload/CMake error so a retry can still identify
  # retired filenames that do not exist in the current ye3t-lammps revision.
  for record_file in "$manifest_file" "$kokkos_manifest_file" "$install_record" "$example_manifest"; do
    rm -f -- "$record_file"
  done
  while IFS= read -r empty_dir; do
    rmdir -- "$lammps_root/$empty_dir" 2>/dev/null || :
  done < "$stage_dir/remove-dirs.sorted"
  echo "Uninstalled YE3T ($count files) from $lammps_root"
  echo "Unlisted files and nonempty directories were preserved; backup: $backup_dir"
  echo "You may now run --apply from the updated ye3t-lammps tree and reconfigure/rebuild LAMMPS."
}

# Guard shared ancestors and reject a symlinked CMake input for every mode.
for managed_dir in src/ML-YE3T src/KOKKOS cmake/Modules/Packages doc/src \
    examples/PACKAGES/ye3t; do
  check_managed_path "$managed_dir" directory
done
check_managed_path cmake/CMakeLists.txt file
if test -L "$cmake_file"; then
  echo "Refusing to edit a symlinked CMakeLists.txt: $cmake_file" >&2
  exit 1
fi

if test "$mode" = --uninstall; then
  uninstall_package
  exit 0
fi

if test ! -d "$kokkos_dir" || test ! -d "$module_dir" || test ! -d "$doc_dir" ||
   test ! -d "$lammps_root/examples/PACKAGES"; then
  echo "Not a supported LAMMPS installation target: $lammps_root" >&2
  exit 1
fi

for source_file in $source_files; do
  if test ! -f "$repo_root/src/$source_file"; then
    echo "Missing ye3t-lammps source: src/$source_file" >&2
    exit 1
  fi
done
for source_file in $kokkos_source_files; do
  if test ! -f "$repo_root/src/KOKKOS/$source_file"; then
    echo "Missing ye3t-lammps Kokkos source: src/KOKKOS/$source_file" >&2
    exit 1
  fi
done
if test ! -f "$repo_root/cmake/ML-YE3T.cmake" ||
   test ! -f "$repo_root/lammps_package/README"; then
  echo "The ye3t-lammps package metadata is incomplete" >&2
  exit 1
fi
if test ! -f "$example_source_manifest"; then
  echo "Missing public example manifest: examples/PACKAGES/ye3t/$example_manifest_name" >&2
  exit 1
fi
manifest_paths "$example_source_manifest" examples > "$stage_dir/example-paths"
example_files=$(sed 's@^examples/PACKAGES/ye3t/@@' "$stage_dir/example-paths")
if ! (cd "$example_source_dir" && sha256sum -c --quiet "$example_manifest_name"); then
  echo "The ye3t-lammps public examples failed their source checksum verification" >&2
  exit 1
fi
if test -z "$example_files"; then
  echo "The public example manifest is empty" >&2
  exit 1
fi
for example_file in $example_files; do
  if test ! -f "$example_source_dir/$example_file"; then
    echo "Missing public example file: examples/PACKAGES/ye3t/$example_file" >&2
    exit 1
  fi
done

standard_count=$(awk '
  /^set\(STANDARD_PACKAGES$/ { in_standard = 1; next }
  in_standard && /^\)$/ { in_standard = 0 }
  in_standard && $0 == "  ML-YE3T" { count++ }
  END { print count + 0 }
' "$cmake_file")
include_count=$(awk '
  /^foreach\(PKG_WITH_INCL / { in_include = 1 }
  in_include && /(^|[[:space:]])ML-YE3T([[:space:]\)]|$)/ { count++ }
  in_include && /\)$/ { in_include = 0 }
  END { print count + 0 }
' "$cmake_file")

if test "$standard_count" -gt 1 || test "$include_count" -gt 1; then
  echo "LAMMPS CMake contains duplicate ML-YE3T entries" >&2
  exit 1
fi
if test "$standard_count" -ne "$include_count"; then
  echo "LAMMPS CMake is partially patched for ML-YE3T" >&2
  exit 1
fi

cmake_state=clean
if test "$standard_count" -eq 1; then
  cmake_state=patched
  cp "$cmake_file" "$stage_dir/CMakeLists.txt"
else
  awk '
    BEGIN { standard = 0; include = 0 }
    /^set\(STANDARD_PACKAGES$/ { in_standard = 1 }
    in_standard && $0 == "  ML-PACE" {
      print
      print "  ML-YE3T"
      standard++
      next
    }
    in_standard && /^\)$/ { in_standard = 0 }
    {
      line = $0
      if (line ~ /^foreach\(PKG_WITH_INCL / || in_include) {
        in_include = 1
        if (line ~ /ML-PACE LEPTON/) {
          sub(/ML-PACE LEPTON/, "ML-PACE ML-YE3T LEPTON", line)
          include++
        }
        if (line ~ /\)$/) in_include = 0
      }
      print line
    }
    END {
      if (standard != 1 || include != 1) exit 42
    }
  ' "$cmake_file" > "$stage_dir/CMakeLists.txt" || {
    echo "Unsupported LAMMPS CMake layout; required semantic anchors were not unique" >&2
    exit 1
  }
fi

if test "$cmake_state" = clean; then
  # A previous uninstall may leave unlisted outputs/notes in these directories.
  # Permit them, but never overwrite an owned destination or follow a symlink.
  : > "$stage_dir/install-destinations"
  for source_file in $source_files README YE3T_LAMMPS_MANIFEST.sha256 \
      YE3T_LAMMPS_KOKKOS_MANIFEST.sha256 YE3T_LAMMPS_INSTALL.txt; do
    printf '%s\n' "src/ML-YE3T/$source_file" >> "$stage_dir/install-destinations"
  done
  for source_file in $kokkos_source_files; do
    printf '%s\n' "src/KOKKOS/$source_file" >> "$stage_dir/install-destinations"
  done
  printf '%s\n' cmake/Modules/Packages/ML-YE3T.cmake doc/src/pair_ye3t.rst \
    doc/src/compute_ye3t_property_atom.rst \
    examples/PACKAGES/ye3t/YE3T_EXAMPLE_MANIFEST.sha256 >> "$stage_dir/install-destinations"
  for example_file in $example_files; do
    printf '%s\n' "examples/PACKAGES/ye3t/$example_file" >> "$stage_dir/install-destinations"
  done
  while IFS= read -r destination_file; do
    check_managed_path "$destination_file" file
    if test -e "$lammps_root/$destination_file" || test -L "$lammps_root/$destination_file"; then
      echo "Found an owned destination without CMake registrations: $destination_file" >&2
      echo "Run --uninstall before replacing a partial/older installation." >&2
      exit 1
    fi
  done < "$stage_dir/install-destinations"
else
  if test ! -d "$package_dir" || test ! -f "$module_file"; then
    echo "CMake is patched but the ML-YE3T package is incomplete" >&2
    exit 1
  fi
  if test ! -f "$manifest_file" || test ! -f "$install_record"; then
    echo "Installed ML-YE3T package lacks its integrity records" >&2
    exit 1
  fi
  if ! (cd "$package_dir" && sha256sum -c --quiet YE3T_LAMMPS_MANIFEST.sha256); then
    echo "Installed ML-YE3T package failed its recorded checksum verification" >&2
    exit 1
  fi
  if test ! -f "$kokkos_manifest_file" ||
     ! (cd "$package_dir" && sha256sum -c --quiet YE3T_LAMMPS_KOKKOS_MANIFEST.sha256); then
    echo "Installed ML-YE3T Kokkos sources failed their checksum verification" >&2
    exit 1
  fi
  if test "$(sed -n '1p' "$install_record")" != "ye3t-lammps source integration" ||
     test "$(grep -c '^lammps_revision=' "$install_record")" -ne 1; then
    echo "Installed ML-YE3T package has an invalid installation record" >&2
    exit 1
  fi
  for source_file in $source_files; do
    if test ! -f "$package_dir/$source_file" ||
       ! cmp -s "$repo_root/src/$source_file" "$package_dir/$source_file"; then
      echo "Installed ML-YE3T source differs: src/ML-YE3T/$source_file" >&2
      exit 1
    fi
  done
  for source_file in $kokkos_source_files; do
    if test ! -f "$kokkos_dir/$source_file" ||
       ! cmp -s "$repo_root/src/KOKKOS/$source_file" "$kokkos_dir/$source_file"; then
      echo "Installed ML-YE3T Kokkos source differs: src/KOKKOS/$source_file" >&2
      exit 1
    fi
  done
  if ! cmp -s "$repo_root/cmake/ML-YE3T.cmake" "$module_file" ||
     ! cmp -s "$repo_root/lammps_package/README" "$package_dir/README"; then
    echo "Installed ML-YE3T package metadata differs" >&2
    exit 1
  fi
  if test ! -f "$doc_file" || ! cmp -s "$doc_source_file" "$doc_file"; then
    echo "Installed ML-YE3T documentation page is missing or differs: doc/src/pair_ye3t.rst" >&2
    exit 1
  fi
  if test ! -f "$property_doc_file" ||
     ! cmp -s "$property_doc_source_file" "$property_doc_file"; then
    echo "Installed ML-YE3T documentation page is missing or differs: doc/src/compute_ye3t_property_atom.rst" >&2
    exit 1
  fi
  if test ! -d "$example_dir" || test ! -f "$example_manifest" ||
     ! cmp -s "$example_source_manifest" "$example_manifest"; then
    echo "Installed ML-YE3T public examples are incomplete or differ" >&2
    exit 1
  fi
  if ! (cd "$example_dir" && sha256sum -c --quiet "$example_manifest_name"); then
    echo "Installed ML-YE3T public examples failed checksum verification" >&2
    exit 1
  fi
  for example_tool in run_examples.sh verify_examples.py; do
    if test ! -x "$example_dir/$example_tool"; then
      echo "Installed ML-YE3T public example tool is not executable: $example_tool" >&2
      exit 1
    fi
  done
  for example_file in $example_files; do
    if test ! -f "$example_dir/$example_file" ||
       ! cmp -s "$example_source_dir/$example_file" "$example_dir/$example_file"; then
      echo "Installed ML-YE3T example differs: examples/PACKAGES/ye3t/$example_file" >&2
      exit 1
    fi
  done
fi

if test "$mode" = --check; then
  if test "$cmake_state" = clean; then
    diff -u "$cmake_file" "$stage_dir/CMakeLists.txt" || true
    echo "Check passed; --apply will install the declared CPU and Kokkos source/header manifests, package metadata, the pair_style ye3t documentation page, and public examples."
  else
    echo "Check passed; identical ML-YE3T package is already installed."
  fi
  exit 0
fi

if test "$cmake_state" = patched; then
  echo "Identical ML-YE3T package is already installed in $lammps_root"
  exit 0
fi

mkdir -p "$package_dir"
for source_file in $source_files; do
  install -m 0644 "$repo_root/src/$source_file" "$package_dir/$source_file"
done
for source_file in $kokkos_source_files; do
  install -m 0644 "$repo_root/src/KOKKOS/$source_file" "$kokkos_dir/$source_file"
done
install -m 0644 "$repo_root/lammps_package/README" "$package_dir/README"
install -m 0644 "$repo_root/cmake/ML-YE3T.cmake" "$module_file"
install -m 0644 "$doc_source_file" "$doc_file"
install -m 0644 "$property_doc_source_file" "$property_doc_file"
mkdir -p "$example_dir"
for example_file in $example_files; do
  destination=$example_dir/$example_file
  mkdir -p "$(dirname -- "$destination")"
  case "$example_file" in
    ./run_examples.sh|./verify_examples.py|run_examples.sh|verify_examples.py)
      file_mode=0755
      ;;
    *) file_mode=0644 ;;
  esac
  install -m "$file_mode" "$example_source_dir/$example_file" "$destination"
done
install -m 0644 "$example_source_manifest" "$example_manifest"
install -m 0644 "$stage_dir/CMakeLists.txt" "$cmake_file"

(
  cd "$package_dir"
  for source_file in $kokkos_source_files; do
    sha256sum "../KOKKOS/$source_file"
  done
) > "$kokkos_manifest_file"

(
  cd "$package_dir"
  for source_file in $source_files README YE3T_LAMMPS_KOKKOS_MANIFEST.sha256; do
    sha256sum "$source_file"
  done
) > "$manifest_file"

if git -C "$lammps_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  revision=$(git -C "$lammps_root" rev-parse HEAD 2>/dev/null || printf unknown)
else
  revision=not-a-git-tree
fi
if git -C "$repo_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  package_revision=$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || printf unknown)
else
  package_revision=not-a-git-tree
fi
printf '%s\n' "ye3t-lammps source integration" \
  "lammps_revision=$revision" \
  "ye3t_lammps_revision=$package_revision" \
  "examples=examples/PACKAGES/ye3t" \
  "documentation=doc/src/pair_ye3t.rst,doc/src/compute_ye3t_property_atom.rst" > "$install_record"

echo "Installed ML-YE3T into $lammps_root"
echo "Configure with -D PKG_ML-YE3T=yes and an explicit ML_YE3T_RUNTIME_SOURCE or ML_YE3T_RUNTIME_ROOT."
