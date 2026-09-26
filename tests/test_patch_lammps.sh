#!/bin/sh

set -eu

if test "$#" -ne 1; then
  echo "Usage: test_patch_lammps.sh YE3T_LAMMPS_ROOT" >&2
  exit 2
fi

repo_root=$(CDPATH= cd -- "$1" && pwd -P)
tmp_base=${TMPDIR:-/tmp}
fixture_root=$(mktemp -d "$tmp_base/ye3t-lammps-patch-test.XXXXXX")
cleanup()
{
  case "$fixture_root" in
    "$tmp_base"/ye3t-lammps-patch-test.*) rm -rf -- "$fixture_root" ;;
    *) echo "Refusing to remove unexpected fixture path: $fixture_root" >&2 ;;
  esac
}
trap cleanup EXIT HUP INT TERM

mkdir -p "$fixture_root/src/KOKKOS" "$fixture_root/cmake/Modules/Packages" \
  "$fixture_root/doc/src" "$fixture_root/examples/PACKAGES"
: > "$fixture_root/src/lammps.cpp"
cat > "$fixture_root/cmake/CMakeLists.txt" <<'EOF'
set(STANDARD_PACKAGES
  ML-IAP
  ML-PACE
  ML-SNAP)

foreach(PKG_WITH_INCL GRAPHICS ML-IAP COMPRESS ML-PACE LEPTON FENIX)
  if(PKG_${PKG_WITH_INCL})
    include(Packages/${PKG_WITH_INCL})
  endif()
endforeach()
EOF

patcher=$repo_root/tools/patch_lammps.sh
"$patcher" --check --lammps-source "$fixture_root" >/dev/null
test ! -e "$fixture_root/src/ML-YE3T"

"$patcher" --apply --lammps-source "$fixture_root" >/dev/null
test -f "$fixture_root/src/ML-YE3T/pair_ye3t.cpp"
test -f "$fixture_root/src/ML-YE3T/ye3t_cpu_batching.h"
test -f "$fixture_root/src/ML-YE3T/ye3t_gpu_dag_schedule.h"
test -f "$fixture_root/src/ML-YE3T/ye3t_gpu_tagged_source.h"
test -f "$fixture_root/src/ML-YE3T/ye3t_tagged_cauchy_readout_plan.h"
test -f "$fixture_root/src/ML-YE3T/ye3t_cpu_source_tiling.h"
test -f "$fixture_root/src/ML-YE3T/ye3t_cpu_source_tiling.cpp"
test -f "$fixture_root/src/ML-YE3T/ye3t_lifted_cauchy_source.cpp"
test -f "$fixture_root/src/ML-YE3T/YE3T_LAMMPS_MANIFEST.sha256"
test -f "$fixture_root/src/ML-YE3T/YE3T_LAMMPS_KOKKOS_MANIFEST.sha256"
test -f "$fixture_root/src/KOKKOS/pair_ye3t_kokkos.cpp"
test -f "$fixture_root/src/KOKKOS/pair_ye3t_kokkos.h"
test -f "$fixture_root/src/KOKKOS/ye3t_lifted_cauchy_kokkos_plan.h"
test -f "$fixture_root/src/KOKKOS/ye3t_tagged_cauchy_kokkos_plan.h"
test -f "$fixture_root/src/KOKKOS/ye3t_kokkos_plan.h"
test -f "$fixture_root/src/KOKKOS/ye3t_kokkos_step_state.h"
test -f "$fixture_root/src/KOKKOS/ye3t_kokkos_types.h"
test -f "$fixture_root/cmake/Modules/Packages/ML-YE3T.cmake"
cmp "$repo_root/doc/src/pair_ye3t.rst" "$fixture_root/doc/src/pair_ye3t.rst"
test -f "$fixture_root/examples/PACKAGES/ye3t/YE3T_EXAMPLE_MANIFEST.sha256"
test -x "$fixture_root/examples/PACKAGES/ye3t/run_examples.sh"
test -x "$fixture_root/examples/PACKAGES/ye3t/verify_examples.py"
test -f "$fixture_root/examples/PACKAGES/ye3t/models/ta_l8_h16/model.yace"
test -f "$fixture_root/examples/PACKAGES/ye3t/cost_comparison/Ni/in.ni_ye3t_tagged_127"
test -f "$fixture_root/examples/PACKAGES/ye3t/cost_comparison/Ni/models/ye3t_tagged_127/model.ye3t.json"
cmp "$repo_root/examples/PACKAGES/ye3t/YE3T_EXAMPLE_MANIFEST.sha256" \
  "$fixture_root/examples/PACKAGES/ye3t/YE3T_EXAMPLE_MANIFEST.sha256"
test "$(grep -c '^  ML-YE3T$' "$fixture_root/cmake/CMakeLists.txt")" -eq 1
test "$(grep -c 'ML-PACE ML-YE3T LEPTON' "$fixture_root/cmake/CMakeLists.txt")" -eq 1

first_hash=$(sha256sum "$fixture_root/cmake/CMakeLists.txt")
"$patcher" --apply --lammps-source "$fixture_root" >/dev/null
second_hash=$(sha256sum "$fixture_root/cmake/CMakeLists.txt")
test "$first_hash" = "$second_hash"

manifest=$fixture_root/src/ML-YE3T/YE3T_LAMMPS_MANIFEST.sha256
kokkos_manifest=$fixture_root/src/ML-YE3T/YE3T_LAMMPS_KOKKOS_MANIFEST.sha256
install_record=$fixture_root/src/ML-YE3T/YE3T_LAMMPS_INSTALL.txt
example_manifest=$fixture_root/examples/PACKAGES/ye3t/YE3T_EXAMPLE_MANIFEST.sha256

mv "$manifest" "$manifest.saved"
if "$patcher" --check --lammps-source "$fixture_root" >/dev/null 2>&1; then
  echo "Patcher accepted an installation without its checksum manifest" >&2
  exit 1
fi
mv "$manifest.saved" "$manifest"

mv "$kokkos_manifest" "$kokkos_manifest.saved"
if "$patcher" --check --lammps-source "$fixture_root" >/dev/null 2>&1; then
  echo "Patcher accepted an installation without its Kokkos checksum manifest" >&2
  exit 1
fi
mv "$kokkos_manifest.saved" "$kokkos_manifest"

cp "$kokkos_manifest" "$kokkos_manifest.saved"
printf '%064d  ../KOKKOS/pair_ye3t_kokkos.cpp\n' 0 >> "$kokkos_manifest"
if "$patcher" --check --lammps-source "$fixture_root" >/dev/null 2>&1; then
  echo "Patcher accepted a corrupted Kokkos checksum manifest" >&2
  exit 1
fi
mv "$kokkos_manifest.saved" "$kokkos_manifest"

cp "$manifest" "$manifest.saved"
printf '%064d  README\n' 0 >> "$manifest"
if "$patcher" --check --lammps-source "$fixture_root" >/dev/null 2>&1; then
  echo "Patcher accepted a corrupted checksum manifest" >&2
  exit 1
fi
mv "$manifest.saved" "$manifest"

mv "$install_record" "$install_record.saved"
if "$patcher" --check --lammps-source "$fixture_root" >/dev/null 2>&1; then
  echo "Patcher accepted an installation without its install record" >&2
  exit 1
fi
mv "$install_record.saved" "$install_record"

mv "$example_manifest" "$example_manifest.saved"
if "$patcher" --check --lammps-source "$fixture_root" >/dev/null 2>&1; then
  echo "Patcher accepted an installation without its example manifest" >&2
  exit 1
fi
mv "$example_manifest.saved" "$example_manifest"

example_readme=$fixture_root/examples/PACKAGES/ye3t/README.md
cp "$example_readme" "$example_readme.saved"
printf '\nlocal example conflict\n' >> "$example_readme"
if "$patcher" --check --lammps-source "$fixture_root" >/dev/null 2>&1; then
  echo "Patcher accepted a locally modified installed example" >&2
  exit 1
fi
mv "$example_readme.saved" "$example_readme"

chmod -x "$fixture_root/examples/PACKAGES/ye3t/run_examples.sh"
if "$patcher" --check --lammps-source "$fixture_root" >/dev/null 2>&1; then
  echo "Patcher accepted a non-executable installed example runner" >&2
  exit 1
fi
chmod +x "$fixture_root/examples/PACKAGES/ye3t/run_examples.sh"

chmod -x "$fixture_root/examples/PACKAGES/ye3t/verify_examples.py"
if "$patcher" --check --lammps-source "$fixture_root" >/dev/null 2>&1; then
  echo "Patcher accepted a non-executable installed example verifier" >&2
  exit 1
fi
chmod +x "$fixture_root/examples/PACKAGES/ye3t/verify_examples.py"

kokkos_source=$fixture_root/src/KOKKOS/pair_ye3t_kokkos.cpp
cp "$kokkos_source" "$kokkos_source.saved"
printf '\n// local Kokkos conflict\n' >> "$kokkos_source"
if "$patcher" --check --lammps-source "$fixture_root" >/dev/null 2>&1; then
  echo "Patcher accepted a locally modified installed Kokkos source" >&2
  exit 1
fi
mv "$kokkos_source.saved" "$kokkos_source"

printf '\n// local conflict\n' >> "$fixture_root/src/ML-YE3T/pair_ye3t.cpp"
if "$patcher" --check --lammps-source "$fixture_root" >/dev/null 2>&1; then
  echo "Patcher accepted a locally modified installed source" >&2
  exit 1
fi
