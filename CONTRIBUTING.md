# Contributing

- C++ sources follow the LAMMPS conventions: the LAMMPS header banner with a
  contributing-author block, `LMP_` include guards, and the repository
  `.clang-format`. `python3 tests/test_source_contracts.py` checks the
  banners, guards, and the installer's file inventory; run it after adding
  or renaming a source file, and add new files to `tools/patch_lammps.sh`
  and `tests/test_patch_lammps.sh`.
- After editing anything under `examples/PACKAGES/ye3t`, run
  `python3 tools/refresh_example_manifest.py`; the installer verifies the
  manifest.
- Keep build trees, raw benchmark output, and generated example output
  outside the repository. The promoted results under `docs/results` carry
  their own checksum files.
- Run the inexpensive checks in `tests/README.md` before opening a pull
  request; the source-package and CUDA tests need a LAMMPS build.
