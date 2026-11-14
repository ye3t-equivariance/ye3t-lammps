# Exact empty tagged assignments: CPU evaluator update

This change consumes `ye3t_distinct_occurrence_support_v1` from the compiler's
V3/V4 physical source plan. For each center it counts species-resolved physical
occurrences inside the directed pair cutoff. It skips a feature and its
derivatives when no contributing assignment can exist. Atom-image occurrences
are distinct; tensor rank is not substituted for distinct tag count.

The inventory belongs to the source binding and its hash. It does not change
coupling coefficients, feature coordinates, fitted weights or radial formulas.
No floating-point threshold determines which features vanish.

## Migration and compatibility

The implementation originated in the YE3T Methods working tree based on
`5796e546e604`. The five shared files are `ye3t_tagged_cauchy_model.{h,cpp}`,
`ye3t_tagged_cauchy_cpu.{h,cpp}` and `ye3t_tagged_cauchy_readout_plan.h`.
They were merged against the Methods base into the corresponding LAMMPS files,
preserving the earlier indexed-loader migration and LAMMPS-specific changes.
Existing GPL and copyright notices remain intact. No external implementation
was copied.

- New exports require this CPU library or an ABI-matched CPU plugin. Old
  readers reject the extended source certificate.
- Old V3/V4 files can acquire the inventory by loading and re-exporting through
  the updated Python model interface. Direct native loading of an unmodified
  old file preserves its old behavior.
- Historical V2 formulas and coefficient coordinates remain unchanged.
- Current Kokkos planning rejects the new source extension explicitly. GPU
  implementation and qualification are still required.
- Fully populated environments retain the selected execution policy. Where a
  folded readout merges available and unavailable features, evaluation uses
  the existing direct terms so that unavailable contributions are skipped
  before multiplication by fitted weights.

## Evidence

The focused CTest group passes 13/13 tests, including existing tagged CPU,
source/portfolio integrity, mean-property and host GPU-math checks. Host
GPU-math tests do not constitute device qualification.

A four-atom, mixed-species, periodic rank-1--4 fitted export was evaluated with
the CPU plugin. Maximum energy/force differences against the corrected Python
model were `8.81e-13 eV` and `1.48e-12 eV/Angstrom`. The first final rerun stalled
in WSL process startup. After the user-authorized WSL restart, the final
populated-loop implementation passed the zero-step LAMMPS rerun with those
same residuals. The rebuilt shared CPU library and 13 native tests also pass.

On a 48-atom populated geometry with 56 features, 30 alternating timing pairs
gave candidate/baseline ratio 0.9678, bootstrap 95% interval [0.9586, 0.9774].
This is a bounded CPU evaluator check, not a replay of the paper benchmark.

Raw evidence, commands, preserved rejected timings and limitations live in
`ye3t-workflows/MLIP/linear_pair_config_20261113/exact_support_fix/`.

## Deferred TODO: LINEAR-CHEB-C2

The user deferred pooled tagged-linear C2 Chebyshev integration on 2026-11-13
to prioritize coefficient remediation and standardization. The current shared
Chebyshev retained-source tests do not qualify this CPU LAMMPS export path.

After the corresponding YE3T/Methods integration, propagate the source-family
reader and radial value/derivative changes into
`src/ye3t_tagged_cauchy_model.{h,cpp}` and
`src/ye3t_tagged_cauchy_cpu.{h,cpp}` as needed. Reuse the existing Chebyshev C2
definition and shared couplings; do not apply Jacobi moment coefficients to
Chebyshev radial values. Preserve native source licenses, old artifact
coordinates and published tagged-model performance.

Gate completion on an actual exported-model run with energy/force agreement,
empty/populated supports, pair cutoffs, periodic images, legacy artifact
regressions and matched performance. Source-family and cache identities must
remain distinguishable. Keep unsupported device routes explicit. The shared
checklist is `LINEAR-CHEB-C2` in YE3T Methods
`docs/linear_standardization.rst`; the core request/payload entry point is
`ye3t/couplings/tagged_cauchy_general.py`.
