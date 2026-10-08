# PACE product / YE3T-symmetric / YE3T-mixed result bundle

This result bundle separates three runtime series:

1. **PACE product** evaluates each ordinary ACE `.yace` with
   `pair_style pace product`.
2. **YE3T-symmetric (AUTO)** evaluates the byte-identical `.yace` with
   `pair_style ye3t`, a compiler-produced plan, and `block_policy auto`.
3. **YE3T-mixed (AUTO)** evaluates the ordinary-plus-tagged model entirely
   through `pair_style ye3t auto`. AUTO calibrates compiled-direct, generic
   DAG, symmetric-power, and block candidates on the active machine and
   catalogue. These fitted catalogues selected compiled-direct; this is
   a measured decision, not a fallback or a restriction on other catalogues.

ZBL is a separately fitted reference overlay in every series. No
YE3T-labelled timing or validation row invokes `pair_style pace`.

Every YE3T-symmetric log must load exactly one compiler plan containing at
least one optimized candidate. The benchmark fails on missing plans and on
the `budget_direct_fallback`, `conservative_direct_fallback`, or
`selected_direct_no_authorized_profile` states. Across these runs,
AUTO resolved 0 model/rank rows with non-direct routes and 18 with an
all-direct schedule selected from a measured candidate portfolio. The
latter is a planner decision, not an unplanned fallback.

Tagged AUTO uses 128 centers, 16 synthetic neighbors per center, 11 rotated
candidate-order repetitions, and a conservative quartile confidence gate. A
candidate replaces direct only when its 75th-percentile calibration time is
below 92% of direct's 25th percentile. The selected route is fixed before the
timed trajectory and calibration time is excluded from steady-state timing.

The byte-identical PACE-product and YE3T-symmetric models agree across all six
systems. The largest force-component discrepancy is 7.12e-12 eV/Å, and all
energy, atomic-energy, and virial parity checks pass.

At 127 matched descriptors, tagged YE3T lowers held-out force RMSE on all six
systems by 3.5–75.6% while costing 0.94–1.11 times PACE product on one CPU
rank. For the same ordinary model, YE3T-symmetric costs 0.30–0.37 times PACE
product at 127 descriptors. Mixed YE3T is faster than PACE product at 127
descriptors for Li, Mo, Si, and Ge. The complete six-system curves place at least
one mixed model on the measured force accuracy–runtime Pareto frontier for
Cu, Ge, Li, Mo, Ni, Si. All six systems use the same 60/127/149
descriptor-count tiers.

The tagged models select component 10 of the frozen catalogue. Its compiler
labels contain both a nontrivial repeated-block permutation partition and
nonzero intermediate angular momentum. A matched 60-descriptor Cu ablation
replacing that component with nonjoint tagged components lowers held-out force
RMSE by 14.6% when the joint component is restored. This is the
causal evidence for the combined permutation/rotation information; the broader
six-system comparison establishes descriptor efficiency and Pareto behavior.

The timing results apply to the tested catalogues and reference CPU. Mixed YE3T
is faster than PACE product at several equal-count points and no more than
1.11 times PACE at 127 descriptors, while adding
new nondominated accuracy/runtime points. Repetition-rich high-order
microcatalogues select block or symmetric-power routes on the same machine,
demonstrating why AUTO must remain catalogue-aware. Black-outlined markers
are mechanically nondominated within the measured model set.

The idle 512-feature repetition-rich microcatalogue ladder makes the catalogue
dependence explicit: AUTO selected direct for N=4,6,8,12, symmetric-power for
N=16,32, and block for N=24. Across all routes, the largest value/adjoint
discrepancy from direct is 8.88e-16.

## Provenance and reproduction

This directory contains compact evidence: tables, figures, source-data CSVs,
claims, provenance, and validation records. The raw LAMMPS inputs, logs,
repetitions, summaries, and accuracy tables that produced them are stored in
a private workflow archive. The `ye3t-workflows/...` and `ye3t-ace/...` paths
recorded in `provenance.json` and `accuracy_runtime_source_data.csv` are
internal study-archive and fitting-package locations kept as frozen
provenance; they are not paths in this repository. The renderer that
regenerates the figures from that archive is part of the `ye3t-ace` fitting
archive outside this repository; its invocation is recorded here as provenance:

```bash
python ye3t-ace/examples/publication/cost_comparison/workflow/render_self_contained_publication.py \
  --workspace "$PWD" \
  --output ye3t-ace/examples/publication/cost_comparison/results/three_way_pace_ye3t_auto_v3_20260921 \
  --reuse-promoted-raw
```

That command checks that every accuracy, timing, parity, ablation,
and qualification input is present before replacing any derived table or
figure. The LAMMPS replay inputs, models, and reference logs for every
element are in `examples/PACKAGES/ye3t/cost_comparison`, with the run
commands in its README and each element directory.

All six self-contained 127-descriptor models also pass LAMMPS force/virial
finite differences and a 2,000-step NVT plus 10,000-step NVE run. The largest
absolute energy drift is 1.78e-06 eV/atom/ps, below the acceptance limit of
1e-3 eV/atom/ps.
