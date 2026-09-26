# ML-YE3T LAMMPS examples

These are source-integrated LAMMPS examples. Run them from this directory in
the `ye3t-lammps` repository or from `examples/PACKAGES/ye3t` in a LAMMPS tree
after `tools/patch_lammps.sh --apply`.

## Contents

| location | contents | use |
|---|---|---|
| `models/ta_l8_compact` | fitted 58-function rank-through-8 Ta model and plan | static checks and short MD; direct/block/AUTO |
| `models/ta_l8_full` | fitted 61-function rank-through-8 Ta model and plan | static checks; direct/coupled-product |
| `models/ta_l8_h16` | compact Ta control plus one rank-16 execution sentinel | rank-16 evaluator availability only |
| `cost_comparison/<element>` | fitted Li, Mo, Cu, Ni, Si, and Ge linear ACE controls and tagged YE3T models on the `mlearn` split | same-model PACE/YE3T comparisons, tagged models, finite differences, NVE |
| `in.ye3t.*` | Ta inputs for every evaluator, MPI, finite-difference, replay, and NVE check | individual runs and `run_examples.sh` |
| `log.*.g++.1`, `log.*.g++.4` | reference logs for the Ta inputs on one and four MPI ranks | expected output |

The rank-16 sentinel coefficients were chosen to exercise kernels and were
not fitted to material targets. They must not be used for material MD or
accuracy claims. Model hashes are listed in `models/MODELS.md`; every file
in this directory is listed with its SHA-256 in `YE3T_EXAMPLE_MANIFEST.sha256`.

The measured one- and four-rank residuals of the reference logs are
summarized in `REFERENCE_RESULTS.md`. Those logs demonstrate correctness; they
are not timing data.

## One-process runs

Set an absolute LAMMPS binary:

```bash
LMP=/absolute/path/to/lmp
```

Each `in.*` file has local defaults and runs independently. For example, the
fitted compact model with the direct evaluator is simply:

```bash
"$LMP" -in in.ye3t.direct
```

Fitted compact model, forced block and AUTO evaluators:

```bash
"$LMP" -in in.ye3t.block
"$LMP" -in in.ye3t.auto
```

The full 61-function model runs through either the direct evaluator or a
compiled plan with one coupled-product DAG and an explicit direct residual:

```bash
"$LMP" -var model models/ta_l8_full/model.yace \
  -var dump_path dump.ye3t.full-direct -in in.ye3t.direct
"$LMP" -in in.ye3t.coupled-product
```

Both commands read the identical standard `.yace`; the coupled command also
reads its hash-bound plan and function map. The plan changes execution, not
the potential or fitted coefficients.

## Same-potential PACE comparison

The comparison requires a LAMMPS binary built with both `ML-PACE` and
`ML-YE3T`. It deliberately supplies the identical model file to both styles.
The product evaluator is the primary baseline; the bounded recursive input is
included as a secondary reference:

```bash
"$LMP" -in in.ye3t.pace-product
"$LMP" -in in.ye3t.pace-recursive
```

`verify_examples.py` compares dumps and logs. It takes no command-line
arguments; the check and its inputs are read from a JSON file named by the
`CONFIG_PATH` environment variable, and every tolerance has a visible default
in the `TOLERANCES` dictionary at the top of the script:

```bash
cat > parity.json <<'EOF'
{"check": "dumps",
 "reference": "dump.ye3t.pace-product",
 "candidates": ["dump.ye3t.direct", "dump.ye3t.block", "dump.ye3t.auto"]}
EOF
CONFIG_PATH=parity.json python3 verify_examples.py
```

The verifier compares atom IDs/types, positions, atomic and total energies,
every force component, and all six per-atom virial components. It prints the
measured residuals and exits nonzero if a tolerance is exceeded. The other
checks are `numdiff` (finite-difference logs), `replay` (fixed-position
repeated evaluations), `md` (bounded energy drift), and `virial_consistency`
(per-atom stress sum against the global pair pressure).

`in.ye3t.replay-direct` and `in.ye3t.pace-replay` repeat six
force/energy/virial evaluations without integrating the atoms. The runner
verifies that positions remain fixed, bounds repeated-evaluation variation,
and compares the final PACE and YE3T frames.

## Fitted element models and tagged YE3T

`cost_comparison/<element>` holds, for each of Li, Mo, Cu, Ni, Si, and Ge,
an ordinary 127-descriptor ACE control and the matched 127-descriptor tagged
YE3T model fitted on the published `mlearn` split, together with runnable
inputs for PACE product, YE3T on the identical `.yace`, and the tagged model
through `pair_style ye3t` alone. Ni additionally carries the 60- and
149-descriptor tiers and the 196-descriptor augmented model. See
`cost_comparison/README.md` and each element README.

## Numerical derivatives

`in.ye3t.numdiff` uses LAMMPS `fix numdiff` on one displaced atom and
`fix numdiff/virial` on the cell. Build LAMMPS with `PKG_EXTRA-FIX=ON`.
The 3x3x3 cell is larger than twice the model cutoff, avoiding repeated
periodic-image neighbors. The runner performs a small displacement/strain
sweep and checks the reported maximum errors.

```bash
YE3T_LMP="$LMP" YE3T_WITH_NUMDIFF=yes ./run_examples.sh
```

`in.ye3t.numdiff-direct` runs the same check without a plan sidecar. Use it
for a newly exported standalone `.yace` file:

```bash
"$LMP" -var model /absolute/path/to/model.yace -in in.ye3t.numdiff-direct
```

## MPI and domain crossing

All CPU YE3T runs require `newton on`. Run the fitted matrix on four ranks:

```bash
YE3T_LMP="$LMP" YE3T_MPI_RANKS=4 ./run_examples.sh
```

For ranks greater than one, the runner also executes the same atom-domain
crossing input (`in.ye3t.mpi-migration`) on one and N ranks and checks the
final energy, forces, and virials. `YE3T_MPIEXEC` selects another launcher:

```bash
YE3T_LMP="$LMP" YE3T_MPI_RANKS=4 YE3T_MPIEXEC=srun ./run_examples.sh
```

## Rank-16 execution

`models/ta_l8_h16` appends one tiny homogeneous rank-16 sentinel to the
fitted compact model so the direct and block evaluators can be exercised
beyond rank 8:

```bash
"$LMP" -in in.ye3t.high-rank-direct
"$LMP" -in in.ye3t.high-rank-block
YE3T_LMP="$LMP" YE3T_WITH_HIGH_RANK=yes ./run_examples.sh
```

This is an availability check, not a fitted high-rank model.

## Kokkos GPU correctness

A LAMMPS binary built with `ML-YE3T`, `KOKKOS`, FP64, and a supported
device backend runs the exact direct or backend-planned AUTO evaluator with
conventional suffix selection:

```bash
"$LMP" -k on g 1 -pk kokkos neigh half -sf kk -in in.ye3t.direct
"$LMP" -k on g 1 -pk kokkos neigh half -sf kk -in in.ye3t.auto
```

Run CPU and GPU checks together, including the same-model
`pace/kk product` comparison when `ML-PACE` is in the same binary:

```bash
YE3T_LMP="$LMP" YE3T_WITH_KOKKOS=yes YE3T_WITH_PACE=yes \
  YE3T_WITH_NUMDIFF=yes YE3T_WITH_HIGH_RANK=yes ./run_examples.sh
```

The Kokkos lane asserts a CUDA or HIP execution space, a successful device
plan-copy sentinel, no CPU evaluator dispatch, energy/force/virial parity,
short NVE, fixed-position replay, and direct, AUTO, block, and coupled-product
finite differences when requested. GPU AUTO validates the full candidate
portfolio, records its profile, reason, and plan identity, and makes one
deterministic catalogue-level choice; without a calibrated replay for the
active GPU class it records a conservative decision and selects direct.

Use `YE3T_MPI_RANKS=N YE3T_KOKKOS_GPUS=N` for a single-node one-rank-per-GPU
correctness run. If ranks exceed the requested GPUs, the result is explicitly
oversubscribed correctness and must not be used as scaling evidence.

GPU calibration, replay generation, and the timing/scaling workflow live in
`tools/gpu_scaling/` of the `ye3t-lammps` repository; they are developer
tools rather than examples and keep their own command-line interfaces.

## Complete validation runner

`run_examples.sh` takes no command-line arguments. Its switches are the
environment variables listed at the top of the script:

```bash
YE3T_LMP="$LMP" YE3T_MPI_RANKS=4 YE3T_WITH_PACE=yes \
  YE3T_WITH_NUMDIFF=yes YE3T_WITH_HIGH_RANK=yes ./run_examples.sh
```

The runner writes every log, dump, and JSON verification report into a new
`generated/run-PID` directory (or `YE3T_EXAMPLE_OUTPUT`), applies
`YE3T_EXAMPLE_TIMEOUT` seconds to every LAMMPS process, and exits nonzero on
the first failed check. Use an otherwise idle machine and a dedicated
benchmark workflow for timing; the example runner is a correctness workflow.

The short NVE lane rejects non-finite thermo output and requires the maximum
total-energy drift to remain below `1e-4` eV/atom over 20 steps.
`in.ye3t.md-direct` provides the same bounded run for a newly exported
standalone `.yace` file without requiring a matching plan sidecar.

## Training and plan generation

LAMMPS performs inference only. The corresponding installed-package examples
live in the other YE3T packages:

- `ye3t/examples/compile_scalar_ace_lammps_plans.py` in the public `ye3t`
  repository compiles representative scalar coordinates and execution plans
  through `ye3t.couplings`;
- the Ta catalogue, split, fit, strict `.yace` export, and validation
  workflow lives in `ye3t-ace` (not yet public);
- the fitting and export workflow for the element models in
  `cost_comparison/` also lives in `ye3t-ace` (not yet public).

The selected checked models are included here so these LAMMPS examples remain
self-contained after `tools/patch_lammps.sh --apply`.
