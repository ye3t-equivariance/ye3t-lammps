# Reference CPU results

The `log.25Sep2026.*.g++.1` and `.g++.4` files beside each input are the
reference logs of the shipped inputs on one and four MPI ranks. They were
produced by `run_examples.sh` and by running every `cost_comparison` input
from its own directory with default variables. They demonstrate correctness;
they are not timing data.

- LAMMPS: stable release 22 Jul 2025 - Update 4
  (`611ca3b7f8525ba802373b04f9e3d632d515f7e3`) with ML-PACE
  `v.2025.12.4.p1`, EXTRA-FIX, and ML-YE3T installed by `tools/patch_lammps.sh`
- build: Release, GCC 15.2.0, MPICH 5.0.1
- precision: `double`
- ranks: one and four MPI processes on one Linux host

## Same-model PACE comparison

The primary comparison used identical `models/ta_l8_compact/model.yace` bytes
for ML-PACE and ML-YE3T. Residuals are the maxima over atoms reported by
`verify_examples.py` (`dumps` check) with `pace product` as the reference.

| comparison against `pace product` | MPI ranks | max atomic energy (eV) | max force component (eV/A) | max per-atom virial component (eV) | max total virial component (eV) |
|---|---:|---:|---:|---:|---:|
| `pace recursive` | 1 | 2.84e-14 | 1.10e-14 | 3.26e-14 | 5.12e-14 |
| `ye3t` direct | 1 | 8.53e-14 | 6.22e-12 | 1.47e-11 | 6.28e-10 |
| `ye3t` block | 1 | 9.24e-14 | 6.22e-12 | 1.47e-11 | 6.28e-10 |
| `ye3t` AUTO | 1 | 8.53e-14 | 6.22e-12 | 1.47e-11 | 6.28e-10 |
| `pace recursive` | 4 | 2.84e-14 | 1.01e-14 | 3.26e-14 | 6.51e-14 |
| `ye3t` direct | 4 | 8.53e-14 | 6.22e-12 | 1.47e-11 | 6.28e-10 |
| `ye3t` block | 4 | 9.24e-14 | 6.22e-12 | 1.47e-11 | 6.28e-10 |
| `ye3t` AUTO | 4 | 8.53e-14 | 6.22e-12 | 1.47e-11 | 6.28e-10 |

Direct, block, and AUTO agree with each other more closely than with PACE:
block versus direct differs by at most `3.55e-15` eV/A in a force component,
and AUTO reproduces direct exactly on this catalogue. The 61-function full
model (`models/ta_l8_full`) also matched `pace product`; its largest force
difference was `6.15e-12` eV/A for both the direct and the coupled-product
route, and coupled-product versus direct agreed within `1.39e-15` eV/A.

The fixed-position replays (`in.ye3t.replay-direct`, `in.ye3t.pace-replay`)
produced six identical frames each; the final YE3T and PACE frames differ by
the same `6.22e-12` eV/A as the static comparison.

## Domain crossing, derivatives, and NVE

The atom-domain-crossing input (`in.ye3t.mpi-migration`) on one and four
ranks gave:

| quantity | maximum absolute difference |
|---|---:|
| atomic energy | 2.84e-14 eV |
| force component | 1.48e-14 eV/A |
| per-atom virial component | 4.77e-14 eV |
| total virial component | 6.28e-14 eV |

LAMMPS numerical differentiation (`in.ye3t.numdiff`, `in.ye3t.numdiff-direct`)
passed at all three requested step sizes. At the default medium steps
(`1e-4` A and `1e-6` strain), the maximum force error was `1.61e-8` eV/A on
one rank and `1.71e-8` eV/A on four ranks; the six-component pressure error
norm was `1.07e-3` bar and `9.73e-4` bar. The coarse and fine steps gave
`1.32e-7` and `3.13e-9` eV/A on one rank.

The per-atom stress sum agrees with the global pair pressure within
`1.42e-14` eV (`virial_consistency` check).

The 128-atom, 300 K, 20-step NVE inputs (`in.ye3t.md`, `in.ye3t.md-direct`)
ran on one and four ranks with a maximum total-energy drift of `1.13e-6`
eV/atom at a 1 fs timestep.

## Rank-16 execution fixture

For `models/ta_l8_h16`, direct and block execution agreed within `2.66e-15`
eV/A on one rank and `1.78e-15` eV/A on four ranks, and both matched
`pace product` on the same file within `1.11e-11` eV/A. These coefficients
are execution sentinels, not a fitted material model, so this establishes
evaluator availability and parity only.

## Fitted element models

Each `cost_comparison/<element>` directory carries its own reference logs.
The finite-difference inputs report the maximum force error of `fix numdiff`
on one displaced atom and the six-component pressure error norm of
`fix numdiff/virial` (one rank):

| model | force error (eV/A) | pressure error norm (bar) |
|---|---:|---:|
| Li `ace_127` / `ye3t_tagged_127` | 1.24e-9 / 1.46e-8 | 2.43e-4 / 3.02e-4 |
| Mo `ace_127` / `ye3t_tagged_127` | 6.05e-9 / 8.52e-9 | 6.79e-3 / 6.05e-3 |
| Cu `ace_127` / `ye3t_tagged_127` | 3.34e-8 / 4.27e-8 | 1.06e-3 / 2.14e-3 |
| Ni `ace_127` / `ye3t_tagged_127` | 1.87e-8 / 3.34e-8 | 1.21e-3 / 8.18e-3 |
| Ni `ye3t_augmented_196` | 7.33e-8 | 5.13e-3 |
| Si `ace_127` / `ye3t_tagged_127` | 1.13e-8 / 1.02e-8 | 2.49e-3 / 2.06e-3 |
| Ge `ace_127` / `ye3t_tagged_127` | 1.62e-8 / 4.75e-8 | 2.61e-3 / 8.43e-4 |

The `.nve` inputs (2,000 NVT steps followed by 10,000 NVE steps of a 2x2x2
conventional cell: 16 atoms for bcc Li and Mo, 32 for fcc Cu and Ni, 64 for
diamond Si and Ge; 0.5 fs timestep) completed on one and four ranks with
identical thermo output. The maximum total-energy excursion over the NVE
segment was between `1.5e-6` eV/atom (Li tagged) and `4.2e-5` eV/atom (Si
tagged); the Ni augmented model gave `1.1e-5` eV/atom.

Re-run the complete machine-checked CPU matrix with:

```bash
YE3T_LMP=/absolute/path/to/lmp YE3T_MPI_RANKS=4 YE3T_WITH_PACE=yes \
  YE3T_WITH_NUMDIFF=yes YE3T_WITH_HIGH_RANK=yes ./run_examples.sh
```

The runner writes JSON verification reports containing the measured residuals
and exits nonzero on failure.
