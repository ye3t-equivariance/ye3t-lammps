# Figure captions

## Three-way accuracy/runtime Pareto figures

Held-out energy or force RMSE versus fixed-position one-rank CPU time for
PACE `product`, compiler-planned YE3T-symmetric AUTO on the byte-identical
ordinary model, and YE3T-mixed tagged AUTO models. Black outlines mark points that
are nondominated within the measured catalogue; numeric labels are descriptor
counts. An all-direct schedule is accepted only when AUTO loaded and scored a
nonempty optimized candidate portfolio and did not enter a fallback state.
Tagged AUTO decisions use an 11-repeat rotated-order quartile confidence gate;
the retained fitted catalogues selected compiled-direct, while repetition-rich
microcatalogues can select block or symmetric-power execution.

## High-order AUTO microcatalogue

Speedup of generic DAG, symmetric-power, and block execution over
compiled-direct for 512-feature repetition-rich catalogues through N=32.
Black rings identify the conservative AUTO decision. These are exact kernel
and adjoint benchmarks, not fitted-potential accuracy results.
