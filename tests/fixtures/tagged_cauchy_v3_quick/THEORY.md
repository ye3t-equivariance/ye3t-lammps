# Bounded tagged-Cauchy V3 construction

This fixture tests a narrow but exact vertical slice.  At fixed tensor order
`N=4`, tagged coordinates with tag counts `s=0,1,2` are compiled in the
globally permutation-invariant, rotationally scalar parent.  Young partitions
are denoted by `kappa`; `s` is only the number of explicit tag positions.

For the one-neighbor source, with `x=r/r_c`, the current certificate uses

```text
phi_{q l m}(r) = N_{q l} x^l (1-x)^2
                 P_q^(4,2l+2)(2x-1) C_{l m}(r_hat).
```

`P_q^(alpha,beta)` is a shifted Jacobi polynomial and `C_lm` is a
Racah-normalized Condon--Shortley spherical harmonic.  Same-neighbor products
use the existing YE3T generalized Clebsch--Gordan convention.  Their angular,
radial, species, support, and parity records are exact and hash-bound to the
model.  The particular fixture uses Ta, `q=0`, `l=1`, and `r_c=4.8 A`.
Source-plan V2 evaluates the Jacobi polynomial and its derivative with the
stable differentiated three-term recurrence shared by Python, native CPU, and
Kokkos.  The exact expanded-power coefficients remain provenance-only.
Source-plan V1 Horner evaluation remains readable for legacy CPU replay but is
deliberately rejected by the V3 Kokkos path.

The tagged sums are lowered to a common commutative moment algebra.  Ordered
distinct tag tuples use the exact set-partition/Moebius identity; this does not
claim a fully self-interaction-free residual density.  Full ACE purification
is deliberately outside this first tagged implementation.

All selected tag counts are reduced together at fixed `N`.  Exact
Gram--Schmidt in the realized free-moment algebra removes an explicit `s=1`
duplicate and produces two orthonormal image coordinates from three raw rows.
This is not a numerical pivot basis and does not use a sampled design matrix.
Orthogonality is with respect to the specified symmetric-Fock compiler metric,
not the empirical distribution of atomic environments.

YE3T compiles and hash-commits the exact real moment schedule, real-form maps,
and division-free transpose adjoint.  `ye3t-methods` binds that compiler-owned
schedule to the source and fitted readout and exports it without re-deriving
the coefficients.  LAMMPS reads only the serialized binary64 schedule and
source coefficients: it performs no symbolic compilation, Gram construction,
matrix inversion, or recoupling at runtime.  The native loader cross-checks
compiler, source, schedule, readout, and convention identities before
accepting the model.

The V3 loader rejects an exact zero separation before evaluating a direction.
The current native spherical-harmonic kernel clamps its unit-vector denominator
to `1e-12 A`, so literal Python/native source-and-derivative parity is validated
here for `r>1e-12 A`; all distances in this fixture are much larger.  An exact
regular-solid-harmonic continuation below that bound remains a production
origin-semantics task, not a claim of this software fixture.

This certificate is bounded to the stated `N=4`, `s<=2`, homogeneous `l=1`
slice.  General tag placement carriers, mixed contents, arbitrary `s`, fitted
Ta models, and multi-element V3 execution are later gates.  A Kokkos V3 direct
path is available only as a correctness reference: it is explicitly logged as
`experimental_reference_unqualified`, retains fixed component/factor limits,
and has no performance, AUTO, block, or multi-GPU scaling claim.

References:

- A. P. Yutsis, I. B. Levinson, and V. V. Vanagas, *Mathematical Apparatus
  of the Theory of Angular Momentum* (1962), for generalized angular-momentum
  coupling trees and products of Clebsch--Gordan coefficients.
- NIST Digital Library of Mathematical Functions, Sections 18.1 and 18.3,
  for Jacobi polynomials and their orthogonality conventions, and Section
  34.3(vii), for spherical-harmonic products and Wigner 3j symbols.
- The full project derivation and scope statement are in the `ye3t-methods`
  package.
