# Tagged-Cauchy V3 CPU/MPI/Kokkos quick example

This directory is a self-contained correctness example for the first
hash-bound tagged-Cauchy physical-image model format.  The bundled model is a
two-feature, tensor-order-4 (`N=4`) Ta software fixture with tag counts
`s=0,1,2`.  It was produced by a tiny deterministic ridge solve so that the
compiler, Python runtime, and native LAMMPS implementation can be compared.

It is **not** a physically fitted Ta potential and must not be used for model
accuracy, molecular-dynamics stability, or timing claims.  `THEORY.md`
describes the exact bounded construction and its current scope.

Run the static 54-atom periodic BCC check from the `tests/fixtures` directory
of this repository (the directory is not installed into the LAMMPS examples
tree):

```bash
cd tests/fixtures
/absolute/path/to/lmp -in tagged_cauchy_v3_quick/in.tagged_cauchy_v3_quick
mpiexec -n 2 /absolute/path/to/lmp \
  -in tagged_cauchy_v3_quick/in.tagged_cauchy_v3_quick
```

The second command exercises domain decomposition and ghost atoms.  The
dedicated repository validation additionally moves an atom across the rank
boundary and compares both ranks with the Python oracle.

A CUDA/Kokkos LAMMPS build runs the byte-identical model with:

```bash
/absolute/path/to/lmp -k on g 1 -pk kokkos neigh half -sf kk \
  -in tagged_cauchy_v3_quick/in.tagged_cauchy_v3_quick
```

The log must report `evaluator physical_image_v3_direct` and qualification
`experimental_reference_unqualified`. This is an unoptimized direct
correctness path, not timing or performance evidence. It intentionally rejects
legacy source-plan V1 models and currently retains fixed `l<=8`, 128-component,
and 32-factors-per-term limits.

For a force and strain-virial finite-difference check, build LAMMPS with
`PKG_EXTRA-FIX=ON` and run:

```bash
/absolute/path/to/lmp \
  -in tagged_cauchy_v3_quick/in.tagged_cauchy_v3_numdiff
```

The same finite-difference input runs on Kokkos by adding
`-k on g 1 -pk kokkos neigh half -sf kk` before `-in`.

The input prints force error in eV/A and the virial error both in LAMMPS metal
pressure units (bar) and, after multiplication by volume, in eV.  Numerical
differentiation is a correctness check and should not be included in timing.

The model generator is `tests/generate_tagged_cauchy_v3_fixture.py`.
Regeneration requires installed `ye3t` and `ye3t-ace` (not yet public);
running this LAMMPS example does not. The two JSON files here are
byte-identical copies of `tests/fixtures/tagged_cauchy_physical_image_v3.json`
and `tests/fixtures/tagged_cauchy_physical_image_v3.fixture.json`. The
repository validation for this fixture is
`tests/validate_tagged_cauchy_v3_lammps.py` (CPU/MPI parity against the
Python oracle) and `tests/test_kokkos_tagged_cauchy_cuda.sh` (Kokkos parity,
per-atom/global virial, and optional numerical-difference checks); the public
`examples/PACKAGES/ye3t/run_examples.sh` runner does not execute it.

Using multiple MPI ranks on one GPU is explicitly oversubscribed correctness
testing. Real multi-GPU scaling remains a separate qualification gate.
