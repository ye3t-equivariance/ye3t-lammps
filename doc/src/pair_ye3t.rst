.. index:: pair_style ye3t
.. index:: pair_style ye3t/kk

pair_style ye3t command
=======================

Accelerator Variants: *ye3t/kk*

Syntax
""""""

.. code-block:: LAMMPS

   pair_style ye3t keyword value ...

* zero or more keyword/value pairs may be appended

  .. parsed-literal::

       keyword = *model_family* or *plan* or *block_policy* or *auto_replay* or *chunksize* or *source_realization*
         *model_family* value = *yace* or *tagged_cauchy* or *lifted_cauchy*
           yace = linear ACE potential in the standard .yace file format (default)
           tagged_cauchy = native YE3T tagged-Cauchy bundle or composite manifest
           lifted_cauchy = native YE3T lifted-Cauchy bundle
         *plan* value = file
           file = compiled execution-plan manifest for a *yace* model
         *block_policy* value = *direct* or *block* or *scalar_power* or *coupled_product* or *auto*
           direct = evaluate every basis function through the exact product DAG
           block = force the compiler-validated block/symmetric-power schedule
           scalar_power = force the compiler-validated homogeneous scalar-power schedule
           coupled_product = force the compiler-validated coupled-product schedule
           auto = choose one exact schedule for the active backend at pair_coeff time
         *auto_replay* value = file
           file = calibrated GPU AUTO replay record (*ye3t/kk* with *block_policy auto* only)
         *chunksize* value = N
           N = maximum number of atoms evaluated in one pass over the neighbor list
         *source_realization* value = *model_default* or *direct* or *factorized*
           (lifted_cauchy models only)

Examples
""""""""

.. code-block:: LAMMPS

   pair_style ye3t
   pair_coeff * * Ta.yace Ta

   pair_style ye3t plan manifest.json block_policy auto
   pair_coeff * * Ta.yace Ta

   pair_style hybrid/overlay ye3t model_family tagged_cauchy block_policy direct zbl 1.28 1.65
   pair_coeff * * ye3t model.ye3t.json Cu
   pair_coeff * * zbl 29 29

   # V4 model with artifact-bound atomic references and pair-specific ZBL:
   pair_style ye3t model_family tagged_cauchy block_policy direct
   pair_coeff * * model.ye3t.json H O K S

**Complete NVE input.** The source package includes a fitted Ta model and
the matching execution plan in
``examples/PACKAGES/ye3t/models/ta_l8_compact``. From
``examples/PACKAGES/ye3t`` in the patched LAMMPS tree, run
``lmp -in in.ye3t.md``. The complete input is:

.. code-block:: LAMMPS

   variable model index models/ta_l8_compact/model.yace
   variable plan index models/ta_l8_compact/manifest.json
   variable policy index auto

   units metal
   atom_style atomic
   boundary p p p
   atom_modify map yes sort 0 0.0
   newton on

   lattice bcc 3.3161146998079496
   region cell block 0 4 0 4 0 4 units lattice
   create_box 1 cell
   create_atoms 1 box
   mass 1 180.94788
   reset_atoms id sort yes
   displace_atoms all random 0.01 0.01 0.01 77123 units box
   velocity all create 300.0 89231 mom yes rot no dist gaussian

   neighbor 0.3 bin
   neigh_modify every 1 delay 0 check yes

   pair_style ye3t plan ${plan} block_policy ${policy} chunksize 256
   pair_coeff * * ${model} Ta

   timestep 0.001
   fix integrate all nve
   thermo 1
   thermo_style custom step atoms temp pe ke etotal press
   thermo_modify format float %.17g
   run 20

The output lists potential, kinetic, and total energy at every step. Change
the model, element type map, lattice, temperature, and run length for a new
system. Export a fitted potential with ``ye3t-methods`` before using a
different model file; ``pair_style ye3t`` performs inference in LAMMPS.

**From an ASE fit to LAMMPS.** For a supported scalar tagged model already
fitted and saved by ``ye3t-methods``, export its native bundle with the public
model object:

.. code-block:: python

   from ye3t_methods import LinearModel

   model = LinearModel.read("fitted_tagged.ye3t.json")
   deployed = model.export_lammps("deployed.ye3t.json")
   print(deployed)

Use the printed path in ``pair_coeff`` and list one model element for each
LAMMPS atom type, in atom-type order:

.. code-block:: LAMMPS

   pair_style ye3t model_family tagged_cauchy block_policy direct
   pair_coeff * * deployed.ye3t.json Ta

The ``ye3t-methods`` ASE fitting and export examples show how to create the
saved model. Ordinary scalar density models that meet the PACE export
requirements produce a ``.yace`` file and use the first syntax example above.

Description
"""""""""""

Pair style *ye3t* computes interactions with linear Young-E(3)-tensor
(YE3T) potentials. A YE3T potential is a linear model whose basis
functions are products of atomic-base functions coupled through exact
permutation and rotation symmetry-adapted coefficients. The ordinary
symmetric sector of that basis is the linear Atomic Cluster Expansion
(ACE), so a standard ``.yace`` potential file written for
:doc:`pair_style pace <pair_pace>` is also a valid *yace* model for this
pair style and evaluates to the same energies, forces, and virials.
Beyond that sector, YE3T models can carry basis functions with
nontrivial permutation intermediates (tagged-Cauchy and lifted-Cauchy
families) that have no ``.yace`` representation.

Models are trained and exported outside LAMMPS with the ``ye3t`` and
``ye3t-methods`` packages. LAMMPS performs inference only: it reads the
exported potential, validates every hash-bound sidecar once during
:doc:`pair_coeff <pair_coeff>`, and then evaluates the model without any
further file parsing or hashing during the run.

The *model_family* keyword selects the potential format. For *yace* the
single argument of :doc:`pair_coeff <pair_coeff>` is the ``.yace`` file.
For *tagged_cauchy* it is either a native ``model.ye3t.json`` bundle or a
composite manifest that binds an ordinary ``.yace`` backbone and a tagged
correction file by SHA-256; both components are evaluated inside one
instance of this pair style. For *lifted_cauchy* it is the
``model.ye3t.json`` file of a lifted bundle. After the model file, list
the element name for each LAMMPS atom type in order, exactly as for
:doc:`pair_style pace <pair_pace>`. Every species named by the model
must be mapped by at least one atom type.

Tagged V4 bundles support directed species-pair descriptor cutoffs and bound
fixed atomic references plus pair-specific ZBL switches on the CPU. The same
compiled polynomial and explicit adjoint used by Python training are consumed
by the native evaluator. If ``readout_binding.payload.reference_terms``
contains ZBL, it is included in energies, forces and virials: do not add an
external ZBL overlay. Residual-only bundles without bound references use a
separate ZBL overlay. V4 bundles are rejected by ``ye3t/kk``;
run them with the CPU pair style.

The *plan* keyword names a compiled execution-plan manifest for a *yace*
model. The plan is produced by the ``ye3t`` compiler together with the
``.yace`` export, records the hash of the potential it belongs to, and
describes exact alternative schedules (block, symmetric-power, scalar-power,
and coupled-product routes) for evaluating the same basis functions. A plan
changes only how the potential is evaluated, never the potential itself, and
a plan that does not match the potential hash is rejected.

The *block_policy* keyword selects the schedule. *direct* is the reference
route that mirrors the product evaluator of :doc:`pair_style pace <pair_pace>`
and needs no plan. The forced policies *block*, *scalar_power*, and
*coupled_product* require a plan that validates the requested route and
fail otherwise. *auto* scores the candidates in the plan for the active
backend during :doc:`pair_coeff <pair_coeff>` and freezes one exact
schedule before the run; the selection, its reason, and the plan identity
are printed to the log. On the CPU the candidates are timed on a bounded
synthetic workload. On the GPU, *auto* selects the direct route unless an
*auto_replay* record calibrated on the same GPU class, executable, model,
and plan is supplied, in which case the replayed selection is validated and
used. Tagged-Cauchy models read their candidate portfolio from the bundle
itself and therefore accept *block_policy* but not *plan* or
*auto_replay*.

The *chunksize* keyword bounds the number of atoms evaluated in one pass.
The default is 4096; the evaluators process fewer atoms per pass when the
model workspace or, for *ye3t/kk*, the available device memory requires it.
The explicit value is an expert override and is recorded in AUTO replay
records.

At :doc:`pair_coeff <pair_coeff>` time the pair style prints the model
identity, the selected schedule, and (for *ye3t/kk*) the device plan probe
results. Any mismatch between the potential, a plan, a composite
component, or a replay record raises an error.

----------

Mixing, shift, table, tail correction, restart, rRESPA info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

For atom type pairs I,J and I != J, where types I and J correspond to two
different element types, mixing is performed by LAMMPS with user-specifiable
parameters as described above. You never need to specify a pair_coeff
command with I != J arguments for this style.

This pair style does not support the :doc:`pair_modify <pair_modify>`
shift, table, and tail options.

This pair style does not write its information to :doc:`binary restart
files <restart>`, since it is stored in potential files. Thus, you need to
re-specify the pair_style and pair_coeff commands in an input script that
reads a restart file.

This pair style can only be used via the *pair* keyword of the
:doc:`run_style respa <run_style>` command. It does not support the
*inner*, *middle*, *outer* keywords.

----------

.. include:: accel_styles.rst

----------

Restrictions
""""""""""""

This pair style is part of the ML-YE3T package. It is only enabled if
LAMMPS was built with that package. See the :doc:`Build package
<Build_package>` page for more info.

This pair style requires :doc:`metal units <units>` and
:doc:`newton pair on <newton>`.

The *ye3t/kk* variant requires a Kokkos build with double precision and a
half neighbor list (``-pk kokkos neigh half``). It executes on the device
backend only; there is no *ye3t/kk/host* variant. CUDA is the validated
device backend. Composite tagged-Cauchy manifests are evaluated on the CPU
only. For native tagged-Cauchy bundles, *ye3t/kk* accepts only
*block_policy direct* and requires angular degrees l <= 8, at most 128 basis
components, and 32 factors per term; bundles older than the physical-image
V3 format are additionally limited to 16 radial functions. Lifted-Cauchy
models accept only *block_policy direct*.

Related commands
""""""""""""""""

:doc:`pair_coeff <pair_coeff>`, :doc:`pair_style pace <pair_pace>`,
:doc:`pair_style hybrid/overlay <pair_hybrid>`, :doc:`pair_style zbl <pair_zbl>`

Default
"""""""

The keyword defaults are model_family = yace, block_policy = direct,
chunksize = 4096, source_realization = model_default, and no plan or
auto_replay file.
