.. index:: compute ye3t/property/atom

compute ye3t/property/atom command
==================================

Syntax
""""""

.. code-block:: LAMMPS

   compute ID group-ID ye3t/property/atom model.ye3t.json element1 element2 ...

Supply one element name or ``NULL`` for each LAMMPS atom type. The names
must occur in the model's saved species order. A ``NULL`` type is excluded
both as a center and as a neighbor.

Example
"""""""

.. code-block:: LAMMPS

   units metal
   pair_style zero 5.0
   pair_coeff * *
   compute p all ye3t/property/atom fitted_property.ye3t.json Ni Cu
   dump d all custom 1 property.dump id type c_p[1] c_p[2] c_p[3]

Description
"""""""""""

This CPU compute evaluates the fitted mean of a compiler-bound full-M
per-atom linear property exported by ``ye3t_methods.LinearModel``. It reads
``ye3t_methods_tagged_full_m_per_atom_v2`` and
``ye3t_methods_density_full_m_per_atom_v2`` artifacts with their native
property plans. The density path currently supports explicit delta chemistry,
identity PACE ChebExpCos radial channels, complex spherical harmonics with
``pace_y00_one`` normalization, and no source or charge normalization.
Models with target angular momentum L=1 or L=2 are supported. The output is
a real-tesseral per-atom array with ``2L+1``
columns in the model's saved axis order. The output units are those of the
training target.

The compute uses a full occasional neighbor list. Each periodic image is a
separate directed neighbor occurrence. Only owned atoms in ``group-ID``
are evaluated; their neighbors may be outside that group. Rows for owned
atoms outside the group or mapped to ``NULL`` are zero. The result does not
depend on the ``newton pair`` setting. The compute produces a mean property,
not forces, stress, posterior variance, or a pair potential.

The model cutoff plus the neighbor skin must fit within the ghost range.
Set a pair style with a sufficient cutoff or increase ``comm_modify cutoff``.
For example, ``pair_style zero`` permits property evaluation without a
YE3T pair potential. ``neighbor multi`` is not supported for this compute.
The model uses Angstrom geometry, so ``units metal`` is required.

For molecular topologies, LAMMPS can omit bonded pairs when
:doc:`special_bonds <special_bonds>` assigns a zero weight. Such omissions,
and explicit ``neigh_modify exclude`` rules, also affect this compute's
neighbor list. Keep the neighbor inclusion policy consistent with the
training environments; this compute does not restore excluded pairs.

The reader checks the artifact and embedded plan hashes and their internal
bindings. These hashes detect ordinary corruption; they are not a signature
from an external trusted authority. The Python exporter independently checks
the embedded schedule against its compiler source before writing it.

Output info
""""""""""

This compute calculates a per-atom array with three columns for L=1 or
five columns for L=2. Values are intensive per-atom quantities. Access a
column as ``c_ID[I]`` in per-atom output commands.

Restrictions
"""""""""""

The CPU compute is available in the ML-YE3T package or the ``ye3tplugin``
loadable module. An experimental ``ye3t/property/atom/kk/device`` variant is
present in Kokkos builds for tagged v2 artifacts. It rejects density v2
artifacts. Matched CPU/device per-atom rows passed 15 single-rank and two
two-rank tagged regression cases, including a rank-3 mixed Young block.
The largest component difference was ``2.85e-13``. The measured device
throughput depends strongly on atom count and MPI/GPU placement; retained
256/2048/16384-atom fixed-cell, moving-atom timings are in the P6 workflow study. Density
device execution remains unsupported.

The native reader currently accepts tagged and density ``L=1`` odd-parity
and ``L=2`` even-parity models. Python can write a tagged ``L=2`` odd-parity
artifact, but this native compute rejects it pending separate covariance
validation.
