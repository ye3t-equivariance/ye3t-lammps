/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: James M. Goff (Sandia National Laboratories)
------------------------------------------------------------------------- */

#include "lammpsplugin.h"
#include "version.h"

#include "pair_ye3t.h"
#include "compute_ye3t_property_atom.h"

using namespace LAMMPS_NS;

namespace {

Pair *create_pair_ye3t(LAMMPS *lmp)
{
  return new PairYE3T(lmp);
}

Compute *create_compute_ye3t_property_atom(LAMMPS *lmp, int narg, char **arg)
{
  return new ComputeYE3TPropertyAtom(lmp, narg, arg);
}

}    // namespace

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *registration)
{
  lammpsplugin_t plugin;
  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "ye3t";
  plugin.info = "YE3T linear ACE CPU pair style";
  plugin.author = "James M. Goff (Sandia National Laboratories)";
  plugin.creator.v1 = reinterpret_cast<lammpsplugin_factory1 *>(&create_pair_ye3t);
  plugin.handle = handle;
  auto register_plugin = reinterpret_cast<lammpsplugin_regfunc>(registration);
  register_plugin(&plugin, lmp);

  plugin.style = "compute";
  plugin.name = "ye3t/property/atom";
  plugin.info = "YE3T compiler-bound per-atom equivariant mean property";
  plugin.creator.v2 = reinterpret_cast<lammpsplugin_factory2 *>(&create_compute_ye3t_property_atom);
  register_plugin(&plugin, lmp);
}
