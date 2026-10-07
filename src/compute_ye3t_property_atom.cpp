/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation. This software is distributed under
   the GNU General Public License. See the top-level LAMMPS README.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: James M. Goff (Sandia National Laboratories)
------------------------------------------------------------------------- */

#include "compute_ye3t_property_atom.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <string>

using namespace LAMMPS_NS;

namespace {

// LAMMPS renamed the fixed-cutoff request method after the 2025 stable ABI.
template <class Request>
auto set_property_cutoff(Request *request, double cutoff, int)
    -> decltype(request->set_cutoff_fixed(cutoff), void())
{
  request->set_cutoff_fixed(cutoff);
}

template <class Request>
auto set_property_cutoff(Request *request, double cutoff, long)
    -> decltype(request->set_cutoff(cutoff), void())
{
  request->set_cutoff(cutoff);
}

}  // namespace

ComputeYE3TPropertyAtom::ComputeYE3TPropertyAtom(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg)
{
  if (strcmp("metal", update->unit_style) != 0)
    error->all(FLERR, "Compute ye3t/property/atom requires units metal");
  if (narg != 4 + atom->ntypes)
    error->all(FLERR,
               "Compute ye3t/property/atom requires a model path and one element or NULL per atom type");
  try {
    evaluator_ = std::make_unique<YE3T_LAMMPS::YE3TMeanPropertyCPU>(arg[3]);
  } catch (const std::exception &exception) {
    error->all(FLERR, "Cannot load YE3T mean property model: {}", exception.what());
  }
  species_by_type_.assign(atom->ntypes + 1, -1);
  for (int type = 1; type <= atom->ntypes; ++type) {
    const std::string name = arg[3 + type];
    if (name == "NULL") continue;
    const auto &species = evaluator_->species_order();
    const auto found = std::find(species.begin(), species.end(), name);
    if (found == species.end())
      error->all(FLERR, "Compute ye3t/property/atom has an unknown mapped element: {}", name);
    species_by_type_[type] = static_cast<int>(found - species.begin());
  }
  peratom_flag = 1;
  size_peratom_cols = evaluator_->width();
}

ComputeYE3TPropertyAtom::~ComputeYE3TPropertyAtom()
{
  memory->destroy(property_);
}

void ComputeYE3TPropertyAtom::init()
{
  const double cutoff = evaluator_->cutoff();
  const double request_cutoff = cutoff + neighbor->skin;
  const double ghost_cutoff = force->pair
      ? std::max(force->pair->cutforce + neighbor->skin, comm->cutghostuser)
      : comm->cutghostuser;
  if (request_cutoff > ghost_cutoff + 1e-10)
    error->all(FLERR,
               "Compute ye3t/property/atom cutoff plus skin exceeds ghost range; increase pair cutoff or comm_modify cutoff");
  if (neighbor->style != Neighbor::BIN && neighbor->style != Neighbor::NSQ)
    error->all(FLERR,
               "Compute ye3t/property/atom with an explicit cutoff requires neighbor style bin or nsq");
  auto *request = neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);
  set_property_cutoff(request, request_cutoff, 0);
}

void ComputeYE3TPropertyAtom::init_list(int, NeighList *list)
{
  list_ = list;
}

void ComputeYE3TPropertyAtom::compute_peratom()
{
  invoked_peratom = update->ntimestep;
  if (atom->nmax > nmax_) {
    memory->destroy(property_);
    nmax_ = atom->nmax;
    memory->create(property_, nmax_, size_peratom_cols, "ye3t/property/atom:values");
    array_atom = property_;
  }
  if (!list_) error->all(FLERR, "Compute ye3t/property/atom has no neighbor list");
  neighbor->build_one(list_);
  for (int i = 0; i < atom->nlocal; ++i)
    std::fill(property_[i], property_[i] + size_peratom_cols, 0.0);

  centers_.clear();
  central_species_.clear();
  edge_offsets_.clear();
  neighbor_species_.clear();
  edge_vectors_.clear();
  edge_offsets_.push_back(0);
  const double cutoff_sq = evaluator_->cutoff() * evaluator_->cutoff();
  double **x = atom->x;
  for (int ii = 0; ii < list_->inum; ++ii) {
    const int i = list_->ilist[ii];
    if (i >= atom->nlocal || !(atom->mask[i] & groupbit)) continue;
    const int central = species_by_type_[atom->type[i]];
    if (central < 0) continue;
    centers_.push_back(i);
    central_species_.push_back(central);
    int *neighbors = list_->firstneigh[i];
    for (int jj = 0; jj < list_->numneigh[i]; ++jj) {
      const int j = neighbors[jj] & NEIGHMASK;
      const int neighbor = species_by_type_[atom->type[j]];
      if (neighbor < 0) continue;
      const double dx = x[j][0] - x[i][0];
      const double dy = x[j][1] - x[i][1];
      const double dz = x[j][2] - x[i][2];
      if (dx * dx + dy * dy + dz * dz >= cutoff_sq) continue;
      neighbor_species_.push_back(neighbor);
      edge_vectors_.push_back(dx);
      edge_vectors_.push_back(dy);
      edge_vectors_.push_back(dz);
    }
    edge_offsets_.push_back(neighbor_species_.size());
  }
  if (centers_.empty()) return;
  values_.resize(centers_.size() * size_peratom_cols);
  try {
    evaluator_->evaluate(static_cast<int>(centers_.size()), central_species_.data(),
                         edge_offsets_.data(), neighbor_species_.data(), edge_vectors_.data(),
                         nullptr, values_.data());
  } catch (const std::exception &exception) {
    error->all(FLERR, "YE3T mean property evaluation failed: {}", exception.what());
  }
  for (std::size_t row = 0; row < centers_.size(); ++row)
    std::copy_n(values_.data() + row * size_peratom_cols, size_peratom_cols,
                property_[centers_[row]]);
}

double ComputeYE3TPropertyAtom::memory_usage()
{
  return static_cast<double>(nmax_) * size_peratom_cols * sizeof(double) +
         static_cast<double>(centers_.capacity() + central_species_.capacity() +
                             neighbor_species_.capacity()) * sizeof(int) +
         static_cast<double>(edge_offsets_.capacity()) * sizeof(std::size_t) +
         static_cast<double>(edge_vectors_.capacity() + values_.capacity()) * sizeof(double);
}
