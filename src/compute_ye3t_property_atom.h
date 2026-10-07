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

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(ye3t/property/atom,ComputeYE3TPropertyAtom);
// clang-format on
#else

#ifndef LMP_COMPUTE_YE3T_PROPERTY_ATOM_H
#define LMP_COMPUTE_YE3T_PROPERTY_ATOM_H

#include "compute.h"
#include "ye3t_mean_property_cpu.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace LAMMPS_NS {

class ComputeYE3TPropertyAtom : public Compute {
 public:
  ComputeYE3TPropertyAtom(class LAMMPS *, int, char **);
  ~ComputeYE3TPropertyAtom() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  void compute_peratom() override;
  double memory_usage() override;

 protected:
  const YE3T_LAMMPS::YE3TMeanPropertyCPU &property_model() const { return *evaluator_; }
  const std::vector<int> &property_type_map() const { return species_by_type_; }
  class NeighList *property_neighbor_list() const { return list_; }

 private:
  std::unique_ptr<YE3T_LAMMPS::YE3TMeanPropertyCPU> evaluator_;
  class NeighList *list_ = nullptr;
  double **property_ = nullptr;
  int nmax_ = 0;
  std::vector<int> species_by_type_;
  std::vector<int> centers_;
  std::vector<int> central_species_;
  std::vector<std::size_t> edge_offsets_;
  std::vector<int> neighbor_species_;
  std::vector<double> edge_vectors_;
  std::vector<double> values_;
};

}  // namespace LAMMPS_NS

#endif
#endif
