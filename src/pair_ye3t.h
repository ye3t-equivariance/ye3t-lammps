/* -*- c++ -*- ----------------------------------------------------------
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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(ye3t,PairYE3T);
// clang-format on
#else

#ifndef LMP_PAIR_YE3T_H
#define LMP_PAIR_YE3T_H

#include "pair.h"

#include "ye3t_lifted_cauchy_cpu.h"
#include "ye3t_tagged_cauchy_cpu.h"
#include "ye3t_yace_model.h"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace YE3T_LAMMPS {
class YE3TCPUEvaluator;
}

namespace LAMMPS_NS {

class PairYE3T : public Pair {
 public:
  explicit PairYE3T(LAMMPS *);
  ~PairYE3T() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void *extract(const char *, int &) override;
  double memory_usage() override;

 protected:
  virtual void initialize_backend();
  virtual void initialize_lifted_backend();
  // Called from coeff() after the tagged model, evaluator, and species map
  // are built, wrapped in the same try/catch as the lifted backend hook.
  // The CPU base only logs (tagged_evaluator_ is already built by the time
  // this runs); PairYE3TKokkos overrides it to also upload the device plan.
  virtual void initialize_tagged_backend();
  virtual YE3T_LAMMPS::YACEBlockPolicy model_load_policy() const;
  virtual bool supports_lifted_cauchy() const;
  // The CPU base always supports tagged_cauchy (defined in pair_ye3t.cpp);
  // PairYE3TKokkos overrides this once its device path is available, in the
  // same way ye3t/kk declares its lifted_cauchy support.
  virtual bool supports_tagged_cauchy() const;

  const YE3T_LAMMPS::YACEModel &model() const { return *model_; }
  bool is_lifted_cauchy_model() const { return lifted_model_ != nullptr; }
  const YE3T_LAMMPS::LiftedCauchyModel &lifted_cauchy_model() const { return *lifted_model_; }
  bool is_tagged_cauchy_model() const { return tagged_model_ != nullptr; }
  const YE3T_LAMMPS::TaggedCauchyModel &tagged_cauchy_model() const { return *tagged_model_; }
  YE3T_LAMMPS::LiftedCauchySourcePolicy lifted_source_policy() const
  {
    return lifted_source_policy_;
  }
  bool model_loaded() const
  {
    return model_ != nullptr || lifted_model_ != nullptr || tagged_model_ != nullptr;
  }
  const std::string &source_model_hash() const { return source_model_hash_; }
  const std::string &logical_direct_plan_hash() const { return logical_direct_plan_hash_; }
  const std::string &semantic_selection_hash() const { return semantic_selection_hash_; }
  int chunksize() const { return chunksize_; }
  YE3T_LAMMPS::YACEBlockPolicy block_policy() const { return block_policy_; }
  const std::string &auto_replay_path() const { return auto_replay_path_; }
  void set_maximum_imaginary_density(double value) { maximum_imaginary_density_ = value; }

 private:
  template <typename Value> static void resize_growing(std::vector<Value> &values, std::size_t size)
  {
    if (values.capacity() < size) {
      const std::size_t grown = values.capacity() + values.capacity() / 2 + 64;
      values.reserve(grown > size ? grown : size);
    }
    values.resize(size);
  }

  void allocate();
  void evaluate_lifted_chunk(int, int);
  void evaluate_tagged_chunk(int, int);

  enum class ModelFamily { YACE, LIFTED_CAUCHY, TAGGED_CAUCHY };
  enum class LiftedSourceSelection { MODEL_DEFAULT, DIRECT_Q, FACTORIZED_T };

  std::unique_ptr<YE3T_LAMMPS::YACEModel> model_;
  std::unique_ptr<YE3T_LAMMPS::YE3TCPUEvaluator> evaluator_;
  std::unique_ptr<YE3T_LAMMPS::LiftedCauchyModel> lifted_model_;
  std::unique_ptr<YE3T_LAMMPS::LiftedCauchyCPUSource> lifted_source_;
  std::unique_ptr<YE3T_LAMMPS::LiftedCauchyCPULoweredReadout> lifted_readout_;
  std::unique_ptr<YE3T_LAMMPS::TaggedCauchyModel> tagged_model_;
  std::unique_ptr<YE3T_LAMMPS::TaggedCauchyCPUEvaluator> tagged_evaluator_;
  std::vector<std::size_t> tagged_edge_offsets_;
  std::vector<int> center_atoms_;
  std::vector<int> central_species_;
  std::vector<int> edge_centers_;
  std::vector<int> edge_neighbors_;
  std::vector<int> edge_neighbor_species_;
  std::vector<double> edge_vectors_;
  std::vector<double> atomic_energies_;
  std::vector<double> edge_gradients_;
  std::vector<double> ordinary_atomic_energies_;
  std::vector<double> ordinary_edge_gradients_;
  std::vector<std::size_t> lifted_edge_offsets_;
  std::vector<double> lifted_source_values_;
  std::vector<double> lifted_source_adjoint_;
  std::vector<YE3T_LAMMPS::LiftedCauchyEdge> lifted_center_edges_;
  std::vector<double> lifted_center_source_;
  std::vector<std::array<double, 3>> lifted_center_edge_gradients_;
  int chunksize_ = 4096;
  std::string sidecar_manifest_;
  std::string auto_replay_path_;
  std::string source_model_hash_;
  std::string logical_direct_plan_hash_;
  std::string semantic_selection_hash_;
  YE3T_LAMMPS::YACEBlockPolicy block_policy_ = YE3T_LAMMPS::YACEBlockPolicy::DIRECT;
  ModelFamily model_family_ = ModelFamily::YACE;
  LiftedSourceSelection lifted_source_selection_ = LiftedSourceSelection::MODEL_DEFAULT;
  bool lifted_source_selection_was_set_ = false;
  YE3T_LAMMPS::LiftedCauchySourcePolicy lifted_source_policy_ =
      YE3T_LAMMPS::LiftedCauchySourcePolicy::DIRECT_Q;
  double maximum_imaginary_density_ = 0.0;
};

}    // namespace LAMMPS_NS

#endif
#endif
