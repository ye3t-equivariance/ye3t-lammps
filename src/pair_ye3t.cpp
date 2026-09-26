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

#include "pair_ye3t.h"

#include "ye3t_cpu_evaluator.h"
#include "ye3t_sha256.h"
#include "ye3t_yace_model.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "info.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"
#include "utils.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>

using namespace LAMMPS_NS;

namespace {
constexpr std::size_t LIFTED_SOURCE_BATCH_BYTES = 32u * 1024u * 1024u;
// Keep the route microbenchmark long enough to rise above sub-millisecond
// scheduler noise while remaining negligible relative to an MD run.  The
// median over seven interleaved repeats is catalogue- and machine-specific.
constexpr int TAGGED_AUTO_CENTERS = 128;
constexpr int TAGGED_AUTO_NEIGHBORS = 16;
constexpr int TAGGED_AUTO_REPEATS = 11;
constexpr double TAGGED_AUTO_SWITCH_RATIO = 0.92;

YE3T_LAMMPS::TaggedCauchyExecutionPolicy
tagged_execution_policy(YE3T_LAMMPS::YACEBlockPolicy policy)
{
  switch (policy) {
    case YE3T_LAMMPS::YACEBlockPolicy::DIRECT:
      return YE3T_LAMMPS::TaggedCauchyExecutionPolicy::COMPILED_DIRECT;
    case YE3T_LAMMPS::YACEBlockPolicy::AUTO:
      return YE3T_LAMMPS::TaggedCauchyExecutionPolicy::AUTO;
    case YE3T_LAMMPS::YACEBlockPolicy::BLOCK:
      return YE3T_LAMMPS::TaggedCauchyExecutionPolicy::BLOCK;
    case YE3T_LAMMPS::YACEBlockPolicy::SCALAR_POWER:
      return YE3T_LAMMPS::TaggedCauchyExecutionPolicy::SYMMETRIC_POWER;
    case YE3T_LAMMPS::YACEBlockPolicy::COUPLED_PRODUCT:
      return YE3T_LAMMPS::TaggedCauchyExecutionPolicy::GENERIC_DAG;
    default:
      throw std::invalid_argument("unsupported tagged-Cauchy execution policy");
  }
}
}    // namespace

PairYE3T::PairYE3T(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
}

PairYE3T::~PairYE3T()
{
  if (copymode) return;
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

void PairYE3T::allocate()
{
  allocated = 1;
  const int count = atom->ntypes + 1;
  memory->create(setflag, count, count, "pair:ye3t:setflag");
  memory->create(cutsq, count, count, "pair:ye3t:cutsq");
  map = new int[count];
}

void PairYE3T::settings(int narg, char **arg)
{
  if (strcmp("metal", update->unit_style) != 0)
    error->all(FLERR, "YE3T ACE potentials require 'metal' units");
  int index = 0;
  while (index < narg) {
    if (strcmp(arg[index], "chunksize") == 0) {
      if (index + 1 >= narg) utils::missing_cmd_args(FLERR, "pair_style ye3t chunksize", error);
      chunksize_ = utils::inumeric(FLERR, arg[index + 1], false, lmp);
      if (chunksize_ <= 0) error->all(FLERR, "Pair style ye3t chunksize must be positive");
      index += 2;
    } else if (strcmp(arg[index], "plan") == 0) {
      if (index + 1 >= narg) utils::missing_cmd_args(FLERR, "pair_style ye3t plan", error);
      sidecar_manifest_ = arg[index + 1];
      index += 2;
    } else if (strcmp(arg[index], "model_family") == 0) {
      if (index + 1 >= narg) utils::missing_cmd_args(FLERR, "pair_style ye3t model_family", error);
      if (strcmp(arg[index + 1], "yace") == 0)
        model_family_ = ModelFamily::YACE;
      else if (strcmp(arg[index + 1], "lifted_cauchy") == 0)
        model_family_ = ModelFamily::LIFTED_CAUCHY;
      else if (strcmp(arg[index + 1], "tagged_cauchy") == 0)
        model_family_ = ModelFamily::TAGGED_CAUCHY;
      else
        error->all(FLERR,
                   "Pair style ye3t model_family must be yace, "
                   "lifted_cauchy, or tagged_cauchy");
      index += 2;
    } else if (strcmp(arg[index], "source_realization") == 0) {
      if (index + 1 >= narg)
        utils::missing_cmd_args(FLERR, "pair_style ye3t source_realization", error);
      if (strcmp(arg[index + 1], "model_default") == 0)
        lifted_source_selection_ = LiftedSourceSelection::MODEL_DEFAULT;
      else if (strcmp(arg[index + 1], "direct") == 0)
        lifted_source_selection_ = LiftedSourceSelection::DIRECT_Q;
      else if (strcmp(arg[index + 1], "factorized") == 0)
        lifted_source_selection_ = LiftedSourceSelection::FACTORIZED_T;
      else
        error->all(FLERR,
                   "Pair style ye3t source_realization must be model_default, "
                   "direct, or factorized");
      lifted_source_selection_was_set_ = true;
      index += 2;
    } else if (strcmp(arg[index], "block_policy") == 0) {
      if (index + 1 >= narg) utils::missing_cmd_args(FLERR, "pair_style ye3t block_policy", error);
      if (strcmp(arg[index + 1], "direct") == 0)
        block_policy_ = YE3T_LAMMPS::YACEBlockPolicy::DIRECT;
      else if (strcmp(arg[index + 1], "auto") == 0)
        block_policy_ = YE3T_LAMMPS::YACEBlockPolicy::AUTO;
      else if (strcmp(arg[index + 1], "block") == 0)
        block_policy_ = YE3T_LAMMPS::YACEBlockPolicy::BLOCK;
      else if (strcmp(arg[index + 1], "scalar_power") == 0)
        block_policy_ = YE3T_LAMMPS::YACEBlockPolicy::SCALAR_POWER;
      else if (strcmp(arg[index + 1], "coupled_product") == 0)
        block_policy_ = YE3T_LAMMPS::YACEBlockPolicy::COUPLED_PRODUCT;
      else
        error->all(FLERR,
                   "Pair style ye3t block_policy must be direct, auto, block, "
                   "scalar_power, or coupled_product");
      index += 2;
    } else if (strcmp(arg[index], "auto_replay") == 0) {
      if (index + 1 >= narg) utils::missing_cmd_args(FLERR, "pair_style ye3t auto_replay", error);
      auto_replay_path_ = arg[index + 1];
      index += 2;
    } else {
      error->all(FLERR, "Unknown pair_style ye3t setting: {}", arg[index]);
    }
  }
  if (!auto_replay_path_.empty() && block_policy_ != YE3T_LAMMPS::YACEBlockPolicy::AUTO)
    error->all(FLERR, "Pair style ye3t auto_replay requires block_policy auto");
  if (model_family_ == ModelFamily::LIFTED_CAUCHY) {
    if (!sidecar_manifest_.empty() || !auto_replay_path_.empty() ||
        block_policy_ != YE3T_LAMMPS::YACEBlockPolicy::DIRECT)
      error->all(FLERR,
                 "Pair style ye3t lifted_cauchy currently "
                 "requires direct block_policy and does not accept plan or "
                 "auto_replay");
  } else if (model_family_ == ModelFamily::TAGGED_CAUCHY) {
    if (!sidecar_manifest_.empty() || !auto_replay_path_.empty())
      error->all(FLERR,
                 "Pair style ye3t tagged_cauchy reads compiler portfolios "
                 "from the model bundle and does not accept plan or "
                 "auto_replay");
    if (lifted_source_selection_was_set_)
      error->all(FLERR,
                 "Pair style ye3t source_realization applies only to "
                 "model_family lifted_cauchy");
  } else if (lifted_source_selection_was_set_) {
    error->all(FLERR,
               "Pair style ye3t source_realization applies only to "
               "model_family lifted_cauchy");
  }
  const char *block_policy = "scalar_power";
  if (block_policy_ == YE3T_LAMMPS::YACEBlockPolicy::DIRECT)
    block_policy = "direct";
  else if (block_policy_ == YE3T_LAMMPS::YACEBlockPolicy::AUTO)
    block_policy = "auto";
  else if (block_policy_ == YE3T_LAMMPS::YACEBlockPolicy::BLOCK)
    block_policy = "block";
  else if (block_policy_ == YE3T_LAMMPS::YACEBlockPolicy::COUPLED_PRODUCT)
    block_policy = "coupled_product";
  const char *model_family = "yace";
  if (model_family_ == ModelFamily::LIFTED_CAUCHY)
    model_family = "lifted_cauchy";
  else if (model_family_ == ModelFamily::TAGGED_CAUCHY)
    model_family = "tagged_cauchy";
  const char *source_realization = "model_default";
  if (lifted_source_selection_ == LiftedSourceSelection::DIRECT_Q)
    source_realization = "direct";
  else if (lifted_source_selection_ == LiftedSourceSelection::FACTORIZED_T)
    source_realization = "factorized";
  if (comm->me == 0)
    utils::logmesg(lmp,
                   "YE3T configuration: model_family {}, source_realization {}, "
                   "chunksize {}, block_policy {}, plan {}, auto_replay {}\n",
                   model_family, source_realization, chunksize_, block_policy,
                   sidecar_manifest_.empty() ? "none" : sidecar_manifest_,
                   auto_replay_path_.empty() ? "none" : auto_replay_path_);
}

void PairYE3T::initialize_backend()
{
  evaluator_ = std::make_unique<YE3T_LAMMPS::YE3TCPUEvaluator>(model_.get());
  if (comm->me == 0)
    utils::logmesg(lmp,
                   "YE3T CPU dispatch: PACE-compatible spline/half-complex source with "
                   "split-real harmonics, exact signed-m expansion/collapse, tiled "
                   "split-real symmetric-power forward/adjoint, source_model {}, "
                   "direct_logical_plan {}\n",
                   source_model_hash_, logical_direct_plan_hash_);
  if (comm->me == 0)
    utils::logmesg(lmp,
                   "YE3T CPU source schedule: complete environments in 8/16-center batches, "
                   "soft cache target {} bytes ({} edges), no source-table replay; "
                   "chunksize is an outer packing limit, not a memory limit\n",
                   evaluator_->source_tile_target_bytes(), evaluator_->source_tile_edge_capacity());
}

void PairYE3T::initialize_lifted_backend()
{
  if (comm->me == 0)
    utils::logmesg(lmp, "YE3T lifted CPU dispatch: source {}, native_plan {}, deployment {}\n",
                   lifted_source_policy_ == YE3T_LAMMPS::LiftedCauchySourcePolicy::DIRECT_Q
                       ? "direct_q"
                       : "factorized_t",
                   lifted_model_->native_self_hash, lifted_model_->deployment_identity_hash);
}

void PairYE3T::initialize_tagged_backend()
{
  if (tagged_evaluator_->auto_requested() && !tagged_evaluator_->auto_calibrated()) {
    std::array<double, 4> candidate_seconds = {
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
    };
    double calibration_seconds = 0.0;
    int selected_candidate = 0;
    if (comm->me == 0) {
      const auto begin = std::chrono::steady_clock::now();
      candidate_seconds = tagged_evaluator_->calibrate_auto(
          TAGGED_AUTO_CENTERS, TAGGED_AUTO_NEIGHBORS, TAGGED_AUTO_REPEATS);
      const auto end = std::chrono::steady_clock::now();
      calibration_seconds = std::chrono::duration<double>(end - begin).count();
      int fastest_candidate = 0;
      for (int candidate = 1; candidate < 4; ++candidate)
        if (candidate_seconds[static_cast<std::size_t>(candidate)] <
            candidate_seconds[static_cast<std::size_t>(fastest_candidate)])
          fastest_candidate = candidate;
      // Keep AUTO on direct when differences are within initialization noise
      // or too small to meet the project's runtime-retention threshold.
      // Non-direct routes must win by at least 8%.
      const std::array<YE3T_LAMMPS::TaggedCauchyExecutionPolicy, 4> policies = {
          YE3T_LAMMPS::TaggedCauchyExecutionPolicy::COMPILED_DIRECT,
          YE3T_LAMMPS::TaggedCauchyExecutionPolicy::GENERIC_DAG,
          YE3T_LAMMPS::TaggedCauchyExecutionPolicy::SYMMETRIC_POWER,
          YE3T_LAMMPS::TaggedCauchyExecutionPolicy::BLOCK,
      };
      if (fastest_candidate != 0 &&
          tagged_evaluator_->auto_candidate_confidently_faster(
              policies[static_cast<std::size_t>(fastest_candidate)], TAGGED_AUTO_SWITCH_RATIO))
        selected_candidate = fastest_candidate;
    }
    MPI_Bcast(candidate_seconds.data(), 4, MPI_DOUBLE, 0, world);
    MPI_Bcast(&calibration_seconds, 1, MPI_DOUBLE, 0, world);
    MPI_Bcast(&selected_candidate, 1, MPI_INT, 0, world);
    const std::array<YE3T_LAMMPS::TaggedCauchyExecutionPolicy, 4> policies = {
        YE3T_LAMMPS::TaggedCauchyExecutionPolicy::COMPILED_DIRECT,
        YE3T_LAMMPS::TaggedCauchyExecutionPolicy::GENERIC_DAG,
        YE3T_LAMMPS::TaggedCauchyExecutionPolicy::SYMMETRIC_POWER,
        YE3T_LAMMPS::TaggedCauchyExecutionPolicy::BLOCK,
    };
    tagged_evaluator_->freeze_auto_policy(policies[static_cast<std::size_t>(selected_candidate)]);
    semantic_selection_hash_ = YE3T_LAMMPS::sha256_string(
        "ye3t_tagged_execution_selection_v2;" + tagged_model_->execution_portfolio.portfolio_hash +
        ";" + tagged_evaluator_->selected_evaluator_name() +
        ";calibration_v3_quartile;centers=" + std::to_string(TAGGED_AUTO_CENTERS) +
        ";neighbors=" + std::to_string(TAGGED_AUTO_NEIGHBORS) +
        ";repeats=" + std::to_string(TAGGED_AUTO_REPEATS) + ";switch_ratio=0.92");
    if (comm->me == 0)
      utils::logmesg(lmp,
                     "YE3T tagged AUTO calibration: centers {}, neighbors/center {}, "
                     "repeats {}, direct {:.9g} s, generic_dag {:.9g} s, "
                     "symmetric_power {:.9g} s, block {:.9g} s; selected {}; "
                     "calibration {:.9g} s (excluded from steady-state timing)\n",
                     TAGGED_AUTO_CENTERS, TAGGED_AUTO_NEIGHBORS, TAGGED_AUTO_REPEATS,
                     candidate_seconds[0], candidate_seconds[1], candidate_seconds[2],
                     candidate_seconds[3], tagged_evaluator_->selected_evaluator_name(),
                     calibration_seconds);
    if (comm->me == 0) {
      const auto &lower = tagged_evaluator_->auto_calibration_lower_seconds();
      const auto &upper = tagged_evaluator_->auto_calibration_upper_seconds();
      utils::logmesg(lmp,
                     "YE3T tagged AUTO quartiles: direct [{:.9g}, {:.9g}] s, "
                     "generic_dag [{:.9g}, {:.9g}] s, symmetric_power [{:.9g}, "
                     "{:.9g}] s, block [{:.9g}, {:.9g}] s; non-direct switch "
                     "requires candidate upper < {:.3g} * direct lower\n",
                     lower[0], upper[0], lower[1], upper[1], lower[2], upper[2], lower[3], upper[3],
                     TAGGED_AUTO_SWITCH_RATIO);
    }
  }
  if (comm->me == 0) {
    if (tagged_model_->has_ordinary_backbone())
      utils::logmesg(lmp,
                     "YE3T self-contained composite: ordinary model {}, tagged model {}, "
                     "manifest {}, ordinary evaluator compiled_direct, tagged evaluator "
                     "{}\n",
                     tagged_model_->ordinary_model_hash, tagged_model_->tagged_component_hash,
                     tagged_model_->composite_self_hash,
                     tagged_evaluator_->selected_evaluator_name());
    if (tagged_model_->deployment_kind == YE3T_LAMMPS::TaggedCauchyDeploymentKind::PhysicalImageV3)
      utils::logmesg(lmp,
                     "YE3T tagged-Cauchy CPU dispatch: compiler-owned physical-image "
                     "schedule, direct shifted-Jacobi source, division-free adjoint, "
                     "tag_count {}, feature_count {}, schedule {}, deployment {}\n",
                     tagged_model_->tag_count, tagged_model_->feature_count,
                     tagged_model_->schedule_hash, tagged_model_->deployment_identity_hash);
    else
      utils::logmesg(lmp,
                     "YE3T tagged-Cauchy CPU dispatch: real-arithmetic exact moment "
                     "reduction, evaluator {}, portfolio {}, tag_count {}, feature_count "
                     "{}, model_self_hash {}\n",
                     tagged_evaluator_->selected_evaluator_name(),
                     tagged_model_->execution_portfolio.present
                         ? tagged_model_->execution_portfolio.portfolio_hash
                         : "legacy_compiled_direct_compatibility",
                     tagged_model_->tag_count, tagged_model_->feature_count,
                     tagged_model_->self_hash);
  }
}

YE3T_LAMMPS::YACEBlockPolicy PairYE3T::model_load_policy() const
{
  return block_policy_;
}

bool PairYE3T::supports_lifted_cauchy() const
{
  return true;
}
bool PairYE3T::supports_tagged_cauchy() const
{
  return true;
}

void PairYE3T::coeff(int narg, char **arg)
{
  if (narg != atom->ntypes + 3) error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  const std::string potential_path = utils::get_potential_file_path(arg[2]);
  const std::string sidecar_path =
      sidecar_manifest_.empty() ? std::string() : utils::get_potential_file_path(sidecar_manifest_);
  const std::string replay_path =
      auto_replay_path_.empty() ? std::string() : utils::get_potential_file_path(auto_replay_path_);
  if (potential_path.empty()) error->all(FLERR, "Could not locate YE3T potential {}", arg[2]);

  const auto map_species = [&](const std::vector<std::string> &species) {
    map_element2type(narg - 3, arg + 3);
    for (int type = 1; type <= atom->ntypes; ++type) {
      const char *name = arg[type + 2];
      if (strcmp(name, "NULL") == 0) {
        map[type] = -1;
        continue;
      }
      const auto found = std::find(species.begin(), species.end(), name);
      if (found == species.end())
        error->all(FLERR, "Element {} is not present in YE3T potential {}", name, potential_path);
      map[type] = static_cast<int>(found - species.begin());
    }
  };

  if (model_family_ == ModelFamily::LIFTED_CAUCHY) {
    if (!supports_lifted_cauchy())
      error->all(FLERR,
                 "Pair style ye3t/kk does not yet support model_family "
                 "lifted_cauchy");
    try {
      lifted_model_ = std::make_unique<YE3T_LAMMPS::LiftedCauchyModel>(
          YE3T_LAMMPS::LiftedCauchyModel::load(potential_path));
      std::string realization = lifted_model_->default_source_realization;
      if (lifted_source_selection_ == LiftedSourceSelection::DIRECT_Q)
        realization = "direct";
      else if (lifted_source_selection_ == LiftedSourceSelection::FACTORIZED_T)
        realization = "factorized";
      if (realization == "direct")
        lifted_source_policy_ = YE3T_LAMMPS::LiftedCauchySourcePolicy::DIRECT_Q;
      else if (realization == "factorized")
        lifted_source_policy_ = YE3T_LAMMPS::LiftedCauchySourcePolicy::FACTORIZED;
      else
        throw std::runtime_error("model default source realization is auto; select direct or "
                                 "factorized explicitly");
      lifted_source_ = std::make_unique<YE3T_LAMMPS::LiftedCauchyCPUSource>(lifted_model_.get());
      lifted_readout_ =
          std::make_unique<YE3T_LAMMPS::LiftedCauchyCPULoweredReadout>(lifted_model_.get());
      model_.reset();
      evaluator_.reset();
      source_model_hash_ = YE3T_LAMMPS::sha256_file(lifted_model_->model_path);
      logical_direct_plan_hash_ = lifted_model_->native_self_hash;
      semantic_selection_hash_ = lifted_model_->deployment_identity_hash;
    } catch (const std::exception &exception) {
      error->all(FLERR, "Could not load YE3T lifted-Cauchy potential: {}", exception.what());
    }
    map_species(lifted_model_->central_species_order);
    try {
      initialize_lifted_backend();
    } catch (const std::exception &exception) {
      error->all(FLERR, "Could not initialize YE3T lifted backend: {}", exception.what());
    }
    if (comm->me == 0)
      utils::logmesg(lmp, "Loaded YE3T lifted-Cauchy potential {}\n", potential_path);
    return;
  }

  if (model_family_ == ModelFamily::TAGGED_CAUCHY) {
    if (!supports_tagged_cauchy())
      error->all(FLERR,
                 "Pair style ye3t/kk does not yet support model_family "
                 "tagged_cauchy");
    try {
      tagged_model_ = std::make_unique<YE3T_LAMMPS::TaggedCauchyModel>(
          YE3T_LAMMPS::TaggedCauchyModel::load(potential_path));
      tagged_evaluator_ = std::make_unique<YE3T_LAMMPS::TaggedCauchyCPUEvaluator>(
          tagged_model_.get(), tagged_execution_policy(block_policy_));
      evaluator_.reset();
      model_.reset();
      if (tagged_model_->has_ordinary_backbone()) {
        model_ = std::make_unique<YE3T_LAMMPS::YACEModel>(
            YE3T_LAMMPS::YACEModel::load(tagged_model_->ordinary_model_path, std::string(),
                                         YE3T_LAMMPS::YACEBlockPolicy::DIRECT));
        if (model_->species_count() != static_cast<int>(tagged_model_->species_order.size()))
          throw std::runtime_error("ordinary and tagged component species counts differ");
        for (int species = 0; species < model_->species_count(); ++species)
          if (model_->species(species).element !=
              tagged_model_->species_order[static_cast<std::size_t>(species)])
            throw std::runtime_error("ordinary and tagged component species ordering differs");
        const double cutoff_tolerance = 2.0e-12 * std::max(1.0, std::abs(tagged_model_->cutoff));
        if (std::abs(model_->maximum_cutoff() - tagged_model_->cutoff) > cutoff_tolerance)
          throw std::runtime_error("ordinary and tagged component cutoffs differ");
      }
      source_model_hash_ = tagged_model_->has_ordinary_backbone()
          ? tagged_model_->composite_self_hash
          : YE3T_LAMMPS::sha256_file(tagged_model_->model_path);
      logical_direct_plan_hash_ =
          tagged_model_->deployment_kind == YE3T_LAMMPS::TaggedCauchyDeploymentKind::PhysicalImageV3
          ? tagged_model_->schedule_hash
          : tagged_model_->self_hash;
      semantic_selection_hash_ = tagged_model_->execution_portfolio.present
          ? YE3T_LAMMPS::sha256_string("ye3t_tagged_execution_selection_v1;" +
                                       tagged_model_->execution_portfolio.portfolio_hash + ";" +
                                       tagged_evaluator_->selected_evaluator_name())
          : (tagged_model_->deployment_kind ==
                     YE3T_LAMMPS::TaggedCauchyDeploymentKind::PhysicalImageV3
                 ? tagged_model_->deployment_identity_hash
                 : tagged_model_->self_hash);
    } catch (const std::exception &exception) {
      error->all(FLERR, "Could not load YE3T tagged-Cauchy potential: {}", exception.what());
    }
    map_species(tagged_model_->species_order);
    {
      // Every artifact species must be mapped by some pair_coeff
      // type, not only the reverse (every mapped type names an artifact
      // species, already enforced inside map_species above).
      std::vector<bool> species_mapped(tagged_model_->species_order.size(), false);
      for (int type = 1; type <= atom->ntypes; ++type)
        if (map[type] >= 0) species_mapped[static_cast<std::size_t>(map[type])] = true;
      for (std::size_t species = 0; species < species_mapped.size(); ++species)
        if (!species_mapped[species])
          error->all(FLERR,
                     "YE3T tagged-Cauchy potential species '{}' is not "
                     "mapped by any pair_coeff type",
                     tagged_model_->species_order[species]);
    }
    try {
      if (model_) initialize_backend();
      initialize_tagged_backend();
    } catch (const std::exception &exception) {
      error->all(FLERR, "Could not initialize YE3T tagged backend: {}", exception.what());
    }
    if (comm->me == 0)
      utils::logmesg(lmp, "Loaded YE3T tagged-Cauchy potential {}\n", potential_path);
    return;
  }

  const auto load_policy = model_load_policy();
  if (!replay_path.empty() && load_policy != YE3T_LAMMPS::YACEBlockPolicy::GPU_AUTO)
    error->all(FLERR, "Pair style ye3t auto_replay is available only for ye3t/kk");
  try {
    model_ = std::make_unique<YE3T_LAMMPS::YACEModel>(
        YE3T_LAMMPS::YACEModel::load(potential_path, sidecar_path, load_policy, replay_path));
    source_model_hash_ = YE3T_LAMMPS::sha256_file(model_->source_path());
    logical_direct_plan_hash_ =
        YE3T_LAMMPS::sha256_string("ye3t_yace_direct_logical_v1;" + source_model_hash_);
    std::string selection_identity = "ye3t_evaluator_semantic_selection_v1;" + source_model_hash_ +
        ";" + std::to_string(model_->species_count()) + ";";
    for (int species = 0; species < model_->species_count(); ++species) {
      const auto &entry = model_->species(species);
      selection_identity += entry.element + ";" + entry.block_program.evaluator_plan_hash + ";";
    }
    semantic_selection_hash_ = YE3T_LAMMPS::sha256_string(selection_identity);
  } catch (const std::exception &exception) {
    error->all(FLERR, "Could not load YE3T YACE potential: {}", exception.what());
  }

  std::vector<std::string> species;
  species.reserve(static_cast<std::size_t>(model_->species_count()));
  for (int index = 0; index < model_->species_count(); ++index)
    species.push_back(model_->species(index).element);
  map_species(species);
  try {
    initialize_backend();
  } catch (const std::exception &exception) {
    error->all(FLERR, "Could not initialize YE3T evaluator: {}", exception.what());
  }

  if (comm->me == 0) {
    utils::logmesg(lmp, "Loaded YE3T YACE potential {}\n", potential_path);
    for (int species = 0; species < model_->species_count(); ++species) {
      const auto &entry = model_->species(species);
      if (entry.block_program.candidate_route_count > 0)
        utils::logmesg(
            lmp,
            "\t{}: {} selected block routes and {} selected coupled-product "
            "DAGs from {} compiled candidate routes, {} total "
            "candidates across {}/{} candidate/catalogue functions, {} "
            "scalar-power routes, {} symmetric-power blocks, {} scalar "
            "bases, {} scalar nodes, "
            "estimated operations {} -> {}, planner {} ({}, {}, {}, {} "
            "scores), plan {}\n",
            entry.element, entry.block_program.routes.size(),
            entry.block_program.coupled_product_plans.size(),
            entry.block_program.candidate_route_count, entry.block_program.candidate_count,
            entry.block_program.candidate_function_count,
            entry.block_program.catalogue_function_count,
            entry.block_program.scalar_program.routes.size(),
            entry.block_program.power_plans.size(), entry.block_program.scalar_program.bases.size(),
            entry.block_program.scalar_program.nodes.size(),
            entry.block_program.direct_operation_estimate,
            entry.block_program.selected_operation_estimate, entry.block_program.planner_algorithm,
            entry.block_program.planner_profile, entry.block_program.planner_status,
            entry.block_program.planner_optimal ? "optimal" : "heuristic",
            entry.block_program.planner_score_evaluations, entry.block_program.plan_hash);
      if (entry.block_program.candidate_route_count > 0)
        utils::logmesg(lmp, "\t{}: evaluator portfolio {}\n", entry.element,
                       entry.block_program.evaluator_plan_hash);
      if (entry.block_program.decisions.size() <= 12)
        for (const auto &decision : entry.block_program.decisions) {
          const char *evaluator = "direct";
          if (decision.selected_evaluator == YE3T_LAMMPS::YACEEvaluatorKind::BLOCK_SYMMETRIC_POWER)
            evaluator = "block";
          else if (decision.selected_evaluator ==
                   YE3T_LAMMPS::YACEEvaluatorKind::SCALAR_INVARIANT_POWER)
            evaluator = "scalar_power";
          else if (decision.selected_evaluator ==
                   YE3T_LAMMPS::YACEEvaluatorKind::COUPLED_PRODUCT_DAG)
            evaluator = "coupled_product";
          utils::logmesg(lmp, "\t{}: function {} evaluator {} candidate {} ({}/{} choices)\n",
                         entry.element, decision.function_index, evaluator,
                         decision.selected_candidate_id, decision.selected_candidate_index + 1,
                         decision.candidates.size());
        }
      if (entry.polynomial.binary_dag) {
        utils::logmesg(lmp,
                       "\t{}: {} full channels, {} source channels, {} shared monomials, "
                       "{} cached powers, {} binary product-DAG nodes ({}), rank <= {}\n",
                       entry.element, entry.channels.size(), entry.source_channels.size(),
                       entry.polynomial.monomial_coefficients.size(),
                       entry.polynomial.power_channels.size(),
                       entry.polynomial.binary_node_left.size(),
                       entry.polynomial.dag_factor_ordering, entry.polynomial.maximum_rank);
      } else {
        utils::logmesg(
            lmp,
            "\t{}: {} full channels, {} source channels, {} shared monomials, "
            "{} cached powers, "
            "{} prefix product-DAG nodes with {} roots ({}), rank <= {}\n",
            entry.element, entry.channels.size(), entry.source_channels.size(),
            entry.polynomial.monomial_coefficients.size(), entry.polynomial.power_channels.size(),
            entry.polynomial.dag_node_parents.size(), entry.polynomial.dag_root_node_count,
            entry.polynomial.dag_factor_ordering, entry.polynomial.maximum_rank);
      }
    }
  }
}

void PairYE3T::init_style()
{
  if (!model_loaded() || (model_family_ == ModelFamily::YACE && !evaluator_) ||
      (model_family_ == ModelFamily::LIFTED_CAUCHY && (!lifted_source_ || !lifted_readout_)) ||
      (model_family_ == ModelFamily::TAGGED_CAUCHY &&
       (!tagged_evaluator_ || (tagged_model_->has_ordinary_backbone() && !evaluator_))))
    error->all(FLERR, "Pair style ye3t requires pair_coeff before run");
  if (atom->tag_enable == 0) error->all(FLERR, "Pair style ye3t requires atom IDs");
  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style ye3t CPU compatibility route requires newton pair on");
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

double PairYE3T::init_one(int i, int j)
{
  if (setflag[i][j] == 0)
    error->all(FLERR, Error::NOLASTLINE,
               "All pair coeffs are not set. Status:\n" + Info::get_pair_coeff_status(lmp));
  const int first = map[i];
  const int second = map[j];
  if (model_family_ == ModelFamily::LIFTED_CAUCHY) return lifted_model_->cutoff;
  if (model_family_ == ModelFamily::TAGGED_CAUCHY)
    return model_ ? std::max(tagged_model_->cutoff, model_->bond(first, second).cutoff)
                  : tagged_model_->cutoff;
  return std::max(model_->bond(first, second).cutoff, model_->bond(second, first).cutoff);
}

void PairYE3T::evaluate_lifted_chunk(int atom_count, int edge_count)
{
  const std::int64_t source_count = lifted_model_->source_variable_count;
  if (static_cast<std::uint64_t>(atom_count) >
      std::numeric_limits<std::size_t>::max() / static_cast<std::uint64_t>(source_count))
    throw std::overflow_error("lifted-Cauchy source batch size overflows");
  const std::size_t source_value_count =
      static_cast<std::size_t>(atom_count) * static_cast<std::size_t>(source_count);
  const std::size_t maximum_source_values = LIFTED_SOURCE_BATCH_BYTES / (2 * sizeof(double));
  if (source_value_count > maximum_source_values)
    throw std::runtime_error("lifted-Cauchy source batch exceeds the internal workspace limit");
  if (lifted_edge_offsets_.size() != static_cast<std::size_t>(atom_count + 1) ||
      lifted_edge_offsets_.back() != static_cast<std::size_t>(edge_count))
    throw std::runtime_error("lifted-Cauchy center/edge offsets are invalid");

  resize_growing(atomic_energies_, static_cast<std::size_t>(atom_count));
  resize_growing(edge_gradients_, static_cast<std::size_t>(edge_count) * 3);
  const auto resize_source_batch = [&](std::vector<double> &values) {
    if (values.capacity() < source_value_count) {
      const std::size_t grown = values.capacity() + values.capacity() / 2 + 64;
      values.reserve(std::min(maximum_source_values, std::max(grown, source_value_count)));
    }
    values.resize(source_value_count);
  };
  resize_source_batch(lifted_source_values_);
  resize_source_batch(lifted_source_adjoint_);

  const auto fill_center_edges = [&](int center) {
    const std::size_t begin = lifted_edge_offsets_[static_cast<std::size_t>(center)];
    const std::size_t end = lifted_edge_offsets_[static_cast<std::size_t>(center + 1)];
    lifted_center_edges_.clear();
    if (lifted_center_edges_.capacity() < end - begin) lifted_center_edges_.reserve(end - begin);
    for (std::size_t edge = begin; edge < end; ++edge) {
      YE3T_LAMMPS::LiftedCauchyEdge value;
      value.neighbor_species_index = edge_neighbor_species_[edge];
      value.displacement = {edge_vectors_[edge * 3], edge_vectors_[edge * 3 + 1],
                            edge_vectors_[edge * 3 + 2]};
      lifted_center_edges_.push_back(value);
    }
  };

  for (int center = 0; center < atom_count; ++center) {
    fill_center_edges(center);
    lifted_source_->accumulate(lifted_center_edges_, lifted_source_policy_, lifted_center_source_);
    if (lifted_center_source_.size() != static_cast<std::size_t>(source_count))
      throw std::runtime_error("lifted-Cauchy source returned an invalid source count");
    std::copy(lifted_center_source_.begin(), lifted_center_source_.end(),
              lifted_source_values_.begin() +
                  static_cast<std::size_t>(center) * static_cast<std::size_t>(source_count));
  }

  lifted_readout_->evaluate(atom_count, central_species_.data(), lifted_source_values_.data(),
                            atomic_energies_.data(), lifted_source_adjoint_.data());

  for (int center = 0; center < atom_count; ++center) {
    fill_center_edges(center);
    const auto source_begin = lifted_source_adjoint_.begin() +
        static_cast<std::size_t>(center) * static_cast<std::size_t>(source_count);
    lifted_center_source_.assign(source_begin,
                                 source_begin + static_cast<std::size_t>(source_count));
    lifted_source_->vjp(lifted_center_edges_, lifted_center_source_, lifted_source_policy_,
                        lifted_center_edge_gradients_);
    const std::size_t edge_begin = lifted_edge_offsets_[static_cast<std::size_t>(center)];
    if (lifted_center_edge_gradients_.size() !=
        lifted_edge_offsets_[static_cast<std::size_t>(center + 1)] - edge_begin)
      throw std::runtime_error("lifted-Cauchy source VJP returned an invalid edge count");
    for (std::size_t edge = 0; edge < lifted_center_edge_gradients_.size(); ++edge) {
      for (int component = 0; component < 3; ++component)
        edge_gradients_[(edge_begin + edge) * 3 + static_cast<std::size_t>(component)] =
            lifted_center_edge_gradients_[edge][static_cast<std::size_t>(component)];
    }
  }
}

void PairYE3T::evaluate_tagged_chunk(int atom_count, int edge_count)
{
  resize_growing(atomic_energies_, static_cast<std::size_t>(atom_count));
  resize_growing(edge_gradients_, static_cast<std::size_t>(edge_count) * 3);
  if (tagged_edge_offsets_.size() != static_cast<std::size_t>(atom_count + 1) ||
      tagged_edge_offsets_.back() != static_cast<std::size_t>(edge_count))
    throw std::runtime_error("tagged-Cauchy center/edge offsets are invalid");
  tagged_evaluator_->evaluate(atom_count, central_species_.data(), tagged_edge_offsets_.data(),
                              edge_neighbor_species_.data(), edge_vectors_.data(),
                              atomic_energies_.data(), edge_gradients_.data());
  if (evaluator_) {
    resize_growing(ordinary_atomic_energies_, static_cast<std::size_t>(atom_count));
    resize_growing(ordinary_edge_gradients_, static_cast<std::size_t>(edge_count) * 3);
    evaluator_->evaluate(atom_count, central_species_.data(), edge_count, edge_centers_.data(),
                         edge_neighbor_species_.data(), edge_vectors_.data(),
                         ordinary_atomic_energies_.data(), ordinary_edge_gradients_.data());
    for (int center = 0; center < atom_count; ++center)
      atomic_energies_[static_cast<std::size_t>(center)] +=
          ordinary_atomic_energies_[static_cast<std::size_t>(center)];
    for (std::size_t value = 0; value < static_cast<std::size_t>(edge_count) * 3; ++value)
      edge_gradients_[value] += ordinary_edge_gradients_[value];
  }
}

void PairYE3T::compute(int eflag, int vflag)
{
  if (copymode)
    ev_init(eflag, vflag, 0);
  else
    ev_init(eflag, vflag, 1);

  double **positions = atom->x;
  double **forces = atom->f;
  int *types = atom->type;
  const int nlocal = atom->nlocal;
  const int newton_pair = force->newton_pair;

  const int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  const bool lifted = model_family_ == ModelFamily::LIFTED_CAUCHY;
  const bool tagged = model_family_ == ModelFamily::TAGGED_CAUCHY;
  int evaluation_chunksize = chunksize_;
  if (lifted) {
    const std::uint64_t source_count =
        static_cast<std::uint64_t>(lifted_model_->source_variable_count);
    const std::uint64_t maximum_source_values = LIFTED_SOURCE_BATCH_BYTES / (2 * sizeof(double));
    if (source_count == 0 || source_count > maximum_source_values)
      error->all(FLERR,
                 "YE3T lifted-Cauchy source width exceeds the internal CPU "
                 "workspace limit");
    const std::uint64_t maximum_centers = maximum_source_values / source_count;
    evaluation_chunksize = std::min(
        chunksize_,
        static_cast<int>(std::min<std::uint64_t>(
            maximum_centers, static_cast<std::uint64_t>(std::numeric_limits<int>::max()))));
  }
  maximum_imaginary_density_ = 0.0;
  int next_chunk_begin = 0;
  while (next_chunk_begin < inum) {
    const int chunk_begin = next_chunk_begin;
    // Subtraction avoids integer overflow for a very large configured chunk.
    const int chunk_end = chunk_begin + std::min(evaluation_chunksize, inum - chunk_begin);
    next_chunk_begin = chunk_end;
    center_atoms_.clear();
    central_species_.clear();
    edge_centers_.clear();
    edge_neighbors_.clear();
    edge_neighbor_species_.clear();
    edge_vectors_.clear();
    if (lifted) {
      lifted_edge_offsets_.clear();
      lifted_edge_offsets_.push_back(0);
    }
    if (tagged) {
      tagged_edge_offsets_.clear();
      tagged_edge_offsets_.push_back(0);
    }
    center_atoms_.reserve(static_cast<std::size_t>(chunk_end - chunk_begin));
    central_species_.reserve(static_cast<std::size_t>(chunk_end - chunk_begin));

    for (int ii = chunk_begin; ii < chunk_end; ++ii) {
      const int atom_i = ilist[ii];
      const int central = map[types[atom_i]];
      if (central < 0) continue;
      const int center_slot = static_cast<int>(center_atoms_.size());
      center_atoms_.push_back(atom_i);
      central_species_.push_back(central);
      int *neighbors = firstneigh[atom_i];
      for (int jj = 0; jj < numneigh[atom_i]; ++jj) {
        const int atom_j = neighbors[jj] & NEIGHMASK;
        const int neighbor = map[types[atom_j]];
        if (neighbor < 0) continue;
        const double x = positions[atom_j][0] - positions[atom_i][0];
        const double y = positions[atom_j][1] - positions[atom_i][1];
        const double z = positions[atom_j][2] - positions[atom_i][2];
        const double cutoff = lifted ? lifted_model_->cutoff
            : tagged
            ? (model_ ? std::max(tagged_model_->cutoff, model_->bond(central, neighbor).cutoff)
                      : tagged_model_->cutoff)
            : model_->bond(central, neighbor).cutoff;
        if (x * x + y * y + z * z >= cutoff * cutoff) continue;
        edge_centers_.push_back(center_slot);
        edge_neighbors_.push_back(atom_j);
        edge_neighbor_species_.push_back(neighbor);
        edge_vectors_.push_back(x);
        edge_vectors_.push_back(y);
        edge_vectors_.push_back(z);
      }
      if (lifted) lifted_edge_offsets_.push_back(edge_centers_.size());
      if (tagged) tagged_edge_offsets_.push_back(edge_centers_.size());
    }

    const int atom_count = static_cast<int>(center_atoms_.size());
    const int edge_count = static_cast<int>(edge_centers_.size());
    if (atom_count == 0) continue;
    try {
      if (lifted) {
        evaluate_lifted_chunk(atom_count, edge_count);
      } else if (tagged) {
        evaluate_tagged_chunk(atom_count, edge_count);
      } else {
        resize_growing(atomic_energies_, static_cast<std::size_t>(atom_count));
        resize_growing(edge_gradients_, static_cast<std::size_t>(edge_count) * 3);
        evaluator_->evaluate(atom_count, central_species_.data(), edge_count, edge_centers_.data(),
                             edge_neighbor_species_.data(), edge_vectors_.data(),
                             atomic_energies_.data(), edge_gradients_.data());
      }
    } catch (const std::exception &exception) {
      error->one(FLERR, "YE3T CPU evaluation failed: {}", exception.what());
    }
    if ((!lifted && !tagged) || (tagged && evaluator_))
      maximum_imaginary_density_ =
          std::max(maximum_imaginary_density_, evaluator_->maximum_imaginary_density());

    for (int edge = 0; edge < edge_count; ++edge) {
      const int atom_i =
          center_atoms_[static_cast<std::size_t>(edge_centers_[static_cast<std::size_t>(edge)])];
      const int atom_j = edge_neighbors_[static_cast<std::size_t>(edge)];
      const double fx = edge_gradients_[static_cast<std::size_t>(edge * 3)];
      const double fy = edge_gradients_[static_cast<std::size_t>(edge * 3 + 1)];
      const double fz = edge_gradients_[static_cast<std::size_t>(edge * 3 + 2)];
      forces[atom_i][0] += fx;
      forces[atom_i][1] += fy;
      forces[atom_i][2] += fz;
      forces[atom_j][0] -= fx;
      forces[atom_j][1] -= fy;
      forces[atom_j][2] -= fz;
      if (vflag_either) {
        const double x = edge_vectors_[static_cast<std::size_t>(edge * 3)];
        const double y = edge_vectors_[static_cast<std::size_t>(edge * 3 + 1)];
        const double z = edge_vectors_[static_cast<std::size_t>(edge * 3 + 2)];
        ev_tally_xyz(atom_i, atom_j, nlocal, newton_pair, 0.0, 0.0, fx, fy, fz, -x, -y, -z);
      }
    }

    if (eflag_either) {
      for (int center = 0; center < atom_count; ++center)
        ev_tally_full(center_atoms_[static_cast<std::size_t>(center)],
                      2.0 * atomic_energies_[static_cast<std::size_t>(center)], 0.0, 0.0, 0.0, 0.0,
                      0.0);
    }
  }
  if (vflag_fdotr) virial_fdotr_compute();
}

void *PairYE3T::extract(const char *name, int &dimension)
{
  dimension = 0;
  if (strcmp(name, "maximum_imaginary_density") == 0)
    return static_cast<void *>(&maximum_imaginary_density_);
  return nullptr;
}

double PairYE3T::memory_usage()
{
  double bytes = Pair::memory_usage();
  bytes += center_atoms_.capacity() * sizeof(int);
  bytes += central_species_.capacity() * sizeof(int);
  bytes += edge_centers_.capacity() * sizeof(int);
  bytes += edge_neighbors_.capacity() * sizeof(int);
  bytes += edge_neighbor_species_.capacity() * sizeof(int);
  bytes += edge_vectors_.capacity() * sizeof(double);
  bytes += atomic_energies_.capacity() * sizeof(double);
  bytes += edge_gradients_.capacity() * sizeof(double);
  bytes += ordinary_atomic_energies_.capacity() * sizeof(double);
  bytes += ordinary_edge_gradients_.capacity() * sizeof(double);
  bytes += lifted_edge_offsets_.capacity() * sizeof(std::size_t);
  bytes += lifted_source_values_.capacity() * sizeof(double);
  bytes += lifted_source_adjoint_.capacity() * sizeof(double);
  bytes += lifted_center_edges_.capacity() * sizeof(YE3T_LAMMPS::LiftedCauchyEdge);
  bytes += lifted_center_source_.capacity() * sizeof(double);
  bytes += lifted_center_edge_gradients_.capacity() * sizeof(std::array<double, 3>);
  bytes += tagged_edge_offsets_.capacity() * sizeof(std::size_t);
  if (model_) bytes += model_->memory_usage();
  if (lifted_source_) bytes += lifted_source_->memory_usage();
  if (evaluator_) bytes += evaluator_->memory_usage();
  if (lifted_model_) bytes += lifted_model_->memory_usage();
  if (lifted_readout_) bytes += lifted_readout_->memory_usage();
  if (tagged_model_) bytes += tagged_model_->memory_usage();
  if (tagged_evaluator_) bytes += tagged_evaluator_->memory_usage();
  return bytes;
}
