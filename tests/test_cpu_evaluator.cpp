// clang-format off
#include "ye3t_cpu_evaluator.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
// clang-format on

namespace YE3T_LAMMPS {
struct YE3TCPUEvaluatorTestAccess {
  static void set_edge_capacity(YE3TCPUEvaluator &evaluator, int capacity)
  {
    if (capacity <= 0) throw std::invalid_argument("invalid test tile capacity");
    evaluator.reference_edge_tiling_ = true;
    evaluator.source_tile_policy_.edge_capacity = capacity;
  }
  static std::size_t resident_edge_rows(const YE3TCPUEvaluator &evaluator)
  {
    return evaluator.radii_.size();
  }
};
}    // namespace YE3T_LAMMPS

namespace {

void require(bool condition, const std::string &message)
{
  if (!condition) throw std::runtime_error(message);
}

struct Environment {
  int central = -1;
  std::vector<int> species;
  std::vector<double> positions;
  std::vector<int> neighbors;
};

Environment load_environment(const std::string &path)
{
  std::ifstream stream(path);
  require(stream.good(), "could not open environment fixture");
  std::string token;
  stream >> token;
  require(token == "pace_environment_v1", "unsupported environment schema");
  stream >> token;
  require(token == "central", "missing central record");
  Environment environment;
  stream >> environment.central;
  stream >> token;
  require(token == "atoms", "missing atom count");
  int atom_count = 0;
  stream >> atom_count;
  require(atom_count > 0, "empty environment");
  environment.species.resize(static_cast<std::size_t>(atom_count));
  environment.positions.resize(static_cast<std::size_t>(atom_count) * 3);
  for (int atom = 0; atom < atom_count; ++atom) {
    stream >> environment.species[static_cast<std::size_t>(atom)];
    stream >> environment.positions[static_cast<std::size_t>(atom * 3)];
    stream >> environment.positions[static_cast<std::size_t>(atom * 3 + 1)];
    stream >> environment.positions[static_cast<std::size_t>(atom * 3 + 2)];
  }
  stream >> token;
  require(token == "neighbors", "missing neighbor count");
  int neighbor_count = 0;
  stream >> neighbor_count;
  require(neighbor_count >= 0, "negative neighbor count");
  environment.neighbors.resize(static_cast<std::size_t>(neighbor_count));
  for (int &neighbor : environment.neighbors) stream >> neighbor;
  require(static_cast<bool>(stream), "malformed environment fixture");
  return environment;
}

std::vector<double> edge_vectors(const Environment &environment)
{
  std::vector<double> vectors(environment.neighbors.size() * 3);
  for (std::size_t edge = 0; edge < environment.neighbors.size(); ++edge) {
    const int neighbor = environment.neighbors[edge];
    for (int axis = 0; axis < 3; ++axis)
      vectors[edge * 3 + static_cast<std::size_t>(axis)] =
          environment.positions[static_cast<std::size_t>(neighbor * 3 + axis)] -
          environment.positions[static_cast<std::size_t>(environment.central * 3 + axis)];
  }
  return vectors;
}

void evaluate(YE3T_LAMMPS::YE3TCPUEvaluator &evaluator, const Environment &environment,
              const std::vector<double> &vectors, double &energy, std::vector<double> &gradients)
{
  const int edge_count = static_cast<int>(environment.neighbors.size());
  const int central_species = environment.species[static_cast<std::size_t>(environment.central)];
  std::vector<int> centers(static_cast<std::size_t>(edge_count), 0);
  std::vector<int> neighbor_species(static_cast<std::size_t>(edge_count));
  for (int edge = 0; edge < edge_count; ++edge)
    neighbor_species[static_cast<std::size_t>(edge)] = environment.species[static_cast<std::size_t>(
        environment.neighbors[static_cast<std::size_t>(edge)])];
  gradients.resize(static_cast<std::size_t>(edge_count) * 3);
  evaluator.evaluate(1, &central_species, edge_count, centers.data(), neighbor_species.data(),
                     vectors.data(), &energy, gradients.data());
}

void evaluate_repeated(YE3T_LAMMPS::YE3TCPUEvaluator &evaluator, const Environment &environment,
                       int atom_count, std::vector<double> &energies,
                       std::vector<double> &gradients)
{
  const auto environment_vectors = edge_vectors(environment);
  const int neighbors_per_atom = static_cast<int>(environment.neighbors.size());
  const int edge_count = atom_count * neighbors_per_atom;
  const int species = environment.species[static_cast<std::size_t>(environment.central)];
  std::vector<int> central_species(static_cast<std::size_t>(atom_count), species);
  std::vector<int> centers(static_cast<std::size_t>(edge_count));
  std::vector<int> neighbor_species(static_cast<std::size_t>(edge_count));
  std::vector<double> vectors(static_cast<std::size_t>(edge_count) * 3);
  for (int atom = 0; atom < atom_count; ++atom)
    for (int local_edge = 0; local_edge < neighbors_per_atom; ++local_edge) {
      const int edge = atom * neighbors_per_atom + local_edge;
      centers[static_cast<std::size_t>(edge)] = atom;
      neighbor_species[static_cast<std::size_t>(edge)] =
          environment.species[static_cast<std::size_t>(
              environment.neighbors[static_cast<std::size_t>(local_edge)])];
      for (int axis = 0; axis < 3; ++axis)
        vectors[static_cast<std::size_t>(edge * 3 + axis)] =
            environment_vectors[static_cast<std::size_t>(local_edge * 3 + axis)];
    }
  energies.resize(static_cast<std::size_t>(atom_count));
  gradients.resize(static_cast<std::size_t>(edge_count) * 3);
  evaluator.evaluate(atom_count, central_species.data(), edge_count, centers.data(),
                     neighbor_species.data(), vectors.data(), energies.data(), gradients.data());
}

// Runs only with the real YE3T runtime linked. The dependency-free schedule
// suite does not stand in for this energy/force and native-kernel regression.
void verify_native_source_tiling(const YE3T_LAMMPS::YACEModel &model,
                                 const Environment &environment)
{
  using YE3T_LAMMPS::YE3TCPUEvaluator;
  using Access = YE3T_LAMMPS::YE3TCPUEvaluatorTestAccess;
  for (int atoms : {0, 1, 9, 37, 257}) {
    YE3TCPUEvaluator untiled(&model);
    Access::set_edge_capacity(untiled, std::numeric_limits<int>::max());
    std::vector<double> expected_energy, expected_gradient;
    evaluate_repeated(untiled, environment, atoms, expected_energy, expected_gradient);
    const int automatic = YE3TCPUEvaluator(&model).source_tile_edge_capacity();
    for (int cap : {1, 2, 3, 7, 31, 256, automatic}) {
      YE3TCPUEvaluator tiled(&model);
      Access::set_edge_capacity(tiled, cap);
      std::vector<double> energy, gradient;
      evaluate_repeated(tiled, environment, atoms, energy, gradient);
      require(energy.size() == expected_energy.size() &&
                  gradient.size() == expected_gradient.size(),
              "source tiling changed the readout or edge batch shape");
      for (std::size_t i = 0; i < energy.size(); ++i)
        require(std::abs(energy[i] - expected_energy[i]) <=
                    3e-12 * (1 + std::abs(expected_energy[i])),
                "native tiled energy differs from single source tile");
      for (std::size_t i = 0; i < gradient.size(); ++i)
        require(std::abs(gradient[i] - expected_gradient[i]) <=
                    3e-11 * (1 + std::abs(expected_gradient[i])),
                "native tiled VJP differs from single source tile");
      require(Access::resident_edge_rows(tiled) <= static_cast<std::size_t>(cap),
              "native source tables grew past internal edge tile");
      // Reuse across calls with a different input size; no stale source/adjoint.
      double single_energy = 0.0;
      evaluate(tiled, environment, edge_vectors(environment), single_energy, gradient);
      double reference_single_energy = 0.0;
      std::vector<double> reference_single_gradient;
      evaluate(untiled, environment, edge_vectors(environment), reference_single_energy,
               reference_single_gradient);
      require(std::abs(single_energy - reference_single_energy) <=
                  3e-12 * (1 + std::abs(reference_single_energy)),
              "reused source tile has stale energy");
      for (std::size_t i = 0; i < gradient.size(); ++i)
        require(std::abs(gradient[i] - reference_single_gradient[i]) <=
                    3e-11 * (1 + std::abs(reference_single_gradient[i])),
                "reused source tile has stale VJP");
      require(Access::resident_edge_rows(tiled) <= static_cast<std::size_t>(cap),
              "source tile capacity changed during reuse");
    }
  }
  std::cout << "native automatic/tiny/full source tiles agree in energy and VJP\n";
}

void run(const std::string &model_path, const std::string &environment_path)
{
  const auto model = YE3T_LAMMPS::YACEModel::load(model_path);
  require(model.species_count() == 1, "fixture species count mismatch");
  require(model.species(0).element == "Ta", "fixture element mismatch");
  require(model.species(0).polynomial.maximum_rank == 2, "fixture maximum rank mismatch");
  require(model.species(0).polynomial.monomial_coefficients.size() == 3,
          "shared monomial lowering did not retain the expected terms");
  require(model.species(0).source_channels.size() < model.species(0).channels.size(),
          "half-basis source lowering did not reduce the fixture channels");
  require(model.species(0).full_channel_sources.size() == model.species(0).channels.size() &&
              model.species(0).full_channel_transforms.size() == model.species(0).channels.size(),
          "half-basis source transform does not cover every full channel");
  require(std::any_of(model.species(0).full_channel_transforms.begin(),
                      model.species(0).full_channel_transforms.end(),
                      [](int transform) {
                        return transform != 0;
                      }),
          "half-basis source fixture does not exercise negative m reconstruction");
  const auto &plan = model.species(0).polynomial;
  require(plan.dag_factor_ordering == "channel-order" ||
              plan.dag_factor_ordering == "frequent-first" ||
              plan.dag_factor_ordering == "balanced-channel-order" ||
              plan.dag_factor_ordering == "balanced-reverse-channel-order" ||
              plan.dag_factor_ordering == "balanced-frequent-first",
          "product-DAG compiler did not record its factor ordering");
  require(plan.monomial_nodes.size() == plan.monomial_coefficients.size(),
          "compiled product-DAG endpoints are incomplete");
  if (plan.binary_dag) {
    require(plan.binary_node_left.size() == plan.binary_node_right.size(),
            "compiled binary product-DAG metadata is incomplete");
    const std::int64_t power_count = static_cast<std::int64_t>(plan.power_channels.size());
    for (std::size_t node = 0; node < plan.binary_node_left.size(); ++node) {
      const std::int64_t limit = power_count + static_cast<std::int64_t>(node);
      require(plan.binary_node_left[node] >= 0 && plan.binary_node_left[node] < limit &&
                  plan.binary_node_right[node] >= 0 && plan.binary_node_right[node] < limit,
              "compiled binary product-DAG is not topological");
    }
  } else {
    require(plan.dag_node_parents.size() == plan.dag_node_powers.size(),
            "compiled prefix product-DAG metadata is incomplete");
    require(plan.dag_root_node_count > 0 &&
                plan.dag_root_node_count < static_cast<std::int64_t>(plan.dag_node_parents.size()),
            "compiled product-DAG root partition is empty or invalid");
    for (std::int64_t node = 1; node <= plan.dag_root_node_count; ++node)
      require(plan.dag_node_parents[static_cast<std::size_t>(node)] == 0,
              "compiled product-DAG root partition is not contiguous");
    for (std::int64_t node = plan.dag_root_node_count + 1;
         node < static_cast<std::int64_t>(plan.dag_node_parents.size()); ++node) {
      const std::int64_t parent = plan.dag_node_parents[static_cast<std::size_t>(node)];
      require(parent > 0 && parent < node,
              "compiled product-DAG non-root partition is not topological");
    }
  }
  const auto &bond = model.bond(0, 0);
  const double minimum_model_storage = plan.monomial_coefficients.size() * sizeof(double) +
      (plan.dag_node_parents.size() + plan.binary_node_left.size() +
       plan.binary_node_right.size()) *
          sizeof(std::int64_t) +
      bond.radial_base_spline.size() * sizeof(double) +
      bond.contracted_spline.size() * sizeof(double);
  require(model.memory_usage() >= minimum_model_storage,
          "model memory accounting omits loaded plan or spline storage");

  const auto environment = load_environment(environment_path);
  verify_native_source_tiling(model, environment);
  const auto vectors = edge_vectors(environment);
  YE3T_LAMMPS::YE3TCPUEvaluator evaluator(&model);
  double energy = 0.0;
  std::vector<double> gradients;
  evaluate(evaluator, environment, vectors, energy, gradients);

  const double expected_energy = 3.97640596846674410e-01;
  require(std::abs(energy - expected_energy) <= 3.0e-13,
          "fixed-environment atomic energy mismatch");
  const std::vector<double> expected_gradients{
      2.05778384562388084e+00,
      -4.08320441381867683e-01,
      -1.13384907597649673e-01,
      -2.16380800087566161e-01,
      1.54787484453499491e-01,
      -2.13976937880321592e-02,
      -2.69575327030088102e-02,
      -9.60006687478199422e-03,
      -3.36821804261611749e-02,
      1.61784891641783196e-01,
      -4.50789838142715091e-03,
      -1.25178068408213130e-03,
      -8.34416198163131216e-08,
      2.37893067050234404e-04,
      -8.25146329657860333e-09,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0,
      -2.10579162754626417e-02,
      -1.52804850686840160e-01,
      -2.08239753293277749e-03,
  };
  require(gradients.size() == expected_gradients.size(),
          "fixed-environment gradient size mismatch");
  for (std::size_t index = 0; index < gradients.size(); ++index) {
    const double tolerance = 3.0e-11 * (1.0 + std::abs(expected_gradients[index]));
    require(std::abs(gradients[index] - expected_gradients[index]) <= tolerance,
            "fixed-environment edge gradient mismatch at index " + std::to_string(index));
  }
  require(evaluator.maximum_imaginary_density() <= 3.0e-13,
          "invariant density has a material imaginary component");

  const double step = 1.0e-6;
  for (std::size_t edge = 0; edge < environment.neighbors.size(); ++edge) {
    const double radius = std::sqrt(vectors[edge * 3] * vectors[edge * 3] +
                                    vectors[edge * 3 + 1] * vectors[edge * 3 + 1] +
                                    vectors[edge * 3 + 2] * vectors[edge * 3 + 2]);
    const double first_spline_radius =
        bond.cutoff / static_cast<double>(bond.spline_interval_count);
    if (radius <= first_spline_radius + 2.0e-3 || radius >= model.maximum_cutoff() - 2.0e-3)
      continue;
    for (int axis = 0; axis < 3; ++axis) {
      auto plus = vectors;
      auto minus = vectors;
      plus[edge * 3 + static_cast<std::size_t>(axis)] += step;
      minus[edge * 3 + static_cast<std::size_t>(axis)] -= step;
      double plus_energy = 0.0;
      double minus_energy = 0.0;
      std::vector<double> scratch;
      evaluate(evaluator, environment, plus, plus_energy, scratch);
      evaluate(evaluator, environment, minus, minus_energy, scratch);
      const double finite_difference = (plus_energy - minus_energy) / (2.0 * step);
      const double analytic = gradients[edge * 3 + static_cast<std::size_t>(axis)];
      const double tolerance = 2.0e-6 * (1.0 + std::abs(analytic));
      require(std::abs(analytic - finite_difference) <= tolerance,
              "fixed-environment finite-difference mismatch");
    }
  }

  Environment reversed = environment;
  std::reverse(reversed.neighbors.begin(), reversed.neighbors.end());
  const auto reversed_vectors = edge_vectors(reversed);
  double reversed_energy = 0.0;
  std::vector<double> reversed_gradients;
  evaluate(evaluator, reversed, reversed_vectors, reversed_energy, reversed_gradients);
  require(std::abs(reversed_energy - energy) <= 3.0e-13,
          "neighbor reordering changed the atomic energy");
  for (std::size_t edge = 0; edge < environment.neighbors.size(); ++edge) {
    const std::size_t reversed_edge = environment.neighbors.size() - edge - 1;
    for (int axis = 0; axis < 3; ++axis)
      require(std::abs(gradients[edge * 3 + static_cast<std::size_t>(axis)] -
                       reversed_gradients[reversed_edge * 3 + static_cast<std::size_t>(axis)]) <=
                  3.0e-12,
              "neighbor reordering changed an edge gradient");
  }
}

void run_block_comparison(const std::string &model_path, const std::string &environment_path,
                          const std::string &manifest_path, std::size_t expected_route_count,
                          std::size_t expected_power_plan_count, bool require_rank8_sentinels,
                          bool expect_automatic, bool expect_scalar, bool gpu_automatic,
                          std::size_t expected_candidate_count,
                          const std::string &reordered_manifest_path)
{
  const auto direct_model = YE3T_LAMMPS::YACEModel::load(model_path);
  const std::size_t catalogue_function_count =
      direct_model.species(0).polynomial.descriptor_offsets.size() - 1;
  const auto block_model =
      YE3T_LAMMPS::YACEModel::load(model_path, manifest_path, YE3T_LAMMPS::YACEBlockPolicy::BLOCK);
  std::unique_ptr<YE3T_LAMMPS::YACEModel> scalar_model;
  if (expect_scalar)
    scalar_model = std::make_unique<YE3T_LAMMPS::YACEModel>(YE3T_LAMMPS::YACEModel::load(
        model_path, manifest_path, YE3T_LAMMPS::YACEBlockPolicy::SCALAR_POWER));
  std::unique_ptr<YE3T_LAMMPS::YACEModel> automatic_model;
  bool automatic_rejected = false;
  try {
    const auto automatic_policy = gpu_automatic ? YE3T_LAMMPS::YACEBlockPolicy::GPU_AUTO
                                                : YE3T_LAMMPS::YACEBlockPolicy::AUTO;
    automatic_model = std::make_unique<YE3T_LAMMPS::YACEModel>(
        YE3T_LAMMPS::YACEModel::load(model_path, manifest_path, automatic_policy));
  } catch (const std::runtime_error &error) {
    automatic_rejected =
        std::string(error.what()).find("explicit alternative routes") != std::string::npos;
  }
  if (expect_automatic) {
    require(automatic_model != nullptr && !automatic_rejected,
            "v2 sidecar did not enable automatic portfolio dispatch");
    const auto &automatic_program = automatic_model->species(0).block_program;
    require(automatic_program.candidate_route_count ==
                static_cast<std::int64_t>(expected_candidate_count),
            "automatic planner saw an unexpected candidate count");
    require(automatic_program.catalogue_function_count ==
                    static_cast<std::int64_t>(catalogue_function_count) &&
                automatic_program.decisions.size() == catalogue_function_count,
            "automatic planner did not retain one decision per catalogue "
            "function");
    require(automatic_program.selected_operation_estimate <=
                automatic_program.direct_operation_estimate,
            "automatic planner selected a portfolio costlier than direct");
    require(automatic_program.evaluator_plan_hash.size() == 64,
            "automatic planner did not record its evaluator-plan hash");
    if (gpu_automatic) {
      require(automatic_program.planner_profile == "kokkos_gpu_conservative_direct_v1" &&
                  automatic_program.planner_algorithm == "conservative_direct_fallback_v1" &&
                  automatic_program.planner_status == "selected_direct_no_authorized_profile" &&
                  automatic_program.planner_calibration_hash == "not_applicable" &&
                  automatic_program.planner_decision_reason ==
                      "no_authorized_non_direct_profile_match" &&
                  !automatic_program.planner_optimal &&
                  automatic_program.planner_score_evaluations == 1 &&
                  automatic_program.selected_operation_estimate ==
                      automatic_program.direct_operation_estimate,
              "GPU automatic planner did not retain its conservative direct "
              "fallback");
      require(std::all_of(automatic_program.decisions.begin(), automatic_program.decisions.end(),
                          [](const auto &decision) {
                            return decision.selected_evaluator ==
                                YE3T_LAMMPS::YACEEvaluatorKind::EXPLICIT_CTILDE;
                          }) &&
                  automatic_program.routes.empty() && automatic_program.power_plans.empty() &&
                  automatic_program.scalar_program.routes.empty() &&
                  automatic_program.coupled_product_plans.empty(),
              "GPU automatic planner lowered work outside its authorized direct "
              "candidate vector");
      const auto cpu_automatic_model = YE3T_LAMMPS::YACEModel::load(
          model_path, manifest_path, YE3T_LAMMPS::YACEBlockPolicy::AUTO);
      require(cpu_automatic_model.species(0).block_program.evaluator_plan_hash !=
                  automatic_program.evaluator_plan_hash,
              "CPU and GPU automatic selections have the same semantic "
              "identity");
    } else if (expected_route_count <= 12) {
      require(automatic_program.planner_algorithm == "exhaustive_candidate_vector_catalogue_v2" &&
                  automatic_program.planner_optimal,
              "small automatic portfolio was not selected exhaustively");
    }
    const auto repeated_automatic_model =
        YE3T_LAMMPS::YACEModel::load(model_path, manifest_path,
                                     gpu_automatic ? YE3T_LAMMPS::YACEBlockPolicy::GPU_AUTO
                                                   : YE3T_LAMMPS::YACEBlockPolicy::AUTO);
    require(repeated_automatic_model.species(0).block_program.evaluator_plan_hash ==
                automatic_program.evaluator_plan_hash,
            "automatic portfolio selection was not deterministic");
    if (!reordered_manifest_path.empty()) {
      const auto reordered_model =
          YE3T_LAMMPS::YACEModel::load(model_path, reordered_manifest_path,
                                       gpu_automatic ? YE3T_LAMMPS::YACEBlockPolicy::GPU_AUTO
                                                     : YE3T_LAMMPS::YACEBlockPolicy::AUTO);
      const auto &reordered_program = reordered_model.species(0).block_program;
      require(reordered_program.evaluator_plan_hash == automatic_program.evaluator_plan_hash &&
                  reordered_program.selected_operation_estimate ==
                      automatic_program.selected_operation_estimate &&
                  reordered_program.decisions.size() == automatic_program.decisions.size(),
              "candidate order changed the compiled evaluator portfolio");
      for (std::size_t index = 0; index < automatic_program.decisions.size(); ++index)
        require(reordered_program.decisions[index].selected_candidate_id ==
                    automatic_program.decisions[index].selected_candidate_id,
                "candidate order changed a selected evaluator ID");
    }
    if (expected_candidate_count != expected_route_count) {
      std::cout << "automatic planner: " << automatic_program.planner_algorithm << " status "
                << automatic_program.planner_status << " scores "
                << automatic_program.planner_score_evaluations << " selected operations "
                << automatic_program.selected_operation_estimate << '\n';
      for (const auto &decision : automatic_program.decisions)
        if (decision.candidates.size() > 1)
          std::cout << "automatic decision " << decision.function_index << " candidate "
                    << decision.selected_candidate_id << '\n';
    }
    if (expect_scalar && !gpu_automatic) {
      require(automatic_program.scalar_program.routes.size() == expected_route_count,
              "automatic planner did not select every scalar-power route");
      require(automatic_program.routes.empty(),
              "automatic planner retained a slower generic block route");
    }
  } else {
    require(automatic_rejected, "v1 sidecar unexpectedly allowed automatic portfolio dispatch");
  }
  require(direct_model.species_count() == block_model.species_count(),
          "block sidecar changed the species count");
  const auto &direct_species = direct_model.species(0);
  const auto &block_species = block_model.species(0);
  if (expect_scalar) {
    const auto &scalar_species = scalar_model->species(0);
    const auto &scalar_program = scalar_species.block_program.scalar_program;
    require(scalar_program.bases.size() == 1, "scalar dispatch did not share its quadratic base");
    require(scalar_program.nodes.size() == expected_route_count + 2,
            "scalar dispatch did not share the expected addition-chain nodes");
    require(scalar_program.routes.size() == expected_route_count,
            "scalar dispatch selected an unexpected route count");
    require(scalar_species.block_program.routes.empty() &&
                scalar_species.block_program.power_plans.empty(),
            "forced scalar dispatch retained generic block work");
    require(std::all_of(scalar_species.block_program.decisions.begin(),
                        scalar_species.block_program.decisions.end(),
                        [](const auto &decision) {
                          const bool scalar_available = std::any_of(
                              decision.candidates.begin(), decision.candidates.end(),
                              [](const auto &candidate) {
                                return candidate.evaluator ==
                                    YE3T_LAMMPS::YACEEvaluatorKind::SCALAR_INVARIANT_POWER;
                              });
                          return scalar_available ? decision.selected_evaluator ==
                                  YE3T_LAMMPS::YACEEvaluatorKind::SCALAR_INVARIANT_POWER
                                                  : decision.selected_evaluator ==
                                  YE3T_LAMMPS::YACEEvaluatorKind::EXPLICIT_CTILDE;
                        }),
            "forced scalar dispatch did not select every certified scalar "
            "candidate or preserve direct-only rows");
    if (catalogue_function_count == expected_route_count)
      require(scalar_species.polynomial.monomial_coefficients.empty(),
              "forced scalar dispatch retained mapped C-tilde monomials");
    else
      require(!scalar_species.polynomial.monomial_coefficients.empty() &&
                  scalar_species.block_program.dispatch == "cpu_ye3t_direct_scalar_portfolio_v1",
              "mixed scalar catalogue lost direct work or reported a false "
              "pure-scalar dispatch");
  }
  require(block_species.block_program.routes.size() == expected_route_count,
          "forced block dispatch selected an unexpected number of fitted rows");
  require(expected_power_plan_count == 0 ||
              block_species.block_program.power_plans.size() == expected_power_plan_count,
          "fitted block dispatch compiled an unexpected number of unique blocks");
  require(!block_species.block_program.power_plans.empty(),
          "fitted block dispatch did not compile any block plans");
  const std::size_t direct_input_plan_count = static_cast<std::size_t>(
      std::count_if(block_species.block_program.power_plans.begin(),
                    block_species.block_program.power_plans.end(), [](const auto &plan) {
                      return plan.direct_input_plan;
                    }));
  if (expected_route_count == 58)
    require(direct_input_plan_count == 15,
            "full fitted block map did not lower every compiler-certified "
            "power-1 plan: expected 15, received " +
                std::to_string(direct_input_plan_count));
  if (require_rank8_sentinels)
    require(std::any_of(block_species.block_program.routes.begin(),
                        block_species.block_program.routes.end(),
                        [](const auto &route) {
                          return route.function_index == 56;
                        }) &&
                std::any_of(block_species.block_program.routes.begin(),
                            block_species.block_program.routes.end(),
                            [](const auto &route) {
                              return route.function_index == 57;
                            }),
            "fitted block dispatch omitted a required rank-8 sentinel route");
  require(std::all_of(block_species.block_program.power_plans.begin(),
                      block_species.block_program.power_plans.end(),
                      [](const auto &plan) {
                        return plan.real_coefficients;
                      }) &&
              std::all_of(block_species.block_program.routes.begin(),
                          block_species.block_program.routes.end(),
                          [](const auto &route) {
                            return route.real_coefficients;
                          }),
          "fitted block plan did not enable its exact-real coefficient "
          "specialization");
  require(std::all_of(block_species.block_program.power_plans.begin(),
                      block_species.block_program.power_plans.end(),
                      [](const auto &plan) {
                        return plan.output_L == 0 || plan.conjugate_half_output;
                      }),
          "fitted block plan did not enable its certified conjugate-half "
          "specialization");
  require(block_species.polynomial.monomial_coefficients.size() <
              direct_species.polynomial.monomial_coefficients.size(),
          "block dispatch did not remove mapped C-tilde monomials");
  if (expected_candidate_count != expected_route_count) {
    require(block_species.block_program.candidate_function_count == 4 &&
                block_species.block_program.candidate_count ==
                    static_cast<std::int64_t>(catalogue_function_count + expected_candidate_count),
            "candidate-vector fixture did not retain its full row portfolio");
    for (const auto &[function, count] : std::array<std::pair<std::int64_t, std::size_t>, 4>{
             std::pair<std::int64_t, std::size_t>{34, 2}, {55, 3}, {56, 3}, {57, 3}}) {
      const auto decision = std::find_if(block_species.block_program.decisions.begin(),
                                         block_species.block_program.decisions.end(),
                                         [function](const auto &candidate) {
                                           return candidate.function_index == function;
                                         });
      require(decision != block_species.block_program.decisions.end() &&
                  decision->candidates.size() == count,
              "candidate-vector fixture has the wrong per-row cardinality");
    }
    require(std::count_if(block_species.block_program.routes.begin(),
                          block_species.block_program.routes.end(),
                          [](const auto &route) {
                            return route.function_index == 55;
                          }) == 2,
            "forced block policy did not select the two-term [7,1] readout");
  }

  const auto environment = load_environment(environment_path);
  const auto vectors = edge_vectors(environment);
  YE3T_LAMMPS::YE3TCPUEvaluator direct_evaluator(&direct_model);
  YE3T_LAMMPS::YE3TCPUEvaluator block_evaluator(&block_model);
  std::unique_ptr<YE3T_LAMMPS::YE3TCPUEvaluator> scalar_evaluator;
  if (scalar_model != nullptr)
    scalar_evaluator = std::make_unique<YE3T_LAMMPS::YE3TCPUEvaluator>(scalar_model.get());
  std::unique_ptr<YE3T_LAMMPS::YE3TCPUEvaluator> automatic_evaluator;
  if (automatic_model != nullptr)
    automatic_evaluator = std::make_unique<YE3T_LAMMPS::YE3TCPUEvaluator>(automatic_model.get());
  double direct_energy = 0.0;
  double block_energy = 0.0;
  double automatic_energy = 0.0;
  double scalar_energy = 0.0;
  std::vector<double> direct_gradients;
  std::vector<double> block_gradients;
  std::vector<double> automatic_gradients;
  std::vector<double> scalar_gradients;
  evaluate(direct_evaluator, environment, vectors, direct_energy, direct_gradients);
  const double maximum_direct_gradient = *std::max_element(
      direct_gradients.begin(), direct_gradients.end(), [](double left, double right) {
        return std::abs(left) < std::abs(right);
      });
  require(std::abs(direct_energy) >= 1.0e-4 && std::abs(maximum_direct_gradient) >= 1.0e-4,
          "route comparison fixture has insufficient energy or gradient signal");
  evaluate(block_evaluator, environment, vectors, block_energy, block_gradients);
  if (scalar_evaluator != nullptr)
    evaluate(*scalar_evaluator, environment, vectors, scalar_energy, scalar_gradients);
  const double energy_residual = std::abs(block_energy - direct_energy);
  double scalar_energy_residual = 0.0;
  double scalar_gradient_residual = 0.0;
  require(std::abs(block_energy - direct_energy) <= 2.0e-10 * (1.0 + std::abs(direct_energy)),
          "block and direct atomic energies disagree");
  require(block_gradients.size() == direct_gradients.size(),
          "block and direct gradient widths disagree");
  double gradient_residual = 0.0;
  for (std::size_t index = 0; index < direct_gradients.size(); ++index) {
    gradient_residual =
        std::max(gradient_residual, std::abs(block_gradients[index] - direct_gradients[index]));
    const double tolerance = 2.0e-9 * (1.0 + std::abs(direct_gradients[index]));
    require(std::abs(block_gradients[index] - direct_gradients[index]) <= tolerance,
            "block and direct edge gradients disagree at index " + std::to_string(index));
  }
  if (scalar_evaluator != nullptr) {
    scalar_energy_residual = std::abs(scalar_energy - direct_energy);
    require(std::abs(scalar_energy - direct_energy) <= 2.0e-10 * (1.0 + std::abs(direct_energy)),
            "scalar and direct atomic energies disagree");
    require(scalar_gradients.size() == direct_gradients.size(),
            "scalar and direct gradient widths disagree");
    for (std::size_t index = 0; index < direct_gradients.size(); ++index) {
      scalar_gradient_residual = std::max(
          scalar_gradient_residual, std::abs(scalar_gradients[index] - direct_gradients[index]));
      require(std::abs(scalar_gradients[index] - direct_gradients[index]) <=
                  2.0e-9 * (1.0 + std::abs(direct_gradients[index])),
              "scalar and direct edge gradients disagree at index " + std::to_string(index));
    }
  }
  if (automatic_model != nullptr) {
    evaluate(*automatic_evaluator, environment, vectors, automatic_energy, automatic_gradients);
    require(std::abs(automatic_energy - direct_energy) <= 2.0e-10 * (1.0 + std::abs(direct_energy)),
            "automatic and direct atomic energies disagree");
    require(automatic_gradients.size() == direct_gradients.size(),
            "automatic and direct gradient widths disagree");
    for (std::size_t index = 0; index < direct_gradients.size(); ++index)
      require(std::abs(automatic_gradients[index] - direct_gradients[index]) <=
                  2.0e-9 * (1.0 + std::abs(direct_gradients[index])),
              "automatic and direct edge gradients disagree at index " + std::to_string(index));
  }

  for (const int atom_count : {1, 7, 8, 9, 16, 17}) {
    std::vector<double> direct_batch_energies;
    std::vector<double> block_batch_energies;
    std::vector<double> automatic_batch_energies;
    std::vector<double> scalar_batch_energies;
    std::vector<double> direct_batch_gradients;
    std::vector<double> block_batch_gradients;
    std::vector<double> automatic_batch_gradients;
    std::vector<double> scalar_batch_gradients;
    evaluate_repeated(direct_evaluator, environment, atom_count, direct_batch_energies,
                      direct_batch_gradients);
    evaluate_repeated(block_evaluator, environment, atom_count, block_batch_energies,
                      block_batch_gradients);
    if (automatic_model != nullptr) {
      evaluate_repeated(*automatic_evaluator, environment, atom_count, automatic_batch_energies,
                        automatic_batch_gradients);
    }
    if (scalar_evaluator != nullptr)
      evaluate_repeated(*scalar_evaluator, environment, atom_count, scalar_batch_energies,
                        scalar_batch_gradients);
    require(direct_batch_energies.size() == block_batch_energies.size() &&
                direct_batch_gradients.size() == block_batch_gradients.size(),
            "tiled block comparison returned inconsistent batch dimensions");
    for (std::size_t index = 0; index < direct_batch_energies.size(); ++index)
      require(std::abs(block_batch_energies[index] - direct_batch_energies[index]) <=
                  2.0e-10 * (1.0 + std::abs(direct_batch_energies[index])),
              "tiled block and direct batch energies disagree");
    for (std::size_t index = 0; index < direct_batch_gradients.size(); ++index)
      require(std::abs(block_batch_gradients[index] - direct_batch_gradients[index]) <=
                  2.0e-9 * (1.0 + std::abs(direct_batch_gradients[index])),
              "tiled block and direct batch gradients disagree");
    if (automatic_model != nullptr) {
      require(automatic_batch_energies.size() == direct_batch_energies.size() &&
                  automatic_batch_gradients.size() == direct_batch_gradients.size(),
              "automatic tiled comparison returned inconsistent dimensions");
      for (std::size_t index = 0; index < direct_batch_energies.size(); ++index)
        require(std::abs(automatic_batch_energies[index] - direct_batch_energies[index]) <=
                    2.0e-10 * (1.0 + std::abs(direct_batch_energies[index])),
                "automatic and direct batch energies disagree");
      for (std::size_t index = 0; index < direct_batch_gradients.size(); ++index)
        require(std::abs(automatic_batch_gradients[index] - direct_batch_gradients[index]) <=
                    2.0e-9 * (1.0 + std::abs(direct_batch_gradients[index])),
                "automatic and direct batch gradients disagree");
    }
    if (scalar_evaluator != nullptr) {
      require(scalar_batch_energies.size() == direct_batch_energies.size() &&
                  scalar_batch_gradients.size() == direct_batch_gradients.size(),
              "scalar tiled comparison returned inconsistent dimensions");
      for (std::size_t index = 0; index < direct_batch_energies.size(); ++index)
        require(std::abs(scalar_batch_energies[index] - direct_batch_energies[index]) <=
                    2.0e-10 * (1.0 + std::abs(direct_batch_energies[index])),
                "scalar and direct batch energies disagree");
      for (std::size_t index = 0; index < direct_batch_gradients.size(); ++index)
        require(std::abs(scalar_batch_gradients[index] - direct_batch_gradients[index]) <=
                    2.0e-9 * (1.0 + std::abs(direct_batch_gradients[index])),
                "scalar and direct batch gradients disagree");
    }
  }

  Environment zero_environment = environment;
  zero_environment.neighbors.clear();
  std::vector<double> direct_zero_energies;
  std::vector<double> block_zero_energies;
  std::vector<double> direct_zero_gradients;
  std::vector<double> block_zero_gradients;
  std::vector<double> automatic_zero_energies;
  std::vector<double> automatic_zero_gradients;
  std::vector<double> scalar_zero_energies;
  std::vector<double> scalar_zero_gradients;
  evaluate_repeated(direct_evaluator, zero_environment, 9, direct_zero_energies,
                    direct_zero_gradients);
  evaluate_repeated(block_evaluator, zero_environment, 9, block_zero_energies,
                    block_zero_gradients);
  if (automatic_model != nullptr) {
    evaluate_repeated(*automatic_evaluator, zero_environment, 9, automatic_zero_energies,
                      automatic_zero_gradients);
  }
  if (scalar_evaluator != nullptr)
    evaluate_repeated(*scalar_evaluator, zero_environment, 9, scalar_zero_energies,
                      scalar_zero_gradients);
  require(direct_zero_gradients.empty() && block_zero_gradients.empty(),
          "zero-neighbor block comparison returned edge gradients");
  for (std::size_t index = 0; index < direct_zero_energies.size(); ++index)
    require(std::abs(block_zero_energies[index] - direct_zero_energies[index]) <= 2.0e-10,
            "zero-neighbor tiled block and direct energies disagree");
  if (automatic_model != nullptr) {
    require(automatic_zero_gradients.empty(),
            "zero-neighbor automatic comparison returned edge gradients");
    for (std::size_t index = 0; index < direct_zero_energies.size(); ++index)
      require(std::abs(automatic_zero_energies[index] - direct_zero_energies[index]) <= 2.0e-10,
              "zero-neighbor automatic and direct energies disagree");
  }
  if (scalar_evaluator != nullptr) {
    require(scalar_zero_gradients.empty(),
            "zero-neighbor scalar comparison returned edge gradients");
    for (std::size_t index = 0; index < direct_zero_energies.size(); ++index)
      require(std::abs(scalar_zero_energies[index] - direct_zero_energies[index]) <= 2.0e-10,
              "zero-neighbor scalar and direct energies disagree");
  }

  const double step = 1.0e-6;
  double finite_difference_residual = 0.0;
  double scalar_finite_difference_residual = 0.0;
  for (std::size_t edge = 0; edge < environment.neighbors.size(); ++edge) {
    for (int axis = 0; axis < 3; ++axis) {
      auto plus = vectors;
      auto minus = vectors;
      plus[edge * 3 + static_cast<std::size_t>(axis)] += step;
      minus[edge * 3 + static_cast<std::size_t>(axis)] -= step;
      double plus_energy = 0.0;
      double minus_energy = 0.0;
      std::vector<double> scratch;
      evaluate(block_evaluator, environment, plus, plus_energy, scratch);
      evaluate(block_evaluator, environment, minus, minus_energy, scratch);
      const double finite_difference = (plus_energy - minus_energy) / (2.0 * step);
      const double analytic = block_gradients[edge * 3 + static_cast<std::size_t>(axis)];
      finite_difference_residual =
          std::max(finite_difference_residual, std::abs(analytic - finite_difference));
      require(std::abs(analytic - finite_difference) <= 3.0e-6 * (1.0 + std::abs(analytic)),
              "block edge gradient finite-difference mismatch");
      if (automatic_model != nullptr) {
        double automatic_plus_energy = 0.0;
        double automatic_minus_energy = 0.0;
        evaluate(*automatic_evaluator, environment, plus, automatic_plus_energy, scratch);
        evaluate(*automatic_evaluator, environment, minus, automatic_minus_energy, scratch);
        const double automatic_finite_difference =
            (automatic_plus_energy - automatic_minus_energy) / (2.0 * step);
        const double automatic_analytic =
            automatic_gradients[edge * 3 + static_cast<std::size_t>(axis)];
        require(std::abs(automatic_analytic - automatic_finite_difference) <=
                    3.0e-6 * (1.0 + std::abs(automatic_analytic)),
                "automatic edge gradient finite-difference mismatch");
      }
      if (scalar_evaluator != nullptr) {
        double scalar_plus_energy = 0.0;
        double scalar_minus_energy = 0.0;
        evaluate(*scalar_evaluator, environment, plus, scalar_plus_energy, scratch);
        evaluate(*scalar_evaluator, environment, minus, scalar_minus_energy, scratch);
        const double scalar_finite_difference =
            (scalar_plus_energy - scalar_minus_energy) / (2.0 * step);
        const double scalar_analytic = scalar_gradients[edge * 3 + static_cast<std::size_t>(axis)];
        scalar_finite_difference_residual =
            std::max(scalar_finite_difference_residual,
                     std::abs(scalar_analytic - scalar_finite_difference));
        require(std::abs(scalar_analytic - scalar_finite_difference) <=
                    3.0e-6 * (1.0 + std::abs(scalar_analytic)),
                "scalar edge gradient finite-difference mismatch");
      }
    }
  }
  std::cout << "block route count: " << block_species.block_program.routes.size() << '\n'
            << "unique block-plan count: " << block_species.block_program.power_plans.size() << '\n'
            << "direct-input block-plan count: " << direct_input_plan_count << '\n'
            << "direct operation estimate: "
            << block_species.block_program.direct_operation_estimate << '\n'
            << "selected operation estimate: "
            << block_species.block_program.selected_operation_estimate << '\n';
  std::cout << "direct atomic energy: " << direct_energy << '\n'
            << "maximum direct edge-gradient magnitude: " << std::abs(maximum_direct_gradient)
            << '\n';
  for (const auto &route : block_species.block_program.routes)
    std::cout << "route " << route.function_index << " direct " << route.direct_operation_estimate
              << " block " << route.block_operation_estimate << '\n';
  std::cout << "block/direct maximum atomic-energy residual: " << energy_residual << '\n'
            << "block/direct maximum edge-gradient residual: " << gradient_residual << '\n'
            << "block finite-difference maximum residual: " << finite_difference_residual << '\n';
  if (scalar_evaluator != nullptr)
    std::cout << "scalar/direct maximum atomic-energy residual: " << scalar_energy_residual << '\n'
              << "scalar/direct maximum edge-gradient residual: " << scalar_gradient_residual
              << '\n'
              << "scalar finite-difference maximum residual: " << scalar_finite_difference_residual
              << '\n';
}

void run_coupled_product_comparison(const std::string &model_path,
                                    const std::string &environment_path,
                                    const std::string &manifest_path)
{
  const auto direct_model = YE3T_LAMMPS::YACEModel::load(model_path);
  const auto coupled_model = YE3T_LAMMPS::YACEModel::load(
      model_path, manifest_path, YE3T_LAMMPS::YACEBlockPolicy::COUPLED_PRODUCT);
  const auto automatic_model =
      YE3T_LAMMPS::YACEModel::load(model_path, manifest_path, YE3T_LAMMPS::YACEBlockPolicy::AUTO);
  require(direct_model.species_count() == 1 && coupled_model.species_count() == 1 &&
              automatic_model.species_count() == 1,
          "coupled-product fixture has an unexpected species count");

  const auto &direct_species = direct_model.species(0);
  const auto &coupled_species = coupled_model.species(0);
  const auto &coupled_program = coupled_species.block_program;
  const auto &automatic_program = automatic_model.species(0).block_program;
  const std::size_t function_count = direct_species.polynomial.descriptor_offsets.size() - 1;
  require(function_count == 61, "coupled-product fixture does not use the full L8 catalogue");
  require(coupled_program.coupled_product_plans.size() == 1 && coupled_program.routes.empty() &&
              coupled_program.power_plans.empty() && coupled_program.scalar_program.bases.empty() &&
              coupled_program.scalar_program.nodes.empty() &&
              coupled_program.scalar_program.routes.empty() &&
              coupled_program.scalar_program.value_count == 0,
          "forced coupled-product dispatch retained unrelated fast-path work");
  require(
      coupled_program.dispatch == "cpu_ye3t_direct_coupled_product_portfolio_v1" &&
          coupled_program.catalogue_function_count == static_cast<std::int64_t>(function_count) &&
          coupled_program.candidate_count == static_cast<std::int64_t>(function_count + 1) &&
          coupled_program.candidate_route_count == 1 &&
          coupled_program.candidate_function_count == 1 &&
          coupled_program.decisions.size() == function_count &&
          coupled_program.planner_profile == "cpu_structural_coupled_product_split_real_tile8_v1" &&
          coupled_program.planner_algorithm == "forced_candidate_vector_catalogue_v2" &&
          coupled_program.planner_status == "selected" && coupled_program.planner_optimal &&
          coupled_program.evaluator_plan_hash.size() == 64,
      "forced coupled-product portfolio metadata is inconsistent");
  require(!coupled_species.polynomial.monomial_coefficients.empty() &&
              coupled_species.polynomial.monomial_coefficients.size() <
                  direct_species.polynomial.monomial_coefficients.size(),
          "forced coupled-product dispatch did not retain direct fallbacks or "
          "remove its mapped C-tilde row");

  const auto &plan = coupled_program.coupled_product_plans.front();
  require(plan.function_index == 38 && plan.operation_estimate == 44 &&
              plan.total_node_components == 15 &&
              plan.node_offsets == std::vector<std::int64_t>({0, 3, 6, 11, 14}) &&
              plan.node_dimensions == std::vector<std::int64_t>({3, 3, 5, 3, 1}),
          "coupled-product plan does not target the certified Ta descriptor");
  const std::size_t node_count = plan.node_offsets.size();
  require(node_count > 0 && plan.node_dimensions.size() == node_count &&
              plan.node_leaf_offsets.size() == node_count &&
              plan.node_coefficient_offsets.size() == node_count + 1 &&
              plan.node_coefficient_offsets.front() == 0 &&
              plan.node_coefficient_offsets.back() ==
                  static_cast<std::int64_t>(plan.coefficient_values.size()) &&
              plan.coefficient_left_components.size() == plan.coefficient_values.size() &&
              plan.coefficient_right_components.size() == plan.coefficient_values.size() &&
              plan.coefficient_output_components.size() == plan.coefficient_values.size() &&
              !plan.readout_components.empty() &&
              plan.readout_components.size() == plan.readout_coefficients.size(),
          "coupled-product native lowering has inconsistent packed arrays");
  std::int64_t next_component = 0;
  for (std::size_t node = 0; node < node_count; ++node) {
    const std::int64_t offset = plan.node_offsets[node];
    const std::int64_t dimension = plan.node_dimensions[node];
    require(offset == next_component && dimension > 0 && dimension % 2 == 1,
            "coupled-product node storage is not compact and contiguous");
    next_component += dimension;
    const std::int64_t begin = plan.node_coefficient_offsets[node];
    const std::int64_t end = plan.node_coefficient_offsets[node + 1];
    require(begin >= 0 && begin <= end &&
                end <= static_cast<std::int64_t>(plan.coefficient_values.size()),
            "coupled-product node coefficient range is invalid");
    if (plan.node_leaf_offsets[node] >= 0) {
      require(begin == end &&
                  plan.node_leaf_offsets[node] + dimension <=
                      static_cast<std::int64_t>(plan.leaf_input_components.size()),
              "coupled-product primitive node binding is invalid");
    } else {
      require(begin < end, "coupled-product product node is empty");
      for (std::int64_t term = begin; term < end; ++term) {
        const std::size_t index = static_cast<std::size_t>(term);
        require(plan.coefficient_left_components[index] >= 0 &&
                    plan.coefficient_left_components[index] < offset &&
                    plan.coefficient_right_components[index] >= 0 &&
                    plan.coefficient_right_components[index] < offset &&
                    plan.coefficient_output_components[index] >= offset &&
                    plan.coefficient_output_components[index] < offset + dimension,
                "coupled-product coefficients violate topological storage");
      }
    }
  }
  require(next_component == plan.total_node_components,
          "coupled-product total component count is inconsistent");
  require(std::all_of(plan.leaf_input_components.begin(), plan.leaf_input_components.end(),
                      [&](std::int64_t component) {
                        return component >= 0 &&
                            component < static_cast<std::int64_t>(direct_species.channels.size());
                      }),
          "coupled-product primitive input lies outside the A basis");
  require(std::all_of(plan.readout_components.begin(), plan.readout_components.end(),
                      [&](std::int64_t component) {
                        return component >= 0 && component < plan.total_node_components;
                      }),
          "coupled-product readout lies outside node storage");

  const auto forced_decision = std::find_if(
      coupled_program.decisions.begin(), coupled_program.decisions.end(), [](const auto &decision) {
        return decision.function_index == 38;
      });
  require(
      forced_decision != coupled_program.decisions.end() &&
          forced_decision->candidates.size() == 2 &&
          forced_decision->selected_candidate_index >= 0 &&
          static_cast<std::size_t>(forced_decision->selected_candidate_index) <
              forced_decision->candidates.size() &&
          forced_decision
                  ->candidates[static_cast<std::size_t>(forced_decision->selected_candidate_index)]
                  .candidate_id == forced_decision->selected_candidate_id &&
          forced_decision->selected_evaluator ==
              YE3T_LAMMPS::YACEEvaluatorKind::COUPLED_PRODUCT_DAG,
      "forced policy did not select the certified coupled-product DAG");
  require(std::all_of(coupled_program.decisions.begin(), coupled_program.decisions.end(),
                      [](const auto &decision) {
                        return decision.function_index == 38 ? decision.selected_evaluator ==
                                YE3T_LAMMPS::YACEEvaluatorKind::COUPLED_PRODUCT_DAG
                                                             : decision.candidates.size() == 1 &&
                                decision.selected_evaluator ==
                                    YE3T_LAMMPS::YACEEvaluatorKind::EXPLICIT_CTILDE;
                      }),
          "forced coupled-product policy changed a direct-only row");

  const auto automatic_decision =
      std::find_if(automatic_program.decisions.begin(), automatic_program.decisions.end(),
                   [](const auto &decision) {
                     return decision.function_index == 38;
                   });
  require(automatic_decision != automatic_program.decisions.end() &&
              automatic_decision->candidates.size() == 2 &&
              automatic_program.selected_operation_estimate <=
                  automatic_program.direct_operation_estimate &&
              automatic_program.evaluator_plan_hash.size() == 64,
          "automatic planner did not retain a valid coupled-product choice");
  require(automatic_program.planner_algorithm == "exhaustive_candidate_vector_catalogue_v2" &&
              automatic_program.planner_status == "selected" && automatic_program.planner_optimal &&
              automatic_program.selected_operation_estimate ==
                  std::min(automatic_program.direct_operation_estimate,
                           coupled_program.selected_operation_estimate),
          "automatic planner did not select the cheapest whole-catalogue "
          "execution plan");
  const bool automatic_coupled =
      automatic_decision->selected_evaluator == YE3T_LAMMPS::YACEEvaluatorKind::COUPLED_PRODUCT_DAG;
  require(automatic_program.routes.empty() && automatic_program.power_plans.empty() &&
              automatic_program.scalar_program.routes.empty() &&
              automatic_program.coupled_product_plans.size() == (automatic_coupled ? 1 : 0) &&
              (automatic_coupled ? automatic_program.selected_operation_estimate <
                       automatic_program.direct_operation_estimate
                                 : automatic_program.selected_operation_estimate ==
                       automatic_program.direct_operation_estimate),
          "automatic coupled-product dispatch does not match its selected "
          "whole-catalogue plan");

  const auto environment = load_environment(environment_path);
  const auto vectors = edge_vectors(environment);
  YE3T_LAMMPS::YE3TCPUEvaluator direct_evaluator(&direct_model);
  YE3T_LAMMPS::YE3TCPUEvaluator coupled_evaluator(&coupled_model);
  YE3T_LAMMPS::YE3TCPUEvaluator automatic_evaluator(&automatic_model);
  double direct_energy = 0.0;
  double coupled_energy = 0.0;
  double automatic_energy = 0.0;
  std::vector<double> direct_gradients;
  std::vector<double> coupled_gradients;
  std::vector<double> automatic_gradients;
  evaluate(direct_evaluator, environment, vectors, direct_energy, direct_gradients);
  evaluate(coupled_evaluator, environment, vectors, coupled_energy, coupled_gradients);
  evaluate(automatic_evaluator, environment, vectors, automatic_energy, automatic_gradients);
  require(std::abs(direct_energy) >= 1.0e-4 && !direct_gradients.empty(),
          "coupled-product fixture has insufficient physical signal");

  double coupled_energy_residual = std::abs(coupled_energy - direct_energy);
  double coupled_gradient_residual = 0.0;
  require(coupled_energy_residual <= 4.0e-11 * (1.0 + std::abs(direct_energy)) &&
              std::abs(automatic_energy - direct_energy) <=
                  4.0e-11 * (1.0 + std::abs(direct_energy)) &&
              coupled_gradients.size() == direct_gradients.size() &&
              automatic_gradients.size() == direct_gradients.size(),
          "coupled-product atomic energy or gradient width disagrees with "
          "direct C-tilde");
  for (std::size_t index = 0; index < direct_gradients.size(); ++index) {
    coupled_gradient_residual = std::max(
        coupled_gradient_residual, std::abs(coupled_gradients[index] - direct_gradients[index]));
    const double tolerance = 4.0e-11 * (1.0 + std::abs(direct_gradients[index]));
    require(std::abs(coupled_gradients[index] - direct_gradients[index]) <= tolerance &&
                std::abs(automatic_gradients[index] - direct_gradients[index]) <= tolerance,
            "coupled-product edge gradient disagrees with direct C-tilde");
  }

  for (const int atom_count : {1, 7, 8, 9, 16, 17}) {
    std::vector<double> direct_energies;
    std::vector<double> coupled_energies;
    std::vector<double> automatic_energies;
    std::vector<double> direct_batch_gradients;
    std::vector<double> coupled_batch_gradients;
    std::vector<double> automatic_batch_gradients;
    evaluate_repeated(direct_evaluator, environment, atom_count, direct_energies,
                      direct_batch_gradients);
    evaluate_repeated(coupled_evaluator, environment, atom_count, coupled_energies,
                      coupled_batch_gradients);
    evaluate_repeated(automatic_evaluator, environment, atom_count, automatic_energies,
                      automatic_batch_gradients);
    require(direct_energies.size() == coupled_energies.size() &&
                direct_energies.size() == automatic_energies.size() &&
                direct_batch_gradients.size() == coupled_batch_gradients.size() &&
                direct_batch_gradients.size() == automatic_batch_gradients.size(),
            "coupled-product tiled batch dimensions disagree");
    for (std::size_t index = 0; index < direct_energies.size(); ++index) {
      const double tolerance = 4.0e-11 * (1.0 + std::abs(direct_energies[index]));
      require(std::abs(coupled_energies[index] - direct_energies[index]) <= tolerance &&
                  std::abs(automatic_energies[index] - direct_energies[index]) <= tolerance,
              "coupled-product tiled batch energy disagrees");
    }
    for (std::size_t index = 0; index < direct_batch_gradients.size(); ++index) {
      const double tolerance = 4.0e-11 * (1.0 + std::abs(direct_batch_gradients[index]));
      require(std::abs(coupled_batch_gradients[index] - direct_batch_gradients[index]) <=
                      tolerance &&
                  std::abs(automatic_batch_gradients[index] - direct_batch_gradients[index]) <=
                      tolerance,
              "coupled-product tiled batch gradient disagrees");
    }
  }

  Environment zero_environment = environment;
  zero_environment.neighbors.clear();
  std::vector<double> direct_zero_energies;
  std::vector<double> coupled_zero_energies;
  std::vector<double> automatic_zero_energies;
  std::vector<double> direct_zero_gradients;
  std::vector<double> coupled_zero_gradients;
  std::vector<double> automatic_zero_gradients;
  evaluate_repeated(direct_evaluator, zero_environment, 9, direct_zero_energies,
                    direct_zero_gradients);
  evaluate_repeated(coupled_evaluator, zero_environment, 9, coupled_zero_energies,
                    coupled_zero_gradients);
  evaluate_repeated(automatic_evaluator, zero_environment, 9, automatic_zero_energies,
                    automatic_zero_gradients);
  require(direct_zero_gradients.empty() && coupled_zero_gradients.empty() &&
              automatic_zero_gradients.empty(),
          "zero-neighbor coupled-product evaluation returned gradients");
  for (std::size_t index = 0; index < direct_zero_energies.size(); ++index)
    require(std::abs(coupled_zero_energies[index] - direct_zero_energies[index]) <= 4.0e-11 &&
                std::abs(automatic_zero_energies[index] - direct_zero_energies[index]) <= 4.0e-11,
            "zero-neighbor coupled-product energy disagrees");

  Environment reversed = environment;
  std::reverse(reversed.neighbors.begin(), reversed.neighbors.end());
  const auto reversed_vectors = edge_vectors(reversed);
  double reversed_energy = 0.0;
  std::vector<double> reversed_gradients;
  evaluate(coupled_evaluator, reversed, reversed_vectors, reversed_energy, reversed_gradients);
  require(std::abs(reversed_energy - coupled_energy) <= 4.0e-11 * (1.0 + std::abs(coupled_energy)),
          "neighbor reordering changed the coupled-product energy");
  for (std::size_t edge = 0; edge < environment.neighbors.size(); ++edge) {
    const std::size_t reversed_edge = environment.neighbors.size() - edge - 1;
    for (int axis = 0; axis < 3; ++axis)
      require(std::abs(coupled_gradients[edge * 3 + axis] -
                       reversed_gradients[reversed_edge * 3 + axis]) <=
                  4.0e-11 * (1.0 + std::abs(coupled_gradients[edge * 3 + axis])),
              "neighbor reordering changed a coupled-product gradient");
  }

  const double step = 1.0e-6;
  double finite_difference_residual = 0.0;
  for (std::size_t edge = 0; edge < environment.neighbors.size(); ++edge)
    for (int axis = 0; axis < 3; ++axis) {
      auto plus = vectors;
      auto minus = vectors;
      plus[edge * 3 + static_cast<std::size_t>(axis)] += step;
      minus[edge * 3 + static_cast<std::size_t>(axis)] -= step;
      double plus_energy = 0.0;
      double minus_energy = 0.0;
      std::vector<double> scratch;
      evaluate(coupled_evaluator, environment, plus, plus_energy, scratch);
      evaluate(coupled_evaluator, environment, minus, minus_energy, scratch);
      const double finite_difference = (plus_energy - minus_energy) / (2.0 * step);
      const double analytic = coupled_gradients[edge * 3 + static_cast<std::size_t>(axis)];
      finite_difference_residual =
          std::max(finite_difference_residual, std::abs(analytic - finite_difference));
      require(std::abs(analytic - finite_difference) <= 3.0e-6 * (1.0 + std::abs(analytic)),
              "coupled-product edge gradient finite-difference mismatch");
    }

  std::cout << "coupled-product function: " << plan.function_index << '\n'
            << "coupled-product nodes: " << node_count << '\n'
            << "coupled-product operation estimate: " << plan.operation_estimate << '\n'
            << "automatic evaluator: "
            << (automatic_decision->selected_evaluator ==
                        YE3T_LAMMPS::YACEEvaluatorKind::COUPLED_PRODUCT_DAG
                    ? "coupled_product_dag"
                    : "explicit_ctilde")
            << '\n'
            << "coupled/direct maximum atomic-energy residual: " << coupled_energy_residual << '\n'
            << "coupled/direct maximum edge-gradient residual: " << coupled_gradient_residual
            << '\n'
            << "coupled finite-difference maximum residual: " << finite_difference_residual << '\n';
}

}    // namespace

int main(int argc, char **argv)
{
  if (argc != 3 && argc != 4 && argc != 6 && argc != 7 && argc != 8 && argc != 9) {
    std::cerr
        << "usage: test_cpu_evaluator MODEL.yace ENVIRONMENT.env "
           "[MANIFEST.json [EXPECTED_ROUTES EXPECTED_POWER_PLANS "
           "[auto|scalar|portfolio|candidate_vector|gpu_auto_candidate_vector|coupled_product "
           "[EXPECTED_CANDIDATES [REORDERED_MANIFEST]]]]]\n";
    return 2;
  }
  try {
    if (argc == 3)
      run(argv[1], argv[2]);
    else {
      const std::size_t expected_routes = argc >= 6 ? std::stoull(argv[4]) : 2;
      const std::size_t expected_power_plans = argc >= 6 ? std::stoull(argv[5]) : 4;
      const std::string extended_mode = argc >= 7 ? std::string(argv[6]) : "";
      const bool expect_automatic = extended_mode == "auto" || extended_mode == "scalar" ||
          extended_mode == "portfolio" || extended_mode == "candidate_vector" ||
          extended_mode == "gpu_auto_candidate_vector";
      const bool expect_scalar = extended_mode == "scalar" || extended_mode == "portfolio";
      const bool gpu_automatic = extended_mode == "gpu_auto_candidate_vector";
      const std::size_t expected_candidates = argc >= 8 ? std::stoull(argv[7]) : expected_routes;
      const std::string reordered_manifest = argc == 9 ? argv[8] : "";
      if (extended_mode == "coupled_product")
        run_coupled_product_comparison(argv[1], argv[2], argv[3]);
      else
        run_block_comparison(argv[1], argv[2], argv[3], expected_routes, expected_power_plans,
                             argc == 4, expect_automatic, expect_scalar, gpu_automatic,
                             expected_candidates, reordered_manifest);
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
