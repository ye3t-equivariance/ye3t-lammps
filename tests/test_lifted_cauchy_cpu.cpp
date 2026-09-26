#include "ye3t_lifted_cauchy_cpu.h"
#include "ye3t_lifted_cauchy_model.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using YE3T_LAMMPS::LiftedCauchyCPULoweredReadout;
using YE3T_LAMMPS::LiftedCauchyCPUSource;
using YE3T_LAMMPS::LiftedCauchyEdge;
using YE3T_LAMMPS::LiftedCauchyModel;
using YE3T_LAMMPS::LiftedCauchySourcePolicy;

namespace {

using Vector3 = std::array<double, 3>;
using Matrix3 = std::array<Vector3, 3>;

void require(bool condition, const std::string &message)
{
  if (!condition) throw std::runtime_error(message);
}

void require_close(double actual, double expected, double tolerance, const std::string &message)
{
  const double scale = std::max({1.0, std::abs(actual), std::abs(expected)});
  if (std::abs(actual - expected) > tolerance * scale)
    throw std::runtime_error(message + ": actual=" + std::to_string(actual) +
                             " expected=" + std::to_string(expected));
}

double dot(const std::vector<double> &left, const std::vector<double> &right)
{
  require(left.size() == right.size(), "dot-product size mismatch");
  double result = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) result += left[index] * right[index];
  return result;
}

std::vector<double> numeric_vector(const YAML::Node &node, const std::string &name)
{
  require(node && node.IsSequence(), name + " must be a sequence");
  std::vector<double> result;
  result.reserve(node.size());
  for (std::size_t index = 0; index < node.size(); ++index)
    result.push_back(node[index].as<double>());
  return result;
}

std::vector<int> integer_vector(const YAML::Node &node, const std::string &name)
{
  require(node && node.IsSequence(), name + " must be a sequence");
  std::vector<int> result;
  result.reserve(node.size());
  for (std::size_t index = 0; index < node.size(); ++index) result.push_back(node[index].as<int>());
  return result;
}

std::vector<Vector3> vector3_rows(const YAML::Node &node, const std::string &name)
{
  require(node && node.IsSequence(), name + " must be a sequence");
  std::vector<Vector3> result;
  result.reserve(node.size());
  for (std::size_t row = 0; row < node.size(); ++row) {
    require(node[row].IsSequence() && node[row].size() == 3,
            name + " rows must have three entries");
    result.push_back(
        {node[row][0].as<double>(), node[row][1].as<double>(), node[row][2].as<double>()});
  }
  return result;
}

Matrix3 matrix3(const YAML::Node &node, const std::string &name)
{
  const auto rows = vector3_rows(node, name);
  require(rows.size() == 3, name + " must have three rows");
  return {rows[0], rows[1], rows[2]};
}

std::vector<double> matrix_rows(const YAML::Node &node, std::size_t columns,
                                const std::string &name)
{
  require(node && node.IsSequence(), name + " must be a sequence");
  std::vector<double> result;
  result.reserve(node.size() * columns);
  for (std::size_t row = 0; row < node.size(); ++row) {
    require(node[row].IsSequence() && node[row].size() == columns,
            name + " has an invalid row width");
    for (std::size_t column = 0; column < columns; ++column)
      result.push_back(node[row][column].as<double>());
  }
  return result;
}

struct FixtureReference {
  std::vector<Vector3> positions;
  std::vector<int> atom_types;
  std::vector<double> atomic_energies;
  double total_energy = 0.0;
  std::vector<double> source_values;
  std::vector<double> source_adjoint;
  std::vector<int> edge_centers;
  std::vector<int> edge_neighbors;
  std::vector<Vector3> edge_displacements;
  std::vector<Vector3> edge_gradients;
  std::vector<Vector3> forces;
  Matrix3 strain_derivative{};
  Matrix3 virial{};
};

FixtureReference load_reference(const std::filesystem::path &path, std::size_t source_count)
{
  const YAML::Node node = YAML::LoadFile(path.string());
  const std::string schema = node["schema"].as<std::string>();
  require(schema == "ye3t_lifted_cauchy_native_fixture_v1" ||
              schema == "ye3t_lifted_cauchy_native_fixture_v2",
          "fixture reference schema mismatch");
  FixtureReference result;
  result.positions = vector3_rows(node["positions_A"], "positions_A");
  result.atom_types = integer_vector(node["atom_types"], "atom_types");
  result.atomic_energies = numeric_vector(node["atomic_energies_eV"], "atomic_energies_eV");
  result.total_energy = node["total_energy_eV"].as<double>();
  result.source_values =
      matrix_rows(node["canonical_source_values_q"], source_count, "canonical_source_values_q");
  result.source_adjoint = matrix_rows(node["canonical_source_adjoints_dE_dq"], source_count,
                                      "canonical_source_adjoints_dE_dq");
  result.edge_centers = integer_vector(node["directed_edge_centers"], "directed_edge_centers");
  result.edge_neighbors =
      integer_vector(node["directed_edge_neighbors"], "directed_edge_neighbors");
  result.edge_displacements =
      vector3_rows(node["directed_edge_displacements_A"], "directed_edge_displacements_A");
  result.edge_gradients = vector3_rows(node["directed_edge_gradients_dE_dd_eV_per_A"],
                                       "directed_edge_gradients_dE_dd_eV_per_A");
  result.forces = vector3_rows(node["forces_eV_per_A"], "forces_eV_per_A");
  result.strain_derivative = matrix3(node["strain_derivative_eV"], "strain_derivative_eV");
  result.virial = matrix3(node["lammps_global_virial_eV"], "lammps_global_virial_eV");
  require(result.positions.size() == result.atom_types.size() &&
              result.positions.size() == result.atomic_energies.size(),
          "fixture atom arrays disagree");
  require(result.edge_centers.size() == result.edge_neighbors.size() &&
              result.edge_centers.size() == result.edge_displacements.size() &&
              result.edge_centers.size() == result.edge_gradients.size(),
          "fixture edge arrays disagree");
  return result;
}

struct SystemResult {
  std::vector<double> source_values;
  std::vector<double> source_adjoint;
  std::vector<double> atomic_energies;
  double total_energy = 0.0;
  std::vector<int> edge_centers;
  std::vector<int> edge_neighbors;
  std::vector<Vector3> edge_displacements;
  std::vector<Vector3> edge_gradients;
  std::vector<Vector3> forces;
  Matrix3 strain_derivative{};
  Matrix3 virial{};
};

SystemResult evaluate_system(const LiftedCauchyModel &model, LiftedCauchyCPUSource &source,
                             LiftedCauchyCPULoweredReadout &readout,
                             const std::vector<Vector3> &positions,
                             const std::vector<int> &atom_types, LiftedCauchySourcePolicy policy,
                             bool reverse_neighbors = false)
{
  require(positions.size() == atom_types.size(), "system position/type count mismatch");
  const int atom_count = static_cast<int>(positions.size());
  const std::size_t source_count = static_cast<std::size_t>(model.source_variable_count);
  SystemResult result;
  result.source_values.resize(positions.size() * source_count);
  result.atomic_energies.resize(positions.size());
  result.source_adjoint.resize(positions.size() * source_count);
  result.forces.resize(positions.size());

  std::vector<std::vector<LiftedCauchyEdge>> environments(positions.size());
  std::vector<std::vector<int>> neighbors(positions.size());
  for (int center = 0; center < atom_count; ++center) {
    for (int neighbor = 0; neighbor < atom_count; ++neighbor) {
      if (neighbor == center) continue;
      Vector3 displacement{};
      for (int component = 0; component < 3; ++component)
        displacement[component] = positions[neighbor][component] - positions[center][component];
      environments[center].push_back(LiftedCauchyEdge{atom_types[neighbor], displacement});
      neighbors[center].push_back(neighbor);
    }
    if (reverse_neighbors) {
      std::reverse(environments[center].begin(), environments[center].end());
      std::reverse(neighbors[center].begin(), neighbors[center].end());
    }
    std::vector<double> values;
    source.accumulate(environments[center], policy, values);
    std::copy(values.begin(), values.end(),
              result.source_values.begin() + static_cast<std::size_t>(center) * source_count);
  }

  readout.evaluate(atom_count, atom_types.data(), result.source_values.data(),
                   result.atomic_energies.data(), result.source_adjoint.data());
  result.total_energy =
      std::accumulate(result.atomic_energies.begin(), result.atomic_energies.end(), 0.0);

  for (int center = 0; center < atom_count; ++center) {
    std::vector<double> adjoint(
        result.source_adjoint.begin() + static_cast<std::size_t>(center) * source_count,
        result.source_adjoint.begin() + static_cast<std::size_t>(center + 1) * source_count);
    std::vector<Vector3> gradients;
    source.vjp(environments[center], adjoint, policy, gradients);
    for (std::size_t edge = 0; edge < gradients.size(); ++edge) {
      const int neighbor = neighbors[center][edge];
      result.edge_centers.push_back(center);
      result.edge_neighbors.push_back(neighbor);
      result.edge_displacements.push_back(environments[center][edge].displacement);
      result.edge_gradients.push_back(gradients[edge]);
      for (int component = 0; component < 3; ++component) {
        result.forces[center][component] += gradients[edge][component];
        result.forces[neighbor][component] -= gradients[edge][component];
        for (int right = 0; right < 3; ++right)
          result.strain_derivative[component][right] +=
              gradients[edge][component] * environments[center][edge].displacement[right];
      }
    }
  }
  for (int left = 0; left < 3; ++left)
    for (int right = 0; right < 3; ++right)
      result.virial[left][right] = -result.strain_derivative[left][right];
  return result;
}

double source_objective(const LiftedCauchyCPUSource &source,
                        const std::vector<LiftedCauchyEdge> &edges,
                        const std::vector<double> &adjoint, LiftedCauchySourcePolicy policy)
{
  std::vector<double> values;
  source.accumulate(edges, policy, values);
  return dot(values, adjoint);
}

void test_axis_convention(const LiftedCauchyModel &model, const LiftedCauchyCPUSource &source)
{
  const auto group =
      std::find_if(model.source_groups.begin(), model.source_groups.end(), [](const auto &item) {
        return item.angular == 1;
      });
  if (group == model.source_groups.end()) return;
  const double coordinate = 0.37;
  const double distance = coordinate * model.cutoff;
  const double coefficient = group->direct_q_polynomials.front().coefficients[0];
  const double expected = coefficient * (1.0 - coordinate) * (1.0 - coordinate) * coordinate;
  const std::int64_t offset = group->q_source_variable_offsets.front();
  const std::array<std::array<double, 3>, 3> axes{{
      {distance, 0.0, 0.0},
      {0.0, 0.0, distance},
      {0.0, distance, 0.0},
  }};
  const std::array<int, 3> active{{0, 1, 2}};
  const std::array<double, 3> signs{{1.0, 1.0, -1.0}};
  for (std::size_t axis = 0; axis < axes.size(); ++axis) {
    std::vector<double> values;
    source.accumulate({LiftedCauchyEdge{0, axes[axis]}}, LiftedCauchySourcePolicy::DIRECT_Q,
                      values);
    for (int component = 0; component < 3; ++component)
      require_close(values[static_cast<std::size_t>(offset + component)],
                    component == active[axis] ? signs[axis] * expected : 0.0, 2.0e-12,
                    "physical-real axis convention mismatch");
  }
}

void test_source_and_vjp(const LiftedCauchyModel &model, const LiftedCauchyCPUSource &source)
{
  std::vector<LiftedCauchyEdge> edges{
      {0, {1.3, -0.4, 0.7}},
      {0, {-0.8, 1.1, 0.2}},
      {0, {0.0, 0.0, 0.0}},
  };
  std::vector<double> direct;
  std::vector<double> factorized;
  source.accumulate(edges, LiftedCauchySourcePolicy::DIRECT_Q, direct);
  source.accumulate(edges, LiftedCauchySourcePolicy::FACTORIZED, factorized);
  require(direct.size() == static_cast<std::size_t>(model.source_variable_count),
          "source variable count mismatch");
  for (std::size_t index = 0; index < direct.size(); ++index)
    require_close(direct[index], factorized[index], 5.0e-11, "direct/factorized source mismatch");

  std::vector<double> adjoint(direct.size());
  for (std::size_t index = 0; index < adjoint.size(); ++index)
    adjoint[index] = 0.17 * static_cast<double>(index + 1);
  std::vector<std::array<double, 3>> direct_gradient;
  std::vector<std::array<double, 3>> factorized_gradient;
  source.vjp(edges, adjoint, LiftedCauchySourcePolicy::DIRECT_Q, direct_gradient);
  source.vjp(edges, adjoint, LiftedCauchySourcePolicy::FACTORIZED, factorized_gradient);
  for (std::size_t edge = 0; edge < edges.size(); ++edge)
    for (int component = 0; component < 3; ++component)
      require_close(direct_gradient[edge][component], factorized_gradient[edge][component], 5.0e-10,
                    "direct/factorized source VJP mismatch");

  const double step = 2.0e-6;
  for (int component = 0; component < 3; ++component) {
    auto plus = edges;
    auto minus = edges;
    plus[0].displacement[component] += step;
    minus[0].displacement[component] -= step;
    const double finite_difference =
        (source_objective(source, plus, adjoint, LiftedCauchySourcePolicy::DIRECT_Q) -
         source_objective(source, minus, adjoint, LiftedCauchySourcePolicy::DIRECT_Q)) /
        (2.0 * step);
    require_close(direct_gradient[0][component], finite_difference, 2.0e-8,
                  "direct source finite-difference mismatch");
  }

  if (model.exclude_zero_separation) {
    for (int component = 0; component < 3; ++component)
      require_close(direct_gradient[2][component], 0.0, 0.0,
                    "excluded zero-separation edge has a source gradient");
  } else {
    const double origin_step = 1.0e-8;
    for (int component = 0; component < 3; ++component) {
      auto plus = edges;
      auto minus = edges;
      plus[2].displacement[component] += origin_step;
      minus[2].displacement[component] -= origin_step;
      const double finite_difference =
          (source_objective(source, plus, adjoint, LiftedCauchySourcePolicy::DIRECT_Q) -
           source_objective(source, minus, adjoint, LiftedCauchySourcePolicy::DIRECT_Q)) /
          (2.0 * origin_step);
      require_close(direct_gradient[2][component], finite_difference, 2.0e-5,
                    "origin source finite-difference mismatch");
    }
  }
}

void test_cutoff(const LiftedCauchyModel &model, const LiftedCauchyCPUSource &source)
{
  const std::vector<LiftedCauchyEdge> edges{
      {0, {model.cutoff, 0.0, 0.0}},
      {0, {model.cutoff + 0.1, 0.0, 0.0}},
  };
  for (const auto policy :
       {LiftedCauchySourcePolicy::DIRECT_Q, LiftedCauchySourcePolicy::FACTORIZED}) {
    std::vector<double> values;
    source.accumulate(edges, policy, values);
    require(std::all_of(values.begin(), values.end(),
                        [](double value) {
                          return value == 0.0;
                        }),
            "source is nonzero at or beyond the cutoff");
    std::vector<double> adjoint(values.size(), 1.0);
    std::vector<std::array<double, 3>> gradients;
    source.vjp(edges, adjoint, policy, gradients);
    for (const auto &gradient : gradients)
      for (double value : gradient)
        require(value == 0.0, "source derivative is nonzero beyond the cutoff");
  }
}

void compare_system_to_reference(const SystemResult &actual, const FixtureReference &expected,
                                 const std::string &policy)
{
  require(actual.atomic_energies.size() == expected.atomic_energies.size(),
          policy + " atomic-energy count mismatch");
  for (std::size_t atom = 0; atom < actual.atomic_energies.size(); ++atom)
    require_close(actual.atomic_energies[atom], expected.atomic_energies[atom], 5.0e-11,
                  policy + " atomic energy mismatch");
  require_close(actual.total_energy, expected.total_energy, 5.0e-11,
                policy + " total energy mismatch");
  require(actual.source_values.size() == expected.source_values.size(),
          policy + " source size mismatch");
  for (std::size_t index = 0; index < actual.source_values.size(); ++index) {
    require_close(actual.source_values[index], expected.source_values[index], 5.0e-11,
                  policy + " canonical source mismatch");
    require_close(actual.source_adjoint[index], expected.source_adjoint[index], 5.0e-10,
                  policy + " source adjoint mismatch");
  }
  require(actual.edge_centers == expected.edge_centers &&
              actual.edge_neighbors == expected.edge_neighbors,
          policy + " directed-edge order mismatch");
  for (std::size_t edge = 0; edge < actual.edge_gradients.size(); ++edge)
    for (int component = 0; component < 3; ++component) {
      require_close(actual.edge_displacements[edge][component],
                    expected.edge_displacements[edge][component], 2.0e-14,
                    policy + " edge displacement mismatch");
      require_close(actual.edge_gradients[edge][component],
                    expected.edge_gradients[edge][component], 5.0e-9,
                    policy + " edge gradient mismatch");
    }
  for (std::size_t atom = 0; atom < actual.forces.size(); ++atom)
    for (int component = 0; component < 3; ++component)
      require_close(actual.forces[atom][component], expected.forces[atom][component], 5.0e-9,
                    policy + " force mismatch");
  for (int left = 0; left < 3; ++left)
    for (int right = 0; right < 3; ++right) {
      require_close(actual.strain_derivative[left][right], expected.strain_derivative[left][right],
                    5.0e-9, policy + " strain derivative mismatch");
      require_close(actual.virial[left][right], expected.virial[left][right], 5.0e-9,
                    policy + " LAMMPS virial mismatch");
      require_close(actual.virial[left][right], -actual.strain_derivative[left][right], 0.0,
                    policy + " virial sign mismatch");
    }
}

void test_lowered_readout_and_reference(const LiftedCauchyModel &model,
                                        LiftedCauchyCPUSource &source,
                                        LiftedCauchyCPULoweredReadout &readout,
                                        const FixtureReference &reference)
{
  for (const auto policy :
       {LiftedCauchySourcePolicy::DIRECT_Q, LiftedCauchySourcePolicy::FACTORIZED}) {
    const std::string name = policy == LiftedCauchySourcePolicy::DIRECT_Q ? "direct-Q"
                                                                          : "factorized-T";
    const SystemResult result =
        evaluate_system(model, source, readout, reference.positions, reference.atom_types, policy);
    compare_system_to_reference(result, reference, name);
  }

  const SystemResult ordered =
      evaluate_system(model, source, readout, reference.positions, reference.atom_types,
                      LiftedCauchySourcePolicy::DIRECT_Q);
  const SystemResult reordered =
      evaluate_system(model, source, readout, reference.positions, reference.atom_types,
                      LiftedCauchySourcePolicy::DIRECT_Q, true);
  require_close(reordered.total_energy, ordered.total_energy, 2.0e-12,
                "neighbor reorder changed the energy");
  for (std::size_t atom = 0; atom < ordered.forces.size(); ++atom)
    for (int component = 0; component < 3; ++component)
      require_close(reordered.forces[atom][component], ordered.forces[atom][component], 2.0e-11,
                    "neighbor reorder changed a force");
  for (int left = 0; left < 3; ++left)
    for (int right = 0; right < 3; ++right)
      require_close(reordered.strain_derivative[left][right],
                    ordered.strain_derivative[left][right], 2.0e-11,
                    "neighbor reorder changed the strain derivative");
}

std::vector<Vector3> transform_vectors(const std::vector<Vector3> &vectors, const Matrix3 &matrix)
{
  std::vector<Vector3> result(vectors.size());
  for (std::size_t vector = 0; vector < vectors.size(); ++vector)
    for (int row = 0; row < 3; ++row)
      for (int column = 0; column < 3; ++column)
        result[vector][row] += matrix[row][column] * vectors[vector][column];
  return result;
}

Matrix3 transform_matrix(const Matrix3 &value, const Matrix3 &matrix)
{
  Matrix3 result{};
  for (int left = 0; left < 3; ++left)
    for (int right = 0; right < 3; ++right)
      for (int inner_left = 0; inner_left < 3; ++inner_left)
        for (int inner_right = 0; inner_right < 3; ++inner_right)
          result[left][right] += matrix[left][inner_left] * value[inner_left][inner_right] *
              matrix[right][inner_right];
  return result;
}

void test_o3_and_conservation(const LiftedCauchyModel &model, LiftedCauchyCPUSource &source,
                              LiftedCauchyCPULoweredReadout &readout,
                              const FixtureReference &reference)
{
  const SystemResult baseline =
      evaluate_system(model, source, readout, reference.positions, reference.atom_types,
                      LiftedCauchySourcePolicy::DIRECT_Q);
  const std::array<Matrix3, 3> transforms{{
      {{{0.0, -1.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, 1.0}}},
      {{{-1.0, 0.0, 0.0}, {0.0, -1.0, 0.0}, {0.0, 0.0, -1.0}}},
      {{{-6.0 / 7.0, 2.0 / 7.0, 3.0 / 7.0},
        {2.0 / 7.0, -3.0 / 7.0, 6.0 / 7.0},
        {3.0 / 7.0, 6.0 / 7.0, 2.0 / 7.0}}},
  }};
  for (const auto &matrix : transforms) {
    const auto positions = transform_vectors(reference.positions, matrix);
    const SystemResult transformed =
        evaluate_system(model, source, readout, positions, reference.atom_types,
                        LiftedCauchySourcePolicy::DIRECT_Q);
    require_close(transformed.total_energy, baseline.total_energy, 5.0e-11,
                  "O(3) transform changed the energy");
    const auto expected_forces = transform_vectors(baseline.forces, matrix);
    for (std::size_t atom = 0; atom < baseline.forces.size(); ++atom)
      for (int component = 0; component < 3; ++component)
        require_close(transformed.forces[atom][component], expected_forces[atom][component], 5.0e-9,
                      "O(3) force covariance mismatch");
    const Matrix3 expected_strain = transform_matrix(baseline.strain_derivative, matrix);
    for (int left = 0; left < 3; ++left)
      for (int right = 0; right < 3; ++right)
        require_close(transformed.strain_derivative[left][right], expected_strain[left][right],
                      5.0e-9, "O(3) strain covariance mismatch");
  }

  auto translated_positions = reference.positions;
  for (auto &position : translated_positions) {
    position[0] += 1.3;
    position[1] -= 0.8;
    position[2] += 0.45;
  }
  const SystemResult translated =
      evaluate_system(model, source, readout, translated_positions, reference.atom_types,
                      LiftedCauchySourcePolicy::DIRECT_Q);
  require_close(translated.total_energy, baseline.total_energy, 5.0e-11,
                "translation changed the energy");
  for (std::size_t atom = 0; atom < baseline.forces.size(); ++atom)
    for (int component = 0; component < 3; ++component)
      require_close(translated.forces[atom][component], baseline.forces[atom][component], 5.0e-9,
                    "translation changed a force");
  for (int left = 0; left < 3; ++left)
    for (int right = 0; right < 3; ++right)
      require_close(translated.strain_derivative[left][right],
                    baseline.strain_derivative[left][right], 5.0e-9,
                    "translation changed the strain derivative");

  Vector3 net_force{};
  Vector3 net_torque{};
  for (std::size_t atom = 0; atom < baseline.forces.size(); ++atom) {
    for (int component = 0; component < 3; ++component)
      net_force[component] += baseline.forces[atom][component];
    net_torque[0] += reference.positions[atom][1] * baseline.forces[atom][2] -
        reference.positions[atom][2] * baseline.forces[atom][1];
    net_torque[1] += reference.positions[atom][2] * baseline.forces[atom][0] -
        reference.positions[atom][0] * baseline.forces[atom][2];
    net_torque[2] += reference.positions[atom][0] * baseline.forces[atom][1] -
        reference.positions[atom][1] * baseline.forces[atom][0];
  }
  for (int component = 0; component < 3; ++component) {
    require_close(net_force[component], 0.0, 2.0e-11,
                  "native forces do not conserve total momentum");
    require_close(net_torque[component], 0.0, 2.0e-9,
                  "native forces do not conserve angular momentum");
  }
  for (int left = 0; left < 3; ++left)
    for (int right = left + 1; right < 3; ++right)
      require_close(baseline.strain_derivative[left][right],
                    baseline.strain_derivative[right][left], 2.0e-9,
                    "native strain derivative is not symmetric");
}

void test_finite_differences(const LiftedCauchyModel &model, LiftedCauchyCPUSource &source,
                             LiftedCauchyCPULoweredReadout &readout,
                             const FixtureReference &reference)
{
  const SystemResult baseline =
      evaluate_system(model, source, readout, reference.positions, reference.atom_types,
                      LiftedCauchySourcePolicy::DIRECT_Q);
  const double coordinate_step = 2.0e-6;
  for (std::size_t atom = 0; atom < reference.positions.size(); ++atom)
    for (int component = 0; component < 3; ++component) {
      auto plus = reference.positions;
      auto minus = reference.positions;
      plus[atom][component] += coordinate_step;
      minus[atom][component] -= coordinate_step;
      const double derivative =
          (evaluate_system(model, source, readout, plus, reference.atom_types,
                           LiftedCauchySourcePolicy::DIRECT_Q)
               .total_energy -
           evaluate_system(model, source, readout, minus, reference.atom_types,
                           LiftedCauchySourcePolicy::DIRECT_Q)
               .total_energy) /
          (2.0 * coordinate_step);
      require_close(-baseline.forces[atom][component], derivative, 2.0e-7,
                    "coordinate finite-difference mismatch");
    }

  const double strain_step = 1.0e-6;
  for (int left = 0; left < 3; ++left)
    for (int right = 0; right < 3; ++right) {
      auto plus = reference.positions;
      auto minus = reference.positions;
      for (std::size_t atom = 0; atom < reference.positions.size(); ++atom) {
        plus[atom][left] += strain_step * reference.positions[atom][right];
        minus[atom][left] -= strain_step * reference.positions[atom][right];
      }
      const double derivative =
          (evaluate_system(model, source, readout, plus, reference.atom_types,
                           LiftedCauchySourcePolicy::DIRECT_Q)
               .total_energy -
           evaluate_system(model, source, readout, minus, reference.atom_types,
                           LiftedCauchySourcePolicy::DIRECT_Q)
               .total_energy) /
          (2.0 * strain_step);
      require_close(baseline.strain_derivative[left][right], derivative, 2.0e-7,
                    "strain finite-difference mismatch");
    }
}

void test_polynomial_vjp_and_head_selection(const LiftedCauchyModel &model)
{
  const std::size_t source_count = static_cast<std::size_t>(model.source_variable_count);
  LiftedCauchyCPULoweredReadout readout(&model);
  std::vector<double> source_values(source_count);
  for (std::size_t index = 0; index < source_count; ++index)
    source_values[index] = 0.11 * static_cast<double>(index + 1) - 0.37;
  const int head = 0;
  double energy = 0.0;
  std::vector<double> adjoint(source_count);
  readout.evaluate(1, &head, source_values.data(), &energy, adjoint.data());
  const double step = 1.0e-6;
  for (std::size_t index = 0; index < source_count; ++index) {
    auto plus = source_values;
    auto minus = source_values;
    plus[index] += step;
    minus[index] -= step;
    double plus_energy = 0.0;
    double minus_energy = 0.0;
    std::vector<double> scratch(source_count);
    readout.evaluate(1, &head, plus.data(), &plus_energy, scratch.data());
    readout.evaluate(1, &head, minus.data(), &minus_energy, scratch.data());
    require_close(adjoint[index], (plus_energy - minus_energy) / (2.0 * step), 2.0e-7,
                  "sparse polynomial finite-difference mismatch");
  }

  LiftedCauchyModel zero_model = model;
  auto &polynomial = zero_model.heads[0].polynomial;
  polynomial.offset = 0.75;
  polynomial.factor_offsets = {0, 2, 4, 6};
  polynomial.factor_indices = {0, 1, 0, 1, 0, 1};
  polynomial.factor_exponents = {1, 1, 2, 1, 3, 2};
  polynomial.monomial_coefficients = {2.0, 3.0, -4.0};
  polynomial.maximum_tensor_rank = 5;
  polynomial.maximum_factor_count = 2;
  LiftedCauchyCPULoweredReadout zero_readout(&zero_model);
  std::vector<double> zero_source(source_count, 0.0);
  zero_source[1] = 3.0;
  double zero_energy = 0.0;
  std::vector<double> zero_adjoint(source_count);
  zero_readout.evaluate(1, &head, zero_source.data(), &zero_energy, zero_adjoint.data());
  require_close(zero_energy, 0.75, 0.0, "zero-safe polynomial value mismatch");
  require_close(zero_adjoint[0], 6.0, 0.0, "zero-safe product adjoint used a division shortcut");
  for (std::size_t index = 1; index < source_count; ++index)
    require_close(zero_adjoint[index], 0.0, 0.0,
                  "zero-safe polynomial produced a spurious adjoint");

  zero_source[0] = -0.4;
  zero_source[1] = 0.7;
  zero_readout.evaluate(1, &head, zero_source.data(), &zero_energy, zero_adjoint.data());
  for (int index = 0; index < 2; ++index) {
    auto plus = zero_source;
    auto minus = zero_source;
    plus[index] += step;
    minus[index] -= step;
    double plus_energy = 0.0;
    double minus_energy = 0.0;
    std::vector<double> scratch(source_count);
    zero_readout.evaluate(1, &head, plus.data(), &plus_energy, scratch.data());
    zero_readout.evaluate(1, &head, minus.data(), &minus_energy, scratch.data());
    require_close(zero_adjoint[index], (plus_energy - minus_energy) / (2.0 * step), 2.0e-9,
                  "repeated-power polynomial adjoint mismatch");
  }

  polynomial.factor_offsets = {0};
  polynomial.factor_indices.clear();
  polynomial.factor_exponents.clear();
  polynomial.monomial_coefficients.clear();
  polynomial.maximum_tensor_rank = 0;
  polynomial.maximum_factor_count = 0;
  LiftedCauchyCPULoweredReadout constant_readout(&zero_model);
  std::fill(zero_adjoint.begin(), zero_adjoint.end(), 1.0);
  constant_readout.evaluate(1, &head, zero_source.data(), &zero_energy, zero_adjoint.data());
  require_close(zero_energy, 0.75, 0.0, "offset-only head value mismatch");
  require(std::all_of(zero_adjoint.begin(), zero_adjoint.end(),
                      [](double value) {
                        return value == 0.0;
                      }),
          "offset-only head has a nonzero source adjoint");

  LiftedCauchyModel two_head = model;
  auto second = two_head.heads[0];
  second.central_species = "W";
  second.central_species_index = 1;
  second.polynomial.offset = 1.5 * second.polynomial.offset + 0.3;
  for (double &coefficient : second.polynomial.monomial_coefficients) coefficient *= 1.5;
  two_head.central_species_order.push_back("W");
  two_head.type_map.push_back(7);
  two_head.heads.push_back(std::move(second));
  LiftedCauchyCPULoweredReadout two_head_readout(&two_head);
  std::vector<double> two_sources;
  two_sources.insert(two_sources.end(), source_values.begin(), source_values.end());
  two_sources.insert(two_sources.end(), source_values.begin(), source_values.end());
  const int heads[2] = {0, 1};
  double energies[2]{};
  std::vector<double> two_adjoints(2 * source_count);
  two_head_readout.evaluate(2, heads, two_sources.data(), energies, two_adjoints.data());
  require_close(energies[1], 1.5 * energies[0] + 0.3, 5.0e-13,
                "central-species head selection mismatch");
  for (std::size_t index = 0; index < source_count; ++index)
    require_close(two_adjoints[source_count + index], 1.5 * two_adjoints[index], 5.0e-13,
                  "central-species head adjoint mismatch");

  bool rejected = false;
  const int invalid_head = 2;
  try {
    two_head_readout.evaluate(1, &invalid_head, source_values.data(), &energy, adjoint.data());
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected, "invalid central-species head was accepted");
  two_head_readout.evaluate(0, nullptr, nullptr, nullptr, nullptr);
}

void test_file_tamper_rejected(const std::filesystem::path &fixture)
{
  namespace fs = std::filesystem;
  const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  const fs::path copy = fs::temp_directory_path() / ("ye3t_lifted_loader_" + std::to_string(stamp));
  fs::create_directory(copy);
  try {
    fs::copy(fixture, copy, fs::copy_options::recursive | fs::copy_options::overwrite_existing);
    std::ofstream stream(copy / "native_runtime.ye3t.json", std::ios::app);
    stream << ' ';
    stream.close();
    bool rejected = false;
    try {
      (void) LiftedCauchyModel::load(copy.string());
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    require(rejected, "native runtime file tamper was accepted");
  } catch (...) {
    fs::remove_all(copy);
    throw;
  }
  fs::remove_all(copy);
}

}    // namespace

int main(int argc, char **argv)
{
  try {
    if (argc != 2 && argc != 3)
      throw std::runtime_error("usage: test_lifted_cauchy_cpu BUNDLE_DIR [COMPOSITE_BUNDLE_DIR]");
    const std::filesystem::path fixture(argv[1]);
    const LiftedCauchyModel model = LiftedCauchyModel::load(fixture.string());
    require(model.central_species_order == std::vector<std::string>{"Ta"},
            "fixture species ordering mismatch");
    require(model.source_variable_count > 0, "fixture source width mismatch");
    require(model.heads.size() == 1 && !model.heads[0].polynomial.monomial_coefficients.empty(),
            "fixture polynomial is empty");
    LiftedCauchyCPUSource source(&model);
    LiftedCauchyCPULoweredReadout readout(&model);
    const FixtureReference reference = load_reference(
        fixture / "reference.json", static_cast<std::size_t>(model.source_variable_count));
    const std::size_t expected_workspace =
        static_cast<std::size_t>(4 * model.heads[0].polynomial.maximum_factor_count + 2);
    require(readout.memory_usage() == expected_workspace * sizeof(double),
            "lowered-readout workspace accounting mismatch");
    test_axis_convention(model, source);
    test_source_and_vjp(model, source);
    test_cutoff(model, source);
    test_lowered_readout_and_reference(model, source, readout, reference);
    test_o3_and_conservation(model, source, readout, reference);
    test_finite_differences(model, source, readout, reference);
    test_polynomial_vjp_and_head_selection(model);
    test_file_tamper_rejected(fixture);
    if (argc == 3) {
      const std::filesystem::path composite_fixture(argv[2]);
      const LiftedCauchyModel composite = LiftedCauchyModel::load(composite_fixture.string());
      require(composite.composite_compiler_binding,
              "composite bundle did not select the compact binding path");
      require(!composite.compiler_binding_self_hash.empty() &&
                  !composite.composite_artifact_hash.empty() &&
                  composite.compiler_artifact_self_hash.empty(),
              "composite bundle identities are incomplete");
      require(composite.source_variable_count > 0 && composite.heads.size() == 1 &&
                  !composite.heads[0].polynomial.monomial_coefficients.empty(),
              "composite native runtime is empty");
      LiftedCauchyCPUSource composite_source(&composite);
      test_source_and_vjp(composite, composite_source);
      test_cutoff(composite, composite_source);
      test_polynomial_vjp_and_head_selection(composite);
      test_file_tamper_rejected(composite_fixture);
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "Lifted-Cauchy native loader/source/readout tests passed.\n";
  return 0;
}
