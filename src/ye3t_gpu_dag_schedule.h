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

#ifndef LMP_YE3T_GPU_DAG_SCHEDULE_H
#define LMP_YE3T_GPU_DAG_SCHEDULE_H

#include "ye3t_yace_model.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace YE3T_LAMMPS {

// Offline transpose of a binary DAG. Runtime adjoints are gathered by their
// owner instead of atomically scattered by all parents. Repeated children
// deliberately produce TWO entries (e.g. d(x*x)/dx = 2*x).
struct GpuDagSchedule {
  std::vector<int> parent_offsets{0};
  std::vector<int> parents;
  std::vector<int> siblings;
  std::vector<double> readout_seed;
  std::vector<int> source_power_offsets{0};
  std::vector<int> source_powers;
};

inline int gpu_schedule_index(std::size_t value)
{
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("YE3T GPU reverse schedule exceeds 32-bit indexing");
  return static_cast<int>(value);
}

// Backend-local lowering of the selected DIRECT residual. The CPU planner may
// prefer a prefix DAG (including ties); that is not an unsupported model. Keep
// the power table and readout coefficients in the model and translate only the
// operand IDs needed by the binary-only device kernels. Nothing in the model,
// block selection, or semantic/replay identity is mutated.
struct GpuBinaryDag {
  std::vector<std::int64_t> left;
  std::vector<std::int64_t> right;
  std::vector<std::int64_t> monomial_operands;
};

inline GpuBinaryDag lower_gpu_binary_dag(const YACESparsePolynomial &plan)
{
  if (plan.power_channels.size() != plan.power_exponents.size() ||
      plan.monomial_nodes.size() != plan.monomial_coefficients.size())
    throw std::invalid_argument("YE3T GPU DIRECT DAG tables are inconsistent");
  const int power_count = gpu_schedule_index(plan.power_channels.size());
  GpuBinaryDag result;
  if (plan.binary_dag) {
    // Preserve existing binary schedules, including shared and repeated children.
    result.left = plan.binary_node_left;
    result.right = plan.binary_node_right;
    result.monomial_operands = plan.monomial_nodes;
  } else {
    if (!plan.binary_node_left.empty() || !plan.binary_node_right.empty() ||
        plan.dag_node_parents.size() != plan.dag_node_powers.size())
      throw std::invalid_argument("YE3T GPU prefix DAG tables are inconsistent");
    const std::size_t count = plan.dag_node_parents.size();
    gpu_schedule_index(count);
    if (count == 0) {
      // without_descriptors() leaves no DAG when all work belongs to specialized
      // routes. Do not fabricate a nonempty DIRECT readout in that case.
      if (power_count != 0 || !plan.monomial_nodes.empty() || plan.dag_root_node_count != 0)
        throw std::invalid_argument("YE3T GPU nonempty prefix DAG has no unit root");
      return result;
    }
    if (plan.dag_node_parents[0] != -1 || plan.dag_node_powers[0] != -1 ||
        plan.dag_root_node_count < 0 ||
        plan.dag_root_node_count > static_cast<std::int64_t>(count - 1))
      throw std::invalid_argument("YE3T GPU prefix DAG has an invalid unit root");

    // Prefix node 0 is the unit. A child of the unit aliases its power input;
    // every other prefix node becomes one binary multiplication. This preserves
    // prefix sharing and multiplication order without expanding any monomials,
    // enumerating factor permutations, or recompiling the basis.
    std::vector<std::int64_t> operands(count, -1);
    const std::size_t products = count - 1 - static_cast<std::size_t>(plan.dag_root_node_count);
    gpu_schedule_index(plan.power_channels.size() + products);
    result.left.reserve(products);
    result.right.reserve(products);
    std::int64_t root_count = 0;
    for (std::size_t node = 1; node < count; ++node) {
      const std::int64_t parent = plan.dag_node_parents[node];
      const std::int64_t power = plan.dag_node_powers[node];
      if (parent < 0 || parent >= static_cast<std::int64_t>(node) || power < 0 ||
          power >= power_count)
        throw std::invalid_argument("YE3T GPU prefix DAG is not topologically ordered");
      if (parent == 0) {
        ++root_count;
        operands[node] = power;
      } else {
        operands[node] = gpu_schedule_index(plan.power_channels.size() + result.left.size());
        result.left.push_back(operands[static_cast<std::size_t>(parent)]);
        result.right.push_back(power);
      }
    }
    if (root_count != plan.dag_root_node_count)
      throw std::invalid_argument("YE3T GPU prefix DAG root count is inconsistent");
    result.monomial_operands.reserve(plan.monomial_nodes.size());
    for (const auto node : plan.monomial_nodes) {
      if (node < 0 || node >= static_cast<std::int64_t>(count))
        throw std::invalid_argument("YE3T GPU prefix readout operand is out of range");
      // The device representation uses -1 for the constant monomial, not 0.
      result.monomial_operands.push_back(operands[static_cast<std::size_t>(node)]);
    }
  }
  if (result.left.size() != result.right.size())
    throw std::invalid_argument("YE3T GPU binary DAG operand tables differ in size");
  const int values = gpu_schedule_index(plan.power_channels.size() + result.left.size());
  for (std::size_t node = 0; node < result.left.size(); ++node) {
    const auto available = power_count + static_cast<std::int64_t>(node);
    if (result.left[node] < 0 || result.right[node] < 0 || result.left[node] >= available ||
        result.right[node] >= available)
      throw std::invalid_argument("YE3T GPU binary DAG is not topologically ordered");
  }
  for (const auto operand : result.monomial_operands)
    if (operand < -1 || operand >= values)
      throw std::invalid_argument("YE3T GPU binary readout operand is out of range");
  return result;
}

template <class IndexVector, class SourceVector>
GpuDagSchedule make_gpu_dag_schedule(int source_count, const IndexVector &power_channels,
                                     const SourceVector &channel_sources, const IndexVector &left,
                                     const IndexVector &right, const IndexVector &terminal,
                                     const std::vector<double> &coefficients)
{
  if (source_count < 0 || left.size() != right.size() || terminal.size() != coefficients.size())
    throw std::invalid_argument("invalid YE3T GPU DAG schedule dimensions");
  const int powers = gpu_schedule_index(power_channels.size());
  const int values = gpu_schedule_index(power_channels.size() + left.size());
  gpu_schedule_index(2 * left.size());
  GpuDagSchedule output;
  output.readout_seed.assign(values, 0.0);
  std::vector<std::vector<int>> parent_lists(values), sibling_lists(values);
  for (std::size_t node = 0; node < left.size(); ++node) {
    const int parent = powers + static_cast<int>(node);
    if (left[node] < 0 || right[node] < 0 || left[node] >= parent || right[node] >= parent)
      throw std::invalid_argument("YE3T GPU reverse schedule requires a topological DAG");
    const int a = static_cast<int>(left[node]), b = static_cast<int>(right[node]);
    parent_lists[a].push_back(parent);
    sibling_lists[a].push_back(b);
    parent_lists[b].push_back(parent);
    sibling_lists[b].push_back(a);
  }
  for (std::size_t term = 0; term < terminal.size(); ++term) {
    if (terminal[term] < -1 || terminal[term] >= values || !std::isfinite(coefficients[term]))
      throw std::invalid_argument("invalid YE3T GPU reverse readout term");
    if (terminal[term] >= 0) {
      double &seed = output.readout_seed[static_cast<std::size_t>(terminal[term])];
      seed += coefficients[term];
      if (!std::isfinite(seed))
        throw std::invalid_argument("non-finite YE3T GPU reverse readout seed");
    }
  }
  for (int value = 0; value < values; ++value) {
    // Reverse parent order matches the serial reverse-DAG traversal more closely.
    output.parents.insert(output.parents.end(), parent_lists[value].rbegin(),
                          parent_lists[value].rend());
    output.siblings.insert(output.siblings.end(), sibling_lists[value].rbegin(),
                           sibling_lists[value].rend());
    output.parent_offsets.push_back(gpu_schedule_index(output.parents.size()));
  }
  std::vector<std::vector<int>> source_lists(source_count);
  for (int power = 0; power < powers; ++power) {
    const auto channel = power_channels[power];
    if (channel < 0 || static_cast<std::size_t>(channel) >= channel_sources.size())
      throw std::invalid_argument("invalid YE3T GPU reverse power channel");
    const auto source = channel_sources[static_cast<std::size_t>(channel)];
    if (source < 0 || source >= source_count)
      throw std::invalid_argument("invalid YE3T GPU reverse source index");
    source_lists[static_cast<std::size_t>(source)].push_back(power);
  }
  for (const auto &list : source_lists) {
    output.source_powers.insert(output.source_powers.end(), list.begin(), list.end());
    output.source_power_offsets.push_back(gpu_schedule_index(output.source_powers.size()));
  }
  return output;
}

#ifdef KOKKOS_INLINE_FUNCTION
#define YE3T_GPU_DAG_INLINE KOKKOS_INLINE_FUNCTION
#else
#define YE3T_GPU_DAG_INLINE inline
#endif

// Also used by the host arithmetic tests. Plan/storage are lightweight view-like
// objects; no allocation, index discovery, or atomics occur inside this helper.
template <class Plan, class View>
YE3T_GPU_DAG_INLINE void gather_gpu_dag_adjoint(const Plan &plan, const View &real,
                                                const View &imaginary, const View &adjoint_real,
                                                const View &adjoint_imaginary, int value_base,
                                                int value, int stride, int lane,
                                                double &result_real, double &result_imaginary)
{
  const int global_value = value_base + value;
  result_real = plan.dag_readout_seed(global_value);
  result_imaginary = 0.0;
  for (int entry = plan.dag_parent_offsets(global_value);
       entry < plan.dag_parent_offsets(global_value + 1); ++entry) {
    const int parent = plan.dag_parents(entry) * stride + lane;
    const int sibling = plan.dag_siblings(entry) * stride + lane;
    const double ar = adjoint_real(parent), ai = adjoint_imaginary(parent);
    const double br = real(sibling), bi = imaginary(sibling);
    result_real += ar * br + ai * bi;
    result_imaginary += ai * br - ar * bi;
  }
}
#undef YE3T_GPU_DAG_INLINE

}    // namespace YE3T_LAMMPS

#endif    // LMP_YE3T_GPU_DAG_SCHEDULE_H
