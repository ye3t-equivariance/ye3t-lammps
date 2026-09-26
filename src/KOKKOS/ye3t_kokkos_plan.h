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

#ifndef LMP_YE3T_KOKKOS_PLAN_H
#define LMP_YE3T_KOKKOS_PLAN_H

// clang-format off
#include "ye3t_runtime_core.h"
#include "ye3t_sha256.h"
#include "ye3t_yace_model.h"
#include <Kokkos_Core.hpp>
#include "ye3t_gpu_dag_schedule.h"
#include "ye3t_gpu_block_schedule.h"
#include "ye3t_gpu_harmonic_stream.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
// clang-format on

namespace YE3T_LAMMPS {

struct YE3TKokkosBlockSpeciesRecord {
  int power_begin = 0;
  int power_count = 0;
  int plan_begin = 0;
  int plan_count = 0;
  int route_begin = 0;
  int route_count = 0;
  int power_storage_size = 0;
  int output_storage_size = 0;
};

struct YE3TKokkosBlockPowerRecord {
  int channel = 0;
  int maximum_exponent = 0;
  int storage_offset = 0;
};

struct YE3TKokkosBlockPlanRecord {
  int input_begin = 0;
  int input_count = 0;
  int monomial_begin = 0;
  int monomial_count = 0;
  int output_begin = 0;
  int output_count = 0;
  int output_storage_offset = 0;
  int output_L = 0;
  int flags = 0;
};

struct YE3TKokkosBlockRouteRecord {
  int term_begin = 0;
  int term_count = 0;
  int function_index = -1;
};

struct YE3TKokkosScalarSpeciesRecord {
  int base_begin = 0;
  int base_count = 0;
  int node_begin = 0;
  int node_count = 0;
  int route_begin = 0;
  int route_count = 0;
  int value_count = 0;
};

struct YE3TKokkosScalarBaseRecord {
  int term_begin = 0;
  int term_count = 0;
};

struct YE3TKokkosScalarNodeRecord {
  int left_value = -1;
  int right_value = -1;
  int exponent = 0;
  int base_index = -1;
};

struct YE3TKokkosScalarRouteRecord {
  int function_index = -1;
  int value_index = -1;
  double scale_real = 0.0;
  double scale_imaginary = 0.0;
};

struct YE3TKokkosCoupledSpeciesRecord {
  int plan_begin = 0;
  int plan_count = 0;
};

struct YE3TKokkosCoupledPlanRecord {
  int node_begin = 0;
  int node_count = 0;
  int leaf_begin = 0;
  int leaf_count = 0;
  int coefficient_begin = 0;
  int coefficient_count = 0;
  int readout_begin = 0;
  int readout_count = 0;
  int total_component_count = 0;
  int function_index = -1;
  int operation_estimate = 0;
};

struct YE3TKokkosCoupledNodeRecord {
  int component_offset = 0;
  int dimension = 0;
  int leaf_offset = -1;
  int coefficient_begin = 0;
  int coefficient_count = 0;
};

enum YE3TKokkosBlockPlanFlag : int {
  YE3T_BLOCK_REAL_COEFFICIENTS = 1,
  YE3T_BLOCK_CONJUGATE_HALF_OUTPUT = 2,
  YE3T_BLOCK_DIRECT_INPUT = 4
};

template <class DeviceType> struct YE3TKokkosSpeciesViews {
  using IntView = Kokkos::View<const int *, DeviceType>;
  using RealView = Kokkos::View<const double *, DeviceType>;

  int species_count = 0;
  IntView channel_offsets;
  IntView source_offsets;
  IntView power_offsets;
  IntView node_offsets;
  IntView monomial_offsets;
  IntView node_level_table_offsets;
  IntView node_level_offsets;
  IntView node_level_nodes;
  RealView reference_energies;
  RealView embedding_scales;
  RealView density_safe_limits;
  IntView full_channel_sources;
  IntView full_channel_transforms;
  IntView power_channels;
  IntView power_exponents;
  IntView binary_node_left;
  IntView binary_node_right;
  IntView monomial_nodes;
  RealView monomial_coefficients;
  // Owner-gather reverse DAG and source-power transpose, compiled once.
  IntView dag_value_offsets;
  IntView dag_parent_offsets;
  IntView dag_parents;
  IntView dag_siblings;
  RealView dag_readout_seed;
  IntView dag_source_power_offsets;
  IntView dag_source_powers;
};

template <class DeviceType> struct YE3TKokkosBondViews {
  using IntView = Kokkos::View<const int *, DeviceType>;
  using RealView = Kokkos::View<const double *, DeviceType>;

  IntView vjp_group_degrees;
  IntView vjp_group_orders;
  IntView vjp_bond_column_offsets;
  IntView vjp_column_group_offsets;
  IntView active_contracted_offsets;
  IntView active_contracted_functions;
  int species_count = 0;
  int bond_count = 0;
  IntView radial_counts;
  IntView angular_maxima;
  IntView radial_base_counts;
  IntView interval_counts;
  RealView cutoffs;
  IntView radial_spline_offsets;
  RealView radial_splines;
  IntView contracted_spline_offsets;
  RealView contracted_splines;
  IntView radial_map_offsets;
  IntView radial_channel_outputs;
  IntView radial_channel_indices;
  IntView angular_map_offsets;
  IntView angular_channel_outputs;
  IntView contracted_channel_indices;
  IntView angular_channel_indices;
  IntView vjp_bond_group_offsets;
  IntView vjp_group_term_offsets;
  IntView vjp_group_harmonics;
  IntView vjp_group_terms;
};

template <class DeviceType> struct YE3TKokkosBlockViews {
  using SpeciesView = Kokkos::View<const YE3TKokkosBlockSpeciesRecord *, DeviceType>;
  using PowerView = Kokkos::View<const YE3TKokkosBlockPowerRecord *, DeviceType>;
  using PlanView = Kokkos::View<const YE3TKokkosBlockPlanRecord *, DeviceType>;
  using RouteView = Kokkos::View<const YE3TKokkosBlockRouteRecord *, DeviceType>;
  using IntView = Kokkos::View<const int *, DeviceType>;
  using RealView = Kokkos::View<const double *, DeviceType>;

  IntView plan_tile_offsets;
  IntView tile_monomial_begin;
  IntView tile_monomial_count;
  IntView tile_output_offsets;
  IntView tile_outputs;
  IntView tile_coefficient_offsets;
  IntView tile_coefficients;
  IntView monomial_output_offsets;
  IntView monomial_outputs;
  IntView monomial_coefficients;

  SpeciesView species;
  PowerView powers;
  PlanView plans;
  RouteView routes;
  IntView plan_input_channels;
  IntView plan_input_power_offsets;
  IntView monomial_factor_offsets;
  IntView monomial_factor_components;
  IntView monomial_factor_exponents;
  IntView output_coefficient_offsets;
  IntView coefficient_terms;
  RealView coefficient_real;
  RealView coefficient_imaginary;
  IntView direct_input_channels;
  RealView direct_input_scales;
  IntView route_term_factor_offsets;
  IntView route_term_factor_plans;
  IntView route_term_factor_components;
  RealView route_term_coefficient_real;
  RealView route_term_coefficient_imaginary;
};

template <class DeviceType> struct YE3TKokkosScalarViews {
  using SpeciesView = Kokkos::View<const YE3TKokkosScalarSpeciesRecord *, DeviceType>;
  using BaseView = Kokkos::View<const YE3TKokkosScalarBaseRecord *, DeviceType>;
  using NodeView = Kokkos::View<const YE3TKokkosScalarNodeRecord *, DeviceType>;
  using RouteView = Kokkos::View<const YE3TKokkosScalarRouteRecord *, DeviceType>;
  using IntView = Kokkos::View<const int *, DeviceType>;
  using RealView = Kokkos::View<const double *, DeviceType>;

  SpeciesView species;
  BaseView bases;
  NodeView nodes;
  RouteView routes;
  IntView left_channels;
  IntView right_channels;
  RealView coefficient_real;
  RealView coefficient_imaginary;
};

template <class DeviceType> struct YE3TKokkosCoupledViews {
  using SpeciesView = Kokkos::View<const YE3TKokkosCoupledSpeciesRecord *, DeviceType>;
  using PlanView = Kokkos::View<const YE3TKokkosCoupledPlanRecord *, DeviceType>;
  using NodeView = Kokkos::View<const YE3TKokkosCoupledNodeRecord *, DeviceType>;
  using IntView = Kokkos::View<const int *, DeviceType>;
  using RealView = Kokkos::View<const double *, DeviceType>;

  SpeciesView species;
  PlanView plans;
  NodeView nodes;
  IntView leaf_input_channels;
  IntView coefficient_left_components;
  IntView coefficient_right_components;
  IntView coefficient_output_components;
  RealView coefficient_values;
  IntView readout_components;
  RealView readout_coefficients;
};

template <class DeviceType> struct YE3TKokkosAngularViews {
  using RealView = Kokkos::View<const double *, DeviceType>;

  int maximum_angular_momentum = 0;
  RealView plan;
  RealView recurrence;
};

template <class DeviceType> class YE3TKokkosDevicePlan {
  struct HostBlockTables {
    GpuBlockExecutionSchedule execution;
    std::vector<YE3TKokkosBlockSpeciesRecord> species;
    std::vector<YE3TKokkosBlockPowerRecord> powers;
    std::vector<YE3TKokkosBlockPlanRecord> plans;
    std::vector<YE3TKokkosBlockRouteRecord> routes;
    std::vector<int> plan_input_channels;
    std::vector<int> plan_input_power_offsets;
    std::vector<int> monomial_factor_offsets{0};
    std::vector<int> monomial_factor_components;
    std::vector<int> monomial_factor_exponents;
    std::vector<int> output_coefficient_offsets{0};
    std::vector<int> coefficient_terms;
    std::vector<double> coefficient_real;
    std::vector<double> coefficient_imaginary;
    std::vector<int> direct_input_channels;
    std::vector<double> direct_input_scales;
    std::vector<int> route_term_factor_offsets{0};
    std::vector<int> route_term_factor_plans;
    std::vector<int> route_term_factor_components;
    std::vector<double> route_term_coefficient_real;
    std::vector<double> route_term_coefficient_imaginary;
    std::vector<std::string> semantic_ids;
  };

  struct HostScalarTables {
    std::vector<YE3TKokkosScalarSpeciesRecord> species;
    std::vector<YE3TKokkosScalarBaseRecord> bases;
    std::vector<YE3TKokkosScalarNodeRecord> nodes;
    std::vector<YE3TKokkosScalarRouteRecord> routes;
    std::vector<int> left_channels;
    std::vector<int> right_channels;
    std::vector<double> coefficient_real;
    std::vector<double> coefficient_imaginary;
    std::vector<std::string> semantic_ids;
  };

  struct HostCoupledTables {
    std::vector<YE3TKokkosCoupledSpeciesRecord> species;
    std::vector<YE3TKokkosCoupledPlanRecord> plans;
    std::vector<YE3TKokkosCoupledNodeRecord> nodes;
    std::vector<int> leaf_input_channels;
    std::vector<int> coefficient_left_components;
    std::vector<int> coefficient_right_components;
    std::vector<int> coefficient_output_components;
    std::vector<double> coefficient_values;
    std::vector<int> readout_components;
    std::vector<double> readout_coefficients;
    std::vector<std::string> semantic_ids;
  };

 public:
  using IntView = Kokkos::View<int *, DeviceType>;
  using RealView = Kokkos::View<double *, DeviceType>;
  using BlockSpeciesView = Kokkos::View<YE3TKokkosBlockSpeciesRecord *, DeviceType>;
  using BlockPowerView = Kokkos::View<YE3TKokkosBlockPowerRecord *, DeviceType>;
  using BlockPlanView = Kokkos::View<YE3TKokkosBlockPlanRecord *, DeviceType>;
  using BlockRouteView = Kokkos::View<YE3TKokkosBlockRouteRecord *, DeviceType>;
  using ScalarSpeciesView = Kokkos::View<YE3TKokkosScalarSpeciesRecord *, DeviceType>;
  using ScalarBaseView = Kokkos::View<YE3TKokkosScalarBaseRecord *, DeviceType>;
  using ScalarNodeView = Kokkos::View<YE3TKokkosScalarNodeRecord *, DeviceType>;
  using ScalarRouteView = Kokkos::View<YE3TKokkosScalarRouteRecord *, DeviceType>;
  using CoupledSpeciesView = Kokkos::View<YE3TKokkosCoupledSpeciesRecord *, DeviceType>;
  using CoupledPlanView = Kokkos::View<YE3TKokkosCoupledPlanRecord *, DeviceType>;
  using CoupledNodeView = Kokkos::View<YE3TKokkosCoupledNodeRecord *, DeviceType>;

  void upload(const YACEModel &model, std::size_t device_byte_budget)
  {
    if (model.species_count() <= 0 || model.bond_count() <= 0)
      throw std::invalid_argument("YE3T Kokkos plan requires a populated model");

    species_count = model.species_count();
    bond_count = model.bond_count();
    maximum_angular_momentum = model.maximum_angular_momentum();
    const std::size_t expected_bonds =
        checked_product(static_cast<std::size_t>(species_count),
                        static_cast<std::size_t>(species_count), "bond count");
    if (static_cast<std::size_t>(bond_count) != expected_bonds)
      throw std::invalid_argument("YE3T Kokkos plan requires one directed bond per species pair");

    maximum_channel_count_ = 0;
    maximum_source_count_ = 0;
    maximum_power_count_ = 0;
    maximum_node_count_ = 0;
    maximum_monomial_count_ = 0;
    maximum_block_power_storage_ = 0;
    maximum_block_output_storage_ = 0;
    maximum_block_monomial_count_ = 0;
    maximum_block_power_count_ = 0;
    maximum_block_plan_count_ = 0;
    maximum_block_route_count_ = 0;
    block_route_count_ = 0;
    maximum_scalar_value_count_ = 0;
    scalar_route_count_ = 0;
    maximum_coupled_component_count_ = 0;
    coupled_plan_count_ = 0;

    std::vector<int> host_species_channel_offsets(1, 0);
    std::vector<int> host_species_source_offsets(1, 0);
    std::vector<int> host_species_power_offsets(1, 0);
    std::vector<int> host_species_node_offsets(1, 0);
    std::vector<int> host_species_monomial_offsets(1, 0);
    std::vector<int> host_species_node_level_table_offsets(1, 0);
    std::vector<int> host_node_level_offsets;
    std::vector<int> host_node_level_nodes;
    std::vector<double> host_reference_energies;
    std::vector<double> host_embedding_scales;
    std::vector<double> host_density_safe_limits;
    std::vector<int> host_full_channel_sources;
    std::vector<int> host_full_channel_transforms;
    std::vector<int> host_power_channels;
    std::vector<int> host_power_exponents;
    std::vector<int> host_binary_node_left;
    std::vector<int> host_binary_node_right;
    std::vector<int> host_monomial_nodes;
    std::vector<double> host_monomial_coefficients;
    std::vector<int> host_dag_value_offsets(1, 0);
    std::vector<int> host_dag_parent_offsets(1, 0);
    std::vector<int> host_dag_parents;
    std::vector<int> host_dag_siblings;
    std::vector<double> host_dag_readout_seed;
    std::vector<int> host_dag_source_power_offsets(1, 0);
    std::vector<int> host_dag_source_powers;

    for (int species_index = 0; species_index < species_count; ++species_index) {
      const auto &species = model.species(species_index);
      const auto &polynomial = species.polynomial;
      const int channel_count = checked_size(species.channels.size(), "channel table");
      const int source_count = checked_size(species.source_channels.size(), "source-channel table");
      const bool has_direct = !polynomial.monomial_coefficients.empty();
      const bool has_block = !species.block_program.routes.empty();
      const bool has_scalar = !species.block_program.scalar_program.routes.empty();
      const bool has_coupled = !species.block_program.coupled_product_plans.empty();
      // The shared loader chooses either a prefix or a binary schedule using a
      // CPU-oriented cost rule. Lower only the selected residual here, before
      // sizing, level scheduling, reverse scheduling, hashing, or device upload.
      const auto binary = lower_gpu_binary_dag(polynomial);
      if (species.channels.empty() || species.source_channels.empty())
        throw std::invalid_argument("YE3T Kokkos foundation requires nonempty channel tables");
      if (!has_direct && !has_block && !has_scalar && !has_coupled)
        throw std::invalid_argument("YE3T Kokkos species has no executable evaluator routes");
      if (species.full_channel_sources.size() != species.channels.size() ||
          species.full_channel_transforms.size() != species.channels.size())
        throw std::invalid_argument("YE3T Kokkos channel map is inconsistent");
      if (polynomial.power_channels.size() != polynomial.power_exponents.size() ||
          binary.left.size() != binary.right.size() ||
          binary.monomial_operands.size() != polynomial.monomial_coefficients.size())
        throw std::invalid_argument("YE3T Kokkos direct polynomial tables are inconsistent");
      if (!std::isfinite(species.reference_energy) || !std::isfinite(species.embedding_scale) ||
          !std::isfinite(species.density_safe_limit))
        throw std::invalid_argument("YE3T Kokkos species scalars must be finite");

      host_reference_energies.push_back(species.reference_energy);
      host_embedding_scales.push_back(species.embedding_scale);
      host_density_safe_limits.push_back(species.density_safe_limit);

      for (std::size_t channel = 0; channel < species.channels.size(); ++channel) {
        const int source = species.full_channel_sources[channel];
        const int transform = species.full_channel_transforms[channel];
        if (source < 0 || source >= static_cast<int>(species.source_channels.size()) ||
            (transform != -1 && transform != 0 && transform != 1))
          throw std::invalid_argument("YE3T Kokkos full-channel transform is out of range");
        host_full_channel_sources.push_back(source);
        host_full_channel_transforms.push_back(transform);
      }

      const int power_count = checked_size(polynomial.power_channels.size(), "power table");
      const int node_count = checked_size(binary.left.size(), "binary DAG");
      const int monomial_count =
          checked_size(polynomial.monomial_coefficients.size(), "monomial table");
      maximum_channel_count_ = std::max(maximum_channel_count_, channel_count);
      maximum_source_count_ = std::max(maximum_source_count_, source_count);
      maximum_power_count_ = std::max(maximum_power_count_, power_count);
      maximum_node_count_ = std::max(maximum_node_count_, node_count);
      maximum_monomial_count_ = std::max(maximum_monomial_count_, monomial_count);
      for (int power = 0; power < power_count; ++power) {
        const int channel = checked_index(
            polynomial.power_channels[static_cast<std::size_t>(power)], "power channel");
        const int exponent = checked_index(
            polynomial.power_exponents[static_cast<std::size_t>(power)], "power exponent");
        if (channel < 0 || channel >= static_cast<int>(species.channels.size()) || exponent <= 0)
          throw std::invalid_argument("YE3T Kokkos power entry is out of range");
        host_power_channels.push_back(channel);
        host_power_exponents.push_back(exponent);
      }
      std::vector<int> value_levels(static_cast<std::size_t>(power_count), 0);
      std::vector<std::vector<int>> nodes_by_level;
      for (int node = 0; node < node_count; ++node) {
        const int left =
            checked_index(binary.left[static_cast<std::size_t>(node)], "binary DAG left operand");
        const int right =
            checked_index(binary.right[static_cast<std::size_t>(node)], "binary DAG right operand");
        const int available = power_count + node;
        if (left < 0 || right < 0 || left >= available || right >= available)
          throw std::invalid_argument("YE3T Kokkos binary DAG is not topologically ordered");
        const int level = 1 +
            std::max(value_levels[static_cast<std::size_t>(left)],
                     value_levels[static_cast<std::size_t>(right)]);
        value_levels.push_back(level);
        if (nodes_by_level.size() < static_cast<std::size_t>(level))
          nodes_by_level.resize(static_cast<std::size_t>(level));
        nodes_by_level[static_cast<std::size_t>(level - 1)].push_back(node);
        host_binary_node_left.push_back(left);
        host_binary_node_right.push_back(right);
      }
      for (const auto &level_nodes : nodes_by_level) {
        host_node_level_offsets.push_back(
            checked_size(host_node_level_nodes.size(), "node-level nodes"));
        for (const int node : level_nodes) host_node_level_nodes.push_back(node);
      }
      host_node_level_offsets.push_back(
          checked_size(host_node_level_nodes.size(), "node-level nodes"));
      append_offset(host_species_node_level_table_offsets, nodes_by_level.size() + 1,
                    "node-level offset table");
      for (std::size_t term = 0; term < binary.monomial_operands.size(); ++term) {
        const int operand = checked_index(binary.monomial_operands[term], "monomial operand");
        if (operand < -1 || operand >= power_count + node_count ||
            !std::isfinite(polynomial.monomial_coefficients[term]))
          throw std::invalid_argument("YE3T Kokkos monomial entry is out of range");
        host_monomial_nodes.push_back(operand);
        host_monomial_coefficients.push_back(polynomial.monomial_coefficients[term]);
      }

      const auto reverse = make_gpu_dag_schedule(
          source_count, polynomial.power_channels, species.full_channel_sources, binary.left,
          binary.right, binary.monomial_operands, polynomial.monomial_coefficients);
      const int parent_base = checked_size(host_dag_parents.size(), "DAG reverse entries");
      for (std::size_t i = 1; i < reverse.parent_offsets.size(); ++i)
        host_dag_parent_offsets.push_back(
            checked_size(static_cast<std::size_t>(parent_base) + reverse.parent_offsets[i],
                         "DAG reverse offset"));
      host_dag_parents.insert(host_dag_parents.end(), reverse.parents.begin(),
                              reverse.parents.end());
      host_dag_siblings.insert(host_dag_siblings.end(), reverse.siblings.begin(),
                               reverse.siblings.end());
      host_dag_readout_seed.insert(host_dag_readout_seed.end(), reverse.readout_seed.begin(),
                                   reverse.readout_seed.end());
      host_dag_value_offsets.push_back(
          checked_size(host_dag_readout_seed.size(), "DAG reverse values"));
      const int source_base = checked_size(host_dag_source_powers.size(), "DAG source powers");
      for (std::size_t i = 1; i < reverse.source_power_offsets.size(); ++i)
        host_dag_source_power_offsets.push_back(
            checked_size(static_cast<std::size_t>(source_base) + reverse.source_power_offsets[i],
                         "DAG source-power offset"));
      host_dag_source_powers.insert(host_dag_source_powers.end(), reverse.source_powers.begin(),
                                    reverse.source_powers.end());

      append_offset(host_species_channel_offsets, species.channels.size(), "channel table");
      append_offset(host_species_source_offsets, species.source_channels.size(),
                    "source-channel table");
      append_offset(host_species_power_offsets, polynomial.power_channels.size(), "power table");
      append_offset(host_species_node_offsets, binary.left.size(), "binary DAG");
      append_offset(host_species_monomial_offsets, binary.monomial_operands.size(),
                    "monomial table");
    }

    HostBlockTables host_block;
    flatten_block_programs(model, host_block);
    host_block.execution = make_gpu_block_execution_schedule(
        host_block.plans, host_block.output_coefficient_offsets, host_block.coefficient_terms,
        checked_size(host_block.monomial_factor_offsets.size() - 1, "block monomials"),
        YE3T_BLOCK_DIRECT_INPUT, YE3T_BLOCK_CONJUGATE_HALF_OUTPUT);
    HostScalarTables host_scalar;
    flatten_scalar_programs(model, host_scalar);
    HostCoupledTables host_coupled;
    flatten_coupled_programs(model, host_coupled);

    std::vector<int> host_bond_radial_counts;
    std::vector<int> host_bond_angular_maxima;
    std::vector<int> host_bond_radial_base_counts;
    std::vector<int> host_bond_interval_counts;
    std::vector<double> host_bond_cutoffs;
    std::vector<int> host_radial_spline_offsets(1, 0);
    std::vector<double> host_radial_splines;
    std::vector<int> host_contracted_spline_offsets(1, 0);
    std::vector<double> host_contracted_splines;
    std::vector<int> host_radial_map_offsets(1, 0);
    std::vector<int> host_radial_channel_outputs;
    std::vector<int> host_radial_channel_indices;
    std::vector<int> host_angular_map_offsets(1, 0);
    std::vector<int> host_angular_channel_outputs;
    std::vector<int> host_contracted_channel_indices;
    std::vector<int> host_angular_channel_indices;
    std::vector<int> host_vjp_group_degrees, host_vjp_group_orders;
    std::vector<int> host_vjp_bond_column_offsets(1, 0), host_vjp_column_group_offsets(1, 0);
    std::vector<int> host_active_contracted_offsets(1, 0), host_active_contracted_functions;
    std::vector<int> host_vjp_bond_group_offsets(1, 0);
    std::vector<int> host_vjp_group_term_offsets(1, 0);
    std::vector<int> host_vjp_group_harmonics;
    std::vector<int> host_vjp_group_terms;

    for (int central = 0; central < species_count; ++central) {
      for (int neighbor = 0; neighbor < species_count; ++neighbor) {
        const auto &bond = model.bond(central, neighbor);
        if (bond.central_species != central || bond.neighbor_species != neighbor ||
            bond.radial_count <= 0 || bond.radial_base_count <= 0 || bond.angular_maximum < 0 ||
            bond.spline_interval_count <= 0 ||
            bond.spline_interval_count > std::numeric_limits<int>::max() || !(bond.cutoff > 0.0) ||
            !std::isfinite(bond.cutoff))
          throw std::invalid_argument("YE3T Kokkos bond metadata is inconsistent");
        if (bond.radial_channel_outputs.size() != bond.radial_channel_indices.size() ||
            bond.angular_channel_outputs.size() != bond.contracted_channel_indices.size() ||
            bond.angular_channel_outputs.size() !=
                bond.angular_channel_nonnegative_indices.size() ||
            bond.radial_base_spline.empty() || bond.contracted_spline.empty())
          throw std::invalid_argument("YE3T Kokkos bond tables are inconsistent");
        if (!all_finite(bond.radial_base_spline) || !all_finite(bond.contracted_spline))
          throw std::invalid_argument("YE3T Kokkos spline coefficients must be finite");

        const int source_count = checked_size(model.species(central).source_channels.size(),
                                              "bond source-channel table");
        const std::size_t interval_width = checked_add(
            static_cast<std::size_t>(bond.spline_interval_count), 1, "spline interval width");
        const std::size_t expected_radial_spline = checked_product(
            checked_product(interval_width, static_cast<std::size_t>(bond.radial_base_count),
                            "radial spline table"),
            4, "radial spline table");
        const std::size_t contracted_width =
            checked_product(static_cast<std::size_t>(bond.radial_count),
                            static_cast<std::size_t>(bond.angular_maximum + 1), "contracted width");
        if (contracted_width > static_cast<std::size_t>(std::numeric_limits<int>::max()))
          throw std::overflow_error("YE3T Kokkos contracted width exceeds 32-bit indexing");
        const std::size_t expected_contracted_spline = checked_product(
            checked_product(interval_width, contracted_width, "contracted spline table"), 4,
            "contracted spline table");
        if (bond.radial_base_spline.size() != expected_radial_spline ||
            bond.contracted_spline.size() != expected_contracted_spline)
          throw std::invalid_argument("YE3T Kokkos spline-table extent is inconsistent");
        for (std::size_t term = 0; term < bond.radial_channel_outputs.size(); ++term) {
          const int output = bond.radial_channel_outputs[term];
          const int radial = bond.radial_channel_indices[term];
          if (output < 0 || output >= source_count || radial < 0 ||
              radial >= bond.radial_base_count)
            throw std::invalid_argument("YE3T Kokkos radial-channel map is out of range");
          host_radial_channel_outputs.push_back(output);
          host_radial_channel_indices.push_back(radial);
        }
        const std::int64_t angular_width_64 =
            ye3t::runtime::complex_spherical_harmonics_nonnegative_table_width(
                bond.angular_maximum);
        if (angular_width_64 > std::numeric_limits<int>::max())
          throw std::overflow_error("YE3T Kokkos angular width exceeds 32-bit indexing");
        const int angular_width = static_cast<int>(angular_width_64);
        for (std::size_t term = 0; term < bond.angular_channel_outputs.size(); ++term) {
          const int output = bond.angular_channel_outputs[term];
          const int contracted = bond.contracted_channel_indices[term];
          const int angular = bond.angular_channel_nonnegative_indices[term];
          if (output < 0 || output >= source_count || contracted < 0 ||
              contracted >= bond.contracted_width() || angular < 0 || angular >= angular_width)
            throw std::invalid_argument("YE3T Kokkos angular-channel map is out of range");
          host_angular_channel_outputs.push_back(output);
          host_contracted_channel_indices.push_back(contracted);
          host_angular_channel_indices.push_back(angular);
        }

        host_bond_radial_counts.push_back(bond.radial_count);
        host_bond_angular_maxima.push_back(bond.angular_maximum);
        host_bond_radial_base_counts.push_back(bond.radial_base_count);
        host_bond_interval_counts.push_back(static_cast<int>(bond.spline_interval_count));
        host_bond_cutoffs.push_back(bond.cutoff);
        append_values(host_radial_splines, bond.radial_base_spline);
        append_values(host_contracted_splines, bond.contracted_spline);
        append_offset(host_radial_spline_offsets, bond.radial_base_spline.size(),
                      "radial spline table");
        append_offset(host_contracted_spline_offsets, bond.contracted_spline.size(),
                      "contracted spline table");
        append_offset(host_radial_map_offsets, bond.radial_channel_outputs.size(),
                      "radial-channel map");
        append_offset(host_angular_map_offsets, bond.angular_channel_outputs.size(),
                      "angular-channel map");
      }
    }

    auto harmonic_labels = [](int component) {
      int l = 0;
      while (component >= static_cast<std::int64_t>(l + 1) * (l + 2) / 2) ++l;
      return std::pair<int, int>{component - l * (l + 1) / 2, l};    // order, degree
    };
    maximum_source_serial_work_count_ = 0;
    vjp_maximum_group_term_count_ = 0;
    for (int bond = 0; bond < bond_count; ++bond) {
      const int begin = host_angular_map_offsets[static_cast<std::size_t>(bond)];
      const int end = host_angular_map_offsets[static_cast<std::size_t>(bond + 1)];
      if (begin < 0 || end < begin ||
          static_cast<std::size_t>(end) > host_angular_channel_indices.size())
        throw std::invalid_argument("YE3T Kokkos VJP angular-map range is invalid");

      const std::size_t bond_group_begin = host_vjp_group_harmonics.size();
      std::vector<int> ordered_terms;
      ordered_terms.reserve(static_cast<std::size_t>(end - begin));
      for (int term = begin; term < end; ++term) ordered_terms.push_back(term);
      std::stable_sort(ordered_terms.begin(), ordered_terms.end(), [&](int first, int second) {
        const int first_harmonic = host_angular_channel_indices[static_cast<std::size_t>(first)];
        const int second_harmonic = host_angular_channel_indices[static_cast<std::size_t>(second)];
        if (first_harmonic != second_harmonic)
          return harmonic_labels(first_harmonic) < harmonic_labels(second_harmonic);
        return first < second;
      });

      std::vector<unsigned char> covered(static_cast<std::size_t>(end - begin), 0);
      std::size_t position = 0;
      while (position < ordered_terms.size()) {
        const int harmonic =
            host_angular_channel_indices[static_cast<std::size_t>(ordered_terms[position])];
        const std::size_t group_begin = host_vjp_group_terms.size();
        host_vjp_group_harmonics.push_back(harmonic);
        const auto labels = harmonic_labels(harmonic);
        host_vjp_group_orders.push_back(labels.first);
        host_vjp_group_degrees.push_back(labels.second);
        while (position < ordered_terms.size() &&
               host_angular_channel_indices[static_cast<std::size_t>(ordered_terms[position])] ==
                   harmonic) {
          const int term = ordered_terms[position++];
          if (term < begin || term >= end || covered[static_cast<std::size_t>(term - begin)] != 0)
            throw std::invalid_argument("YE3T Kokkos VJP schedule term coverage is invalid");
          covered[static_cast<std::size_t>(term - begin)] = 1;
          host_vjp_group_terms.push_back(term);
        }
        const std::size_t group_size = host_vjp_group_terms.size() - group_begin;
        append_offset(host_vjp_group_term_offsets, group_size, "VJP group-term table");
        vjp_maximum_group_term_count_ =
            std::max(vjp_maximum_group_term_count_, checked_size(group_size, "VJP group size"));
      }
      if (std::find(covered.begin(), covered.end(), 0) != covered.end())
        throw std::invalid_argument("YE3T Kokkos VJP schedule does not cover every angular term");
      append_offset(host_vjp_bond_group_offsets,
                    host_vjp_group_harmonics.size() -
                        static_cast<std::size_t>(host_vjp_bond_group_offsets.back()),
                    "VJP bond-group table");
      // Columns permit edge teams to share the recurrence along l without
      // evaluating unused (l,m) outputs. Neighbor-major and VJP stream the same
      // groups in increasing (m,l), with only a few live recurrence states.
      for (int group = static_cast<int>(bond_group_begin);
           group < static_cast<int>(host_vjp_group_harmonics.size());) {
        const int order = host_vjp_group_orders[group];
        do {
          ++group;
        } while (group < static_cast<int>(host_vjp_group_harmonics.size()) &&
                 host_vjp_group_orders[group] == order);
        host_vjp_column_group_offsets.push_back(group);
      }
      host_vjp_bond_column_offsets.push_back(
          checked_size(host_vjp_column_group_offsets.size() - 1, "harmonic columns"));
      std::vector<int> active_radials;
      for (int term = begin; term < end; ++term)
        active_radials.push_back(host_contracted_channel_indices[term]);
      std::sort(active_radials.begin(), active_radials.end());
      active_radials.erase(std::unique(active_radials.begin(), active_radials.end()),
                           active_radials.end());
      host_active_contracted_functions.insert(host_active_contracted_functions.end(),
                                              active_radials.begin(), active_radials.end());
      host_active_contracted_offsets.push_back(
          checked_size(host_active_contracted_functions.size(), "active contracted functions"));
      const std::size_t radial_term_count =
          static_cast<std::size_t>(host_radial_map_offsets[static_cast<std::size_t>(bond + 1)] -
                                   host_radial_map_offsets[static_cast<std::size_t>(bond)]);
      const std::size_t angular_term_count = static_cast<std::size_t>(end - begin);
      const std::size_t harmonic_group_count = host_vjp_group_harmonics.size() - bond_group_begin;
      const std::size_t serial_work =
          checked_add(checked_add(radial_term_count, active_radials.size(), "source serial work"),
                      checked_add(harmonic_group_count, angular_term_count, "source serial work"),
                      "source serial work");
      maximum_source_serial_work_count_ = std::max(maximum_source_serial_work_count_,
                                                   checked_size(serial_work, "source serial work"));
    }
    if (host_vjp_group_terms.size() != host_angular_channel_indices.size())
      throw std::invalid_argument("YE3T Kokkos VJP schedule term count is inconsistent");
    vjp_harmonic_group_count_ =
        checked_size(host_vjp_group_harmonics.size(), "VJP harmonic group table");
    vjp_angular_term_count_ = checked_size(host_vjp_group_terms.size(), "VJP angular term table");

    const std::int64_t angular_plan_size =
        ye3t::runtime::complex_spherical_harmonics_table_plan_size(maximum_angular_momentum);
    if (angular_plan_size <= 0 || angular_plan_size > std::numeric_limits<int>::max())
      throw std::overflow_error("YE3T Kokkos angular plan is too large");
    std::vector<double> host_angular_plan(static_cast<std::size_t>(angular_plan_size));
    ye3t::runtime::build_complex_spherical_harmonics_table_plan<double>(
        maximum_angular_momentum, host_angular_plan.data(), angular_plan_size,
        std::sqrt(4.0 * std::acos(-1.0)));

    const auto recurrence_size =
        ye3t::runtime::complex_spherical_harmonics_recurrence_plan_size(maximum_angular_momentum);
    if (recurrence_size <= 0 || recurrence_size > std::numeric_limits<int>::max())
      throw std::overflow_error("YE3T Kokkos harmonic recurrence is too large");
    std::vector<double> host_angular_recurrence(static_cast<std::size_t>(recurrence_size));
    ye3t::runtime::build_complex_spherical_harmonics_recurrence_plan<double>(
        maximum_angular_momentum, host_angular_recurrence.data(), recurrence_size,
        std::sqrt(4.0 * std::acos(-1.0)));
    static_bytes_ = 0;
    count_bytes<double>(host_angular_recurrence.size());
    count_bytes<int>(host_vjp_group_degrees.size());
    count_bytes<int>(host_vjp_group_orders.size());
    count_bytes<int>(host_vjp_bond_column_offsets.size());
    count_bytes<int>(host_vjp_column_group_offsets.size());
    count_bytes<int>(host_active_contracted_offsets.size());
    count_bytes<int>(host_active_contracted_functions.size());
    count_bytes<int>(host_dag_value_offsets.size());
    count_bytes<int>(host_dag_parent_offsets.size());
    count_bytes<int>(host_dag_parents.size());
    count_bytes<int>(host_dag_siblings.size());
    count_bytes<double>(host_dag_readout_seed.size());
    count_bytes<int>(host_dag_source_power_offsets.size());
    count_bytes<int>(host_dag_source_powers.size());

    count_bytes<int>(host_species_channel_offsets.size());
    count_bytes<int>(host_species_source_offsets.size());
    count_bytes<int>(host_species_power_offsets.size());
    count_bytes<int>(host_species_node_offsets.size());
    count_bytes<int>(host_species_monomial_offsets.size());
    count_bytes<int>(host_species_node_level_table_offsets.size());
    count_bytes<int>(host_node_level_offsets.size());
    count_bytes<int>(host_node_level_nodes.size());
    count_bytes<double>(host_reference_energies.size());
    count_bytes<double>(host_embedding_scales.size());
    count_bytes<double>(host_density_safe_limits.size());
    count_bytes<int>(host_full_channel_sources.size());
    count_bytes<int>(host_full_channel_transforms.size());
    count_bytes<int>(host_power_channels.size());
    count_bytes<int>(host_power_exponents.size());
    count_bytes<int>(host_binary_node_left.size());
    count_bytes<int>(host_binary_node_right.size());
    count_bytes<int>(host_monomial_nodes.size());
    count_bytes<double>(host_monomial_coefficients.size());
    count_bytes<int>(host_bond_radial_counts.size());
    count_bytes<int>(host_bond_angular_maxima.size());
    count_bytes<int>(host_bond_radial_base_counts.size());
    count_bytes<int>(host_bond_interval_counts.size());
    count_bytes<double>(host_bond_cutoffs.size());
    count_bytes<int>(host_radial_spline_offsets.size());
    count_bytes<double>(host_radial_splines.size());
    count_bytes<int>(host_contracted_spline_offsets.size());
    count_bytes<double>(host_contracted_splines.size());
    count_bytes<int>(host_radial_map_offsets.size());
    count_bytes<int>(host_radial_channel_outputs.size());
    count_bytes<int>(host_radial_channel_indices.size());
    count_bytes<int>(host_angular_map_offsets.size());
    count_bytes<int>(host_angular_channel_outputs.size());
    count_bytes<int>(host_contracted_channel_indices.size());
    count_bytes<int>(host_angular_channel_indices.size());
    count_bytes<int>(host_vjp_bond_group_offsets.size());
    count_bytes<int>(host_vjp_group_term_offsets.size());
    count_bytes<int>(host_vjp_group_harmonics.size());
    count_bytes<int>(host_vjp_group_terms.size());
    count_bytes<double>(host_angular_plan.size());
    count_bytes<int>(host_block.execution.plan_tile_offsets.size());
    count_bytes<int>(host_block.execution.tile_monomial_begin.size());
    count_bytes<int>(host_block.execution.tile_monomial_count.size());
    count_bytes<int>(host_block.execution.tile_output_offsets.size());
    count_bytes<int>(host_block.execution.tile_outputs.size());
    count_bytes<int>(host_block.execution.tile_coefficient_offsets.size());
    count_bytes<int>(host_block.execution.tile_coefficients.size());
    count_bytes<int>(host_block.execution.monomial_output_offsets.size());
    count_bytes<int>(host_block.execution.monomial_outputs.size());
    count_bytes<int>(host_block.execution.monomial_coefficients.size());
    count_bytes<YE3TKokkosBlockSpeciesRecord>(host_block.species.size());
    count_bytes<YE3TKokkosBlockPowerRecord>(host_block.powers.size());
    count_bytes<YE3TKokkosBlockPlanRecord>(host_block.plans.size());
    count_bytes<YE3TKokkosBlockRouteRecord>(host_block.routes.size());
    count_bytes<int>(host_block.plan_input_channels.size());
    count_bytes<int>(host_block.plan_input_power_offsets.size());
    count_bytes<int>(host_block.monomial_factor_offsets.size());
    count_bytes<int>(host_block.monomial_factor_components.size());
    count_bytes<int>(host_block.monomial_factor_exponents.size());
    count_bytes<int>(host_block.output_coefficient_offsets.size());
    count_bytes<int>(host_block.coefficient_terms.size());
    count_bytes<double>(host_block.coefficient_real.size());
    count_bytes<double>(host_block.coefficient_imaginary.size());
    count_bytes<int>(host_block.direct_input_channels.size());
    count_bytes<double>(host_block.direct_input_scales.size());
    count_bytes<int>(host_block.route_term_factor_offsets.size());
    count_bytes<int>(host_block.route_term_factor_plans.size());
    count_bytes<int>(host_block.route_term_factor_components.size());
    count_bytes<double>(host_block.route_term_coefficient_real.size());
    count_bytes<double>(host_block.route_term_coefficient_imaginary.size());
    count_bytes<YE3TKokkosScalarSpeciesRecord>(host_scalar.species.size());
    count_bytes<YE3TKokkosScalarBaseRecord>(host_scalar.bases.size());
    count_bytes<YE3TKokkosScalarNodeRecord>(host_scalar.nodes.size());
    count_bytes<YE3TKokkosScalarRouteRecord>(host_scalar.routes.size());
    count_bytes<int>(host_scalar.left_channels.size());
    count_bytes<int>(host_scalar.right_channels.size());
    count_bytes<double>(host_scalar.coefficient_real.size());
    count_bytes<double>(host_scalar.coefficient_imaginary.size());
    count_bytes<YE3TKokkosCoupledSpeciesRecord>(host_coupled.species.size());
    count_bytes<YE3TKokkosCoupledPlanRecord>(host_coupled.plans.size());
    count_bytes<YE3TKokkosCoupledNodeRecord>(host_coupled.nodes.size());
    count_bytes<int>(host_coupled.leaf_input_channels.size());
    count_bytes<int>(host_coupled.coefficient_left_components.size());
    count_bytes<int>(host_coupled.coefficient_right_components.size());
    count_bytes<int>(host_coupled.coefficient_output_components.size());
    count_bytes<double>(host_coupled.coefficient_values.size());
    count_bytes<int>(host_coupled.readout_components.size());
    count_bytes<double>(host_coupled.readout_coefficients.size());
    const std::size_t minimum_resident =
        checked_add(static_bytes_, minimum_dynamic_bytes(), "minimum resident storage");
    if (minimum_resident > device_byte_budget)
      throw std::length_error("YE3T Kokkos plan cannot preserve 15 percent "
                              "free device-memory headroom");

    const std::string direct_signature = signature_hash(
        host_species_channel_offsets, host_species_source_offsets, host_species_power_offsets,
        host_species_node_offsets, host_species_monomial_offsets,
        host_species_node_level_table_offsets, host_node_level_offsets, host_node_level_nodes,
        host_reference_energies, host_embedding_scales, host_density_safe_limits,
        host_full_channel_sources, host_full_channel_transforms, host_power_channels,
        host_power_exponents, host_binary_node_left, host_binary_node_right, host_monomial_nodes,
        host_monomial_coefficients, host_bond_radial_counts, host_bond_angular_maxima,
        host_bond_radial_base_counts, host_bond_interval_counts, host_bond_cutoffs,
        host_radial_spline_offsets, host_radial_splines, host_contracted_spline_offsets,
        host_contracted_splines, host_radial_map_offsets, host_radial_channel_outputs,
        host_radial_channel_indices, host_angular_map_offsets, host_angular_channel_outputs,
        host_contracted_channel_indices, host_angular_channel_indices, host_angular_plan,
        host_dag_value_offsets, host_dag_parent_offsets, host_dag_parents, host_dag_siblings,
        host_dag_readout_seed, host_dag_source_power_offsets, host_dag_source_powers);
    vjp_schedule_hash_ = vjp_schedule_signature_hash(
        direct_signature, host_vjp_bond_group_offsets, host_vjp_group_term_offsets,
        host_vjp_group_harmonics, host_vjp_group_terms);
    signature_hash_ = host_block.routes.empty()
        ? direct_signature
        : block_signature_hash(direct_signature, host_block);
    if (!host_scalar.routes.empty())
      signature_hash_ = scalar_signature_hash(signature_hash_, host_scalar);
    if (!host_coupled.plans.empty())
      signature_hash_ = coupled_signature_hash(signature_hash_, host_coupled);
    const auto direct_probe = probe_values(
        host_species_channel_offsets, host_species_source_offsets, host_species_power_offsets,
        host_species_node_offsets, host_species_monomial_offsets,
        host_species_node_level_table_offsets, host_node_level_offsets, host_node_level_nodes,
        host_reference_energies, host_embedding_scales, host_density_safe_limits,
        host_full_channel_sources, host_full_channel_transforms, host_power_channels,
        host_power_exponents, host_binary_node_left, host_binary_node_right, host_monomial_nodes,
        host_monomial_coefficients, host_bond_radial_counts, host_bond_angular_maxima,
        host_bond_radial_base_counts, host_bond_interval_counts, host_bond_cutoffs,
        host_radial_spline_offsets, host_radial_splines, host_contracted_spline_offsets,
        host_contracted_splines, host_radial_map_offsets, host_radial_channel_outputs,
        host_radial_channel_indices, host_angular_map_offsets, host_angular_channel_outputs,
        host_contracted_channel_indices, host_angular_channel_indices, host_vjp_bond_group_offsets,
        host_vjp_group_term_offsets, host_vjp_group_harmonics, host_vjp_group_terms,
        host_angular_plan);
    std::copy(direct_probe.begin(), direct_probe.end(), expected_probe_.begin());
    const auto block_probe = block_probe_values(host_block);
    std::copy(block_probe.begin(), block_probe.end(),
              expected_probe_.begin() + direct_probe.size());
    const auto scalar_probe = scalar_probe_values(host_scalar);
    std::copy(scalar_probe.begin(), scalar_probe.end(),
              expected_probe_.begin() + direct_probe.size() + block_probe.size());
    const auto coupled_probe = coupled_probe_values(host_coupled);
    std::copy(coupled_probe.begin(), coupled_probe.end(),
              expected_probe_.begin() + direct_probe.size() + block_probe.size() +
                  scalar_probe.size());

    copy_checked_view(dag_value_offsets, "ye3t:dag_value_offsets", host_dag_value_offsets);
    copy_checked_view(dag_parent_offsets, "ye3t:dag_parent_offsets", host_dag_parent_offsets);
    copy_checked_view(dag_parents, "ye3t:dag_parents", host_dag_parents);
    copy_checked_view(dag_siblings, "ye3t:dag_siblings", host_dag_siblings);
    copy_checked_view(dag_readout_seed, "ye3t:dag_readout_seed", host_dag_readout_seed);
    copy_checked_view(dag_source_power_offsets, "ye3t:dag_source_power_offsets",
                      host_dag_source_power_offsets);
    copy_checked_view(dag_source_powers, "ye3t:dag_source_powers", host_dag_source_powers);

    copy_view(species_channel_offsets, "ye3t:species_channel_offsets",
              host_species_channel_offsets);
    copy_view(species_source_offsets, "ye3t:species_source_offsets", host_species_source_offsets);
    copy_view(species_power_offsets, "ye3t:species_power_offsets", host_species_power_offsets);
    copy_view(species_node_offsets, "ye3t:species_node_offsets", host_species_node_offsets);
    copy_view(species_monomial_offsets, "ye3t:species_monomial_offsets",
              host_species_monomial_offsets);
    copy_view(node_level_table_offsets, "ye3t:node_level_table_offsets",
              host_species_node_level_table_offsets);
    copy_view(node_level_offsets, "ye3t:node_level_offsets", host_node_level_offsets);
    copy_view(node_level_nodes, "ye3t:node_level_nodes", host_node_level_nodes);
    copy_view(reference_energies, "ye3t:reference_energies", host_reference_energies);
    copy_view(embedding_scales, "ye3t:embedding_scales", host_embedding_scales);
    copy_view(density_safe_limits, "ye3t:density_safe_limits", host_density_safe_limits);
    copy_view(full_channel_sources, "ye3t:full_channel_sources", host_full_channel_sources);
    copy_view(full_channel_transforms, "ye3t:full_channel_transforms",
              host_full_channel_transforms);
    copy_view(power_channels, "ye3t:power_channels", host_power_channels);
    copy_view(power_exponents, "ye3t:power_exponents", host_power_exponents);
    copy_view(binary_node_left, "ye3t:binary_node_left", host_binary_node_left);
    copy_view(binary_node_right, "ye3t:binary_node_right", host_binary_node_right);
    copy_view(monomial_nodes, "ye3t:monomial_nodes", host_monomial_nodes);
    copy_view(monomial_coefficients, "ye3t:monomial_coefficients", host_monomial_coefficients);
    copy_view(bond_radial_counts, "ye3t:bond_radial_counts", host_bond_radial_counts);
    copy_view(bond_angular_maxima, "ye3t:bond_angular_maxima", host_bond_angular_maxima);
    copy_view(bond_radial_base_counts, "ye3t:bond_radial_base_counts",
              host_bond_radial_base_counts);
    copy_view(bond_interval_counts, "ye3t:bond_interval_counts", host_bond_interval_counts);
    copy_view(bond_cutoffs, "ye3t:bond_cutoffs", host_bond_cutoffs);
    copy_view(radial_spline_offsets, "ye3t:radial_spline_offsets", host_radial_spline_offsets);
    copy_view(radial_splines, "ye3t:radial_splines", host_radial_splines);
    copy_view(contracted_spline_offsets, "ye3t:contracted_spline_offsets",
              host_contracted_spline_offsets);
    copy_view(contracted_splines, "ye3t:contracted_splines", host_contracted_splines);
    copy_view(radial_map_offsets, "ye3t:radial_map_offsets", host_radial_map_offsets);
    copy_view(radial_channel_outputs, "ye3t:radial_channel_outputs", host_radial_channel_outputs);
    copy_view(radial_channel_indices, "ye3t:radial_channel_indices", host_radial_channel_indices);
    copy_view(angular_map_offsets, "ye3t:angular_map_offsets", host_angular_map_offsets);
    copy_view(angular_channel_outputs, "ye3t:angular_channel_outputs",
              host_angular_channel_outputs);
    copy_view(contracted_channel_indices, "ye3t:contracted_channel_indices",
              host_contracted_channel_indices);
    copy_view(angular_channel_indices, "ye3t:angular_channel_indices",
              host_angular_channel_indices);
    copy_view(vjp_bond_group_offsets, "ye3t:vjp_bond_group_offsets", host_vjp_bond_group_offsets);
    copy_view(vjp_group_term_offsets, "ye3t:vjp_group_term_offsets", host_vjp_group_term_offsets);
    copy_view(vjp_group_harmonics, "ye3t:vjp_group_harmonics", host_vjp_group_harmonics);
    copy_view(vjp_group_terms, "ye3t:vjp_group_terms", host_vjp_group_terms);
    copy_view(angular_plan, "ye3t:angular_plan", host_angular_plan);
    copy_checked_view(angular_recurrence, "ye3t:angular_recurrence", host_angular_recurrence);
    copy_checked_view(vjp_group_degrees, "ye3t:vjp_group_degrees", host_vjp_group_degrees);
    copy_checked_view(vjp_group_orders, "ye3t:vjp_group_orders", host_vjp_group_orders);
    copy_checked_view(vjp_bond_column_offsets, "ye3t:vjp_bond_column_offsets",
                      host_vjp_bond_column_offsets);
    copy_checked_view(vjp_column_group_offsets, "ye3t:vjp_column_group_offsets",
                      host_vjp_column_group_offsets);
    copy_checked_view(active_contracted_offsets, "ye3t:active_contracted_offsets",
                      host_active_contracted_offsets);
    copy_checked_view(active_contracted_functions, "ye3t:active_contracted_functions",
                      host_active_contracted_functions);

    copy_checked_view(block_plan_tile_offsets, "ye3t:block_plan_tile_offsets",
                      host_block.execution.plan_tile_offsets);
    copy_checked_view(block_tile_monomial_begin, "ye3t:block_tile_monomial_begin",
                      host_block.execution.tile_monomial_begin);
    copy_checked_view(block_tile_monomial_count, "ye3t:block_tile_monomial_count",
                      host_block.execution.tile_monomial_count);
    copy_checked_view(block_tile_output_offsets, "ye3t:block_tile_output_offsets",
                      host_block.execution.tile_output_offsets);
    copy_checked_view(block_tile_outputs, "ye3t:block_tile_outputs",
                      host_block.execution.tile_outputs);
    copy_checked_view(block_tile_coefficient_offsets, "ye3t:block_tile_coefficient_offsets",
                      host_block.execution.tile_coefficient_offsets);
    copy_checked_view(block_tile_coefficients, "ye3t:block_tile_coefficients",
                      host_block.execution.tile_coefficients);
    copy_checked_view(block_monomial_output_offsets, "ye3t:block_monomial_output_offsets",
                      host_block.execution.monomial_output_offsets);
    copy_checked_view(block_monomial_outputs, "ye3t:block_monomial_outputs",
                      host_block.execution.monomial_outputs);
    copy_checked_view(block_monomial_coefficients, "ye3t:block_monomial_coefficients",
                      host_block.execution.monomial_coefficients);
    copy_view(block_species, "ye3t:block_species", host_block.species);
    copy_view(block_powers, "ye3t:block_powers", host_block.powers);
    copy_view(block_plans, "ye3t:block_plans", host_block.plans);
    copy_view(block_routes, "ye3t:block_routes", host_block.routes);
    copy_view(block_plan_input_channels, "ye3t:block_plan_input_channels",
              host_block.plan_input_channels);
    copy_view(block_plan_input_power_offsets, "ye3t:block_plan_input_power_offsets",
              host_block.plan_input_power_offsets);
    copy_view(block_monomial_factor_offsets, "ye3t:block_monomial_factor_offsets",
              host_block.monomial_factor_offsets);
    copy_view(block_monomial_factor_components, "ye3t:block_monomial_factor_components",
              host_block.monomial_factor_components);
    copy_view(block_monomial_factor_exponents, "ye3t:block_monomial_factor_exponents",
              host_block.monomial_factor_exponents);
    copy_view(block_output_coefficient_offsets, "ye3t:block_output_coefficient_offsets",
              host_block.output_coefficient_offsets);
    copy_view(block_coefficient_terms, "ye3t:block_coefficient_terms",
              host_block.coefficient_terms);
    copy_view(block_coefficient_real, "ye3t:block_coefficient_real", host_block.coefficient_real);
    copy_view(block_coefficient_imaginary, "ye3t:block_coefficient_imaginary",
              host_block.coefficient_imaginary);
    copy_view(block_direct_input_channels, "ye3t:block_direct_input_channels",
              host_block.direct_input_channels);
    copy_view(block_direct_input_scales, "ye3t:block_direct_input_scales",
              host_block.direct_input_scales);
    copy_view(block_route_term_factor_offsets, "ye3t:block_route_term_factor_offsets",
              host_block.route_term_factor_offsets);
    copy_view(block_route_term_factor_plans, "ye3t:block_route_term_factor_plans",
              host_block.route_term_factor_plans);
    copy_view(block_route_term_factor_components, "ye3t:block_route_term_factor_components",
              host_block.route_term_factor_components);
    copy_view(block_route_term_coefficient_real, "ye3t:block_route_term_coefficient_real",
              host_block.route_term_coefficient_real);
    copy_view(block_route_term_coefficient_imaginary, "ye3t:block_route_term_coefficient_imaginary",
              host_block.route_term_coefficient_imaginary);
    copy_view(scalar_species, "ye3t:scalar_species", host_scalar.species);
    copy_view(scalar_bases, "ye3t:scalar_bases", host_scalar.bases);
    copy_view(scalar_nodes, "ye3t:scalar_nodes", host_scalar.nodes);
    copy_view(scalar_routes, "ye3t:scalar_routes", host_scalar.routes);
    copy_view(scalar_left_channels, "ye3t:scalar_left_channels", host_scalar.left_channels);
    copy_view(scalar_right_channels, "ye3t:scalar_right_channels", host_scalar.right_channels);
    copy_view(scalar_coefficient_real, "ye3t:scalar_coefficient_real",
              host_scalar.coefficient_real);
    copy_view(scalar_coefficient_imaginary, "ye3t:scalar_coefficient_imaginary",
              host_scalar.coefficient_imaginary);
    copy_view(coupled_species, "ye3t:coupled_species", host_coupled.species);
    copy_view(coupled_plans, "ye3t:coupled_plans", host_coupled.plans);
    copy_view(coupled_nodes, "ye3t:coupled_nodes", host_coupled.nodes);
    copy_view(coupled_leaf_input_channels, "ye3t:coupled_leaf_inputs",
              host_coupled.leaf_input_channels);
    copy_view(coupled_coefficient_left_components, "ye3t:coupled_coefficient_left",
              host_coupled.coefficient_left_components);
    copy_view(coupled_coefficient_right_components, "ye3t:coupled_coefficient_right",
              host_coupled.coefficient_right_components);
    copy_view(coupled_coefficient_output_components, "ye3t:coupled_coefficient_output",
              host_coupled.coefficient_output_components);
    copy_view(coupled_coefficient_values, "ye3t:coupled_coefficients",
              host_coupled.coefficient_values);
    copy_view(coupled_readout_components, "ye3t:coupled_readout_components",
              host_coupled.readout_components);
    copy_view(coupled_readout_coefficients, "ye3t:coupled_readout_coefficients",
              host_coupled.readout_coefficients);
  }

  double run_probe() const
  {
    RealView result("ye3t:device_plan_probe", expected_probe_.size());
    const auto species_channel_offsets_view = species_channel_offsets;
    const auto species_source_offsets_view = species_source_offsets;
    const auto species_power_offsets_view = species_power_offsets;
    const auto species_node_offsets_view = species_node_offsets;
    const auto species_monomial_offsets_view = species_monomial_offsets;
    const auto node_level_table_offsets_view = node_level_table_offsets;
    const auto node_level_offsets_view = node_level_offsets;
    const auto node_level_nodes_view = node_level_nodes;
    const auto reference_energies_view = reference_energies;
    const auto embedding_scales_view = embedding_scales;
    const auto density_safe_limits_view = density_safe_limits;
    const auto full_channel_sources_view = full_channel_sources;
    const auto full_channel_transforms_view = full_channel_transforms;
    const auto power_channels_view = power_channels;
    const auto power_exponents_view = power_exponents;
    const auto binary_node_left_view = binary_node_left;
    const auto binary_node_right_view = binary_node_right;
    const auto monomial_nodes_view = monomial_nodes;
    const auto monomial_coefficients_view = monomial_coefficients;
    const auto bond_radial_counts_view = bond_radial_counts;
    const auto bond_angular_maxima_view = bond_angular_maxima;
    const auto bond_radial_base_counts_view = bond_radial_base_counts;
    const auto bond_interval_counts_view = bond_interval_counts;
    const auto bond_cutoffs_view = bond_cutoffs;
    const auto radial_spline_offsets_view = radial_spline_offsets;
    const auto radial_splines_view = radial_splines;
    const auto contracted_spline_offsets_view = contracted_spline_offsets;
    const auto contracted_splines_view = contracted_splines;
    const auto radial_map_offsets_view = radial_map_offsets;
    const auto radial_channel_outputs_view = radial_channel_outputs;
    const auto radial_channel_indices_view = radial_channel_indices;
    const auto angular_map_offsets_view = angular_map_offsets;
    const auto angular_channel_outputs_view = angular_channel_outputs;
    const auto contracted_channel_indices_view = contracted_channel_indices;
    const auto angular_channel_indices_view = angular_channel_indices;
    const auto vjp_bond_group_offsets_view = vjp_bond_group_offsets;
    const auto vjp_group_term_offsets_view = vjp_group_term_offsets;
    const auto vjp_group_harmonics_view = vjp_group_harmonics;
    const auto vjp_group_terms_view = vjp_group_terms;
    const auto angular_plan_view = angular_plan;
    const auto block_species_view = block_species;
    const auto block_powers_view = block_powers;
    const auto block_plans_view = block_plans;
    const auto block_routes_view = block_routes;
    const auto block_plan_input_channels_view = block_plan_input_channels;
    const auto block_plan_input_power_offsets_view = block_plan_input_power_offsets;
    const auto block_monomial_factor_offsets_view = block_monomial_factor_offsets;
    const auto block_monomial_factor_components_view = block_monomial_factor_components;
    const auto block_monomial_factor_exponents_view = block_monomial_factor_exponents;
    const auto block_output_coefficient_offsets_view = block_output_coefficient_offsets;
    const auto block_coefficient_terms_view = block_coefficient_terms;
    const auto block_coefficient_real_view = block_coefficient_real;
    const auto block_coefficient_imaginary_view = block_coefficient_imaginary;
    const auto block_direct_input_channels_view = block_direct_input_channels;
    const auto block_direct_input_scales_view = block_direct_input_scales;
    const auto block_route_term_factor_offsets_view = block_route_term_factor_offsets;
    const auto block_route_term_factor_plans_view = block_route_term_factor_plans;
    const auto block_route_term_factor_components_view = block_route_term_factor_components;
    const auto block_route_term_coefficient_real_view = block_route_term_coefficient_real;
    const auto block_route_term_coefficient_imaginary_view = block_route_term_coefficient_imaginary;
    const auto scalar_species_view = scalar_species;
    const auto scalar_bases_view = scalar_bases;
    const auto scalar_nodes_view = scalar_nodes;
    const auto scalar_routes_view = scalar_routes;
    const auto scalar_left_channels_view = scalar_left_channels;
    const auto scalar_right_channels_view = scalar_right_channels;
    const auto scalar_coefficient_real_view = scalar_coefficient_real;
    const auto scalar_coefficient_imaginary_view = scalar_coefficient_imaginary;
    const auto coupled_species_view = coupled_species;
    const auto coupled_plans_view = coupled_plans;
    const auto coupled_nodes_view = coupled_nodes;
    const auto coupled_leaf_input_channels_view = coupled_leaf_input_channels;
    const auto coupled_coefficient_left_components_view = coupled_coefficient_left_components;
    const auto coupled_coefficient_right_components_view = coupled_coefficient_right_components;
    const auto coupled_coefficient_output_components_view = coupled_coefficient_output_components;
    const auto coupled_coefficient_values_view = coupled_coefficient_values;
    const auto coupled_readout_components_view = coupled_readout_components;
    const auto coupled_readout_coefficients_view = coupled_readout_coefficients;
    const int local_species_count = species_count;
    const int local_bond_count = bond_count;
    const int local_maximum_angular_momentum = maximum_angular_momentum;

    Kokkos::parallel_for(
        "YE3TDevicePlanProbe", Kokkos::RangePolicy<DeviceType>(0, 1), KOKKOS_LAMBDA(const int) {
          result(0) = static_cast<double>(local_species_count) +
              static_cast<double>(species_channel_offsets_view.extent(0)) +
              reference_energies_view(0) + embedding_scales_view(0);
          result(1) = static_cast<double>(local_bond_count) + bond_cutoffs_view(0) +
              static_cast<double>(bond_interval_counts_view(0)) + radial_splines_view(0) +
              contracted_splines_view(0);
          result(2) = static_cast<double>(local_maximum_angular_momentum) + angular_plan_view(0) +
              angular_plan_view(angular_plan_view.extent(0) - 1);
          result(3) = static_cast<double>(species_source_offsets_view(1)) +
              static_cast<double>(full_channel_sources_view(0)) +
              static_cast<double>(full_channel_transforms_view(0));
          result(4) = static_cast<double>(species_power_offsets_view(1)) +
              static_cast<double>(power_channels_view.extent(0) > 0 ? power_channels_view(0) : 0) +
              static_cast<double>(power_exponents_view.extent(0) > 0 ? power_exponents_view(0) : 0);
          result(5) = static_cast<double>(species_node_offsets_view(1)) +
              static_cast<double>(binary_node_left_view.extent(0) > 0 ? binary_node_left_view(0)
                                                                      : 0) +
              static_cast<double>(binary_node_right_view.extent(0) > 0 ? binary_node_right_view(0)
                                                                       : 0) +
              static_cast<double>(node_level_table_offsets_view(1)) +
              static_cast<double>(node_level_offsets_view(0)) +
              static_cast<double>(node_level_nodes_view.extent(0) > 0 ? node_level_nodes_view(0)
                                                                      : 0);
          result(6) = static_cast<double>(species_monomial_offsets_view(1)) +
              static_cast<double>(monomial_nodes_view.extent(0) > 0 ? monomial_nodes_view(0) : 0) +
              (monomial_coefficients_view.extent(0) > 0 ? monomial_coefficients_view(0) : 0.0) +
              density_safe_limits_view(0);
          result(7) =
              static_cast<double>(bond_radial_counts_view(0)) +
              static_cast<double>(bond_angular_maxima_view(0)) +
              static_cast<double>(bond_radial_base_counts_view(0)) +
              static_cast<double>(radial_spline_offsets_view(1)) +
              static_cast<double>(contracted_spline_offsets_view(1)) +
              static_cast<double>(radial_map_offsets_view(1)) +
              static_cast<double>(
                  radial_channel_outputs_view.extent(0) > 0 ? radial_channel_outputs_view(0) : 0) +
              static_cast<double>(
                  radial_channel_indices_view.extent(0) > 0 ? radial_channel_indices_view(0) : 0) +
              static_cast<double>(angular_map_offsets_view(1)) +
              static_cast<double>(angular_channel_outputs_view.extent(0) > 0
                                      ? angular_channel_outputs_view(0)
                                      : 0) +
              static_cast<double>(contracted_channel_indices_view.extent(0) > 0
                                      ? contracted_channel_indices_view(0)
                                      : 0) +
              static_cast<double>(angular_channel_indices_view.extent(0) > 0
                                      ? angular_channel_indices_view(0)
                                      : 0) +
              static_cast<double>(vjp_bond_group_offsets_view(1)) +
              static_cast<double>(vjp_group_term_offsets_view.extent(0)) +
              static_cast<double>(
                  vjp_group_harmonics_view.extent(0) > 0 ? vjp_group_harmonics_view(0) : 0) +
              static_cast<double>(vjp_group_terms_view.extent(0) > 0 ? vjp_group_terms_view(0) : 0);
          const auto block_species_first = block_species_view(0);
          result(8) = static_cast<double>(block_species_view.extent(0)) +
              block_species_first.power_begin + block_species_first.power_count +
              block_species_first.plan_begin + block_species_first.plan_count +
              block_species_first.route_begin + block_species_first.route_count +
              block_species_first.power_storage_size + block_species_first.output_storage_size;
          result(9) = static_cast<double>(block_powers_view.extent(0)) +
              static_cast<double>(block_plans_view.extent(0));
          if (block_powers_view.extent(0) > 0) {
            const auto power = block_powers_view(0);
            result(9) += power.channel + power.maximum_exponent + power.storage_offset;
          }
          if (block_plans_view.extent(0) > 0) {
            const auto plan = block_plans_view(0);
            result(9) += plan.input_begin + plan.input_count + plan.monomial_begin +
                plan.monomial_count + plan.output_begin + plan.output_count +
                plan.output_storage_offset + plan.output_L + plan.flags;
          }
          result(10) = static_cast<double>(block_routes_view.extent(0)) +
              static_cast<double>(block_monomial_factor_offsets_view.extent(0)) +
              static_cast<double>(block_output_coefficient_offsets_view.extent(0)) +
              static_cast<double>(block_coefficient_terms_view.extent(0));
          if (block_routes_view.extent(0) > 0) {
            const auto route = block_routes_view(0);
            result(10) += route.term_begin + route.term_count + route.function_index;
          }
          result(10) += static_cast<double>(block_monomial_factor_offsets_view.extent(0) > 0
                                                ? block_monomial_factor_offsets_view(0)
                                                : 0) +
              static_cast<double>(block_monomial_factor_components_view.extent(0) > 0
                                      ? block_monomial_factor_components_view(0)
                                      : 0) +
              static_cast<double>(block_monomial_factor_exponents_view.extent(0) > 0
                                      ? block_monomial_factor_exponents_view(0)
                                      : 0) +
              static_cast<double>(block_output_coefficient_offsets_view.extent(0) > 0
                                      ? block_output_coefficient_offsets_view(0)
                                      : 0) +
              static_cast<double>(block_coefficient_terms_view.extent(0) > 0
                                      ? block_coefficient_terms_view(0)
                                      : 0) +
              (block_coefficient_real_view.extent(0) > 0 ? block_coefficient_real_view(0) : 0.0) +
              (block_coefficient_imaginary_view.extent(0) > 0 ? block_coefficient_imaginary_view(0)
                                                              : 0.0);
          result(11) = static_cast<double>(block_plan_input_channels_view.extent(0)) +
              static_cast<double>(block_plan_input_power_offsets_view.extent(0)) +
              static_cast<double>(block_direct_input_channels_view.extent(0)) +
              static_cast<double>(block_direct_input_scales_view.extent(0)) +
              static_cast<double>(block_route_term_factor_offsets_view.extent(0)) +
              static_cast<double>(block_route_term_factor_plans_view.extent(0)) +
              static_cast<double>(block_route_term_factor_components_view.extent(0)) +
              static_cast<double>(block_route_term_coefficient_real_view.extent(0)) +
              static_cast<double>(block_route_term_coefficient_imaginary_view.extent(0)) +
              static_cast<double>(block_plan_input_channels_view.extent(0) > 0
                                      ? block_plan_input_channels_view(0)
                                      : 0) +
              static_cast<double>(block_plan_input_power_offsets_view.extent(0) > 0
                                      ? block_plan_input_power_offsets_view(0)
                                      : 0) +
              static_cast<double>(block_direct_input_channels_view.extent(0) > 0
                                      ? block_direct_input_channels_view(0)
                                      : 0) +
              (block_direct_input_scales_view.extent(0) > 0 ? block_direct_input_scales_view(0)
                                                            : 0.0) +
              static_cast<double>(block_route_term_factor_offsets_view.extent(0) > 0
                                      ? block_route_term_factor_offsets_view(0)
                                      : 0) +
              static_cast<double>(block_route_term_factor_plans_view.extent(0) > 0
                                      ? block_route_term_factor_plans_view(0)
                                      : 0) +
              static_cast<double>(block_route_term_factor_components_view.extent(0) > 0
                                      ? block_route_term_factor_components_view(0)
                                      : 0) +
              (block_route_term_coefficient_real_view.extent(0) > 0
                   ? block_route_term_coefficient_real_view(0)
                   : 0.0) +
              (block_route_term_coefficient_imaginary_view.extent(0) > 0
                   ? block_route_term_coefficient_imaginary_view(0)
                   : 0.0);
          const auto scalar_species_first = scalar_species_view(0);
          result(12) = static_cast<double>(scalar_species_view.extent(0)) +
              scalar_species_first.base_begin + scalar_species_first.base_count +
              scalar_species_first.node_begin + scalar_species_first.node_count +
              scalar_species_first.route_begin + scalar_species_first.route_count +
              scalar_species_first.value_count;
          result(13) = static_cast<double>(scalar_bases_view.extent(0)) +
              static_cast<double>(scalar_nodes_view.extent(0));
          if (scalar_bases_view.extent(0) > 0) {
            const auto base = scalar_bases_view(0);
            result(13) += base.term_begin + base.term_count;
          }
          if (scalar_nodes_view.extent(0) > 0) {
            const auto node = scalar_nodes_view(0);
            result(13) += node.left_value + node.right_value + node.exponent + node.base_index;
          }
          result(14) = static_cast<double>(scalar_routes_view.extent(0));
          if (scalar_routes_view.extent(0) > 0) {
            const auto route = scalar_routes_view(0);
            result(14) +=
                route.function_index + route.value_index + route.scale_real + route.scale_imaginary;
          }
          result(15) =
              static_cast<double>(scalar_left_channels_view.extent(0)) +
              static_cast<double>(scalar_right_channels_view.extent(0)) +
              static_cast<double>(scalar_coefficient_real_view.extent(0)) +
              static_cast<double>(scalar_coefficient_imaginary_view.extent(0)) +
              static_cast<double>(
                  scalar_left_channels_view.extent(0) > 0 ? scalar_left_channels_view(0) : 0) +
              static_cast<double>(
                  scalar_right_channels_view.extent(0) > 0 ? scalar_right_channels_view(0) : 0) +
              (scalar_coefficient_real_view.extent(0) > 0 ? scalar_coefficient_real_view(0) : 0.0) +
              (scalar_coefficient_imaginary_view.extent(0) > 0
                   ? scalar_coefficient_imaginary_view(0)
                   : 0.0);
          const auto coupled_species_first = coupled_species_view(0);
          result(16) = static_cast<double>(coupled_species_view.extent(0)) +
              coupled_species_first.plan_begin + coupled_species_first.plan_count +
              static_cast<double>(coupled_plans_view.extent(0));
          if (coupled_plans_view.extent(0) > 0) {
            const auto plan = coupled_plans_view(0);
            result(16) += plan.node_begin + plan.node_count + plan.leaf_begin + plan.leaf_count +
                plan.coefficient_begin + plan.coefficient_count + plan.readout_begin +
                plan.readout_count + plan.total_component_count + plan.function_index +
                plan.operation_estimate;
          }
          result(17) = static_cast<double>(coupled_nodes_view.extent(0));
          if (coupled_nodes_view.extent(0) > 0) {
            const auto first = coupled_nodes_view(0);
            const auto last = coupled_nodes_view(coupled_nodes_view.extent(0) - 1);
            result(17) += first.component_offset + first.dimension + first.leaf_offset +
                first.coefficient_begin + first.coefficient_count + last.component_offset +
                last.dimension + last.leaf_offset + last.coefficient_begin + last.coefficient_count;
          }
          result(18) = static_cast<double>(coupled_leaf_input_channels_view.extent(0)) +
              static_cast<double>(coupled_coefficient_left_components_view.extent(0)) +
              static_cast<double>(coupled_coefficient_right_components_view.extent(0)) +
              static_cast<double>(coupled_coefficient_output_components_view.extent(0)) +
              static_cast<double>(coupled_coefficient_values_view.extent(0)) +
              static_cast<double>(coupled_leaf_input_channels_view.extent(0) > 0
                                      ? coupled_leaf_input_channels_view(0)
                                      : 0) +
              static_cast<double>(coupled_coefficient_left_components_view.extent(0) > 0
                                      ? coupled_coefficient_left_components_view(0)
                                      : 0) +
              static_cast<double>(coupled_coefficient_right_components_view.extent(0) > 0
                                      ? coupled_coefficient_right_components_view(0)
                                      : 0) +
              static_cast<double>(coupled_coefficient_output_components_view.extent(0) > 0
                                      ? coupled_coefficient_output_components_view(0)
                                      : 0) +
              (coupled_coefficient_values_view.extent(0) > 0 ? coupled_coefficient_values_view(0)
                                                             : 0.0);
          result(19) = static_cast<double>(coupled_readout_components_view.extent(0)) +
              static_cast<double>(coupled_readout_coefficients_view.extent(0)) +
              static_cast<double>(coupled_readout_components_view.extent(0) > 0
                                      ? coupled_readout_components_view(0)
                                      : 0) +
              (coupled_readout_coefficients_view.extent(0) > 0
                   ? coupled_readout_coefficients_view(0)
                   : 0.0);
        });
    Kokkos::fence("YE3T device-plan probe");
    auto host_result = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
    double maximum_error = 0.0;
    for (std::size_t index = 0; index < expected_probe_.size(); ++index)
      maximum_error =
          std::max(maximum_error, std::abs(host_result(index) - expected_probe_[index]));
    return maximum_error;
  }

  std::size_t memory_usage() const { return static_bytes_; }
  const std::string &signature_hash() const { return signature_hash_; }
  const std::string &vjp_schedule_hash() const { return vjp_schedule_hash_; }
  int vjp_harmonic_group_count() const { return vjp_harmonic_group_count_; }
  int vjp_angular_term_count() const { return vjp_angular_term_count_; }
  int vjp_maximum_group_term_count() const { return vjp_maximum_group_term_count_; }
  int maximum_source_serial_work_count() const { return maximum_source_serial_work_count_; }
  int maximum_channel_count() const { return maximum_channel_count_; }
  int maximum_source_count() const { return maximum_source_count_; }
  int maximum_power_count() const { return maximum_power_count_; }
  int maximum_node_count() const { return maximum_node_count_; }
  int maximum_value_count() const { return maximum_power_count_ + maximum_node_count_; }
  int maximum_monomial_count() const { return maximum_monomial_count_; }
  int maximum_block_power_storage() const { return maximum_block_power_storage_; }
  int maximum_block_output_storage() const { return maximum_block_output_storage_; }
  int maximum_block_monomial_count() const { return maximum_block_monomial_count_; }
  int block_monomial_tile_count() const
  {
    return std::min(maximum_block_monomial_count_, GPU_BLOCK_MONOMIAL_TILE);
  }
  int maximum_block_power_count() const { return maximum_block_power_count_; }
  int maximum_block_plan_count() const { return maximum_block_plan_count_; }
  int maximum_block_route_count() const { return maximum_block_route_count_; }
  int block_route_count() const { return block_route_count_; }
  bool has_block_program() const { return block_route_count_ > 0; }
  bool has_direct_residual() const { return maximum_monomial_count_ > 0; }
  int maximum_scalar_value_count() const { return maximum_scalar_value_count_; }
  int scalar_route_count() const { return scalar_route_count_; }
  bool has_scalar_program() const { return scalar_route_count_ > 0; }
  int maximum_coupled_component_count() const { return maximum_coupled_component_count_; }
  int coupled_plan_count() const { return coupled_plan_count_; }
  bool has_coupled_program() const { return coupled_plan_count_ > 0; }
  bool has_optimized_program() const
  {
    return has_block_program() || has_scalar_program() || has_coupled_program();
  }

  std::size_t bytes_per_center(bool resident_direct = false) const
  {
    const std::size_t complex_components =
        checked_add(static_cast<std::size_t>(maximum_source_count_),
                    static_cast<std::size_t>(resident_direct ? 0 : maximum_value_count()),
                    "per-center complex workspace");
    std::size_t real_components =
        checked_product(complex_components, 4, "per-center split-complex workspace");
    real_components = checked_add(real_components, 1, "per-center energy workspace");
    if (has_optimized_program()) {
      real_components = checked_add(real_components, 2, "per-center residual-density workspace");
    }
    if (has_block_program()) {
      real_components =
          checked_add(real_components,
                      checked_product(static_cast<std::size_t>(maximum_block_power_storage_), 2,
                                      "per-center block-power workspace"),
                      "per-center block-power workspace");
      real_components =
          checked_add(real_components,
                      checked_product(static_cast<std::size_t>(maximum_block_output_storage_), 4,
                                      "per-center block-output workspace"),
                      "per-center block-output workspace");
      real_components =
          checked_add(real_components,
                      checked_product(static_cast<std::size_t>(block_monomial_tile_count()), 2,
                                      "per-center block-monomial workspace"),
                      "per-center block-monomial workspace");
      real_components =
          checked_add(real_components,
                      checked_product(static_cast<std::size_t>(maximum_block_route_count_), 2,
                                      "per-center block-route density workspace"),
                      "per-center block-route density workspace");
    }
    if (has_scalar_program()) {
      real_components =
          checked_add(real_components,
                      checked_product(static_cast<std::size_t>(maximum_scalar_value_count_), 4,
                                      "per-center scalar-power workspace"),
                      "per-center scalar-power workspace");
    }
    if (has_coupled_program()) {
      real_components =
          checked_add(real_components,
                      checked_product(static_cast<std::size_t>(maximum_coupled_component_count_), 4,
                                      "per-center coupled-product workspace"),
                      "per-center coupled-product workspace");
    }
    return checked_product(real_components, sizeof(double), "per-center real workspace");
  }

  std::size_t minimum_dynamic_bytes() const
  {
    return checked_add(bytes_per_center(), 2 * sizeof(int), "minimum dynamic workspace");
  }

  YE3TKokkosSpeciesViews<DeviceType> species_views() const
  {
    return {species_count,
            species_channel_offsets,
            species_source_offsets,
            species_power_offsets,
            species_node_offsets,
            species_monomial_offsets,
            node_level_table_offsets,
            node_level_offsets,
            node_level_nodes,
            reference_energies,
            embedding_scales,
            density_safe_limits,
            full_channel_sources,
            full_channel_transforms,
            power_channels,
            power_exponents,
            binary_node_left,
            binary_node_right,
            monomial_nodes,
            monomial_coefficients,
            dag_value_offsets,
            dag_parent_offsets,
            dag_parents,
            dag_siblings,
            dag_readout_seed,
            dag_source_power_offsets,
            dag_source_powers};
  }

  YE3TKokkosBondViews<DeviceType> bond_views() const
  {
    return {vjp_group_degrees,
            vjp_group_orders,
            vjp_bond_column_offsets,
            vjp_column_group_offsets,
            active_contracted_offsets,
            active_contracted_functions,
            species_count,
            bond_count,
            bond_radial_counts,
            bond_angular_maxima,
            bond_radial_base_counts,
            bond_interval_counts,
            bond_cutoffs,
            radial_spline_offsets,
            radial_splines,
            contracted_spline_offsets,
            contracted_splines,
            radial_map_offsets,
            radial_channel_outputs,
            radial_channel_indices,
            angular_map_offsets,
            angular_channel_outputs,
            contracted_channel_indices,
            angular_channel_indices,
            vjp_bond_group_offsets,
            vjp_group_term_offsets,
            vjp_group_harmonics,
            vjp_group_terms};
  }

  YE3TKokkosAngularViews<DeviceType> angular_views() const
  {
    return {maximum_angular_momentum, angular_plan, angular_recurrence};
  }

  YE3TKokkosBlockViews<DeviceType> block_views() const
  {
    return {block_plan_tile_offsets,
            block_tile_monomial_begin,
            block_tile_monomial_count,
            block_tile_output_offsets,
            block_tile_outputs,
            block_tile_coefficient_offsets,
            block_tile_coefficients,
            block_monomial_output_offsets,
            block_monomial_outputs,
            block_monomial_coefficients,
            block_species,
            block_powers,
            block_plans,
            block_routes,
            block_plan_input_channels,
            block_plan_input_power_offsets,
            block_monomial_factor_offsets,
            block_monomial_factor_components,
            block_monomial_factor_exponents,
            block_output_coefficient_offsets,
            block_coefficient_terms,
            block_coefficient_real,
            block_coefficient_imaginary,
            block_direct_input_channels,
            block_direct_input_scales,
            block_route_term_factor_offsets,
            block_route_term_factor_plans,
            block_route_term_factor_components,
            block_route_term_coefficient_real,
            block_route_term_coefficient_imaginary};
  }

  YE3TKokkosScalarViews<DeviceType> scalar_views() const
  {
    return {scalar_species,
            scalar_bases,
            scalar_nodes,
            scalar_routes,
            scalar_left_channels,
            scalar_right_channels,
            scalar_coefficient_real,
            scalar_coefficient_imaginary};
  }

  YE3TKokkosCoupledViews<DeviceType> coupled_views() const
  {
    return {coupled_species,
            coupled_plans,
            coupled_nodes,
            coupled_leaf_input_channels,
            coupled_coefficient_left_components,
            coupled_coefficient_right_components,
            coupled_coefficient_output_components,
            coupled_coefficient_values,
            coupled_readout_components,
            coupled_readout_coefficients};
  }

 private:
  int species_count = 0;
  int bond_count = 0;
  int maximum_angular_momentum = 0;
  int maximum_channel_count_ = 0;
  int maximum_source_count_ = 0;
  int maximum_power_count_ = 0;
  int maximum_node_count_ = 0;
  int maximum_monomial_count_ = 0;
  int maximum_block_power_storage_ = 0;
  int maximum_block_output_storage_ = 0;
  int maximum_block_monomial_count_ = 0;
  int maximum_block_power_count_ = 0;
  int maximum_block_plan_count_ = 0;
  int maximum_block_route_count_ = 0;
  int block_route_count_ = 0;
  int maximum_scalar_value_count_ = 0;
  int scalar_route_count_ = 0;
  int maximum_coupled_component_count_ = 0;
  int coupled_plan_count_ = 0;
  IntView species_channel_offsets;
  IntView species_source_offsets;
  IntView species_power_offsets;
  IntView species_node_offsets;
  IntView species_monomial_offsets;
  IntView node_level_table_offsets;
  IntView node_level_offsets;
  IntView node_level_nodes;
  RealView reference_energies;
  RealView embedding_scales;
  RealView density_safe_limits;
  IntView full_channel_sources;
  IntView full_channel_transforms;
  IntView power_channels;
  IntView power_exponents;
  IntView binary_node_left;
  IntView binary_node_right;
  IntView monomial_nodes;
  RealView monomial_coefficients;
  // Owner-gather reverse DAG and source-power transpose, compiled once.
  IntView dag_value_offsets;
  IntView dag_parent_offsets;
  IntView dag_parents;
  IntView dag_siblings;
  RealView dag_readout_seed;
  IntView dag_source_power_offsets;
  IntView dag_source_powers;

  IntView bond_radial_counts;
  IntView bond_angular_maxima;
  IntView bond_radial_base_counts;
  IntView bond_interval_counts;
  RealView bond_cutoffs;
  IntView radial_spline_offsets;
  RealView radial_splines;
  IntView contracted_spline_offsets;
  RealView contracted_splines;
  IntView radial_map_offsets;
  IntView radial_channel_outputs;
  IntView radial_channel_indices;
  IntView angular_map_offsets;
  IntView angular_channel_outputs;
  IntView contracted_channel_indices;
  IntView angular_channel_indices;
  IntView vjp_bond_group_offsets;
  IntView vjp_group_term_offsets;
  IntView vjp_group_harmonics;
  IntView vjp_group_terms;
  RealView angular_plan;
  RealView angular_recurrence;
  IntView vjp_group_degrees;
  IntView vjp_group_orders;
  IntView vjp_bond_column_offsets;
  IntView vjp_column_group_offsets;
  IntView active_contracted_offsets;
  IntView active_contracted_functions;

  BlockSpeciesView block_species;
  BlockPowerView block_powers;
  BlockPlanView block_plans;
  BlockRouteView block_routes;
  IntView block_plan_tile_offsets;
  IntView block_tile_monomial_begin;
  IntView block_tile_monomial_count;
  IntView block_tile_output_offsets;
  IntView block_tile_outputs;
  IntView block_tile_coefficient_offsets;
  IntView block_tile_coefficients;
  IntView block_monomial_output_offsets;
  IntView block_monomial_outputs;
  IntView block_monomial_coefficients;
  IntView block_plan_input_channels;
  IntView block_plan_input_power_offsets;
  IntView block_monomial_factor_offsets;
  IntView block_monomial_factor_components;
  IntView block_monomial_factor_exponents;
  IntView block_output_coefficient_offsets;
  IntView block_coefficient_terms;
  RealView block_coefficient_real;
  RealView block_coefficient_imaginary;
  IntView block_direct_input_channels;
  RealView block_direct_input_scales;
  IntView block_route_term_factor_offsets;
  IntView block_route_term_factor_plans;
  IntView block_route_term_factor_components;
  RealView block_route_term_coefficient_real;
  RealView block_route_term_coefficient_imaginary;
  ScalarSpeciesView scalar_species;
  ScalarBaseView scalar_bases;
  ScalarNodeView scalar_nodes;
  ScalarRouteView scalar_routes;
  IntView scalar_left_channels;
  IntView scalar_right_channels;
  RealView scalar_coefficient_real;
  RealView scalar_coefficient_imaginary;
  CoupledSpeciesView coupled_species;
  CoupledPlanView coupled_plans;
  CoupledNodeView coupled_nodes;
  IntView coupled_leaf_input_channels;
  IntView coupled_coefficient_left_components;
  IntView coupled_coefficient_right_components;
  IntView coupled_coefficient_output_components;
  RealView coupled_coefficient_values;
  IntView coupled_readout_components;
  RealView coupled_readout_coefficients;

  // Initialization-only integrity check for derived execution tables. The older
  // scalar plan probe remains in place for the original tables.
  template <class View, class Value>
  static void copy_checked_view(View &view, const char *label, const std::vector<Value> &values)
  {
    copy_view(view, label, values);
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
    for (std::size_t i = 0; i < values.size(); ++i)
      if (host(i) != values[i])
        throw std::runtime_error("YE3T GPU execution-table upload verification failed");
  }

  static int checked_size(std::size_t size, const char *name)
  {
    if (size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error(std::string("YE3T Kokkos ") + name +
                                " exceeds 32-bit device indexing");
    return static_cast<int>(size);
  }

  static int checked_index(std::int64_t value, const char *name)
  {
    if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
      throw std::overflow_error(std::string("YE3T Kokkos ") + name +
                                " exceeds 32-bit device indexing");
    return static_cast<int>(value);
  }

  static std::size_t checked_add(std::size_t first, std::size_t second, const char *name)
  {
    if (second > std::numeric_limits<std::size_t>::max() - first)
      throw std::overflow_error(std::string("YE3T Kokkos ") + name + " byte/count overflow");
    return first + second;
  }

  static std::size_t checked_product(std::size_t first, std::size_t second, const char *name)
  {
    if (first != 0 && second > std::numeric_limits<std::size_t>::max() / first)
      throw std::overflow_error(std::string("YE3T Kokkos ") + name + " byte/count overflow");
    return first * second;
  }

  static void append_offset(std::vector<int> &offsets, std::size_t count, const char *name)
  {
    const std::size_t previous = static_cast<std::size_t>(offsets.back());
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()) - previous)
      throw std::overflow_error(std::string("YE3T Kokkos ") + name +
                                " exceeds 32-bit device indexing");
    offsets.push_back(static_cast<int>(previous + count));
  }

  static void append_values(std::vector<double> &destination, const std::vector<double> &source)
  {
    if (source.size() > destination.max_size() - destination.size())
      throw std::overflow_error("YE3T Kokkos flattened table is too large");
    destination.insert(destination.end(), source.begin(), source.end());
  }

  static bool finite_complex(const std::complex<double> &value)
  {
    return std::isfinite(value.real()) && std::isfinite(value.imag());
  }

  void flatten_block_programs(const YACEModel &model, HostBlockTables &tables)
  {
    for (int species_index = 0; species_index < species_count; ++species_index) {
      const auto &species_entry = model.species(species_index);
      const auto &program = species_entry.block_program;
      tables.semantic_ids.push_back(program.plan_hash);
      tables.semantic_ids.push_back(program.evaluator_plan_hash);
      tables.semantic_ids.push_back(program.dispatch);
      YE3TKokkosBlockSpeciesRecord species_record;
      species_record.power_begin = checked_size(tables.powers.size(), "block power table");
      species_record.plan_begin = checked_size(tables.plans.size(), "block plan table");
      species_record.route_begin = checked_size(tables.routes.size(), "block route table");
      species_record.power_storage_size =
          checked_index(program.power_storage_size, "block power storage");
      if (species_record.power_storage_size < 0)
        throw std::invalid_argument("YE3T Kokkos block power storage is negative");
      if (program.power_channels.size() != program.power_maximum_exponents.size() ||
          program.power_channels.size() != program.power_offsets.size())
        throw std::invalid_argument("YE3T Kokkos block power cache tables are inconsistent");

      for (std::size_t power = 0; power < program.power_channels.size(); ++power) {
        const int channel = checked_index(program.power_channels[power], "block power channel");
        const int maximum =
            checked_index(program.power_maximum_exponents[power], "block maximum exponent");
        const int offset = checked_index(program.power_offsets[power], "block power offset");
        if (channel < 0 || channel >= static_cast<int>(species_entry.channels.size()) ||
            maximum < 0 || offset < 0 ||
            static_cast<std::int64_t>(offset) + maximum >= species_record.power_storage_size)
          throw std::invalid_argument("YE3T Kokkos block power cache entry is out of range");
        tables.powers.push_back({channel, maximum, offset});
      }
      species_record.power_count = checked_size(program.power_channels.size(), "block power table");

      int output_storage_size = 0;
      for (const auto &plan : program.power_plans) {
        YE3TKokkosBlockPlanRecord plan_record;
        plan_record.input_begin =
            checked_size(tables.plan_input_channels.size(), "block plan input table");
        const int declared_input_count =
            checked_index(plan.input_dimension, "block input dimension");
        plan_record.input_count = plan.direct_input_plan ? 0 : declared_input_count;
        plan_record.monomial_begin =
            checked_size(tables.monomial_factor_offsets.size() - 1, "block monomial table");
        const int declared_monomial_count =
            checked_index(plan.monomial_count, "block monomial count");
        plan_record.monomial_count = plan.direct_input_plan ? 0 : declared_monomial_count;
        plan_record.output_begin =
            checked_size(tables.output_coefficient_offsets.size() - 1, "block output table");
        plan_record.output_count = checked_index(plan.output_dimension, "block output dimension");
        plan_record.output_storage_offset =
            checked_index(plan.output_storage_offset, "block output storage offset");
        plan_record.output_L = plan.output_L;
        if (plan.real_coefficients) plan_record.flags |= YE3T_BLOCK_REAL_COEFFICIENTS;
        if (plan.conjugate_half_output) plan_record.flags |= YE3T_BLOCK_CONJUGATE_HALF_OUTPUT;
        if (plan.direct_input_plan) plan_record.flags |= YE3T_BLOCK_DIRECT_INPUT;

        if (plan_record.input_count < 0 || plan_record.monomial_count < 0 ||
            plan_record.output_count <= 0 || plan_record.output_storage_offset < 0 ||
            plan_record.output_L < 0)
          throw std::invalid_argument("YE3T Kokkos block plan dimensions are invalid");
        const std::int64_t output_end =
            static_cast<std::int64_t>(plan_record.output_storage_offset) + plan_record.output_count;
        if (output_end > std::numeric_limits<int>::max())
          throw std::overflow_error("YE3T Kokkos block output storage exceeds 32-bit indexing");
        output_storage_size = std::max(output_storage_size, static_cast<int>(output_end));
        const int magnetic_width = 2 * plan_record.output_L + 1;
        if (magnetic_width <= 0 || plan_record.output_count % magnetic_width != 0)
          throw std::invalid_argument("YE3T Kokkos block magnetic layout is inconsistent");
        if (!plan.direct_input_plan &&
            (plan.input_channels.size() != static_cast<std::size_t>(plan_record.input_count) ||
             plan.input_power_offsets.size() != static_cast<std::size_t>(plan_record.input_count)))
          throw std::invalid_argument("YE3T Kokkos block input tables are inconsistent");

        std::vector<int> input_maximum_exponents(static_cast<std::size_t>(plan_record.input_count),
                                                 -1);
        for (int input = 0; input < plan_record.input_count; ++input) {
          const int channel = checked_index(plan.input_channels[static_cast<std::size_t>(input)],
                                            "block input channel");
          const int power_offset =
              checked_index(plan.input_power_offsets[static_cast<std::size_t>(input)],
                            "block input power offset");
          if (channel < 0 || channel >= static_cast<int>(species_entry.channels.size()) ||
              power_offset < 0)
            throw std::invalid_argument("YE3T Kokkos block input entry is out of range");
          for (int power = 0; power < species_record.power_count; ++power) {
            const auto &record =
                tables.powers[static_cast<std::size_t>(species_record.power_begin + power)];
            if (record.channel == channel && record.storage_offset == power_offset) {
              input_maximum_exponents[static_cast<std::size_t>(input)] = record.maximum_exponent;
              break;
            }
          }
          if (!plan.direct_input_plan &&
              input_maximum_exponents[static_cast<std::size_t>(input)] < 0)
            throw std::invalid_argument("YE3T Kokkos block input is absent from the shared power "
                                        "cache");
          tables.plan_input_channels.push_back(channel);
          tables.plan_input_power_offsets.push_back(power_offset);
        }

        if (plan.direct_input_plan) {
          if (plan.direct_input_channels.size() !=
                  static_cast<std::size_t>(plan_record.output_count) ||
              plan.direct_input_scales.size() != static_cast<std::size_t>(plan_record.output_count))
            throw std::invalid_argument("YE3T Kokkos direct-input block plan is inconsistent");
        } else {
          if (plan_record.input_count <= 0 ||
              plan.monomial_counts.size() !=
                  static_cast<std::size_t>(plan_record.input_count) *
                      static_cast<std::size_t>(plan_record.monomial_count) ||
              plan.monomial_factor_offsets.size() !=
                  static_cast<std::size_t>(plan_record.monomial_count + 1) ||
              plan.monomial_factor_components.size() != plan.monomial_factor_exponents.size() ||
              plan.output_offsets.size() !=
                  static_cast<std::size_t>(plan_record.output_count + 1) ||
              plan.coefficient_terms.size() != plan.coefficient_values.size())
            throw std::invalid_argument(
                "YE3T Kokkos symmetric-power block tables are inconsistent");
        }

        for (int monomial = 0; monomial < plan_record.monomial_count; ++monomial) {
          const int begin =
              checked_index(plan.monomial_factor_offsets[static_cast<std::size_t>(monomial)],
                            "block monomial factor offset");
          const int end =
              checked_index(plan.monomial_factor_offsets[static_cast<std::size_t>(monomial + 1)],
                            "block monomial factor offset");
          if (begin < 0 || begin >= end ||
              end > static_cast<int>(plan.monomial_factor_components.size()))
            throw std::invalid_argument("YE3T Kokkos block monomial support is invalid");
          for (int factor = begin; factor < end; ++factor) {
            const int component =
                checked_index(plan.monomial_factor_components[static_cast<std::size_t>(factor)],
                              "block monomial factor component");
            const int exponent =
                checked_index(plan.monomial_factor_exponents[static_cast<std::size_t>(factor)],
                              "block monomial factor exponent");
            if (component < 0 || component >= plan_record.input_count || exponent <= 0 ||
                exponent > input_maximum_exponents[static_cast<std::size_t>(component)])
              throw std::invalid_argument("YE3T Kokkos block monomial factor is out of range");
            tables.monomial_factor_components.push_back(component);
            tables.monomial_factor_exponents.push_back(exponent);
          }
          tables.monomial_factor_offsets.push_back(checked_size(
              tables.monomial_factor_components.size(), "block monomial factor table"));
        }

        for (int output = 0; output < plan_record.output_count; ++output) {
          int direct_channel = -1;
          double direct_scale = 0.0;
          if (plan.direct_input_plan) {
            direct_channel =
                checked_index(plan.direct_input_channels[static_cast<std::size_t>(output)],
                              "block direct-input channel");
            direct_scale = plan.direct_input_scales[static_cast<std::size_t>(output)];
            if (direct_channel < -1 ||
                direct_channel >= static_cast<int>(species_entry.channels.size()) ||
                !std::isfinite(direct_scale))
              throw std::invalid_argument("YE3T Kokkos direct-input block entry is invalid");
          }
          tables.direct_input_channels.push_back(direct_channel);
          tables.direct_input_scales.push_back(direct_scale);

          if (!plan.direct_input_plan) {
            const int begin = checked_index(plan.output_offsets[static_cast<std::size_t>(output)],
                                            "block coefficient offset");
            const int end = checked_index(plan.output_offsets[static_cast<std::size_t>(output + 1)],
                                          "block coefficient offset");
            if (begin < 0 || begin > end || end > static_cast<int>(plan.coefficient_values.size()))
              throw std::invalid_argument("YE3T Kokkos block coefficient range is invalid");
            for (int coefficient = begin; coefficient < end; ++coefficient) {
              const int term =
                  checked_index(plan.coefficient_terms[static_cast<std::size_t>(coefficient)],
                                "block coefficient term");
              const auto value = plan.coefficient_values[static_cast<std::size_t>(coefficient)];
              if (term < 0 || term >= plan_record.monomial_count || !finite_complex(value) ||
                  (plan.real_coefficients && value.imag() != 0.0))
                throw std::invalid_argument("YE3T Kokkos block coefficient is invalid");
              tables.coefficient_terms.push_back(term);
              tables.coefficient_real.push_back(value.real());
              tables.coefficient_imaginary.push_back(value.imag());
            }
          }
          tables.output_coefficient_offsets.push_back(
              checked_size(tables.coefficient_terms.size(), "block coefficient table"));
        }

        maximum_block_monomial_count_ =
            std::max(maximum_block_monomial_count_, plan_record.monomial_count);
        tables.plans.push_back(plan_record);
      }
      species_record.plan_count =
          checked_size(tables.plans.size() - static_cast<std::size_t>(species_record.plan_begin),
                       "block plan table");
      species_record.output_storage_size = output_storage_size;

      for (const auto &route : program.routes) {
        YE3TKokkosBlockRouteRecord route_record;
        route_record.term_begin =
            checked_size(tables.route_term_coefficient_real.size(), "block route term table");
        route_record.term_count = checked_size(route.coefficients.size(), "block route term table");
        route_record.function_index =
            checked_index(route.function_index, "block route function index");
        if (route_record.term_count <= 0 || route_record.function_index < 0 ||
            route.coefficients.size() == 0)
          throw std::invalid_argument("YE3T Kokkos block route is empty");

        const bool general_route = !route.term_factor_offsets.empty();
        if (general_route) {
          if (route.term_factor_offsets.size() != route.coefficients.size() + 1 ||
              route.term_factor_plans.size() != route.term_factor_components.size())
            throw std::invalid_argument("YE3T Kokkos general block route is inconsistent");
        } else if (route.left_components.size() != route.coefficients.size() ||
                   (route.right_plan >= 0 &&
                    route.right_components.size() != route.coefficients.size())) {
          throw std::invalid_argument("YE3T Kokkos unary/binary block route is inconsistent");
        }

        for (int term = 0; term < route_record.term_count; ++term) {
          const auto coefficient = route.coefficients[static_cast<std::size_t>(term)];
          if (!finite_complex(coefficient) ||
              (route.real_coefficients && coefficient.imag() != 0.0))
            throw std::invalid_argument("YE3T Kokkos block route coefficient is invalid");
          tables.route_term_coefficient_real.push_back(coefficient.real());
          tables.route_term_coefficient_imaginary.push_back(coefficient.imag());

          auto append_factor = [&](std::int64_t local_plan_value, std::int64_t component_value) {
            const int local_plan = checked_index(local_plan_value, "block route plan");
            const int component = checked_index(component_value, "block route component");
            if (local_plan < 0 || local_plan >= species_record.plan_count)
              throw std::invalid_argument("YE3T Kokkos block route plan is out of range");
            const int global_plan = species_record.plan_begin + local_plan;
            if (component < 0 ||
                component >= tables.plans[static_cast<std::size_t>(global_plan)].output_count)
              throw std::invalid_argument("YE3T Kokkos block route component is out of range");
            tables.route_term_factor_plans.push_back(global_plan);
            tables.route_term_factor_components.push_back(component);
          };

          if (general_route) {
            const int begin =
                checked_index(route.term_factor_offsets[static_cast<std::size_t>(term)],
                              "block route factor offset");
            const int end =
                checked_index(route.term_factor_offsets[static_cast<std::size_t>(term + 1)],
                              "block route factor offset");
            if (begin < 0 || begin >= end || end > static_cast<int>(route.term_factor_plans.size()))
              throw std::invalid_argument("YE3T Kokkos block route factor range is invalid");
            for (int factor = begin; factor < end; ++factor)
              append_factor(route.term_factor_plans[static_cast<std::size_t>(factor)],
                            route.term_factor_components[static_cast<std::size_t>(factor)]);
          } else {
            append_factor(route.left_plan, route.left_components[static_cast<std::size_t>(term)]);
            if (route.right_plan >= 0)
              append_factor(route.right_plan,
                            route.right_components[static_cast<std::size_t>(term)]);
          }
          tables.route_term_factor_offsets.push_back(
              checked_size(tables.route_term_factor_plans.size(), "block route factor table"));
        }
        tables.routes.push_back(route_record);
      }
      species_record.route_count =
          checked_size(tables.routes.size() - static_cast<std::size_t>(species_record.route_begin),
                       "block route table");
      maximum_block_power_count_ = std::max(maximum_block_power_count_, species_record.power_count);
      maximum_block_plan_count_ = std::max(maximum_block_plan_count_, species_record.plan_count);
      maximum_block_route_count_ = std::max(maximum_block_route_count_, species_record.route_count);
      maximum_block_power_storage_ =
          std::max(maximum_block_power_storage_, species_record.power_storage_size);
      maximum_block_output_storage_ =
          std::max(maximum_block_output_storage_, species_record.output_storage_size);
      tables.species.push_back(species_record);
    }
    block_route_count_ = checked_size(tables.routes.size(), "block route table");
  }

  void flatten_scalar_programs(const YACEModel &model, HostScalarTables &tables)
  {
    for (int species_index = 0; species_index < species_count; ++species_index) {
      const auto &species_entry = model.species(species_index);
      const auto &block_program = species_entry.block_program;
      const auto &program = block_program.scalar_program;
      YE3TKokkosScalarSpeciesRecord species_record;
      species_record.base_begin = checked_size(tables.bases.size(), "scalar base table");
      species_record.node_begin = checked_size(tables.nodes.size(), "scalar node table");
      species_record.route_begin = checked_size(tables.routes.size(), "scalar route table");
      species_record.value_count = checked_index(program.value_count, "scalar value count");
      if (species_record.value_count < 0)
        throw std::invalid_argument("YE3T Kokkos scalar value count is negative");

      const bool has_payload = !program.bases.empty() || !program.nodes.empty() ||
          !program.routes.empty() || program.value_count != 0;
      if (has_payload && program.routes.empty())
        throw std::invalid_argument("YE3T Kokkos scalar program has unreferenced payload");
      if (!program.routes.empty()) {
        if (block_program.plan_hash.empty() || block_program.evaluator_plan_hash.empty() ||
            block_program.dispatch.empty())
          throw std::invalid_argument(
              "YE3T Kokkos scalar program has incomplete semantic identity");
        tables.semantic_ids.push_back(block_program.plan_hash);
        tables.semantic_ids.push_back(block_program.evaluator_plan_hash);
        tables.semantic_ids.push_back(block_program.dispatch);
      }

      std::vector<int> value_bases;
      std::vector<int> value_exponents;
      value_bases.reserve(static_cast<std::size_t>(species_record.value_count));
      value_exponents.reserve(static_cast<std::size_t>(species_record.value_count));
      for (std::size_t base_index = 0; base_index < program.bases.size(); ++base_index) {
        const auto &base = program.bases[base_index];
        if (base.base_id.empty() || base.left_channels.empty() ||
            base.left_channels.size() != base.right_channels.size() ||
            base.left_channels.size() != base.coefficients.size())
          throw std::invalid_argument("YE3T Kokkos scalar base tables are inconsistent");
        YE3TKokkosScalarBaseRecord base_record;
        base_record.term_begin = checked_size(tables.left_channels.size(), "scalar term table");
        base_record.term_count = checked_size(base.coefficients.size(), "scalar term table");
        for (std::size_t term = 0; term < base.coefficients.size(); ++term) {
          const int left = checked_index(base.left_channels[term], "scalar left channel");
          const int right = checked_index(base.right_channels[term], "scalar right channel");
          const auto coefficient = base.coefficients[term];
          if (left < 0 || right < 0 || left >= static_cast<int>(species_entry.channels.size()) ||
              right >= static_cast<int>(species_entry.channels.size()) ||
              !finite_complex(coefficient))
            throw std::invalid_argument("YE3T Kokkos scalar base term is invalid");
          tables.left_channels.push_back(left);
          tables.right_channels.push_back(right);
          tables.coefficient_real.push_back(coefficient.real());
          tables.coefficient_imaginary.push_back(coefficient.imag());
        }
        tables.bases.push_back(base_record);
        tables.semantic_ids.push_back(base.base_id);
        value_bases.push_back(static_cast<int>(base_index));
        value_exponents.push_back(1);
      }
      species_record.base_count = checked_size(program.bases.size(), "scalar base table");

      for (std::size_t node_index = 0; node_index < program.nodes.size(); ++node_index) {
        const auto &node = program.nodes[node_index];
        YE3TKokkosScalarNodeRecord node_record;
        node_record.left_value = checked_index(node.left_value, "scalar left value");
        node_record.right_value = checked_index(node.right_value, "scalar right value");
        node_record.exponent = checked_index(node.exponent, "scalar exponent");
        node_record.base_index = checked_index(node.base_index, "scalar base index");
        const int available = species_record.base_count + static_cast<int>(node_index);
        if (node_record.left_value < 0 || node_record.right_value < 0 ||
            node_record.left_value >= available || node_record.right_value >= available ||
            node_record.exponent <= 1 || node_record.base_index < 0 ||
            node_record.base_index >= species_record.base_count ||
            value_bases[static_cast<std::size_t>(node_record.left_value)] !=
                node_record.base_index ||
            value_bases[static_cast<std::size_t>(node_record.right_value)] !=
                node_record.base_index ||
            value_exponents[static_cast<std::size_t>(node_record.left_value)] +
                    value_exponents[static_cast<std::size_t>(node_record.right_value)] !=
                node_record.exponent)
          throw std::invalid_argument(
              "YE3T Kokkos scalar node is not a valid topological power step");
        tables.nodes.push_back(node_record);
        value_bases.push_back(node_record.base_index);
        value_exponents.push_back(node_record.exponent);
      }
      species_record.node_count = checked_size(program.nodes.size(), "scalar node table");
      if (species_record.value_count != species_record.base_count + species_record.node_count)
        throw std::invalid_argument("YE3T Kokkos scalar value count is inconsistent");

      const int catalogue_count =
          checked_index(block_program.catalogue_function_count, "scalar catalogue function count");
      for (const auto &route : program.routes) {
        YE3TKokkosScalarRouteRecord route_record;
        route_record.function_index = checked_index(route.function_index, "scalar route function");
        route_record.value_index = checked_index(route.value_index, "scalar route value");
        route_record.scale_real = route.scale.real();
        route_record.scale_imaginary = route.scale.imag();
        if (catalogue_count <= 0 || route_record.function_index < 0 ||
            route_record.function_index >= catalogue_count || route_record.value_index < 0 ||
            route_record.value_index >= species_record.value_count ||
            !finite_complex(route.scale) || route.factorization_id.empty())
          throw std::invalid_argument("YE3T Kokkos scalar route is invalid");
        tables.routes.push_back(route_record);
        tables.semantic_ids.push_back(route.factorization_id);
      }
      species_record.route_count = checked_size(program.routes.size(), "scalar route table");
      maximum_scalar_value_count_ =
          std::max(maximum_scalar_value_count_, species_record.value_count);
      tables.species.push_back(species_record);
    }
    scalar_route_count_ = checked_size(tables.routes.size(), "scalar route table");
  }

  void flatten_coupled_programs(const YACEModel &model, HostCoupledTables &tables)
  {
    for (int species_index = 0; species_index < species_count; ++species_index) {
      const auto &species_entry = model.species(species_index);
      const auto &program = species_entry.block_program;
      YE3TKokkosCoupledSpeciesRecord species_record;
      species_record.plan_begin = checked_size(tables.plans.size(), "coupled-product plan table");
      if (!program.coupled_product_plans.empty()) {
        if (program.plan_hash.empty() || program.evaluator_plan_hash.empty() ||
            program.dispatch.empty())
          throw std::invalid_argument("YE3T Kokkos coupled-product program has incomplete semantic "
                                      "identity");
        tables.semantic_ids.push_back(program.plan_hash);
        tables.semantic_ids.push_back(program.evaluator_plan_hash);
        tables.semantic_ids.push_back(program.dispatch);
      }

      const int catalogue_count = checked_index(program.catalogue_function_count,
                                                "coupled-product catalogue function count");
      std::vector<int> selected_functions;
      for (const auto &plan : program.coupled_product_plans) {
        YE3TKokkosCoupledPlanRecord plan_record;
        plan_record.node_begin = checked_size(tables.nodes.size(), "coupled-product node table");
        plan_record.node_count =
            checked_size(plan.node_offsets.size(), "coupled-product node table");
        plan_record.leaf_begin =
            checked_size(tables.leaf_input_channels.size(), "coupled-product leaf table");
        plan_record.leaf_count =
            checked_size(plan.leaf_input_components.size(), "coupled-product leaf table");
        plan_record.coefficient_begin =
            checked_size(tables.coefficient_values.size(), "coupled-product coefficient table");
        plan_record.coefficient_count =
            checked_size(plan.coefficient_values.size(), "coupled-product coefficient table");
        plan_record.readout_begin =
            checked_size(tables.readout_components.size(), "coupled-product readout table");
        plan_record.readout_count =
            checked_size(plan.readout_components.size(), "coupled-product readout table");
        plan_record.total_component_count =
            checked_index(plan.total_node_components, "coupled-product component workspace");
        plan_record.function_index =
            checked_index(plan.function_index, "coupled-product function index");
        plan_record.operation_estimate =
            checked_index(plan.operation_estimate, "coupled-product operation estimate");

        if (plan_record.node_count <= 0 || plan_record.leaf_count <= 0 ||
            plan_record.coefficient_count <= 0 || plan_record.readout_count <= 0 ||
            plan_record.total_component_count <= 0 || catalogue_count <= 0 ||
            plan_record.function_index < 0 || plan_record.function_index >= catalogue_count ||
            plan_record.operation_estimate < 0 ||
            plan.node_dimensions.size() != plan.node_offsets.size() ||
            plan.node_leaf_offsets.size() != plan.node_offsets.size() ||
            plan.node_coefficient_offsets.size() != plan.node_offsets.size() + 1 ||
            plan.coefficient_left_components.size() != plan.coefficient_values.size() ||
            plan.coefficient_right_components.size() != plan.coefficient_values.size() ||
            plan.coefficient_output_components.size() != plan.coefficient_values.size() ||
            plan.readout_components.size() != plan.readout_coefficients.size())
          throw std::invalid_argument("YE3T Kokkos coupled-product plan tables are inconsistent");
        if (std::find(selected_functions.begin(), selected_functions.end(),
                      plan_record.function_index) != selected_functions.end())
          throw std::invalid_argument("YE3T Kokkos coupled-product function is selected twice");
        selected_functions.push_back(plan_record.function_index);

        const auto decision =
            std::find_if(program.decisions.begin(), program.decisions.end(),
                         [&](const YACEEvaluatorDecision &entry) {
                           return entry.function_index == plan_record.function_index;
                         });
        if (decision == program.decisions.end() ||
            decision->selected_evaluator != YACEEvaluatorKind::COUPLED_PRODUCT_DAG ||
            decision->selected_candidate_id.empty() || decision->selected_candidate_index < 0 ||
            decision->selected_candidate_index >=
                static_cast<std::int64_t>(decision->candidates.size()) ||
            decision->candidates[static_cast<std::size_t>(decision->selected_candidate_index)]
                    .candidate_id != decision->selected_candidate_id ||
            decision->candidates[static_cast<std::size_t>(decision->selected_candidate_index)]
                    .evaluator != YACEEvaluatorKind::COUPLED_PRODUCT_DAG)
          throw std::invalid_argument("YE3T Kokkos coupled-product plan is not the selected "
                                      "compiler candidate");
        tables.semantic_ids.push_back(decision->selected_candidate_id);

        if (plan.node_coefficient_offsets.front() != 0 ||
            plan.node_coefficient_offsets.back() !=
                static_cast<std::int64_t>(plan.coefficient_values.size()))
          throw std::invalid_argument(
              "YE3T Kokkos coupled-product coefficient offsets are invalid");
        int next_component = 0;
        for (int node_index = 0; node_index < plan_record.node_count; ++node_index) {
          YE3TKokkosCoupledNodeRecord node_record;
          node_record.component_offset =
              checked_index(plan.node_offsets[static_cast<std::size_t>(node_index)],
                            "coupled-product node component offset");
          node_record.dimension =
              checked_index(plan.node_dimensions[static_cast<std::size_t>(node_index)],
                            "coupled-product node dimension");
          node_record.leaf_offset =
              checked_index(plan.node_leaf_offsets[static_cast<std::size_t>(node_index)],
                            "coupled-product node leaf offset");
          const int coefficient_begin =
              checked_index(plan.node_coefficient_offsets[static_cast<std::size_t>(node_index)],
                            "coupled-product node coefficient offset");
          const int coefficient_end =
              checked_index(plan.node_coefficient_offsets[static_cast<std::size_t>(node_index + 1)],
                            "coupled-product node coefficient offset");
          node_record.coefficient_begin = coefficient_begin;
          node_record.coefficient_count = coefficient_end - coefficient_begin;
          if (node_record.component_offset != next_component || node_record.dimension <= 0 ||
              node_record.dimension % 2 != 1 || coefficient_begin < 0 ||
              coefficient_begin > coefficient_end ||
              coefficient_end > plan_record.coefficient_count)
            throw std::invalid_argument("YE3T Kokkos coupled-product node layout is invalid");

          if (node_record.leaf_offset >= 0) {
            if (node_record.coefficient_count != 0 ||
                node_record.leaf_offset + node_record.dimension > plan_record.leaf_count)
              throw std::invalid_argument("YE3T Kokkos coupled-product leaf node is invalid");
          } else {
            if (node_record.leaf_offset != -1 || node_record.coefficient_count <= 0)
              throw std::invalid_argument("YE3T Kokkos coupled-product product node is invalid");
            for (int coefficient = coefficient_begin; coefficient < coefficient_end;
                 ++coefficient) {
              const std::size_t index = static_cast<std::size_t>(coefficient);
              const int left = checked_index(plan.coefficient_left_components[index],
                                             "coupled-product left component");
              const int right = checked_index(plan.coefficient_right_components[index],
                                              "coupled-product right component");
              const int output = checked_index(plan.coefficient_output_components[index],
                                               "coupled-product output component");
              if (left < 0 || right < 0 || left >= node_record.component_offset ||
                  right >= node_record.component_offset || output < node_record.component_offset ||
                  output >= node_record.component_offset + node_record.dimension ||
                  !std::isfinite(plan.coefficient_values[index]))
                throw std::invalid_argument("YE3T Kokkos coupled-product coefficient is invalid");
            }
          }
          next_component += node_record.dimension;
          tables.nodes.push_back(node_record);
        }
        if (next_component != plan_record.total_component_count)
          throw std::invalid_argument(
              "YE3T Kokkos coupled-product component count is inconsistent");

        for (const std::int64_t component_value : plan.leaf_input_components) {
          const int component =
              checked_index(component_value, "coupled-product leaf input channel");
          if (component < 0 || component >= static_cast<int>(species_entry.channels.size()))
            throw std::invalid_argument("YE3T Kokkos coupled-product leaf input is out of range");
          tables.leaf_input_channels.push_back(component);
        }
        for (std::size_t coefficient = 0; coefficient < plan.coefficient_values.size();
             ++coefficient) {
          tables.coefficient_left_components.push_back(checked_index(
              plan.coefficient_left_components[coefficient], "coupled-product left component"));
          tables.coefficient_right_components.push_back(checked_index(
              plan.coefficient_right_components[coefficient], "coupled-product right component"));
          tables.coefficient_output_components.push_back(checked_index(
              plan.coefficient_output_components[coefficient], "coupled-product output component"));
          tables.coefficient_values.push_back(plan.coefficient_values[coefficient]);
        }
        for (std::size_t readout = 0; readout < plan.readout_components.size(); ++readout) {
          const int component =
              checked_index(plan.readout_components[readout], "coupled-product readout component");
          bool scalar_root = false;
          for (int node_index = 0; node_index < plan_record.node_count; ++node_index) {
            const auto &node =
                tables.nodes[static_cast<std::size_t>(plan_record.node_begin + node_index)];
            if (component >= node.component_offset &&
                component < node.component_offset + node.dimension) {
              scalar_root = node.dimension == 1;
              break;
            }
          }
          if (!scalar_root || !std::isfinite(plan.readout_coefficients[readout]))
            throw std::invalid_argument("YE3T Kokkos coupled-product readout is invalid");
          tables.readout_components.push_back(component);
          tables.readout_coefficients.push_back(plan.readout_coefficients[readout]);
        }

        maximum_coupled_component_count_ =
            std::max(maximum_coupled_component_count_, plan_record.total_component_count);
        tables.plans.push_back(plan_record);
      }
      for (const auto &decision : program.decisions) {
        if (decision.selected_evaluator != YACEEvaluatorKind::COUPLED_PRODUCT_DAG) continue;
        const int function_index =
            checked_index(decision.function_index, "coupled-product decision function");
        if (function_index < 0 || function_index >= catalogue_count ||
            std::count(selected_functions.begin(), selected_functions.end(), function_index) != 1)
          throw std::invalid_argument("YE3T Kokkos coupled-product decision has no unique device "
                                      "plan");
      }
      species_record.plan_count =
          checked_size(tables.plans.size() - static_cast<std::size_t>(species_record.plan_begin),
                       "coupled-product plan table");
      tables.species.push_back(species_record);
    }
    coupled_plan_count_ = checked_size(tables.plans.size(), "coupled-product plan table");
  }

  static bool all_finite(const std::vector<double> &values)
  {
    return std::all_of(values.begin(), values.end(), [](double value) {
      return std::isfinite(value);
    });
  }

  template <class Value> void count_bytes(std::size_t count)
  {
    if (count > (std::numeric_limits<std::size_t>::max() - static_bytes_) / sizeof(Value))
      throw std::overflow_error("YE3T Kokkos plan byte count overflow");
    static_bytes_ += count * sizeof(Value);
  }

  template <class View, class Value>
  static void copy_view(View &destination, const char *label, const std::vector<Value> &source)
  {
    destination = View(std::string(label), source.size());
    auto mirror = Kokkos::create_mirror_view(destination);
    for (std::size_t index = 0; index < source.size(); ++index) mirror(index) = source[index];
    Kokkos::deep_copy(destination, mirror);
  }

  static void hash_unsigned(SHA256Builder &hash, std::uint64_t value)
  {
    std::array<unsigned char, 8> bytes{};
    for (int byte = 7; byte >= 0; --byte) {
      bytes[static_cast<std::size_t>(7 - byte)] = static_cast<unsigned char>(value >> (byte * 8));
    }
    hash.update(bytes.data(), bytes.size());
  }

  template <class Value> static void hash_value(SHA256Builder &hash, Value value)
  {
    if constexpr (std::is_integral_v<Value>) {
      hash_unsigned(hash, static_cast<std::uint64_t>(static_cast<std::int64_t>(value)));
    } else {
      static_assert(std::is_same_v<Value, double>);
      static_assert(sizeof(double) == sizeof(std::uint64_t));
      static_assert(std::numeric_limits<double>::is_iec559);
      std::uint64_t bits = 0;
      std::memcpy(&bits, &value, sizeof(bits));
      hash_unsigned(hash, bits);
    }
  }

  template <class Value>
  static void hash_vector(SHA256Builder &hash, const std::vector<Value> &values)
  {
    hash_unsigned(hash, static_cast<std::uint64_t>(values.size()));
    for (const Value value : values) hash_value(hash, value);
  }

  static void hash_string(SHA256Builder &hash, const std::string &value)
  {
    hash_unsigned(hash, static_cast<std::uint64_t>(value.size()));
    hash.update(value);
  }

  static void hash_block_species_records(SHA256Builder &hash,
                                         const std::vector<YE3TKokkosBlockSpeciesRecord> &records)
  {
    hash_unsigned(hash, static_cast<std::uint64_t>(records.size()));
    for (const auto &record : records) {
      hash_value(hash, record.power_begin);
      hash_value(hash, record.power_count);
      hash_value(hash, record.plan_begin);
      hash_value(hash, record.plan_count);
      hash_value(hash, record.route_begin);
      hash_value(hash, record.route_count);
      hash_value(hash, record.power_storage_size);
      hash_value(hash, record.output_storage_size);
    }
  }

  static void hash_block_power_records(SHA256Builder &hash,
                                       const std::vector<YE3TKokkosBlockPowerRecord> &records)
  {
    hash_unsigned(hash, static_cast<std::uint64_t>(records.size()));
    for (const auto &record : records) {
      hash_value(hash, record.channel);
      hash_value(hash, record.maximum_exponent);
      hash_value(hash, record.storage_offset);
    }
  }

  static void hash_block_plan_records(SHA256Builder &hash,
                                      const std::vector<YE3TKokkosBlockPlanRecord> &records)
  {
    hash_unsigned(hash, static_cast<std::uint64_t>(records.size()));
    for (const auto &record : records) {
      hash_value(hash, record.input_begin);
      hash_value(hash, record.input_count);
      hash_value(hash, record.monomial_begin);
      hash_value(hash, record.monomial_count);
      hash_value(hash, record.output_begin);
      hash_value(hash, record.output_count);
      hash_value(hash, record.output_storage_offset);
      hash_value(hash, record.output_L);
      hash_value(hash, record.flags);
    }
  }

  static void hash_block_route_records(SHA256Builder &hash,
                                       const std::vector<YE3TKokkosBlockRouteRecord> &records)
  {
    hash_unsigned(hash, static_cast<std::uint64_t>(records.size()));
    for (const auto &record : records) {
      hash_value(hash, record.term_begin);
      hash_value(hash, record.term_count);
      hash_value(hash, record.function_index);
    }
  }

  static void hash_scalar_species_records(SHA256Builder &hash,
                                          const std::vector<YE3TKokkosScalarSpeciesRecord> &records)
  {
    hash_unsigned(hash, static_cast<std::uint64_t>(records.size()));
    for (const auto &record : records) {
      hash_value(hash, record.base_begin);
      hash_value(hash, record.base_count);
      hash_value(hash, record.node_begin);
      hash_value(hash, record.node_count);
      hash_value(hash, record.route_begin);
      hash_value(hash, record.route_count);
      hash_value(hash, record.value_count);
    }
  }

  static void hash_scalar_base_records(SHA256Builder &hash,
                                       const std::vector<YE3TKokkosScalarBaseRecord> &records)
  {
    hash_unsigned(hash, static_cast<std::uint64_t>(records.size()));
    for (const auto &record : records) {
      hash_value(hash, record.term_begin);
      hash_value(hash, record.term_count);
    }
  }

  static void hash_scalar_node_records(SHA256Builder &hash,
                                       const std::vector<YE3TKokkosScalarNodeRecord> &records)
  {
    hash_unsigned(hash, static_cast<std::uint64_t>(records.size()));
    for (const auto &record : records) {
      hash_value(hash, record.left_value);
      hash_value(hash, record.right_value);
      hash_value(hash, record.exponent);
      hash_value(hash, record.base_index);
    }
  }

  static void hash_scalar_route_records(SHA256Builder &hash,
                                        const std::vector<YE3TKokkosScalarRouteRecord> &records)
  {
    hash_unsigned(hash, static_cast<std::uint64_t>(records.size()));
    for (const auto &record : records) {
      hash_value(hash, record.function_index);
      hash_value(hash, record.value_index);
      hash_value(hash, record.scale_real);
      hash_value(hash, record.scale_imaginary);
    }
  }

  static void
  hash_coupled_species_records(SHA256Builder &hash,
                               const std::vector<YE3TKokkosCoupledSpeciesRecord> &records)
  {
    hash_unsigned(hash, static_cast<std::uint64_t>(records.size()));
    for (const auto &record : records) {
      hash_value(hash, record.plan_begin);
      hash_value(hash, record.plan_count);
    }
  }

  static void hash_coupled_plan_records(SHA256Builder &hash,
                                        const std::vector<YE3TKokkosCoupledPlanRecord> &records)
  {
    hash_unsigned(hash, static_cast<std::uint64_t>(records.size()));
    for (const auto &record : records) {
      hash_value(hash, record.node_begin);
      hash_value(hash, record.node_count);
      hash_value(hash, record.leaf_begin);
      hash_value(hash, record.leaf_count);
      hash_value(hash, record.coefficient_begin);
      hash_value(hash, record.coefficient_count);
      hash_value(hash, record.readout_begin);
      hash_value(hash, record.readout_count);
      hash_value(hash, record.total_component_count);
      hash_value(hash, record.function_index);
      hash_value(hash, record.operation_estimate);
    }
  }

  static void hash_coupled_node_records(SHA256Builder &hash,
                                        const std::vector<YE3TKokkosCoupledNodeRecord> &records)
  {
    hash_unsigned(hash, static_cast<std::uint64_t>(records.size()));
    for (const auto &record : records) {
      hash_value(hash, record.component_offset);
      hash_value(hash, record.dimension);
      hash_value(hash, record.leaf_offset);
      hash_value(hash, record.coefficient_begin);
      hash_value(hash, record.coefficient_count);
    }
  }

  std::string block_signature_hash(const std::string &direct_signature,
                                   const HostBlockTables &tables) const
  {
    SHA256Builder hash;
    hash.update("ye3t_kokkos_block_flat_v1");
    hash_string(hash, direct_signature);
    hash_value(hash, maximum_block_power_storage_);
    hash_value(hash, maximum_block_output_storage_);
    hash_value(hash, maximum_block_monomial_count_);
    hash_value(hash, maximum_block_power_count_);
    hash_value(hash, maximum_block_plan_count_);
    hash_value(hash, maximum_block_route_count_);
    hash_value(hash, block_route_count_);
    hash_block_species_records(hash, tables.species);
    hash_block_power_records(hash, tables.powers);
    hash_block_plan_records(hash, tables.plans);
    hash_block_route_records(hash, tables.routes);
    hash_vector(hash, tables.plan_input_channels);
    hash_vector(hash, tables.plan_input_power_offsets);
    hash_vector(hash, tables.monomial_factor_offsets);
    hash_vector(hash, tables.monomial_factor_components);
    hash_vector(hash, tables.monomial_factor_exponents);
    hash_vector(hash, tables.output_coefficient_offsets);
    hash_vector(hash, tables.coefficient_terms);
    hash_vector(hash, tables.coefficient_real);
    hash_vector(hash, tables.coefficient_imaginary);
    hash_vector(hash, tables.direct_input_channels);
    hash_vector(hash, tables.direct_input_scales);
    hash_vector(hash, tables.route_term_factor_offsets);
    hash_vector(hash, tables.route_term_factor_plans);
    hash_vector(hash, tables.route_term_factor_components);
    hash_vector(hash, tables.route_term_coefficient_real);
    hash_vector(hash, tables.route_term_coefficient_imaginary);
    hash_unsigned(hash, static_cast<std::uint64_t>(tables.semantic_ids.size()));
    for (const auto &identity : tables.semantic_ids) hash_string(hash, identity);
    return hash.finish();
  }

  std::string scalar_signature_hash(const std::string &previous_signature,
                                    const HostScalarTables &tables) const
  {
    SHA256Builder hash;
    hash.update("ye3t_kokkos_scalar_power_flat_v1");
    hash_string(hash, previous_signature);
    hash_value(hash, maximum_scalar_value_count_);
    hash_value(hash, scalar_route_count_);
    hash_scalar_species_records(hash, tables.species);
    hash_scalar_base_records(hash, tables.bases);
    hash_scalar_node_records(hash, tables.nodes);
    hash_scalar_route_records(hash, tables.routes);
    hash_vector(hash, tables.left_channels);
    hash_vector(hash, tables.right_channels);
    hash_vector(hash, tables.coefficient_real);
    hash_vector(hash, tables.coefficient_imaginary);
    hash_unsigned(hash, static_cast<std::uint64_t>(tables.semantic_ids.size()));
    for (const auto &identity : tables.semantic_ids) hash_string(hash, identity);
    return hash.finish();
  }

  std::string coupled_signature_hash(const std::string &previous_signature,
                                     const HostCoupledTables &tables) const
  {
    SHA256Builder hash;
    hash.update("ye3t_kokkos_coupled_product_flat_v1");
    hash_string(hash, previous_signature);
    hash_value(hash, maximum_coupled_component_count_);
    hash_value(hash, coupled_plan_count_);
    hash_coupled_species_records(hash, tables.species);
    hash_coupled_plan_records(hash, tables.plans);
    hash_coupled_node_records(hash, tables.nodes);
    hash_vector(hash, tables.leaf_input_channels);
    hash_vector(hash, tables.coefficient_left_components);
    hash_vector(hash, tables.coefficient_right_components);
    hash_vector(hash, tables.coefficient_output_components);
    hash_vector(hash, tables.coefficient_values);
    hash_vector(hash, tables.readout_components);
    hash_vector(hash, tables.readout_coefficients);
    hash_unsigned(hash, static_cast<std::uint64_t>(tables.semantic_ids.size()));
    for (const auto &identity : tables.semantic_ids) hash_string(hash, identity);
    return hash.finish();
  }

  std::string vjp_schedule_signature_hash(const std::string &direct_signature,
                                          const std::vector<int> &bond_group_offsets,
                                          const std::vector<int> &group_term_offsets,
                                          const std::vector<int> &group_harmonics,
                                          const std::vector<int> &group_terms) const
  {
    SHA256Builder hash;
    hash.update("ye3t_kokkos_vjp_grouped_harmonic_v1");
    hash_string(hash, direct_signature);
    hash_vector(hash, bond_group_offsets);
    hash_vector(hash, group_term_offsets);
    hash_vector(hash, group_harmonics);
    hash_vector(hash, group_terms);
    return hash.finish();
  }

  template <class... Vectors> std::string signature_hash(const Vectors &...vectors) const
  {
    SHA256Builder hash;
    hash.update("ye3t_kokkos_direct_flat_v4_owned_reverse");
    hash_value(hash, species_count);
    hash_value(hash, bond_count);
    hash_value(hash, maximum_angular_momentum);
    (hash_vector(hash, vectors), ...);
    return hash.finish();
  }

  template <class Value> static double first_or_zero(const std::vector<Value> &values)
  {
    return values.empty() ? 0.0 : static_cast<double>(values.front());
  }

  static std::array<double, 4> block_probe_values(const HostBlockTables &tables)
  {
    double species_value = static_cast<double>(tables.species.size());
    if (!tables.species.empty()) {
      const auto &record = tables.species.front();
      species_value += record.power_begin + record.power_count + record.plan_begin +
          record.plan_count + record.route_begin + record.route_count + record.power_storage_size +
          record.output_storage_size;
    }

    double plan_value =
        static_cast<double>(tables.powers.size()) + static_cast<double>(tables.plans.size());
    if (!tables.powers.empty()) {
      const auto &record = tables.powers.front();
      plan_value += record.channel + record.maximum_exponent + record.storage_offset;
    }
    if (!tables.plans.empty()) {
      const auto &record = tables.plans.front();
      plan_value += record.input_begin + record.input_count + record.monomial_begin +
          record.monomial_count + record.output_begin + record.output_count +
          record.output_storage_offset + record.output_L + record.flags;
    }

    double coefficient_value = static_cast<double>(tables.routes.size()) +
        tables.monomial_factor_offsets.size() + tables.output_coefficient_offsets.size() +
        tables.coefficient_terms.size();
    if (!tables.routes.empty()) {
      const auto &record = tables.routes.front();
      coefficient_value += record.term_begin + record.term_count + record.function_index;
    }
    coefficient_value += first_or_zero(tables.monomial_factor_offsets) +
        first_or_zero(tables.monomial_factor_components) +
        first_or_zero(tables.monomial_factor_exponents) +
        first_or_zero(tables.output_coefficient_offsets) + first_or_zero(tables.coefficient_terms) +
        first_or_zero(tables.coefficient_real) + first_or_zero(tables.coefficient_imaginary);

    const double route_value = static_cast<double>(tables.plan_input_channels.size()) +
        tables.plan_input_power_offsets.size() + tables.direct_input_channels.size() +
        tables.direct_input_scales.size() + tables.route_term_factor_offsets.size() +
        tables.route_term_factor_plans.size() + tables.route_term_factor_components.size() +
        tables.route_term_coefficient_real.size() + tables.route_term_coefficient_imaginary.size() +
        first_or_zero(tables.plan_input_channels) + first_or_zero(tables.plan_input_power_offsets) +
        first_or_zero(tables.direct_input_channels) + first_or_zero(tables.direct_input_scales) +
        first_or_zero(tables.route_term_factor_offsets) +
        first_or_zero(tables.route_term_factor_plans) +
        first_or_zero(tables.route_term_factor_components) +
        first_or_zero(tables.route_term_coefficient_real) +
        first_or_zero(tables.route_term_coefficient_imaginary);
    return {species_value, plan_value, coefficient_value, route_value};
  }

  static std::array<double, 4> scalar_probe_values(const HostScalarTables &tables)
  {
    double species_value = static_cast<double>(tables.species.size());
    if (!tables.species.empty()) {
      const auto &record = tables.species.front();
      species_value += record.base_begin + record.base_count + record.node_begin +
          record.node_count + record.route_begin + record.route_count + record.value_count;
    }

    double program_value =
        static_cast<double>(tables.bases.size()) + static_cast<double>(tables.nodes.size());
    if (!tables.bases.empty()) {
      const auto &record = tables.bases.front();
      program_value += record.term_begin + record.term_count;
    }
    if (!tables.nodes.empty()) {
      const auto &record = tables.nodes.front();
      program_value += record.left_value + record.right_value + record.exponent + record.base_index;
    }

    double route_value = static_cast<double>(tables.routes.size());
    if (!tables.routes.empty()) {
      const auto &record = tables.routes.front();
      route_value +=
          record.function_index + record.value_index + record.scale_real + record.scale_imaginary;
    }

    const double coefficient_value = static_cast<double>(tables.left_channels.size()) +
        static_cast<double>(tables.right_channels.size()) +
        static_cast<double>(tables.coefficient_real.size()) +
        static_cast<double>(tables.coefficient_imaginary.size()) +
        first_or_zero(tables.left_channels) + first_or_zero(tables.right_channels) +
        first_or_zero(tables.coefficient_real) + first_or_zero(tables.coefficient_imaginary);
    return {species_value, program_value, route_value, coefficient_value};
  }

  static std::array<double, 4> coupled_probe_values(const HostCoupledTables &tables)
  {
    double species_value =
        static_cast<double>(tables.species.size()) + static_cast<double>(tables.plans.size());
    if (!tables.species.empty()) {
      const auto &record = tables.species.front();
      species_value += record.plan_begin + record.plan_count;
    }
    if (!tables.plans.empty()) {
      const auto &record = tables.plans.front();
      species_value += record.node_begin + record.node_count + record.leaf_begin +
          record.leaf_count + record.coefficient_begin + record.coefficient_count +
          record.readout_begin + record.readout_count + record.total_component_count +
          record.function_index + record.operation_estimate;
    }

    double node_value = static_cast<double>(tables.nodes.size());
    if (!tables.nodes.empty()) {
      const auto &first = tables.nodes.front();
      const auto &last = tables.nodes.back();
      node_value += first.component_offset + first.dimension + first.leaf_offset +
          first.coefficient_begin + first.coefficient_count + last.component_offset +
          last.dimension + last.leaf_offset + last.coefficient_begin + last.coefficient_count;
    }

    const double coefficient_value = static_cast<double>(tables.leaf_input_channels.size()) +
        static_cast<double>(tables.coefficient_left_components.size()) +
        static_cast<double>(tables.coefficient_right_components.size()) +
        static_cast<double>(tables.coefficient_output_components.size()) +
        static_cast<double>(tables.coefficient_values.size()) +
        first_or_zero(tables.leaf_input_channels) +
        first_or_zero(tables.coefficient_left_components) +
        first_or_zero(tables.coefficient_right_components) +
        first_or_zero(tables.coefficient_output_components) +
        first_or_zero(tables.coefficient_values);
    const double readout_value = static_cast<double>(tables.readout_components.size()) +
        static_cast<double>(tables.readout_coefficients.size()) +
        first_or_zero(tables.readout_components) + first_or_zero(tables.readout_coefficients);
    return {species_value, node_value, coefficient_value, readout_value};
  }

  std::array<double, 8> probe_values(const std::vector<int> &host_species_channel_offsets,
                                     const std::vector<int> &host_species_source_offsets,
                                     const std::vector<int> &host_species_power_offsets,
                                     const std::vector<int> &host_species_node_offsets,
                                     const std::vector<int> &host_species_monomial_offsets,
                                     const std::vector<int> &host_species_node_level_table_offsets,
                                     const std::vector<int> &host_node_level_offsets,
                                     const std::vector<int> &host_node_level_nodes,
                                     const std::vector<double> &host_reference_energies,
                                     const std::vector<double> &host_embedding_scales,
                                     const std::vector<double> &host_density_safe_limits,
                                     const std::vector<int> &host_full_channel_sources,
                                     const std::vector<int> &host_full_channel_transforms,
                                     const std::vector<int> &host_power_channels,
                                     const std::vector<int> &host_power_exponents,
                                     const std::vector<int> &host_binary_node_left,
                                     const std::vector<int> &host_binary_node_right,
                                     const std::vector<int> &host_monomial_nodes,
                                     const std::vector<double> &host_monomial_coefficients,
                                     const std::vector<int> &host_bond_radial_counts,
                                     const std::vector<int> &host_bond_angular_maxima,
                                     const std::vector<int> &host_bond_radial_base_counts,
                                     const std::vector<int> &host_bond_interval_counts,
                                     const std::vector<double> &host_bond_cutoffs,
                                     const std::vector<int> &host_radial_spline_offsets,
                                     const std::vector<double> &host_radial_splines,
                                     const std::vector<int> &host_contracted_spline_offsets,
                                     const std::vector<double> &host_contracted_splines,
                                     const std::vector<int> &host_radial_map_offsets,
                                     const std::vector<int> &host_radial_channel_outputs,
                                     const std::vector<int> &host_radial_channel_indices,
                                     const std::vector<int> &host_angular_map_offsets,
                                     const std::vector<int> &host_angular_channel_outputs,
                                     const std::vector<int> &host_contracted_channel_indices,
                                     const std::vector<int> &host_angular_channel_indices,
                                     const std::vector<int> &host_vjp_bond_group_offsets,
                                     const std::vector<int> &host_vjp_group_term_offsets,
                                     const std::vector<int> &host_vjp_group_harmonics,
                                     const std::vector<int> &host_vjp_group_terms,
                                     const std::vector<double> &host_angular_plan) const
  {
    return {static_cast<double>(species_count) +
                static_cast<double>(host_species_channel_offsets.size()) +
                host_reference_energies.front() + host_embedding_scales.front(),
            static_cast<double>(bond_count) + host_bond_cutoffs.front() +
                static_cast<double>(host_bond_interval_counts.front()) +
                host_radial_splines.front() + host_contracted_splines.front(),
            static_cast<double>(maximum_angular_momentum) + host_angular_plan.front() +
                host_angular_plan.back(),
            static_cast<double>(host_species_source_offsets[1]) +
                static_cast<double>(host_full_channel_sources.front()) +
                static_cast<double>(host_full_channel_transforms.front()),
            static_cast<double>(host_species_power_offsets[1]) +
                first_or_zero(host_power_channels) + first_or_zero(host_power_exponents),
            static_cast<double>(host_species_node_offsets[1]) +
                first_or_zero(host_binary_node_left) + first_or_zero(host_binary_node_right) +
                static_cast<double>(host_species_node_level_table_offsets[1]) +
                static_cast<double>(host_node_level_offsets.front()) +
                first_or_zero(host_node_level_nodes),
            static_cast<double>(host_species_monomial_offsets[1]) +
                first_or_zero(host_monomial_nodes) + first_or_zero(host_monomial_coefficients) +
                host_density_safe_limits.front(),
            static_cast<double>(host_bond_radial_counts.front()) +
                static_cast<double>(host_bond_angular_maxima.front()) +
                static_cast<double>(host_bond_radial_base_counts.front()) +
                static_cast<double>(host_radial_spline_offsets[1]) +
                static_cast<double>(host_contracted_spline_offsets[1]) +
                static_cast<double>(host_radial_map_offsets[1]) +
                first_or_zero(host_radial_channel_outputs) +
                first_or_zero(host_radial_channel_indices) +
                static_cast<double>(host_angular_map_offsets[1]) +
                first_or_zero(host_angular_channel_outputs) +
                first_or_zero(host_contracted_channel_indices) +
                first_or_zero(host_angular_channel_indices) +
                static_cast<double>(host_vjp_bond_group_offsets[1]) +
                static_cast<double>(host_vjp_group_term_offsets.size()) +
                first_or_zero(host_vjp_group_harmonics) + first_or_zero(host_vjp_group_terms)};
  }

  std::array<double, 20> expected_probe_{};
  std::size_t static_bytes_ = 0;
  int vjp_harmonic_group_count_ = 0;
  int vjp_angular_term_count_ = 0;
  int vjp_maximum_group_term_count_ = 0;
  int maximum_source_serial_work_count_ = 0;
  std::string signature_hash_;
  std::string vjp_schedule_hash_;
};

}    // namespace YE3T_LAMMPS

#endif    // LMP_YE3T_KOKKOS_PLAN_H
