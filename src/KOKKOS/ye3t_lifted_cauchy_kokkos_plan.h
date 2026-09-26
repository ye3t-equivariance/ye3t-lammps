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

#ifndef LMP_YE3T_LIFTED_CAUCHY_KOKKOS_PLAN_H
#define LMP_YE3T_LIFTED_CAUCHY_KOKKOS_PLAN_H

#include "ye3t_lifted_cauchy_model.h"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace YE3T_LAMMPS {

template <class DeviceType> struct LiftedCauchyKokkosViews {
  using IntView = Kokkos::View<const int *, DeviceType>;
  using OffsetView = Kokkos::View<const std::int64_t *, DeviceType>;
  using RealView = Kokkos::View<const double *, DeviceType>;

  int species_count = 0;
  int source_row_count = 0;
  std::int64_t source_variable_count = 0;
  int maximum_factor_count = 0;
  double cutoff = 0.0;
  IntView species_row_offsets;
  OffsetView row_source_offsets;
  OffsetView polynomial_offsets;
  RealView polynomial_coefficients;
  OffsetView head_term_offsets;
  OffsetView term_factor_offsets;
  OffsetView factor_indices;
  IntView factor_exponents;
  RealView term_coefficients;
  RealView head_offsets;
};

template <class DeviceType> class LiftedCauchyKokkosPlan {
 public:
  using IntView = Kokkos::View<int *, DeviceType>;
  using OffsetView = Kokkos::View<std::int64_t *, DeviceType>;
  using RealView = Kokkos::View<double *, DeviceType>;

  void upload(const LiftedCauchyModel &model, std::size_t allocation_budget)
  {
    if (model.central_species_order.empty() || model.heads.empty() ||
        model.source_variable_count <= 0 || model.cutoff <= 0.0)
      throw std::invalid_argument("invalid lifted-Cauchy device model");
    if (model.central_species_order.size() != model.heads.size() ||
        model.central_species_order.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("invalid lifted-Cauchy device species count");
    for (const auto &group : model.source_groups) {
      if (group.angular != 1 || group.real_component_count != 3)
        throw std::invalid_argument("mixed-angular lifted-Cauchy models are not yet supported by "
                                    "pair_style ye3t/kk; use pair_style ye3t on the CPU");
    }

    species_count_ = static_cast<int>(model.central_species_order.size());
    source_variable_count_ = model.source_variable_count;
    cutoff_ = model.cutoff;

    std::vector<int> species_row_offsets;
    std::vector<std::int64_t> row_source_offsets;
    std::vector<std::int64_t> polynomial_offsets(1, 0);
    std::vector<double> polynomial_coefficients;
    species_row_offsets.reserve(static_cast<std::size_t>(species_count_) + 1);
    for (int species = 0; species < species_count_; ++species) {
      species_row_offsets.push_back(static_cast<int>(row_source_offsets.size()));
      for (const auto &group : model.source_groups) {
        if (group.neighbor_species_index != species) continue;
        for (int q = 0; q < group.source_dimension; ++q) {
          row_source_offsets.push_back(
              group.q_source_variable_offsets[static_cast<std::size_t>(q)]);
          const auto &coefficients =
              group.direct_q_polynomials[static_cast<std::size_t>(q)].coefficients;
          polynomial_coefficients.insert(polynomial_coefficients.end(), coefficients.begin(),
                                         coefficients.end());
          polynomial_offsets.push_back(static_cast<std::int64_t>(polynomial_coefficients.size()));
        }
      }
    }
    species_row_offsets.push_back(static_cast<int>(row_source_offsets.size()));
    if (row_source_offsets.empty() ||
        row_source_offsets.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("invalid lifted-Cauchy device source rows");
    source_row_count_ = static_cast<int>(row_source_offsets.size());

    std::vector<std::int64_t> head_term_offsets(1, 0);
    std::vector<std::int64_t> term_factor_offsets(1, 0);
    std::vector<std::int64_t> factor_indices;
    std::vector<int> factor_exponents;
    std::vector<double> term_coefficients;
    std::vector<double> head_offsets;
    maximum_factor_count_ = 0;
    for (int head_index = 0; head_index < species_count_; ++head_index) {
      const auto &head = model.heads[static_cast<std::size_t>(head_index)];
      if (head.central_species_index != head_index)
        throw std::invalid_argument("lifted-Cauchy device head order is not dense");
      const auto &polynomial = head.polynomial;
      if (polynomial.maximum_factor_count < 0 ||
          polynomial.maximum_factor_count > (std::numeric_limits<int>::max() / 2 - 1))
        throw std::invalid_argument("lifted-Cauchy device factor count exceeds index limits");
      maximum_factor_count_ =
          std::max(maximum_factor_count_, static_cast<int>(polynomial.maximum_factor_count));
      const std::size_t term_count = polynomial.monomial_coefficients.size();
      for (std::size_t term = 0; term < term_count; ++term) {
        const std::int64_t begin = polynomial.factor_offsets[term];
        const std::int64_t end = polynomial.factor_offsets[term + 1];
        for (std::int64_t factor = begin; factor < end; ++factor) {
          const std::int64_t index = polynomial.factor_indices[static_cast<std::size_t>(factor)];
          const std::int64_t exponent =
              polynomial.factor_exponents[static_cast<std::size_t>(factor)];
          if (index < 0 || index >= source_variable_count_ || exponent < 1 ||
              exponent > std::numeric_limits<int>::max())
            throw std::invalid_argument("invalid lifted-Cauchy device polynomial factor");
          factor_indices.push_back(index);
          factor_exponents.push_back(static_cast<int>(exponent));
        }
        term_factor_offsets.push_back(static_cast<std::int64_t>(factor_indices.size()));
        term_coefficients.push_back(polynomial.monomial_coefficients[term]);
      }
      head_term_offsets.push_back(static_cast<std::int64_t>(term_coefficients.size()));
      head_offsets.push_back(polynomial.offset);
    }

    std::size_t bytes = checked_array_bytes(species_row_offsets.size(), sizeof(int));
    for (const std::size_t count :
         {row_source_offsets.size(), polynomial_offsets.size(), head_term_offsets.size(),
          term_factor_offsets.size(), factor_indices.size()})
      bytes = checked_add(bytes, checked_array_bytes(count, sizeof(std::int64_t)));
    bytes = checked_add(bytes, checked_array_bytes(factor_exponents.size(), sizeof(int)));
    for (const std::size_t count :
         {polynomial_coefficients.size(), term_coefficients.size(), head_offsets.size()})
      bytes = checked_add(bytes, checked_array_bytes(count, sizeof(double)));
    if (bytes > allocation_budget)
      throw std::runtime_error("lifted-Cauchy device plan exceeds the allocation budget");

    copy(species_row_offsets_, species_row_offsets, "ye3t:lifted_species_row_offsets");
    copy(row_source_offsets_, row_source_offsets, "ye3t:lifted_row_source_offsets");
    copy(polynomial_offsets_, polynomial_offsets, "ye3t:lifted_polynomial_offsets");
    copy(polynomial_coefficients_, polynomial_coefficients, "ye3t:lifted_polynomial_coefficients");
    copy(head_term_offsets_, head_term_offsets, "ye3t:lifted_head_term_offsets");
    copy(term_factor_offsets_, term_factor_offsets, "ye3t:lifted_term_factor_offsets");
    copy(factor_indices_, factor_indices, "ye3t:lifted_factor_indices");
    copy(factor_exponents_, factor_exponents, "ye3t:lifted_factor_exponents");
    copy(term_coefficients_, term_coefficients, "ye3t:lifted_term_coefficients");
    copy(head_offsets_, head_offsets, "ye3t:lifted_head_offsets");
    memory_usage_ = bytes;
  }

  LiftedCauchyKokkosViews<DeviceType> views() const
  {
    return {species_count_,
            source_row_count_,
            source_variable_count_,
            maximum_factor_count_,
            cutoff_,
            species_row_offsets_,
            row_source_offsets_,
            polynomial_offsets_,
            polynomial_coefficients_,
            head_term_offsets_,
            term_factor_offsets_,
            factor_indices_,
            factor_exponents_,
            term_coefficients_,
            head_offsets_};
  }

  int species_count() const { return species_count_; }
  int source_row_count() const { return source_row_count_; }
  std::int64_t source_variable_count() const { return source_variable_count_; }
  int maximum_factor_count() const { return maximum_factor_count_; }
  int workspace_values_per_center() const { return 2 * (maximum_factor_count_ + 1); }
  double cutoff() const { return cutoff_; }
  std::size_t memory_usage() const { return memory_usage_; }

 private:
  static std::size_t checked_add(std::size_t first, std::size_t second)
  {
    if (second > std::numeric_limits<std::size_t>::max() - first)
      throw std::overflow_error("lifted-Cauchy device plan byte-count overflow");
    return first + second;
  }

  static std::size_t checked_array_bytes(std::size_t count, std::size_t width)
  {
    if (count != 0 && width > std::numeric_limits<std::size_t>::max() / count)
      throw std::overflow_error("lifted-Cauchy device plan byte-count overflow");
    return count * width;
  }

  template <class View, class Value>
  static void copy(View &destination, const std::vector<Value> &source, const char *name)
  {
    destination = View(std::string(name), source.size());
    auto host = Kokkos::create_mirror_view(destination);
    for (std::size_t index = 0; index < source.size(); ++index) host(index) = source[index];
    Kokkos::deep_copy(destination, host);
  }

  IntView species_row_offsets_;
  OffsetView row_source_offsets_;
  OffsetView polynomial_offsets_;
  RealView polynomial_coefficients_;
  OffsetView head_term_offsets_;
  OffsetView term_factor_offsets_;
  OffsetView factor_indices_;
  IntView factor_exponents_;
  RealView term_coefficients_;
  RealView head_offsets_;
  int species_count_ = 0;
  int source_row_count_ = 0;
  std::int64_t source_variable_count_ = 0;
  int maximum_factor_count_ = 0;
  double cutoff_ = 0.0;
  std::size_t memory_usage_ = 0;
};

}    // namespace YE3T_LAMMPS

#endif    // LMP_YE3T_LIFTED_CAUCHY_KOKKOS_PLAN_H
