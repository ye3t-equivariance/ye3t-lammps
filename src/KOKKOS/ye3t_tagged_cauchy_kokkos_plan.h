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

#ifndef LMP_YE3T_TAGGED_CAUCHY_KOKKOS_PLAN_H
#define LMP_YE3T_TAGGED_CAUCHY_KOKKOS_PLAN_H

#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

#include "ye3t_gpu_tagged_source.h"
#include "ye3t_shifted_jacobi.h"
#include "ye3t_tagged_cauchy_model.h"
#include "ye3t_tagged_cauchy_readout_plan.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace YE3T_LAMMPS {

template <class DeviceType> struct TaggedCauchyKokkosViews {
  using IntView = Kokkos::View<const int *, DeviceType>;
  using OffsetView = Kokkos::View<const std::int64_t *, DeviceType>;
  using RealView = Kokkos::View<const double *, DeviceType>;
  using ComplexView = Kokkos::View<const Kokkos::complex<double> *, DeviceType>;

  int species_count = 0;
  int feature_count = 0;
  int tag_count = 0;
  int physical_image_v3 = 0;
  int stable_shifted_jacobi = 0;
  double cutoff = 0.0;
  double radial_cutoff_width = 0.0;
  double radial_lambda = 0.0;
  int radial_count = 0;
  int channel_count = 0;
  int total_component_count = 0;
  int density_key_count = 0;
  int moment_key_count = 0;
  int term_count = 0;
  int adjoint_term_count = 0;
  int source_group_count = 0;
  IntView source_group_offsets;
  IntView source_group_channels;

  IntView channel_neighbor_species;
  IntView channel_radial_column;
  IntView channel_l;
  IntView channel_component_offset;
  IntView channel_inverse_offset;
  ComplexView inverse_matrix;
  RealView channel_source_normalization;
  RealView channel_angular_scale;

  RealView offsets;    // [species]

  IntView density_flat_index;    // [density_key_count]

  OffsetView moment_factor_offsets;    // [moment_key_count + 1]
  IntView moment_factor_flat_index;    // [moment_factor_offsets(moment_key_count)]

  OffsetView species_term_offsets;    // [species_count + 1]
  RealView term_coefficient;
  IntView term_p;
  OffsetView term_density_offsets;    // [term_count + 1]
  IntView term_density_factor_index;
  OffsetView term_moment_offsets;    // [term_count + 1]
  IntView term_moment_factor_index;

  IntView adjoint_source_index;
  OffsetView adjoint_remaining_offsets;
  IntView adjoint_remaining_source_index;
  RealView folded_adjoint_coefficient;

  RealView legendre_derivative;    // [channel_count * LEGENDRE_M_STRIDE * LEGENDRE_STRIDE]
  RealView normalization;          // [channel_count * LEGENDRE_M_STRIDE]
};

// Device instantiation of the shared, host-testable arithmetic, not a host fallback.
struct TaggedKokkosMath {
  using Complex = Kokkos::complex<double>;
  KOKKOS_INLINE_FUNCTION static double sqrt(double x) { return Kokkos::sqrt(x); }
  KOKKOS_INLINE_FUNCTION static double cos(double x) { return Kokkos::cos(x); }
  KOKKOS_INLINE_FUNCTION static double sin(double x) { return Kokkos::sin(x); }
  KOKKOS_INLINE_FUNCTION static double exp(double x) { return Kokkos::exp(x); }
};
template <class DeviceType>
KOKKOS_INLINE_FUNCTION void
tagged_kokkos_edge_components(const TaggedCauchyKokkosViews<DeviceType> &plan, int species,
                              double dx, double dy, double dz, bool need_gradient, double *phi_real,
                              double *phi_grad_x, double *phi_grad_y, double *phi_grad_z)
{
  if (need_gradient)
    gpu_tagged_edge_components<true, TaggedKokkosMath>(plan, species, dx, dy, dz, phi_real,
                                                       phi_grad_x, phi_grad_y, phi_grad_z);
  else
    gpu_tagged_edge_components<false, TaggedKokkosMath>(plan, species, dx, dy, dz, phi_real,
                                                        nullptr, nullptr, nullptr);
}

template <class DeviceType> class TaggedCauchyKokkosPlan {
 public:
  using IntView = Kokkos::View<int *, DeviceType>;
  using OffsetView = Kokkos::View<std::int64_t *, DeviceType>;
  using RealView = Kokkos::View<double *, DeviceType>;
  using ComplexView = Kokkos::View<Kokkos::complex<double> *, DeviceType>;

  void upload(const TaggedCauchyModel &model, std::size_t allocation_budget)
  {
    if (model.deployment_kind == TaggedCauchyDeploymentKind::PhysicalImageV4)
      throw std::runtime_error("Tagged V4 pair-specific sources and references require the CPU runtime");
    if (model.species_order.empty() || model.channels.empty() || model.feature_count <= 0 ||
        model.cutoff <= 0.0)
      throw std::invalid_argument("invalid tagged-Cauchy device model");
    if (model.total_component_count > TAGGED_KOKKOS_MAX_COMPONENTS)
      throw std::invalid_argument("tagged-Cauchy device plan exceeds TAGGED_KOKKOS_MAX_COMPONENTS");
    const bool physical_image_v3 =
        model.deployment_kind == TaggedCauchyDeploymentKind::PhysicalImageV3;
    const bool stable_shifted_jacobi =
        model.source_realization == TaggedCauchySourceRealization::ShiftedJacobiThreeTermV1;
    if (physical_image_v3 && !stable_shifted_jacobi)
      throw std::invalid_argument("tagged-Cauchy V3 Kokkos reference requires source-plan V2");
    if (!physical_image_v3 && model.radial_count > TAGGED_KOKKOS_MAX_RADIAL_COUNT)
      throw std::invalid_argument(
          "tagged-Cauchy device plan exceeds TAGGED_KOKKOS_MAX_RADIAL_COUNT");
    for (const auto &channel : model.channels)
      if (channel.l > TAGGED_KOKKOS_MAX_L)
        throw std::invalid_argument("tagged-Cauchy device plan exceeds TAGGED_KOKKOS_MAX_L");
    for (const auto &term : model.terms)
      if (static_cast<int>(term.density_factor_indices.size()) > TAGGED_KOKKOS_MAX_TERM_FACTORS ||
          static_cast<int>(term.moment_indices.size()) > TAGGED_KOKKOS_MAX_TERM_FACTORS)
        throw std::invalid_argument(
            "tagged-Cauchy device plan exceeds TAGGED_KOKKOS_MAX_TERM_FACTORS "
            "(term density/moment factor count)");
    for (const auto &key : model.real_moment_keys)
      if (static_cast<int>(key.size()) > TAGGED_KOKKOS_MAX_TERM_FACTORS)
        throw std::invalid_argument(
            "tagged-Cauchy device plan exceeds TAGGED_KOKKOS_MAX_TERM_FACTORS "
            "(moment key factor count)");

    const auto source_groups = make_gpu_tagged_source_groups(model);
    source_group_count_ = static_cast<int>(source_groups.offsets.size()) - 1;

    species_count_ = static_cast<int>(model.species_order.size());
    feature_count_ = model.feature_count;
    tag_count_ = model.tag_count;
    physical_image_v3_ = physical_image_v3 ? 1 : 0;
    stable_shifted_jacobi_ = stable_shifted_jacobi ? 1 : 0;
    cutoff_ = model.cutoff;
    radial_cutoff_width_ = model.radial_cutoff_width;
    radial_lambda_ = model.radial_lambda;
    radial_count_ = model.radial_count;
    channel_count_ = static_cast<int>(model.channels.size());
    total_component_count_ = model.total_component_count;
    density_key_count_ = static_cast<int>(model.real_density_keys.size());
    moment_key_count_ = static_cast<int>(model.real_moment_keys.size());
    raw_term_count_ = static_cast<int>(model.terms.size());
    adjoint_term_count_ = static_cast<int>(model.adjoint_terms.size());

    std::vector<int> channel_neighbor_species(channel_count_);
    std::vector<int> channel_radial_column(channel_count_);
    std::vector<int> channel_l(channel_count_);
    std::vector<int> channel_component_offset(channel_count_);
    std::vector<int> channel_inverse_offset(channel_count_);
    std::vector<double> channel_source_normalization(channel_count_, 1.0);
    std::vector<double> channel_angular_scale(channel_count_, 1.0);
    std::vector<Kokkos::complex<double>> inverse_matrix;
    std::vector<double> legendre_derivative(static_cast<std::size_t>(channel_count_) *
                                                TAGGED_KOKKOS_LEGENDRE_M_STRIDE *
                                                TAGGED_KOKKOS_LEGENDRE_STRIDE,
                                            0.0);
    std::vector<double> normalization(
        static_cast<std::size_t>(channel_count_) * TAGGED_KOKKOS_LEGENDRE_M_STRIDE, 0.0);

    for (int c = 0; c < channel_count_; ++c) {
      const auto &channel = model.channels[static_cast<std::size_t>(c)];
      channel_neighbor_species[static_cast<std::size_t>(c)] = channel.neighbor_species_index;
      channel_radial_column[static_cast<std::size_t>(c)] = channel.radial_channel;
      channel_l[static_cast<std::size_t>(c)] = channel.l;
      channel_component_offset[static_cast<std::size_t>(c)] = channel.component_offset;
      channel_source_normalization[static_cast<std::size_t>(c)] = channel.normalization;
      channel_angular_scale[static_cast<std::size_t>(c)] = channel.angular_scale;
      const auto &form = model.real_forms[static_cast<std::size_t>(channel.real_form_index)];
      channel_inverse_offset[static_cast<std::size_t>(c)] = static_cast<int>(inverse_matrix.size());
      for (const auto &value : form.inverse)
        inverse_matrix.emplace_back(value.real(), value.imag());

      build_gpu_tagged_legendre_table(
          channel.l,
          legendre_derivative.data() +
              static_cast<std::size_t>(c) * TAGGED_KOKKOS_LEGENDRE_M_STRIDE *
                  TAGGED_KOKKOS_LEGENDRE_STRIDE,
          normalization.data() + static_cast<std::size_t>(c) * TAGGED_KOKKOS_LEGENDRE_M_STRIDE);
    }

    std::vector<double> offsets = model.offsets;

    std::vector<int> density_flat_index = model.real_density_flat_index;

    std::vector<std::int64_t> moment_factor_offsets(1, 0);
    std::vector<int> moment_factor_flat_index;
    for (const auto &flat : model.real_moment_flat_indices) {
      moment_factor_flat_index.insert(moment_factor_flat_index.end(), flat.begin(), flat.end());
      moment_factor_offsets.push_back(static_cast<std::int64_t>(moment_factor_flat_index.size()));
    }

    const TaggedCauchyReadoutPlan readout = compile_tagged_cauchy_readout(model);
    term_count_ = static_cast<int>(readout.term_coefficient.size());

    std::vector<int> adjoint_source_index(static_cast<std::size_t>(adjoint_term_count_));
    std::vector<std::int64_t> adjoint_remaining_offsets(1, 0);
    std::vector<int> adjoint_remaining_source_index;
    std::vector<double> folded_adjoint_coefficient(static_cast<std::size_t>(species_count_) *
                                                   static_cast<std::size_t>(adjoint_term_count_));
    for (int t = 0; t < adjoint_term_count_; ++t) {
      const auto &term = model.adjoint_terms[static_cast<std::size_t>(t)];
      adjoint_source_index[static_cast<std::size_t>(t)] = term.source_index;
      adjoint_remaining_source_index.insert(adjoint_remaining_source_index.end(),
                                            term.remaining_source_indices.begin(),
                                            term.remaining_source_indices.end());
      adjoint_remaining_offsets.push_back(
          static_cast<std::int64_t>(adjoint_remaining_source_index.size()));
      for (int species = 0; species < species_count_; ++species)
        folded_adjoint_coefficient[static_cast<std::size_t>(species) * adjoint_term_count_ + t] =
            model.beta[static_cast<std::size_t>(species)]
                      [static_cast<std::size_t>(term.feature_index)] *
            term.coefficient;
    }

    std::size_t bytes =
        checked_array_bytes(static_cast<std::size_t>(channel_count_) * 5, sizeof(int));
    bytes = checked_add(
        bytes, checked_array_bytes(inverse_matrix.size(), sizeof(Kokkos::complex<double>)));
    bytes = checked_add(bytes,
                        checked_array_bytes(channel_source_normalization.size(), sizeof(double)));
    bytes = checked_add(bytes, checked_array_bytes(channel_angular_scale.size(), sizeof(double)));
    bytes = checked_add(bytes, checked_array_bytes(offsets.size(), sizeof(double)));
    bytes = checked_add(bytes, checked_array_bytes(density_flat_index.size(), sizeof(int)));
    bytes =
        checked_add(bytes, checked_array_bytes(moment_factor_offsets.size(), sizeof(std::int64_t)));
    bytes = checked_add(bytes, checked_array_bytes(moment_factor_flat_index.size(), sizeof(int)));
    bytes = checked_add(
        bytes, checked_array_bytes(readout.species_term_offsets.size(), sizeof(std::int64_t)));
    bytes =
        checked_add(bytes, checked_array_bytes(readout.term_coefficient.size(), sizeof(double)));
    bytes = checked_add(bytes, checked_array_bytes(readout.term_p.size(), sizeof(int)));
    bytes = checked_add(
        bytes, checked_array_bytes(readout.term_density_offsets.size(), sizeof(std::int64_t)));
    bytes = checked_add(bytes, checked_array_bytes(readout.term_density_flat.size(), sizeof(int)));
    bytes = checked_add(
        bytes, checked_array_bytes(readout.term_moment_offsets.size(), sizeof(std::int64_t)));
    bytes = checked_add(bytes, checked_array_bytes(readout.term_moment_flat.size(), sizeof(int)));
    bytes = checked_add(bytes, checked_array_bytes(adjoint_source_index.size(), sizeof(int)));
    bytes = checked_add(
        bytes, checked_array_bytes(adjoint_remaining_offsets.size(), sizeof(std::int64_t)));
    bytes =
        checked_add(bytes, checked_array_bytes(adjoint_remaining_source_index.size(), sizeof(int)));
    bytes =
        checked_add(bytes, checked_array_bytes(folded_adjoint_coefficient.size(), sizeof(double)));
    bytes = checked_add(bytes, checked_array_bytes(legendre_derivative.size(), sizeof(double)));
    bytes = checked_add(bytes, checked_array_bytes(normalization.size(), sizeof(double)));
    bytes = checked_add(bytes, checked_array_bytes(source_groups.offsets.size(), sizeof(int)));
    bytes = checked_add(bytes, checked_array_bytes(source_groups.channels.size(), sizeof(int)));
    if (bytes > allocation_budget)
      throw std::runtime_error("tagged-Cauchy device plan exceeds the allocation budget");

    copy(channel_neighbor_species_, channel_neighbor_species, "ye3t:tagged_channel_species");
    copy(channel_radial_column_, channel_radial_column, "ye3t:tagged_channel_radial_column");
    copy(channel_l_, channel_l, "ye3t:tagged_channel_l");
    copy(channel_component_offset_, channel_component_offset,
         "ye3t:tagged_channel_component_offset");
    copy(channel_inverse_offset_, channel_inverse_offset, "ye3t:tagged_channel_inverse_offset");
    copy(inverse_matrix_, inverse_matrix, "ye3t:tagged_inverse_matrix");
    copy(channel_source_normalization_, channel_source_normalization,
         "ye3t:tagged_source_normalization");
    copy(channel_angular_scale_, channel_angular_scale, "ye3t:tagged_angular_scale");
    copy(offsets_, offsets, "ye3t:tagged_offsets");
    copy(density_flat_index_, density_flat_index, "ye3t:tagged_density_flat_index");
    copy(moment_factor_offsets_, moment_factor_offsets, "ye3t:tagged_moment_factor_offsets");
    copy(moment_factor_flat_index_, moment_factor_flat_index,
         "ye3t:tagged_moment_factor_flat_index");
    copy(species_term_offsets_, readout.species_term_offsets, "ye3t:tagged_species_term_offsets");
    copy(term_coefficient_, readout.term_coefficient, "ye3t:tagged_term_coefficient");
    copy(term_p_, readout.term_p, "ye3t:tagged_term_p");
    copy(term_density_offsets_, readout.term_density_offsets, "ye3t:tagged_term_density_offsets");
    copy(term_density_factor_index_, readout.term_density_flat,
         "ye3t:tagged_term_density_factor_index");
    copy(term_moment_offsets_, readout.term_moment_offsets, "ye3t:tagged_term_moment_offsets");
    copy(term_moment_factor_index_, readout.term_moment_flat,
         "ye3t:tagged_term_moment_factor_index");
    copy(adjoint_source_index_, adjoint_source_index, "ye3t:tagged_adjoint_source_index");
    copy(adjoint_remaining_offsets_, adjoint_remaining_offsets,
         "ye3t:tagged_adjoint_remaining_offsets");
    copy(adjoint_remaining_source_index_, adjoint_remaining_source_index,
         "ye3t:tagged_adjoint_remaining_source_index");
    copy(folded_adjoint_coefficient_, folded_adjoint_coefficient,
         "ye3t:tagged_folded_adjoint_coefficient");
    copy(legendre_derivative_, legendre_derivative, "ye3t:tagged_legendre_derivative");
    copy(normalization_, normalization, "ye3t:tagged_normalization");
    copy(source_group_offsets_, source_groups.offsets, "ye3t:tagged_source_group_offsets");
    copy(source_group_channels_, source_groups.channels, "ye3t:tagged_source_group_channels");
    // Validate all new schedule entries once, before any device kernel uses them.
    for (const auto *view : {&source_group_offsets_, &source_group_channels_}) {
      const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), *view);
      const auto &expected = view == &source_group_offsets_ ? source_groups.offsets
                                                            : source_groups.channels;
      for (std::size_t i = 0; i < expected.size(); ++i)
        if (host(i) != expected[i])
          throw std::runtime_error("tagged source schedule upload mismatch");
    }
    memory_usage_ = bytes;
  }

  TaggedCauchyKokkosViews<DeviceType> views() const
  {
    TaggedCauchyKokkosViews<DeviceType> result;
    result.species_count = species_count_;
    result.feature_count = feature_count_;
    result.tag_count = tag_count_;
    result.physical_image_v3 = physical_image_v3_;
    result.stable_shifted_jacobi = stable_shifted_jacobi_;
    result.cutoff = cutoff_;
    result.radial_cutoff_width = radial_cutoff_width_;
    result.radial_lambda = radial_lambda_;
    result.radial_count = radial_count_;
    result.channel_count = channel_count_;
    result.total_component_count = total_component_count_;
    result.density_key_count = density_key_count_;
    result.moment_key_count = moment_key_count_;
    result.term_count = term_count_;
    result.adjoint_term_count = adjoint_term_count_;
    result.source_group_count = source_group_count_;
    result.source_group_offsets = source_group_offsets_;
    result.source_group_channels = source_group_channels_;
    result.channel_neighbor_species = channel_neighbor_species_;
    result.channel_radial_column = channel_radial_column_;
    result.channel_l = channel_l_;
    result.channel_component_offset = channel_component_offset_;
    result.channel_inverse_offset = channel_inverse_offset_;
    result.inverse_matrix = inverse_matrix_;
    result.channel_source_normalization = channel_source_normalization_;
    result.channel_angular_scale = channel_angular_scale_;
    result.offsets = offsets_;
    result.density_flat_index = density_flat_index_;
    result.moment_factor_offsets = moment_factor_offsets_;
    result.moment_factor_flat_index = moment_factor_flat_index_;
    result.species_term_offsets = species_term_offsets_;
    result.term_coefficient = term_coefficient_;
    result.term_p = term_p_;
    result.term_density_offsets = term_density_offsets_;
    result.term_density_factor_index = term_density_factor_index_;
    result.term_moment_offsets = term_moment_offsets_;
    result.term_moment_factor_index = term_moment_factor_index_;
    result.adjoint_source_index = adjoint_source_index_;
    result.adjoint_remaining_offsets = adjoint_remaining_offsets_;
    result.adjoint_remaining_source_index = adjoint_remaining_source_index_;
    result.folded_adjoint_coefficient = folded_adjoint_coefficient_;
    result.legendre_derivative = legendre_derivative_;
    result.normalization = normalization_;
    return result;
  }

  int species_count() const { return species_count_; }
  int feature_count() const { return feature_count_; }
  int density_key_count() const { return density_key_count_; }
  int moment_key_count() const { return moment_key_count_; }
  int raw_term_count() const { return raw_term_count_; }
  int term_count() const { return term_count_; }
  int total_component_count() const { return total_component_count_; }
  double cutoff() const { return cutoff_; }
  std::size_t memory_usage() const { return memory_usage_; }

 private:
  static std::size_t checked_add(std::size_t first, std::size_t second)
  {
    if (second > std::numeric_limits<std::size_t>::max() - first)
      throw std::overflow_error("tagged-Cauchy device plan byte-count overflow");
    return first + second;
  }

  static std::size_t checked_array_bytes(std::size_t count, std::size_t width)
  {
    if (count != 0 && width > std::numeric_limits<std::size_t>::max() / count)
      throw std::overflow_error("tagged-Cauchy device plan byte-count overflow");
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

  IntView source_group_offsets_;
  IntView source_group_channels_;
  int source_group_count_ = 0;
  IntView channel_neighbor_species_;
  IntView channel_radial_column_;
  IntView channel_l_;
  IntView channel_component_offset_;
  IntView channel_inverse_offset_;
  ComplexView inverse_matrix_;
  RealView channel_source_normalization_;
  RealView channel_angular_scale_;
  RealView offsets_;
  IntView density_flat_index_;
  OffsetView moment_factor_offsets_;
  IntView moment_factor_flat_index_;
  OffsetView species_term_offsets_;
  RealView term_coefficient_;
  IntView term_p_;
  OffsetView term_density_offsets_;
  IntView term_density_factor_index_;
  OffsetView term_moment_offsets_;
  IntView term_moment_factor_index_;
  IntView adjoint_source_index_;
  OffsetView adjoint_remaining_offsets_;
  IntView adjoint_remaining_source_index_;
  RealView folded_adjoint_coefficient_;
  RealView legendre_derivative_;
  RealView normalization_;

  int species_count_ = 0;
  int feature_count_ = 0;
  int tag_count_ = 0;
  int physical_image_v3_ = 0;
  int stable_shifted_jacobi_ = 0;
  double cutoff_ = 0.0;
  double radial_cutoff_width_ = 0.0;
  double radial_lambda_ = 0.0;
  int radial_count_ = 0;
  int channel_count_ = 0;
  int total_component_count_ = 0;
  int density_key_count_ = 0;
  int moment_key_count_ = 0;
  int raw_term_count_ = 0;
  int term_count_ = 0;
  int adjoint_term_count_ = 0;
  std::size_t memory_usage_ = 0;
};

}    // namespace YE3T_LAMMPS

#endif    // LMP_YE3T_TAGGED_CAUCHY_KOKKOS_PLAN_H
