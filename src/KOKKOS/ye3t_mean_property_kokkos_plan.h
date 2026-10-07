/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   Contributing author: James M. Goff (Sandia National Laboratories)
------------------------------------------------------------------------- */

#ifndef LMP_YE3T_MEAN_PROPERTY_KOKKOS_PLAN_H
#define LMP_YE3T_MEAN_PROPERTY_KOKKOS_PLAN_H

#include "ye3t_mean_property_cpu.h"
#include "ye3t_gpu_tagged_source.h"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace YE3T_LAMMPS {

template <class DeviceType> struct MeanPropertyKokkosViews {
  using Integers = Kokkos::View<const int *, DeviceType>;
  using Reals = Kokkos::View<const double *, DeviceType>;

  int species_count = 0;
  int component_count = 0;
  int width = 0;
  int schedule_count = 0;
  int feature_count = 0;
  double cutoff = 0.0;
  Reals pair_cutoffs;
  Integers channel_species, channel_l, channel_offset, jacobi_offsets;
  Reals channel_normalization, jacobi_coefficients;
  Reals legendre_derivative, harmonic_normalization;
  Integers schedule_support, schedule_input_offsets, schedule_term_offsets;
  Integers schedule_coefficient_offsets, input_component, input_role;
  Integers term_factor_offsets, term_input, term_exponent;
  Integers coefficient_term, coefficient_output;
  Reals coefficient_value;
  Integers feature_schedule, feature_start;
  Reals readout;
};

template <class DeviceType> class MeanPropertyKokkosPlan {
 public:
  using IntView = Kokkos::View<int *, DeviceType>;
  using RealView = Kokkos::View<double *, DeviceType>;

  void upload(const YE3TMeanPropertyCPU &model)
  {
    if (!model.tagged_model())
      throw std::runtime_error("YE3T density property device plan is not implemented");
    if (model.component_count() > TAGGED_KOKKOS_MAX_COMPONENTS)
      throw std::runtime_error("YE3T property device component count exceeds 128");
    species_count_ = static_cast<int>(model.species_order().size());
    component_count_ = model.component_count();
    width_ = model.width();
    schedule_count_ = static_cast<int>(model.schedules().size());
    feature_count_ = static_cast<int>(model.features().size());
    cutoff_ = model.cutoff();
    copy(pair_cutoffs_, model.pair_cutoffs(), "ye3t:property_pair_cutoffs");

    std::vector<int> species, angular, offsets, jacobi_offsets{0};
    std::vector<double> scale, jacobi, legendre, harmonic;
    for (const auto &channel : model.channels()) {
      if (channel.l > TAGGED_KOKKOS_MAX_L)
        throw std::runtime_error("YE3T property device angular degree exceeds tagged Kokkos limit");
      species.push_back(channel.neighbor_species);
      angular.push_back(channel.l);
      offsets.push_back(channel.offset);
      scale.push_back(channel.normalization);
      jacobi.insert(jacobi.end(), channel.jacobi_coefficients.begin(),
                    channel.jacobi_coefficients.end());
      jacobi_offsets.push_back(static_cast<int>(jacobi.size()));
      const std::size_t begin = legendre.size();
      legendre.resize(begin + TAGGED_KOKKOS_LEGENDRE_M_STRIDE * TAGGED_KOKKOS_LEGENDRE_STRIDE);
      harmonic.resize(harmonic.size() + TAGGED_KOKKOS_LEGENDRE_M_STRIDE);
      build_gpu_tagged_legendre_table(
          channel.l, legendre.data() + begin,
          harmonic.data() + harmonic.size() - TAGGED_KOKKOS_LEGENDRE_M_STRIDE);
    }
    copy(channel_species_, species, "ye3t:property_channel_species");
    copy(channel_l_, angular, "ye3t:property_channel_l");
    copy(channel_offset_, offsets, "ye3t:property_channel_offset");
    copy(channel_normalization_, scale, "ye3t:property_channel_norm");
    copy(jacobi_offsets_, jacobi_offsets, "ye3t:property_jacobi_offsets");
    copy(jacobi_coefficients_, jacobi, "ye3t:property_jacobi_coefficients");
    copy(legendre_derivative_, legendre, "ye3t:property_legendre");
    copy(harmonic_normalization_, harmonic, "ye3t:property_harmonic_norm");

    std::vector<int> support, input_offsets{0}, term_offsets{0}, coefficient_offsets{0};
    std::vector<int> input_component, input_role, factor_offsets{0};
    std::vector<int> factor_input, factor_exponent, coefficient_term, coefficient_output;
    std::vector<double> coefficient_value;
    for (const auto &schedule : model.schedules()) {
      support.push_back(schedule.support_tag_count);
      for (const auto &input : schedule.inputs) {
        input_component.push_back(input.component);
        input_role.push_back(input.role);
      }
      input_offsets.push_back(static_cast<int>(input_component.size()));
      for (std::size_t term = 0; term + 1 < schedule.term_offsets.size(); ++term) {
        for (int factor = schedule.term_offsets[term];
             factor < schedule.term_offsets[term + 1]; ++factor) {
          factor_input.push_back(input_offsets[input_offsets.size() - 2] +
                                 schedule.term_components[factor]);
          factor_exponent.push_back(schedule.term_exponents[factor]);
        }
        factor_offsets.push_back(static_cast<int>(factor_input.size()));
      }
      for (std::size_t coefficient = 0; coefficient < schedule.coefficient_values.size();
           ++coefficient) {
        coefficient_term.push_back(term_offsets.back() + schedule.coefficient_terms[coefficient]);
        coefficient_output.push_back(schedule.coefficient_outputs[coefficient]);
        coefficient_value.push_back(schedule.coefficient_values[coefficient]);
      }
      term_offsets.push_back(static_cast<int>(factor_offsets.size()) - 1);
      coefficient_offsets.push_back(static_cast<int>(coefficient_value.size()));
    }
    copy(schedule_support_, support, "ye3t:property_schedule_support");
    copy(schedule_input_offsets_, input_offsets, "ye3t:property_schedule_inputs");
    copy(schedule_term_offsets_, term_offsets, "ye3t:property_schedule_terms");
    copy(schedule_coefficient_offsets_, coefficient_offsets,
         "ye3t:property_schedule_coefficients");
    copy(input_component_, input_component, "ye3t:property_input_component");
    copy(input_role_, input_role, "ye3t:property_input_role");
    copy(term_factor_offsets_, factor_offsets, "ye3t:property_factor_offsets");
    copy(term_input_, factor_input, "ye3t:property_factor_inputs");
    copy(term_exponent_, factor_exponent, "ye3t:property_factor_exponents");
    copy(coefficient_term_, coefficient_term, "ye3t:property_coefficient_term");
    copy(coefficient_output_, coefficient_output, "ye3t:property_coefficient_output");
    copy(coefficient_value_, coefficient_value, "ye3t:property_coefficient_value");
    std::vector<int> feature_schedule, feature_start;
    for (const auto &feature : model.features()) {
      feature_schedule.push_back(feature.schedule_index);
      feature_start.push_back(feature.component_start);
    }
    copy(feature_schedule_, feature_schedule, "ye3t:property_feature_schedule");
    copy(feature_start_, feature_start, "ye3t:property_feature_start");
    copy(readout_, model.readout(), "ye3t:property_readout");
  }

  MeanPropertyKokkosViews<DeviceType> views() const
  {
    MeanPropertyKokkosViews<DeviceType> result;
    result.species_count = species_count_;
    result.component_count = component_count_;
    result.width = width_;
    result.schedule_count = schedule_count_;
    result.feature_count = feature_count_;
    result.cutoff = cutoff_;
    result.pair_cutoffs = pair_cutoffs_;
    result.channel_species = channel_species_;
    result.channel_l = channel_l_;
    result.channel_offset = channel_offset_;
    result.channel_normalization = channel_normalization_;
    result.jacobi_offsets = jacobi_offsets_;
    result.jacobi_coefficients = jacobi_coefficients_;
    result.legendre_derivative = legendre_derivative_;
    result.harmonic_normalization = harmonic_normalization_;
    result.schedule_support = schedule_support_;
    result.schedule_input_offsets = schedule_input_offsets_;
    result.schedule_term_offsets = schedule_term_offsets_;
    result.schedule_coefficient_offsets = schedule_coefficient_offsets_;
    result.input_component = input_component_;
    result.input_role = input_role_;
    result.term_factor_offsets = term_factor_offsets_;
    result.term_input = term_input_;
    result.term_exponent = term_exponent_;
    result.coefficient_term = coefficient_term_;
    result.coefficient_output = coefficient_output_;
    result.coefficient_value = coefficient_value_;
    result.feature_schedule = feature_schedule_;
    result.feature_start = feature_start_;
    result.readout = readout_;
    return result;
  }

 private:
  template <class View, class Value>
  static void copy(View &destination, const std::vector<Value> &source, const char *name)
  {
    destination = View(std::string(name), source.size());
    auto host = Kokkos::create_mirror_view(destination);
    for (std::size_t index = 0; index < source.size(); ++index) host(index) = source[index];
    Kokkos::deep_copy(destination, host);
  }

  int species_count_ = 0, component_count_ = 0, width_ = 0;
  int schedule_count_ = 0, feature_count_ = 0;
  double cutoff_ = 0.0;
  RealView pair_cutoffs_, channel_normalization_, jacobi_coefficients_;
  RealView legendre_derivative_, harmonic_normalization_, coefficient_value_, readout_;
  IntView channel_species_, channel_l_, channel_offset_, jacobi_offsets_;
  IntView schedule_support_, schedule_input_offsets_, schedule_term_offsets_;
  IntView schedule_coefficient_offsets_, input_component_, input_role_;
  IntView term_factor_offsets_, term_input_, term_exponent_;
  IntView coefficient_term_, coefficient_output_, feature_schedule_, feature_start_;
};

}  // namespace YE3T_LAMMPS

#endif
