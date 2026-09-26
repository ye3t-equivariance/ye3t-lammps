#pragma once
// clang-format off
#include "ye3t_gpu_tagged_source.h"
#include <complex>
#include <cmath>
// clang-format on
using namespace YE3T_LAMMPS;
using Complex = std::complex<double>;
struct HostMath {
  using Complex = std::complex<double>;
  static double sqrt(double x) { return std::sqrt(x); }
  static double cos(double x) { return std::cos(x); }
  static double sin(double x) { return std::sin(x); }
  static double exp(double x) { return std::exp(x); }
};
template <class T> struct Array {
  std::vector<T> values;
  const T &operator()(int i) const { return values.at(i); }
  const T *data() const { return values.data(); }
};
struct HostSource {
  int physical_image_v3, radial_count, channel_count, total_component_count, source_group_count;
  double cutoff, radial_cutoff_width, radial_lambda;
  Array<int> channel_neighbor_species, channel_l, channel_radial_column;
  Array<int> channel_component_offset, channel_inverse_offset;
  Array<int> source_group_offsets, source_group_channels;
  Array<double> channel_source_normalization, channel_angular_scale;
  Array<double> legendre_derivative, normalization;
  Array<Complex> inverse_matrix;
  explicit HostSource(const TaggedCauchyModel &model)
  {
    physical_image_v3 = model.deployment_kind == TaggedCauchyDeploymentKind::PhysicalImageV3;
    radial_count = model.radial_count;
    channel_count = static_cast<int>(model.channels.size());
    total_component_count = model.total_component_count;
    cutoff = model.cutoff;
    radial_cutoff_width = model.radial_cutoff_width;
    radial_lambda = model.radial_lambda;
    const auto groups = make_gpu_tagged_source_groups(model);
    source_group_offsets.values = groups.offsets;
    source_group_channels.values = groups.channels;
    source_group_count = static_cast<int>(groups.offsets.size()) - 1;
    legendre_derivative.values.resize(
        channel_count * TAGGED_KOKKOS_LEGENDRE_M_STRIDE * TAGGED_KOKKOS_LEGENDRE_STRIDE, 0.0);
    normalization.values.resize(channel_count * TAGGED_KOKKOS_LEGENDRE_M_STRIDE, 0.0);
    for (int i = 0; i < channel_count; ++i) {
      const auto &c = model.channels[i];
      channel_neighbor_species.values.push_back(c.neighbor_species_index);
      channel_l.values.push_back(c.l);
      channel_radial_column.values.push_back(c.radial_channel);
      channel_component_offset.values.push_back(c.component_offset);
      channel_inverse_offset.values.push_back(static_cast<int>(inverse_matrix.values.size()));
      const auto &inverse = model.real_forms[c.real_form_index].inverse;
      inverse_matrix.values.insert(inverse_matrix.values.end(), inverse.begin(), inverse.end());
      channel_source_normalization.values.push_back(c.normalization);
      channel_angular_scale.values.push_back(c.angular_scale);
      build_gpu_tagged_legendre_table(
          c.l,
          legendre_derivative.values.data() +
              i * TAGGED_KOKKOS_LEGENDRE_M_STRIDE * TAGGED_KOKKOS_LEGENDRE_STRIDE,
          normalization.values.data() + i * TAGGED_KOKKOS_LEGENDRE_M_STRIDE);
    }
  }
};
inline TaggedCauchyModel make_model(int l, bool v3)
{
  TaggedCauchyModel model;
  model.species_order = {"X", "Y", "no_channels"};
  model.feature_count = 1;
  model.offsets = {0.0, 0.0, 0.0};
  model.beta = {{1.0}, {1.0}, {1.0}};
  model.cutoff = model.radial_rc = 4.8;
  model.radial_cutoff_width = .27;
  model.radial_lambda = .5723;
  model.radial_count = 8;
  model.deployment_kind = v3 ? TaggedCauchyDeploymentKind::PhysicalImageV3
                             : TaggedCauchyDeploymentKind::LegacyMomentV2;
  model.source_realization = v3 ? TaggedCauchySourceRealization::ShiftedJacobiThreeTermV1
                                : TaggedCauchySourceRealization::PaceChebExpCos;
  // Two distinct unitary real forms at the SAME l must not be grouped together.
  for (int alternate = 0; alternate < 2; ++alternate) {
    TaggedCauchyRealForm form;
    form.l = l;
    form.width = 2 * l + 1;
    form.inverse.resize(form.width * form.width);
    form.inverse[l * form.width + l] = 1.0;
    const double scale = 1 / std::sqrt(2.0);
    for (int m = 1; m <= l; ++m) {
      const double phase = (m % 2) ? -1.0 : 1.0;
      form.inverse[(l + m) * form.width + l + m] = phase * scale;
      form.inverse[(l + m) * form.width + l - m] = scale;
      form.inverse[(l - m) * form.width + l + m] = Complex(0, -phase * scale);
      form.inverse[(l - m) * form.width + l - m] = Complex(0, scale);
    }
    if (alternate)
      for (int row = 0; row < form.width; row += 2)
        for (int col = 0; col < form.width; ++col) form.inverse[row * form.width + col] *= -1.0;
    model.real_forms.push_back(form);
  }
  // Deliberately unsorted, repeated radial degrees. The V3 test exceeds the
  // legacy radial-array cap without extending the deployed component/l caps.
  const int degree[] = {v3 ? 31 : 5, 0, v3 ? 11 : 5, 0, v3 ? 11 : 2, v3 ? 11 : 1};
  const int species[] = {0, 1, 0, 0, 1, 0};
  for (int i = 0; i < 6; ++i) {
    TaggedCauchyChannel c;
    c.channel_index = i;
    c.l = l;
    c.radial_channel = degree[i];
    c.neighbor_species_index = species[i];
    c.real_form_index = i == 5 ? 1 : 0;
    c.component_offset = model.total_component_count;
    c.normalization = 1.1 + i * .07;
    c.angular_scale = .9 + i * .03;
    model.total_component_count += 2 * l + 1;
    model.channels.push_back(c);
  }
  return model;
}
