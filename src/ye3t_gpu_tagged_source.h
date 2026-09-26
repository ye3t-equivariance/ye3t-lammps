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

#ifndef LMP_YE3T_GPU_TAGGED_SOURCE_H
#define LMP_YE3T_GPU_TAGGED_SOURCE_H

#include "ye3t_tagged_cauchy_model.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace YE3T_LAMMPS {

// Bounds retained from the existing backend. Kernel-local storage is NOT
// guaranteed to be registers; inspect compiler/profiler output on the target.
constexpr int TAGGED_KOKKOS_MAX_L = 8;
constexpr int TAGGED_KOKKOS_MAX_COMPONENTS = 128;
constexpr int TAGGED_KOKKOS_MAX_TERM_FACTORS = 32;
constexpr int TAGGED_KOKKOS_MAX_RADIAL_COUNT = 16;
constexpr int TAGGED_KOKKOS_LEGENDRE_STRIDE = TAGGED_KOKKOS_MAX_L + 1;
constexpr int TAGGED_KOKKOS_LEGENDRE_M_STRIDE = TAGGED_KOKKOS_MAX_L + 2;

// Offline scheduling only: component numbering, real forms and all source
// channels remain unchanged. A group shares species, l and the real-form ID.
struct GpuTaggedSourceGroups {
  std::vector<int> offsets{0};
  std::vector<int> channels;
};
inline GpuTaggedSourceGroups make_gpu_tagged_source_groups(const TaggedCauchyModel &model)
{
  if (model.channels.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      model.total_component_count < 0 || model.total_component_count > TAGGED_KOKKOS_MAX_COMPONENTS)
    throw std::invalid_argument("invalid tagged GPU source dimensions");
  std::map<std::tuple<int, int, int>, std::vector<int>> groups;
  std::vector<bool> occupied(model.total_component_count, false);
  const bool v3 = model.deployment_kind == TaggedCauchyDeploymentKind::PhysicalImageV3;
  for (std::size_t index = 0; index < model.channels.size(); ++index) {
    const auto &c = model.channels[index];
    if (c.l < 0 || c.l > TAGGED_KOKKOS_MAX_L || c.radial_channel < 0 ||
        c.neighbor_species_index < 0 ||
        c.neighbor_species_index >= static_cast<int>(model.species_order.size()) ||
        c.real_form_index < 0 || c.real_form_index >= static_cast<int>(model.real_forms.size()) ||
        (!v3 &&
         (model.radial_count <= 0 || model.radial_count > TAGGED_KOKKOS_MAX_RADIAL_COUNT ||
          c.radial_channel >= model.radial_count)))
      throw std::invalid_argument("invalid tagged GPU source channel");
    const auto &form = model.real_forms[c.real_form_index];
    const int width = 2 * c.l + 1;
    if (form.l != c.l || form.width != width ||
        form.inverse.size() != static_cast<std::size_t>(width * width) || c.component_offset < 0 ||
        c.component_offset > model.total_component_count - width)
      throw std::invalid_argument("invalid tagged GPU real-form/component layout");
    for (int a = 0; a < width; ++a) {
      if (occupied[c.component_offset + a])
        throw std::invalid_argument("overlapping tagged GPU component layout");
      occupied[c.component_offset + a] = true;
    }
    groups[{c.neighbor_species_index, c.l, c.real_form_index}].push_back(static_cast<int>(index));
  }
  GpuTaggedSourceGroups result;
  for (auto &entry : groups) {
    auto &channels = entry.second;
    std::stable_sort(channels.begin(), channels.end(), [&](int a, int b) {
      return model.channels[a].radial_channel < model.channels[b].radial_channel;
    });
    result.channels.insert(result.channels.end(), channels.begin(), channels.end());
    result.offsets.push_back(static_cast<int>(result.channels.size()));
  }
  return result;
}

inline void build_gpu_tagged_legendre_table(int l, double *derivative_table,
                                            double *normalization_row)
{
  // Ordinary Legendre P_l(z), ascending powers (same 3-term recursion as
  // ye3t_lifted_cauchy_cpu.cpp's legendre_coefficients / the Python
  // reference's _legendre_coefficients_ascending).
  std::vector<double> previous{1.0};
  std::vector<double> current = (l == 0) ? std::vector<double>{1.0} : std::vector<double>{0.0, 1.0};
  for (int degree = 1; degree < l; ++degree) {
    std::vector<double> next(static_cast<std::size_t>(degree + 2), 0.0);
    for (std::size_t power = 0; power < current.size(); ++power)
      next[power + 1] += (2.0 * degree + 1.0) / (degree + 1.0) * current[power];
    for (std::size_t power = 0; power < previous.size(); ++power)
      next[power] -= static_cast<double>(degree) / (degree + 1.0) * previous[power];
    previous = std::move(current);
    current = std::move(next);
  }
  std::vector<double> polynomial = current;    // P_l, ascending, degree l
  for (int m = 0; m <= l; ++m) {
    double *row = derivative_table + m * TAGGED_KOKKOS_LEGENDRE_STRIDE;
    for (int power = 0; power <= l; ++power)
      row[power] = (power < static_cast<int>(polynomial.size())) ? polynomial[power] : 0.0;
    const double log_norm = 0.5 *
        (std::log(2.0 * l + 1.0) - std::log(4.0 * 3.14159265358979323846) +
         std::lgamma(l - m + 1.0) - std::lgamma(l + m + 1.0));
    normalization_row[m] = std::exp(log_norm);
    if (m == l) break;
    std::vector<double> next_polynomial(polynomial.size() > 1 ? polynomial.size() - 1 : 0);
    for (std::size_t power = 1; power < polynomial.size(); ++power)
      next_polynomial[power - 1] = static_cast<double>(power) * polynomial[power];
    polynomial = std::move(next_polynomial);
  }
}

#ifdef KOKKOS_INLINE_FUNCTION
#define YE3T_GPU_TAGGED_INLINE KOKKOS_INLINE_FUNCTION
#else
#define YE3T_GPU_TAGGED_INLINE inline
#endif

template <class Math>
YE3T_GPU_TAGGED_INLINE void gpu_tagged_radial(double r, double rc, double cutoff_width,
                                              double lmbda, int radial_count, double *values,
                                              double *derivatives)
{
  const double pi = 3.14159265358979323846;
  if (r >= rc) {
    for (int n = 0; n < radial_count; ++n) {
      values[n] = 0.0;
      derivatives[n] = 0.0;
    }
    return;
  }
  const double scaled = r / rc;
  const double envelope = 0.5 * (1.0 + Math::cos(pi * scaled));
  const double envelope_derivative = -0.5 * pi * Math::sin(pi * scaled) / rc;
  double outer_switch = 1.0;
  double outer_switch_derivative = 0.0;
  if (cutoff_width > 0.0 && r > rc - cutoff_width) {
    const double phase = pi * (r - (rc - cutoff_width)) / cutoff_width;
    outer_switch = 0.5 * (1.0 + Math::cos(phase));
    outer_switch_derivative = -0.5 * pi * Math::sin(phase) / cutoff_width;
  }
  const double combined = envelope * outer_switch;
  const double combined_derivative =
      envelope_derivative * outer_switch + envelope * outer_switch_derivative;
  values[0] = combined;
  derivatives[0] = combined_derivative;
  if (radial_count == 1) return;
  const double exponential = Math::exp(-lmbda * (scaled - 1.0));
  const double denominator = Math::exp(lmbda) - 1.0;
  const double warped = 1.0 - 2.0 * (exponential - 1.0) / denominator;
  const double warped_derivative = 2.0 * lmbda * exponential / (denominator * rc);
  double chebyshev_previous = 1.0;
  double chebyshev = warped;
  double chebyshev_derivative_previous = 0.0;
  double chebyshev_derivative = warped_derivative;
  for (int radial = 1; radial < radial_count; ++radial) {
    if (radial > 1) {
      const double next = 2.0 * warped * chebyshev - chebyshev_previous;
      const double next_derivative =
          2.0 * (warped_derivative * chebyshev + warped * chebyshev_derivative) -
          chebyshev_derivative_previous;
      chebyshev_previous = chebyshev;
      chebyshev = next;
      chebyshev_derivative_previous = chebyshev_derivative;
      chebyshev_derivative = next_derivative;
    }
    const double base = 0.5 * (1.0 - chebyshev);
    const double base_derivative = -0.5 * chebyshev_derivative;
    values[radial] = base * combined;
    derivatives[radial] = base_derivative * combined + base * combined_derivative;
  }
}

template <bool NeedGradient, class Math>
YE3T_GPU_TAGGED_INLINE void
gpu_tagged_complex_ylm(int l, double dx, double dy, double dz,
                       const double *legendre_derivative_slice, const double *normalization_slice,
                       typename Math::Complex *Yc, typename Math::Complex *dYc_dx,
                       typename Math::Complex *dYc_dy, typename Math::Complex *dYc_dz)
{
  constexpr double epsilon = 1.0e-12;
  const double r = Math::sqrt(dx * dx + dy * dy + dz * dz);
  const double safe_r = r > epsilon ? r : epsilon;
  const double ux = dx / safe_r;
  const double uy = dy / safe_r;
  const double uz = dz / safe_r;
  const int width = 2 * l + 1;

  // P_l^{(m)}(z) (poly) and P_l^{(m+1)}(z) (dpoly; 0 at m == l), via the
  // uploaded ascending-power coefficient table, and (ux+-i uy)^m built
  // incrementally once for the whole shared angular/real-form group.
  double poly[TAGGED_KOKKOS_MAX_L + 1];
  double dpoly[NeedGradient ? TAGGED_KOKKOS_MAX_L + 1 : 1];
  typename Math::Complex power_plus[TAGGED_KOKKOS_MAX_L + 1];     // (ux + i uy)^m
  typename Math::Complex power_minus[TAGGED_KOKKOS_MAX_L + 1];    // (ux - i uy)^m
  const typename Math::Complex xy(ux, uy);
  const typename Math::Complex xy_conj(ux, -uy);
  power_plus[0] = typename Math::Complex(1.0, 0.0);
  power_minus[0] = typename Math::Complex(1.0, 0.0);
  for (int m = 1; m <= l; ++m) {
    power_plus[m] = power_plus[m - 1] * xy;
    power_minus[m] = power_minus[m - 1] * xy_conj;
  }
  for (int m = 0; m <= l; ++m) {
    const int degree = l - m;
    const double *coefficients = legendre_derivative_slice + m * TAGGED_KOKKOS_LEGENDRE_STRIDE;
    double value = coefficients[degree];
    for (int power_index = degree - 1; power_index >= 0; --power_index)
      value = value * uz + coefficients[power_index];
    poly[m] = value;

    if constexpr (NeedGradient) {
      double derivative = 0.0;
      if (m < l) {
        const double *next_coefficients =
            legendre_derivative_slice + (m + 1) * TAGGED_KOKKOS_LEGENDRE_STRIDE;
        const int next_degree = l - m - 1;
        derivative = next_coefficients[next_degree];
        for (int power_index = next_degree - 1; power_index >= 0; --power_index)
          derivative = derivative * uz + next_coefficients[power_index];
      }
      dpoly[m] = derivative;
    }
  }

  // Y_l^{+m} = N_{l,m} (-1)^m P_l^{(m)}(z) (ux+i uy)^m,
  // Y_l^{-m} = N_{l,m} P_l^{(m)}(z) (ux-i uy)^m  (m = 1..l; no (-1)^m factor)
  // (the same convention as
  // ye3t::runtime::complex_spherical_harmonics_with_derivative).
  for (int m = 0; m <= l; ++m) {
    const double norm = normalization_slice[m];
    const double sign = (m % 2 == 0) ? 1.0 : -1.0;
    const double coefficient_pos = norm * sign;

    Yc[l + m] = coefficient_pos * poly[m] * power_plus[m];
    if constexpr (NeedGradient) {
      const typename Math::Complex dpower_plus_dux = (m == 0)
          ? typename Math::Complex(0.0, 0.0)
          : typename Math::Complex(static_cast<double>(m), 0.0) * power_plus[m - 1];
      const typename Math::Complex dpower_plus_duy = (m == 0)
          ? typename Math::Complex(0.0, 0.0)
          : typename Math::Complex(0.0, static_cast<double>(m)) * power_plus[m - 1];
      dYc_dx[l + m] = coefficient_pos * poly[m] * dpower_plus_dux;
      dYc_dy[l + m] = coefficient_pos * poly[m] * dpower_plus_duy;
      dYc_dz[l + m] = coefficient_pos * dpoly[m] * power_plus[m];
    }

    if (m > 0) {
      Yc[l - m] = norm * poly[m] * power_minus[m];
      if constexpr (NeedGradient) {
        const typename Math::Complex dpower_minus_dux =
            typename Math::Complex(static_cast<double>(m), 0.0) * power_minus[m - 1];
        const typename Math::Complex dpower_minus_duy =
            typename Math::Complex(0.0, -static_cast<double>(m)) * power_minus[m - 1];
        dYc_dx[l - m] = norm * poly[m] * dpower_minus_dux;
        dYc_dy[l - m] = norm * poly[m] * dpower_minus_duy;
        dYc_dz[l - m] = norm * dpoly[m] * power_minus[m];
      }
    }
  }

  // Chain rule d(unit_k)/d(raw_j) = (delta_kj - unit_k*unit_j)/r (r>epsilon;
  // degenerate diagonal fallback otherwise, matching the CPU/runtime
  // convention exactly).
  if constexpr (NeedGradient) {
    const double unit[3] = {ux, uy, uz};
    double jacobian[3][3];
    for (int k = 0; k < 3; ++k)
      for (int j = 0; j < 3; ++j)
        jacobian[k][j] = (r > epsilon) ? ((k == j ? 1.0 : 0.0) - unit[k] * unit[j]) / safe_r
                                       : ((k == j) ? 1.0 / safe_r : 0.0);
    for (int row = 0; row < width; ++row) {
      const typename Math::Complex du[3] = {dYc_dx[row], dYc_dy[row], dYc_dz[row]};
      typename Math::Complex out[3] = {0.0, 0.0, 0.0};
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) out[j] += du[k] * jacobian[k][j];
      dYc_dx[row] = out[0];
      dYc_dy[row] = out[1];
      dYc_dz[row] = out[2];
    }
  }
}

// Streaming form of the EXISTING shifted_jacobi_value_with_derivative
// recurrence. Sorted radial columns share prefixes; no new degree cap or
// full radial table is introduced. Repeated columns do not repeat work.
template <bool NeedGradient> struct GpuTaggedJacobiStream {
  int degree = 0;
  double value = 1.0, derivative = 0.0;
  double previous = 0.0, previous_derivative = 0.0;
  YE3T_GPU_TAGGED_INLINE void advance(int target, int angular_l, double x)
  {
    const double alpha = 4.0, beta = 2.0 * angular_l + 2.0;
    if (degree == 0 && target > 0) {
      previous = 1.0;
      value = (alpha + beta + 2.0) * x - (beta + 1.0);
      if constexpr (NeedGradient) derivative = alpha + beta + 2.0;
      degree = 1;
    }
    while (degree < target) {
      const double n = degree;
      const double total = 2.0 * n + alpha + beta;
      const double a = (total + 1.0) * (total + 2.0) / (2.0 * (n + 1.0) * (n + alpha + beta + 1.0));
      const double b = (alpha * alpha - beta * beta) * (total + 1.0) /
          (2.0 * (n + 1.0) * (n + alpha + beta + 1.0) * total);
      const double c =
          (n + alpha) * (n + beta) * (total + 2.0) / ((n + 1.0) * (n + alpha + beta + 1.0) * total);
      const double multiplier = a * (2.0 * x - 1.0) + b;
      const double next = multiplier * value - c * previous;
      if constexpr (NeedGradient) {
        const double next_derivative =
            2.0 * a * value + multiplier * derivative - c * previous_derivative;
        previous_derivative = derivative;
        derivative = next_derivative;
      }
      previous = value;
      value = next;
      ++degree;
    }
  }
};

// Math supplies the complex type and device-callable sqrt/cos/sin/exp. This
// exact body is instantiated with Kokkos for production and std math in tests.
template <bool NeedGradient, class Math, class Plan, class Consumer>
YE3T_GPU_TAGGED_INLINE void gpu_tagged_visit_components(const Plan &plan, int species, double dx,
                                                        double dy, double dz,
                                                        const Consumer &consume)
{
  const double r = Math::sqrt(dx * dx + dy * dy + dz * dz);
  if (r >= plan.cutoff) return;
  double radial[TAGGED_KOKKOS_MAX_RADIAL_COUNT];
  double radial_derivative[TAGGED_KOKKOS_MAX_RADIAL_COUNT];
  if (!plan.physical_image_v3)
    gpu_tagged_radial<Math>(r, plan.cutoff, plan.radial_cutoff_width, plan.radial_lambda,
                            plan.radial_count, radial, radial_derivative);
  const double safe_r = r > 1.0e-12 ? r : 1.0e-12;
  const double unit[3] = {dx / safe_r, dy / safe_r, dz / safe_r};
  const double x = r / plan.cutoff;
  for (int group = 0; group < plan.source_group_count; ++group) {
    const int begin = plan.source_group_offsets(group), end = plan.source_group_offsets(group + 1);
    const int first = plan.source_group_channels(begin);
    if (plan.channel_neighbor_species(first) != species) continue;
    const int l = plan.channel_l(first), width = 2 * l + 1;
    typename Math::Complex Y[2 * TAGGED_KOKKOS_MAX_L + 1];
    typename Math::Complex Yx[NeedGradient ? 2 * TAGGED_KOKKOS_MAX_L + 1 : 1];
    typename Math::Complex Yy[NeedGradient ? 2 * TAGGED_KOKKOS_MAX_L + 1 : 1];
    typename Math::Complex Yz[NeedGradient ? 2 * TAGGED_KOKKOS_MAX_L + 1 : 1];
    gpu_tagged_complex_ylm<NeedGradient, Math>(
        l, dx, dy, dz,
        plan.legendre_derivative.data() +
            static_cast<std::size_t>(first) * TAGGED_KOKKOS_LEGENDRE_M_STRIDE *
                TAGGED_KOKKOS_LEGENDRE_STRIDE,
        plan.normalization.data() +
            static_cast<std::size_t>(first) * TAGGED_KOKKOS_LEGENDRE_M_STRIDE,
        Y, Yx, Yy, Yz);
    double yr[2 * TAGGED_KOKKOS_MAX_L + 1];
    double yx[NeedGradient ? 2 * TAGGED_KOKKOS_MAX_L + 1 : 1];
    double yy[NeedGradient ? 2 * TAGGED_KOKKOS_MAX_L + 1 : 1];
    double yz[NeedGradient ? 2 * TAGGED_KOKKOS_MAX_L + 1 : 1];
    const int inverse_offset = plan.channel_inverse_offset(first);
    for (int a = 0; a < width; ++a) {
      typename Math::Complex value(0.0, 0.0), gx(0.0, 0.0), gy(0.0, 0.0), gz(0.0, 0.0);
      for (int row = 0; row < width; ++row) {
        const auto inverse = plan.inverse_matrix(inverse_offset + a * width + row);
        value += inverse * Y[row];
        if constexpr (NeedGradient) {
          gx += inverse * Yx[row];
          gy += inverse * Yy[row];
          gz += inverse * Yz[row];
        }
      }
      yr[a] = value.real();
      if constexpr (NeedGradient) {
        yx[a] = gx.real();
        yy[a] = gy.real();
        yz[a] = gz.real();
      }
    }
    double x_l = 1.0;
    for (int power = 0; power < l; ++power) x_l *= x;
    double x_l_dx = 0.0;
    if constexpr (NeedGradient) {
      if (l > 0) {
        x_l_dx = static_cast<double>(l);
        for (int power = 1; power < l; ++power) x_l_dx *= x;
      }
    }
    GpuTaggedJacobiStream<NeedGradient> jacobi;
    for (int entry = begin; entry < end; ++entry) {
      const int channel = plan.source_group_channels(entry);
      const int column = plan.channel_radial_column(channel);
      double R, dR = 0.0;
      if (plan.physical_image_v3) {
        jacobi.advance(column, l, x);
        const double one_minus = 1.0 - x, envelope = one_minus * one_minus;
        const double scale =
            plan.channel_source_normalization(channel) * plan.channel_angular_scale(channel);
        R = scale * envelope * jacobi.value * x_l;
        if constexpr (NeedGradient)
          dR = scale *
              ((-2.0 * one_minus) * jacobi.value * x_l + envelope * jacobi.derivative * x_l +
               envelope * jacobi.value * x_l_dx) /
              plan.cutoff;
      } else {
        R = radial[column];
        if constexpr (NeedGradient) dR = radial_derivative[column];
      }
      const int offset = plan.channel_component_offset(channel);
      for (int a = 0; a < width; ++a) {
        if constexpr (NeedGradient) {
          consume(offset + a, R * yr[a], dR * unit[0] * yr[a] + R * yx[a],
                  dR * unit[1] * yr[a] + R * yy[a], dR * unit[2] * yr[a] + R * yz[a]);
        } else {
          consume(offset + a, R * yr[a], 0.0, 0.0, 0.0);
        }
      }
    }
  }
}

// Store only what the caller requested. The forward instantiation never
// allocates the angular derivative arrays or writes derivative output.
template <bool NeedGradient> struct GpuTaggedArrayWriter {
  double *value, *gx, *gy, *gz;
  YE3T_GPU_TAGGED_INLINE void operator()(int index, double v, double x, double y, double z) const
  {
    value[index] = v;
    if constexpr (NeedGradient) {
      gx[index] = x;
      gy[index] = y;
      gz[index] = z;
    }
  }
};
template <bool NeedGradient, class Math, class Plan>
YE3T_GPU_TAGGED_INLINE void gpu_tagged_edge_components(const Plan &plan, int species, double dx,
                                                       double dy, double dz, double *value,
                                                       double *gx, double *gy, double *gz)
{
  for (int i = 0; i < plan.total_component_count; ++i) {
    value[i] = 0.0;
    if constexpr (NeedGradient) {
      gx[i] = 0.0;
      gy[i] = 0.0;
      gz[i] = 0.0;
    }
  }
  gpu_tagged_visit_components<NeedGradient, Math>(
      plan, species, dx, dy, dz, GpuTaggedArrayWriter<NeedGradient>{value, gx, gy, gz});
}

// For a V3 source with no real-moment keys, all component seeds are known before
// source evaluation. Contract gradients as they are produced, eliminating the
// 4 x MAX_COMPONENTS value/gradient arrays WITHOUT an extra source pass.
struct GpuTaggedGradientAccumulator {
  const double *seed;
  double *gradient;
  YE3T_GPU_TAGGED_INLINE void operator()(int index, double, double x, double y, double z) const
  {
    const double weight = seed[index];
    if (weight == 0.0) return;
    gradient[0] += weight * x;
    gradient[1] += weight * y;
    gradient[2] += weight * z;
  }
};
template <class Math, class Plan>
YE3T_GPU_TAGGED_INLINE void gpu_tagged_edge_vjp(const Plan &plan, int species, double dx, double dy,
                                                double dz, const double *seed, double *gradient)
{
  gradient[0] = gradient[1] = gradient[2] = 0.0;
  gpu_tagged_visit_components<true, Math>(plan, species, dx, dy, dz,
                                          GpuTaggedGradientAccumulator{seed, gradient});
}
#undef YE3T_GPU_TAGGED_INLINE
}    // namespace YE3T_LAMMPS

#endif    // LMP_YE3T_GPU_TAGGED_SOURCE_H
