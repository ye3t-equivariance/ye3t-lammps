// Arithmetic reference for the tagged GPU source helper.
// Only annotations, math namespace, and generic plan type changed for host tests.
// This is NOT a Kokkos emulation or a GPU execution test.
#pragma once
// clang-format off
#include <complex>
#include <cmath>
// clang-format on
namespace YE3T_LAMMPS {
namespace tagged_reference {
  inline void tagged_kokkos_radial(double r, double rc, double cutoff_width, double lmbda,
                                   int radial_count, double *values, double *derivatives)
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
    const double envelope = 0.5 * (1.0 + std::cos(pi * scaled));
    const double envelope_derivative = -0.5 * pi * std::sin(pi * scaled) / rc;
    double outer_switch = 1.0;
    double outer_switch_derivative = 0.0;
    if (cutoff_width > 0.0 && r > rc - cutoff_width) {
      const double phase = pi * (r - (rc - cutoff_width)) / cutoff_width;
      outer_switch = 0.5 * (1.0 + std::cos(phase));
      outer_switch_derivative = -0.5 * pi * std::sin(phase) / cutoff_width;
    }
    const double combined = envelope * outer_switch;
    const double combined_derivative =
        envelope_derivative * outer_switch + envelope * outer_switch_derivative;
    values[0] = combined;
    derivatives[0] = combined_derivative;
    if (radial_count == 1) return;
    const double exponential = std::exp(-lmbda * (scaled - 1.0));
    const double denominator = std::exp(lmbda) - 1.0;
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

  // Complex Y_l^m, m = -l..l (row index = l + m), for a raw (non-unit)
  // displacement, plus d(Y_l^m)/d(displacement axis) -- full chain rule
  // through r and the unit vector already applied, matching
  // ye3t::runtime::complex_spherical_harmonics_with_derivative's contract.
  // `legendre_derivative_slice`/`normalization_slice` are this channel's
  // TAGGED_KOKKOS_LEGENDRE_M_STRIDE * TAGGED_KOKKOS_LEGENDRE_STRIDE (resp.
  // TAGGED_KOKKOS_LEGENDRE_M_STRIDE) row out of the uploaded plan tables.
  // Derivation: P_l^{(m)}(z) times (x+-iy)^m, verified against the runtime
  // function and against complex_to_artifact_real of ComplexSphericalHarmonics
  // Basis.cartesian_values.
  inline void tagged_kokkos_complex_ylm(int l, double dx, double dy, double dz,
                                        const double *legendre_derivative_slice,
                                        const double *normalization_slice, std::complex<double> *Yc,
                                        std::complex<double> *dYc_dx, std::complex<double> *dYc_dy,
                                        std::complex<double> *dYc_dz)
  {
    constexpr double epsilon = 1.0e-12;
    const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
    const double safe_r = r > epsilon ? r : epsilon;
    const double ux = dx / safe_r;
    const double uy = dy / safe_r;
    const double uz = dz / safe_r;
    const int width = 2 * l + 1;

    // P_l^{(m)}(z) (poly) and P_l^{(m+1)}(z) (dpoly; 0 at m == l), via the
    // uploaded ascending-power coefficient table, and (ux+-i uy)^m built
    // incrementally once (l is tiny -- <= TAGGED_KOKKOS_MAX_L -- so this and
    // every loop below is O(l), not a performance concern).
    double poly[TAGGED_KOKKOS_MAX_L + 1];
    double dpoly[TAGGED_KOKKOS_MAX_L + 1];
    std::complex<double> power_plus[TAGGED_KOKKOS_MAX_L + 1];     // (ux + i uy)^m
    std::complex<double> power_minus[TAGGED_KOKKOS_MAX_L + 1];    // (ux - i uy)^m
    const std::complex<double> xy(ux, uy);
    const std::complex<double> xy_conj(ux, -uy);
    power_plus[0] = std::complex<double>(1.0, 0.0);
    power_minus[0] = std::complex<double>(1.0, 0.0);
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

    // Y_l^{+m} = N_{l,m} (-1)^m P_l^{(m)}(z) (ux+i uy)^m,
    // Y_l^{-m} = N_{l,m} P_l^{(m)}(z) (ux-i uy)^m  (m = 1..l; no (-1)^m factor)
    // -- verified against ye3t::runtime::complex_spherical_harmonics_with_
    // derivative to 1.1e-16.
    for (int m = 0; m <= l; ++m) {
      const double norm = normalization_slice[m];
      const double sign = (m % 2 == 0) ? 1.0 : -1.0;
      const double coefficient_pos = norm * sign;

      Yc[l + m] = coefficient_pos * poly[m] * power_plus[m];
      const std::complex<double> dpower_plus_dux = (m == 0)
          ? std::complex<double>(0.0, 0.0)
          : std::complex<double>(static_cast<double>(m), 0.0) * power_plus[m - 1];
      const std::complex<double> dpower_plus_duy = (m == 0)
          ? std::complex<double>(0.0, 0.0)
          : std::complex<double>(0.0, static_cast<double>(m)) * power_plus[m - 1];
      dYc_dx[l + m] = coefficient_pos * poly[m] * dpower_plus_dux;
      dYc_dy[l + m] = coefficient_pos * poly[m] * dpower_plus_duy;
      dYc_dz[l + m] = coefficient_pos * dpoly[m] * power_plus[m];

      if (m > 0) {
        Yc[l - m] = norm * poly[m] * power_minus[m];
        const std::complex<double> dpower_minus_dux =
            std::complex<double>(static_cast<double>(m), 0.0) * power_minus[m - 1];
        const std::complex<double> dpower_minus_duy =
            std::complex<double>(0.0, -static_cast<double>(m)) * power_minus[m - 1];
        dYc_dx[l - m] = norm * poly[m] * dpower_minus_dux;
        dYc_dy[l - m] = norm * poly[m] * dpower_minus_duy;
        dYc_dz[l - m] = norm * dpoly[m] * power_minus[m];
      }
    }

    // Chain rule d(unit_k)/d(raw_j) = (delta_kj - unit_k*unit_j)/r (r>epsilon;
    // degenerate diagonal fallback otherwise, matching the CPU/runtime
    // convention exactly).
    const double unit[3] = {ux, uy, uz};
    double jacobian[3][3];
    for (int k = 0; k < 3; ++k)
      for (int j = 0; j < 3; ++j)
        jacobian[k][j] = (r > epsilon) ? ((k == j ? 1.0 : 0.0) - unit[k] * unit[j]) / safe_r
                                       : ((k == j) ? 1.0 / safe_r : 0.0);
    for (int row = 0; row < width; ++row) {
      const std::complex<double> du[3] = {dYc_dx[row], dYc_dy[row], dYc_dz[row]};
      std::complex<double> out[3] = {0.0, 0.0, 0.0};
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) out[j] += du[k] * jacobian[k][j];
      dYc_dx[row] = out[0];
      dYc_dy[row] = out[1];
      dYc_dz[row] = out[2];
    }
  }

  // Per-edge phi_real[c][a] (flat over total_component_count) and, if
  // need_gradient, d(phi_real)/d(raw displacement) -- the exact CPU formula
  // (ye3t_tagged_cauchy_cpu.cpp's compute_edge_components): radial(column)
  // times real-form-transformed complex harmonics, zero for channels whose
  // neighbor_species doesn't match this edge, product rule for the gradient.
  // Shared by both TaggedBuildDensityMoment (need_gradient = false) and
  // TaggedEdgeVJP (need_gradient = true) in pair_ye3t_kokkos.cpp.
  template <class Plan>
  inline void tagged_kokkos_edge_components(const Plan &plan, int species, double dx, double dy,
                                            double dz, bool need_gradient, double *phi_real,
                                            double *phi_grad_x, double *phi_grad_y,
                                            double *phi_grad_z)
  {
    for (int c = 0; c < plan.total_component_count; ++c) {
      phi_real[c] = 0.0;
      if (need_gradient) {
        phi_grad_x[c] = 0.0;
        phi_grad_y[c] = 0.0;
        phi_grad_z[c] = 0.0;
      }
    }
    const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
    double radial_values[TAGGED_KOKKOS_MAX_RADIAL_COUNT];
    double radial_derivatives[TAGGED_KOKKOS_MAX_RADIAL_COUNT];
    if (!plan.physical_image_v3)
      tagged_kokkos_radial(r, plan.cutoff, plan.radial_cutoff_width, plan.radial_lambda,
                           plan.radial_count, radial_values, radial_derivatives);
    const double safe_r = r > 1.0e-12 ? r : 1.0e-12;
    const double unit[3] = {dx / safe_r, dy / safe_r, dz / safe_r};

    for (int c = 0; c < plan.channel_count; ++c) {
      if (plan.channel_neighbor_species(c) != species) continue;
      const int l = plan.channel_l(c);
      const int width = 2 * l + 1;
      std::complex<double> Yc[2 * TAGGED_KOKKOS_MAX_L + 1];
      std::complex<double> dYc_dx[2 * TAGGED_KOKKOS_MAX_L + 1];
      std::complex<double> dYc_dy[2 * TAGGED_KOKKOS_MAX_L + 1];
      std::complex<double> dYc_dz[2 * TAGGED_KOKKOS_MAX_L + 1];
      tagged_kokkos_complex_ylm(l, dx, dy, dz,
                                plan.legendre_derivative.data() +
                                    static_cast<std::size_t>(c) * TAGGED_KOKKOS_LEGENDRE_M_STRIDE *
                                        TAGGED_KOKKOS_LEGENDRE_STRIDE,
                                plan.normalization.data() +
                                    static_cast<std::size_t>(c) * TAGGED_KOKKOS_LEGENDRE_M_STRIDE,
                                Yc, dYc_dx, dYc_dy, dYc_dz);

      double R = 0.0;
      double dR = 0.0;
      if (plan.physical_image_v3) {
        const double x = r / plan.cutoff;
        if (x < 1.0) {
          double polynomial = 0.0;
          double polynomial_dx = 0.0;
          shifted_jacobi_value_with_derivative(plan.channel_radial_column(c), l, x, polynomial,
                                               polynomial_dx);
          const double one_minus = 1.0 - x;
          const double envelope = one_minus * one_minus;
          const double envelope_dx = -2.0 * one_minus;
          double x_l = 1.0;
          for (int power = 0; power < l; ++power) x_l *= x;
          double x_l_dx = 0.0;
          if (l > 0) {
            x_l_dx = static_cast<double>(l);
            for (int power = 1; power < l; ++power) x_l_dx *= x;
          }
          const double scale = plan.channel_source_normalization(c) * plan.channel_angular_scale(c);
          R = scale * envelope * polynomial * x_l;
          dR = scale *
              (envelope_dx * polynomial * x_l + envelope * polynomial_dx * x_l +
               envelope * polynomial * x_l_dx) /
              plan.cutoff;
        }
      } else {
        const int radial_column = plan.channel_radial_column(c);
        R = radial_values[radial_column];
        dR = radial_derivatives[radial_column];
      }
      const int inverse_offset = plan.channel_inverse_offset(c);
      const int component_offset = plan.channel_component_offset(c);
      for (int a = 0; a < width; ++a) {
        std::complex<double> acc(0.0, 0.0), gx(0.0, 0.0), gy(0.0, 0.0), gz(0.0, 0.0);
        for (int row = 0; row < width; ++row) {
          const std::complex<double> inv = plan.inverse_matrix(inverse_offset + a * width + row);
          acc += inv * Yc[row];
          if (need_gradient) {
            gx += inv * dYc_dx[row];
            gy += inv * dYc_dy[row];
            gz += inv * dYc_dz[row];
          }
        }
        const double y_real = acc.real();
        const int flat = component_offset + a;
        phi_real[flat] = R * y_real;
        if (need_gradient) {
          phi_grad_x[flat] = dR * unit[0] * y_real + R * gx.real();
          phi_grad_y[flat] = dR * unit[1] * y_real + R * gy.real();
          phi_grad_z[flat] = dR * unit[2] * y_real + R * gz.real();
        }
      }
    }
  }

}    // namespace tagged_reference
}    // namespace YE3T_LAMMPS
