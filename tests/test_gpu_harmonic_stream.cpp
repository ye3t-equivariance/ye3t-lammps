#include "ye3t_gpu_harmonic_stream.h"
#include "ye3t_runtime_core.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
using namespace YE3T_LAMMPS;
struct Plan {
  const double *p;
  double operator()(int i) const { return p[i]; }
};
void near(double a, double b, double tolerance = 2e-10)
{
  if (!std::isfinite(a) || !std::isfinite(b) ||
      std::abs(a - b) > tolerance * (1 + std::abs(a) + std::abs(b)))
    throw std::runtime_error("streamed harmonic mismatch " + std::to_string(a) + " " +
                             std::to_string(b));
}
int main()
try {
  namespace rt = ye3t::runtime;
  std::mt19937 gen(7623);
  std::uniform_real_distribution<double> d(-2., 2.);
  double maximum_value_error = 0., maximum_gradient_error = 0.;
  int cases = 0;
  for (int lmax : {0, 1, 2, 4, 8, 12, 16}) {
    auto n = rt::complex_spherical_harmonics_nonnegative_table_width(lmax);
    std::vector<double> polynomial(rt::complex_spherical_harmonics_table_plan_size(lmax));
    std::vector<double> recurrence(rt::complex_spherical_harmonics_recurrence_plan_size(lmax));
    const double scale = std::sqrt(4 * std::acos(-1.));
    rt::build_complex_spherical_harmonics_table_plan(lmax, polynomial.data(), polynomial.size(),
                                                     scale);
    rt::build_complex_spherical_harmonics_recurrence_plan(lmax, recurrence.data(),
                                                          recurrence.size(), scale);
    Plan plan{recurrence.data()};
    for (int sample = 0; sample < 75; ++sample) {
      double v[3] = {d(gen), d(gen), d(gen)};
      if (sample < 6) {
        v[0] = v[1] = v[2] = 0;
        v[sample % 3] = sample < 3 ? 1. : -1.;
      }
      if (sample == 6) {
        v[0] = 1e-12;
        v[1] = -1e-12;
        v[2] = 1.;
      }
      double radius = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]), u[3];
      for (int a = 0; a < 3; ++a) u[a] = v[a] / radius;
      std::vector<std::complex<double>> ref(n), gref(3 * n), tmp(lmax + 1);
      rt::complex_spherical_harmonics_nonnegative_unit_table_with_derivative_prevalidated(
          u, &radius, 1, lmax, 1e-14, polynomial.data(), polynomial.size(), tmp.data(), tmp.size(),
          ref.data(), gref.data());
      GpuHarmonicStream<true, Plan> grads(plan, lmax, u, radius);
      GpuHarmonicStream<false, Plan> values(plan, lmax, u);
      for (int m = 0; m <= lmax; ++m)
        for (int l = m; l <= lmax; ++l) {
          int i = l * (l + 1) / 2 + m;
          auto g = grads.advance(l, m);
          auto y = values.advance(l, m);
          if (!g.valid || !y.valid) throw std::runtime_error("valid stream was rejected");
          near(y.real, g.real);
          near(y.imaginary, g.imaginary);
          near(y.real, ref[i].real());
          near(y.imaginary, ref[i].imag());
          maximum_value_error = std::max(
              maximum_value_error, std::abs(std::complex<double>(y.real, y.imaginary) - ref[i]));
          double dotr = 0., doti = 0.;
          for (int a = 0; a < 3; ++a) {
            near(g.gradient_real[a], gref[i * 3 + a].real());
            near(g.gradient_imaginary[a], gref[i * 3 + a].imag());
            maximum_gradient_error = std::max(
                maximum_gradient_error,
                std::abs(std::complex<double>(g.gradient_real[a], g.gradient_imaginary[a]) -
                         gref[i * 3 + a]));
            dotr += u[a] * g.gradient_real[a];
            doti += u[a] * g.gradient_imaginary[a];
          }
          near(dotr, 0.);
          near(doti, 0.);
          if (sample < 8 && lmax <= 8)
            for (int a = 0; a < 3; ++a) {
              const double h = 1e-6;
              double yp[2], ym[2];
              for (int sign : {-1, 1}) {
                double vec[3] = {v[0], v[1], v[2]};
                vec[a] += sign * h;
                double r = std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
                for (int aa = 0; aa < 3; ++aa) vec[aa] /= r;
                GpuHarmonicStream<false, Plan> perturbed(plan, lmax, vec);
                auto x = perturbed.advance(l, m);
                double *dest = sign < 0 ? ym : yp;
                dest[0] = x.real;
                dest[1] = x.imaginary;
              }
              near((yp[0] - ym[0]) / (2 * h), g.gradient_real[a], 2e-7);
              near((yp[1] - ym[1]) / (2 * h), g.gradient_imaginary[a], 2e-7);
            }
        }
      // Sparse groups, skipped degrees, independent edge-team columns.
      GpuHarmonicStream<true, Plan> sparse(plan, lmax, u, radius);
      for (int m = 0; m <= lmax; m += 2)
        for (int l = m; l <= lmax; l += 3) {
          auto y = sparse.advance(l, m);
          int i = l * (l + 1) / 2 + m;
          near(y.real, ref[i].real());
          near(y.gradient_real[2], gref[3 * i + 2].real());
        }
      for (int m = 0; m <= lmax; ++m) {
        GpuHarmonicStream<false, Plan> column(plan, lmax, u);
        auto y = column.advance(lmax, m);
        near(y.real, ref[lmax * (lmax + 1) / 2 + m].real());
      }
      GpuHarmonicStream<true, Plan> bad(plan, lmax, u, 0.);
      if (bad.advance(0, 0).valid) throw std::runtime_error("zero radius accepted");
      ++cases;
    }
  }
  std::cout << cases
            << " harmonic cases PASS, lmax through 16, axes, sparse columns, Cartesian finite "
               "differences; max abs value error "
            << maximum_value_error << ", gradient " << maximum_gradient_error << '\n';
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
