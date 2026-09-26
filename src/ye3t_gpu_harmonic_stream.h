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

#ifndef LMP_YE3T_GPU_HARMONIC_STREAM_H
#define LMP_YE3T_GPU_HARMONIC_STREAM_H

// Normalized nonnegative-m harmonics using the same recurrence-plan convention
// as the native CPU runtime. A column is streamed, not materialized as an
// edge-by-harmonic Jacobian. Call advance(l,m) in increasing (m,l) order.
namespace YE3T_LAMMPS {
#ifdef KOKKOS_INLINE_FUNCTION
#define YE3T_HARMONIC_INLINE KOKKOS_INLINE_FUNCTION
#else
#define YE3T_HARMONIC_INLINE inline
#endif

template <bool Derivatives> struct GpuHarmonicPoint;
template <> struct GpuHarmonicPoint<false> {
  double real = 0.0, imaginary = 0.0;
  bool valid = true;
};
template <> struct GpuHarmonicPoint<true> {
  double real = 0.0, imaginary = 0.0;
  double gradient_real[3] = {0.0, 0.0, 0.0};
  double gradient_imaginary[3] = {0.0, 0.0, 0.0};
  bool valid = true;
};

template <bool Derivatives, class Plan> struct GpuHarmonicStream {
  using Point = GpuHarmonicPoint<Derivatives>;
  Plan plan;
  int maximum_degree = 0, order = 0, degree = 0;
  double u[3] = {};
  // Empty gradient state in the values-only specialization is optimized away.
  double dx[Derivatives ? 3 : 1] = {}, dy[Derivatives ? 3 : 1] = {}, dz[Derivatives ? 3 : 1] = {};
  Point diagonal, previous, previous2;
  bool valid = true;

  YE3T_HARMONIC_INLINE GpuHarmonicStream(Plan p, int lmax, const double unit[3],
                                         double radius = 1.0) : plan(p), maximum_degree(lmax)
  {
    for (int a = 0; a < 3; ++a) u[a] = unit[a];
    diagonal.real = plan(0);
    previous = diagonal;
    if constexpr (Derivatives) {
      valid = radius > 1.0e-14;
      const double inverse = valid ? 1.0 / radius : 0.0;
      for (int a = 0; a < 3; ++a) {
        dx[a] = ((a == 0 ? 1.0 : 0.0) - u[0] * u[a]) * inverse;
        dy[a] = ((a == 1 ? 1.0 : 0.0) - u[1] * u[a]) * inverse;
        dz[a] = ((a == 2 ? 1.0 : 0.0) - u[2] * u[a]) * inverse;
      }
    }
  }

  YE3T_HARMONIC_INLINE Point advance(int l, int m)
  {
    if (!valid || m < order || l < m || l > maximum_degree || (m == order && l < degree)) {
      Point result;
      result.valid = false;
      return result;
    }
    while (order < m) {
      ++order;
      const int index = order * (order + 1) / 2 + order;
      const double a = plan(1 + 2 * index);
      Point next;
      next.real = a * (u[0] * diagonal.real - u[1] * diagonal.imaginary);
      next.imaginary = a * (u[0] * diagonal.imaginary + u[1] * diagonal.real);
      if constexpr (Derivatives)
        for (int axis = 0; axis < 3; ++axis) {
          next.gradient_real[axis] = a *
              (u[0] * diagonal.gradient_real[axis] - u[1] * diagonal.gradient_imaginary[axis] +
               dx[axis] * diagonal.real - dy[axis] * diagonal.imaginary);
          next.gradient_imaginary[axis] = a *
              (u[0] * diagonal.gradient_imaginary[axis] + u[1] * diagonal.gradient_real[axis] +
               dx[axis] * diagonal.imaginary + dy[axis] * diagonal.real);
        }
      diagonal = next;
      previous = next;
      previous2 = Point{};
      degree = order;
    }
    while (degree < l) {
      ++degree;
      const int index = degree * (degree + 1) / 2 + order;
      const double a = plan(1 + 2 * index), b = plan(2 + 2 * index);
      Point next;
      next.real = a * u[2] * previous.real - b * previous2.real;
      next.imaginary = a * u[2] * previous.imaginary - b * previous2.imaginary;
      if constexpr (Derivatives)
        for (int axis = 0; axis < 3; ++axis) {
          next.gradient_real[axis] =
              a * (u[2] * previous.gradient_real[axis] + dz[axis] * previous.real) -
              b * previous2.gradient_real[axis];
          next.gradient_imaginary[axis] =
              a * (u[2] * previous.gradient_imaginary[axis] + dz[axis] * previous.imaginary) -
              b * previous2.gradient_imaginary[axis];
        }
      previous2 = previous;
      previous = next;
    }
    return previous;
  }
};
#undef YE3T_HARMONIC_INLINE
}    // namespace YE3T_LAMMPS

#endif    // LMP_YE3T_GPU_HARMONIC_STREAM_H
