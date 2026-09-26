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

#ifndef LMP_YE3T_GPU_BLOCK_SCHEDULE_H
#define LMP_YE3T_GPU_BLOCK_SCHEDULE_H

// Execution metadata only. The serialized block coefficients, occupation
// normalization, source binding, and descriptor selection are not touched here.
// clang-format off
#include <algorithm>
#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>
#include <utility>
// clang-format on

namespace YE3T_LAMMPS {
constexpr int GPU_BLOCK_MONOMIAL_TILE = 16;
constexpr int GPU_BLOCK_PREFIX_CAPACITY = 32;

struct GpuBlockExecutionSchedule {
  std::vector<int> plan_tile_offsets{0};
  std::vector<int> tile_monomial_begin;
  std::vector<int> tile_monomial_count;
  std::vector<int> tile_output_offsets{0};
  std::vector<int> tile_outputs;
  std::vector<int> tile_coefficient_offsets{0};
  std::vector<int> tile_coefficients;
  std::vector<int> monomial_output_offsets{0};
  std::vector<int> monomial_outputs;
  std::vector<int> monomial_coefficients;
};

inline int gpu_block_index(std::size_t size)
{
  if (size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("YE3T GPU block schedule exceeds 32-bit indexing");
  return static_cast<int>(size);
}

// A forward coefficient is visited exactly once, in an output-grouped bucket
// for its monomial tile. The reverse table gathers by monomial, retaining
// duplicate coefficient entries and without runtime index searches or atomics.
template <class Plans, class Offsets, class Terms>
GpuBlockExecutionSchedule
make_gpu_block_execution_schedule(const Plans &plans, const Offsets &output_offsets,
                                  const Terms &terms, int monomial_count, int direct_flag,
                                  int conjugate_half_flag)
{
  if (monomial_count < 0 || output_offsets.empty())
    throw std::invalid_argument("invalid YE3T GPU block schedule dimensions");
  GpuBlockExecutionSchedule result;
  std::vector<std::vector<std::pair<int, int>>> reverse(monomial_count);
  int expected_monomial = 0;
  for (const auto &plan : plans) {
    if (plan.monomial_begin != expected_monomial || plan.monomial_count < 0 ||
        plan.monomial_count > monomial_count - expected_monomial || plan.output_begin < 0 ||
        plan.output_count < 0 ||
        static_cast<std::size_t>(plan.output_begin) + plan.output_count >= output_offsets.size() ||
        (plan.output_L) < 0 || (plan.output_L) > (std::numeric_limits<int>::max() - 1) / 2)
      throw std::invalid_argument("invalid YE3T GPU block plan range");
    expected_monomial += plan.monomial_count;
    const bool direct = (plan.flags & direct_flag) != 0;
    const bool half = (plan.flags & conjugate_half_flag) != 0;
    const int width = 2 * plan.output_L + 1;
    if ((half && plan.output_count % width != 0) || (direct && plan.monomial_count != 0))
      throw std::invalid_argument("invalid YE3T GPU block identity/half-output shape");
    const int tiles = plan.monomial_count / GPU_BLOCK_MONOMIAL_TILE +
        (plan.monomial_count % GPU_BLOCK_MONOMIAL_TILE != 0);
    std::vector<std::map<int, std::vector<int>>> buckets(tiles);
    if (!direct)
      for (int component = 0; component < plan.output_count; ++component) {
        const int output = plan.output_begin + component;
        const auto begin = output_offsets[output], end = output_offsets[output + 1];
        if (begin < 0 || end < begin || static_cast<std::size_t>(end) > terms.size())
          throw std::invalid_argument("invalid YE3T GPU block coefficient range");
        for (auto coefficient = begin; coefficient < end; ++coefficient) {
          const auto monomial = terms[coefficient];
          if (monomial < 0 || monomial >= plan.monomial_count)
            throw std::invalid_argument("invalid YE3T GPU block coefficient monomial");
          // Negative-m rows are reconstructed from the positive half. Their
          // reverse roots are folded first, so these coefficients are unused.
          if (half && component % width < plan.output_L) continue;
          const int index = gpu_block_index(static_cast<std::size_t>(coefficient));
          buckets[monomial / GPU_BLOCK_MONOMIAL_TILE][component].push_back(index);
          reverse[plan.monomial_begin + monomial].emplace_back(component, index);
        }
      }
    for (int tile = 0; tile < tiles; ++tile) {
      const int begin = tile * GPU_BLOCK_MONOMIAL_TILE;
      result.tile_monomial_begin.push_back(plan.monomial_begin + begin);
      result.tile_monomial_count.push_back(
          std::min(GPU_BLOCK_MONOMIAL_TILE, plan.monomial_count - begin));
      for (const auto &row : buckets[tile]) {
        result.tile_outputs.push_back(row.first);
        result.tile_coefficients.insert(result.tile_coefficients.end(), row.second.begin(),
                                        row.second.end());
        result.tile_coefficient_offsets.push_back(gpu_block_index(result.tile_coefficients.size()));
      }
      result.tile_output_offsets.push_back(gpu_block_index(result.tile_outputs.size()));
    }
    result.plan_tile_offsets.push_back(gpu_block_index(result.tile_monomial_begin.size()));
  }
  if (expected_monomial != monomial_count)
    throw std::invalid_argument("YE3T GPU block schedule does not cover monomials");
  for (const auto &row : reverse) {
    for (const auto &entry : row) {
      result.monomial_outputs.push_back(entry.first);
      result.monomial_coefficients.push_back(entry.second);
    }
    result.monomial_output_offsets.push_back(gpu_block_index(result.monomial_outputs.size()));
  }
  return result;
}

#ifdef KOKKOS_INLINE_FUNCTION
#define YE3T_BLOCK_INLINE KOKKOS_INLINE_FUNCTION
#else
#define YE3T_BLOCK_INLINE inline
#endif

struct GpuBlockComplex {
  double real = 0.0, imaginary = 0.0;
};
YE3T_BLOCK_INLINE GpuBlockComplex gpu_block_multiply(GpuBlockComplex a, GpuBlockComplex b)
{
  return {a.real * b.real - a.imaginary * b.imaginary, a.real * b.imaginary + a.imaginary * b.real};
}

// Also used for a center-owned DIRECT graph. This is a non-owning view of one
// lane/slice; indexing arithmetic is shared by production and host tests.
template <class Scalar> struct GpuStridedValues {
  Scalar *data = nullptr;
  std::size_t stride = 1;
  std::size_t offset = 0;
  YE3T_BLOCK_INLINE Scalar &operator()(int value) const
  {
    return data[offset + static_cast<std::size_t>(value) * stride];
  }
};

using GpuStridedDoubles = GpuStridedValues<double>;
using GpuConstStridedDoubles = GpuStridedValues<const double>;

template <class Block, class Record, class Power>
YE3T_BLOCK_INLINE GpuBlockComplex gpu_block_factor(const Block &b, const Record &p, int factor,
                                                   const Power &pr, const Power &pi,
                                                   bool lower = false)
{
  const int component = b.monomial_factor_components(factor);
  const int exponent = b.monomial_factor_exponents(factor) - (lower ? 1 : 0);
  const int index = b.plan_input_power_offsets(p.input_begin + component) + exponent;
  return {pr(index), pi(index)};
}

// Original output rows are accumulated tile by tile. At most 16 monomials are
// resident per center, irrespective of other plans in the model. No dense
// ambient tensor, division by a source value, or source replay is introduced.
template <class Block, class Record, class Power, class Output, class Scratch>
YE3T_BLOCK_INLINE void
gpu_block_forward_tiles(const Block &b, const Record &p, int plan_index, const Power &pr,
                        const Power &pi, const Output &out_r, const Output &out_i,
                        const Scratch &tmp_r, const Scratch &tmp_i, bool real_coefficients)
{
  for (int component = 0; component < p.output_count; ++component) {
    out_r(p.output_storage_offset + component) = 0.0;
    out_i(p.output_storage_offset + component) = 0.0;
  }
  for (int tile = b.plan_tile_offsets(plan_index); tile < b.plan_tile_offsets(plan_index + 1);
       ++tile) {
    const int begin = b.tile_monomial_begin(tile), count = b.tile_monomial_count(tile);
    for (int m = 0; m < count; ++m) {
      GpuBlockComplex value{1.0, 0.0};
      const int f0 = b.monomial_factor_offsets(begin + m),
                f1 = b.monomial_factor_offsets(begin + m + 1);
      for (int factor = f0; factor < f1; ++factor) {
        const auto f = gpu_block_factor(b, p, factor, pr, pi);
        value = factor == f0 ? f : gpu_block_multiply(value, f);
      }
      tmp_r(m) = value.real;
      tmp_i(m) = value.imaginary;
    }
    for (int row = b.tile_output_offsets(tile); row < b.tile_output_offsets(tile + 1); ++row) {
      GpuBlockComplex sum;
      for (int entry = b.tile_coefficient_offsets(row); entry < b.tile_coefficient_offsets(row + 1);
           ++entry) {
        const int c = b.tile_coefficients(entry);
        const int m = p.monomial_begin + b.coefficient_terms(c) - begin;
        const double cr = b.coefficient_real(c),
                     ci = real_coefficients ? 0.0 : b.coefficient_imaginary(c);
        const auto term = gpu_block_multiply({cr, ci}, {tmp_r(m), tmp_i(m)});
        sum.real += term.real;
        sum.imaginary += term.imaginary;
      }
      const int output = p.output_storage_offset + b.tile_outputs(row);
      out_r(output) += sum.real;
      out_i(output) += sum.imaginary;
    }
  }
}

// Linear-work zero-safe pullback for the usual bounded monomial support. This
// stores prefixes, streams a suffix, and includes x^(e-1), never x^e/x. Larger
// supports use the original bounded-memory algorithm; no model is rejected.
template <class Block, class Record, class Power, class Add>
YE3T_BLOCK_INLINE void gpu_block_monomial_pullback(const Block &b, const Record &p, int monomial,
                                                   GpuBlockComplex root, const Power &pr,
                                                   const Power &pi, const Add &add, double scale)
{
  const int begin = b.monomial_factor_offsets(monomial),
            end = b.monomial_factor_offsets(monomial + 1);
  const int count = end - begin;
  if (count <= GPU_BLOCK_PREFIX_CAPACITY) {
    GpuBlockComplex prefix[GPU_BLOCK_PREFIX_CAPACITY];
    GpuBlockComplex product{1.0, 0.0};
    for (int f = 0; f < count; ++f) {
      prefix[f] = product;
      if (f + 1 < count)
        product = f == 0 ? gpu_block_factor(b, p, begin + f, pr, pi)
                         : gpu_block_multiply(product, gpu_block_factor(b, p, begin + f, pr, pi));
    }
    GpuBlockComplex suffix{1.0, 0.0};
    for (int f = count - 1; f >= 0; --f) {
      auto other = prefix[f];
      if (f == 0)
        other = suffix;
      else if (f + 1 < count)
        other = gpu_block_multiply(other, suffix);
      auto derivative = gpu_block_factor(b, p, begin + f, pr, pi, true);
      if (count > 1) derivative = gpu_block_multiply(other, derivative);
      const double weight = scale * b.monomial_factor_exponents(begin + f);
      const int component = b.monomial_factor_components(begin + f);
      add(b.plan_input_channels(p.input_begin + component),
          weight * (root.real * derivative.real + root.imaginary * derivative.imaginary),
          weight * (root.imaginary * derivative.real - root.real * derivative.imaginary));
      if (f > 0)
        suffix = f == count - 1
            ? gpu_block_factor(b, p, begin + f, pr, pi)
            : gpu_block_multiply(gpu_block_factor(b, p, begin + f, pr, pi), suffix);
    }
  } else {
    GpuBlockComplex prefix{1.0, 0.0};
    for (int target = begin; target < end; ++target) {
      GpuBlockComplex suffix{1.0, 0.0};
      for (int f = target + 1; f < end; ++f)
        suffix = f == target + 1 ? gpu_block_factor(b, p, f, pr, pi)
                                 : gpu_block_multiply(suffix, gpu_block_factor(b, p, f, pr, pi));
      auto other = target == begin
          ? suffix
          : (target + 1 == end ? prefix : gpu_block_multiply(prefix, suffix));
      const auto derivative =
          gpu_block_multiply(other, gpu_block_factor(b, p, target, pr, pi, true));
      const double weight = scale * b.monomial_factor_exponents(target);
      add(b.plan_input_channels(p.input_begin + b.monomial_factor_components(target)),
          weight * (root.real * derivative.real + root.imaginary * derivative.imaginary),
          weight * (root.imaginary * derivative.real - root.real * derivative.imaginary));
      const auto factor = gpu_block_factor(b, p, target, pr, pi);
      prefix = target == begin ? factor : gpu_block_multiply(prefix, factor);
    }
  }
}

template <class Block, class Record, class Power, class Root, class Add>
YE3T_BLOCK_INLINE void gpu_block_adjoint_transpose(const Block &b, const Record &p, const Power &pr,
                                                   const Power &pi, const Root &root_r,
                                                   const Root &root_i, const Add &add, double scale,
                                                   bool real_coefficients)
{
  for (int monomial = p.monomial_begin; monomial < p.monomial_begin + p.monomial_count;
       ++monomial) {
    GpuBlockComplex root;
    for (int entry = b.monomial_output_offsets(monomial);
         entry < b.monomial_output_offsets(monomial + 1); ++entry) {
      const int c = b.monomial_coefficients(entry);
      const int output = p.output_storage_offset + b.monomial_outputs(entry);
      const double cr = b.coefficient_real(c),
                   ci = real_coefficients ? 0.0 : b.coefficient_imaginary(c);
      root.real += root_r(output) * cr + root_i(output) * ci;
      root.imaginary += root_i(output) * cr - root_r(output) * ci;
    }
    gpu_block_monomial_pullback(b, p, monomial, root, pr, pi, add, scale);
  }
}
#undef YE3T_BLOCK_INLINE
}    // namespace YE3T_LAMMPS

#endif    // LMP_YE3T_GPU_BLOCK_SCHEDULE_H
