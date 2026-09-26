/* ----------------------------------------------------------------------
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

#include "pair_ye3t_kokkos.h"

#include "ye3t_sha256.h"
#include <Kokkos_Profiling_ScopedRegion.hpp>

#include "atom.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "neighbor_kokkos.h"
#include "utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime_api.h>
#elif defined(KOKKOS_ENABLE_HIP)
#include <hip/hip_runtime_api.h>
#endif

using namespace LAMMPS_NS;

static_assert(std::is_same<ye3t_kokkos::position_float, double>::value &&
                  std::is_same<ye3t_kokkos::force_float, double>::value &&
                  std::is_same<ye3t_kokkos::energy_float, double>::value,
              "Pair style ye3t/kk requires Kokkos double precision for "
              "positions, forces, and energies");

namespace {

// Host-side launch regions only: never add a fence or read a device result.
// Profilers can correlate these with the existing named kernels and with the
// LAMMPS/MPI timeline. A region's host duration is NOT a GPU elapsed time.
class YE3TProfilePhase {
 public:
  explicit YE3TProfilePhase(const char *name) { Kokkos::Profiling::pushRegion(name); }
  ~YE3TProfilePhase() { Kokkos::Profiling::popRegion(); }
  YE3TProfilePhase(const YE3TProfilePhase &) = delete;
  YE3TProfilePhase &operator=(const YE3TProfilePhase &) = delete;
  void next(const char *name)
  {
    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::pushRegion(name);
  }
};

template <class DeviceType> const char *execution_space_name()
{
#if defined(KOKKOS_ENABLE_CUDA)
  if constexpr (std::is_same_v<DeviceType, Kokkos::Cuda>) return "Cuda";
#elif defined(KOKKOS_ENABLE_HIP)
  if constexpr (std::is_same_v<DeviceType, Kokkos::HIP>) return "HIP";
#elif defined(KOKKOS_ENABLE_SYCL)
  if constexpr (std::is_same_v<DeviceType, Kokkos::Experimental::SYCL>) return "SYCL";
#elif defined(KOKKOS_ENABLE_OPENMPTARGET)
  if constexpr (std::is_same_v<DeviceType, Kokkos::Experimental::OpenMPTarget>)
    return "OpenMPTarget";
#endif
  return "non-device";
}

std::size_t checked_add_bytes(std::size_t first, std::size_t second, const char *name)
{
  if (second > std::numeric_limits<std::size_t>::max() - first)
    throw std::overflow_error(std::string("YE3T Kokkos ") + name + " byte overflow");
  return first + second;
}

std::size_t checked_multiply_bytes(std::size_t first, std::size_t second, const char *name)
{
  if (first != 0 && second > std::numeric_limits<std::size_t>::max() / first)
    throw std::overflow_error(std::string("YE3T Kokkos ") + name + " byte overflow");
  return first * second;
}

constexpr std::size_t YE3T_MAX_PLANNED_WORKSPACE_BYTES =
    (13ULL * 1024ULL * 1024ULL * 1024ULL) / 2ULL;

bool within_lifted_workspace_ceiling(std::size_t plan_bytes, std::size_t fixed_bytes,
                                     std::size_t dynamic_bytes, std::size_t replaced_bytes,
                                     std::size_t proposed_bytes)
{
  if (replaced_bytes > dynamic_bytes)
    throw std::logic_error("YE3T Kokkos dynamic-byte accounting underflow");
  const std::size_t proposed_dynamic = checked_add_bytes(
      dynamic_bytes - replaced_bytes, proposed_bytes, "lifted proposed dynamic storage");
  const std::size_t proposed_resident = checked_add_bytes(
      checked_add_bytes(plan_bytes, fixed_bytes, "lifted proposed resident storage"),
      proposed_dynamic, "lifted proposed resident storage");
  return proposed_resident <= YE3T_MAX_PLANNED_WORKSPACE_BYTES;
}

enum YE3TDeviceStatus : int {
  YE3T_STATUS_ZERO_RADIUS = 1,
  YE3T_STATUS_SPLINE_INTERVAL = 2,
  YE3T_STATUS_CSR_MISMATCH = 4,
  YE3T_STATUS_DENSITY_LIMIT = 8,
  YE3T_STATUS_NONFINITE_GEOMETRY = 16,
  YE3T_STATUS_NONFINITE_SOURCE = 32,
  YE3T_STATUS_NONFINITE_READOUT = 64,
  YE3T_STATUS_NONFINITE_VJP = 128,
  YE3T_STATUS_NONFINITE_TAGGED_FORWARD = 256,
  YE3T_STATUS_NONFINITE_TAGGED_READOUT = 512,
  YE3T_STATUS_NONFINITE_TAGGED_VJP = 1024
};

constexpr int YE3T_NEIGHBOR_MAJOR_SOURCE_WORK_LIMIT = 256;
constexpr int YE3T_NEIGHBOR_MAJOR_SOURCE_TEAM_SIZE = 32;
constexpr int YE3T_NEIGHBOR_MAJOR_SOURCE_PADDING_LIMIT = 2;

struct YE3TRankDeviceRecord {
  int world_rank = -1;
  int world_size = 0;
  int local_rank = -1;
  int local_size = 0;
  int launcher_local_rank = -1;
  int device_ordinal = -1;
  int gpu_aware_mpi = 0;
  int runtime_version = 0;
  int driver_version = 0;
  unsigned long long total_device_bytes = 0;
  char hostname[MPI_MAX_PROCESSOR_NAME]{};
  char execution_space[16]{};
  char uuid[48]{};
  char uuid_status[16]{};
  char pci_bus_id[32]{};
  char device_class_hash[65]{};
  char visible_mask_hash[65]{};
};

struct YE3TCapacityRecord {
  int world_rank = -1;
  int configured_chunk = 0;
  int local_inum = 0;
  int requested_chunk = 0;
  int effective_chunk = 0;
  int center_capacity = 0;
  int edge_capacity = 0;
  int reallocations_before = 0;
  int reallocations_after = 0;
  int center_reduced = 0;
  int edge_reductions = 0;
  int exact_chunk_required = 0;
  unsigned long long dynamic_bytes = 0;
  unsigned long long resident_bytes = 0;
  unsigned long long free_device_bytes = 0;
  unsigned long long total_device_bytes = 0;
};

template <std::size_t Size> void store_text(char (&destination)[Size], const std::string &value)
{
  std::snprintf(destination, Size, "%s", value.c_str());
}

std::string visible_device_mask_hash(bool cuda_backend)
{
  std::string signature;
  const char *cuda_names[] = {"CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"};
  const char *hip_names[] = {"ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"};
  const char **names = cuda_backend ? cuda_names : hip_names;
  const int count = cuda_backend ? 2 : 3;
  for (int index = 0; index < count; ++index) {
    const char *value = std::getenv(names[index]);
    signature += names[index];
    signature += "=";
    signature += value == nullptr ? "<unset>" : value;
    signature += ";";
  }
  return YE3T_LAMMPS::sha256_string(signature);
}

int launcher_local_rank()
{
  const char *names[] = {"OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID",   "SLURM_LOCALID",
                         "MV2_COMM_WORLD_LOCAL_RANK",  "PALS_LOCAL_RANKID", "PMI_LOCAL_RANK"};
  int selected = -1;
  for (const char *name : names) {
    const char *value = std::getenv(name);
    if (value == nullptr || *value == '\0') continue;
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 0 || parsed > std::numeric_limits<int>::max())
      return -2;
    if (selected >= 0 && selected != static_cast<int>(parsed)) return -2;
    selected = static_cast<int>(parsed);
  }
  return selected;
}

std::string format_device_uuid(const unsigned char *bytes)
{
  std::ostringstream stream;
  stream << "GPU-" << std::hex << std::setfill('0');
  for (int index = 0; index < 16; ++index) {
    if (index == 4 || index == 6 || index == 8 || index == 10) stream << '-';
    stream << std::setw(2) << static_cast<unsigned int>(bytes[index]);
  }
  return stream.str();
}

KOKKOS_INLINE_FUNCTION
void set_device_status(int *status, int flag)
{
  Kokkos::atomic_fetch_or(status, flag);
}

struct DeviceComplex {
  double real = 0.0;
  double imaginary = 0.0;
};

KOKKOS_INLINE_FUNCTION
DeviceComplex multiply(DeviceComplex left, DeviceComplex right)
{
  return {left.real * right.real - left.imaginary * right.imaginary,
          left.real * right.imaginary + left.imaginary * right.real};
}

KOKKOS_INLINE_FUNCTION
DeviceComplex multiply_coefficient(DeviceComplex coefficient, DeviceComplex value)
{
  if (coefficient.imaginary == 0.0)
    return {coefficient.real * value.real, coefficient.real * value.imaginary};
  return multiply(coefficient, value);
}

KOKKOS_INLINE_FUNCTION
DeviceComplex conjugate(DeviceComplex value)
{
  return {value.real, -value.imaginary};
}

KOKKOS_INLINE_FUNCTION
DeviceComplex integer_power(DeviceComplex base, int exponent)
{
  DeviceComplex value{1.0, 0.0};
  while (exponent > 0) {
    if ((exponent & 1) != 0) value = multiply(value, base);
    exponent >>= 1;
    if (exponent > 0) base = multiply(base, base);
  }
  return value;
}

KOKKOS_INLINE_FUNCTION
double integer_power(double base, int exponent)
{
  double result = 1.0;
  double factor = base;
  int remaining = exponent;
  while (remaining > 0) {
    if (remaining & 1) result *= factor;
    factor *= factor;
    remaining >>= 1;
  }
  return result;
}

struct SplineValue {
  double value = 0.0;
  double derivative = 0.0;
  bool valid = true;
};

struct SplineScalarValue {
  double value = 0.0;
  bool valid = true;
};

template <class RealView>
KOKKOS_INLINE_FUNCTION SplineValue evaluate_spline(const RealView &coefficients, int table_offset,
                                                   int function_count, int interval_count,
                                                   double cutoff, int function, double radius)
{
  if (radius >= cutoff) return {};
  const double scale = static_cast<double>(interval_count) / cutoff;
  const double scaled = radius * scale;
  const int interval = static_cast<int>(Kokkos::floor(scaled));
  if (interval <= 0) return {0.0, 0.0, false};
  const double local = scaled - static_cast<double>(interval);
  const double local2 = local * local;
  const int coefficient = table_offset + (interval * function_count + function) * 4;
  const double c0 = coefficients(coefficient);
  const double c1 = coefficients(coefficient + 1);
  const double c2 = coefficients(coefficient + 2);
  const double c3 = coefficients(coefficient + 3);
  return {c0 + c1 * local + c2 * local2 + c3 * local2 * local,
          (c1 + 2.0 * c2 * local + 3.0 * c3 * local2) * scale, true};
}

template <class RealView>
KOKKOS_INLINE_FUNCTION SplineScalarValue evaluate_spline_value(const RealView &coefficients,
                                                               int table_offset, int function_count,
                                                               int interval_count, double cutoff,
                                                               int function, double radius)
{
  if (radius >= cutoff) return {};
  const double scale = static_cast<double>(interval_count) / cutoff;
  const double scaled = radius * scale;
  const int interval = static_cast<int>(Kokkos::floor(scaled));
  if (interval <= 0) return {0.0, false};
  const double local = scaled - static_cast<double>(interval);
  const double local2 = local * local;
  const int coefficient = table_offset + (interval * function_count + function) * 4;
  const double c0 = coefficients(coefficient);
  const double c1 = coefficients(coefficient + 1);
  const double c2 = coefficients(coefficient + 2);
  const double c3 = coefficients(coefficient + 3);
  return {c0 + c1 * local + c2 * local2 + c3 * local2 * local, true};
}

template <class RealView>
KOKKOS_INLINE_FUNCTION double polynomial_value(const RealView &plan, int offset, int count,
                                               double argument)
{
  double value = 0.0;
  for (int coefficient = count; coefficient > 0; --coefficient)
    value = value * argument + plan(offset + coefficient - 1);
  return value;
}

struct HarmonicValue {
  double real = 0.0;
  double imaginary = 0.0;
  double gradient_real[3] = {0.0, 0.0, 0.0};
  double gradient_imaginary[3] = {0.0, 0.0, 0.0};
  bool valid = true;
};

struct HarmonicScalarValue {
  double real = 0.0;
  double imaginary = 0.0;
  bool valid = true;
};

template <class DeviceType>
KOKKOS_INLINE_FUNCTION HarmonicScalarValue
evaluate_harmonic_value(const YE3T_LAMMPS::YE3TKokkosAngularViews<DeviceType> &angular,
                        int component, const double unit[3])
{
  HarmonicScalarValue output;
  int degree = 0;
  while (degree <= angular.maximum_angular_momentum && component >= (degree + 1) * (degree + 2) / 2)
    ++degree;
  if (degree > angular.maximum_angular_momentum) {
    output.valid = false;
    return output;
  }
  const int magnetic = component - degree * (degree + 1) / 2;
  if (magnetic < 0 || magnetic > degree) {
    output.valid = false;
    return output;
  }

  const int degree_count = angular.maximum_angular_momentum + 1;
  const int order_count = degree_count + 1;
  const int normalization_offset = degree_count * order_count * degree_count;
  const int degree_offset = degree * order_count * degree_count;
  const double normalization =
      angular.plan(normalization_offset + degree * degree_count + magnetic);

  if (magnetic == 0) {
    output.real =
        normalization * polynomial_value(angular.plan, degree_offset, degree + 1, unit[2]);
    return output;
  }

  double power_real = 1.0;
  double power_imaginary = 0.0;
  for (int order = 1; order <= magnetic; ++order) {
    const double next_real = power_real * unit[0] - power_imaginary * unit[1];
    const double next_imaginary = power_real * unit[1] + power_imaginary * unit[0];
    power_real = next_real;
    power_imaginary = next_imaginary;
  }
  const int coefficient_offset = degree_offset + magnetic * degree_count;
  const double polynomial =
      polynomial_value(angular.plan, coefficient_offset, degree - magnetic + 1, unit[2]);
  const double sign = magnetic % 2 == 0 ? 1.0 : -1.0;
  const double value_scale = sign * normalization * polynomial;
  output.real = value_scale * power_real;
  output.imaginary = value_scale * power_imaginary;
  return output;
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION HarmonicValue
evaluate_harmonic(const YE3T_LAMMPS::YE3TKokkosAngularViews<DeviceType> &angular, int component,
                  const double unit[3], double radius)
{
  HarmonicValue output;
  if (!(radius > 1.0e-14)) {
    output.valid = false;
    return output;
  }

  int degree = 0;
  while (degree <= angular.maximum_angular_momentum && component >= (degree + 1) * (degree + 2) / 2)
    ++degree;
  if (degree > angular.maximum_angular_momentum) {
    output.valid = false;
    return output;
  }
  const int magnetic = component - degree * (degree + 1) / 2;
  if (magnetic < 0 || magnetic > degree) {
    output.valid = false;
    return output;
  }

  const int degree_count = angular.maximum_angular_momentum + 1;
  const int order_count = degree_count + 1;
  const int normalization_offset = degree_count * order_count * degree_count;
  const int degree_offset = degree * order_count * degree_count;
  const double normalization =
      angular.plan(normalization_offset + degree * degree_count + magnetic);

  double unit_gradient_real[3] = {0.0, 0.0, 0.0};
  double unit_gradient_imaginary[3] = {0.0, 0.0, 0.0};
  if (magnetic == 0) {
    output.real =
        normalization * polynomial_value(angular.plan, degree_offset, degree + 1, unit[2]);
    unit_gradient_real[2] = normalization *
        polynomial_value(angular.plan, degree_offset + degree_count, degree > 0 ? degree : 1,
                         unit[2]);
  } else {
    double power_real = 1.0;
    double power_imaginary = 0.0;
    double previous_real = 1.0;
    double previous_imaginary = 0.0;
    for (int order = 1; order <= magnetic; ++order) {
      previous_real = power_real;
      previous_imaginary = power_imaginary;
      const double next_real = power_real * unit[0] - power_imaginary * unit[1];
      const double next_imaginary = power_real * unit[1] + power_imaginary * unit[0];
      power_real = next_real;
      power_imaginary = next_imaginary;
    }
    const int coefficient_offset = degree_offset + magnetic * degree_count;
    const double polynomial =
        polynomial_value(angular.plan, coefficient_offset, degree - magnetic + 1, unit[2]);
    const double polynomial_z =
        polynomial_value(angular.plan, coefficient_offset + degree_count,
                         degree > magnetic ? degree - magnetic : 1, unit[2]);
    const double sign = magnetic % 2 == 0 ? 1.0 : -1.0;
    const double value_scale = sign * normalization * polynomial;
    const double xy_gradient_scale = value_scale * magnetic;
    const double z_gradient_scale = sign * normalization * polynomial_z;
    output.real = value_scale * power_real;
    output.imaginary = value_scale * power_imaginary;
    unit_gradient_real[0] = xy_gradient_scale * previous_real;
    unit_gradient_real[1] = -xy_gradient_scale * previous_imaginary;
    unit_gradient_real[2] = z_gradient_scale * power_real;
    unit_gradient_imaginary[0] = xy_gradient_scale * previous_imaginary;
    unit_gradient_imaginary[1] = xy_gradient_scale * previous_real;
    unit_gradient_imaginary[2] = z_gradient_scale * power_imaginary;
  }

  double radial_real = 0.0;
  double radial_imaginary = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    radial_real += unit_gradient_real[axis] * unit[axis];
    radial_imaginary += unit_gradient_imaginary[axis] * unit[axis];
  }
  for (int axis = 0; axis < 3; ++axis) {
    output.gradient_real[axis] = (unit_gradient_real[axis] - unit[axis] * radial_real) / radius;
    output.gradient_imaginary[axis] =
        (unit_gradient_imaginary[axis] - unit[axis] * radial_imaginary) / radius;
  }
  return output;
}

template <class DeviceType> struct CountEdges {
  using AT = ArrayTypes<DeviceType>;
  typename ye3t_kokkos::ArrayAliases<DeviceType>::t_positions_randomread positions;
  typename AT::t_int_1d_randomread types;
  typename AT::t_neighbors_2d neighbors;
  typename AT::t_int_1d_randomread ilist;
  typename AT::t_int_1d_randomread neighbor_counts;
  Kokkos::View<const int *, DeviceType> type_to_species;
  YE3T_LAMMPS::YE3TKokkosBondViews<DeviceType> bonds;
  Kokkos::View<std::int64_t *, DeviceType> counts;
  Kokkos::View<int *, DeviceType> status;
  int chunk_begin = 0;

  // No host maximum is needed by every source policy.
  KOKKOS_INLINE_FUNCTION
  void operator()(int lane) const
  {
    int unused_maximum = 0;
    (*this)(lane, unused_maximum);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(int lane, int &maximum_count) const
  {
    const int atom_i = ilist(chunk_begin + lane);
    const int central = type_to_species(types(atom_i));
    std::int64_t count = 0;
    if (central >= 0) {
      const int neighbor_count = neighbor_counts(atom_i);
      for (int jj = 0; jj < neighbor_count; ++jj) {
        const int atom_j = neighbors(atom_i, jj) & NEIGHMASK;
        const int neighbor = type_to_species(types(atom_j));
        if (neighbor < 0) continue;
        const int bond = central * bonds.species_count + neighbor;
        const double edge_x = positions(atom_j, 0) - positions(atom_i, 0);
        const double edge_y = positions(atom_j, 1) - positions(atom_i, 1);
        const double edge_z = positions(atom_j, 2) - positions(atom_i, 2);
        const double radius_squared = edge_x * edge_x + edge_y * edge_y + edge_z * edge_z;
        if (!Kokkos::isfinite(edge_x) || !Kokkos::isfinite(edge_y) || !Kokkos::isfinite(edge_z) ||
            !Kokkos::isfinite(radius_squared)) {
          set_device_status(status.data(), YE3T_STATUS_NONFINITE_GEOMETRY);
          continue;
        }
        if (radius_squared <= 0.0) {
          set_device_status(status.data(), YE3T_STATUS_ZERO_RADIUS);
          continue;
        }
        const double cutoff = bonds.cutoffs(bond);
        if (radius_squared < cutoff * cutoff) ++count;
      }
    }
    counts(lane) = count;
    if (count > maximum_count) maximum_count = static_cast<int>(count);
  }
};

template <class DeviceType> using ScanEdges = YE3T_LAMMPS::KokkosScanEdgesSummary<DeviceType>;

template <class DeviceType> struct FillEdges {
  using AT = ArrayTypes<DeviceType>;
  typename ye3t_kokkos::ArrayAliases<DeviceType>::t_positions_randomread positions;
  typename AT::t_int_1d_randomread types;
  typename AT::t_neighbors_2d neighbors;
  typename AT::t_int_1d_randomread ilist;
  typename AT::t_int_1d_randomread neighbor_counts;
  Kokkos::View<const int *, DeviceType> type_to_species;
  YE3T_LAMMPS::YE3TKokkosBondViews<DeviceType> bonds;
  Kokkos::View<const std::int64_t *, DeviceType> offsets;
  Kokkos::View<int *, DeviceType> edge_centers;
  Kokkos::View<int *, DeviceType> edge_neighbors;
  Kokkos::View<int *, DeviceType> edge_bonds;
  Kokkos::View<double *, DeviceType> edge_radius;
  Kokkos::View<double *, DeviceType> edge_unit;
  Kokkos::View<int *, DeviceType> status;
  int chunk_begin = 0;
  int edge_capacity = 0;

  KOKKOS_INLINE_FUNCTION
  void operator()(int lane) const
  {
    const int atom_i = ilist(chunk_begin + lane);
    const int central = type_to_species(types(atom_i));
    std::int64_t edge = offsets(lane);
    if (central >= 0) {
      const int neighbor_count = neighbor_counts(atom_i);
      for (int jj = 0; jj < neighbor_count; ++jj) {
        const int atom_j = neighbors(atom_i, jj) & NEIGHMASK;
        const int neighbor = type_to_species(types(atom_j));
        if (neighbor < 0) continue;
        const int bond = central * bonds.species_count + neighbor;
        const double edge_x = positions(atom_j, 0) - positions(atom_i, 0);
        const double edge_y = positions(atom_j, 1) - positions(atom_i, 1);
        const double edge_z = positions(atom_j, 2) - positions(atom_i, 2);
        const double radius_squared = edge_x * edge_x + edge_y * edge_y + edge_z * edge_z;
        const double cutoff = bonds.cutoffs(bond);
        if (radius_squared > 0.0 && radius_squared < cutoff * cutoff) {
          const int edge_index = static_cast<int>(edge);
          const double radius = Kokkos::sqrt(radius_squared);
          edge_centers(edge_index) = lane;
          edge_neighbors(edge_index) = atom_j;
          edge_bonds(edge_index) = bond;
          edge_radius(edge_index) = radius;
          edge_unit(edge_index) = edge_x / radius;
          edge_unit(edge_capacity + edge_index) = edge_y / radius;
          edge_unit(2 * edge_capacity + edge_index) = edge_z / radius;
          ++edge;
        }
      }
    }
    if (edge != offsets(lane + 1)) set_device_status(status.data(), YE3T_STATUS_CSR_MISMATCH);
  }
};

template <class DeviceType> struct CountLiftedEdges {
  using AT = ArrayTypes<DeviceType>;
  typename ye3t_kokkos::ArrayAliases<DeviceType>::t_positions_randomread positions;
  typename AT::t_int_1d_randomread types;
  typename AT::t_neighbors_2d neighbors;
  typename AT::t_int_1d_randomread ilist;
  typename AT::t_int_1d_randomread neighbor_counts;
  Kokkos::View<const int *, DeviceType> type_to_species;
  Kokkos::View<std::int64_t *, DeviceType> counts;
  Kokkos::View<int *, DeviceType> status;
  double cutoff = 0.0;
  int chunk_begin = 0;

  // No host maximum is needed by every source policy.
  KOKKOS_INLINE_FUNCTION
  void operator()(int lane) const
  {
    int unused_maximum = 0;
    (*this)(lane, unused_maximum);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(int lane, int &maximum_count) const
  {
    const int atom_i = ilist(chunk_begin + lane);
    std::int64_t count = 0;
    if (type_to_species(types(atom_i)) >= 0) {
      const int neighbor_count = neighbor_counts(atom_i);
      for (int jj = 0; jj < neighbor_count; ++jj) {
        const int atom_j = neighbors(atom_i, jj) & NEIGHMASK;
        if (type_to_species(types(atom_j)) < 0) continue;
        const double dx = positions(atom_j, 0) - positions(atom_i, 0);
        const double dy = positions(atom_j, 1) - positions(atom_i, 1);
        const double dz = positions(atom_j, 2) - positions(atom_i, 2);
        const double radius_squared = dx * dx + dy * dy + dz * dz;
        if (!Kokkos::isfinite(dx) || !Kokkos::isfinite(dy) || !Kokkos::isfinite(dz) ||
            !Kokkos::isfinite(radius_squared)) {
          set_device_status(status.data(), YE3T_STATUS_NONFINITE_GEOMETRY);
          continue;
        }
        if (radius_squared < cutoff * cutoff) ++count;
      }
    }
    counts(lane) = count;
    if (count > maximum_count) maximum_count = static_cast<int>(count);
  }
};

template <class DeviceType> struct FillLiftedEdges {
  using AT = ArrayTypes<DeviceType>;
  typename ye3t_kokkos::ArrayAliases<DeviceType>::t_positions_randomread positions;
  typename AT::t_int_1d_randomread types;
  typename AT::t_neighbors_2d neighbors;
  typename AT::t_int_1d_randomread ilist;
  typename AT::t_int_1d_randomread neighbor_counts;
  Kokkos::View<const int *, DeviceType> type_to_species;
  Kokkos::View<const std::int64_t *, DeviceType> offsets;
  Kokkos::View<int *, DeviceType> edge_centers;
  Kokkos::View<int *, DeviceType> edge_neighbors;
  Kokkos::View<int *, DeviceType> edge_bonds;
  Kokkos::View<double *, DeviceType> edge_radius;
  Kokkos::View<double *, DeviceType> edge_unit;
  Kokkos::View<int *, DeviceType> status;
  double cutoff = 0.0;
  int chunk_begin = 0;
  int edge_capacity = 0;

  KOKKOS_INLINE_FUNCTION
  void operator()(int lane) const
  {
    const int atom_i = ilist(chunk_begin + lane);
    const int central = type_to_species(types(atom_i));
    std::int64_t edge = offsets(lane);
    if (central >= 0) {
      const int neighbor_count = neighbor_counts(atom_i);
      for (int jj = 0; jj < neighbor_count; ++jj) {
        const int atom_j = neighbors(atom_i, jj) & NEIGHMASK;
        const int neighbor = type_to_species(types(atom_j));
        if (neighbor < 0) continue;
        const double dx = positions(atom_j, 0) - positions(atom_i, 0);
        const double dy = positions(atom_j, 1) - positions(atom_i, 1);
        const double dz = positions(atom_j, 2) - positions(atom_i, 2);
        const double radius_squared = dx * dx + dy * dy + dz * dz;
        if (!Kokkos::isfinite(dx) || !Kokkos::isfinite(dy) || !Kokkos::isfinite(dz) ||
            !Kokkos::isfinite(radius_squared)) {
          set_device_status(status.data(), YE3T_STATUS_NONFINITE_GEOMETRY);
          continue;
        }
        if (radius_squared >= 0.0 && radius_squared < cutoff * cutoff) {
          const int edge_index = static_cast<int>(edge);
          const double radius = Kokkos::sqrt(radius_squared);
          edge_centers(edge_index) = lane;
          edge_neighbors(edge_index) = atom_j;
          edge_bonds(edge_index) = neighbor;
          edge_radius(edge_index) = radius;
          edge_unit(edge_index) = radius > 0.0 ? dx / radius : 0.0;
          edge_unit(edge_capacity + edge_index) = radius > 0.0 ? dy / radius : 0.0;
          edge_unit(2 * edge_capacity + edge_index) = radius > 0.0 ? dz / radius : 0.0;
          ++edge;
        }
      }
    }
    if (edge != offsets(lane + 1)) set_device_status(status.data(), YE3T_STATUS_CSR_MISMATCH);
  }
};

template <class DeviceType> struct BuildLiftedDirectQSource {
  YE3T_LAMMPS::LiftedCauchyKokkosViews<DeviceType> plan;
  Kokkos::View<const std::int64_t *, DeviceType> offsets;
  Kokkos::View<const int *, DeviceType> edge_bonds;
  Kokkos::View<const double *, DeviceType> edge_radius;
  Kokkos::View<const double *, DeviceType> edge_unit;
  Kokkos::View<double *, DeviceType> source;
  Kokkos::View<int *, DeviceType> status;
  int center_count = 0;
  int center_capacity = 0;
  int edge_capacity = 0;

  KOKKOS_INLINE_FUNCTION
  void operator()(std::int64_t iteration) const
  {
    const int row = iteration / center_count;
    const int lane = iteration - row * center_count;
    int neighbor_species = 0;
    while (neighbor_species + 1 < plan.species_count &&
           row >= plan.species_row_offsets(neighbor_species + 1))
      ++neighbor_species;
    double accumulated[3] = {0.0, 0.0, 0.0};
    bool invalid = false;
    const std::int64_t coefficient_begin = plan.polynomial_offsets(row);
    const std::int64_t coefficient_end = plan.polynomial_offsets(row + 1);
    for (std::int64_t edge_64 = offsets(lane); edge_64 < offsets(lane + 1); ++edge_64) {
      const int edge = static_cast<int>(edge_64);
      if (edge_bonds(edge) != neighbor_species) continue;
      const double coordinate = edge_radius(edge) / plan.cutoff;
      double polynomial = plan.polynomial_coefficients(coefficient_end - 1);
      for (std::int64_t reverse = coefficient_end - 1; reverse > coefficient_begin; --reverse) {
        polynomial = polynomial * coordinate + plan.polynomial_coefficients(reverse - 1);
        invalid = invalid || !Kokkos::isfinite(polynomial);
      }
      const double envelope = 1.0 - coordinate;
      const double radial = envelope * envelope * polynomial;
      invalid = invalid || !Kokkos::isfinite(coordinate) || !Kokkos::isfinite(radial);
      accumulated[0] += radial * coordinate * edge_unit(edge);
      accumulated[1] += radial * coordinate * edge_unit(2 * edge_capacity + edge);
      accumulated[2] -= radial * coordinate * edge_unit(edge_capacity + edge);
      invalid = invalid || !Kokkos::isfinite(accumulated[0]) || !Kokkos::isfinite(accumulated[1]) ||
          !Kokkos::isfinite(accumulated[2]);
    }
    if (invalid) set_device_status(status.data(), YE3T_STATUS_NONFINITE_SOURCE);
    const std::int64_t output = plan.row_source_offsets(row);
    for (int component = 0; component < 3; ++component)
      source[(output + component) * center_capacity + lane] = invalid ? 0.0
                                                                      : accumulated[component];
  }
};

template <class DeviceType> struct LiftedSparseForwardAdjoint {
  using AT = ArrayTypes<DeviceType>;
  typename AT::t_int_1d_randomread types;
  typename AT::t_int_1d_randomread ilist;
  Kokkos::View<const int *, DeviceType> type_to_species;
  YE3T_LAMMPS::LiftedCauchyKokkosViews<DeviceType> plan;
  Kokkos::View<const double *, DeviceType> source;
  Kokkos::View<double *, DeviceType> source_adjoint;
  Kokkos::View<double *, DeviceType> workspace;
  Kokkos::View<double *, DeviceType> atomic_energies;
  Kokkos::View<int *, DeviceType> status;
  int chunk_begin = 0;
  int center_capacity = 0;

  KOKKOS_INLINE_FUNCTION
  void operator()(int lane) const
  {
    for (std::int64_t variable = 0; variable < plan.source_variable_count; ++variable)
      source_adjoint[variable * center_capacity + lane] = 0.0;
    const int atom_i = ilist(chunk_begin + lane);
    const int head = type_to_species(types(atom_i));
    if (head < 0 || head >= plan.species_count) {
      atomic_energies(lane) = 0.0;
      return;
    }
    double energy = plan.head_offsets(head);
    bool invalid = !Kokkos::isfinite(energy);
    const std::int64_t term_begin = plan.head_term_offsets(head);
    const std::int64_t term_end = plan.head_term_offsets(head + 1);
    const int suffix_base = plan.maximum_factor_count + 1;
    for (std::int64_t term = term_begin; term < term_end; ++term) {
      const std::int64_t factor_begin = plan.term_factor_offsets(term);
      const std::int64_t factor_end = plan.term_factor_offsets(term + 1);
      const int factor_count = static_cast<int>(factor_end - factor_begin);
      workspace(lane) = 1.0;
      for (int local = 0; local < factor_count; ++local) {
        const std::int64_t factor = factor_begin + local;
        const std::int64_t variable = plan.factor_indices(factor);
        const int exponent = plan.factor_exponents(factor);
        const double value = source[variable * center_capacity + lane];
        workspace((local + 1) * center_capacity + lane) =
            workspace(local * center_capacity + lane) * integer_power(value, exponent);
        invalid = invalid || !Kokkos::isfinite(value) ||
            !Kokkos::isfinite(workspace((local + 1) * center_capacity + lane));
      }
      workspace((suffix_base + factor_count) * center_capacity + lane) = 1.0;
      for (int local = factor_count; local > 0; --local) {
        const std::int64_t factor = factor_begin + local - 1;
        const std::int64_t variable = plan.factor_indices(factor);
        const int exponent = plan.factor_exponents(factor);
        const double value = source[variable * center_capacity + lane];
        workspace((suffix_base + local - 1) * center_capacity + lane) =
            integer_power(value, exponent) *
            workspace((suffix_base + local) * center_capacity + lane);
        invalid = invalid || !Kokkos::isfinite(value) ||
            !Kokkos::isfinite(workspace((suffix_base + local - 1) * center_capacity + lane));
      }
      const double coefficient = plan.term_coefficients(term);
      energy += coefficient * workspace(factor_count * center_capacity + lane);
      invalid = invalid || !Kokkos::isfinite(coefficient) || !Kokkos::isfinite(energy);
      for (int local = 0; local < factor_count; ++local) {
        const std::int64_t factor = factor_begin + local;
        const std::int64_t variable = plan.factor_indices(factor);
        const int exponent = plan.factor_exponents(factor);
        const double value = source[variable * center_capacity + lane];
        source_adjoint[variable * center_capacity + lane] += coefficient *
            workspace(local * center_capacity + lane) * static_cast<double>(exponent) *
            integer_power(value, exponent - 1) *
            workspace((suffix_base + local + 1) * center_capacity + lane);
        invalid = invalid || !Kokkos::isfinite(source_adjoint[variable * center_capacity + lane]);
      }
    }
    if (invalid) {
      set_device_status(status.data(), YE3T_STATUS_NONFINITE_READOUT);
      for (std::int64_t variable = 0; variable < plan.source_variable_count; ++variable)
        source_adjoint[variable * center_capacity + lane] = 0.0;
      atomic_energies(lane) = 0.0;
      return;
    }
    atomic_energies(lane) = energy;
  }
};

template <class DeviceType> struct LiftedDirectQSourceVJP {
  YE3T_LAMMPS::LiftedCauchyKokkosViews<DeviceType> plan;
  Kokkos::View<const int *, DeviceType> edge_centers;
  Kokkos::View<const int *, DeviceType> edge_bonds;
  Kokkos::View<const double *, DeviceType> edge_radius;
  Kokkos::View<const double *, DeviceType> edge_unit;
  Kokkos::View<const double *, DeviceType> source_adjoint;
  Kokkos::View<double *, DeviceType> edge_gradient;
  Kokkos::View<int *, DeviceType> status;
  int center_capacity = 0;
  int edge_capacity = 0;

  KOKKOS_INLINE_FUNCTION
  void operator()(int edge) const
  {
    const int lane = edge_centers(edge);
    const int neighbor_species = edge_bonds(edge);
    const double direction[3] = {edge_unit(edge), edge_unit(edge_capacity + edge),
                                 edge_unit(2 * edge_capacity + edge)};
    const double coordinate = edge_radius(edge) / plan.cutoff;
    const double solid[3] = {coordinate * direction[0], coordinate * direction[2],
                             -coordinate * direction[1]};
    double gradient[3] = {0.0, 0.0, 0.0};
    bool invalid = !Kokkos::isfinite(coordinate);
    const int row_begin = plan.species_row_offsets(neighbor_species);
    const int row_end = plan.species_row_offsets(neighbor_species + 1);
    for (int row = row_begin; row < row_end; ++row) {
      const std::int64_t coefficient_begin = plan.polynomial_offsets(row);
      const std::int64_t coefficient_end = plan.polynomial_offsets(row + 1);
      double polynomial = plan.polynomial_coefficients(coefficient_end - 1);
      double polynomial_derivative = 0.0;
      for (std::int64_t reverse = coefficient_end - 1; reverse > coefficient_begin; --reverse) {
        polynomial_derivative = polynomial_derivative * coordinate + polynomial;
        polynomial = polynomial * coordinate + plan.polynomial_coefficients(reverse - 1);
        invalid = invalid || !Kokkos::isfinite(polynomial) ||
            !Kokkos::isfinite(polynomial_derivative);
      }
      const double envelope = 1.0 - coordinate;
      const double radial = envelope * envelope * polynomial;
      const double radial_derivative =
          -2.0 * envelope * polynomial + envelope * envelope * polynomial_derivative;
      invalid = invalid || !Kokkos::isfinite(radial) || !Kokkos::isfinite(radial_derivative);
      const std::int64_t output = plan.row_source_offsets(row);
      for (int component = 0; component < 3; ++component) {
        const double seed = source_adjoint[(output + component) * center_capacity + lane];
        invalid = invalid || !Kokkos::isfinite(seed);
        for (int axis = 0; axis < 3; ++axis) {
          double component_map = 0.0;
          if ((component == 0 && axis == 0) || (component == 1 && axis == 2))
            component_map = 1.0;
          else if (component == 2 && axis == 1)
            component_map = -1.0;
          gradient[axis] += seed *
              (radial * component_map + radial_derivative * solid[component] * direction[axis]) /
              plan.cutoff;
          invalid = invalid || !Kokkos::isfinite(gradient[axis]);
        }
      }
    }
    if (invalid) set_device_status(status.data(), YE3T_STATUS_NONFINITE_VJP);
    for (int axis = 0; axis < 3; ++axis)
      edge_gradient(axis * edge_capacity + edge) = invalid ? 0.0 : gradient[axis];
  }
};

// Tagged-Cauchy device kernels. CountLiftedEdges/ScanEdges/FillLiftedEdges
// (above) and ScatterForceEnergyVirial (below) are shared with the lifted
// family: both are family-agnostic (a single scalar cutoff, edge_bonds =
// neighbor species index, and generic edge_gradient_/atomic_energies_
// consumption respectively). The three algorithm passes below are one
// RangePolicy kernel each: "one thread per center" for passes 1-2 and "one
// thread per edge" for pass 3.

// Pass 1: for every edge of center `lane`, compute phi_real (no gradient --
// need_gradient=false) and accumulate the real densities and real moments
// (CPU: ye3t_tagged_cauchy_cpu.cpp's forward accumulation loop).
template <class DeviceType> struct TaggedBuildDensityMoment {
  YE3T_LAMMPS::TaggedCauchyKokkosViews<DeviceType> plan;
  Kokkos::View<const std::int64_t *, DeviceType> offsets;
  Kokkos::View<const int *, DeviceType> edge_bonds;
  Kokkos::View<const double *, DeviceType> edge_radius;
  Kokkos::View<const double *, DeviceType> edge_unit;
  Kokkos::View<double *, DeviceType> density;
  Kokkos::View<double *, DeviceType> moment;
  Kokkos::View<int *, DeviceType> status;
  int center_capacity = 0;
  int edge_capacity = 0;

  KOKKOS_INLINE_FUNCTION
  void operator()(int lane) const
  {
    for (int dk = 0; dk < plan.density_key_count; ++dk) density[dk * center_capacity + lane] = 0.0;
    for (int mk = 0; mk < plan.moment_key_count; ++mk) moment[mk * center_capacity + lane] = 0.0;
    bool invalid = false;
    for (std::int64_t edge_64 = offsets(lane); edge_64 < offsets(lane + 1); ++edge_64) {
      const int edge = static_cast<int>(edge_64);
      const int species = edge_bonds(edge);
      const double radius = edge_radius(edge);
      if (plan.physical_image_v3 && radius == 0.0) {
        set_device_status(status.data(), YE3T_STATUS_ZERO_RADIUS);
        invalid = true;
        continue;
      }
      const double dx = radius * edge_unit(edge);
      const double dy = radius * edge_unit(edge_capacity + edge);
      const double dz = radius * edge_unit(2 * edge_capacity + edge);
      double phi_real[YE3T_LAMMPS::TAGGED_KOKKOS_MAX_COMPONENTS];
      YE3T_LAMMPS::tagged_kokkos_edge_components<DeviceType>(plan, species, dx, dy, dz, false,
                                                             phi_real, nullptr, nullptr, nullptr);
      for (int dk = 0; dk < plan.density_key_count; ++dk) {
        const double value = phi_real[plan.density_flat_index(dk)];
        invalid = invalid || !Kokkos::isfinite(value);
        density[dk * center_capacity + lane] += value;
      }
      for (int mk = 0; mk < plan.moment_key_count; ++mk) {
        const std::int64_t factor_begin = plan.moment_factor_offsets(mk);
        const std::int64_t factor_end = plan.moment_factor_offsets(mk + 1);
        double product = 1.0;
        for (std::int64_t factor = factor_begin; factor < factor_end; ++factor)
          product *= phi_real[plan.moment_factor_flat_index(factor)];
        invalid = invalid || !Kokkos::isfinite(product);
        moment[mk * center_capacity + lane] += product;
      }
    }
    if (invalid) set_device_status(status.data(), YE3T_STATUS_NONFINITE_TAGGED_FORWARD);
  }
};

// Pass 2: evaluate the load-time species-folded readout and its term-level
// adjoints dE/dA, dE/dM. Legacy V2 coefficients have exact-zero work removed
// and identical commutative monomials coalesced. Prefix/suffix product-rule
// distribution remains division-free, including repeated factors and zeros.
// No atomics are needed because one lane owns every adjoint slot for a center.
template <class DeviceType> struct TaggedTermForwardAdjoint {
  using AT = ArrayTypes<DeviceType>;
  typename AT::t_int_1d_randomread types;
  typename AT::t_int_1d_randomread ilist;
  Kokkos::View<const int *, DeviceType> type_to_species;
  YE3T_LAMMPS::TaggedCauchyKokkosViews<DeviceType> plan;
  Kokkos::View<const std::int64_t *, DeviceType> offsets;
  Kokkos::View<const double *, DeviceType> density;
  Kokkos::View<const double *, DeviceType> moment;
  Kokkos::View<double *, DeviceType> density_adjoint;
  Kokkos::View<double *, DeviceType> moment_adjoint;
  Kokkos::View<double *, DeviceType> atomic_energies;
  Kokkos::View<int *, DeviceType> status;
  int chunk_begin = 0;
  int center_capacity = 0;

  KOKKOS_INLINE_FUNCTION
  void operator()(int lane) const
  {
    for (int dk = 0; dk < plan.density_key_count; ++dk)
      density_adjoint[dk * center_capacity + lane] = 0.0;
    for (int mk = 0; mk < plan.moment_key_count; ++mk)
      moment_adjoint[mk * center_capacity + lane] = 0.0;
    const int atom_i = ilist(chunk_begin + lane);
    const int species = type_to_species(types(atom_i));
    if (species < 0 || species >= plan.species_count) {
      atomic_energies(lane) = 0.0;
      set_device_status(status.data(), YE3T_STATUS_NONFINITE_TAGGED_READOUT);
      return;
    }
    const std::int64_t z = offsets(lane + 1) - offsets(lane);
    double energy = plan.offsets(species);
    bool invalid = !Kokkos::isfinite(energy);
    const std::int64_t term_begin = plan.species_term_offsets(species);
    const std::int64_t term_end = plan.species_term_offsets(species + 1);
    for (std::int64_t t = term_begin; t < term_end; ++t) {
      const std::int64_t density_begin = plan.term_density_offsets(t);
      const std::int64_t density_end = plan.term_density_offsets(t + 1);
      const int density_count = static_cast<int>(density_end - density_begin);
      double density_values[YE3T_LAMMPS::TAGGED_KOKKOS_MAX_TERM_FACTORS];
      double density_product = 1.0;
      for (int k = 0; k < density_count; ++k) {
        density_values[k] =
            density[plan.term_density_factor_index(density_begin + k) * center_capacity + lane];
        density_product *= density_values[k];
      }
      const std::int64_t moment_begin = plan.term_moment_offsets(t);
      const std::int64_t moment_end = plan.term_moment_offsets(t + 1);
      const int moment_count = static_cast<int>(moment_end - moment_begin);
      double moment_values[YE3T_LAMMPS::TAGGED_KOKKOS_MAX_TERM_FACTORS];
      double moment_product = 1.0;
      for (int k = 0; k < moment_count; ++k) {
        moment_values[k] =
            moment[plan.term_moment_factor_index(moment_begin + k) * center_capacity + lane];
        moment_product *= moment_values[k];
      }
      const int p = plan.term_p(t);
      double free_count = 1.0;
      if (!plan.physical_image_v3) {
        free_count = 0.0;
        if (z >= plan.tag_count) {
          free_count = 1.0;
          const int count = plan.tag_count - p;
          const double top = static_cast<double>(z) - static_cast<double>(p);
          for (int i = 0; i < count; ++i) free_count *= (top - static_cast<double>(i));
        }
      }
      const double coefficient = plan.term_coefficient(t);
      const double term_value = coefficient * density_product * free_count * moment_product;
      energy += term_value;
      invalid = invalid || !Kokkos::isfinite(term_value) || !Kokkos::isfinite(energy);

      if (!plan.physical_image_v3 && free_count != 0.0) {
        const double seed = coefficient * free_count;
        if (density_count > 0) {
          double prefix[YE3T_LAMMPS::TAGGED_KOKKOS_MAX_TERM_FACTORS];
          prefix[0] = 1.0;
          for (int k = 1; k < density_count; ++k) prefix[k] = prefix[k - 1] * density_values[k - 1];
          double suffix = 1.0;
          const double gA = seed * moment_product;
          for (int k = density_count - 1; k >= 0; --k) {
            const int idx = plan.term_density_factor_index(density_begin + k);
            density_adjoint[idx * center_capacity + lane] += gA * prefix[k] * suffix;
            suffix *= density_values[k];
          }
        }
        if (moment_count > 0) {
          double prefix[YE3T_LAMMPS::TAGGED_KOKKOS_MAX_TERM_FACTORS];
          prefix[0] = 1.0;
          for (int k = 1; k < moment_count; ++k) prefix[k] = prefix[k - 1] * moment_values[k - 1];
          double suffix = 1.0;
          const double gM = seed * density_product;
          for (int k = moment_count - 1; k >= 0; --k) {
            const int idx = plan.term_moment_factor_index(moment_begin + k);
            moment_adjoint[idx * center_capacity + lane] += gM * prefix[k] * suffix;
            suffix *= moment_values[k];
          }
        }
      }
    }
    if (plan.physical_image_v3) {
      for (int t = 0; t < plan.adjoint_term_count; ++t) {
        double value = plan.folded_adjoint_coefficient(species * plan.adjoint_term_count + t);
        const std::int64_t begin = plan.adjoint_remaining_offsets(t);
        const std::int64_t end = plan.adjoint_remaining_offsets(t + 1);
        for (std::int64_t factor = begin; factor < end; ++factor)
          value *= density[plan.adjoint_remaining_source_index(factor) * center_capacity + lane];
        density_adjoint[plan.adjoint_source_index(t) * center_capacity + lane] += value;
        invalid = invalid || !Kokkos::isfinite(value);
      }
    }
    if (invalid) {
      set_device_status(status.data(), YE3T_STATUS_NONFINITE_TAGGED_READOUT);
      atomic_energies(lane) = 0.0;
      for (int dk = 0; dk < plan.density_key_count; ++dk)
        density_adjoint[dk * center_capacity + lane] = 0.0;
      for (int mk = 0; mk < plan.moment_key_count; ++mk)
        moment_adjoint[mk * center_capacity + lane] = 0.0;
      return;
    }
    atomic_energies(lane) = energy;
  }
};

// Pass 3: recompute phi_real and its Cartesian gradient for edge `edge`,
// form dE/dphi from dE/dA (direct) and dE/dM (product rule over that
// moment's factors, multiplicities handled by the same prefix/suffix
// distribution as pass 2 and as the CPU path), and contract with the
// gradient to get dE_i/d(edge_vector). Feeds edge_gradient_, which
// ScatterForceEnergyVirial (shared with the lifted path, unmodified)
// scatters to forces/energy/virial exactly as it already does for lifted.
template <class DeviceType, bool DensityOnly = false> struct TaggedEdgeVJP {
  YE3T_LAMMPS::TaggedCauchyKokkosViews<DeviceType> plan;
  Kokkos::View<const int *, DeviceType> edge_centers;
  Kokkos::View<const int *, DeviceType> edge_bonds;
  Kokkos::View<const double *, DeviceType> edge_radius;
  Kokkos::View<const double *, DeviceType> edge_unit;
  Kokkos::View<const double *, DeviceType> density_adjoint;
  Kokkos::View<const double *, DeviceType> moment_adjoint;
  Kokkos::View<double *, DeviceType> edge_gradient;
  Kokkos::View<int *, DeviceType> status;
  int center_capacity = 0;
  int edge_capacity = 0;

  KOKKOS_INLINE_FUNCTION
  void operator()(int edge) const
  {
    const int lane = edge_centers(edge);
    const int species = edge_bonds(edge);
    const double radius = edge_radius(edge);
    if (plan.physical_image_v3 && radius == 0.0) {
      set_device_status(status.data(), YE3T_STATUS_ZERO_RADIUS);
      edge_gradient(edge) = 0.0;
      edge_gradient(edge_capacity + edge) = 0.0;
      edge_gradient(2 * edge_capacity + edge) = 0.0;
      return;
    }
    const double dx = radius * edge_unit(edge);
    const double dy = radius * edge_unit(edge_capacity + edge);
    const double dz = radius * edge_unit(2 * edge_capacity + edge);
    if constexpr (DensityOnly) {
      double seed[YE3T_LAMMPS::TAGGED_KOKKOS_MAX_COMPONENTS];
      for (int c = 0; c < plan.total_component_count; ++c) seed[c] = 0.0;
      bool invalid = false;
      for (int dk = 0; dk < plan.density_key_count; ++dk) {
        const double value = density_adjoint[dk * center_capacity + lane];
        invalid = invalid || !Kokkos::isfinite(value);
        seed[plan.density_flat_index(dk)] += value;
      }
      double gradient[3];
      YE3T_LAMMPS::gpu_tagged_edge_vjp<YE3T_LAMMPS::TaggedKokkosMath>(plan, species, dx, dy, dz,
                                                                      seed, gradient);
      for (int axis = 0; axis < 3; ++axis) invalid = invalid || !Kokkos::isfinite(gradient[axis]);
      if (invalid) set_device_status(status.data(), YE3T_STATUS_NONFINITE_TAGGED_VJP);
      for (int axis = 0; axis < 3; ++axis)
        edge_gradient(axis * edge_capacity + edge) = invalid ? 0.0 : gradient[axis];
    } else {
      double phi_real[YE3T_LAMMPS::TAGGED_KOKKOS_MAX_COMPONENTS];
      double phi_grad_x[YE3T_LAMMPS::TAGGED_KOKKOS_MAX_COMPONENTS];
      double phi_grad_y[YE3T_LAMMPS::TAGGED_KOKKOS_MAX_COMPONENTS];
      double phi_grad_z[YE3T_LAMMPS::TAGGED_KOKKOS_MAX_COMPONENTS];
      YE3T_LAMMPS::tagged_kokkos_edge_components<DeviceType>(
          plan, species, dx, dy, dz, true, phi_real, phi_grad_x, phi_grad_y, phi_grad_z);

      double seed[YE3T_LAMMPS::TAGGED_KOKKOS_MAX_COMPONENTS];
      for (int c = 0; c < plan.total_component_count; ++c) seed[c] = 0.0;
      bool invalid = false;
      for (int dk = 0; dk < plan.density_key_count; ++dk) {
        const double adjoint = density_adjoint[dk * center_capacity + lane];
        invalid = invalid || !Kokkos::isfinite(adjoint);
        seed[plan.density_flat_index(dk)] += adjoint;
      }
      for (int mk = 0; mk < plan.moment_key_count; ++mk) {
        const double adjoint = moment_adjoint[mk * center_capacity + lane];
        invalid = invalid || !Kokkos::isfinite(adjoint);
        if (adjoint == 0.0) continue;
        const std::int64_t factor_begin = plan.moment_factor_offsets(mk);
        const std::int64_t factor_end = plan.moment_factor_offsets(mk + 1);
        const int count = static_cast<int>(factor_end - factor_begin);
        double values[YE3T_LAMMPS::TAGGED_KOKKOS_MAX_TERM_FACTORS];
        for (int k = 0; k < count; ++k)
          values[k] = phi_real[plan.moment_factor_flat_index(factor_begin + k)];
        double prefix[YE3T_LAMMPS::TAGGED_KOKKOS_MAX_TERM_FACTORS];
        prefix[0] = 1.0;
        for (int k = 1; k < count; ++k) prefix[k] = prefix[k - 1] * values[k - 1];
        double suffix = 1.0;
        for (int k = count - 1; k >= 0; --k) {
          const int flat = plan.moment_factor_flat_index(factor_begin + k);
          seed[flat] += adjoint * prefix[k] * suffix;
          suffix *= values[k];
        }
      }
      double gx = 0.0, gy = 0.0, gz = 0.0;
      for (int c = 0; c < plan.total_component_count; ++c) {
        if (seed[c] == 0.0) continue;
        gx += seed[c] * phi_grad_x[c];
        gy += seed[c] * phi_grad_y[c];
        gz += seed[c] * phi_grad_z[c];
      }
      invalid = invalid || !Kokkos::isfinite(gx) || !Kokkos::isfinite(gy) || !Kokkos::isfinite(gz);
      if (invalid) set_device_status(status.data(), YE3T_STATUS_NONFINITE_TAGGED_VJP);
      edge_gradient(edge) = invalid ? 0.0 : gx;
      edge_gradient(edge_capacity + edge) = invalid ? 0.0 : gy;
      edge_gradient(2 * edge_capacity + edge) = invalid ? 0.0 : gz;
    }
  }
};

template <class DeviceType> struct BuildOrdinarySource {
  using AT = ArrayTypes<DeviceType>;
  typename ye3t_kokkos::ArrayAliases<DeviceType>::t_positions_randomread positions;
  typename AT::t_int_1d_randomread types;
  typename AT::t_int_1d_randomread ilist;
  Kokkos::View<const int *, DeviceType> type_to_species;
  YE3T_LAMMPS::YE3TKokkosSpeciesViews<DeviceType> species;
  YE3T_LAMMPS::YE3TKokkosBondViews<DeviceType> bonds;
  YE3T_LAMMPS::YE3TKokkosAngularViews<DeviceType> angular;
  Kokkos::View<const std::int64_t *, DeviceType> offsets;
  Kokkos::View<const int *, DeviceType> edge_neighbors;
  Kokkos::View<const int *, DeviceType> edge_bonds;
  Kokkos::View<double *, DeviceType> source_real;
  Kokkos::View<double *, DeviceType> source_imaginary;
  Kokkos::View<int *, DeviceType> status;
  int chunk_begin = 0;
  int center_count = 0;
  int center_capacity = 0;

  KOKKOS_INLINE_FUNCTION
  void operator()(int iteration) const
  {
    const int component = iteration / center_count;
    const int lane = iteration - component * center_count;
    const int atom_i = ilist(chunk_begin + lane);
    const int central = type_to_species(types(atom_i));
    const int output = component * center_capacity + lane;
    double accumulated_real = 0.0;
    double accumulated_imaginary = 0.0;
    if (central < 0 ||
        component >= species.source_offsets(central + 1) - species.source_offsets(central)) {
      source_real(output) = 0.0;
      source_imaginary(output) = 0.0;
      return;
    }

    for (std::int64_t edge_64 = offsets(lane); edge_64 < offsets(lane + 1); ++edge_64) {
      const int edge = static_cast<int>(edge_64);
      const int atom_j = edge_neighbors(edge);
      const int bond = edge_bonds(edge);
      const double edge_x = positions(atom_j, 0) - positions(atom_i, 0);
      const double edge_y = positions(atom_j, 1) - positions(atom_i, 1);
      const double edge_z = positions(atom_j, 2) - positions(atom_i, 2);
      const double radius = Kokkos::sqrt(edge_x * edge_x + edge_y * edge_y + edge_z * edge_z);
      const int interval_count = bonds.interval_counts(bond);
      const double cutoff = bonds.cutoffs(bond);

      for (int term = bonds.radial_map_offsets(bond); term < bonds.radial_map_offsets(bond + 1);
           ++term) {
        if (bonds.radial_channel_outputs(term) != component) continue;
        const SplineValue radial = evaluate_spline(
            bonds.radial_splines, bonds.radial_spline_offsets(bond), bonds.radial_base_counts(bond),
            interval_count, cutoff, bonds.radial_channel_indices(term), radius);
        if (!radial.valid) {
          set_device_status(status.data(), YE3T_STATUS_SPLINE_INTERVAL);
          continue;
        }
        accumulated_real += radial.value;
      }

      const int contracted_width = bonds.radial_counts(bond) * (bonds.angular_maxima(bond) + 1);
      for (int term = bonds.angular_map_offsets(bond); term < bonds.angular_map_offsets(bond + 1);
           ++term) {
        if (bonds.angular_channel_outputs(term) != component) continue;
        const SplineValue radial = evaluate_spline(
            bonds.contracted_splines, bonds.contracted_spline_offsets(bond), contracted_width,
            interval_count, cutoff, bonds.contracted_channel_indices(term), radius);
        const double direction[3] = {edge_x / radius, edge_y / radius, edge_z / radius};
        const HarmonicValue harmonic =
            evaluate_harmonic(angular, bonds.angular_channel_indices(term), direction, radius);
        if (!radial.valid || !harmonic.valid) {
          set_device_status(status.data(),
                            radial.valid ? YE3T_STATUS_ZERO_RADIUS : YE3T_STATUS_SPLINE_INTERVAL);
          continue;
        }
        accumulated_real += radial.value * harmonic.real;
        accumulated_imaginary += radial.value * harmonic.imaginary;
      }
    }
    source_real(output) = accumulated_real;
    source_imaginary(output) = accumulated_imaginary;
  }
};

template <class DeviceType> struct ZeroOrdinarySource {
  Kokkos::View<double *, DeviceType> source_real;
  Kokkos::View<double *, DeviceType> source_imaginary;
  int center_count = 0;
  int center_capacity = 0;

  KOKKOS_INLINE_FUNCTION
  void operator()(int iteration) const
  {
    const int component = iteration / center_count;
    const int lane = iteration - component * center_count;
    const int output = component * center_capacity + lane;
    source_real(output) = 0.0;
    source_imaginary(output) = 0.0;
  }
};

template <class DeviceType> struct BuildOrdinarySourceByEdge {
  using TeamPolicy = Kokkos::TeamPolicy<DeviceType>;
  using TeamMember = typename TeamPolicy::member_type;
  using ScratchView = Kokkos::View<double *, typename DeviceType::scratch_memory_space,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  YE3T_LAMMPS::YE3TKokkosBondViews<DeviceType> bonds;
  YE3T_LAMMPS::YE3TKokkosAngularViews<DeviceType> angular;
  Kokkos::View<const int *, DeviceType> edge_centers;
  Kokkos::View<const int *, DeviceType> edge_bonds;
  Kokkos::View<const double *, DeviceType> edge_radius;
  Kokkos::View<const double *, DeviceType> edge_unit;
  Kokkos::View<double *, DeviceType> source_real;
  Kokkos::View<double *, DeviceType> source_imaginary;
  Kokkos::View<double *, DeviceType> radial_derivative;
  Kokkos::View<double *, DeviceType> contracted_value;
  Kokkos::View<double *, DeviceType> contracted_derivative;
  Kokkos::View<int *, DeviceType> status;
  int center_capacity = 0;
  int edge_capacity = 0;
  int maximum_radial_base_count = 0;
  int maximum_contracted_width = 0;
  int maximum_angular_width = 0;
  int scratch_value_count = 0;

  static std::size_t scratch_bytes(int value_count) { return ScratchView::shmem_size(value_count); }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamMember &team) const
  {
    const int edge = team.league_rank();
    const int lane = edge_centers(edge);
    const int bond = edge_bonds(edge);
    const double radius = edge_radius(edge);
    if (!(radius > 1.0e-14)) {
      set_device_status(status.data(), YE3T_STATUS_ZERO_RADIUS);
      return;
    }
    const double direction[3] = {edge_unit(edge), edge_unit(edge_capacity + edge),
                                 edge_unit(2 * edge_capacity + edge)};
    const int interval_count = bonds.interval_counts(bond);
    const double cutoff = bonds.cutoffs(bond);
    const int radial_base_count = bonds.radial_base_counts(bond);
    const int contracted_width = bonds.radial_counts(bond) * (bonds.angular_maxima(bond) + 1);
    const int contracted_offset = maximum_radial_base_count;
    const int harmonic_real_offset = contracted_offset + maximum_contracted_width;
    const int harmonic_imaginary_offset = harmonic_real_offset + maximum_angular_width;
    ScratchView scratch(team.team_scratch(0), scratch_value_count);

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, radial_base_count), [&](int function) {
      const SplineValue value =
          evaluate_spline(bonds.radial_splines, bonds.radial_spline_offsets(bond),
                          radial_base_count, interval_count, cutoff, function, radius);
      if (!value.valid) set_device_status(status.data(), YE3T_STATUS_SPLINE_INTERVAL);
      scratch(function) = value.value;
      const std::size_t cache = static_cast<std::size_t>(function) * edge_capacity + edge;
      radial_derivative(cache) = value.derivative;
    });
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, bonds.active_contracted_offsets(bond),
                                                 bonds.active_contracted_offsets(bond + 1)),
                         [&](int entry) {
                           const int function = bonds.active_contracted_functions(entry);
                           const SplineValue value = evaluate_spline(
                               bonds.contracted_splines, bonds.contracted_spline_offsets(bond),
                               contracted_width, interval_count, cutoff, function, radius);
                           if (!value.valid)
                             set_device_status(status.data(), YE3T_STATUS_SPLINE_INTERVAL);
                           scratch(contracted_offset + function) = value.value;
                           const std::size_t cache =
                               static_cast<std::size_t>(function) * edge_capacity + edge;
                           contracted_value(cache) = value.value;
                           contracted_derivative(cache) = value.derivative;
                         });
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, bonds.vjp_bond_column_offsets(bond),
                                bonds.vjp_bond_column_offsets(bond + 1)),
        [&](int column) {
          YE3T_LAMMPS::GpuHarmonicStream<false, decltype(angular.recurrence)> harmonics(
              angular.recurrence, angular.maximum_angular_momentum, direction);
          for (int group = bonds.vjp_column_group_offsets(column);
               group < bonds.vjp_column_group_offsets(column + 1); ++group) {
            const auto value =
                harmonics.advance(bonds.vjp_group_degrees(group), bonds.vjp_group_orders(group));
            if (!value.valid) set_device_status(status.data(), YE3T_STATUS_ZERO_RADIUS);
            const int component = bonds.vjp_group_harmonics(group);
            scratch(harmonic_real_offset + component) = value.real;
            scratch(harmonic_imaginary_offset + component) = value.imaginary;
          }
        });
    team.team_barrier();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, bonds.radial_map_offsets(bond),
                                                 bonds.radial_map_offsets(bond + 1)),
                         [&](int term) {
                           const int output = bonds.radial_channel_outputs(term);
                           const int function = bonds.radial_channel_indices(term);
                           Kokkos::atomic_add(&source_real(output * center_capacity + lane),
                                              scratch(function));
                         });
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, bonds.angular_map_offsets(bond),
                                bonds.angular_map_offsets(bond + 1)),
        [&](int term) {
          const int output = bonds.angular_channel_outputs(term);
          const int radial = bonds.contracted_channel_indices(term);
          const int harmonic = bonds.angular_channel_indices(term);
          const double radial_value = scratch(contracted_offset + radial);
          Kokkos::atomic_add(&source_real(output * center_capacity + lane),
                             radial_value * scratch(harmonic_real_offset + harmonic));
          Kokkos::atomic_add(&source_imaginary(output * center_capacity + lane),
                             radial_value * scratch(harmonic_imaginary_offset + harmonic));
        });
  }
};

template <class DeviceType> struct BuildOrdinarySourceNeighborMajor {
  using TeamPolicy = Kokkos::TeamPolicy<DeviceType>;
  using TeamMember = typename TeamPolicy::member_type;

  YE3T_LAMMPS::YE3TKokkosBondViews<DeviceType> bonds;
  YE3T_LAMMPS::YE3TKokkosAngularViews<DeviceType> angular;
  Kokkos::View<const std::int64_t *, DeviceType> center_counts;
  Kokkos::View<const std::int64_t *, DeviceType> center_offsets;
  Kokkos::View<const int *, DeviceType> edge_bonds;
  Kokkos::View<const double *, DeviceType> edge_radius;
  Kokkos::View<const double *, DeviceType> edge_unit;
  Kokkos::View<double *, DeviceType> source_real;
  Kokkos::View<double *, DeviceType> source_imaginary;
  Kokkos::View<double *, DeviceType> radial_derivative;
  Kokkos::View<double *, DeviceType> contracted_value;
  Kokkos::View<double *, DeviceType> contracted_derivative;
  Kokkos::View<int *, DeviceType> status;
  int center_count = 0;
  int center_capacity = 0;
  int edge_capacity = 0;
  int center_tile_count = 0;

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamMember &team) const
  {
    const int lane = team.team_rank() + team.team_size() * (team.league_rank() % center_tile_count);
    if (lane >= center_count) return;
    const int neighbor = team.league_rank() / center_tile_count;
    if (neighbor >= center_counts(lane)) return;
    const int edge = static_cast<int>(center_offsets(lane) + neighbor);
    const int bond = edge_bonds(edge);
    const double radius = edge_radius(edge);
    if (!(radius > 1.0e-14)) {
      set_device_status(status.data(), YE3T_STATUS_ZERO_RADIUS);
      return;
    }
    const double direction[3] = {edge_unit(edge), edge_unit(edge_capacity + edge),
                                 edge_unit(2 * edge_capacity + edge)};
    const int interval_count = bonds.interval_counts(bond);
    const double cutoff = bonds.cutoffs(bond);
    const int radial_base_count = bonds.radial_base_counts(bond);
    const int contracted_width = bonds.radial_counts(bond) * (bonds.angular_maxima(bond) + 1);

    for (int term = bonds.radial_map_offsets(bond); term < bonds.radial_map_offsets(bond + 1);
         ++term) {
      const int function = bonds.radial_channel_indices(term);
      const SplineValue value =
          evaluate_spline(bonds.radial_splines, bonds.radial_spline_offsets(bond),
                          radial_base_count, interval_count, cutoff, function, radius);
      if (!value.valid) set_device_status(status.data(), YE3T_STATUS_SPLINE_INTERVAL);
      const std::size_t cache = static_cast<std::size_t>(function) * edge_capacity + edge;
      radial_derivative(cache) = value.derivative;
      const int output = bonds.radial_channel_outputs(term);
      Kokkos::atomic_add(&source_real(output * center_capacity + lane), value.value);
    }

    for (int entry = bonds.active_contracted_offsets(bond);
         entry < bonds.active_contracted_offsets(bond + 1); ++entry) {
      const int function = bonds.active_contracted_functions(entry);
      const SplineValue value =
          evaluate_spline(bonds.contracted_splines, bonds.contracted_spline_offsets(bond),
                          contracted_width, interval_count, cutoff, function, radius);
      if (!value.valid) set_device_status(status.data(), YE3T_STATUS_SPLINE_INTERVAL);
      const std::size_t cache = static_cast<std::size_t>(function) * edge_capacity + edge;
      contracted_value(cache) = value.value;
      contracted_derivative(cache) = value.derivative;
    }

    YE3T_LAMMPS::GpuHarmonicStream<false, decltype(angular.recurrence)> harmonics(
        angular.recurrence, angular.maximum_angular_momentum, direction);
    for (int group = bonds.vjp_bond_group_offsets(bond);
         group < bonds.vjp_bond_group_offsets(bond + 1); ++group) {
      const auto harmonic_value =
          harmonics.advance(bonds.vjp_group_degrees(group), bonds.vjp_group_orders(group));
      if (!harmonic_value.valid) {
        set_device_status(status.data(), YE3T_STATUS_ZERO_RADIUS);
        continue;
      }
      for (int entry = bonds.vjp_group_term_offsets(group);
           entry < bonds.vjp_group_term_offsets(group + 1); ++entry) {
        const int term = bonds.vjp_group_terms(entry);
        const int output = bonds.angular_channel_outputs(term);
        const int radial = bonds.contracted_channel_indices(term);
        const std::size_t cache = static_cast<std::size_t>(radial) * edge_capacity + edge;
        const double radial_value = contracted_value(cache);
        Kokkos::atomic_add(&source_real(output * center_capacity + lane),
                           radial_value * harmonic_value.real);
        Kokkos::atomic_add(&source_imaginary(output * center_capacity + lane),
                           radial_value * harmonic_value.imaginary);
      }
    }
  }
};

template <class DeviceType>
KOKKOS_INLINE_FUNCTION DeviceComplex
read_full_channel(const YE3T_LAMMPS::YE3TKokkosSpeciesViews<DeviceType> &species,
                  const Kokkos::View<const double *, DeviceType> &source_real,
                  const Kokkos::View<const double *, DeviceType> &source_imaginary, int central,
                  int channel, int center_capacity, int lane)
{
  const int global_channel = species.channel_offsets(central) + channel;
  const int source = species.full_channel_sources(global_channel);
  const int transform = species.full_channel_transforms(global_channel);
  DeviceComplex value{source_real(source * center_capacity + lane),
                      source_imaginary(source * center_capacity + lane)};
  if (transform != 0) {
    value.real *= transform;
    value.imaginary *= -transform;
  }
  return value;
}

template <class DeviceType> struct DirectForwardAdjoint {
  using AT = ArrayTypes<DeviceType>;
  typename AT::t_int_1d_randomread types;
  typename AT::t_int_1d_randomread ilist;
  Kokkos::View<const int *, DeviceType> type_to_species;
  YE3T_LAMMPS::YE3TKokkosSpeciesViews<DeviceType> species;
  Kokkos::View<const double *, DeviceType> source_real;
  Kokkos::View<const double *, DeviceType> source_imaginary;
  Kokkos::View<double *, DeviceType> source_adjoint_real;
  Kokkos::View<double *, DeviceType> source_adjoint_imaginary;
  Kokkos::View<double *, DeviceType> dag_real;
  Kokkos::View<double *, DeviceType> dag_imaginary;
  Kokkos::View<double *, DeviceType> dag_adjoint_real;
  Kokkos::View<double *, DeviceType> dag_adjoint_imaginary;
  Kokkos::View<double *, DeviceType> atomic_energies;
  Kokkos::View<int *, DeviceType> status;
  int chunk_begin = 0;
  int center_capacity = 0;

  using value_type = double;

  KOKKOS_INLINE_FUNCTION
  void operator()(int lane, double &maximum_imaginary) const
  {
    const int atom_i = ilist(chunk_begin + lane);
    const int central = type_to_species(types(atom_i));
    if (central < 0) {
      atomic_energies(lane) = 0.0;
      return;
    }
    const int source_count = species.source_offsets(central + 1) - species.source_offsets(central);
    const int power_begin = species.power_offsets(central);
    const int power_count = species.power_offsets(central + 1) - power_begin;
    const int node_begin = species.node_offsets(central);
    const int node_count = species.node_offsets(central + 1) - node_begin;
    const int monomial_begin = species.monomial_offsets(central);
    const int monomial_count = species.monomial_offsets(central + 1) - monomial_begin;
    const int value_count = power_count + node_count;

    for (int value = 0; value < value_count; ++value) {
      dag_adjoint_real(value * center_capacity + lane) = 0.0;
      dag_adjoint_imaginary(value * center_capacity + lane) = 0.0;
    }
    for (int source = 0; source < source_count; ++source) {
      source_adjoint_real(source * center_capacity + lane) = 0.0;
      source_adjoint_imaginary(source * center_capacity + lane) = 0.0;
    }

    for (int power = 0; power < power_count; ++power) {
      const int global_power = power_begin + power;
      const int channel = species.power_channels(global_power);
      const int exponent = species.power_exponents(global_power);
      const DeviceComplex input = read_full_channel(species, source_real, source_imaginary, central,
                                                    channel, center_capacity, lane);
      const DeviceComplex value = integer_power(input, exponent);
      dag_real(power * center_capacity + lane) = value.real;
      dag_imaginary(power * center_capacity + lane) = value.imaginary;
    }
    for (int node = 0; node < node_count; ++node) {
      const int global_node = node_begin + node;
      const int left = species.binary_node_left(global_node);
      const int right = species.binary_node_right(global_node);
      const DeviceComplex left_value{dag_real(left * center_capacity + lane),
                                     dag_imaginary(left * center_capacity + lane)};
      const DeviceComplex right_value{dag_real(right * center_capacity + lane),
                                      dag_imaginary(right * center_capacity + lane)};
      const DeviceComplex value = multiply(left_value, right_value);
      const int output = power_count + node;
      dag_real(output * center_capacity + lane) = value.real;
      dag_imaginary(output * center_capacity + lane) = value.imaginary;
    }

    DeviceComplex density;
    for (int term = 0; term < monomial_count; ++term) {
      const int global_term = monomial_begin + term;
      const int operand = species.monomial_nodes(global_term);
      const double coefficient = species.monomial_coefficients(global_term);
      if (operand < 0) {
        density.real += coefficient;
      } else {
        density.real += coefficient * dag_real(operand * center_capacity + lane);
        density.imaginary += coefficient * dag_imaginary(operand * center_capacity + lane);
        dag_adjoint_real(operand * center_capacity + lane) += coefficient;
      }
    }

    for (int node = node_count - 1; node >= 0; --node) {
      const int output = power_count + node;
      const int global_node = node_begin + node;
      const int left = species.binary_node_left(global_node);
      const int right = species.binary_node_right(global_node);
      const double gradient_real = dag_adjoint_real(output * center_capacity + lane);
      const double gradient_imaginary = dag_adjoint_imaginary(output * center_capacity + lane);
      const double left_real = dag_real(left * center_capacity + lane);
      const double left_imaginary = dag_imaginary(left * center_capacity + lane);
      const double right_real = dag_real(right * center_capacity + lane);
      const double right_imaginary = dag_imaginary(right * center_capacity + lane);
      dag_adjoint_real(left * center_capacity + lane) +=
          gradient_real * right_real + gradient_imaginary * right_imaginary;
      dag_adjoint_imaginary(left * center_capacity + lane) +=
          gradient_imaginary * right_real - gradient_real * right_imaginary;
      dag_adjoint_real(right * center_capacity + lane) +=
          gradient_real * left_real + gradient_imaginary * left_imaginary;
      dag_adjoint_imaginary(right * center_capacity + lane) +=
          gradient_imaginary * left_real - gradient_real * left_imaginary;
    }

    const double embedding_scale = species.embedding_scales(central);
    for (int power = 0; power < power_count; ++power) {
      const int global_power = power_begin + power;
      const int channel = species.power_channels(global_power);
      const int exponent = species.power_exponents(global_power);
      const DeviceComplex input = read_full_channel(species, source_real, source_imaginary, central,
                                                    channel, center_capacity, lane);
      DeviceComplex derivative = integer_power(input, exponent - 1);
      derivative.real *= exponent;
      derivative.imaginary *= exponent;
      const double root_real = dag_adjoint_real(power * center_capacity + lane);
      const double root_imaginary = dag_adjoint_imaginary(power * center_capacity + lane);
      double gradient_real =
          embedding_scale * (root_real * derivative.real + root_imaginary * derivative.imaginary);
      double gradient_imaginary =
          embedding_scale * (root_imaginary * derivative.real - root_real * derivative.imaginary);
      const int global_channel = species.channel_offsets(central) + channel;
      const int source = species.full_channel_sources(global_channel);
      const int transform = species.full_channel_transforms(global_channel);
      if (transform != 0) {
        gradient_real *= transform;
        gradient_imaginary *= -transform;
      }
      source_adjoint_real(source * center_capacity + lane) += gradient_real;
      source_adjoint_imaginary(source * center_capacity + lane) += gradient_imaginary;
    }

    if (density.real >= species.density_safe_limits(central))
      set_device_status(status.data(), YE3T_STATUS_DENSITY_LIMIT);
    atomic_energies(lane) = species.reference_energies(central) + embedding_scale * density.real;
    const double imaginary = Kokkos::abs(density.imaginary);
    if (imaginary > maximum_imaginary) maximum_imaginary = imaginary;
  }
};

template <class DeviceType> struct InitializeOptimizedState {
  using AT = ArrayTypes<DeviceType>;
  using TeamPolicy = Kokkos::TeamPolicy<DeviceType>;
  using MemberType = typename TeamPolicy::member_type;
  typename AT::t_int_1d_randomread types;
  typename AT::t_int_1d_randomread ilist;
  Kokkos::View<const int *, DeviceType> type_to_species;
  YE3T_LAMMPS::YE3TKokkosSpeciesViews<DeviceType> species;
  Kokkos::View<double *, DeviceType> source_adjoint_real;
  Kokkos::View<double *, DeviceType> source_adjoint_imaginary;
  Kokkos::View<double *, DeviceType> total_density_real;
  Kokkos::View<double *, DeviceType> total_density_imaginary;
  Kokkos::View<double *, DeviceType> atomic_energies;
  int chunk_begin = 0;
  int center_capacity = 0;

  KOKKOS_INLINE_FUNCTION
  void operator()(const MemberType &team) const
  {
    const int lane = team.league_rank();
    const int atom_i = ilist(chunk_begin + lane);
    const int central = type_to_species(types(atom_i));
    if (central < 0) {
      if (team.team_rank() == 0) {
        total_density_real(lane) = 0.0;
        total_density_imaginary(lane) = 0.0;
        atomic_energies(lane) = 0.0;
      }
      return;
    }

    const int source_count = species.source_offsets(central + 1) - species.source_offsets(central);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, source_count), [&](const int source) {
      source_adjoint_real(source * center_capacity + lane) = 0.0;
      source_adjoint_imaginary(source * center_capacity + lane) = 0.0;
    });
    if (team.team_rank() == 0) {
      total_density_real(lane) = 0.0;
      total_density_imaginary(lane) = 0.0;
      atomic_energies(lane) = 0.0;
    }
  }
};

template <class DeviceType, bool FinalizeDensity> struct DirectTeamForwardAdjoint {
  using AT = ArrayTypes<DeviceType>;
  using TeamPolicy = Kokkos::TeamPolicy<DeviceType>;
  using MemberType = typename TeamPolicy::member_type;
  typename AT::t_int_1d_randomread types;
  typename AT::t_int_1d_randomread ilist;
  Kokkos::View<const int *, DeviceType> type_to_species;
  YE3T_LAMMPS::YE3TKokkosSpeciesViews<DeviceType> species;
  Kokkos::View<const double *, DeviceType> source_real;
  Kokkos::View<const double *, DeviceType> source_imaginary;
  Kokkos::View<double *, DeviceType> source_adjoint_real;
  Kokkos::View<double *, DeviceType> source_adjoint_imaginary;
  Kokkos::View<double *, DeviceType> dag_real;
  Kokkos::View<double *, DeviceType> dag_imaginary;
  Kokkos::View<double *, DeviceType> dag_adjoint_real;
  Kokkos::View<double *, DeviceType> dag_adjoint_imaginary;
  Kokkos::View<double *, DeviceType> total_density_real;
  Kokkos::View<double *, DeviceType> total_density_imaginary;
  Kokkos::View<double *, DeviceType> atomic_energies;
  Kokkos::View<int *, DeviceType> status;
  int chunk_begin = 0;
  int center_capacity = 0;
  int value_capacity = 0;
  int resident_value_capacity = 0;

  using value_type = double;

  KOKKOS_INLINE_FUNCTION
  void operator()(const MemberType &team) const
  {
    static_assert(!FinalizeDensity, "Final density diagnostics require the reduction overload");
    double unused_maximum = 0.0;
    (*this)(team, unused_maximum);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const MemberType &team, double &maximum_imaginary) const
  {
    const int lane = team.league_rank();
    const int atom_i = ilist(chunk_begin + lane);
    const int central = type_to_species(types(atom_i));
    if (central < 0) {
      if (team.team_rank() == 0) {
        if constexpr (!FinalizeDensity) {
          total_density_real(lane) = 0.0;
          total_density_imaginary(lane) = 0.0;
        }
        atomic_energies(lane) = 0.0;
      }
      return;
    }

    const int source_count = species.source_offsets(central + 1) - species.source_offsets(central);
    const int power_begin = species.power_offsets(central);
    const int power_count = species.power_offsets(central + 1) - power_begin;
    const int node_begin = species.node_offsets(central);
    const int monomial_begin = species.monomial_offsets(central);
    const int monomial_count = species.monomial_offsets(central + 1) - monomial_begin;

    // Team threads vary the NODE, not the center: keep each center's graph
    // contiguous. Small graphs remain in team scratch across forward + reverse;
    // larger graphs use the same contiguous layout in global memory.
    double *local = resident_value_capacity > 0
        ? static_cast<double *>(team.team_shmem().get_shmem(
              static_cast<std::size_t>(4) * resident_value_capacity * sizeof(double)))
        : nullptr;
    const std::size_t base = static_cast<std::size_t>(lane) * value_capacity;
    const YE3T_LAMMPS::GpuStridedDoubles value_r{local ? local : dag_real.data(), 1,
                                                 local ? 0 : base};
    const YE3T_LAMMPS::GpuStridedDoubles value_i{
        local ? local + resident_value_capacity : dag_imaginary.data(), 1, local ? 0 : base};
    const YE3T_LAMMPS::GpuStridedDoubles adjoint_r{
        local ? local + 2 * resident_value_capacity : dag_adjoint_real.data(), 1, local ? 0 : base};
    const YE3T_LAMMPS::GpuStridedDoubles adjoint_i{local ? local + 3 * resident_value_capacity
                                                         : dag_adjoint_imaginary.data(),
                                                   1, local ? 0 : base};
    const int reverse_value_begin = species.dag_value_offsets(central);
    // Every DAG adjoint and source adjoint is overwritten by its owner below.
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, power_count), [&](const int power) {
      const int global_power = power_begin + power;
      const int channel = species.power_channels(global_power);
      const int exponent = species.power_exponents(global_power);
      const DeviceComplex input = read_full_channel(species, source_real, source_imaginary, central,
                                                    channel, center_capacity, lane);
      const DeviceComplex value = integer_power(input, exponent);
      value_r(power) = value.real;
      value_i(power) = value.imaginary;
    });
    team.team_barrier();

    const int level_table_begin = species.node_level_table_offsets(central);
    const int level_count = species.node_level_table_offsets(central + 1) - level_table_begin - 1;
    for (int level = 0; level < level_count; ++level) {
      const int level_begin = species.node_level_offsets(level_table_begin + level);
      const int level_end = species.node_level_offsets(level_table_begin + level + 1);
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, level_end - level_begin),
                           [&](const int level_offset) {
                             const int node = species.node_level_nodes(level_begin + level_offset);
                             const int global_node = node_begin + node;
                             const int left = species.binary_node_left(global_node);
                             const int right = species.binary_node_right(global_node);
                             const DeviceComplex left_value{value_r(left), value_i(left)};
                             const DeviceComplex right_value{value_r(right), value_i(right)};
                             const DeviceComplex value = multiply(left_value, right_value);
                             const int output = power_count + node;
                             value_r(output) = value.real;
                             value_i(output) = value.imaginary;
                           });
      team.team_barrier();
    }

    double density_real = 0.0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, monomial_count),
        [&](const int term, double &sum) {
          const int global_term = monomial_begin + term;
          const int operand = species.monomial_nodes(global_term);
          const double coefficient = species.monomial_coefficients(global_term);
          if (operand < 0) {
            sum += coefficient;
          } else {
            sum += coefficient * value_r(operand);
          }
        },
        density_real);
    double density_imaginary = 0.0;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team, monomial_count),
        [&](const int term, double &sum) {
          const int global_term = monomial_begin + term;
          const int operand = species.monomial_nodes(global_term);
          if (operand >= 0) sum += species.monomial_coefficients(global_term) * value_i(operand);
        },
        density_imaginary);
    team.team_barrier();

    // Parents live at strictly higher levels. Each owner gathers once after
    // those levels complete, so same-child and shared-parent cases need no atomics.
    for (int level = level_count - 1; level >= 0; --level) {
      const int level_begin = species.node_level_offsets(level_table_begin + level);
      const int level_end = species.node_level_offsets(level_table_begin + level + 1);
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team, level_end - level_begin), [&](const int level_offset) {
            const int node = species.node_level_nodes(level_begin + level_offset);
            const int output = power_count + node;
            double ar, ai;
            YE3T_LAMMPS::gather_gpu_dag_adjoint(species, value_r, value_i, adjoint_r, adjoint_i,
                                                reverse_value_begin, output, 1, 0, ar, ai);
            adjoint_r(output) = ar;
            adjoint_i(output) = ai;
          });
      team.team_barrier();
    }
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, power_count), [&](const int power) {
      double ar, ai;
      YE3T_LAMMPS::gather_gpu_dag_adjoint(species, value_r, value_i, adjoint_r, adjoint_i,
                                          reverse_value_begin, power, 1, 0, ar, ai);
      adjoint_r(power) = ar;
      adjoint_i(power) = ai;
    });
    team.team_barrier();

    const double embedding_scale = species.embedding_scales(central);
    // Full channels (including negative-m transforms) may share one source.
    // Gather their power roots by SOURCE, not by power, before writing that row.
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, source_count), [&](const int source) {
      double source_real_sum = 0.0, source_imaginary_sum = 0.0;
      const int global_source = species.source_offsets(central) + source;
      for (int entry = species.dag_source_power_offsets(global_source);
           entry < species.dag_source_power_offsets(global_source + 1); ++entry) {
        const int power = species.dag_source_powers(entry);
        const int global_power = power_begin + power;
        const int channel = species.power_channels(global_power);
        const int exponent = species.power_exponents(global_power);
        DeviceComplex derivative{1.0, 0.0};
        if (exponent != 1) {
          const DeviceComplex input = read_full_channel(species, source_real, source_imaginary,
                                                        central, channel, center_capacity, lane);
          derivative = integer_power(input, exponent - 1);
          derivative.real *= exponent;
          derivative.imaginary *= exponent;
        }
        const double root_real = adjoint_r(power);
        const double root_imaginary = adjoint_i(power);
        double gr =
            embedding_scale * (root_real * derivative.real + root_imaginary * derivative.imaginary);
        double gi =
            embedding_scale * (root_imaginary * derivative.real - root_real * derivative.imaginary);
        const int global_channel = species.channel_offsets(central) + channel;
        const int transform = species.full_channel_transforms(global_channel);
        if (transform != 0) {
          gr *= transform;
          gi *= -transform;
        }
        source_real_sum += gr;
        source_imaginary_sum += gi;
      }
      source_adjoint_real(source * center_capacity + lane) = source_real_sum;
      source_adjoint_imaginary(source * center_capacity + lane) = source_imaginary_sum;
      if (!Kokkos::isfinite(source_real_sum) || !Kokkos::isfinite(source_imaginary_sum))
        set_device_status(status.data(), YE3T_STATUS_NONFINITE_READOUT);
    });
    team.team_barrier();

    if (team.team_rank() == 0) {
      if (!Kokkos::isfinite(density_real) || !Kokkos::isfinite(density_imaginary))
        set_device_status(status.data(), YE3T_STATUS_NONFINITE_READOUT);
      if constexpr (FinalizeDensity) {
        if (density_real >= species.density_safe_limits(central))
          set_device_status(status.data(), YE3T_STATUS_DENSITY_LIMIT);
        atomic_energies(lane) =
            species.reference_energies(central) + embedding_scale * density_real;
        const double imaginary = Kokkos::abs(density_imaginary);
        if (imaginary > maximum_imaginary) maximum_imaginary = imaginary;
      } else {
        total_density_real(lane) = density_real;
        total_density_imaginary(lane) = density_imaginary;
      }
    }
  }
};

template <class DeviceType>
KOKKOS_INLINE_FUNCTION void
add_full_channel_adjoint(const YE3T_LAMMPS::YE3TKokkosSpeciesViews<DeviceType> &species,
                         const Kokkos::View<double *, DeviceType> &source_adjoint_real,
                         const Kokkos::View<double *, DeviceType> &source_adjoint_imaginary,
                         int central, int channel, int center_capacity, int lane,
                         double gradient_real, double gradient_imaginary)
{
  const int global_channel = species.channel_offsets(central) + channel;
  const int source = species.full_channel_sources(global_channel);
  const int transform = species.full_channel_transforms(global_channel);
  if (transform != 0) {
    gradient_real *= transform;
    gradient_imaginary *= -transform;
  }
  Kokkos::atomic_add(&source_adjoint_real(source * center_capacity + lane), gradient_real);
  Kokkos::atomic_add(&source_adjoint_imaginary(source * center_capacity + lane),
                     gradient_imaginary);
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION void
add_owned_full_channel_adjoint(const YE3T_LAMMPS::YE3TKokkosSpeciesViews<DeviceType> &species,
                               const Kokkos::View<double *, DeviceType> &source_adjoint_real,
                               const Kokkos::View<double *, DeviceType> &source_adjoint_imaginary,
                               int central, int channel, int center_capacity, int lane,
                               double gradient_real, double gradient_imaginary)
{
  const int global_channel = species.channel_offsets(central) + channel;
  const int source = species.full_channel_sources(global_channel);
  const int transform = species.full_channel_transforms(global_channel);
  if (transform != 0) {
    gradient_real *= transform;
    gradient_imaginary *= -transform;
  }
  source_adjoint_real(source * center_capacity + lane) += gradient_real;
  source_adjoint_imaginary(source * center_capacity + lane) += gradient_imaginary;
}

template <class DeviceType>
KOKKOS_INLINE_FUNCTION void scalar_power_forward_adjoint_center(
    const YE3T_LAMMPS::YE3TKokkosSpeciesViews<DeviceType> &species,
    const YE3T_LAMMPS::YE3TKokkosScalarViews<DeviceType> &scalar,
    const Kokkos::View<const double *, DeviceType> &source_real,
    const Kokkos::View<const double *, DeviceType> &source_imaginary,
    const Kokkos::View<double *, DeviceType> &source_adjoint_real,
    const Kokkos::View<double *, DeviceType> &source_adjoint_imaginary,
    const Kokkos::View<double *, DeviceType> &value_real,
    const Kokkos::View<double *, DeviceType> &value_imaginary,
    const Kokkos::View<double *, DeviceType> &adjoint_real,
    const Kokkos::View<double *, DeviceType> &adjoint_imaginary,
    const Kokkos::View<double *, DeviceType> &total_density_real,
    const Kokkos::View<double *, DeviceType> &total_density_imaginary, int central,
    int center_capacity, int lane)
{
  const auto program = scalar.species(central);
  if (program.route_count == 0) return;

  for (int value = 0; value < program.value_count; ++value) {
    const int offset = value * center_capacity + lane;
    adjoint_real(offset) = 0.0;
    adjoint_imaginary(offset) = 0.0;
  }

  for (int base_index = 0; base_index < program.base_count; ++base_index) {
    const auto base = scalar.bases(program.base_begin + base_index);
    DeviceComplex value;
    for (int term = 0; term < base.term_count; ++term) {
      const int global_term = base.term_begin + term;
      const DeviceComplex coefficient{scalar.coefficient_real(global_term),
                                      scalar.coefficient_imaginary(global_term)};
      const DeviceComplex left =
          read_full_channel(species, source_real, source_imaginary, central,
                            scalar.left_channels(global_term), center_capacity, lane);
      const DeviceComplex right =
          read_full_channel(species, source_real, source_imaginary, central,
                            scalar.right_channels(global_term), center_capacity, lane);
      const DeviceComplex contribution = multiply(multiply(coefficient, left), right);
      value.real += contribution.real;
      value.imaginary += contribution.imaginary;
    }
    const int destination = base_index * center_capacity + lane;
    value_real(destination) = value.real;
    value_imaginary(destination) = value.imaginary;
  }

  for (int node_index = 0; node_index < program.node_count; ++node_index) {
    const auto node = scalar.nodes(program.node_begin + node_index);
    const int left_offset = node.left_value * center_capacity + lane;
    const int right_offset = node.right_value * center_capacity + lane;
    const DeviceComplex value = multiply({value_real(left_offset), value_imaginary(left_offset)},
                                         {value_real(right_offset), value_imaginary(right_offset)});
    const int destination = (program.base_count + node_index) * center_capacity + lane;
    value_real(destination) = value.real;
    value_imaginary(destination) = value.imaginary;
  }

  DeviceComplex density;
  for (int route_index = 0; route_index < program.route_count; ++route_index) {
    const auto route = scalar.routes(program.route_begin + route_index);
    const int source = route.value_index * center_capacity + lane;
    const DeviceComplex scale{route.scale_real, route.scale_imaginary};
    const DeviceComplex value{value_real(source), value_imaginary(source)};
    const DeviceComplex contribution = multiply(scale, value);
    density.real += contribution.real;
    density.imaginary += contribution.imaginary;
    adjoint_real(source) += route.scale_real;
    adjoint_imaginary(source) -= route.scale_imaginary;
  }
  total_density_real(lane) += density.real;
  total_density_imaginary(lane) += density.imaginary;

  for (int node_index = program.node_count - 1; node_index >= 0; --node_index) {
    const auto node = scalar.nodes(program.node_begin + node_index);
    const int source = (program.base_count + node_index) * center_capacity + lane;
    const int left_offset = node.left_value * center_capacity + lane;
    const int right_offset = node.right_value * center_capacity + lane;
    const DeviceComplex root{adjoint_real(source), adjoint_imaginary(source)};
    const DeviceComplex left{value_real(left_offset), value_imaginary(left_offset)};
    const DeviceComplex right{value_real(right_offset), value_imaginary(right_offset)};
    const DeviceComplex left_root = multiply(root, conjugate(right));
    const DeviceComplex right_root = multiply(root, conjugate(left));
    adjoint_real(left_offset) += left_root.real;
    adjoint_imaginary(left_offset) += left_root.imaginary;
    adjoint_real(right_offset) += right_root.real;
    adjoint_imaginary(right_offset) += right_root.imaginary;
  }

  const double embedding_scale = species.embedding_scales(central);
  for (int base_index = 0; base_index < program.base_count; ++base_index) {
    const auto base = scalar.bases(program.base_begin + base_index);
    const int source = base_index * center_capacity + lane;
    const DeviceComplex root{adjoint_real(source), adjoint_imaginary(source)};
    for (int term = 0; term < base.term_count; ++term) {
      const int global_term = base.term_begin + term;
      const int left_channel = scalar.left_channels(global_term);
      const int right_channel = scalar.right_channels(global_term);
      const DeviceComplex coefficient{scalar.coefficient_real(global_term),
                                      scalar.coefficient_imaginary(global_term)};
      const DeviceComplex left = read_full_channel(species, source_real, source_imaginary, central,
                                                   left_channel, center_capacity, lane);
      const DeviceComplex right = read_full_channel(species, source_real, source_imaginary, central,
                                                    right_channel, center_capacity, lane);
      DeviceComplex left_root = multiply(root, conjugate(multiply(coefficient, right)));
      DeviceComplex right_root = multiply(root, conjugate(multiply(coefficient, left)));
      left_root.real *= embedding_scale;
      left_root.imaginary *= embedding_scale;
      right_root.real *= embedding_scale;
      right_root.imaginary *= embedding_scale;
      add_full_channel_adjoint(species, source_adjoint_real, source_adjoint_imaginary, central,
                               left_channel, center_capacity, lane, left_root.real,
                               left_root.imaginary);
      add_full_channel_adjoint(species, source_adjoint_real, source_adjoint_imaginary, central,
                               right_channel, center_capacity, lane, right_root.real,
                               right_root.imaginary);
    }
  }
}

template <class DeviceType> struct ScalarPowerForwardAdjoint {
  using AT = ArrayTypes<DeviceType>;
  typename AT::t_int_1d_randomread types;
  typename AT::t_int_1d_randomread ilist;
  Kokkos::View<const int *, DeviceType> type_to_species;
  YE3T_LAMMPS::YE3TKokkosSpeciesViews<DeviceType> species;
  YE3T_LAMMPS::YE3TKokkosScalarViews<DeviceType> scalar;
  Kokkos::View<const double *, DeviceType> source_real;
  Kokkos::View<const double *, DeviceType> source_imaginary;
  Kokkos::View<double *, DeviceType> source_adjoint_real;
  Kokkos::View<double *, DeviceType> source_adjoint_imaginary;
  Kokkos::View<double *, DeviceType> value_real;
  Kokkos::View<double *, DeviceType> value_imaginary;
  Kokkos::View<double *, DeviceType> adjoint_real;
  Kokkos::View<double *, DeviceType> adjoint_imaginary;
  Kokkos::View<double *, DeviceType> total_density_real;
  Kokkos::View<double *, DeviceType> total_density_imaginary;
  int chunk_begin = 0;
  int center_capacity = 0;

  KOKKOS_INLINE_FUNCTION
  void operator()(int lane) const
  {
    const int atom_i = ilist(chunk_begin + lane);
    const int central = type_to_species(types(atom_i));
    if (central < 0) return;
    scalar_power_forward_adjoint_center(
        species, scalar, source_real, source_imaginary, source_adjoint_real,
        source_adjoint_imaginary, value_real, value_imaginary, adjoint_real, adjoint_imaginary,
        total_density_real, total_density_imaginary, central, center_capacity, lane);
  }
};

template <class DeviceType>
KOKKOS_INLINE_FUNCTION void coupled_product_forward_adjoint_center(
    const YE3T_LAMMPS::YE3TKokkosSpeciesViews<DeviceType> &species,
    const YE3T_LAMMPS::YE3TKokkosCoupledViews<DeviceType> &coupled,
    const Kokkos::View<const double *, DeviceType> &source_real,
    const Kokkos::View<const double *, DeviceType> &source_imaginary,
    const Kokkos::View<double *, DeviceType> &source_adjoint_real,
    const Kokkos::View<double *, DeviceType> &source_adjoint_imaginary,
    const Kokkos::View<double *, DeviceType> &value_real,
    const Kokkos::View<double *, DeviceType> &value_imaginary,
    const Kokkos::View<double *, DeviceType> &adjoint_real,
    const Kokkos::View<double *, DeviceType> &adjoint_imaginary,
    const Kokkos::View<double *, DeviceType> &total_density_real,
    const Kokkos::View<double *, DeviceType> &total_density_imaginary, int central,
    int center_capacity, int lane)
{
  const auto program = coupled.species(central);
  if (program.plan_count == 0) return;

  DeviceComplex density;
  const double embedding_scale = species.embedding_scales(central);
  for (int plan_index = 0; plan_index < program.plan_count; ++plan_index) {
    const auto plan = coupled.plans(program.plan_begin + plan_index);
    for (int component = 0; component < plan.total_component_count; ++component) {
      const int offset = component * center_capacity + lane;
      adjoint_real(offset) = 0.0;
      adjoint_imaginary(offset) = 0.0;
    }

    for (int node_index = 0; node_index < plan.node_count; ++node_index) {
      const auto node = coupled.nodes(plan.node_begin + node_index);
      if (node.leaf_offset >= 0) {
        for (int component = 0; component < node.dimension; ++component) {
          const int channel =
              coupled.leaf_input_channels(plan.leaf_begin + node.leaf_offset + component);
          const DeviceComplex value = read_full_channel(species, source_real, source_imaginary,
                                                        central, channel, center_capacity, lane);
          const int destination = (node.component_offset + component) * center_capacity + lane;
          value_real(destination) = value.real;
          value_imaginary(destination) = value.imaginary;
        }
        continue;
      }

      for (int component = 0; component < node.dimension; ++component) {
        const int destination = (node.component_offset + component) * center_capacity + lane;
        value_real(destination) = 0.0;
        value_imaginary(destination) = 0.0;
      }
      for (int term = 0; term < node.coefficient_count; ++term) {
        const int coefficient = plan.coefficient_begin + node.coefficient_begin + term;
        const int left = coupled.coefficient_left_components(coefficient) * center_capacity + lane;
        const int right =
            coupled.coefficient_right_components(coefficient) * center_capacity + lane;
        const int output =
            coupled.coefficient_output_components(coefficient) * center_capacity + lane;
        const DeviceComplex product = multiply({value_real(left), value_imaginary(left)},
                                               {value_real(right), value_imaginary(right)});
        const double scale = coupled.coefficient_values(coefficient);
        value_real(output) += scale * product.real;
        value_imaginary(output) += scale * product.imaginary;
      }
    }

    for (int readout_index = 0; readout_index < plan.readout_count; ++readout_index) {
      const int readout = plan.readout_begin + readout_index;
      const int source = coupled.readout_components(readout) * center_capacity + lane;
      const double coefficient = coupled.readout_coefficients(readout);
      density.real += coefficient * value_real(source);
      density.imaginary += coefficient * value_imaginary(source);
      adjoint_real(source) += coefficient;
    }

    for (int node_index = plan.node_count - 1; node_index >= 0; --node_index) {
      const auto node = coupled.nodes(plan.node_begin + node_index);
      if (node.leaf_offset >= 0) continue;
      for (int term = node.coefficient_count - 1; term >= 0; --term) {
        const int coefficient = plan.coefficient_begin + node.coefficient_begin + term;
        const int left = coupled.coefficient_left_components(coefficient) * center_capacity + lane;
        const int right =
            coupled.coefficient_right_components(coefficient) * center_capacity + lane;
        const int output =
            coupled.coefficient_output_components(coefficient) * center_capacity + lane;
        const double scale = coupled.coefficient_values(coefficient);
        const DeviceComplex root{scale * adjoint_real(output), scale * adjoint_imaginary(output)};
        const DeviceComplex left_value{value_real(left), value_imaginary(left)};
        const DeviceComplex right_value{value_real(right), value_imaginary(right)};
        const DeviceComplex left_root = multiply(root, conjugate(right_value));
        const DeviceComplex right_root = multiply(root, conjugate(left_value));
        adjoint_real(left) += left_root.real;
        adjoint_imaginary(left) += left_root.imaginary;
        adjoint_real(right) += right_root.real;
        adjoint_imaginary(right) += right_root.imaginary;
      }
    }

    for (int node_index = 0; node_index < plan.node_count; ++node_index) {
      const auto node = coupled.nodes(plan.node_begin + node_index);
      if (node.leaf_offset < 0) continue;
      for (int component = 0; component < node.dimension; ++component) {
        const int source = (node.component_offset + component) * center_capacity + lane;
        const int channel =
            coupled.leaf_input_channels(plan.leaf_begin + node.leaf_offset + component);
        add_full_channel_adjoint(species, source_adjoint_real, source_adjoint_imaginary, central,
                                 channel, center_capacity, lane,
                                 embedding_scale * adjoint_real(source),
                                 embedding_scale * adjoint_imaginary(source));
      }
    }
  }

  total_density_real(lane) += density.real;
  total_density_imaginary(lane) += density.imaginary;
}

template <class DeviceType> struct CoupledProductForwardAdjoint {
  using AT = ArrayTypes<DeviceType>;
  typename AT::t_int_1d_randomread types;
  typename AT::t_int_1d_randomread ilist;
  Kokkos::View<const int *, DeviceType> type_to_species;
  YE3T_LAMMPS::YE3TKokkosSpeciesViews<DeviceType> species;
  YE3T_LAMMPS::YE3TKokkosCoupledViews<DeviceType> coupled;
  Kokkos::View<const double *, DeviceType> source_real;
  Kokkos::View<const double *, DeviceType> source_imaginary;
  Kokkos::View<double *, DeviceType> source_adjoint_real;
  Kokkos::View<double *, DeviceType> source_adjoint_imaginary;
  Kokkos::View<double *, DeviceType> value_real;
  Kokkos::View<double *, DeviceType> value_imaginary;
  Kokkos::View<double *, DeviceType> adjoint_real;
  Kokkos::View<double *, DeviceType> adjoint_imaginary;
  Kokkos::View<double *, DeviceType> total_density_real;
  Kokkos::View<double *, DeviceType> total_density_imaginary;
  int chunk_begin = 0;
  int center_capacity = 0;

  KOKKOS_INLINE_FUNCTION
  void operator()(int lane) const
  {
    const int atom_i = ilist(chunk_begin + lane);
    const int central = type_to_species(types(atom_i));
    if (central < 0) return;
    coupled_product_forward_adjoint_center(
        species, coupled, source_real, source_imaginary, source_adjoint_real,
        source_adjoint_imaginary, value_real, value_imaginary, adjoint_real, adjoint_imaginary,
        total_density_real, total_density_imaginary, central, center_capacity, lane);
  }
};

struct ScalarPowerMathProbeResult {
  double forward_error = 0.0;
  double adjoint_dot_error = 0.0;
  double zero_error = 0.0;
  bool finite = true;
};

template <class DeviceType> ScalarPowerMathProbeResult run_scalar_power_math_probe()
{
  using IntView = Kokkos::View<int *, DeviceType>;
  using RealView = Kokkos::View<double *, DeviceType>;
  using ScalarSpeciesView = Kokkos::View<YE3T_LAMMPS::YE3TKokkosScalarSpeciesRecord *, DeviceType>;
  using ScalarBaseView = Kokkos::View<YE3T_LAMMPS::YE3TKokkosScalarBaseRecord *, DeviceType>;
  using ScalarNodeView = Kokkos::View<YE3T_LAMMPS::YE3TKokkosScalarNodeRecord *, DeviceType>;
  using ScalarRouteView = Kokkos::View<YE3T_LAMMPS::YE3TKokkosScalarRouteRecord *, DeviceType>;

  constexpr int lane_count = 3;
  constexpr int source_count = 2;
  constexpr int value_count = 6;
  const std::complex<double> coefficient_pair(0.7, -0.2);
  const std::complex<double> coefficient_zero(-1.1, 0.35);
  const std::array<std::complex<double>, 3> route_scales{std::complex<double>(0.3, 0.4),
                                                         std::complex<double>(-0.2, 0.15),
                                                         std::complex<double>(0.05, -0.12)};
  constexpr double embedding_scale = -1.7;

  IntView channel_offsets("ye3t:scalar_math_probe_channel_offsets", 2);
  IntView full_channel_sources("ye3t:scalar_math_probe_channel_sources", 3);
  IntView full_channel_transforms("ye3t:scalar_math_probe_channel_transforms", 3);
  RealView embedding_scales("ye3t:scalar_math_probe_embedding", 1);
  auto host_channel_offsets = Kokkos::create_mirror_view(channel_offsets);
  auto host_full_channel_sources = Kokkos::create_mirror_view(full_channel_sources);
  auto host_full_channel_transforms = Kokkos::create_mirror_view(full_channel_transforms);
  auto host_embedding_scales = Kokkos::create_mirror_view(embedding_scales);
  host_channel_offsets(0) = 0;
  host_channel_offsets(1) = 3;
  host_full_channel_sources(0) = 0;
  host_full_channel_sources(1) = 1;
  host_full_channel_sources(2) = 0;
  host_full_channel_transforms(0) = -1;
  host_full_channel_transforms(1) = 0;
  host_full_channel_transforms(2) = 0;
  host_embedding_scales(0) = embedding_scale;
  Kokkos::deep_copy(channel_offsets, host_channel_offsets);
  Kokkos::deep_copy(full_channel_sources, host_full_channel_sources);
  Kokkos::deep_copy(full_channel_transforms, host_full_channel_transforms);
  Kokkos::deep_copy(embedding_scales, host_embedding_scales);

  ScalarSpeciesView scalar_species("ye3t:scalar_math_probe_species", 1);
  ScalarBaseView scalar_bases("ye3t:scalar_math_probe_bases", 1);
  ScalarNodeView scalar_nodes("ye3t:scalar_math_probe_nodes", 5);
  ScalarRouteView scalar_routes("ye3t:scalar_math_probe_routes", 3);
  IntView left_channels("ye3t:scalar_math_probe_left_channels", 2);
  IntView right_channels("ye3t:scalar_math_probe_right_channels", 2);
  RealView coefficient_real("ye3t:scalar_math_probe_coefficient_real", 2);
  RealView coefficient_imaginary("ye3t:scalar_math_probe_coefficient_imaginary", 2);

  auto host_scalar_species = Kokkos::create_mirror_view(scalar_species);
  auto host_scalar_bases = Kokkos::create_mirror_view(scalar_bases);
  auto host_scalar_nodes = Kokkos::create_mirror_view(scalar_nodes);
  auto host_scalar_routes = Kokkos::create_mirror_view(scalar_routes);
  auto host_left_channels = Kokkos::create_mirror_view(left_channels);
  auto host_right_channels = Kokkos::create_mirror_view(right_channels);
  auto host_coefficient_real = Kokkos::create_mirror_view(coefficient_real);
  auto host_coefficient_imaginary = Kokkos::create_mirror_view(coefficient_imaginary);

  host_scalar_species(0).base_begin = 0;
  host_scalar_species(0).base_count = 1;
  host_scalar_species(0).node_begin = 0;
  host_scalar_species(0).node_count = 5;
  host_scalar_species(0).route_begin = 0;
  host_scalar_species(0).route_count = 3;
  host_scalar_species(0).value_count = value_count;
  host_scalar_bases(0).term_begin = 0;
  host_scalar_bases(0).term_count = 2;
  const std::array<std::array<int, 4>, 5> nodes{{
      {{0, 0, 2, 0}},
      {{1, 1, 4, 0}},
      {{2, 2, 8, 0}},
      {{3, 2, 12, 0}},
      {{3, 3, 16, 0}},
  }};
  for (int node = 0; node < 5; ++node) {
    host_scalar_nodes(node).left_value = nodes[node][0];
    host_scalar_nodes(node).right_value = nodes[node][1];
    host_scalar_nodes(node).exponent = nodes[node][2];
    host_scalar_nodes(node).base_index = nodes[node][3];
  }
  for (int route = 0; route < 3; ++route) {
    host_scalar_routes(route).function_index = route;
    host_scalar_routes(route).value_index = route + 3;
    host_scalar_routes(route).scale_real = route_scales[route].real();
    host_scalar_routes(route).scale_imaginary = route_scales[route].imag();
  }
  host_left_channels(0) = 0;
  host_right_channels(0) = 2;
  host_left_channels(1) = 1;
  host_right_channels(1) = 1;
  host_coefficient_real(0) = coefficient_pair.real();
  host_coefficient_imaginary(0) = coefficient_pair.imag();
  host_coefficient_real(1) = coefficient_zero.real();
  host_coefficient_imaginary(1) = coefficient_zero.imag();
  Kokkos::deep_copy(scalar_species, host_scalar_species);
  Kokkos::deep_copy(scalar_bases, host_scalar_bases);
  Kokkos::deep_copy(scalar_nodes, host_scalar_nodes);
  Kokkos::deep_copy(scalar_routes, host_scalar_routes);
  Kokkos::deep_copy(left_channels, host_left_channels);
  Kokkos::deep_copy(right_channels, host_right_channels);
  Kokkos::deep_copy(coefficient_real, host_coefficient_real);
  Kokkos::deep_copy(coefficient_imaginary, host_coefficient_imaginary);

  RealView source_real("ye3t:scalar_math_probe_source_real", source_count * lane_count);
  RealView source_imaginary("ye3t:scalar_math_probe_source_imaginary", source_count * lane_count);
  RealView source_adjoint_real("ye3t:scalar_math_probe_source_adjoint_real",
                               source_count * lane_count);
  RealView source_adjoint_imaginary("ye3t:scalar_math_probe_source_adjoint_imaginary",
                                    source_count * lane_count);
  RealView value_real("ye3t:scalar_math_probe_value_real", value_count * lane_count);
  RealView value_imaginary("ye3t:scalar_math_probe_value_imaginary", value_count * lane_count);
  RealView adjoint_real("ye3t:scalar_math_probe_adjoint_real", value_count * lane_count);
  RealView adjoint_imaginary("ye3t:scalar_math_probe_adjoint_imaginary", value_count * lane_count);
  RealView density_real("ye3t:scalar_math_probe_density_real", lane_count);
  RealView density_imaginary("ye3t:scalar_math_probe_density_imaginary", lane_count);
  auto host_source_real = Kokkos::create_mirror_view(source_real);
  auto host_source_imaginary = Kokkos::create_mirror_view(source_imaginary);
  const std::array<std::complex<double>, lane_count> source_m1{
      std::complex<double>(0.8, -0.2), std::complex<double>(0.0, 0.0),
      std::complex<double>(1.0e-20, -2.0e-20)};
  const std::array<std::complex<double>, lane_count> source_0{
      std::complex<double>(-0.5, 0.35), std::complex<double>(0.0, 0.0),
      std::complex<double>(-3.0e-20, 1.0e-20)};
  for (int lane = 0; lane < lane_count; ++lane) {
    host_source_real(lane) = source_m1[lane].real();
    host_source_imaginary(lane) = source_m1[lane].imag();
    host_source_real(lane_count + lane) = source_0[lane].real();
    host_source_imaginary(lane_count + lane) = source_0[lane].imag();
  }
  Kokkos::deep_copy(source_real, host_source_real);
  Kokkos::deep_copy(source_imaginary, host_source_imaginary);
  Kokkos::deep_copy(source_adjoint_real, 0.0);
  Kokkos::deep_copy(source_adjoint_imaginary, 0.0);
  Kokkos::deep_copy(value_real, 0.0);
  Kokkos::deep_copy(value_imaginary, 0.0);
  Kokkos::deep_copy(adjoint_real, 0.0);
  Kokkos::deep_copy(adjoint_imaginary, 0.0);
  Kokkos::deep_copy(density_real, 0.0);
  Kokkos::deep_copy(density_imaginary, 0.0);

  YE3T_LAMMPS::YE3TKokkosSpeciesViews<DeviceType> species;
  species.species_count = 1;
  species.channel_offsets = channel_offsets;
  species.embedding_scales = embedding_scales;
  species.full_channel_sources = full_channel_sources;
  species.full_channel_transforms = full_channel_transforms;
  YE3T_LAMMPS::YE3TKokkosScalarViews<DeviceType> scalar{
      scalar_species, scalar_bases,   scalar_nodes,     scalar_routes,
      left_channels,  right_channels, coefficient_real, coefficient_imaginary};
  const Kokkos::View<const double *, DeviceType> source_real_const = source_real;
  const Kokkos::View<const double *, DeviceType> source_imaginary_const = source_imaginary;
  Kokkos::parallel_for(
      "YE3TScalarPowerMathProbe", Kokkos::RangePolicy<DeviceType>(0, lane_count),
      KOKKOS_LAMBDA(int lane) {
        scalar_power_forward_adjoint_center(
            species, scalar, source_real_const, source_imaginary_const, source_adjoint_real,
            source_adjoint_imaginary, value_real, value_imaginary, adjoint_real, adjoint_imaginary,
            density_real, density_imaginary, 0, lane_count, lane);
      });
  Kokkos::fence("YE3T scalar-power math probe");

  const auto host_density_real =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), density_real);
  const auto host_density_imaginary =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), density_imaginary);
  const auto host_source_adjoint_real =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), source_adjoint_real);
  const auto host_source_adjoint_imaginary =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), source_adjoint_imaginary);

  const auto density = [&](std::complex<double> compact_m1, std::complex<double> compact_0) {
    const std::complex<double> negative_m = -std::conj(compact_m1);
    const std::complex<double> quadratic =
        coefficient_pair * negative_m * compact_m1 + coefficient_zero * compact_0 * compact_0;
    const auto power_2 = quadratic * quadratic;
    const auto power_4 = power_2 * power_2;
    const auto power_8 = power_4 * power_4;
    const auto power_12 = power_8 * power_4;
    const auto power_16 = power_8 * power_8;
    return route_scales[0] * power_8 + route_scales[1] * power_12 + route_scales[2] * power_16;
  };

  ScalarPowerMathProbeResult result;
  for (int lane = 0; lane < lane_count; ++lane) {
    const std::complex<double> actual(host_density_real(lane), host_density_imaginary(lane));
    const std::complex<double> expected = density(source_m1[lane], source_0[lane]);
    result.forward_error = std::max(result.forward_error, std::abs(actual - expected));
    for (int source = 0; source < source_count; ++source) {
      const int offset = source * lane_count + lane;
      result.finite = result.finite && std::isfinite(host_source_adjoint_real(offset)) &&
          std::isfinite(host_source_adjoint_imaginary(offset));
      if (lane == 1)
        result.zero_error = std::max(
            result.zero_error,
            std::hypot(host_source_adjoint_real(offset), host_source_adjoint_imaginary(offset)));
    }
    result.finite = result.finite && std::isfinite(actual.real()) && std::isfinite(actual.imag());
  }

  const std::complex<double> direction_m1(0.31, 0.27);
  const std::complex<double> direction_0(-0.22, 0.19);
  constexpr double step = 1.0e-6;
  const double plus = embedding_scale *
      density(source_m1[0] + step * direction_m1, source_0[0] + step * direction_0).real();
  const double minus = embedding_scale *
      density(source_m1[0] - step * direction_m1, source_0[0] - step * direction_0).real();
  const double finite_difference = (plus - minus) / (2.0 * step);
  const std::complex<double> gradient_m1(host_source_adjoint_real(0),
                                         host_source_adjoint_imaginary(0));
  const std::complex<double> gradient_0(host_source_adjoint_real(lane_count),
                                        host_source_adjoint_imaginary(lane_count));
  const double adjoint_dot =
      (std::conj(gradient_m1) * direction_m1 + std::conj(gradient_0) * direction_0).real();
  result.adjoint_dot_error = std::abs(finite_difference - adjoint_dot);
  result.finite = result.finite && std::isfinite(result.forward_error) &&
      std::isfinite(result.adjoint_dot_error) && std::isfinite(result.zero_error);
  return result;
}

struct CoupledProductMathProbeResult {
  double forward_error = 0.0;
  double adjoint_dot_error = 0.0;
  double zero_error = 0.0;
  bool finite = true;
};

template <class DeviceType> CoupledProductMathProbeResult run_coupled_product_math_probe()
{
  using IntView = Kokkos::View<int *, DeviceType>;
  using RealView = Kokkos::View<double *, DeviceType>;
  using CoupledSpeciesView =
      Kokkos::View<YE3T_LAMMPS::YE3TKokkosCoupledSpeciesRecord *, DeviceType>;
  using CoupledPlanView = Kokkos::View<YE3T_LAMMPS::YE3TKokkosCoupledPlanRecord *, DeviceType>;
  using CoupledNodeView = Kokkos::View<YE3T_LAMMPS::YE3TKokkosCoupledNodeRecord *, DeviceType>;

  constexpr int lane_count = 3;
  constexpr int source_count = 4;
  constexpr int component_count = 11;
  constexpr double embedding_scale = -1.7;
  const std::array<double, 10> product_coefficients{
      {0.7, -0.4, 1.2, -0.6, 0.9, -1.3, 0.55, -0.75, 1.1, -0.8}};
  const std::array<double, 3> readout_coefficients{{0.8, -1.1, 0.35}};

  IntView channel_offsets("ye3t:coupled_math_probe_channel_offsets", 2);
  IntView full_channel_sources("ye3t:coupled_math_probe_channel_sources", 6);
  IntView full_channel_transforms("ye3t:coupled_math_probe_channel_transforms", 6);
  RealView embedding_scales("ye3t:coupled_math_probe_embedding", 1);
  auto host_channel_offsets = Kokkos::create_mirror_view(channel_offsets);
  auto host_full_channel_sources = Kokkos::create_mirror_view(full_channel_sources);
  auto host_full_channel_transforms = Kokkos::create_mirror_view(full_channel_transforms);
  auto host_embedding_scales = Kokkos::create_mirror_view(embedding_scales);
  host_channel_offsets(0) = 0;
  host_channel_offsets(1) = 6;
  host_full_channel_sources(0) = 0;
  host_full_channel_sources(1) = 1;
  host_full_channel_sources(2) = 0;
  host_full_channel_sources(3) = 2;
  host_full_channel_sources(4) = 3;
  host_full_channel_sources(5) = 2;
  host_full_channel_transforms(0) = -1;
  host_full_channel_transforms(1) = 0;
  host_full_channel_transforms(2) = 0;
  host_full_channel_transforms(3) = -1;
  host_full_channel_transforms(4) = 0;
  host_full_channel_transforms(5) = 0;
  host_embedding_scales(0) = embedding_scale;
  Kokkos::deep_copy(channel_offsets, host_channel_offsets);
  Kokkos::deep_copy(full_channel_sources, host_full_channel_sources);
  Kokkos::deep_copy(full_channel_transforms, host_full_channel_transforms);
  Kokkos::deep_copy(embedding_scales, host_embedding_scales);

  CoupledSpeciesView coupled_species("ye3t:coupled_math_probe_species", 1);
  CoupledPlanView coupled_plans("ye3t:coupled_math_probe_plans", 2);
  CoupledNodeView coupled_nodes("ye3t:coupled_math_probe_nodes", 8);
  IntView leaf_input_channels("ye3t:coupled_math_probe_leaf_inputs", 8);
  IntView coefficient_left_components("ye3t:coupled_math_probe_coefficient_left", 10);
  IntView coefficient_right_components("ye3t:coupled_math_probe_coefficient_right", 10);
  IntView coefficient_output_components("ye3t:coupled_math_probe_coefficient_output", 10);
  RealView coefficient_values("ye3t:coupled_math_probe_coefficients", 10);
  IntView readout_components("ye3t:coupled_math_probe_readout_components", 3);
  RealView readout_values("ye3t:coupled_math_probe_readout_coefficients", 3);

  auto host_coupled_species = Kokkos::create_mirror_view(coupled_species);
  auto host_coupled_plans = Kokkos::create_mirror_view(coupled_plans);
  auto host_coupled_nodes = Kokkos::create_mirror_view(coupled_nodes);
  auto host_leaf_input_channels = Kokkos::create_mirror_view(leaf_input_channels);
  auto host_coefficient_left_components = Kokkos::create_mirror_view(coefficient_left_components);
  auto host_coefficient_right_components = Kokkos::create_mirror_view(coefficient_right_components);
  auto host_coefficient_output_components =
      Kokkos::create_mirror_view(coefficient_output_components);
  auto host_coefficient_values = Kokkos::create_mirror_view(coefficient_values);
  auto host_readout_components = Kokkos::create_mirror_view(readout_components);
  auto host_readout_values = Kokkos::create_mirror_view(readout_values);

  host_coupled_species(0) = {0, 2};
  host_coupled_plans(0) = {0, 5, 0, 6, 0, 9, 0, 2, 11, 0, 24};
  host_coupled_plans(1) = {5, 3, 6, 2, 9, 1, 2, 1, 3, 1, 4};
  const std::array<YE3T_LAMMPS::YE3TKokkosCoupledNodeRecord, 8> nodes{{
      {0, 3, 0, 0, 0},
      {3, 3, 3, 0, 0},
      {6, 3, -1, 0, 6},
      {9, 1, -1, 6, 2},
      {10, 1, -1, 8, 1},
      {0, 1, 0, 0, 0},
      {1, 1, 1, 0, 0},
      {2, 1, -1, 0, 1},
  }};
  for (int node = 0; node < 8; ++node)
    host_coupled_nodes(node) = nodes[static_cast<std::size_t>(node)];
  const std::array<int, 8> leaf_inputs{{0, 1, 2, 3, 4, 5, 1, 4}};
  const std::array<int, 10> coefficient_left{{0, 2, 0, 1, 1, 2, 6, 7, 6, 0}};
  const std::array<int, 10> coefficient_right{{5, 3, 4, 3, 5, 4, 6, 8, 8, 1}};
  const std::array<int, 10> coefficient_output{{6, 6, 7, 7, 8, 8, 9, 9, 10, 2}};
  const std::array<int, 3> readout_indices{{9, 10, 2}};
  for (int leaf = 0; leaf < 8; ++leaf)
    host_leaf_input_channels(leaf) = leaf_inputs[static_cast<std::size_t>(leaf)];
  for (int coefficient = 0; coefficient < 10; ++coefficient) {
    const std::size_t index = static_cast<std::size_t>(coefficient);
    host_coefficient_left_components(coefficient) = coefficient_left[index];
    host_coefficient_right_components(coefficient) = coefficient_right[index];
    host_coefficient_output_components(coefficient) = coefficient_output[index];
    host_coefficient_values(coefficient) = product_coefficients[index];
  }
  for (int readout = 0; readout < 3; ++readout) {
    const std::size_t index = static_cast<std::size_t>(readout);
    host_readout_components(readout) = readout_indices[index];
    host_readout_values(readout) = readout_coefficients[index];
  }
  Kokkos::deep_copy(coupled_species, host_coupled_species);
  Kokkos::deep_copy(coupled_plans, host_coupled_plans);
  Kokkos::deep_copy(coupled_nodes, host_coupled_nodes);
  Kokkos::deep_copy(leaf_input_channels, host_leaf_input_channels);
  Kokkos::deep_copy(coefficient_left_components, host_coefficient_left_components);
  Kokkos::deep_copy(coefficient_right_components, host_coefficient_right_components);
  Kokkos::deep_copy(coefficient_output_components, host_coefficient_output_components);
  Kokkos::deep_copy(coefficient_values, host_coefficient_values);
  Kokkos::deep_copy(readout_components, host_readout_components);
  Kokkos::deep_copy(readout_values, host_readout_values);

  RealView source_real("ye3t:coupled_math_probe_source_real", source_count * lane_count);
  RealView source_imaginary("ye3t:coupled_math_probe_source_imaginary", source_count * lane_count);
  RealView source_adjoint_real("ye3t:coupled_math_probe_source_adjoint_real",
                               source_count * lane_count);
  RealView source_adjoint_imaginary("ye3t:coupled_math_probe_source_adjoint_imaginary",
                                    source_count * lane_count);
  RealView value_real("ye3t:coupled_math_probe_value_real", component_count * lane_count);
  RealView value_imaginary("ye3t:coupled_math_probe_value_imaginary", component_count * lane_count);
  RealView adjoint_real("ye3t:coupled_math_probe_adjoint_real", component_count * lane_count);
  RealView adjoint_imaginary("ye3t:coupled_math_probe_adjoint_imaginary",
                             component_count * lane_count);
  RealView density_real("ye3t:coupled_math_probe_density_real", lane_count);
  RealView density_imaginary("ye3t:coupled_math_probe_density_imaginary", lane_count);
  auto host_source_real = Kokkos::create_mirror_view(source_real);
  auto host_source_imaginary = Kokkos::create_mirror_view(source_imaginary);
  auto host_source_adjoint_real = Kokkos::create_mirror_view(source_adjoint_real);
  auto host_source_adjoint_imaginary = Kokkos::create_mirror_view(source_adjoint_imaginary);
  auto host_density_real = Kokkos::create_mirror_view(density_real);
  auto host_density_imaginary = Kokkos::create_mirror_view(density_imaginary);
  const std::array<std::complex<double>, lane_count> source_a{
      std::complex<double>(0.8, -0.2), std::complex<double>(0.0, 0.0),
      std::complex<double>(1.0e-20, -2.0e-20)};
  const std::array<std::complex<double>, lane_count> source_b{
      std::complex<double>(-0.5, 0.35), std::complex<double>(0.0, 0.0),
      std::complex<double>(-3.0e-20, 1.0e-20)};
  const std::array<std::complex<double>, lane_count> source_c{
      std::complex<double>(0.27, 0.41), std::complex<double>(0.0, 0.0),
      std::complex<double>(2.0e-20, 3.0e-20)};
  const std::array<std::complex<double>, lane_count> source_d{
      std::complex<double>(-0.33, -0.18), std::complex<double>(0.0, 0.0),
      std::complex<double>(-1.0e-20, 4.0e-20)};
  std::array<std::complex<double>, source_count * lane_count> adjoint_seed{};
  for (int offset = 0; offset < source_count * lane_count; ++offset)
    adjoint_seed[static_cast<std::size_t>(offset)] =
        std::complex<double>(0.013 * (offset + 1), -0.017 * (offset + 1));
  const std::array<std::complex<double>, lane_count> density_seed{
      std::complex<double>(0.11, -0.07), std::complex<double>(-0.23, 0.13),
      std::complex<double>(0.37, -0.17)};
  for (int lane = 0; lane < lane_count; ++lane) {
    const std::array<std::complex<double>, source_count> sources{source_a[lane], source_b[lane],
                                                                 source_c[lane], source_d[lane]};
    for (int source = 0; source < source_count; ++source) {
      const int offset = source * lane_count + lane;
      host_source_real(offset) = sources[static_cast<std::size_t>(source)].real();
      host_source_imaginary(offset) = sources[static_cast<std::size_t>(source)].imag();
    }
    host_density_real(lane) = density_seed[lane].real();
    host_density_imaginary(lane) = density_seed[lane].imag();
  }
  for (int offset = 0; offset < source_count * lane_count; ++offset) {
    host_source_adjoint_real(offset) = adjoint_seed[offset].real();
    host_source_adjoint_imaginary(offset) = adjoint_seed[offset].imag();
  }
  Kokkos::deep_copy(source_real, host_source_real);
  Kokkos::deep_copy(source_imaginary, host_source_imaginary);
  Kokkos::deep_copy(source_adjoint_real, host_source_adjoint_real);
  Kokkos::deep_copy(source_adjoint_imaginary, host_source_adjoint_imaginary);
  const double probe_nan = std::numeric_limits<double>::quiet_NaN();
  Kokkos::deep_copy(value_real, probe_nan);
  Kokkos::deep_copy(value_imaginary, probe_nan);
  Kokkos::deep_copy(adjoint_real, probe_nan);
  Kokkos::deep_copy(adjoint_imaginary, probe_nan);
  Kokkos::deep_copy(density_real, host_density_real);
  Kokkos::deep_copy(density_imaginary, host_density_imaginary);

  YE3T_LAMMPS::YE3TKokkosSpeciesViews<DeviceType> species;
  species.species_count = 1;
  species.channel_offsets = channel_offsets;
  species.embedding_scales = embedding_scales;
  species.full_channel_sources = full_channel_sources;
  species.full_channel_transforms = full_channel_transforms;
  YE3T_LAMMPS::YE3TKokkosCoupledViews<DeviceType> coupled{
      coupled_species,
      coupled_plans,
      coupled_nodes,
      leaf_input_channels,
      coefficient_left_components,
      coefficient_right_components,
      coefficient_output_components,
      coefficient_values,
      readout_components,
      readout_values,
  };
  const Kokkos::View<const double *, DeviceType> source_real_const = source_real;
  const Kokkos::View<const double *, DeviceType> source_imaginary_const = source_imaginary;
  Kokkos::parallel_for(
      "YE3TCoupledProductMathProbe", Kokkos::RangePolicy<DeviceType>(0, lane_count),
      KOKKOS_LAMBDA(int lane) {
        coupled_product_forward_adjoint_center(
            species, coupled, source_real_const, source_imaginary_const, source_adjoint_real,
            source_adjoint_imaginary, value_real, value_imaginary, adjoint_real, adjoint_imaginary,
            density_real, density_imaginary, 0, lane_count, lane);
      });
  Kokkos::fence("YE3T coupled-product math probe");

  const auto result_density_real =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), density_real);
  const auto result_density_imaginary =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), density_imaginary);
  const auto result_source_adjoint_real =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), source_adjoint_real);
  const auto result_source_adjoint_imaginary =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), source_adjoint_imaginary);

  const auto density = [&](std::complex<double> a, std::complex<double> b, std::complex<double> c,
                           std::complex<double> d) {
    const std::complex<double> a_negative = -std::conj(a);
    const std::complex<double> c_negative = -std::conj(c);
    const std::complex<double> u0 =
        product_coefficients[0] * a_negative * c + product_coefficients[1] * a * c_negative;
    const std::complex<double> u1 =
        product_coefficients[2] * a_negative * d + product_coefficients[3] * b * c_negative;
    const std::complex<double> u2 =
        product_coefficients[4] * b * c + product_coefficients[5] * a * d;
    const std::complex<double> first =
        product_coefficients[6] * u0 * u0 + product_coefficients[7] * u1 * u2;
    const std::complex<double> second = product_coefficients[8] * u0 * u2;
    const std::complex<double> third = product_coefficients[9] * b * d;
    return readout_coefficients[0] * first + readout_coefficients[1] * second +
        readout_coefficients[2] * third;
  };

  CoupledProductMathProbeResult result;
  for (int lane = 0; lane < lane_count; ++lane) {
    const std::complex<double> actual_density(result_density_real(lane),
                                              result_density_imaginary(lane));
    const std::complex<double> expected_density = density_seed[lane] +
        density(source_a[lane], source_b[lane], source_c[lane], source_d[lane]);
    result.forward_error =
        std::max(result.forward_error, std::abs(actual_density - expected_density));
    result.finite = result.finite && std::isfinite(actual_density.real()) &&
        std::isfinite(actual_density.imag());
    for (int source = 0; source < source_count; ++source) {
      const int offset = source * lane_count + lane;
      const std::complex<double> actual_adjoint(result_source_adjoint_real(offset),
                                                result_source_adjoint_imaginary(offset));
      result.finite = result.finite && std::isfinite(actual_adjoint.real()) &&
          std::isfinite(actual_adjoint.imag());
      if (lane == 1)
        result.zero_error =
            std::max(result.zero_error, std::abs(actual_adjoint - adjoint_seed[offset]));
    }
    if (lane == 1)
      result.zero_error =
          std::max(result.zero_error, std::abs(actual_density - density_seed[lane]));
  }

  const std::array<std::array<std::complex<double>, source_count>, 2> directions{{
      {std::complex<double>(0.31, 0.27), std::complex<double>(-0.22, 0.19),
       std::complex<double>(0.17, -0.29), std::complex<double>(-0.13, -0.11)},
      {std::complex<double>(-0.07, 0.23), std::complex<double>(0.37, -0.05),
       std::complex<double>(-0.19, -0.31), std::complex<double>(0.29, 0.41)},
  }};
  const std::array<std::complex<double>, source_count> sources{source_a[0], source_b[0],
                                                               source_c[0], source_d[0]};
  std::array<std::complex<double>, source_count> gradients{};
  for (int source = 0; source < source_count; ++source) {
    const int offset = source * lane_count;
    gradients[static_cast<std::size_t>(source)] = {
        result_source_adjoint_real(offset) - adjoint_seed[offset].real(),
        result_source_adjoint_imaginary(offset) - adjoint_seed[offset].imag()};
  }
  constexpr double step = 1.0e-6;
  for (const auto &direction : directions) {
    std::array<std::complex<double>, source_count> plus_sources{};
    std::array<std::complex<double>, source_count> minus_sources{};
    std::complex<double> adjoint_dot;
    for (int source = 0; source < source_count; ++source) {
      const std::size_t index = static_cast<std::size_t>(source);
      plus_sources[index] = sources[index] + step * direction[index];
      minus_sources[index] = sources[index] - step * direction[index];
      adjoint_dot += std::conj(gradients[index]) * direction[index];
    }
    const double plus = embedding_scale *
        density(plus_sources[0], plus_sources[1], plus_sources[2], plus_sources[3]).real();
    const double minus = embedding_scale *
        density(minus_sources[0], minus_sources[1], minus_sources[2], minus_sources[3]).real();
    const double finite_difference = (plus - minus) / (2.0 * step);
    result.adjoint_dot_error =
        std::max(result.adjoint_dot_error, std::abs(finite_difference - adjoint_dot.real()));
  }
  result.finite = result.finite && std::isfinite(result.forward_error) &&
      std::isfinite(result.adjoint_dot_error) && std::isfinite(result.zero_error);
  return result;
}

template <class DeviceType> struct BlockPowerForward {
  using AT = ArrayTypes<DeviceType>;
  typename AT::t_int_1d_randomread types;
  typename AT::t_int_1d_randomread ilist;
  Kokkos::View<const int *, DeviceType> type_to_species;
  YE3T_LAMMPS::YE3TKokkosSpeciesViews<DeviceType> species;
  YE3T_LAMMPS::YE3TKokkosBlockViews<DeviceType> block;
  Kokkos::View<const double *, DeviceType> source_real;
  Kokkos::View<const double *, DeviceType> source_imaginary;
  Kokkos::View<double *, DeviceType> block_power_real;
  Kokkos::View<double *, DeviceType> block_power_imaginary;
  Kokkos::View<double *, DeviceType> block_output_real;
  Kokkos::View<double *, DeviceType> block_output_imaginary;
  Kokkos::View<double *, DeviceType> block_output_adjoint_real;
  Kokkos::View<double *, DeviceType> block_output_adjoint_imaginary;
  int chunk_begin = 0;
  int center_capacity = 0;

  KOKKOS_INLINE_FUNCTION
  void operator()(int lane) const
  {
    const int atom_i = ilist(chunk_begin + lane);
    const int central = type_to_species(types(atom_i));
    if (central < 0) return;
    const auto species_block = block.species(central);

    for (int component = 0; component < species_block.output_storage_size; ++component) {
      const int index = component * center_capacity + lane;
      block_output_real(index) = 0.0;
      block_output_imaginary(index) = 0.0;
      block_output_adjoint_real(index) = 0.0;
      block_output_adjoint_imaginary(index) = 0.0;
    }
    for (int power_index = 0; power_index < species_block.power_count; ++power_index) {
      const auto power = block.powers(species_block.power_begin + power_index);
      const int base = power.storage_offset * center_capacity + lane;
      block_power_real(base) = 1.0;
      block_power_imaginary(base) = 0.0;
      if (power.maximum_exponent == 0) continue;
      const DeviceComplex input = read_full_channel(species, source_real, source_imaginary, central,
                                                    power.channel, center_capacity, lane);
      DeviceComplex value{1.0, 0.0};
      for (int exponent = 1; exponent <= power.maximum_exponent; ++exponent) {
        value = multiply(value, input);
        const int destination = (power.storage_offset + exponent) * center_capacity + lane;
        block_power_real(destination) = value.real;
        block_power_imaginary(destination) = value.imaginary;
      }
    }
  }
};

template <class DeviceType> struct BlockPlanForward {
  using AT = ArrayTypes<DeviceType>;
  typename AT::t_int_1d_randomread types;
  typename AT::t_int_1d_randomread ilist;
  Kokkos::View<const int *, DeviceType> type_to_species;
  YE3T_LAMMPS::YE3TKokkosSpeciesViews<DeviceType> species;
  YE3T_LAMMPS::YE3TKokkosBlockViews<DeviceType> block;
  Kokkos::View<const double *, DeviceType> source_real;
  Kokkos::View<const double *, DeviceType> source_imaginary;
  Kokkos::View<const double *, DeviceType> block_power_real;
  Kokkos::View<const double *, DeviceType> block_power_imaginary;
  Kokkos::View<double *, DeviceType> block_output_real;
  Kokkos::View<double *, DeviceType> block_output_imaginary;
  Kokkos::View<double *, DeviceType> block_monomial_real;
  Kokkos::View<double *, DeviceType> block_monomial_imaginary;
  int chunk_begin = 0;
  int center_capacity = 0;

  KOKKOS_INLINE_FUNCTION
  void operator()(int lane) const
  {
    const int atom_i = ilist(chunk_begin + lane);
    const int central = type_to_species(types(atom_i));
    if (central < 0) return;
    const auto species_block = block.species(central);

    for (int local_plan = 0; local_plan < species_block.plan_count; ++local_plan) {
      const auto plan = block.plans(species_block.plan_begin + local_plan);
      const bool direct_input = (plan.flags & YE3T_LAMMPS::YE3T_BLOCK_DIRECT_INPUT) != 0;
      const bool conjugate_half = (plan.flags & YE3T_LAMMPS::YE3T_BLOCK_CONJUGATE_HALF_OUTPUT) != 0;
      const bool real_coefficients = (plan.flags & YE3T_LAMMPS::YE3T_BLOCK_REAL_COEFFICIENTS) != 0;

      if (direct_input) {
        for (int component = 0; component < plan.output_count; ++component) {
          const int output = plan.output_begin + component;
          const int channel = block.direct_input_channels(output);
          if (channel < 0) continue;
          const double scale = block.direct_input_scales(output);
          const DeviceComplex input = read_full_channel(species, source_real, source_imaginary,
                                                        central, channel, center_capacity, lane);
          const int destination = (plan.output_storage_offset + component) * center_capacity + lane;
          block_output_real(destination) = scale * input.real;
          block_output_imaginary(destination) = scale * input.imaginary;
        }
      } else {
        const YE3T_LAMMPS::GpuStridedDoubles tmp_r{block_monomial_real.data(),
                                                   static_cast<std::size_t>(center_capacity),
                                                   static_cast<std::size_t>(lane)};
        const YE3T_LAMMPS::GpuStridedDoubles tmp_i{block_monomial_imaginary.data(),
                                                   static_cast<std::size_t>(center_capacity),
                                                   static_cast<std::size_t>(lane)};
        // A single view-like type is used for both components by the helper.
        const YE3T_LAMMPS::GpuConstStridedDoubles power_r{block_power_real.data(),
                                                          static_cast<std::size_t>(center_capacity),
                                                          static_cast<std::size_t>(lane)};
        const YE3T_LAMMPS::GpuConstStridedDoubles power_i{block_power_imaginary.data(),
                                                          static_cast<std::size_t>(center_capacity),
                                                          static_cast<std::size_t>(lane)};
        const YE3T_LAMMPS::GpuStridedDoubles out_r{block_output_real.data(),
                                                   static_cast<std::size_t>(center_capacity),
                                                   static_cast<std::size_t>(lane)};
        const YE3T_LAMMPS::GpuStridedDoubles out_i{block_output_imaginary.data(),
                                                   static_cast<std::size_t>(center_capacity),
                                                   static_cast<std::size_t>(lane)};
        YE3T_LAMMPS::gpu_block_forward_tiles(block, plan, species_block.plan_begin + local_plan,
                                             power_r, power_i, out_r, out_i, tmp_r, tmp_i,
                                             real_coefficients);
      }

      if (conjugate_half && plan.output_L > 0) {
        const int width = 2 * plan.output_L + 1;
        const int multiplicity = plan.output_count / width;
        for (int entry = 0; entry < multiplicity * plan.output_L; ++entry) {
          const int copy = entry / plan.output_L;
          const int magnetic = entry % plan.output_L + 1;
          const double phase = magnetic % 2 == 0 ? 1.0 : -1.0;
          const int negative_component = copy * width + plan.output_L - magnetic;
          const int positive_component = copy * width + plan.output_L + magnetic;
          const int negative =
              (plan.output_storage_offset + negative_component) * center_capacity + lane;
          const int positive =
              (plan.output_storage_offset + positive_component) * center_capacity + lane;
          block_output_real(negative) = phase * block_output_real(positive);
          block_output_imaginary(negative) = -phase * block_output_imaginary(positive);
        }
      }
    }
  }
};

template <class DeviceType> struct BlockRouteForwardAdjoint {
  using AT = ArrayTypes<DeviceType>;
  typename AT::t_int_1d_randomread types;
  typename AT::t_int_1d_randomread ilist;
  Kokkos::View<const int *, DeviceType> type_to_species;
  YE3T_LAMMPS::YE3TKokkosBlockViews<DeviceType> block;
  Kokkos::View<const double *, DeviceType> block_output_real;
  Kokkos::View<const double *, DeviceType> block_output_imaginary;
  Kokkos::View<double *, DeviceType> block_output_adjoint_real;
  Kokkos::View<double *, DeviceType> block_output_adjoint_imaginary;
  Kokkos::View<double *, DeviceType> total_density_real;
  Kokkos::View<double *, DeviceType> total_density_imaginary;
  int chunk_begin = 0;
  int center_capacity = 0;

  KOKKOS_INLINE_FUNCTION
  void operator()(int lane) const
  {
    const int atom_i = ilist(chunk_begin + lane);
    const int central = type_to_species(types(atom_i));
    if (central < 0) return;
    const auto species_block = block.species(central);
    double density_real = 0.0;
    double density_imaginary = 0.0;

    for (int local_route = 0; local_route < species_block.route_count; ++local_route) {
      const auto route = block.routes(species_block.route_begin + local_route);
      for (int local_term = 0; local_term < route.term_count; ++local_term) {
        const int term = route.term_begin + local_term;
        const int begin = block.route_term_factor_offsets(term);
        const int end = block.route_term_factor_offsets(term + 1);
        DeviceComplex product{1.0, 0.0};
        bool product_initialized = false;
        for (int factor = begin; factor < end; ++factor) {
          const auto plan = block.plans(block.route_term_factor_plans(factor));
          const int component = block.route_term_factor_components(factor);
          const int source = (plan.output_storage_offset + component) * center_capacity + lane;
          const DeviceComplex factor_value{block_output_real(source),
                                           block_output_imaginary(source)};
          product = product_initialized ? multiply(product, factor_value) : factor_value;
          product_initialized = true;
        }
        const DeviceComplex coefficient{block.route_term_coefficient_real(term),
                                        block.route_term_coefficient_imaginary(term)};
        const DeviceComplex weighted = multiply_coefficient(coefficient, product);
        density_real += weighted.real;
        density_imaginary += weighted.imaginary;

        for (int target = begin; target < end; ++target) {
          DeviceComplex remaining{1.0, 0.0};
          bool remaining_initialized = false;
          for (int factor = begin; factor < end; ++factor) {
            if (factor == target) continue;
            const auto plan = block.plans(block.route_term_factor_plans(factor));
            const int component = block.route_term_factor_components(factor);
            const int source = (plan.output_storage_offset + component) * center_capacity + lane;
            const DeviceComplex factor_value{block_output_real(source),
                                             block_output_imaginary(source)};
            remaining = remaining_initialized ? multiply(remaining, factor_value) : factor_value;
            remaining_initialized = true;
          }
          const DeviceComplex derivative = multiply_coefficient(coefficient, remaining);
          const auto target_plan = block.plans(block.route_term_factor_plans(target));
          const int target_component = block.route_term_factor_components(target);
          const int destination =
              (target_plan.output_storage_offset + target_component) * center_capacity + lane;
          block_output_adjoint_real(destination) += derivative.real;
          block_output_adjoint_imaginary(destination) -= derivative.imaginary;
        }
      }
    }
    total_density_real(lane) += density_real;
    total_density_imaginary(lane) += density_imaginary;
  }
};

template <class DeviceType> struct BlockPlanAdjoint {
  using AT = ArrayTypes<DeviceType>;
  typename AT::t_int_1d_randomread types;
  typename AT::t_int_1d_randomread ilist;
  Kokkos::View<const int *, DeviceType> type_to_species;
  YE3T_LAMMPS::YE3TKokkosSpeciesViews<DeviceType> species;
  YE3T_LAMMPS::YE3TKokkosBlockViews<DeviceType> block;
  Kokkos::View<const double *, DeviceType> block_power_real;
  Kokkos::View<const double *, DeviceType> block_power_imaginary;
  Kokkos::View<const double *, DeviceType> block_output_real;
  Kokkos::View<const double *, DeviceType> block_output_imaginary;
  Kokkos::View<double *, DeviceType> block_output_adjoint_real;
  Kokkos::View<double *, DeviceType> block_output_adjoint_imaginary;
  Kokkos::View<double *, DeviceType> block_monomial_adjoint_real;
  Kokkos::View<double *, DeviceType> block_monomial_adjoint_imaginary;
  Kokkos::View<double *, DeviceType> source_adjoint_real;
  Kokkos::View<double *, DeviceType> source_adjoint_imaginary;
  int chunk_begin = 0;
  int center_capacity = 0;

  KOKKOS_INLINE_FUNCTION
  void operator()(int lane) const
  {
    const int atom_i = ilist(chunk_begin + lane);
    const int central = type_to_species(types(atom_i));
    if (central < 0) return;
    const auto species_block = block.species(central);
    const double embedding_scale = species.embedding_scales(central);

    for (int local_plan = 0; local_plan < species_block.plan_count; ++local_plan) {
      const auto plan = block.plans(species_block.plan_begin + local_plan);
      const bool direct_input = (plan.flags & YE3T_LAMMPS::YE3T_BLOCK_DIRECT_INPUT) != 0;
      const bool conjugate_half = (plan.flags & YE3T_LAMMPS::YE3T_BLOCK_CONJUGATE_HALF_OUTPUT) != 0;
      const bool real_coefficients = (plan.flags & YE3T_LAMMPS::YE3T_BLOCK_REAL_COEFFICIENTS) != 0;

      if (conjugate_half && plan.output_L > 0) {
        const int width = 2 * plan.output_L + 1;
        const int multiplicity = plan.output_count / width;
        for (int entry = 0; entry < multiplicity * plan.output_L; ++entry) {
          const int copy = entry / plan.output_L;
          const int magnetic = entry % plan.output_L + 1;
          const double phase = magnetic % 2 == 0 ? 1.0 : -1.0;
          const int negative_component = copy * width + plan.output_L - magnetic;
          const int positive_component = copy * width + plan.output_L + magnetic;
          const int negative =
              (plan.output_storage_offset + negative_component) * center_capacity + lane;
          const int positive =
              (plan.output_storage_offset + positive_component) * center_capacity + lane;
          block_output_adjoint_real(positive) += phase * block_output_adjoint_real(negative);
          block_output_adjoint_imaginary(positive) -=
              phase * block_output_adjoint_imaginary(negative);
          block_output_adjoint_real(negative) = 0.0;
          block_output_adjoint_imaginary(negative) = 0.0;
        }
      }

      if (direct_input) {
        for (int component = 0; component < plan.output_count; ++component) {
          const int output = plan.output_begin + component;
          const int channel = block.direct_input_channels(output);
          if (channel < 0) continue;
          const int source = (plan.output_storage_offset + component) * center_capacity + lane;
          const double scale = embedding_scale * block.direct_input_scales(output);
          add_owned_full_channel_adjoint(species, source_adjoint_real, source_adjoint_imaginary,
                                         central, channel, center_capacity, lane,
                                         scale * block_output_adjoint_real(source),
                                         scale * block_output_adjoint_imaginary(source));
        }
        continue;
      }
      const YE3T_LAMMPS::GpuConstStridedDoubles power_r{block_power_real.data(),
                                                        static_cast<std::size_t>(center_capacity),
                                                        static_cast<std::size_t>(lane)};
      const YE3T_LAMMPS::GpuConstStridedDoubles power_i{block_power_imaginary.data(),
                                                        static_cast<std::size_t>(center_capacity),
                                                        static_cast<std::size_t>(lane)};
      const YE3T_LAMMPS::GpuStridedDoubles root_r{block_output_adjoint_real.data(),
                                                  static_cast<std::size_t>(center_capacity),
                                                  static_cast<std::size_t>(lane)};
      const YE3T_LAMMPS::GpuStridedDoubles root_i{block_output_adjoint_imaginary.data(),
                                                  static_cast<std::size_t>(center_capacity),
                                                  static_cast<std::size_t>(lane)};
      const auto add = [&](int channel, double gr, double gi) {
        add_owned_full_channel_adjoint(species, source_adjoint_real, source_adjoint_imaginary,
                                       central, channel, center_capacity, lane, gr, gi);
      };
      YE3T_LAMMPS::gpu_block_adjoint_transpose(block, plan, power_r, power_i, root_r, root_i, add,
                                               embedding_scale, real_coefficients);
    }
  }
};

template <class DeviceType> struct BlockFusedForwardAdjoint {
  BlockPowerForward<DeviceType> power;
  BlockPlanForward<DeviceType> plan_forward;
  BlockRouteForwardAdjoint<DeviceType> route;
  BlockPlanAdjoint<DeviceType> plan_adjoint;

  KOKKOS_INLINE_FUNCTION
  void operator()(int lane) const
  {
    power(lane);
    plan_forward(lane);
    route(lane);
    plan_adjoint(lane);
  }
};

template <class DeviceType> struct BlockPowerWorkMajor {
  using AT = ArrayTypes<DeviceType>;
  using TeamMember = typename Kokkos::TeamPolicy<DeviceType>::member_type;
  typename AT::t_int_1d_randomread types;
  typename AT::t_int_1d_randomread ilist;
  Kokkos::View<const int *, DeviceType> type_to_species;
  YE3T_LAMMPS::YE3TKokkosSpeciesViews<DeviceType> species;
  YE3T_LAMMPS::YE3TKokkosBlockViews<DeviceType> block;
  Kokkos::View<const double *, DeviceType> source_real;
  Kokkos::View<const double *, DeviceType> source_imaginary;
  Kokkos::View<double *, DeviceType> block_power_real;
  Kokkos::View<double *, DeviceType> block_power_imaginary;
  Kokkos::View<double *, DeviceType> block_output_real;
  Kokkos::View<double *, DeviceType> block_output_imaginary;
  Kokkos::View<double *, DeviceType> block_output_adjoint_real;
  Kokkos::View<double *, DeviceType> block_output_adjoint_imaginary;
  int chunk_begin = 0;
  int center_count = 0;
  int center_tiles = 0;
  int center_capacity = 0;

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamMember &team) const
  {
    const int local_power = team.league_rank() / center_tiles;
    const int center_tile = team.league_rank() % center_tiles;
    const int lane = center_tile * team.team_size() + team.team_rank();
    if (lane >= center_count) return;
    const int atom_i = ilist(chunk_begin + lane);
    const int central = type_to_species(types(atom_i));
    if (central < 0) return;
    const auto species_block = block.species(central);

    if (local_power == 0) {
      for (int component = 0; component < species_block.output_storage_size; ++component) {
        const int index = component * center_capacity + lane;
        block_output_real(index) = 0.0;
        block_output_imaginary(index) = 0.0;
        block_output_adjoint_real(index) = 0.0;
        block_output_adjoint_imaginary(index) = 0.0;
      }
    }
    if (local_power >= species_block.power_count) return;

    const auto power = block.powers(species_block.power_begin + local_power);
    const int base = power.storage_offset * center_capacity + lane;
    block_power_real(base) = 1.0;
    block_power_imaginary(base) = 0.0;
    if (power.maximum_exponent == 0) return;
    const DeviceComplex input = read_full_channel(species, source_real, source_imaginary, central,
                                                  power.channel, center_capacity, lane);
    DeviceComplex value{1.0, 0.0};
    for (int exponent = 1; exponent <= power.maximum_exponent; ++exponent) {
      value = multiply(value, input);
      const int destination = (power.storage_offset + exponent) * center_capacity + lane;
      block_power_real(destination) = value.real;
      block_power_imaginary(destination) = value.imaginary;
    }
  }
};

template <class DeviceType> struct BlockPlanForwardWorkMajor {
  using AT = ArrayTypes<DeviceType>;
  using TeamMember = typename Kokkos::TeamPolicy<DeviceType>::member_type;
  typename AT::t_int_1d_randomread types;
  typename AT::t_int_1d_randomread ilist;
  Kokkos::View<const int *, DeviceType> type_to_species;
  YE3T_LAMMPS::YE3TKokkosSpeciesViews<DeviceType> species;
  YE3T_LAMMPS::YE3TKokkosBlockViews<DeviceType> block;
  Kokkos::View<const double *, DeviceType> source_real;
  Kokkos::View<const double *, DeviceType> source_imaginary;
  Kokkos::View<const double *, DeviceType> block_power_real;
  Kokkos::View<const double *, DeviceType> block_power_imaginary;
  Kokkos::View<double *, DeviceType> block_output_real;
  Kokkos::View<double *, DeviceType> block_output_imaginary;
  int chunk_begin = 0;
  int center_count = 0;
  int center_tiles = 0;
  int center_capacity = 0;
  int maximum_monomials = 0;

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamMember &team) const
  {
    const int local_plan = team.league_rank() / center_tiles;
    const int center_tile = team.league_rank() % center_tiles;
    const int lane = center_tile * team.team_size() + team.team_rank();
    if (lane >= center_count) return;
    const int atom_i = ilist(chunk_begin + lane);
    const int central = type_to_species(types(atom_i));
    if (central < 0) return;
    const auto species_block = block.species(central);
    if (local_plan >= species_block.plan_count) return;
    const auto plan = block.plans(species_block.plan_begin + local_plan);
    const bool direct_input = (plan.flags & YE3T_LAMMPS::YE3T_BLOCK_DIRECT_INPUT) != 0;
    const bool conjugate_half = (plan.flags & YE3T_LAMMPS::YE3T_BLOCK_CONJUGATE_HALF_OUTPUT) != 0;
    const bool real_coefficients = (plan.flags & YE3T_LAMMPS::YE3T_BLOCK_REAL_COEFFICIENTS) != 0;

    if (direct_input) {
      for (int component = 0; component < plan.output_count; ++component) {
        const int output = plan.output_begin + component;
        const int channel = block.direct_input_channels(output);
        if (channel < 0) continue;
        const double scale = block.direct_input_scales(output);
        const DeviceComplex input = read_full_channel(species, source_real, source_imaginary,
                                                      central, channel, center_capacity, lane);
        const int destination = (plan.output_storage_offset + component) * center_capacity + lane;
        block_output_real(destination) = scale * input.real;
        block_output_imaginary(destination) = scale * input.imaginary;
      }
    } else {
      double *scratch = static_cast<double *>(team.team_shmem().get_shmem(
          static_cast<std::size_t>(2 * team.team_size() * maximum_monomials) * sizeof(double)));
      const YE3T_LAMMPS::GpuStridedDoubles tmp_r{scratch,
                                                 static_cast<std::size_t>(team.team_size()),
                                                 static_cast<std::size_t>(team.team_rank())};
      const YE3T_LAMMPS::GpuStridedDoubles tmp_i{scratch + team.team_size() * maximum_monomials,
                                                 static_cast<std::size_t>(team.team_size()),
                                                 static_cast<std::size_t>(team.team_rank())};
      // A single view-like type is used for both components by the helper.
      const YE3T_LAMMPS::GpuConstStridedDoubles power_r{block_power_real.data(),
                                                        static_cast<std::size_t>(center_capacity),
                                                        static_cast<std::size_t>(lane)};
      const YE3T_LAMMPS::GpuConstStridedDoubles power_i{block_power_imaginary.data(),
                                                        static_cast<std::size_t>(center_capacity),
                                                        static_cast<std::size_t>(lane)};
      const YE3T_LAMMPS::GpuStridedDoubles out_r{block_output_real.data(),
                                                 static_cast<std::size_t>(center_capacity),
                                                 static_cast<std::size_t>(lane)};
      const YE3T_LAMMPS::GpuStridedDoubles out_i{block_output_imaginary.data(),
                                                 static_cast<std::size_t>(center_capacity),
                                                 static_cast<std::size_t>(lane)};
      YE3T_LAMMPS::gpu_block_forward_tiles(block, plan, species_block.plan_begin + local_plan,
                                           power_r, power_i, out_r, out_i, tmp_r, tmp_i,
                                           real_coefficients);
    }

    if (conjugate_half && plan.output_L > 0) {
      const int width = 2 * plan.output_L + 1;
      const int multiplicity = plan.output_count / width;
      for (int entry = 0; entry < multiplicity * plan.output_L; ++entry) {
        const int copy = entry / plan.output_L;
        const int magnetic = entry % plan.output_L + 1;
        const double phase = magnetic % 2 == 0 ? 1.0 : -1.0;
        const int negative_component = copy * width + plan.output_L - magnetic;
        const int positive_component = copy * width + plan.output_L + magnetic;
        const int negative =
            (plan.output_storage_offset + negative_component) * center_capacity + lane;
        const int positive =
            (plan.output_storage_offset + positive_component) * center_capacity + lane;
        block_output_real(negative) = phase * block_output_real(positive);
        block_output_imaginary(negative) = -phase * block_output_imaginary(positive);
      }
    }
  }
};

template <class DeviceType> struct BlockRouteWorkMajor {
  using AT = ArrayTypes<DeviceType>;
  using TeamMember = typename Kokkos::TeamPolicy<DeviceType>::member_type;
  typename AT::t_int_1d_randomread types;
  typename AT::t_int_1d_randomread ilist;
  Kokkos::View<const int *, DeviceType> type_to_species;
  YE3T_LAMMPS::YE3TKokkosBlockViews<DeviceType> block;
  Kokkos::View<const double *, DeviceType> block_output_real;
  Kokkos::View<const double *, DeviceType> block_output_imaginary;
  Kokkos::View<double *, DeviceType> block_output_adjoint_real;
  Kokkos::View<double *, DeviceType> block_output_adjoint_imaginary;
  Kokkos::View<double *, DeviceType> route_density_real;
  Kokkos::View<double *, DeviceType> route_density_imaginary;
  int chunk_begin = 0;
  int center_count = 0;
  int center_tiles = 0;
  int center_capacity = 0;

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamMember &team) const
  {
    const int local_route = team.league_rank() / center_tiles;
    const int center_tile = team.league_rank() % center_tiles;
    const int lane = center_tile * team.team_size() + team.team_rank();
    if (lane >= center_count) return;
    const int atom_i = ilist(chunk_begin + lane);
    const int central = type_to_species(types(atom_i));
    if (central < 0) return;
    const auto species_block = block.species(central);
    if (local_route >= species_block.route_count) return;
    const auto route = block.routes(species_block.route_begin + local_route);
    double density_real = 0.0;
    double density_imaginary = 0.0;

    for (int local_term = 0; local_term < route.term_count; ++local_term) {
      const int term = route.term_begin + local_term;
      const int begin = block.route_term_factor_offsets(term);
      const int end = block.route_term_factor_offsets(term + 1);
      DeviceComplex product{1.0, 0.0};
      bool product_initialized = false;
      for (int factor = begin; factor < end; ++factor) {
        const auto plan = block.plans(block.route_term_factor_plans(factor));
        const int component = block.route_term_factor_components(factor);
        const int source = (plan.output_storage_offset + component) * center_capacity + lane;
        const DeviceComplex factor_value{block_output_real(source), block_output_imaginary(source)};
        product = product_initialized ? multiply(product, factor_value) : factor_value;
        product_initialized = true;
      }
      const DeviceComplex coefficient{block.route_term_coefficient_real(term),
                                      block.route_term_coefficient_imaginary(term)};
      const DeviceComplex weighted = multiply_coefficient(coefficient, product);
      density_real += weighted.real;
      density_imaginary += weighted.imaginary;

      for (int target = begin; target < end; ++target) {
        DeviceComplex remaining{1.0, 0.0};
        bool remaining_initialized = false;
        for (int factor = begin; factor < end; ++factor) {
          if (factor == target) continue;
          const auto plan = block.plans(block.route_term_factor_plans(factor));
          const int component = block.route_term_factor_components(factor);
          const int source = (plan.output_storage_offset + component) * center_capacity + lane;
          const DeviceComplex factor_value{block_output_real(source),
                                           block_output_imaginary(source)};
          remaining = remaining_initialized ? multiply(remaining, factor_value) : factor_value;
          remaining_initialized = true;
        }
        const DeviceComplex derivative = multiply_coefficient(coefficient, remaining);
        const auto target_plan = block.plans(block.route_term_factor_plans(target));
        const int target_component = block.route_term_factor_components(target);
        const int destination =
            (target_plan.output_storage_offset + target_component) * center_capacity + lane;
        Kokkos::atomic_add(&block_output_adjoint_real(destination), derivative.real);
        Kokkos::atomic_add(&block_output_adjoint_imaginary(destination), -derivative.imaginary);
      }
    }
    const int destination = local_route * center_capacity + lane;
    route_density_real(destination) = density_real;
    route_density_imaginary(destination) = density_imaginary;
  }
};

template <class DeviceType> struct BlockPlanAdjointWorkMajor {
  using AT = ArrayTypes<DeviceType>;
  using TeamMember = typename Kokkos::TeamPolicy<DeviceType>::member_type;
  typename AT::t_int_1d_randomread types;
  typename AT::t_int_1d_randomread ilist;
  Kokkos::View<const int *, DeviceType> type_to_species;
  YE3T_LAMMPS::YE3TKokkosSpeciesViews<DeviceType> species;
  YE3T_LAMMPS::YE3TKokkosBlockViews<DeviceType> block;
  Kokkos::View<const double *, DeviceType> block_power_real;
  Kokkos::View<const double *, DeviceType> block_power_imaginary;
  Kokkos::View<double *, DeviceType> block_output_adjoint_real;
  Kokkos::View<double *, DeviceType> block_output_adjoint_imaginary;
  Kokkos::View<double *, DeviceType> source_adjoint_real;
  Kokkos::View<double *, DeviceType> source_adjoint_imaginary;
  int chunk_begin = 0;
  int center_count = 0;
  int center_tiles = 0;
  int center_capacity = 0;
  int maximum_monomials = 0;

  KOKKOS_INLINE_FUNCTION
  void operator()(const TeamMember &team) const
  {
    const int local_plan = team.league_rank() / center_tiles;
    const int center_tile = team.league_rank() % center_tiles;
    const int lane = center_tile * team.team_size() + team.team_rank();
    if (lane >= center_count) return;
    const int atom_i = ilist(chunk_begin + lane);
    const int central = type_to_species(types(atom_i));
    if (central < 0) return;
    const auto species_block = block.species(central);
    if (local_plan >= species_block.plan_count) return;
    const auto plan = block.plans(species_block.plan_begin + local_plan);
    const bool direct_input = (plan.flags & YE3T_LAMMPS::YE3T_BLOCK_DIRECT_INPUT) != 0;
    const bool conjugate_half = (plan.flags & YE3T_LAMMPS::YE3T_BLOCK_CONJUGATE_HALF_OUTPUT) != 0;
    const bool real_coefficients = (plan.flags & YE3T_LAMMPS::YE3T_BLOCK_REAL_COEFFICIENTS) != 0;
    const double embedding_scale = species.embedding_scales(central);

    if (conjugate_half && plan.output_L > 0) {
      const int width = 2 * plan.output_L + 1;
      const int multiplicity = plan.output_count / width;
      for (int entry = 0; entry < multiplicity * plan.output_L; ++entry) {
        const int copy = entry / plan.output_L;
        const int magnetic = entry % plan.output_L + 1;
        const double phase = magnetic % 2 == 0 ? 1.0 : -1.0;
        const int negative_component = copy * width + plan.output_L - magnetic;
        const int positive_component = copy * width + plan.output_L + magnetic;
        const int negative =
            (plan.output_storage_offset + negative_component) * center_capacity + lane;
        const int positive =
            (plan.output_storage_offset + positive_component) * center_capacity + lane;
        block_output_adjoint_real(positive) += phase * block_output_adjoint_real(negative);
        block_output_adjoint_imaginary(positive) -=
            phase * block_output_adjoint_imaginary(negative);
        block_output_adjoint_real(negative) = 0.0;
        block_output_adjoint_imaginary(negative) = 0.0;
      }
    }

    if (direct_input) {
      for (int component = 0; component < plan.output_count; ++component) {
        const int output = plan.output_begin + component;
        const int channel = block.direct_input_channels(output);
        if (channel < 0) continue;
        const int source = (plan.output_storage_offset + component) * center_capacity + lane;
        const double scale = embedding_scale * block.direct_input_scales(output);
        add_full_channel_adjoint(species, source_adjoint_real, source_adjoint_imaginary, central,
                                 channel, center_capacity, lane,
                                 scale * block_output_adjoint_real(source),
                                 scale * block_output_adjoint_imaginary(source));
      }
      return;
    }
    const YE3T_LAMMPS::GpuConstStridedDoubles power_r{block_power_real.data(),
                                                      static_cast<std::size_t>(center_capacity),
                                                      static_cast<std::size_t>(lane)};
    const YE3T_LAMMPS::GpuConstStridedDoubles power_i{block_power_imaginary.data(),
                                                      static_cast<std::size_t>(center_capacity),
                                                      static_cast<std::size_t>(lane)};
    const YE3T_LAMMPS::GpuStridedDoubles root_r{block_output_adjoint_real.data(),
                                                static_cast<std::size_t>(center_capacity),
                                                static_cast<std::size_t>(lane)};
    const YE3T_LAMMPS::GpuStridedDoubles root_i{block_output_adjoint_imaginary.data(),
                                                static_cast<std::size_t>(center_capacity),
                                                static_cast<std::size_t>(lane)};
    const auto add = [&](int channel, double gr, double gi) {
      add_full_channel_adjoint(species, source_adjoint_real, source_adjoint_imaginary, central,
                               channel, center_capacity, lane, gr, gi);
    };
    YE3T_LAMMPS::gpu_block_adjoint_transpose(block, plan, power_r, power_i, root_r, root_i, add,
                                             embedding_scale, real_coefficients);
  }
};

template <class DeviceType> struct FinalizeTotalDensity {
  using AT = ArrayTypes<DeviceType>;
  typename AT::t_int_1d_randomread types;
  typename AT::t_int_1d_randomread ilist;
  Kokkos::View<const int *, DeviceType> type_to_species;
  YE3T_LAMMPS::YE3TKokkosSpeciesViews<DeviceType> species;
  YE3T_LAMMPS::YE3TKokkosBlockViews<DeviceType> block;
  Kokkos::View<const double *, DeviceType> total_density_real;
  Kokkos::View<const double *, DeviceType> total_density_imaginary;
  Kokkos::View<const double *, DeviceType> route_density_real;
  Kokkos::View<const double *, DeviceType> route_density_imaginary;
  Kokkos::View<double *, DeviceType> atomic_energies;
  Kokkos::View<int *, DeviceType> status;
  int chunk_begin = 0;
  int center_capacity = 0;
  bool include_route_density = false;

  using value_type = double;

  KOKKOS_INLINE_FUNCTION
  void operator()(int lane, double &maximum_imaginary) const
  {
    const int atom_i = ilist(chunk_begin + lane);
    const int central = type_to_species(types(atom_i));
    if (central < 0) {
      atomic_energies(lane) = 0.0;
      return;
    }
    double density_real = total_density_real(lane);
    double density_imaginary = total_density_imaginary(lane);
    if (include_route_density) {
      const auto species_block = block.species(central);
      for (int route = 0; route < species_block.route_count; ++route) {
        const int source = route * center_capacity + lane;
        density_real += route_density_real(source);
        density_imaginary += route_density_imaginary(source);
      }
    }
    if (density_real >= species.density_safe_limits(central))
      set_device_status(status.data(), YE3T_STATUS_DENSITY_LIMIT);
    atomic_energies(lane) =
        species.reference_energies(central) + species.embedding_scales(central) * density_real;
    const double imaginary = Kokkos::abs(density_imaginary);
    if (imaginary > maximum_imaginary) maximum_imaginary = imaginary;
  }
};

template <class DeviceType> struct SourceVJP {
  YE3T_LAMMPS::YE3TKokkosBondViews<DeviceType> bonds;
  YE3T_LAMMPS::YE3TKokkosAngularViews<DeviceType> angular;
  Kokkos::View<const int *, DeviceType> edge_centers;
  Kokkos::View<const int *, DeviceType> edge_bonds;
  Kokkos::View<const double *, DeviceType> edge_radius;
  Kokkos::View<const double *, DeviceType> edge_unit;
  Kokkos::View<const double *, DeviceType> source_adjoint_real;
  Kokkos::View<const double *, DeviceType> source_adjoint_imaginary;
  Kokkos::View<double *, DeviceType> edge_gradient;
  Kokkos::View<int *, DeviceType> status;
  int center_capacity = 0;
  int edge_capacity = 0;

  KOKKOS_INLINE_FUNCTION
  void operator()(int edge) const
  {
    const int lane = edge_centers(edge);
    const int bond = edge_bonds(edge);
    const double radius = edge_radius(edge);
    if (!(radius > 1.0e-14)) {
      set_device_status(status.data(), YE3T_STATUS_ZERO_RADIUS);
      return;
    }
    const double direction[3] = {edge_unit(edge), edge_unit(edge_capacity + edge),
                                 edge_unit(2 * edge_capacity + edge)};
    const int interval_count = bonds.interval_counts(bond);
    const double cutoff = bonds.cutoffs(bond);
    double gradient[3] = {0.0, 0.0, 0.0};

    for (int term = bonds.radial_map_offsets(bond); term < bonds.radial_map_offsets(bond + 1);
         ++term) {
      const int output = bonds.radial_channel_outputs(term);
      const double root = source_adjoint_real(output * center_capacity + lane);
      const SplineValue radial = evaluate_spline(
          bonds.radial_splines, bonds.radial_spline_offsets(bond), bonds.radial_base_counts(bond),
          interval_count, cutoff, bonds.radial_channel_indices(term), radius);
      if (!radial.valid) {
        set_device_status(status.data(), YE3T_STATUS_SPLINE_INTERVAL);
        continue;
      }
      const double weight = root * radial.derivative;
      for (int axis = 0; axis < 3; ++axis) gradient[axis] += weight * direction[axis];
    }

    const int contracted_width = bonds.radial_counts(bond) * (bonds.angular_maxima(bond) + 1);
    for (int term = bonds.angular_map_offsets(bond); term < bonds.angular_map_offsets(bond + 1);
         ++term) {
      const int output = bonds.angular_channel_outputs(term);
      const double root_real = source_adjoint_real(output * center_capacity + lane);
      const double root_imaginary = source_adjoint_imaginary(output * center_capacity + lane);
      const SplineValue radial = evaluate_spline(
          bonds.contracted_splines, bonds.contracted_spline_offsets(bond), contracted_width,
          interval_count, cutoff, bonds.contracted_channel_indices(term), radius);
      const HarmonicValue harmonic =
          evaluate_harmonic(angular, bonds.angular_channel_indices(term), direction, radius);
      if (!radial.valid || !harmonic.valid) {
        set_device_status(status.data(),
                          radial.valid ? YE3T_STATUS_ZERO_RADIUS : YE3T_STATUS_SPLINE_INTERVAL);
        continue;
      }
      const double radial_weight =
          radial.derivative * (root_real * harmonic.real + root_imaginary * harmonic.imaginary);
      const double angular_weight_real = radial.value * root_real;
      const double angular_weight_imaginary = radial.value * root_imaginary;
      for (int axis = 0; axis < 3; ++axis)
        gradient[axis] += radial_weight * direction[axis] +
            angular_weight_real * harmonic.gradient_real[axis] +
            angular_weight_imaginary * harmonic.gradient_imaginary[axis];
    }
    for (int axis = 0; axis < 3; ++axis)
      edge_gradient(axis * edge_capacity + edge) = gradient[axis];
  }
};

template <class DeviceType> struct RadialCachedSourceVJP {
  YE3T_LAMMPS::YE3TKokkosBondViews<DeviceType> bonds;
  YE3T_LAMMPS::YE3TKokkosAngularViews<DeviceType> angular;
  Kokkos::View<const int *, DeviceType> edge_centers;
  Kokkos::View<const int *, DeviceType> edge_bonds;
  Kokkos::View<const double *, DeviceType> edge_radius;
  Kokkos::View<const double *, DeviceType> edge_unit;
  Kokkos::View<const double *, DeviceType> radial_derivative;
  Kokkos::View<const double *, DeviceType> contracted_value;
  Kokkos::View<const double *, DeviceType> contracted_derivative;
  Kokkos::View<const double *, DeviceType> source_adjoint_real;
  Kokkos::View<const double *, DeviceType> source_adjoint_imaginary;
  Kokkos::View<double *, DeviceType> edge_gradient;
  Kokkos::View<int *, DeviceType> status;
  int center_capacity = 0;
  int edge_capacity = 0;

  KOKKOS_INLINE_FUNCTION
  void operator()(int edge) const
  {
    const int lane = edge_centers(edge);
    const int bond = edge_bonds(edge);
    const double radius = edge_radius(edge);
    if (!(radius > 1.0e-14)) {
      set_device_status(status.data(), YE3T_STATUS_ZERO_RADIUS);
      return;
    }
    const double direction[3] = {edge_unit(edge), edge_unit(edge_capacity + edge),
                                 edge_unit(2 * edge_capacity + edge)};
    double gradient[3] = {0.0, 0.0, 0.0};

    for (int term = bonds.radial_map_offsets(bond); term < bonds.radial_map_offsets(bond + 1);
         ++term) {
      const int output = bonds.radial_channel_outputs(term);
      const int function = bonds.radial_channel_indices(term);
      const double root = source_adjoint_real(output * center_capacity + lane);
      const std::size_t cache = static_cast<std::size_t>(function) * edge_capacity + edge;
      const double weight = root * radial_derivative(cache);
      for (int axis = 0; axis < 3; ++axis) gradient[axis] += weight * direction[axis];
    }

    YE3T_LAMMPS::GpuHarmonicStream<true, decltype(angular.recurrence)> harmonics(
        angular.recurrence, angular.maximum_angular_momentum, direction, radius);
    for (int group = bonds.vjp_bond_group_offsets(bond);
         group < bonds.vjp_bond_group_offsets(bond + 1); ++group) {
      const auto harmonic_value =
          harmonics.advance(bonds.vjp_group_degrees(group), bonds.vjp_group_orders(group));
      if (!harmonic_value.valid) {
        set_device_status(status.data(), YE3T_STATUS_ZERO_RADIUS);
        continue;
      }
      for (int entry = bonds.vjp_group_term_offsets(group);
           entry < bonds.vjp_group_term_offsets(group + 1); ++entry) {
        const int term = bonds.vjp_group_terms(entry);
        const int output = bonds.angular_channel_outputs(term);
        const int radial = bonds.contracted_channel_indices(term);
        const double root_real = source_adjoint_real(output * center_capacity + lane);
        const double root_imaginary = source_adjoint_imaginary(output * center_capacity + lane);
        const std::size_t radial_cache = static_cast<std::size_t>(radial) * edge_capacity + edge;
        const double radial_weight = contracted_derivative(radial_cache) *
            (root_real * harmonic_value.real + root_imaginary * harmonic_value.imaginary);
        const double angular_weight_real = contracted_value(radial_cache) * root_real;
        const double angular_weight_imaginary = contracted_value(radial_cache) * root_imaginary;
        for (int axis = 0; axis < 3; ++axis)
          gradient[axis] += radial_weight * direction[axis] +
              angular_weight_real * harmonic_value.gradient_real[axis] +
              angular_weight_imaginary * harmonic_value.gradient_imaginary[axis];
      }
    }
    for (int axis = 0; axis < 3; ++axis)
      edge_gradient(axis * edge_capacity + edge) = gradient[axis];
  }
};

template <class DeviceType, int NEIGHFLAG, bool TALLY_ATOM_VIRIAL> struct ScatterForceEnergyVirial {
  using PairType = PairYE3TKokkos<DeviceType>;
  using AT = ArrayTypes<DeviceType>;
  using value_type = EV_FLOAT;
  typename ye3t_kokkos::ArrayAliases<DeviceType>::t_positions_randomread positions;
  typename AT::t_int_1d_randomread ilist;
  Kokkos::View<const std::int64_t *, DeviceType> offsets;
  Kokkos::View<const int *, DeviceType> edge_neighbors;
  Kokkos::View<const double *, DeviceType> edge_gradient;
  Kokkos::View<const double *, DeviceType> atomic_energies;
  typename ye3t_kokkos::ArrayAliases<DeviceType>::t_eatom eatom;
  typename ye3t_kokkos::ArrayAliases<DeviceType>::t_vatom vatom;
  typename PairType::DupForceView duplicated_force;
  typename PairType::NonDupForceView atomic_force;
  typename PairType::DupVirialView duplicated_vatom;
  typename PairType::NonDupVirialView atomic_vatom;
  int chunk_begin = 0;
  int edge_capacity = 0;
  int eflag_global = 0;
  int eflag_atom = 0;
  int vflag_global = 0;
  int vflag_either = 0;

  // Force-only dispatch uses the same scatter/ownership implementation, but
  // avoids an EV reduction and its host result. Tallying uses the overload below.
  KOKKOS_INLINE_FUNCTION
  void operator()(int lane) const
  {
    EV_FLOAT unused_ev{};
    (*this)(lane, unused_ev);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(int lane, EV_FLOAT &ev) const
  {
    const auto force_scatter =
        ScatterViewHelper<NeedDup_v<NEIGHFLAG, DeviceType>, typename PairType::DupForceView,
                          typename PairType::NonDupForceView>::get(duplicated_force, atomic_force);
    const auto force_access = force_scatter.template access<AtomicDup_v<NEIGHFLAG, DeviceType>>();
    const int atom_i = ilist(chunk_begin + lane);
    double center_force[3] = {0.0, 0.0, 0.0};
    for (std::int64_t edge_64 = offsets(lane); edge_64 < offsets(lane + 1); ++edge_64) {
      const int edge = static_cast<int>(edge_64);
      const int atom_j = edge_neighbors(edge);
      const double gradient[3] = {edge_gradient(edge), edge_gradient(edge_capacity + edge),
                                  edge_gradient(2 * edge_capacity + edge)};
      for (int axis = 0; axis < 3; ++axis) {
        center_force[axis] += gradient[axis];
        force_access(atom_j, axis) -= gradient[axis];
      }
      if (vflag_global || TALLY_ATOM_VIRIAL) {
        const double delta[3] = {positions(atom_i, 0) - positions(atom_j, 0),
                                 positions(atom_i, 1) - positions(atom_j, 1),
                                 positions(atom_i, 2) - positions(atom_j, 2)};
        const double contribution[6] = {delta[0] * gradient[0], delta[1] * gradient[1],
                                        delta[2] * gradient[2], delta[0] * gradient[1],
                                        delta[0] * gradient[2], delta[1] * gradient[2]};
        if (vflag_global)
          for (int component = 0; component < 6; ++component)
            ev.v[component] += contribution[component];
        if constexpr (TALLY_ATOM_VIRIAL) {
          const auto virial_scatter =
              ScatterViewHelper<NeedDup_v<NEIGHFLAG, DeviceType>, typename PairType::DupVirialView,
                                typename PairType::NonDupVirialView>::get(duplicated_vatom,
                                                                          atomic_vatom);
          const auto virial_access =
              virial_scatter.template access<AtomicDup_v<NEIGHFLAG, DeviceType>>();
          for (int component = 0; component < 6; ++component) {
            virial_access(atom_i, component) += 0.5 * contribution[component];
            virial_access(atom_j, component) += 0.5 * contribution[component];
          }
        }
      }
    }
    for (int axis = 0; axis < 3; ++axis) force_access(atom_i, axis) += center_force[axis];

    const double energy = atomic_energies(lane);
    if (eflag_global) ev.evdwl += energy;
    if (eflag_atom) eatom(atom_i) += energy;
  }
};

}    // namespace

template <class DeviceType> PairYE3TKokkos<DeviceType>::PairYE3TKokkos(LAMMPS *lmp) : PairYE3T(lmp)
{
  respa_enable = 0;
  kokkosable = 1;
  atomKK = static_cast<AtomKokkos *>(atom);
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK;
  datamask_modify = F_MASK;
}

template <class DeviceType> PairYE3TKokkos<DeviceType>::~PairYE3TKokkos()
{
  if (copymode) return;
  memoryKK->destroy_kokkos(k_eatom_, eatom);
  memoryKK->destroy_kokkos(k_vatom_, vatom);
}

template <class DeviceType>
std::pair<std::size_t, std::size_t> PairYE3TKokkos<DeviceType>::device_memory_info() const
{
#if defined(KOKKOS_ENABLE_CUDA)
  if constexpr (std::is_same_v<DeviceType, Kokkos::Cuda>) {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess)
      throw std::runtime_error(std::string("cudaMemGetInfo failed: ") + cudaGetErrorString(status));
    return {free_bytes, total_bytes};
  }
#elif defined(KOKKOS_ENABLE_HIP)
  if constexpr (std::is_same_v<DeviceType, Kokkos::HIP>) {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    const hipError_t status = hipMemGetInfo(&free_bytes, &total_bytes);
    if (status != hipSuccess)
      throw std::runtime_error(std::string("hipMemGetInfo failed: ") + hipGetErrorString(status));
    return {free_bytes, total_bytes};
  }
#endif
  throw std::runtime_error("device-memory preflight is not implemented for this Kokkos backend");
}

template <class DeviceType> void PairYE3TKokkos<DeviceType>::initialize_rank_device_identity()
{
  YE3TRankDeviceRecord local;
  MPI_Comm_rank(world, &local.world_rank);
  MPI_Comm_size(world, &local.world_size);
  local.launcher_local_rank = launcher_local_rank();
  local.gpu_aware_mpi = lmp->kokkos->gpu_aware_flag;
  int hostname_length = 0;
  MPI_Get_processor_name(local.hostname, &hostname_length);
  if (hostname_length < 0 || hostname_length >= MPI_MAX_PROCESSOR_NAME)
    error->one(FLERR, "MPI returned an invalid processor name for ye3t/kk");
  local.hostname[hostname_length] = '\0';

  std::vector<char> hostnames(static_cast<std::size_t>(local.world_size) * MPI_MAX_PROCESSOR_NAME,
                              0);
  MPI_Allgather(local.hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, hostnames.data(),
                MPI_MAX_PROCESSOR_NAME, MPI_CHAR, world);
  for (int rank = 0; rank < local.world_size; ++rank) {
    const char *hostname = hostnames.data() + rank * MPI_MAX_PROCESSOR_NAME;
    if (std::strcmp(local.hostname, hostname) == 0) {
      if (rank < local.world_rank) ++local.local_rank;
      ++local.local_size;
    }
  }
  ++local.local_rank;

  std::string device_class_signature;
#if defined(KOKKOS_ENABLE_CUDA)
  if constexpr (std::is_same_v<DeviceType, Kokkos::Cuda>) {
    cudaError_t status = cudaGetDevice(&local.device_ordinal);
    if (status != cudaSuccess)
      error->one(FLERR, "cudaGetDevice failed for ye3t/kk: {}", cudaGetErrorString(status));
    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, local.device_ordinal);
    if (status != cudaSuccess)
      error->one(FLERR, "cudaGetDeviceProperties failed for ye3t/kk: {}",
                 cudaGetErrorString(status));
    status = cudaDeviceGetPCIBusId(local.pci_bus_id, static_cast<int>(sizeof(local.pci_bus_id)),
                                   local.device_ordinal);
    if (status != cudaSuccess)
      error->one(FLERR, "cudaDeviceGetPCIBusId failed for ye3t/kk: {}", cudaGetErrorString(status));
    status = cudaRuntimeGetVersion(&local.runtime_version);
    if (status != cudaSuccess)
      error->one(FLERR, "cudaRuntimeGetVersion failed for ye3t/kk: {}", cudaGetErrorString(status));
    status = cudaDriverGetVersion(&local.driver_version);
    if (status != cudaSuccess)
      error->one(FLERR, "cudaDriverGetVersion failed for ye3t/kk: {}", cudaGetErrorString(status));
    local.total_device_bytes = properties.totalGlobalMem;
    store_text(local.execution_space, "Cuda");
#if CUDART_VERSION >= 10000
    store_text(local.uuid,
               format_device_uuid(reinterpret_cast<const unsigned char *>(properties.uuid.bytes)));
    store_text(local.uuid_status, "available");
#else
    store_text(local.uuid, "unavailable");
    store_text(local.uuid_status, "unavailable");
#endif
    store_text(local.visible_mask_hash, visible_device_mask_hash(true));
    device_class_signature = std::string("Cuda;") + properties.name + ";" +
        std::to_string(properties.major) + "." + std::to_string(properties.minor) + ";" +
        std::to_string(properties.multiProcessorCount) + ";" + std::to_string(properties.warpSize) +
        ";" + std::to_string(properties.totalGlobalMem) + ";" +
        std::to_string(local.runtime_version) + ";" + std::to_string(local.driver_version);
  }
#elif defined(KOKKOS_ENABLE_HIP)
  if constexpr (std::is_same_v<DeviceType, Kokkos::HIP>) {
    hipError_t status = hipGetDevice(&local.device_ordinal);
    if (status != hipSuccess)
      error->one(FLERR, "hipGetDevice failed for ye3t/kk: {}", hipGetErrorString(status));
    hipDeviceProp_t properties{};
    status = hipGetDeviceProperties(&properties, local.device_ordinal);
    if (status != hipSuccess)
      error->one(FLERR, "hipGetDeviceProperties failed for ye3t/kk: {}", hipGetErrorString(status));
    status = hipDeviceGetPCIBusId(local.pci_bus_id, static_cast<int>(sizeof(local.pci_bus_id)),
                                  local.device_ordinal);
    if (status != hipSuccess)
      error->one(FLERR, "hipDeviceGetPCIBusId failed for ye3t/kk: {}", hipGetErrorString(status));
    status = hipRuntimeGetVersion(&local.runtime_version);
    if (status != hipSuccess)
      error->one(FLERR, "hipRuntimeGetVersion failed for ye3t/kk: {}", hipGetErrorString(status));
    status = hipDriverGetVersion(&local.driver_version);
    if (status != hipSuccess)
      error->one(FLERR, "hipDriverGetVersion failed for ye3t/kk: {}", hipGetErrorString(status));
    local.total_device_bytes = properties.totalGlobalMem;
    store_text(local.execution_space, "HIP");
    store_text(local.uuid, "unavailable");
    store_text(local.uuid_status, "unavailable");
    store_text(local.visible_mask_hash, visible_device_mask_hash(false));
    device_class_signature = std::string("HIP;") + properties.name + ";" + properties.gcnArchName +
        ";" + std::to_string(properties.multiProcessorCount) + ";" +
        std::to_string(properties.warpSize) + ";" + std::to_string(properties.totalGlobalMem) +
        ";" + std::to_string(local.runtime_version) + ";" + std::to_string(local.driver_version);
  }
#endif
  if (device_class_signature.empty())
    error->one(FLERR, "Pair style ye3t/kk cannot identify its device backend");
  device_class_hash_ = YE3T_LAMMPS::sha256_string(device_class_signature);
  store_text(local.device_class_hash, device_class_hash_);

  std::vector<YE3TRankDeviceRecord> records(
      comm->me == 0 ? static_cast<std::size_t>(local.world_size) : 0);
  MPI_Gather(&local, static_cast<int>(sizeof(local)), MPI_BYTE,
             records.empty() ? nullptr : records.data(), static_cast<int>(sizeof(local)), MPI_BYTE,
             0, world);

  std::string rank_map_hash;
  if (comm->me == 0) {
    std::string canonical;
    int node_count = 0;
    bool all_uuids = true;
    bool one_device_class = true;
    for (int rank = 0; rank < local.world_size; ++rank) {
      const auto &record = records[rank];
      canonical += std::to_string(record.world_rank) + "|" + record.hostname + "|" +
          std::to_string(record.local_rank) + "|" + std::to_string(record.device_ordinal) + "|" +
          record.uuid + "|" + record.pci_bus_id + "|" + record.device_class_hash + "|" +
          record.visible_mask_hash + ";";
      all_uuids = all_uuids && std::strcmp(record.uuid_status, "available") == 0;
      one_device_class = one_device_class &&
          std::strcmp(records[0].device_class_hash, record.device_class_hash) == 0;
      bool first_on_node = true;
      for (int previous = 0; previous < rank; ++previous)
        first_on_node =
            first_on_node && std::strcmp(records[previous].hostname, record.hostname) != 0;
      if (first_on_node) ++node_count;
      utils::logmesg(lmp,
                     "YE3T Kokkos rank-device: world_rank {}, world_size {}, local_rank "
                     "{}, local_size {}, launcher_local_rank {}, hostname {}, "
                     "execution_space {}, device_ordinal {}, uuid {}, uuid_status {}, "
                     "pci_bus_id {}, device_class {}, visible_mask {}, gpu_aware_mpi {}, "
                     "runtime_version {}, driver_version {}, total_device_bytes {}\n",
                     record.world_rank, record.world_size, record.local_rank, record.local_size,
                     record.launcher_local_rank, record.hostname, record.execution_space,
                     record.device_ordinal, record.uuid, record.uuid_status, record.pci_bus_id,
                     record.device_class_hash, record.visible_mask_hash, record.gpu_aware_mpi,
                     record.runtime_version, record.driver_version, record.total_device_bytes);
    }
    rank_map_hash = YE3T_LAMMPS::sha256_string(canonical);
    utils::logmesg(lmp,
                   "YE3T Kokkos rank-device summary: world_size {}, node_count {}, "
                   "rank_map {}, uuid_status {}, device_class_consensus {}\n",
                   local.world_size, node_count, rank_map_hash,
                   all_uuids ? "complete" : "unavailable", one_device_class ? "passed" : "mixed");
  }
  std::array<char, 65> root_hash{};
  if (comm->me == 0) std::memcpy(root_hash.data(), rank_map_hash.c_str(), rank_map_hash.size());
  MPI_Bcast(root_hash.data(), static_cast<int>(root_hash.size()), MPI_CHAR, 0, world);
  rank_device_map_hash_ = root_hash.data();
}

template <class DeviceType>
void PairYE3TKokkos<DeviceType>::report_initial_capacity_state(int inum, int requested_chunk,
                                                               int effective_chunk,
                                                               int center_reduced,
                                                               int edge_reductions,
                                                               int reallocations_before)
{
  Kokkos::fence("YE3T initial-capacity report");
  const auto [free_bytes, total_bytes] = device_memory_info();
  YE3TCapacityRecord local;
  local.world_rank = comm->me;
  local.configured_chunk = chunksize();
  local.local_inum = inum;
  local.requested_chunk = requested_chunk;
  local.effective_chunk = effective_chunk;
  local.center_capacity = center_capacity_;
  local.edge_capacity = edge_capacity_;
  local.reallocations_before = reallocations_before;
  local.reallocations_after = reallocation_count_;
  local.center_reduced = center_reduced;
  local.edge_reductions = edge_reductions;
  local.exact_chunk_required = require_exact_chunksize_ ? 1 : 0;
  local.dynamic_bytes = dynamic_bytes_;
  const std::size_t plan_bytes = is_lifted_cauchy_model()
      ? lifted_device_plan_.memory_usage()
      : (is_tagged_cauchy_model() ? tagged_device_plan_.memory_usage()
                                  : device_plan_.memory_usage());
  const std::size_t fixed_bytes =
      (type_to_species_.extent(0) * sizeof(int) + step_state_.memory_usage());
  local.resident_bytes = checked_add_bytes(
      checked_add_bytes(plan_bytes, fixed_bytes, "capacity-report resident storage"),
      dynamic_bytes_, "capacity-report resident storage");
  local.free_device_bytes = free_bytes;
  local.total_device_bytes = total_bytes;
  int world_size = 0;
  MPI_Comm_size(world, &world_size);
  std::vector<YE3TCapacityRecord> records(comm->me == 0 ? static_cast<std::size_t>(world_size) : 0);
  MPI_Gather(&local, static_cast<int>(sizeof(local)), MPI_BYTE,
             records.empty() ? nullptr : records.data(), static_cast<int>(sizeof(local)), MPI_BYTE,
             0, world);
  if (comm->me == 0)
    for (const auto &record : records)
      utils::logmesg(lmp,
                     "YE3T Kokkos capacity: world_rank {}, configured_chunk {}, "
                     "local_inum {}, requested_chunk {}, effective_chunk {}, "
                     "center_capacity {}, edge_capacity {}, dynamic_bytes {}, "
                     "resident_bytes {}, reallocations_before {}, reallocations_after {}, "
                     "center_reduced {}, edge_reductions {}, free_device_bytes {}, "
                     "total_device_bytes {}, exact_chunk_required {}\n",
                     record.world_rank, record.configured_chunk, record.local_inum,
                     record.requested_chunk, record.effective_chunk, record.center_capacity,
                     record.edge_capacity, record.dynamic_bytes, record.resident_bytes,
                     record.reallocations_before, record.reallocations_after, record.center_reduced,
                     record.edge_reductions, record.free_device_bytes, record.total_device_bytes,
                     record.exact_chunk_required);
  capacity_state_reported_ = true;
}

template <class DeviceType> std::size_t PairYE3TKokkos<DeviceType>::allocation_budget() const
{
  const auto [free_bytes, total_bytes] = device_memory_info();
  const std::size_t reserve =
      checked_add_bytes(checked_multiply_bytes(total_bytes / 100, 15, "headroom reserve"),
                        ((total_bytes % 100) * 15 + 99) / 100, "headroom reserve");
  constexpr std::size_t allocation_guard = 1024 * 1024;
  if (reserve >= free_bytes || allocation_guard >= free_bytes - reserve) return 0;
  return free_bytes - reserve - allocation_guard;
}

template <class DeviceType>
std::size_t PairYE3TKokkos<DeviceType>::center_storage_bytes(int capacity) const
{
  if (capacity < 0) throw std::invalid_argument("negative YE3T Kokkos center capacity");
  if (capacity == 0) return 0;
  const std::size_t centers = static_cast<std::size_t>(capacity);
  const std::size_t offset_entries = checked_add_bytes(
      checked_multiply_bytes(centers, 2, "center index storage"), 1, "center index storage");
  const std::size_t index_bytes =
      checked_multiply_bytes(offset_entries, sizeof(std::int64_t), "center index storage");
  if (is_tagged_cauchy_model()) {
    const std::size_t density_values =
        checked_multiply_bytes(static_cast<std::size_t>(tagged_device_plan_.density_key_count()), 2,
                               "tagged density and adjoint storage");
    const std::size_t moment_values =
        checked_multiply_bytes(static_cast<std::size_t>(tagged_device_plan_.moment_key_count()), 2,
                               "tagged moment and adjoint storage");
    const std::size_t real_values = checked_add_bytes(
        checked_add_bytes(density_values, moment_values, "tagged center real storage"), 1,
        "tagged center real storage");
    const std::size_t real_bytes = checked_multiply_bytes(
        checked_multiply_bytes(real_values, centers, "tagged center real storage"), sizeof(double),
        "tagged center real storage");
    return checked_add_bytes(index_bytes, real_bytes, "tagged center workspace storage");
  }
  if (is_lifted_cauchy_model()) {
    const std::size_t source_values = checked_multiply_bytes(
        static_cast<std::size_t>(lifted_device_plan_.source_variable_count()), 2,
        "lifted source and adjoint storage");
    const std::size_t real_values =
        checked_add_bytes(checked_add_bytes(source_values,
                                            static_cast<std::size_t>(
                                                lifted_device_plan_.workspace_values_per_center()),
                                            "lifted center real storage"),
                          1, "lifted center real storage");
    const std::size_t real_bytes = checked_multiply_bytes(
        checked_multiply_bytes(real_values, centers, "lifted center real storage"), sizeof(double),
        "lifted center real storage");
    return checked_add_bytes(index_bytes, real_bytes, "lifted center workspace storage");
  }
  const std::size_t real_bytes = checked_multiply_bytes(
      device_plan_.bytes_per_center(direct_resident_values_ > 0), centers, "center real storage");
  return checked_add_bytes(index_bytes, real_bytes, "center workspace storage");
}

template <class DeviceType>
std::size_t PairYE3TKokkos<DeviceType>::edge_storage_bytes(int capacity) const
{
  if (capacity < 0) throw std::invalid_argument("negative YE3T Kokkos edge capacity");
  const std::size_t edges = static_cast<std::size_t>(capacity);
  const std::size_t index_bytes = checked_multiply_bytes(
      checked_multiply_bytes(edges, 3, "edge index storage"), sizeof(int), "edge index storage");
  const std::size_t gradient_bytes =
      checked_multiply_bytes(checked_multiply_bytes(edges, 3, "edge gradient storage"),
                             sizeof(double), "edge gradient storage");
  const std::size_t geometry_bytes =
      checked_multiply_bytes(checked_multiply_bytes(edges, 4, "edge geometry storage"),
                             sizeof(double), "edge geometry storage");
  const std::size_t cache_bytes = checked_multiply_bytes(
      checked_multiply_bytes(edges, static_cast<std::size_t>(edge_basis_cache_value_count_),
                             "edge basis-cache storage"),
      sizeof(double), "edge basis-cache storage");
  return checked_add_bytes(
      checked_add_bytes(checked_add_bytes(index_bytes, gradient_bytes, "edge workspace storage"),
                        geometry_bytes, "edge workspace storage"),
      cache_bytes, "edge workspace storage");
}

template <class DeviceType>
bool PairYE3TKokkos<DeviceType>::allocation_preflight(std::size_t new_bytes) const
{
  return new_bytes <= allocation_budget();
}

template <class DeviceType> int PairYE3TKokkos<DeviceType>::ensure_center_capacity(int requested)
{
  if (is_lifted_cauchy_model()) return ensure_lifted_center_capacity(requested);
  if (is_tagged_cauchy_model()) return ensure_tagged_center_capacity(requested);
  if (requested <= center_capacity_) return requested;
  std::int64_t grown = static_cast<std::int64_t>(center_capacity_) + center_capacity_ / 2 + 64;
  int candidate = static_cast<int>(std::min<std::int64_t>(
      std::numeric_limits<int>::max(), std::max<std::int64_t>(requested, grown)));
  bool tried_requested = candidate == requested;
  while (candidate > 0) {
    const std::size_t new_bytes = center_storage_bytes(candidate);
    if (allocation_preflight(new_bytes)) {
      OffsetView new_counts("ye3t:center_counts", candidate);
      OffsetView new_offsets("ye3t:center_offsets", candidate + 1);
      const std::size_t source_size =
          checked_multiply_bytes(static_cast<std::size_t>(device_plan_.maximum_source_count()),
                                 static_cast<std::size_t>(candidate), "source workspace extent");
      const std::size_t value_size = checked_multiply_bytes(
          static_cast<std::size_t>(direct_resident_values_ ? 0
                                                           : device_plan_.maximum_value_count()),
          static_cast<std::size_t>(candidate), "DAG workspace extent");
      RealView new_source_real("ye3t:source_real", source_size);
      RealView new_source_imaginary("ye3t:source_imaginary", source_size);
      RealView new_source_adjoint_real("ye3t:source_adjoint_real", source_size);
      RealView new_source_adjoint_imaginary("ye3t:source_adjoint_imaginary", source_size);
      RealView new_dag_real("ye3t:dag_real", value_size);
      RealView new_dag_imaginary("ye3t:dag_imaginary", value_size);
      RealView new_dag_adjoint_real("ye3t:dag_adjoint_real", value_size);
      RealView new_dag_adjoint_imaginary("ye3t:dag_adjoint_imaginary", value_size);
      RealView new_total_density_real;
      RealView new_total_density_imaginary;
      RealView new_block_power_real;
      RealView new_block_power_imaginary;
      RealView new_block_output_real;
      RealView new_block_output_imaginary;
      RealView new_block_output_adjoint_real;
      RealView new_block_output_adjoint_imaginary;
      RealView new_block_monomial_real;
      RealView new_block_monomial_imaginary;
      RealView new_block_monomial_adjoint_real;
      RealView new_block_monomial_adjoint_imaginary;
      RealView new_block_route_density_real;
      RealView new_block_route_density_imaginary;
      RealView new_scalar_value_real;
      RealView new_scalar_value_imaginary;
      RealView new_scalar_adjoint_real;
      RealView new_scalar_adjoint_imaginary;
      RealView new_coupled_value_real;
      RealView new_coupled_value_imaginary;
      RealView new_coupled_adjoint_real;
      RealView new_coupled_adjoint_imaginary;
      if (device_plan_.has_optimized_program()) {
        new_total_density_real = RealView("ye3t:total_density_real", candidate);
        new_total_density_imaginary = RealView("ye3t:total_density_imaginary", candidate);
      }
      if (device_plan_.has_block_program()) {
        const std::size_t block_power_size = checked_multiply_bytes(
            static_cast<std::size_t>(device_plan_.maximum_block_power_storage()),
            static_cast<std::size_t>(candidate), "block power workspace extent");
        const std::size_t block_output_size = checked_multiply_bytes(
            static_cast<std::size_t>(device_plan_.maximum_block_output_storage()),
            static_cast<std::size_t>(candidate), "block output workspace extent");
        const std::size_t block_monomial_size = checked_multiply_bytes(
            static_cast<std::size_t>(device_plan_.block_monomial_tile_count()),
            static_cast<std::size_t>(candidate), "block monomial workspace extent");
        const std::size_t block_route_density_size = checked_multiply_bytes(
            static_cast<std::size_t>(device_plan_.maximum_block_route_count()),
            static_cast<std::size_t>(candidate), "block route-density workspace extent");
        new_block_power_real = RealView("ye3t:block_power_real", block_power_size);
        new_block_power_imaginary = RealView("ye3t:block_power_imaginary", block_power_size);
        new_block_output_real = RealView("ye3t:block_output_real", block_output_size);
        new_block_output_imaginary = RealView("ye3t:block_output_imaginary", block_output_size);
        new_block_output_adjoint_real =
            RealView("ye3t:block_output_adjoint_real", block_output_size);
        new_block_output_adjoint_imaginary =
            RealView("ye3t:block_output_adjoint_imaginary", block_output_size);
        new_block_monomial_real = RealView("ye3t:block_monomial_real", block_monomial_size);
        new_block_monomial_imaginary =
            RealView("ye3t:block_monomial_imaginary", block_monomial_size);
        // Transpose pullback consumes each monomial root immediately.
        // No full monomial-adjoint workspace is needed.
        new_block_route_density_real =
            RealView("ye3t:block_route_density_real", block_route_density_size);
        new_block_route_density_imaginary =
            RealView("ye3t:block_route_density_imaginary", block_route_density_size);
      }
      if (device_plan_.has_scalar_program()) {
        const std::size_t scalar_size = checked_multiply_bytes(
            static_cast<std::size_t>(device_plan_.maximum_scalar_value_count()),
            static_cast<std::size_t>(candidate), "scalar-power workspace extent");
        new_scalar_value_real = RealView("ye3t:scalar_value_real", scalar_size);
        new_scalar_value_imaginary = RealView("ye3t:scalar_value_imaginary", scalar_size);
        new_scalar_adjoint_real = RealView("ye3t:scalar_adjoint_real", scalar_size);
        new_scalar_adjoint_imaginary = RealView("ye3t:scalar_adjoint_imaginary", scalar_size);
      }
      if (device_plan_.has_coupled_program()) {
        const std::size_t coupled_size = checked_multiply_bytes(
            static_cast<std::size_t>(device_plan_.maximum_coupled_component_count()),
            static_cast<std::size_t>(candidate), "coupled-product workspace extent");
        new_coupled_value_real = RealView("ye3t:coupled_value_real", coupled_size);
        new_coupled_value_imaginary = RealView("ye3t:coupled_value_imaginary", coupled_size);
        new_coupled_adjoint_real = RealView("ye3t:coupled_adjoint_real", coupled_size);
        new_coupled_adjoint_imaginary = RealView("ye3t:coupled_adjoint_imaginary", coupled_size);
      }
      RealView new_atomic_energies("ye3t:atomic_energies", candidate);

      dynamic_bytes_ -= center_storage_bytes(center_capacity_);
      center_counts_ = std::move(new_counts);
      center_offsets_ = std::move(new_offsets);
      source_real_ = std::move(new_source_real);
      source_imaginary_ = std::move(new_source_imaginary);
      source_adjoint_real_ = std::move(new_source_adjoint_real);
      source_adjoint_imaginary_ = std::move(new_source_adjoint_imaginary);
      dag_real_ = std::move(new_dag_real);
      dag_imaginary_ = std::move(new_dag_imaginary);
      dag_adjoint_real_ = std::move(new_dag_adjoint_real);
      dag_adjoint_imaginary_ = std::move(new_dag_adjoint_imaginary);
      total_density_real_ = std::move(new_total_density_real);
      total_density_imaginary_ = std::move(new_total_density_imaginary);
      block_power_real_ = std::move(new_block_power_real);
      block_power_imaginary_ = std::move(new_block_power_imaginary);
      block_output_real_ = std::move(new_block_output_real);
      block_output_imaginary_ = std::move(new_block_output_imaginary);
      block_output_adjoint_real_ = std::move(new_block_output_adjoint_real);
      block_output_adjoint_imaginary_ = std::move(new_block_output_adjoint_imaginary);
      block_monomial_real_ = std::move(new_block_monomial_real);
      block_monomial_imaginary_ = std::move(new_block_monomial_imaginary);
      block_monomial_adjoint_real_ = std::move(new_block_monomial_adjoint_real);
      block_monomial_adjoint_imaginary_ = std::move(new_block_monomial_adjoint_imaginary);
      block_route_density_real_ = std::move(new_block_route_density_real);
      block_route_density_imaginary_ = std::move(new_block_route_density_imaginary);
      scalar_value_real_ = std::move(new_scalar_value_real);
      scalar_value_imaginary_ = std::move(new_scalar_value_imaginary);
      scalar_adjoint_real_ = std::move(new_scalar_adjoint_real);
      scalar_adjoint_imaginary_ = std::move(new_scalar_adjoint_imaginary);
      coupled_value_real_ = std::move(new_coupled_value_real);
      coupled_value_imaginary_ = std::move(new_coupled_value_imaginary);
      coupled_adjoint_real_ = std::move(new_coupled_adjoint_real);
      coupled_adjoint_imaginary_ = std::move(new_coupled_adjoint_imaginary);
      atomic_energies_ = std::move(new_atomic_energies);
      center_capacity_ = candidate;
      dynamic_bytes_ += new_bytes;
      ++reallocation_count_;
      return std::min(requested, candidate);
    }
    if (!tried_requested && candidate > requested) {
      candidate = requested;
      tried_requested = true;
    } else {
      candidate /= 2;
    }
  }
  if (center_capacity_ > 0) return std::min(requested, center_capacity_);
  error->all(FLERR,
             "Pair style ye3t/kk cannot allocate one center while "
             "preserving device-memory headroom");
  return 0;
}

template <class DeviceType>
int PairYE3TKokkos<DeviceType>::ensure_lifted_center_capacity(int requested)
{
  if (requested <= center_capacity_) return requested;
  std::int64_t grown = static_cast<std::int64_t>(center_capacity_) + center_capacity_ / 2 + 64;
  int candidate = static_cast<int>(std::min<std::int64_t>(
      std::numeric_limits<int>::max(), std::max<std::int64_t>(requested, grown)));
  bool tried_requested = candidate == requested;
  while (candidate > 0) {
    const std::size_t new_bytes = center_storage_bytes(candidate);
    const std::size_t fixed_bytes =
        (type_to_species_.extent(0) * sizeof(int) + step_state_.memory_usage());
    const bool within_ceiling = within_lifted_workspace_ceiling(
        lifted_device_plan_.memory_usage(), fixed_bytes, dynamic_bytes_,
        center_storage_bytes(center_capacity_), new_bytes);
    if (within_ceiling && allocation_preflight(new_bytes)) {
      OffsetView new_counts("ye3t:lifted_center_counts", candidate);
      OffsetView new_offsets("ye3t:lifted_center_offsets", candidate + 1);
      const std::size_t source_size = checked_multiply_bytes(
          static_cast<std::size_t>(lifted_device_plan_.source_variable_count()),
          static_cast<std::size_t>(candidate), "lifted source workspace extent");
      const std::size_t workspace_size = checked_multiply_bytes(
          static_cast<std::size_t>(lifted_device_plan_.workspace_values_per_center()),
          static_cast<std::size_t>(candidate), "lifted polynomial workspace extent");
      RealView new_source("ye3t:lifted_source", source_size);
      RealView new_source_adjoint("ye3t:lifted_source_adjoint", source_size);
      RealView new_workspace("ye3t:lifted_polynomial_workspace", workspace_size);
      RealView new_atomic_energies("ye3t:lifted_atomic_energies", candidate);

      dynamic_bytes_ -= center_storage_bytes(center_capacity_);
      center_counts_ = std::move(new_counts);
      center_offsets_ = std::move(new_offsets);
      source_real_ = std::move(new_source);
      source_adjoint_real_ = std::move(new_source_adjoint);
      lifted_workspace_ = std::move(new_workspace);
      atomic_energies_ = std::move(new_atomic_energies);
      center_capacity_ = candidate;
      dynamic_bytes_ += new_bytes;
      ++reallocation_count_;
      return std::min(requested, candidate);
    }
    if (!tried_requested && candidate > requested) {
      candidate = requested;
      tried_requested = true;
    } else {
      candidate /= 2;
    }
  }
  if (center_capacity_ > 0) return std::min(requested, center_capacity_);
  error->all(FLERR,
             "Pair style ye3t/kk cannot allocate one lifted center "
             "while preserving device-memory headroom");
  return 0;
}

template <class DeviceType>
int PairYE3TKokkos<DeviceType>::ensure_tagged_center_capacity(int requested)
{
  if (requested <= center_capacity_) return requested;
  std::int64_t grown = static_cast<std::int64_t>(center_capacity_) + center_capacity_ / 2 + 64;
  int candidate = static_cast<int>(std::min<std::int64_t>(
      std::numeric_limits<int>::max(), std::max<std::int64_t>(requested, grown)));
  bool tried_requested = candidate == requested;
  while (candidate > 0) {
    const std::size_t new_bytes = center_storage_bytes(candidate);
    if (allocation_preflight(new_bytes)) {
      OffsetView new_counts("ye3t:tagged_center_counts", candidate);
      OffsetView new_offsets("ye3t:tagged_center_offsets", candidate + 1);
      const std::size_t density_size = checked_multiply_bytes(
          static_cast<std::size_t>(tagged_device_plan_.density_key_count()),
          static_cast<std::size_t>(candidate), "tagged density workspace extent");
      const std::size_t moment_size = checked_multiply_bytes(
          static_cast<std::size_t>(tagged_device_plan_.moment_key_count()),
          static_cast<std::size_t>(candidate), "tagged moment workspace extent");
      RealView new_density("ye3t:tagged_density", density_size);
      RealView new_moment("ye3t:tagged_moment", moment_size);
      RealView new_density_adjoint("ye3t:tagged_density_adjoint", density_size);
      RealView new_moment_adjoint("ye3t:tagged_moment_adjoint", moment_size);
      RealView new_atomic_energies("ye3t:tagged_atomic_energies", candidate);

      dynamic_bytes_ -= center_storage_bytes(center_capacity_);
      center_counts_ = std::move(new_counts);
      center_offsets_ = std::move(new_offsets);
      tagged_density_ = std::move(new_density);
      tagged_moment_ = std::move(new_moment);
      tagged_density_adjoint_ = std::move(new_density_adjoint);
      tagged_moment_adjoint_ = std::move(new_moment_adjoint);
      atomic_energies_ = std::move(new_atomic_energies);
      center_capacity_ = candidate;
      dynamic_bytes_ += new_bytes;
      ++reallocation_count_;
      return std::min(requested, candidate);
    }
    if (!tried_requested && candidate > requested) {
      candidate = requested;
      tried_requested = true;
    } else {
      candidate /= 2;
    }
  }
  if (center_capacity_ > 0) return std::min(requested, center_capacity_);
  error->all(FLERR,
             "Pair style ye3t/kk cannot allocate one tagged-Cauchy center "
             "while preserving device-memory headroom");
  return 0;
}

template <class DeviceType> bool PairYE3TKokkos<DeviceType>::ensure_edge_capacity(int requested)
{
  if (requested <= edge_capacity_) return true;
  const std::int64_t grown = static_cast<std::int64_t>(edge_capacity_) + edge_capacity_ / 2 + 256;
  int candidate = static_cast<int>(std::min<std::int64_t>(
      std::numeric_limits<int>::max(), std::max<std::int64_t>(requested, grown)));
  auto can_allocate = [&](std::size_t proposed_bytes) {
    if (!allocation_preflight(proposed_bytes)) return false;
    if (!is_lifted_cauchy_model()) return true;
    const std::size_t fixed_bytes =
        (type_to_species_.extent(0) * sizeof(int) + step_state_.memory_usage());
    return within_lifted_workspace_ceiling(lifted_device_plan_.memory_usage(), fixed_bytes,
                                           dynamic_bytes_, edge_storage_bytes(edge_capacity_),
                                           proposed_bytes);
  };
  if (!can_allocate(edge_storage_bytes(candidate))) candidate = requested;
  const std::size_t new_bytes = edge_storage_bytes(candidate);
  if (!can_allocate(new_bytes)) return false;

  IntView new_centers("ye3t:edge_centers", candidate);
  IntView new_neighbors("ye3t:edge_neighbors", candidate);
  IntView new_bonds("ye3t:edge_bonds", candidate);
  RealView new_radius("ye3t:edge_radius", candidate);
  RealView new_unit(
      "ye3t:edge_unit",
      checked_multiply_bytes(static_cast<std::size_t>(candidate), 3, "edge unit-vector extent"));
  RealView new_gradient(
      "ye3t:edge_gradient",
      checked_multiply_bytes(static_cast<std::size_t>(candidate), 3, "edge gradient extent"));
  const bool use_edge_cache = use_neighbor_major_source_ || use_edge_team_source_;
  const std::size_t radial_size = use_edge_cache
      ? checked_multiply_bytes(static_cast<std::size_t>(candidate),
                               static_cast<std::size_t>(source_maximum_radial_base_count_),
                               "edge radial-cache extent")
      : 0;
  const std::size_t contracted_size = use_edge_cache
      ? checked_multiply_bytes(static_cast<std::size_t>(candidate),
                               static_cast<std::size_t>(source_maximum_contracted_width_),
                               "edge contracted-cache extent")
      : 0;
  RealView new_radial_derivative("ye3t:edge_radial_derivative", radial_size);
  RealView new_contracted_value("ye3t:edge_contracted_value", contracted_size);
  RealView new_contracted_derivative("ye3t:edge_contracted_derivative", contracted_size);
  dynamic_bytes_ -= edge_storage_bytes(edge_capacity_);
  edge_centers_ = std::move(new_centers);
  edge_neighbors_ = std::move(new_neighbors);
  edge_bonds_ = std::move(new_bonds);
  edge_radius_ = std::move(new_radius);
  edge_unit_ = std::move(new_unit);
  edge_radial_derivative_ = std::move(new_radial_derivative);
  edge_contracted_value_ = std::move(new_contracted_value);
  edge_contracted_derivative_ = std::move(new_contracted_derivative);
  edge_gradient_ = std::move(new_gradient);
  edge_capacity_ = candidate;
  dynamic_bytes_ += new_bytes;
  ++reallocation_count_;
  return true;
}

template <class DeviceType> int PairYE3TKokkos<DeviceType>::copy_device_status() const
{
  int status = 0;
  const DeviceType exec{};
  Kokkos::deep_copy(exec, status, Kokkos::subview(device_status_, 0));
  exec.fence("YE3T device status");
  return status;
}

template <class DeviceType> void PairYE3TKokkos<DeviceType>::report_device_status(int status) const
{
  if (status & YE3T_STATUS_ZERO_RADIUS)
    error->one(FLERR, "YE3T Kokkos evaluation encountered a zero-length edge");
  if (status & YE3T_STATUS_SPLINE_INTERVAL)
    error->one(FLERR, "YE3T Kokkos edge radius is below the first spline interval");
  if (status & YE3T_STATUS_CSR_MISMATCH)
    error->one(FLERR, "YE3T Kokkos CSR edge construction was inconsistent");
  if (status & YE3T_STATUS_DENSITY_LIMIT)
    error->one(FLERR, "YE3T density entered the unsupported core-smoothing interval");
  if (status & YE3T_STATUS_NONFINITE_GEOMETRY)
    error->one(FLERR, "YE3T Kokkos evaluation encountered non-finite geometry");
  if (status & YE3T_STATUS_NONFINITE_SOURCE)
    error->one(FLERR, "YE3T Kokkos lifted source construction became non-finite");
  if (status & YE3T_STATUS_NONFINITE_READOUT)
    error->one(FLERR, "YE3T Kokkos sparse readout became non-finite");
  if (status & YE3T_STATUS_NONFINITE_VJP)
    error->one(FLERR, "YE3T Kokkos lifted source VJP became non-finite");
  if (status & YE3T_STATUS_NONFINITE_TAGGED_FORWARD)
    error->one(FLERR, "YE3T Kokkos tagged density/moment construction became non-finite");
  if (status & YE3T_STATUS_NONFINITE_TAGGED_READOUT)
    error->one(FLERR, "YE3T Kokkos tagged term readout became non-finite");
  if (status & YE3T_STATUS_NONFINITE_TAGGED_VJP)
    error->one(FLERR, "YE3T Kokkos tagged edge VJP became non-finite");
}

template <class DeviceType>
YE3T_LAMMPS::YACEBlockPolicy PairYE3TKokkos<DeviceType>::model_load_policy() const
{
  return block_policy() == YE3T_LAMMPS::YACEBlockPolicy::AUTO
      ? YE3T_LAMMPS::YACEBlockPolicy::GPU_AUTO
      : block_policy();
}

template <class DeviceType> void PairYE3TKokkos<DeviceType>::initialize_lifted_backend()
{
  if (lifted_source_policy() != YE3T_LAMMPS::LiftedCauchySourcePolicy::DIRECT_Q)
    error->all(FLERR,
               "Pair style ye3t/kk lifted_cauchy currently supports "
               "source_realization direct only");
  if (execution_space != Device)
    error->all(FLERR, "Pair style ye3t/kk requires a Kokkos device execution space");

  const char *exact_chunk_environment = std::getenv("YE3T_KOKKOS_REQUIRE_EXACT_CHUNKSIZE");
  require_exact_chunksize_ = false;
  if (exact_chunk_environment != nullptr && std::strcmp(exact_chunk_environment, "0") != 0) {
    if (std::strcmp(exact_chunk_environment, "1") != 0)
      error->all(FLERR, "YE3T_KOKKOS_REQUIRE_EXACT_CHUNKSIZE must be absent, 0, or 1");
    require_exact_chunksize_ = true;
  }
  initialize_rank_device_identity();

  const std::size_t type_map_bytes = checked_multiply_bytes(
      static_cast<std::size_t>(atom->ntypes + 1), sizeof(int), "lifted type-map storage");
  const std::size_t status_bytes = sizeof(int);
  std::size_t candidate_budget = allocation_budget();
  const std::size_t auxiliary_bytes =
      checked_add_bytes(type_map_bytes, status_bytes, "lifted candidate auxiliary storage");
  if (auxiliary_bytes >= candidate_budget || auxiliary_bytes >= YE3T_MAX_PLANNED_WORKSPACE_BYTES)
    error->all(FLERR,
               "Pair style ye3t/kk cannot preserve device-memory "
               "headroom while loading the lifted type map");
  candidate_budget = std::min(candidate_budget - auxiliary_bytes,
                              YE3T_MAX_PLANNED_WORKSPACE_BYTES - auxiliary_bytes);

  YE3T_LAMMPS::LiftedCauchyKokkosPlan<DeviceType> candidate;
  candidate.upload(lifted_cauchy_model(), candidate_budget);
  IntView candidate_type_map("ye3t:lifted_type_to_species", atom->ntypes + 1);
  auto host_type_map = Kokkos::create_mirror_view(candidate_type_map);
  for (int type = 0; type <= atom->ntypes; ++type) host_type_map(type) = type == 0 ? -1 : map[type];
  Kokkos::deep_copy(candidate_type_map, host_type_map);
  IntView candidate_status("ye3t:lifted_device_status", 1);
  Kokkos::deep_copy(candidate_status, 0);

  const std::string plan_signature =
      "ye3t_lifted_cauchy_kokkos_direct_q_deferred_tallies_status_v3;" +
      lifted_cauchy_model().deployment_identity_hash + ";" + execution_space_name<DeviceType>() +
      ";fp64;source_major_center_fast;" + std::to_string(candidate.source_variable_count()) + ";" +
      std::to_string(candidate.source_row_count()) + ";" +
      std::to_string(candidate.maximum_factor_count()) + ";";
  device_plan_hash_ = YE3T_LAMMPS::sha256_string(plan_signature);
  std::array<char, 65> root_hash{};
  if (comm->me == 0)
    std::memcpy(root_hash.data(), device_plan_hash_.c_str(), device_plan_hash_.size());
  MPI_Bcast(root_hash.data(), static_cast<int>(root_hash.size()), MPI_CHAR, 0, world);
  const int mismatch = device_plan_hash_ != std::string(root_hash.data());
  int any_mismatch = 0;
  MPI_Allreduce(&mismatch, &any_mismatch, 1, MPI_INT, MPI_MAX, world);
  if (any_mismatch) error->all(FLERR, "Pair style ye3t/kk lifted device plan differs across ranks");

  lifted_device_plan_ = std::move(candidate);
  type_to_species_ = std::move(candidate_type_map);
  device_status_ = std::move(candidate_status);
  center_counts_ = {};
  center_offsets_ = {};
  edge_centers_ = {};
  edge_neighbors_ = {};
  edge_bonds_ = {};
  edge_radius_ = {};
  edge_unit_ = {};
  edge_radial_derivative_ = {};
  edge_contracted_value_ = {};
  edge_contracted_derivative_ = {};
  source_real_ = {};
  source_imaginary_ = {};
  source_adjoint_real_ = {};
  source_adjoint_imaginary_ = {};
  lifted_workspace_ = {};
  dag_real_ = {};
  dag_imaginary_ = {};
  dag_adjoint_real_ = {};
  dag_adjoint_imaginary_ = {};
  total_density_real_ = {};
  total_density_imaginary_ = {};
  block_power_real_ = {};
  block_power_imaginary_ = {};
  block_output_real_ = {};
  block_output_imaginary_ = {};
  block_output_adjoint_real_ = {};
  block_output_adjoint_imaginary_ = {};
  block_monomial_real_ = {};
  block_monomial_imaginary_ = {};
  block_monomial_adjoint_real_ = {};
  block_monomial_adjoint_imaginary_ = {};
  block_route_density_real_ = {};
  block_route_density_imaginary_ = {};
  scalar_value_real_ = {};
  scalar_value_imaginary_ = {};
  scalar_adjoint_real_ = {};
  scalar_adjoint_imaginary_ = {};
  coupled_value_real_ = {};
  coupled_value_imaginary_ = {};
  coupled_adjoint_real_ = {};
  coupled_adjoint_imaginary_ = {};
  atomic_energies_ = {};
  edge_gradient_ = {};
  center_capacity_ = 0;
  edge_capacity_ = 0;
  dynamic_bytes_ = status_bytes;
  reallocation_count_ = 0;
  recorded_chunk_limit_ = 0;
  capacity_state_reported_ = false;
  replay_workload_validated_ = false;
  source_maximum_radial_base_count_ = 0;
  source_maximum_contracted_width_ = 0;
  source_maximum_angular_width_ = 0;
  source_serial_work_count_ = 0;
  edge_basis_cache_value_count_ = 0;
  use_neighbor_major_source_ = false;
  use_edge_team_source_ = false;
  use_block_program_ = false;
  use_work_major_block_ = false;
  use_scalar_program_ = false;
  use_coupled_program_ = false;
  device_plan_ready_ = false;
  lifted_device_plan_ready_ = true;
  flat_plan_hash_ = lifted_cauchy_model().native_self_hash;

  const auto [free_bytes, total_bytes] = device_memory_info();
  if (comm->me == 0)
    utils::logmesg(lmp,
                   "YE3T lifted Kokkos dispatch: source direct_q, evaluator "
                   "compiler_lowered_sparse_symmetric_power, execution_space {}, "
                   "native_plan {}, deployment {}, device_plan {}, source_variables {}, "
                   "source_rows {}, maximum_distinct_factors {}, static_bytes {}, "
                   "free_device_bytes {}, total_device_bytes {}\n",
                   execution_space_name<DeviceType>(), lifted_cauchy_model().native_self_hash,
                   lifted_cauchy_model().deployment_identity_hash, device_plan_hash_,
                   lifted_device_plan_.source_variable_count(),
                   lifted_device_plan_.source_row_count(),
                   lifted_device_plan_.maximum_factor_count(), lifted_device_plan_.memory_usage(),
                   free_bytes, total_bytes);
}

template <class DeviceType> void PairYE3TKokkos<DeviceType>::initialize_tagged_backend()
{
  if (execution_space != Device)
    error->all(FLERR, "Pair style ye3t/kk requires a Kokkos device execution space");
  if (block_policy() != YE3T_LAMMPS::YACEBlockPolicy::DIRECT)
    error->all(FLERR,
               "Pair style ye3t/kk tagged_cauchy currently supports only "
               "block_policy direct; tagged GPU AUTO is a separate "
               "qualification leaf");
  if (tagged_cauchy_model().has_ordinary_backbone())
    error->all(FLERR,
               "Pair style ye3t/kk does not yet support a self-contained ordinary-plus-"
               "tagged composite; use the CPU pair style until the two device plans "
               "share one force/energy accumulation path");
  initialize_rank_device_identity();

  const std::size_t type_map_bytes = checked_multiply_bytes(
      static_cast<std::size_t>(atom->ntypes + 1), sizeof(int), "tagged type-map storage");
  const std::size_t status_bytes = sizeof(int);
  std::size_t candidate_budget = allocation_budget();
  const std::size_t auxiliary_bytes =
      checked_add_bytes(type_map_bytes, status_bytes, "tagged candidate auxiliary storage");
  if (auxiliary_bytes >= candidate_budget || auxiliary_bytes >= YE3T_MAX_PLANNED_WORKSPACE_BYTES)
    error->all(FLERR,
               "Pair style ye3t/kk cannot preserve device-memory headroom "
               "while loading the tagged type map");
  candidate_budget = std::min(candidate_budget - auxiliary_bytes,
                              YE3T_MAX_PLANNED_WORKSPACE_BYTES - auxiliary_bytes);

  YE3T_LAMMPS::TaggedCauchyKokkosPlan<DeviceType> candidate;
  candidate.upload(tagged_cauchy_model(), candidate_budget);
  IntView candidate_type_map("ye3t:tagged_type_to_species", atom->ntypes + 1);
  auto host_type_map = Kokkos::create_mirror_view(candidate_type_map);
  for (int type = 0; type <= atom->ntypes; ++type) host_type_map(type) = type == 0 ? -1 : map[type];
  Kokkos::deep_copy(candidate_type_map, host_type_map);
  IntView candidate_status("ye3t:tagged_device_status", 1);
  Kokkos::deep_copy(candidate_status, 0);

  const std::string plan_signature = "ye3t_tagged_cauchy_kokkos_folded_coalesced_readout_v5;" +
      tagged_cauchy_model().self_hash + ";" + tagged_cauchy_model().source_plan_hash + ";" +
      tagged_cauchy_model().schedule_hash + ";" + tagged_cauchy_model().deployment_identity_hash +
      ";" + execution_space_name<DeviceType>() + ";fp64;reference_one_thread_per_center;" +
      std::to_string(candidate.feature_count()) + ";" +
      std::to_string(candidate.density_key_count()) + ";" +
      std::to_string(candidate.moment_key_count()) + ";" +
      std::to_string(candidate.raw_term_count()) + ";" + std::to_string(candidate.term_count()) +
      ";" + std::to_string(candidate.total_component_count()) + ";";
  device_plan_hash_ = YE3T_LAMMPS::sha256_string(plan_signature);
  std::array<char, 65> root_hash{};
  if (comm->me == 0)
    std::memcpy(root_hash.data(), device_plan_hash_.c_str(), device_plan_hash_.size());
  MPI_Bcast(root_hash.data(), static_cast<int>(root_hash.size()), MPI_CHAR, 0, world);
  const int mismatch = device_plan_hash_ != std::string(root_hash.data());
  int any_mismatch = 0;
  MPI_Allreduce(&mismatch, &any_mismatch, 1, MPI_INT, MPI_MAX, world);
  if (any_mismatch) error->all(FLERR, "Pair style ye3t/kk tagged device plan differs across ranks");

  tagged_device_plan_ = std::move(candidate);
  type_to_species_ = std::move(candidate_type_map);
  device_status_ = std::move(candidate_status);
  center_counts_ = {};
  center_offsets_ = {};
  edge_centers_ = {};
  edge_neighbors_ = {};
  edge_bonds_ = {};
  edge_radius_ = {};
  edge_unit_ = {};
  edge_radial_derivative_ = {};
  edge_contracted_value_ = {};
  edge_contracted_derivative_ = {};
  source_real_ = {};
  source_imaginary_ = {};
  source_adjoint_real_ = {};
  source_adjoint_imaginary_ = {};
  lifted_workspace_ = {};
  dag_real_ = {};
  dag_imaginary_ = {};
  dag_adjoint_real_ = {};
  dag_adjoint_imaginary_ = {};
  total_density_real_ = {};
  total_density_imaginary_ = {};
  block_power_real_ = {};
  block_power_imaginary_ = {};
  block_output_real_ = {};
  block_output_imaginary_ = {};
  block_output_adjoint_real_ = {};
  block_output_adjoint_imaginary_ = {};
  block_monomial_real_ = {};
  block_monomial_imaginary_ = {};
  block_monomial_adjoint_real_ = {};
  block_monomial_adjoint_imaginary_ = {};
  block_route_density_real_ = {};
  block_route_density_imaginary_ = {};
  scalar_value_real_ = {};
  scalar_value_imaginary_ = {};
  scalar_adjoint_real_ = {};
  scalar_adjoint_imaginary_ = {};
  coupled_value_real_ = {};
  coupled_value_imaginary_ = {};
  coupled_adjoint_real_ = {};
  coupled_adjoint_imaginary_ = {};
  atomic_energies_ = {};
  edge_gradient_ = {};
  tagged_density_ = {};
  tagged_moment_ = {};
  tagged_density_adjoint_ = {};
  tagged_moment_adjoint_ = {};
  center_capacity_ = 0;
  edge_capacity_ = 0;
  dynamic_bytes_ = status_bytes;
  reallocation_count_ = 0;
  recorded_chunk_limit_ = 0;
  capacity_state_reported_ = false;
  replay_workload_validated_ = false;
  source_maximum_radial_base_count_ = 0;
  source_maximum_contracted_width_ = 0;
  source_maximum_angular_width_ = 0;
  source_serial_work_count_ = 0;
  edge_basis_cache_value_count_ = 0;
  use_neighbor_major_source_ = false;
  use_edge_team_source_ = false;
  use_block_program_ = false;
  use_work_major_block_ = false;
  use_scalar_program_ = false;
  use_coupled_program_ = false;
  device_plan_ready_ = false;
  lifted_device_plan_ready_ = false;
  tagged_device_plan_ready_ = true;
  flat_plan_hash_ = tagged_cauchy_model().self_hash;

  const auto [free_bytes, total_bytes] = device_memory_info();
  const bool physical_image_v3 = tagged_cauchy_model().deployment_kind ==
      YE3T_LAMMPS::TaggedCauchyDeploymentKind::PhysicalImageV3;
  if (comm->me == 0)
    utils::logmesg(lmp,
                   "YE3T tagged Kokkos dispatch: evaluator {}, qualification {}, "
                   "execution_space {}, model_self_hash {}, source_plan {}, schedule {}, "
                   "device_plan {}, feature_count {}, "
                   "density_key_count {}, moment_key_count {}, raw_term_count {}, "
                   "compiled_term_count {}, total_component_count {}, "
                   "static_bytes {}, free_device_bytes {}, total_device_bytes {}\n",
                   physical_image_v3 ? "physical_image_v3_folded_direct"
                                     : "legacy_moment_v2_folded_coalesced",
                   "experimental_reference_unqualified", execution_space_name<DeviceType>(),
                   tagged_cauchy_model().self_hash, tagged_cauchy_model().source_plan_hash,
                   tagged_cauchy_model().schedule_hash, device_plan_hash_,
                   tagged_device_plan_.feature_count(), tagged_device_plan_.density_key_count(),
                   tagged_device_plan_.moment_key_count(), tagged_device_plan_.raw_term_count(),
                   tagged_device_plan_.term_count(), tagged_device_plan_.total_component_count(),
                   tagged_device_plan_.memory_usage(), free_bytes, total_bytes);
}

template <class DeviceType> void PairYE3TKokkos<DeviceType>::initialize_backend()
{
  const bool request_direct = block_policy() == YE3T_LAMMPS::YACEBlockPolicy::DIRECT;
  const bool request_auto = block_policy() == YE3T_LAMMPS::YACEBlockPolicy::AUTO;
  const bool request_block = block_policy() == YE3T_LAMMPS::YACEBlockPolicy::BLOCK;
  const bool request_scalar = block_policy() == YE3T_LAMMPS::YACEBlockPolicy::SCALAR_POWER;
  const bool request_coupled = block_policy() == YE3T_LAMMPS::YACEBlockPolicy::COUPLED_PRODUCT;
  if (!request_direct && !request_auto && !request_block && !request_scalar && !request_coupled)
    error->all(FLERR,
               "Pair style ye3t/kk currently supports block_policy "
               "direct, auto, block, scalar_power, or coupled_product");

  bool selected_direct = false;
  bool selected_block = false;
  bool selected_scalar = false;
  bool selected_coupled = false;
  std::string planner_profile = "not_applicable";
  std::string planner_algorithm = "not_applicable";
  std::string planner_status = "not_applicable";
  std::string planner_calibration_hash = "not_applicable";
  std::string planner_decision_reason = "not_applicable";
  const auto &auto_replay = model().auto_replay();
  for (int species = 0; species < model().species_count(); ++species) {
    const auto &entry = model().species(species);
    const auto &program = entry.block_program;
    selected_direct = selected_direct || !entry.polynomial.monomial_coefficients.empty();
    selected_block = selected_block || !program.routes.empty();
    selected_scalar = selected_scalar || !program.scalar_program.routes.empty();
    selected_coupled = selected_coupled || !program.coupled_product_plans.empty();
    if (request_auto) {
      const bool valid_replay_record = auto_replay.enabled() &&
          program.planner_profile == "kokkos_gpu_device_replay_v1" &&
          program.planner_algorithm == "device_bound_candidate_replay_v1" &&
          program.planner_status == "selected" &&
          program.planner_calibration_hash == auto_replay.replay_hash &&
          program.planner_decision_reason == "offline_device_bound_interleaved_calibration";
      const bool valid_conservative_record = !auto_replay.enabled() &&
          program.planner_profile == "kokkos_gpu_conservative_direct_v1" &&
          program.planner_algorithm == "conservative_direct_fallback_v1" &&
          program.planner_status == "selected_direct_no_authorized_profile" &&
          program.planner_calibration_hash == "not_applicable" &&
          program.planner_decision_reason == "no_authorized_non_direct_profile_match";
      if ((!valid_replay_record && !valid_conservative_record) ||
          program.evaluator_plan_hash.size() != 64)
        error->all(FLERR,
                   "Pair style ye3t/kk auto requires a complete "
                   "conservative or device-replay selection record");
      if (species == 0) {
        planner_profile = program.planner_profile;
        planner_algorithm = program.planner_algorithm;
        planner_status = program.planner_status;
        planner_calibration_hash = program.planner_calibration_hash;
        planner_decision_reason = program.planner_decision_reason;
      } else if (planner_profile != program.planner_profile ||
                 planner_algorithm != program.planner_algorithm ||
                 planner_status != program.planner_status ||
                 planner_calibration_hash != program.planner_calibration_hash ||
                 planner_decision_reason != program.planner_decision_reason) {
        error->all(FLERR,
                   "Pair style ye3t/kk auto requires one deterministic "
                   "GPU calibration policy across species");
      }
    }
  }
  if (!selected_direct && !selected_block && !selected_scalar && !selected_coupled)
    error->all(FLERR, "Pair style ye3t/kk selected an empty evaluator plan");
  if (auto_replay.enabled()) {
    const bool family_matches = (auto_replay.selected_evaluator == "direct" && selected_direct &&
                                 !selected_block && !selected_scalar && !selected_coupled) ||
        (auto_replay.selected_evaluator == "block" && selected_block && !selected_scalar &&
         !selected_coupled) ||
        (auto_replay.selected_evaluator == "scalar_power" && !selected_block && selected_scalar &&
         !selected_coupled) ||
        (auto_replay.selected_evaluator == "coupled_product" && !selected_block &&
         !selected_scalar && selected_coupled);
    if (!family_matches)
      error->all(FLERR,
                 "Pair style ye3t/kk AUTO replay selected evaluator "
                 "disagrees with its compiled catalogue plan");
  }
  if (request_block) {
    bool any_route = false;
    for (int species = 0; species < model().species_count(); ++species) {
      const auto &program = model().species(species).block_program;
      any_route = any_route || !program.routes.empty();
      if (!program.scalar_program.routes.empty() || !program.coupled_product_plans.empty())
        error->all(FLERR,
                   "Pair style ye3t/kk block_policy block received a "
                   "non-block selected evaluator payload");
    }
    if (!any_route)
      error->all(FLERR,
                 "Pair style ye3t/kk block_policy block requires at "
                 "least one selected block route");
  }
  if (request_scalar) {
    bool any_route = false;
    for (int species = 0; species < model().species_count(); ++species) {
      const auto &program = model().species(species).block_program;
      any_route = any_route || !program.scalar_program.routes.empty();
      if (!program.routes.empty() || !program.coupled_product_plans.empty())
        error->all(FLERR,
                   "Pair style ye3t/kk block_policy scalar_power received a "
                   "non-scalar selected evaluator payload");
    }
    if (!any_route)
      error->all(FLERR,
                 "Pair style ye3t/kk block_policy scalar_power requires "
                 "at least one selected scalar-power route");
  }
  if (request_coupled) {
    bool any_plan = false;
    for (int species = 0; species < model().species_count(); ++species) {
      const auto &program = model().species(species).block_program;
      any_plan = any_plan || !program.coupled_product_plans.empty();
      if (!program.routes.empty() || !program.scalar_program.routes.empty())
        error->all(FLERR,
                   "Pair style ye3t/kk block_policy coupled_product received a "
                   "non-coupled selected evaluator payload");
    }
    if (!any_plan)
      error->all(FLERR,
                 "Pair style ye3t/kk block_policy coupled_product requires at least "
                 "one selected coupled-product plan");
  }
  if (execution_space != Device)
    error->all(FLERR, "Pair style ye3t/kk requires a Kokkos device execution space");

  use_work_major_block_ = false;
  block_work_major_team_size_ = 0;
  block_work_major_scratch_bytes_ = 0;
  if (auto_replay.enabled() && !request_auto)
    error->all(FLERR, "Pair style ye3t/kk AUTO replay requires block_policy auto");
  const char *block_schedule_environment = std::getenv("YE3T_KOKKOS_BLOCK_SCHEDULE");
  const char *block_team_environment = std::getenv("YE3T_KOKKOS_BLOCK_TEAM_SIZE");
  if (auto_replay.enabled()) {
    if (block_schedule_environment != nullptr || block_team_environment != nullptr)
      error->all(FLERR,
                 "YE3T Kokkos block calibration environment cannot "
                 "override an AUTO replay schedule");
    if (selected_block) {
      if (auto_replay.block_schedule == "plan_route_major_v1")
        use_work_major_block_ = true;
      else if (auto_replay.block_schedule == "fused_center_lane_v2")
        use_work_major_block_ = false;
      else
        error->all(FLERR,
                   "Pair style ye3t/kk AUTO replay selected block "
                   "routes without a block schedule");
    } else if (auto_replay.block_schedule != "not_selected") {
      error->all(FLERR,
                 "Pair style ye3t/kk AUTO replay names a block "
                 "schedule without selected block routes");
    }
  } else if (selected_block && block_schedule_environment != nullptr) {
    if (std::strcmp(block_schedule_environment, "fused") == 0) {
      use_work_major_block_ = false;
    } else if (std::strcmp(block_schedule_environment, "work_major") == 0) {
      use_work_major_block_ = true;
    } else {
      error->all(FLERR,
                 "YE3T_KOKKOS_BLOCK_SCHEDULE must be fused or "
                 "work_major");
    }
  }
  int calibrated_block_team_size = 0;
  if (!auto_replay.enabled() && selected_block && block_team_environment != nullptr) {
    if (!use_work_major_block_)
      error->all(FLERR,
                 "YE3T_KOKKOS_BLOCK_TEAM_SIZE requires the work_major "
                 "block schedule");
    try {
      std::size_t consumed = 0;
      calibrated_block_team_size = std::stoi(block_team_environment, &consumed);
      if (consumed != std::strlen(block_team_environment))
        throw std::invalid_argument("trailing characters");
    } catch (const std::exception &) {
      error->all(FLERR, "YE3T_KOKKOS_BLOCK_TEAM_SIZE must be 32, 64, 128, or 256");
    }
    if (calibrated_block_team_size != 32 && calibrated_block_team_size != 64 &&
        calibrated_block_team_size != 128 && calibrated_block_team_size != 256)
      error->all(FLERR, "YE3T_KOKKOS_BLOCK_TEAM_SIZE must be 32, 64, 128, or 256");
  }

  const char *exact_chunk_environment = std::getenv("YE3T_KOKKOS_REQUIRE_EXACT_CHUNKSIZE");
  require_exact_chunksize_ = false;
  if (exact_chunk_environment != nullptr && std::strcmp(exact_chunk_environment, "0") != 0) {
    if (std::strcmp(exact_chunk_environment, "1") != 0)
      error->all(FLERR, "YE3T_KOKKOS_REQUIRE_EXACT_CHUNKSIZE must be absent, 0, or 1");
    require_exact_chunksize_ = true;
  }
  initialize_rank_device_identity();
  if (auto_replay.enabled()) {
#if defined(__linux__)
    const std::string executable_hash = YE3T_LAMMPS::sha256_file("/proc/self/exe");
    if (auto_replay.lammps_executable_hash != executable_hash)
      error->all(FLERR,
                 "Pair style ye3t/kk AUTO replay was calibrated for a "
                 "different LAMMPS executable");
#else
    error->all(FLERR,
               "Pair style ye3t/kk executable-bound AUTO replay is "
               "currently supported only on Linux");
#endif
    if (auto_replay.execution_space != execution_space_name<DeviceType>() ||
        auto_replay.device_class_hash != device_class_hash_)
      error->all(FLERR,
                 "Pair style ye3t/kk AUTO replay was calibrated for a "
                 "different device class");
    if (auto_replay.chunksize != chunksize())
      error->all(FLERR,
                 "Pair style ye3t/kk AUTO replay chunksize differs "
                 "from the configured chunksize");
#if defined(LMP_KOKKOS_LAYOUT_LEGACY)
    constexpr const char *compiled_layout = "legacy";
#else
    constexpr const char *compiled_layout = "default";
#endif
    if (auto_replay.layout != compiled_layout || auto_replay.precision != "fp64")
      error->all(FLERR,
                 "Pair style ye3t/kk AUTO replay layout or precision "
                 "does not match this build");
  }

  const std::size_t type_map_bytes = checked_multiply_bytes(
      static_cast<std::size_t>(atom->ntypes + 1), sizeof(int), "type-map storage");
  const std::size_t status_bytes = sizeof(int);
  std::size_t candidate_budget = allocation_budget();
  const std::size_t auxiliary_bytes =
      checked_add_bytes(type_map_bytes, status_bytes, "candidate auxiliary storage");
  if (auxiliary_bytes >= candidate_budget)
    error->all(FLERR,
               "Pair style ye3t/kk cannot preserve device-memory "
               "headroom while loading the type map");
  candidate_budget -= auxiliary_bytes;

  YE3T_LAMMPS::YE3TKokkosDevicePlan<DeviceType> candidate;
  candidate.upload(model(), candidate_budget);
  // At most 16 KiB for a whole graph; larger models keep node-contiguous slices
  // for each center in global memory. This is an internal residency cap, not a
  // cache claim: for rank-8 graphs of tens of KiB, occupancy matters more
  // than shared-memory residency.
  const std::size_t direct_scratch_limit =
      std::min<std::size_t>(16 * 1024, Kokkos::TeamPolicy<DeviceType>::scratch_size_max(0));
  direct_resident_values_ = candidate.maximum_value_count() > 0 &&
          static_cast<std::size_t>(candidate.maximum_value_count()) <=
              direct_scratch_limit / (4 * sizeof(double))
      ? candidate.maximum_value_count()
      : 0;

  if (use_work_major_block_) {
    if (auto_replay.enabled()) {
      block_work_major_team_size_ = auto_replay.block_team_size;
    } else if (calibrated_block_team_size > 0) {
      block_work_major_team_size_ = calibrated_block_team_size;
    } else {
#if defined(KOKKOS_ENABLE_HIP)
      if constexpr (std::is_same_v<DeviceType, Kokkos::HIP>)
        block_work_major_team_size_ = 64;
      else
#endif
        block_work_major_team_size_ = 32;
    }
    block_work_major_scratch_bytes_ = checked_multiply_bytes(
        checked_multiply_bytes(static_cast<std::size_t>(2 * block_work_major_team_size_),
                               static_cast<std::size_t>(candidate.block_monomial_tile_count()),
                               "block work-major scratch"),
        sizeof(double), "block work-major scratch");
    using BlockTeamPolicy = Kokkos::TeamPolicy<DeviceType>;
    if (block_work_major_scratch_bytes_ >
        static_cast<std::size_t>(BlockTeamPolicy::scratch_size_max(0)))
      error->all(FLERR,
                 "Pair style ye3t/kk work-major block scratch exceeds "
                 "the device team-scratch limit");
    if (auto_replay.enabled() &&
        (auto_replay.kernel_abi != "ye3t_kokkos_candidate_runtime_v2" ||
         auto_replay.block_scratch_layout != "monomial_tile16_lane_fast_v2" ||
         static_cast<std::uint64_t>(auto_replay.block_scratch_bytes) !=
             block_work_major_scratch_bytes_))
      error->all(FLERR,
                 "Pair style ye3t/kk AUTO replay block scratch "
                 "contract does not match the selected model");
  } else if (auto_replay.enabled() &&
             auto_replay.kernel_abi != "ye3t_kokkos_candidate_runtime_v2") {
    error->all(FLERR, "Pair style ye3t/kk AUTO replay kernel ABI does not match");
  }
  int block_identity_child_plans = 0;
  int block_symmetric_power_child_plans = 0;
  if (selected_block) {
    for (const auto &species_entry : model().species()) {
      for (const auto &plan : species_entry.block_program.power_plans) {
        if (plan.maximum_power < 1)
          error->all(FLERR,
                     "Pair style ye3t/kk block child plan has an "
                     "invalid symmetric-power degree");
        if (plan.direct_input_plan) {
          if (plan.maximum_power != 1)
            error->all(FLERR,
                       "Pair style ye3t/kk may use direct-input child "
                       "plans only for the exact Sym^1 identity");
          ++block_identity_child_plans;
        } else {
          ++block_symmetric_power_child_plans;
        }
      }
    }
  }
  const double probe_error = candidate.run_probe();
  if (!std::isfinite(probe_error) || probe_error > 1.0e-12)
    error->all(FLERR, "YE3T Kokkos device-plan probe failed with error {}", probe_error);
  ScalarPowerMathProbeResult scalar_math_probe;
  if (selected_scalar) {
    scalar_math_probe = run_scalar_power_math_probe<DeviceType>();
    if (!scalar_math_probe.finite || scalar_math_probe.forward_error > 5.0e-12 ||
        scalar_math_probe.adjoint_dot_error > 2.0e-7 || scalar_math_probe.zero_error > 1.0e-30)
      error->all(FLERR,
                 "YE3T Kokkos scalar-power math probe failed: forward {}, adjoint "
                 "dot {}, zero {}",
                 scalar_math_probe.forward_error, scalar_math_probe.adjoint_dot_error,
                 scalar_math_probe.zero_error);
  }
  CoupledProductMathProbeResult coupled_math_probe;
  if (selected_coupled) {
    coupled_math_probe = run_coupled_product_math_probe<DeviceType>();
    if (!coupled_math_probe.finite || coupled_math_probe.forward_error > 5.0e-12 ||
        coupled_math_probe.adjoint_dot_error > 2.0e-7 || coupled_math_probe.zero_error > 1.0e-30)
      error->all(FLERR,
                 "YE3T Kokkos coupled-product math probe failed: forward {}, "
                 "adjoint dot {}, zero {}",
                 coupled_math_probe.forward_error, coupled_math_probe.adjoint_dot_error,
                 coupled_math_probe.zero_error);
  }

  IntView candidate_type_map("ye3t:type_to_species", atom->ntypes + 1);
  auto host_type_map = Kokkos::create_mirror_view(candidate_type_map);
  for (int type = 0; type <= atom->ntypes; ++type) host_type_map(type) = type == 0 ? -1 : map[type];
  Kokkos::deep_copy(candidate_type_map, host_type_map);
  IntView candidate_status("ye3t:device_status", 1);
  Kokkos::deep_copy(candidate_status, 0);

  source_maximum_radial_base_count_ = model().maximum_radial_base_count();
  source_maximum_contracted_width_ = model().maximum_contracted_width();
  const std::int64_t angular_width =
      (static_cast<std::int64_t>(model().maximum_angular_momentum()) + 1) *
      (static_cast<std::int64_t>(model().maximum_angular_momentum()) + 2) / 2;
  if (source_maximum_radial_base_count_ <= 0 || source_maximum_contracted_width_ <= 0 ||
      angular_width <= 0 || angular_width > std::numeric_limits<int>::max())
    error->all(FLERR,
               "Pair style ye3t/kk source scratch dimensions are "
               "invalid or exceed 32-bit indexing");
  source_maximum_angular_width_ = static_cast<int>(angular_width);
  const std::size_t edge_cache_values = checked_add_bytes(
      static_cast<std::size_t>(source_maximum_radial_base_count_),
      checked_multiply_bytes(static_cast<std::size_t>(source_maximum_contracted_width_), 2,
                             "edge radial basis-cache extent"),
      "edge radial basis-cache extent");
  if (edge_cache_values > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    error->all(FLERR, "Pair style ye3t/kk edge basis cache exceeds 32-bit indexing");
  const std::size_t scratch_values = checked_add_bytes(
      checked_add_bytes(static_cast<std::size_t>(source_maximum_radial_base_count_),
                        static_cast<std::size_t>(source_maximum_contracted_width_),
                        "source scratch extent"),
      checked_multiply_bytes(static_cast<std::size_t>(source_maximum_angular_width_), 2,
                             "source scratch extent"),
      "source scratch extent");
  if (scratch_values > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    error->all(FLERR, "Pair style ye3t/kk source scratch exceeds 32-bit indexing");
  source_scratch_value_count_ = static_cast<int>(scratch_values);
  source_scratch_bytes_ =
      BuildOrdinarySourceByEdge<DeviceType>::scratch_bytes(source_scratch_value_count_);
  source_serial_work_count_ = candidate.maximum_source_serial_work_count();
  if (source_serial_work_count_ <= 0)
    error->all(FLERR, "Pair style ye3t/kk source work is invalid");
  use_neighbor_major_source_ = std::strcmp(execution_space_name<DeviceType>(), "non-device") != 0 &&
      source_serial_work_count_ <= YE3T_NEIGHBOR_MAJOR_SOURCE_WORK_LIMIT;
  using SourceTeamPolicy = Kokkos::TeamPolicy<DeviceType>;
  use_edge_team_source_ =
      source_scratch_bytes_ <= static_cast<std::size_t>(SourceTeamPolicy::scratch_size_max(0));
  edge_basis_cache_value_count_ = (use_neighbor_major_source_ || use_edge_team_source_)
      ? static_cast<int>(edge_cache_values)
      : 0;
  if (auto_replay.enabled()) {
    const char *source_policy = use_neighbor_major_source_ ? "center_tiled_neighbor_major_v1"
        : use_edge_team_source_                            ? "edge_team_atomic_v1"
                                                           : "component_reference_v1";
    const char *vjp_policy = (use_neighbor_major_source_ || use_edge_team_source_)
        ? "edge_grouped_harmonic_cached_radial_v1"
        : "edge_reference_v1";
    if (auto_replay.source_policy != source_policy || auto_replay.vjp_policy != vjp_policy)
      error->all(FLERR,
                 "Pair style ye3t/kk AUTO replay source or VJP policy "
                 "does not match the compiled device plan");
  }

  if (request_block && !candidate.has_block_program())
    error->all(FLERR,
               "Pair style ye3t/kk block device plan contains no "
               "selected block routes");
  if (request_scalar && !candidate.has_scalar_program())
    error->all(FLERR,
               "Pair style ye3t/kk scalar-power device plan contains no "
               "selected scalar-power routes");
  if (request_coupled && !candidate.has_coupled_program())
    error->all(FLERR,
               "Pair style ye3t/kk coupled-product device plan contains "
               "no selected coupled-product plans");
  if (request_direct && candidate.has_optimized_program())
    error->all(FLERR,
               "Pair style ye3t/kk direct device plan unexpectedly "
               "contains selected optimized routes");
  if (request_block && (candidate.has_scalar_program() || candidate.has_coupled_program()))
    error->all(FLERR,
               "Pair style ye3t/kk block device plan unexpectedly "
               "contains another optimized evaluator family");
  if (request_scalar && (candidate.has_block_program() || candidate.has_coupled_program()))
    error->all(FLERR,
               "Pair style ye3t/kk scalar-power device plan "
               "unexpectedly contains another optimized evaluator "
               "family");
  if (request_coupled && (candidate.has_block_program() || candidate.has_scalar_program()))
    error->all(FLERR,
               "Pair style ye3t/kk coupled-product device plan "
               "unexpectedly contains another optimized evaluator "
               "family");
  if (selected_block != candidate.has_block_program() ||
      selected_scalar != candidate.has_scalar_program() ||
      selected_coupled != candidate.has_coupled_program())
    error->all(FLERR,
               "Pair style ye3t/kk host and device evaluator-family "
               "selections disagree");

  flat_plan_hash_ = candidate.signature_hash();
  const char *evaluator_signature = request_auto ? "ye3t_kokkos_auto_catalogue_v1;"
                                                 : "ye3t_kokkos_direct_team_owned_reverse_v2;";
  if (!request_auto && request_block)
    evaluator_signature = use_work_major_block_ ? "ye3t_kokkos_block_plan_route_major_v1;"
                                                : "ye3t_kokkos_block_fused_center_lane_v2;";
  else if (!request_auto && request_scalar)
    evaluator_signature = "ye3t_kokkos_scalar_power_range_v1;";
  else if (!request_auto && request_coupled)
    evaluator_signature = "ye3t_kokkos_coupled_product_range_serial_v1;";
  std::string device_signature = std::string(evaluator_signature) +
      "owned_reverse_and_deferred_tallies_v1;deferred_status_v1;structured_block_tiles_v1;direct_"
      "center_owned_v1;harmonic_stream_v1;active_radial_evaluation_v1;" +
      (use_neighbor_major_source_  ? "source_center_tiled_neighbor_major_v1;"
           : use_edge_team_source_ ? "source_edge_team_v1;"
                                   : "source_component_reference_v1;") +
      ((use_neighbor_major_source_ || use_edge_team_source_)
           ? "vjp_edge_grouped_harmonic_cached_radial_v1;"
           : "vjp_edge_reference_v1;") +
      execution_space_name<DeviceType>() + ";fp64;device_class=" + device_class_hash_ + ";" +
      source_model_hash() + ";" + flat_plan_hash_ + ";" + candidate.vjp_schedule_hash() + ";" +
      std::to_string(source_scratch_value_count_) + ";" + std::to_string(source_scratch_bytes_) +
      ";" + std::to_string(source_serial_work_count_) + ";" +
      std::to_string(YE3T_NEIGHBOR_MAJOR_SOURCE_WORK_LIMIT) + ";" +
      std::to_string(YE3T_NEIGHBOR_MAJOR_SOURCE_TEAM_SIZE) + ";" +
      std::to_string(YE3T_NEIGHBOR_MAJOR_SOURCE_PADDING_LIMIT) + ";" +
      std::to_string(edge_basis_cache_value_count_) +
      (selected_block ? (use_work_major_block_ ? ";block_schedule=plan_route_major_v1"
                                               : ";block_schedule=fused_center_lane_v2")
                      : ";block_schedule=not_selected") +
      ";block_team_size=" + std::to_string(block_work_major_team_size_) + ";block_scratch_layout=" +
      (use_work_major_block_ ? "monomial_tile16_lane_fast_v2" : "not_selected") +
      ";block_scratch_bytes=" + std::to_string(block_work_major_scratch_bytes_) +
      ";exact_chunksize=" + (require_exact_chunksize_ ? "1" : "0") +
      ";direct_resident_values=" + std::to_string(direct_resident_values_) +
      ";configured_chunksize=" + std::to_string(chunksize()) + ";";
  if (request_auto)
    device_signature += semantic_selection_hash() + ";" + planner_profile + ";" +
        planner_calibration_hash + ";" + planner_decision_reason +
        ";auto_replay=" + (auto_replay.enabled() ? auto_replay.replay_hash : "not_applicable") +
        ";";
#if defined(LMP_KOKKOS_LAYOUT_LEGACY)
  device_signature += "layout_legacy";
#else
  device_signature += "layout_default";
#endif
  device_plan_hash_ = YE3T_LAMMPS::sha256_string(device_signature);
  if (comm->me == 0)
    utils::logmesg(lmp,
                   "YE3T GPU structured execution: DIRECT node-contiguous per center, resident "
                   "values {}; block monomial tile {}, transpose adjoints\n",
                   direct_resident_values_, YE3T_LAMMPS::GPU_BLOCK_MONOMIAL_TILE);

  auto require_rank_consensus = [&](const std::string &hash, const char *name) {
    if (hash.size() != 64) error->all(FLERR, "Pair style ye3t/kk {} is not a SHA-256 digest", name);
    std::array<char, 65> root_hash{};
    if (comm->me == 0) std::memcpy(root_hash.data(), hash.c_str(), hash.size());
    MPI_Bcast(root_hash.data(), static_cast<int>(root_hash.size()), MPI_CHAR, 0, world);
    const int mismatch = hash != std::string(root_hash.data());
    int any_mismatch = 0;
    MPI_Allreduce(&mismatch, &any_mismatch, 1, MPI_INT, MPI_MAX, world);
    if (any_mismatch) error->all(FLERR, "Pair style ye3t/kk {} differs across MPI ranks", name);
  };
  require_rank_consensus(semantic_selection_hash(), "semantic evaluator-selection hash");
  if (auto_replay.enabled()) require_rank_consensus(auto_replay.replay_hash, "AUTO replay hash");
  require_rank_consensus(device_plan_hash_, "device-plan hash");
  device_plan_ = std::move(candidate);
  type_to_species_ = std::move(candidate_type_map);
  device_status_ = std::move(candidate_status);
  center_counts_ = {};
  center_offsets_ = {};
  edge_centers_ = {};
  edge_neighbors_ = {};
  edge_bonds_ = {};
  edge_radius_ = {};
  edge_unit_ = {};
  edge_radial_derivative_ = {};
  edge_contracted_value_ = {};
  edge_contracted_derivative_ = {};
  source_real_ = {};
  source_imaginary_ = {};
  source_adjoint_real_ = {};
  source_adjoint_imaginary_ = {};
  dag_real_ = {};
  dag_imaginary_ = {};
  dag_adjoint_real_ = {};
  dag_adjoint_imaginary_ = {};
  total_density_real_ = {};
  total_density_imaginary_ = {};
  block_power_real_ = {};
  block_power_imaginary_ = {};
  block_output_real_ = {};
  block_output_imaginary_ = {};
  block_output_adjoint_real_ = {};
  block_output_adjoint_imaginary_ = {};
  block_monomial_real_ = {};
  block_monomial_imaginary_ = {};
  block_monomial_adjoint_real_ = {};
  block_monomial_adjoint_imaginary_ = {};
  block_route_density_real_ = {};
  block_route_density_imaginary_ = {};
  scalar_value_real_ = {};
  scalar_value_imaginary_ = {};
  scalar_adjoint_real_ = {};
  scalar_adjoint_imaginary_ = {};
  coupled_value_real_ = {};
  coupled_value_imaginary_ = {};
  coupled_adjoint_real_ = {};
  coupled_adjoint_imaginary_ = {};
  atomic_energies_ = {};
  edge_gradient_ = {};
  center_capacity_ = 0;
  edge_capacity_ = 0;
  dynamic_bytes_ = status_bytes;
  reallocation_count_ = 0;
  recorded_chunk_limit_ = 0;
  capacity_state_reported_ = false;
  replay_workload_validated_ = false;
  use_block_program_ = selected_block;
  use_scalar_program_ = selected_scalar;
  use_coupled_program_ = selected_coupled;
  device_plan_ready_ = true;

  const auto [free_bytes, total_bytes] = device_memory_info();
  std::string active_families;
  auto append_family = [&](const char *family, bool present) {
    if (!present) return;
    if (!active_families.empty()) active_families += "+";
    active_families += family;
  };
  append_family("direct", selected_direct);
  append_family("block", selected_block);
  append_family("scalar_power", selected_scalar);
  append_family("coupled_product", selected_coupled);
  std::string evaluator = "direct";
  if (request_auto)
    evaluator = "auto[" + active_families + "]";
  else if (use_block_program_)
    evaluator = "block";
  else if (use_scalar_program_)
    evaluator = "scalar_power";
  else if (use_coupled_program_)
    evaluator = "coupled_product";
  if (comm->me == 0)
    utils::logmesg(
        lmp,
        "YE3T Kokkos device-plan probe: execution_space {}, source_model {}, "
        "direct_logical_plan {}, requested_policy {}, evaluator {}, "
        "active_families {}, flat_plan {}, device_schedule {}, "
        "semantic_selection {}, planner_profile {}, planner_algorithm {}, "
        "planner_status {}, calibration_hash {}, decision_reason {}, "
        "auto_replay {}, kokkos_layout {}, "
        "selection_rank_consensus passed, rank_device_map {}, device_class {}, "
        "exact_chunk_required {}, configured_chunk {}, active_direct {}, "
        "active_block {}, "
        "active_scalar_power {}, active_coupled_product {}, "
        "static_bytes {}, direct_monomial_storage {}, block_routes {}, "
        "scalar_routes {}, coupled_plans {}, "
        "block_power_storage {}, scalar_value_storage {}, "
        "coupled_component_storage {}, "
        "block_output_storage {}, block_monomial_workspace {}, block_schedule "
        "{}, block_identity_child_plans {}, "
        "block_symmetric_power_child_plans {}, block_team_size {}, "
        "block_scratch_layout {}, block_scratch_bytes {}, source_policy {}, "
        "source_serial_terms {}, source_scratch_bytes {}, "
        "edge_basis_cache_values_per_edge {}, "
        "free_device_bytes {}, VJP_policy {}, "
        "VJP_schedule {}, VJP_harmonic_groups {}, VJP_angular_terms {}, "
        "VJP_maximum_group_terms {}, "
        "total_device_bytes {}, copy_sentinel_maximum_error {}, "
        "scalar_math_probe {}, scalar_math_forward_error {}, "
        "scalar_math_adjoint_dot_error {}, scalar_math_zero_error {}, "
        "coupled_math_probe {}, coupled_math_forward_error {}, "
        "coupled_math_adjoint_dot_error {}, coupled_math_zero_error {}\n",
        execution_space_name<DeviceType>(), source_model_hash(), logical_direct_plan_hash(),
        request_auto          ? "auto"
            : request_block   ? "block"
            : request_scalar  ? "scalar_power"
            : request_coupled ? "coupled_product"
                              : "direct",
        evaluator, active_families, flat_plan_hash_, device_plan_hash_, semantic_selection_hash(),
        planner_profile, planner_algorithm, planner_status, planner_calibration_hash,
        planner_decision_reason, auto_replay.enabled() ? auto_replay.replay_hash : "not_applicable",
#if defined(LMP_KOKKOS_LAYOUT_LEGACY)
        "legacy",
#else
        "default",
#endif
        rank_device_map_hash_, device_class_hash_, require_exact_chunksize_ ? "yes" : "no",
        chunksize(), selected_direct ? "yes" : "no", selected_block ? "yes" : "no",
        selected_scalar ? "yes" : "no", selected_coupled ? "yes" : "no",
        device_plan_.memory_usage(), device_plan_.maximum_monomial_count(),
        device_plan_.block_route_count(), device_plan_.scalar_route_count(),
        device_plan_.coupled_plan_count(), device_plan_.maximum_block_power_storage(),
        device_plan_.maximum_scalar_value_count(), device_plan_.maximum_coupled_component_count(),
        device_plan_.maximum_block_output_storage(), device_plan_.block_monomial_tile_count(),
        selected_block ? (use_work_major_block_ ? "plan_route_major_v1" : "fused_center_lane_v2")
                       : "not_selected",
        block_identity_child_plans, block_symmetric_power_child_plans, block_work_major_team_size_,
        use_work_major_block_ ? "monomial_tile16_lane_fast_v2" : "not_selected",
        block_work_major_scratch_bytes_,
        use_neighbor_major_source_  ? "center_tiled_neighbor_major_v1"
            : use_edge_team_source_ ? "edge_team_atomic_v1"
                                    : "component_reference_v1",
        source_serial_work_count_, source_scratch_bytes_, edge_basis_cache_value_count_, free_bytes,
        (use_neighbor_major_source_ || use_edge_team_source_)
            ? "edge_grouped_harmonic_cached_radial_v1"
            : "edge_reference_v1",
        device_plan_.vjp_schedule_hash(), device_plan_.vjp_harmonic_group_count(),
        device_plan_.vjp_angular_term_count(), device_plan_.vjp_maximum_group_term_count(),
        total_bytes, probe_error, selected_scalar ? "passed" : "not_run",
        scalar_math_probe.forward_error, scalar_math_probe.adjoint_dot_error,
        scalar_math_probe.zero_error, selected_coupled ? "passed" : "not_run",
        coupled_math_probe.forward_error, coupled_math_probe.adjoint_dot_error,
        coupled_math_probe.zero_error);
}

template <class DeviceType> void PairYE3TKokkos<DeviceType>::init_style()
{
  const bool device_plan_missing = is_lifted_cauchy_model()
      ? !lifted_device_plan_ready_
      : (is_tagged_cauchy_model() ? !tagged_device_plan_ready_ : !device_plan_ready_);
  if (!model_loaded() || device_plan_missing)
    error->all(FLERR, "Pair style ye3t/kk requires pair_coeff before run");
  if (atom->tag_enable == 0) error->all(FLERR, "Pair style ye3t/kk requires atom IDs");
  if (force->newton_pair == 0) error->all(FLERR, "Pair style ye3t/kk requires newton pair on");
  if (execution_space != Device)
    error->all(FLERR, "Pair style ye3t/kk requires a Kokkos device execution space");

  neighflag_ = lmp->kokkos->neighflag;
  auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
  request->set_kokkos_host(false);
  request->set_kokkos_device(true);
  if (neighflag_ == FULL)
    error->all(FLERR, "Pair style ye3t/kk requires Kokkos half or half/thread neighbor mode");
}

template <class DeviceType> void PairYE3TKokkos<DeviceType>::compute_lifted(int eflag, int vflag)
{
  const Kokkos::Profiling::ScopedRegion compute_region("YE3T::lifted::compute");
  ev_init(eflag, vflag, 0);
  step_state_.reset();
  // The status word accumulates across chunks and is read once per step.
  Kokkos::deep_copy(DeviceType{}, device_status_, 0);

  if (eflag_atom) {
    if (d_eatom_.extent(0) < static_cast<std::size_t>(maxeatom)) {
      memoryKK->destroy_kokkos(k_eatom_, eatom);
      memoryKK->create_kokkos(k_eatom_, eatom, maxeatom, "pair:eatom");
    }
    d_eatom_ = k_eatom_.template view<DeviceType>();
    Kokkos::deep_copy(DeviceType{}, d_eatom_, 0.0);
  }
  if (vflag_atom) {
    if (d_vatom_.extent(0) < static_cast<std::size_t>(maxvatom)) {
      memoryKK->destroy_kokkos(k_vatom_, vatom);
      memoryKK->create_kokkos(k_vatom_, vatom, maxvatom, "pair:vatom");
    }
    d_vatom_ = k_vatom_.template view<DeviceType>();
    Kokkos::deep_copy(DeviceType{}, d_vatom_, 0.0);
  }

  atomKK->sync(execution_space, X_MASK | F_MASK | TYPE_MASK);
  x = atomKK->k_x.template view<DeviceType>();
  f = atomKK->k_f.template view<DeviceType>();
  type_ = atomKK->k_type.template view<DeviceType>();

  auto *k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  d_neighbors_ = k_list->d_neighbors;
  d_ilist_ = k_list->d_ilist;
  d_numneigh_ = k_list->d_numneigh;
  const int inum = list->inum;

  need_dup_ = lmp->kokkos->need_dup<DeviceType>();
  if (need_dup_) {
    duplicated_force_ =
        Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum,
                                                  Kokkos::Experimental::ScatterDuplicated>(f);
    if (vflag_atom)
      duplicated_vatom_ = Kokkos::Experimental::create_scatter_view<
          Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_vatom_);
  } else {
    atomic_force_ =
        Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum,
                                                  Kokkos::Experimental::ScatterNonDuplicated>(f);
    if (vflag_atom)
      atomic_vatom_ = Kokkos::Experimental::create_scatter_view<
          Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_vatom_);
  }

  EV_FLOAT total_ev;
  const int reallocations_before = reallocation_count_;
  const int requested_chunk = std::min(chunksize(), inum);
  const int allocation_request = require_exact_chunksize_ ? chunksize() : requested_chunk;
  const int allocated_chunk = allocation_request > 0 ? ensure_center_capacity(allocation_request)
                                                     : 0;
  if (require_exact_chunksize_ && allocated_chunk != allocation_request)
    error->one(FLERR,
               "Pair style ye3t/kk exact lifted chunksize allocation failed: "
               "configured {}, allocated {}, center capacity {}, dynamic bytes {}",
               allocation_request, allocated_chunk, center_capacity_, dynamic_bytes_);
  int effective_chunk = std::min(requested_chunk, allocated_chunk);
  if (recorded_chunk_limit_ > 0) effective_chunk = std::min(effective_chunk, recorded_chunk_limit_);
  if (inum > 0 && effective_chunk <= 0)
    error->one(FLERR, "Pair style ye3t/kk has no usable lifted center chunk");
  const int center_reduced = effective_chunk < requested_chunk ? 1 : 0;
  int smallest_effective_chunk = effective_chunk;
  int edge_reductions = 0;
  int chunk_begin = 0;
  const auto lifted_plan = lifted_device_plan_.views();
  while (chunk_begin < inum) {
    YE3TProfilePhase phase("YE3T::edges_and_capacity");
    int center_count = std::min(effective_chunk, inum - chunk_begin);
    std::int64_t edge_count_64 = 0;
    while (true) {
      Kokkos::parallel_for("YE3TCountLiftedEdges", Kokkos::RangePolicy<DeviceType>(0, center_count),
                           CountLiftedEdges<DeviceType>{
                               x, type_, d_neighbors_, d_ilist_, d_numneigh_, type_to_species_,
                               center_counts_, device_status_, lifted_plan.cutoff, chunk_begin});
      Kokkos::parallel_scan("YE3TScanLiftedEdges",
                            Kokkos::RangePolicy<DeviceType>(0, center_count + 1),
                            ScanEdges<DeviceType>{center_counts_, center_offsets_, center_count,
                                                  step_state_.summary_view(), device_status_,
                                                  step_state_.maximum_edges, false});
      const auto edge_summary = step_state_.read_edge_summary();
      edge_count_64 = edge_summary.edge_count;
      report_device_status(edge_summary.status);
      if (edge_count_64 > std::numeric_limits<int>::max()) {
        if (center_count == 1)
          error->one(FLERR, "One YE3T Kokkos lifted center exceeds 32-bit edge indexing");
      } else if (ensure_edge_capacity(static_cast<int>(edge_count_64))) {
        break;
      }
      if (require_exact_chunksize_)
        error->one(FLERR,
                   "Pair style ye3t/kk exact lifted chunksize edge allocation failed: "
                   "center count {}, requested edges {}, edge capacity {}",
                   center_count, edge_count_64, edge_capacity_);
      if (capacity_state_reported_)
        error->one(FLERR,
                   "Pair style ye3t/kk would reduce its reported lifted chunk: "
                   "center count {}, requested edges {}, edge capacity {}",
                   center_count, edge_count_64, edge_capacity_);
      if (center_count == 1)
        error->one(FLERR,
                   "One YE3T Kokkos lifted center cannot fit while "
                   "preserving device-memory headroom");
      center_count = (center_count + 1) / 2;
      smallest_effective_chunk = std::min(smallest_effective_chunk, center_count);
      ++edge_reductions;
    }
    const int edge_count = static_cast<int>(edge_count_64);

    Kokkos::parallel_for(
        "YE3TFillLiftedEdges", Kokkos::RangePolicy<DeviceType>(0, center_count),
        FillLiftedEdges<DeviceType>{x, type_, d_neighbors_, d_ilist_, d_numneigh_, type_to_species_,
                                    center_offsets_, edge_centers_, edge_neighbors_, edge_bonds_,
                                    edge_radius_, edge_unit_, device_status_, lifted_plan.cutoff,
                                    chunk_begin, edge_capacity_});

    phase.next("YE3T::source");
    const std::int64_t source_iterations =
        static_cast<std::int64_t>(lifted_plan.source_row_count) * center_count;
    Kokkos::parallel_for(
        "YE3TBuildLiftedDirectQSource",
        Kokkos::RangePolicy<DeviceType, Kokkos::IndexType<std::int64_t>>(0, source_iterations),
        BuildLiftedDirectQSource<DeviceType>{lifted_plan, center_offsets_, edge_bonds_,
                                             edge_radius_, edge_unit_, source_real_, device_status_,
                                             center_count, center_capacity_, edge_capacity_});
    phase.next("YE3T::readout_forward_adjoint");
    Kokkos::parallel_for(
        "YE3TLiftedSparseForwardAdjoint", Kokkos::RangePolicy<DeviceType>(0, center_count),
        LiftedSparseForwardAdjoint<DeviceType>{
            type_, d_ilist_, type_to_species_, lifted_plan, source_real_, source_adjoint_real_,
            lifted_workspace_, atomic_energies_, device_status_, chunk_begin, center_capacity_});
    phase.next("YE3T::source_pullback");
    if (edge_count > 0)
      Kokkos::parallel_for(
          "YE3TLiftedDirectQSourceVJP", Kokkos::RangePolicy<DeviceType>(0, edge_count),
          LiftedDirectQSourceVJP<DeviceType>{lifted_plan, edge_centers_, edge_bonds_, edge_radius_,
                                             edge_unit_, source_adjoint_real_, edge_gradient_,
                                             device_status_, center_capacity_, edge_capacity_});

    phase.next("YE3T::force_virial_scatter");
    auto run_scatter = [&](auto neighbor_tag, auto virial_tag) {
      constexpr int neighbor_mode = decltype(neighbor_tag)::value;
      constexpr bool tally_atom_virial = decltype(virial_tag)::value;
      using Functor = ScatterForceEnergyVirial<DeviceType, neighbor_mode, tally_atom_virial>;
      const Functor functor{x,
                            d_ilist_,
                            center_offsets_,
                            edge_neighbors_,
                            edge_gradient_,
                            atomic_energies_,
                            d_eatom_,
                            d_vatom_,
                            duplicated_force_,
                            atomic_force_,
                            duplicated_vatom_,
                            atomic_vatom_,
                            chunk_begin,
                            edge_capacity_,
                            eflag_global,
                            eflag_atom,
                            vflag_global,
                            vflag_either};
      if (eflag_global || vflag_global) {
        Kokkos::parallel_reduce("YE3TScatterLiftedForceEnergyVirial",
                                Kokkos::RangePolicy<DeviceType>(0, center_count), functor,
                                step_state_.ev_chunk);
      } else {
        Kokkos::parallel_for("YE3TScatterLiftedForceOnly",
                             Kokkos::RangePolicy<DeviceType>(0, center_count), functor);
      }
    };

    if (neighflag_ == HALF) {
      if (vflag_atom)
        run_scatter(std::integral_constant<int, HALF>{}, std::true_type{});
      else
        run_scatter(std::integral_constant<int, HALF>{}, std::false_type{});
    } else {
      if (vflag_atom)
        run_scatter(std::integral_constant<int, HALFTHREAD>{}, std::true_type{});
      else
        run_scatter(std::integral_constant<int, HALFTHREAD>{}, std::false_type{});
    }
    step_state_.accumulate(eflag_global || vflag_global, false);
    chunk_begin += center_count;
  }

  // One fence per step: tallies and the deferred status word together. Every
  // kernel that raises a status writes finite placeholders, so deferral cannot
  // index out of range, and the error still precedes any integration.
  int deferred_status = 0;
  const auto step_totals = step_state_.read_totals(device_status_, deferred_status);
  report_device_status(deferred_status);
  if (eflag_global || vflag_global) total_ev = step_totals.ev;

  if (need_dup_) Kokkos::Experimental::contribute(f, duplicated_force_);
  if (eflag_global) eng_vdwl += static_cast<double>(total_ev.evdwl);
  if (vflag_global)
    for (int component = 0; component < 6; ++component)
      virial[component] += static_cast<double>(total_ev.v[component]);
  if (vflag_fdotr) pair_virial_fdotr_compute(this);
  if (eflag_atom) {
    k_eatom_.template modify<DeviceType>();
    k_eatom_.sync_host();
  }
  if (vflag_atom) {
    if (need_dup_) Kokkos::Experimental::contribute(d_vatom_, duplicated_vatom_);
    k_vatom_.template modify<DeviceType>();
    k_vatom_.sync_host();
  }
  atomKK->modified(execution_space, F_MASK);
  set_maximum_imaginary_density(0.0);

  if (!capacity_state_reported_) {
    recorded_chunk_limit_ = smallest_effective_chunk;
    report_initial_capacity_state(inum, requested_chunk, smallest_effective_chunk, center_reduced,
                                  edge_reductions, reallocations_before);
  }
  duplicated_force_ = {};
  atomic_force_ = {};
  duplicated_vatom_ = {};
  atomic_vatom_ = {};
}

template <class DeviceType> void PairYE3TKokkos<DeviceType>::compute_tagged(int eflag, int vflag)
{
  const Kokkos::Profiling::ScopedRegion compute_region("YE3T::tagged::compute");
  ev_init(eflag, vflag, 0);
  step_state_.reset();
  // The status word accumulates across chunks and is read once per step.
  Kokkos::deep_copy(DeviceType{}, device_status_, 0);

  if (eflag_atom) {
    if (d_eatom_.extent(0) < static_cast<std::size_t>(maxeatom)) {
      memoryKK->destroy_kokkos(k_eatom_, eatom);
      memoryKK->create_kokkos(k_eatom_, eatom, maxeatom, "pair:eatom");
    }
    d_eatom_ = k_eatom_.template view<DeviceType>();
    Kokkos::deep_copy(DeviceType{}, d_eatom_, 0.0);
  }
  if (vflag_atom) {
    if (d_vatom_.extent(0) < static_cast<std::size_t>(maxvatom)) {
      memoryKK->destroy_kokkos(k_vatom_, vatom);
      memoryKK->create_kokkos(k_vatom_, vatom, maxvatom, "pair:vatom");
    }
    d_vatom_ = k_vatom_.template view<DeviceType>();
    Kokkos::deep_copy(DeviceType{}, d_vatom_, 0.0);
  }

  atomKK->sync(execution_space, X_MASK | F_MASK | TYPE_MASK);
  x = atomKK->k_x.template view<DeviceType>();
  f = atomKK->k_f.template view<DeviceType>();
  type_ = atomKK->k_type.template view<DeviceType>();

  auto *k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  d_neighbors_ = k_list->d_neighbors;
  d_ilist_ = k_list->d_ilist;
  d_numneigh_ = k_list->d_numneigh;
  const int inum = list->inum;

  need_dup_ = lmp->kokkos->need_dup<DeviceType>();
  if (need_dup_) {
    duplicated_force_ =
        Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum,
                                                  Kokkos::Experimental::ScatterDuplicated>(f);
    if (vflag_atom)
      duplicated_vatom_ = Kokkos::Experimental::create_scatter_view<
          Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_vatom_);
  } else {
    atomic_force_ =
        Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum,
                                                  Kokkos::Experimental::ScatterNonDuplicated>(f);
    if (vflag_atom)
      atomic_vatom_ = Kokkos::Experimental::create_scatter_view<
          Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_vatom_);
  }

  EV_FLOAT total_ev;
  const int reallocations_before = reallocation_count_;
  const int requested_chunk = std::min(chunksize(), inum);
  const int allocation_request = require_exact_chunksize_ ? chunksize() : requested_chunk;
  const int allocated_chunk = allocation_request > 0 ? ensure_center_capacity(allocation_request)
                                                     : 0;
  if (require_exact_chunksize_ && allocated_chunk != allocation_request)
    error->one(FLERR,
               "Pair style ye3t/kk exact tagged chunksize allocation failed: "
               "configured {}, allocated {}, center capacity {}, dynamic bytes {}",
               allocation_request, allocated_chunk, center_capacity_, dynamic_bytes_);
  int effective_chunk = std::min(requested_chunk, allocated_chunk);
  if (recorded_chunk_limit_ > 0) effective_chunk = std::min(effective_chunk, recorded_chunk_limit_);
  if (inum > 0 && effective_chunk <= 0)
    error->one(FLERR, "Pair style ye3t/kk has no usable tagged center chunk");
  const int center_reduced = effective_chunk < requested_chunk ? 1 : 0;
  int smallest_effective_chunk = effective_chunk;
  int edge_reductions = 0;
  int chunk_begin = 0;
  const auto tagged_plan = tagged_device_plan_.views();
  while (chunk_begin < inum) {
    YE3TProfilePhase phase("YE3T::edges_and_capacity");
    int center_count = std::min(effective_chunk, inum - chunk_begin);
    std::int64_t edge_count_64 = 0;
    while (true) {
      Kokkos::parallel_for("YE3TCountTaggedEdges", Kokkos::RangePolicy<DeviceType>(0, center_count),
                           CountLiftedEdges<DeviceType>{
                               x, type_, d_neighbors_, d_ilist_, d_numneigh_, type_to_species_,
                               center_counts_, device_status_, tagged_plan.cutoff, chunk_begin});
      Kokkos::parallel_scan("YE3TScanTaggedEdges",
                            Kokkos::RangePolicy<DeviceType>(0, center_count + 1),
                            ScanEdges<DeviceType>{center_counts_, center_offsets_, center_count,
                                                  step_state_.summary_view(), device_status_,
                                                  step_state_.maximum_edges, false});
      const auto edge_summary = step_state_.read_edge_summary();
      edge_count_64 = edge_summary.edge_count;
      report_device_status(edge_summary.status);
      if (edge_count_64 > std::numeric_limits<int>::max()) {
        if (center_count == 1)
          error->one(FLERR, "One YE3T Kokkos tagged center exceeds 32-bit edge indexing");
      } else if (ensure_edge_capacity(static_cast<int>(edge_count_64))) {
        break;
      }
      if (require_exact_chunksize_)
        error->one(FLERR,
                   "Pair style ye3t/kk exact tagged chunksize edge allocation failed: "
                   "center count {}, requested edges {}, edge capacity {}",
                   center_count, edge_count_64, edge_capacity_);
      if (capacity_state_reported_)
        error->one(FLERR,
                   "Pair style ye3t/kk would reduce its reported tagged chunk: "
                   "center count {}, requested edges {}, edge capacity {}",
                   center_count, edge_count_64, edge_capacity_);
      if (center_count == 1)
        error->one(FLERR,
                   "One YE3T Kokkos tagged center cannot fit while "
                   "preserving device-memory headroom");
      center_count = (center_count + 1) / 2;
      smallest_effective_chunk = std::min(smallest_effective_chunk, center_count);
      ++edge_reductions;
    }
    const int edge_count = static_cast<int>(edge_count_64);

    Kokkos::parallel_for(
        "YE3TFillTaggedEdges", Kokkos::RangePolicy<DeviceType>(0, center_count),
        FillLiftedEdges<DeviceType>{x, type_, d_neighbors_, d_ilist_, d_numneigh_, type_to_species_,
                                    center_offsets_, edge_centers_, edge_neighbors_, edge_bonds_,
                                    edge_radius_, edge_unit_, device_status_, tagged_plan.cutoff,
                                    chunk_begin, edge_capacity_});

    phase.next("YE3T::source");
    Kokkos::parallel_for(
        "YE3TTaggedBuildDensityMoment", Kokkos::RangePolicy<DeviceType>(0, center_count),
        TaggedBuildDensityMoment<DeviceType>{
            tagged_plan, center_offsets_, edge_bonds_, edge_radius_, edge_unit_, tagged_density_,
            tagged_moment_, device_status_, center_capacity_, edge_capacity_});

    phase.next("YE3T::readout_forward_adjoint");
    Kokkos::parallel_for(
        "YE3TTaggedTermForwardAdjoint", Kokkos::RangePolicy<DeviceType>(0, center_count),
        TaggedTermForwardAdjoint<DeviceType>{
            type_, d_ilist_, type_to_species_, tagged_plan, center_offsets_, tagged_density_,
            tagged_moment_, tagged_density_adjoint_, tagged_moment_adjoint_, atomic_energies_,
            device_status_, chunk_begin, center_capacity_});

    phase.next("YE3T::source_pullback");
    if (edge_count > 0) {
      auto run_tagged_vjp = [&](auto density_only_tag) {
        constexpr bool density_only = decltype(density_only_tag)::value;
        Kokkos::parallel_for(density_only ? "YE3TTaggedDensityOnlyEdgeVJP" : "YE3TTaggedEdgeVJP",
                             Kokkos::RangePolicy<DeviceType>(0, edge_count),
                             TaggedEdgeVJP<DeviceType, density_only>{
                                 tagged_plan, edge_centers_, edge_bonds_, edge_radius_, edge_unit_,
                                 tagged_density_adjoint_, tagged_moment_adjoint_, edge_gradient_,
                                 device_status_, center_capacity_, edge_capacity_});
      };
      if (tagged_plan.physical_image_v3 && tagged_plan.moment_key_count == 0)
        run_tagged_vjp(std::true_type{});
      else
        run_tagged_vjp(std::false_type{});
    }

    phase.next("YE3T::force_virial_scatter");
    auto run_scatter = [&](auto neighbor_tag, auto virial_tag) {
      constexpr int neighbor_mode = decltype(neighbor_tag)::value;
      constexpr bool tally_atom_virial = decltype(virial_tag)::value;
      using Functor = ScatterForceEnergyVirial<DeviceType, neighbor_mode, tally_atom_virial>;
      const Functor functor{x,
                            d_ilist_,
                            center_offsets_,
                            edge_neighbors_,
                            edge_gradient_,
                            atomic_energies_,
                            d_eatom_,
                            d_vatom_,
                            duplicated_force_,
                            atomic_force_,
                            duplicated_vatom_,
                            atomic_vatom_,
                            chunk_begin,
                            edge_capacity_,
                            eflag_global,
                            eflag_atom,
                            vflag_global,
                            vflag_either};
      if (eflag_global || vflag_global) {
        Kokkos::parallel_reduce("YE3TScatterTaggedForceEnergyVirial",
                                Kokkos::RangePolicy<DeviceType>(0, center_count), functor,
                                step_state_.ev_chunk);
      } else {
        Kokkos::parallel_for("YE3TScatterTaggedForceOnly",
                             Kokkos::RangePolicy<DeviceType>(0, center_count), functor);
      }
    };

    if (neighflag_ == HALF) {
      if (vflag_atom)
        run_scatter(std::integral_constant<int, HALF>{}, std::true_type{});
      else
        run_scatter(std::integral_constant<int, HALF>{}, std::false_type{});
    } else {
      if (vflag_atom)
        run_scatter(std::integral_constant<int, HALFTHREAD>{}, std::true_type{});
      else
        run_scatter(std::integral_constant<int, HALFTHREAD>{}, std::false_type{});
    }
    step_state_.accumulate(eflag_global || vflag_global, false);
    chunk_begin += center_count;
  }

  // One fence per step: tallies and the deferred status word together. Every
  // kernel that raises a status writes finite placeholders, so deferral cannot
  // index out of range, and the error still precedes any integration.
  int deferred_status = 0;
  const auto step_totals = step_state_.read_totals(device_status_, deferred_status);
  report_device_status(deferred_status);
  if (eflag_global || vflag_global) total_ev = step_totals.ev;

  if (need_dup_) Kokkos::Experimental::contribute(f, duplicated_force_);
  if (eflag_global) eng_vdwl += static_cast<double>(total_ev.evdwl);
  if (vflag_global)
    for (int component = 0; component < 6; ++component)
      virial[component] += static_cast<double>(total_ev.v[component]);
  if (vflag_fdotr) pair_virial_fdotr_compute(this);
  if (eflag_atom) {
    k_eatom_.template modify<DeviceType>();
    k_eatom_.sync_host();
  }
  if (vflag_atom) {
    if (need_dup_) Kokkos::Experimental::contribute(d_vatom_, duplicated_vatom_);
    k_vatom_.template modify<DeviceType>();
    k_vatom_.sync_host();
  }
  atomKK->modified(execution_space, F_MASK);
  set_maximum_imaginary_density(0.0);

  if (!capacity_state_reported_) {
    recorded_chunk_limit_ = smallest_effective_chunk;
    report_initial_capacity_state(inum, requested_chunk, smallest_effective_chunk, center_reduced,
                                  edge_reductions, reallocations_before);
  }
  duplicated_force_ = {};
  atomic_force_ = {};
  duplicated_vatom_ = {};
  atomic_vatom_ = {};
}

template <class DeviceType> void PairYE3TKokkos<DeviceType>::compute(int eflag, int vflag)
{
  if (is_tagged_cauchy_model()) {
    compute_tagged(eflag, vflag);
    return;
  }
  if (is_lifted_cauchy_model()) {
    compute_lifted(eflag, vflag);
    return;
  }
  const Kokkos::Profiling::ScopedRegion compute_region("YE3T::ace::compute");
  ev_init(eflag, vflag, 0);
  step_state_.reset();
  // The status word accumulates across chunks and is read once per step.
  Kokkos::deep_copy(DeviceType{}, device_status_, 0);

  if (eflag_atom) {
    if (d_eatom_.extent(0) < static_cast<std::size_t>(maxeatom)) {
      memoryKK->destroy_kokkos(k_eatom_, eatom);
      memoryKK->create_kokkos(k_eatom_, eatom, maxeatom, "pair:eatom");
    }
    d_eatom_ = k_eatom_.template view<DeviceType>();
    Kokkos::deep_copy(DeviceType{}, d_eatom_, 0.0);
  }
  if (vflag_atom) {
    if (d_vatom_.extent(0) < static_cast<std::size_t>(maxvatom)) {
      memoryKK->destroy_kokkos(k_vatom_, vatom);
      memoryKK->create_kokkos(k_vatom_, vatom, maxvatom, "pair:vatom");
    }
    d_vatom_ = k_vatom_.template view<DeviceType>();
    Kokkos::deep_copy(DeviceType{}, d_vatom_, 0.0);
  }

  atomKK->sync(execution_space, X_MASK | F_MASK | TYPE_MASK);
  x = atomKK->k_x.template view<DeviceType>();
  f = atomKK->k_f.template view<DeviceType>();
  type_ = atomKK->k_type.template view<DeviceType>();

  auto *k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  d_neighbors_ = k_list->d_neighbors;
  d_ilist_ = k_list->d_ilist;
  d_numneigh_ = k_list->d_numneigh;
  const int inum = list->inum;
  const auto &auto_replay = model().auto_replay();
  if (auto_replay.enabled() && inum > 0) {
    if (inum < auto_replay.minimum_centers_per_rank || inum > auto_replay.maximum_centers_per_rank)
      error->one(FLERR,
                 "Pair style ye3t/kk local center count {} is outside the AUTO "
                 "replay envelope [{}, {}]",
                 inum, auto_replay.minimum_centers_per_rank, auto_replay.maximum_centers_per_rank);
    if (!replay_workload_validated_ && comm->me == 0)
      utils::logmesg(lmp,
                     "YE3T Kokkos AUTO replay workload: hash {}, local "
                     "centers {}, calibrated envelope [{}, {}], passed\n",
                     auto_replay.replay_hash, inum, auto_replay.minimum_centers_per_rank,
                     auto_replay.maximum_centers_per_rank);
    replay_workload_validated_ = true;
  }

  need_dup_ = lmp->kokkos->need_dup<DeviceType>();
  if (need_dup_) {
    duplicated_force_ =
        Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum,
                                                  Kokkos::Experimental::ScatterDuplicated>(f);
    if (vflag_atom)
      duplicated_vatom_ = Kokkos::Experimental::create_scatter_view<
          Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_vatom_);
  } else {
    atomic_force_ =
        Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum,
                                                  Kokkos::Experimental::ScatterNonDuplicated>(f);
    if (vflag_atom)
      atomic_vatom_ = Kokkos::Experimental::create_scatter_view<
          Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_vatom_);
  }

  EV_FLOAT total_ev;
  const int reallocations_before = reallocation_count_;
  const int requested_chunk = std::min(chunksize(), inum);
  const int allocation_request = require_exact_chunksize_ ? chunksize() : requested_chunk;
  const int allocated_chunk = allocation_request > 0 ? ensure_center_capacity(allocation_request)
                                                     : 0;
  if (require_exact_chunksize_ && allocated_chunk != allocation_request)
    error->one(FLERR,
               "Pair style ye3t/kk exact chunksize allocation failed: configured {}, "
               "allocated {}, center capacity {}, dynamic bytes {}",
               allocation_request, allocated_chunk, center_capacity_, dynamic_bytes_);
  int effective_chunk = std::min(requested_chunk, allocated_chunk);
  if (recorded_chunk_limit_ > 0) effective_chunk = std::min(effective_chunk, recorded_chunk_limit_);
  if (inum > 0 && effective_chunk <= 0)
    error->one(FLERR, "Pair style ye3t/kk has no usable center chunk");
  const int center_reduced = effective_chunk < requested_chunk ? 1 : 0;
  int smallest_effective_chunk = effective_chunk;
  int edge_reductions = 0;
  int chunk_begin = 0;
  while (chunk_begin < inum) {
    YE3TProfilePhase phase("YE3T::edges_and_capacity");
    int center_count = std::min(effective_chunk, inum - chunk_begin);
    std::int64_t edge_count_64 = 0;
    int maximum_edge_count = 0;
    while (true) {
      maximum_edge_count = 0;
      const CountEdges<DeviceType> count_edges{x,
                                               type_,
                                               d_neighbors_,
                                               d_ilist_,
                                               d_numneigh_,
                                               type_to_species_,
                                               device_plan_.bond_views(),
                                               center_counts_,
                                               device_status_,
                                               chunk_begin};
      if (use_neighbor_major_source_) {
        Kokkos::parallel_reduce(
            "YE3TCountEdges", Kokkos::RangePolicy<DeviceType>(0, center_count), count_edges,
            Kokkos::Max<int, typename DeviceType::memory_space>(step_state_.maximum_edges));
      } else {
        Kokkos::parallel_for("YE3TCountEdges", Kokkos::RangePolicy<DeviceType>(0, center_count),
                             count_edges);
      }
      Kokkos::parallel_scan("YE3TScanEdges", Kokkos::RangePolicy<DeviceType>(0, center_count + 1),
                            ScanEdges<DeviceType>{center_counts_, center_offsets_, center_count,
                                                  step_state_.summary_view(), device_status_,
                                                  step_state_.maximum_edges,
                                                  use_neighbor_major_source_});
      const auto edge_summary = step_state_.read_edge_summary();
      edge_count_64 = edge_summary.edge_count;
      maximum_edge_count = edge_summary.maximum_edges;
      report_device_status(edge_summary.status);
      if (edge_count_64 > std::numeric_limits<int>::max()) {
        if (center_count == 1)
          error->one(FLERR,
                     "One YE3T Kokkos center environment exceeds 32-bit "
                     "edge indexing");
      } else if (ensure_edge_capacity(static_cast<int>(edge_count_64))) {
        break;
      }
      if (require_exact_chunksize_)
        error->one(FLERR,
                   "Pair style ye3t/kk exact chunksize edge allocation failed: "
                   "center count {}, requested edges {}, edge capacity {}, dynamic "
                   "bytes {}",
                   center_count, edge_count_64, edge_capacity_, dynamic_bytes_);
      if (capacity_state_reported_)
        error->one(FLERR,
                   "Pair style ye3t/kk would reduce its previously reported chunk: "
                   "center count {}, requested edges {}, edge capacity {}",
                   center_count, edge_count_64, edge_capacity_);
      if (center_count == 1)
        error->one(FLERR,
                   "One YE3T Kokkos center environment cannot fit while "
                   "preserving device-memory headroom");
      center_count = (center_count + 1) / 2;
      smallest_effective_chunk = std::min(smallest_effective_chunk, center_count);
      ++edge_reductions;
    }
    const int edge_count = static_cast<int>(edge_count_64);

    Kokkos::parallel_for("YE3TFillEdges", Kokkos::RangePolicy<DeviceType>(0, center_count),
                         FillEdges<DeviceType>{x, type_, d_neighbors_, d_ilist_, d_numneigh_,
                                               type_to_species_, device_plan_.bond_views(),
                                               center_offsets_, edge_centers_, edge_neighbors_,
                                               edge_bonds_, edge_radius_, edge_unit_,
                                               device_status_, chunk_begin, edge_capacity_});

    phase.next("YE3T::source");
    const int source_iterations = device_plan_.maximum_source_count() * center_count;
    const bool use_edge_cache = use_neighbor_major_source_ || use_edge_team_source_;
    if (use_edge_cache) {
      Kokkos::parallel_for("YE3TZeroOrdinarySource",
                           Kokkos::RangePolicy<DeviceType>(0, source_iterations),
                           ZeroOrdinarySource<DeviceType>{source_real_, source_imaginary_,
                                                          center_count, center_capacity_});
      if (edge_count > 0) {
        const int center_tile_count = (center_count + YE3T_NEIGHBOR_MAJOR_SOURCE_TEAM_SIZE - 1) /
            YE3T_NEIGHBOR_MAJOR_SOURCE_TEAM_SIZE;
        const std::int64_t neighbor_major_league =
            static_cast<std::int64_t>(center_tile_count) * maximum_edge_count;
        const std::int64_t neighbor_major_threads =
            neighbor_major_league * YE3T_NEIGHBOR_MAJOR_SOURCE_TEAM_SIZE;
        const bool use_neighbor_major_chunk = use_neighbor_major_source_ &&
            (!use_edge_team_source_ ||
             neighbor_major_threads <=
                 static_cast<std::int64_t>(YE3T_NEIGHBOR_MAJOR_SOURCE_PADDING_LIMIT) * edge_count);
        if (use_neighbor_major_chunk) {
          if (neighbor_major_league > std::numeric_limits<int>::max())
            error->one(FLERR,
                       "Pair style ye3t/kk neighbor-major source launch "
                       "exceeds 32-bit indexing");
          Kokkos::parallel_for(
              "YE3TBuildOrdinarySourceNeighborMajor",
              Kokkos::TeamPolicy<DeviceType>(static_cast<int>(neighbor_major_league),
                                             YE3T_NEIGHBOR_MAJOR_SOURCE_TEAM_SIZE),
              BuildOrdinarySourceNeighborMajor<DeviceType>{
                  device_plan_.bond_views(), device_plan_.angular_views(), center_counts_,
                  center_offsets_, edge_bonds_, edge_radius_, edge_unit_, source_real_,
                  source_imaginary_, edge_radial_derivative_, edge_contracted_value_,
                  edge_contracted_derivative_, device_status_, center_count, center_capacity_,
                  edge_capacity_, center_tile_count});
        } else {
          BuildOrdinarySourceByEdge<DeviceType> source_functor{device_plan_.bond_views(),
                                                               device_plan_.angular_views(),
                                                               edge_centers_,
                                                               edge_bonds_,
                                                               edge_radius_,
                                                               edge_unit_,
                                                               source_real_,
                                                               source_imaginary_,
                                                               edge_radial_derivative_,
                                                               edge_contracted_value_,
                                                               edge_contracted_derivative_,
                                                               device_status_,
                                                               center_capacity_,
                                                               edge_capacity_,
                                                               source_maximum_radial_base_count_,
                                                               source_maximum_contracted_width_,
                                                               source_maximum_angular_width_,
                                                               source_scratch_value_count_};
          auto source_policy = Kokkos::TeamPolicy<DeviceType>(edge_count, Kokkos::AUTO);
          source_policy = source_policy.set_scratch_size(0, Kokkos::PerTeam(source_scratch_bytes_));
          Kokkos::parallel_for("YE3TBuildOrdinarySourceByEdge", source_policy, source_functor);
        }
      }
    } else {
      Kokkos::parallel_for(
          "YE3TBuildOrdinarySourceReference", Kokkos::RangePolicy<DeviceType>(0, source_iterations),
          BuildOrdinarySource<DeviceType>{
              x, type_, d_ilist_, type_to_species_, device_plan_.species_views(),
              device_plan_.bond_views(), device_plan_.angular_views(), center_offsets_,
              edge_neighbors_, edge_bonds_, source_real_, source_imaginary_, device_status_,
              chunk_begin, center_count, center_capacity_});
    }

    phase.next("YE3T::readout_forward_adjoint");
    auto direct_policy = Kokkos::TeamPolicy<DeviceType>(center_count, Kokkos::AUTO);
    if (direct_resident_values_ > 0)
      direct_policy = direct_policy.set_scratch_size(
          0,
          Kokkos::PerTeam(static_cast<std::size_t>(4) * direct_resident_values_ * sizeof(double)));
    if (!use_block_program_ && !use_scalar_program_ && !use_coupled_program_) {
      Kokkos::parallel_reduce(
          "YE3TDirectTeamForwardAdjoint", direct_policy,
          DirectTeamForwardAdjoint<DeviceType, true>{type_,
                                                     d_ilist_,
                                                     type_to_species_,
                                                     device_plan_.species_views(),
                                                     source_real_,
                                                     source_imaginary_,
                                                     source_adjoint_real_,
                                                     source_adjoint_imaginary_,
                                                     dag_real_,
                                                     dag_imaginary_,
                                                     dag_adjoint_real_,
                                                     dag_adjoint_imaginary_,
                                                     total_density_real_,
                                                     total_density_imaginary_,
                                                     atomic_energies_,
                                                     device_status_,
                                                     chunk_begin,
                                                     center_capacity_,
                                                     device_plan_.maximum_value_count(),
                                                     direct_resident_values_},
          Kokkos::Max<double, typename DeviceType::memory_space>(step_state_.imaginary_chunk));
    } else {
      if (device_plan_.has_direct_residual()) {
        Kokkos::parallel_for(
            "YE3TDirectResidualTeamForwardAdjoint", direct_policy,
            DirectTeamForwardAdjoint<DeviceType, false>{type_,
                                                        d_ilist_,
                                                        type_to_species_,
                                                        device_plan_.species_views(),
                                                        source_real_,
                                                        source_imaginary_,
                                                        source_adjoint_real_,
                                                        source_adjoint_imaginary_,
                                                        dag_real_,
                                                        dag_imaginary_,
                                                        dag_adjoint_real_,
                                                        dag_adjoint_imaginary_,
                                                        total_density_real_,
                                                        total_density_imaginary_,
                                                        atomic_energies_,
                                                        device_status_,
                                                        chunk_begin,
                                                        center_capacity_,
                                                        device_plan_.maximum_value_count(),
                                                        direct_resident_values_});
      } else {
        Kokkos::parallel_for("YE3TInitializeOptimizedState",
                             Kokkos::TeamPolicy<DeviceType>(center_count, Kokkos::AUTO),
                             InitializeOptimizedState<DeviceType>{
                                 type_, d_ilist_, type_to_species_, device_plan_.species_views(),
                                 source_adjoint_real_, source_adjoint_imaginary_,
                                 total_density_real_, total_density_imaginary_, atomic_energies_,
                                 chunk_begin, center_capacity_});
      }
      if (use_block_program_) {
        if (use_work_major_block_) {
          const int team_size = block_work_major_team_size_;
          const int center_tiles = center_count / team_size + (center_count % team_size != 0);
          auto checked_league_size = [&](int cohorts, const char *name) {
            if (cohorts <= 0 || center_tiles <= 0 ||
                cohorts > std::numeric_limits<int>::max() / center_tiles)
              error->all(FLERR, "Pair style ye3t/kk {} league size overflows", name);
            return cohorts * center_tiles;
          };
          const int power_cohorts = std::max(1, device_plan_.maximum_block_power_count());
          const int power_league = checked_league_size(power_cohorts, "block-power");
          const int plan_league =
              checked_league_size(device_plan_.maximum_block_plan_count(), "block-plan");
          const int route_league =
              checked_league_size(device_plan_.maximum_block_route_count(), "block-route");
          Kokkos::parallel_for(
              "YE3TBlockPowerWorkMajor", Kokkos::TeamPolicy<DeviceType>(power_league, team_size),
              BlockPowerWorkMajor<DeviceType>{
                  type_, d_ilist_, type_to_species_, device_plan_.species_views(),
                  device_plan_.block_views(), source_real_, source_imaginary_, block_power_real_,
                  block_power_imaginary_, block_output_real_, block_output_imaginary_,
                  block_output_adjoint_real_, block_output_adjoint_imaginary_, chunk_begin,
                  center_count, center_tiles, center_capacity_});
          auto plan_forward_policy = Kokkos::TeamPolicy<DeviceType>(plan_league, team_size);
          plan_forward_policy = plan_forward_policy.set_scratch_size(
              0, Kokkos::PerTeam(block_work_major_scratch_bytes_));
          Kokkos::parallel_for("YE3TBlockPlanForwardWorkMajor", plan_forward_policy,
                               BlockPlanForwardWorkMajor<DeviceType>{
                                   type_, d_ilist_, type_to_species_, device_plan_.species_views(),
                                   device_plan_.block_views(), source_real_, source_imaginary_,
                                   block_power_real_, block_power_imaginary_, block_output_real_,
                                   block_output_imaginary_, chunk_begin, center_count, center_tiles,
                                   center_capacity_, device_plan_.block_monomial_tile_count()});
          Kokkos::parallel_for("YE3TBlockRouteWorkMajor",
                               Kokkos::TeamPolicy<DeviceType>(route_league, team_size),
                               BlockRouteWorkMajor<DeviceType>{
                                   type_, d_ilist_, type_to_species_, device_plan_.block_views(),
                                   block_output_real_, block_output_imaginary_,
                                   block_output_adjoint_real_, block_output_adjoint_imaginary_,
                                   block_route_density_real_, block_route_density_imaginary_,
                                   chunk_begin, center_count, center_tiles, center_capacity_});
          auto plan_adjoint_policy = Kokkos::TeamPolicy<DeviceType>(plan_league, team_size);
          Kokkos::parallel_for(
              "YE3TBlockPlanAdjointWorkMajor", plan_adjoint_policy,
              BlockPlanAdjointWorkMajor<DeviceType>{
                  type_, d_ilist_, type_to_species_, device_plan_.species_views(),
                  device_plan_.block_views(), block_power_real_, block_power_imaginary_,
                  block_output_adjoint_real_, block_output_adjoint_imaginary_, source_adjoint_real_,
                  source_adjoint_imaginary_, chunk_begin, center_count, center_tiles,
                  center_capacity_, device_plan_.block_monomial_tile_count()});
        } else {
          Kokkos::parallel_for(
              "YE3TBlockFusedCenterLane", Kokkos::RangePolicy<DeviceType>(0, center_count),
              BlockFusedForwardAdjoint<DeviceType>{
                  {type_, d_ilist_, type_to_species_, device_plan_.species_views(),
                   device_plan_.block_views(), source_real_, source_imaginary_, block_power_real_,
                   block_power_imaginary_, block_output_real_, block_output_imaginary_,
                   block_output_adjoint_real_, block_output_adjoint_imaginary_, chunk_begin,
                   center_capacity_},
                  {type_, d_ilist_, type_to_species_, device_plan_.species_views(),
                   device_plan_.block_views(), source_real_, source_imaginary_, block_power_real_,
                   block_power_imaginary_, block_output_real_, block_output_imaginary_,
                   block_monomial_real_, block_monomial_imaginary_, chunk_begin, center_capacity_},
                  {type_, d_ilist_, type_to_species_, device_plan_.block_views(),
                   block_output_real_, block_output_imaginary_, block_output_adjoint_real_,
                   block_output_adjoint_imaginary_, total_density_real_, total_density_imaginary_,
                   chunk_begin, center_capacity_},
                  {type_, d_ilist_, type_to_species_, device_plan_.species_views(),
                   device_plan_.block_views(), block_power_real_, block_power_imaginary_,
                   block_output_real_, block_output_imaginary_, block_output_adjoint_real_,
                   block_output_adjoint_imaginary_, block_monomial_adjoint_real_,
                   block_monomial_adjoint_imaginary_, source_adjoint_real_,
                   source_adjoint_imaginary_, chunk_begin, center_capacity_}});
        }
      }
      if (use_scalar_program_) {
        Kokkos::parallel_for(
            "YE3TScalarPowerForwardAdjoint", Kokkos::RangePolicy<DeviceType>(0, center_count),
            ScalarPowerForwardAdjoint<DeviceType>{
                type_, d_ilist_, type_to_species_, device_plan_.species_views(),
                device_plan_.scalar_views(), source_real_, source_imaginary_, source_adjoint_real_,
                source_adjoint_imaginary_, scalar_value_real_, scalar_value_imaginary_,
                scalar_adjoint_real_, scalar_adjoint_imaginary_, total_density_real_,
                total_density_imaginary_, chunk_begin, center_capacity_});
      }
      if (use_coupled_program_) {
        Kokkos::parallel_for(
            "YE3TCoupledProductForwardAdjoint", Kokkos::RangePolicy<DeviceType>(0, center_count),
            CoupledProductForwardAdjoint<DeviceType>{
                type_, d_ilist_, type_to_species_, device_plan_.species_views(),
                device_plan_.coupled_views(), source_real_, source_imaginary_, source_adjoint_real_,
                source_adjoint_imaginary_, coupled_value_real_, coupled_value_imaginary_,
                coupled_adjoint_real_, coupled_adjoint_imaginary_, total_density_real_,
                total_density_imaginary_, chunk_begin, center_capacity_});
      }
      Kokkos::parallel_reduce(
          "YE3TFinalizeTotalDensity", Kokkos::RangePolicy<DeviceType>(0, center_count),
          FinalizeTotalDensity<DeviceType>{
              type_, d_ilist_, type_to_species_, device_plan_.species_views(),
              device_plan_.block_views(), total_density_real_, total_density_imaginary_,
              block_route_density_real_, block_route_density_imaginary_, atomic_energies_,
              device_status_, chunk_begin, center_capacity_,
              use_block_program_ && use_work_major_block_},
          Kokkos::Max<double, typename DeviceType::memory_space>(step_state_.imaginary_chunk));
    }

    phase.next("YE3T::source_pullback");
    if (edge_count > 0) {
      if (use_neighbor_major_source_ || use_edge_team_source_) {
        Kokkos::parallel_for(
            "YE3TRadialCachedSourceVJP", Kokkos::RangePolicy<DeviceType>(0, edge_count),
            RadialCachedSourceVJP<DeviceType>{
                device_plan_.bond_views(), device_plan_.angular_views(), edge_centers_, edge_bonds_,
                edge_radius_, edge_unit_, edge_radial_derivative_, edge_contracted_value_,
                edge_contracted_derivative_, source_adjoint_real_, source_adjoint_imaginary_,
                edge_gradient_, device_status_, center_capacity_, edge_capacity_});
      } else {
        Kokkos::parallel_for(
            "YE3TSourceVJPReference", Kokkos::RangePolicy<DeviceType>(0, edge_count),
            SourceVJP<DeviceType>{device_plan_.bond_views(), device_plan_.angular_views(),
                                  edge_centers_, edge_bonds_, edge_radius_, edge_unit_,
                                  source_adjoint_real_, source_adjoint_imaginary_, edge_gradient_,
                                  device_status_, center_capacity_, edge_capacity_});
      }
    }

    phase.next("YE3T::force_virial_scatter");
    auto run_scatter = [&](auto neighbor_tag, auto virial_tag) {
      constexpr int neighbor_mode = decltype(neighbor_tag)::value;
      constexpr bool tally_atom_virial = decltype(virial_tag)::value;
      using Functor = ScatterForceEnergyVirial<DeviceType, neighbor_mode, tally_atom_virial>;
      const Functor functor{x,
                            d_ilist_,
                            center_offsets_,
                            edge_neighbors_,
                            edge_gradient_,
                            atomic_energies_,
                            d_eatom_,
                            d_vatom_,
                            duplicated_force_,
                            atomic_force_,
                            duplicated_vatom_,
                            atomic_vatom_,
                            chunk_begin,
                            edge_capacity_,
                            eflag_global,
                            eflag_atom,
                            vflag_global,
                            vflag_either};
      if (eflag_global || vflag_global) {
        Kokkos::parallel_reduce("YE3TScatterForceEnergyVirial",
                                Kokkos::RangePolicy<DeviceType>(0, center_count), functor,
                                step_state_.ev_chunk);
      } else {
        Kokkos::parallel_for("YE3TScatterForceOnly",
                             Kokkos::RangePolicy<DeviceType>(0, center_count), functor);
      }
    };

    if (neighflag_ == HALF) {
      if (vflag_atom)
        run_scatter(std::integral_constant<int, HALF>{}, std::true_type{});
      else
        run_scatter(std::integral_constant<int, HALF>{}, std::false_type{});
    } else {
      if (vflag_atom)
        run_scatter(std::integral_constant<int, HALFTHREAD>{}, std::true_type{});
      else
        run_scatter(std::integral_constant<int, HALFTHREAD>{}, std::false_type{});
    }
    step_state_.accumulate(eflag_global || vflag_global, true);
    chunk_begin += center_count;
  }

  // One fence per step: tallies and the deferred status word together.
  int deferred_status = 0;
  const auto step_totals = step_state_.read_totals(device_status_, deferred_status);
  report_device_status(deferred_status);
  total_ev = step_totals.ev;

  if (need_dup_) Kokkos::Experimental::contribute(f, duplicated_force_);
  if (eflag_global) eng_vdwl += static_cast<double>(total_ev.evdwl);
  if (vflag_global)
    for (int component = 0; component < 6; ++component)
      virial[component] += static_cast<double>(total_ev.v[component]);
  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  if (eflag_atom) {
    k_eatom_.template modify<DeviceType>();
    k_eatom_.sync_host();
  }
  if (vflag_atom) {
    if (need_dup_) Kokkos::Experimental::contribute(d_vatom_, duplicated_vatom_);
    k_vatom_.template modify<DeviceType>();
    k_vatom_.sync_host();
  }
  atomKK->modified(execution_space, F_MASK);
  set_maximum_imaginary_density(step_totals.maximum_imaginary);

  if (!capacity_state_reported_) {
    recorded_chunk_limit_ = smallest_effective_chunk;
    report_initial_capacity_state(inum, requested_chunk, smallest_effective_chunk, center_reduced,
                                  edge_reductions, reallocations_before);
  }

  duplicated_force_ = {};
  atomic_force_ = {};
  duplicated_vatom_ = {};
  atomic_vatom_ = {};
}

template <class DeviceType> double PairYE3TKokkos<DeviceType>::memory_usage()
{
  const std::size_t plan_bytes = is_lifted_cauchy_model()
      ? lifted_device_plan_.memory_usage()
      : (is_tagged_cauchy_model() ? tagged_device_plan_.memory_usage()
                                  : device_plan_.memory_usage());
  return PairYE3T::memory_usage() +
      static_cast<double>(plan_bytes +
                          (type_to_species_.extent(0) * sizeof(int) + step_state_.memory_usage()) +
                          dynamic_bytes_);
}

template class PairYE3TKokkos<LMPDeviceType>;
