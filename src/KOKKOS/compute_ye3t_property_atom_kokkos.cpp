/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   Contributing author: James M. Goff (Sandia National Laboratories)
------------------------------------------------------------------------- */

#include "compute_ye3t_property_atom_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "neighbor_kokkos.h"
#include "update.h"

#include <cmath>
#include <exception>
#include <type_traits>

using namespace LAMMPS_NS;

namespace {

struct PropertyKokkosMath {
  using Complex = Kokkos::complex<double>;
  KOKKOS_INLINE_FUNCTION static double sqrt(double x) { return Kokkos::sqrt(x); }
};

template <class DeviceType>
KOKKOS_INLINE_FUNCTION void edge_components(
    const YE3T_LAMMPS::MeanPropertyKokkosViews<DeviceType> &plan,
    int center_species, int neighbor_species, double dx, double dy, double dz, double *phi)
{
  for (int component = 0; component < plan.component_count; ++component) phi[component] = 0.0;
  const double radius = Kokkos::sqrt(dx * dx + dy * dy + dz * dz);
  const double cutoff = plan.pair_cutoffs(center_species * plan.species_count + neighbor_species);
  if (radius <= 0.0 || radius >= cutoff) return;
  const double x = radius / cutoff;
  for (int channel = 0; channel < static_cast<int>(plan.channel_species.extent(0)); ++channel) {
    if (plan.channel_species(channel) != neighbor_species) continue;
    const int l = plan.channel_l(channel);
    const int width = 2 * l + 1;
    PropertyKokkosMath::Complex y[2 * YE3T_LAMMPS::TAGGED_KOKKOS_MAX_L + 1];
    YE3T_LAMMPS::gpu_tagged_complex_ylm<false, PropertyKokkosMath>(
        l, dx, dy, dz,
        plan.legendre_derivative.data() +
            channel * YE3T_LAMMPS::TAGGED_KOKKOS_LEGENDRE_M_STRIDE *
                YE3T_LAMMPS::TAGGED_KOKKOS_LEGENDRE_STRIDE,
        plan.harmonic_normalization.data() +
            channel * YE3T_LAMMPS::TAGGED_KOKKOS_LEGENDRE_M_STRIDE,
        y, nullptr, nullptr, nullptr);
    double polynomial = 0.0;
    for (int power = plan.jacobi_offsets(channel + 1) - 1;
         power >= plan.jacobi_offsets(channel); --power)
      polynomial = polynomial * x + plan.jacobi_coefficients(power);
    double x_l = 1.0;
    for (int power = 0; power < l; ++power) x_l *= x;
    const double envelope = 1.0 - x;
    const double radial = plan.channel_normalization(channel) * x_l *
                          envelope * envelope * polynomial *
                          Kokkos::sqrt(4.0 * 3.14159265358979323846 / width);
    const int offset = plan.channel_offset(channel);
    phi[offset + l] = radial * y[l].real();
    for (int m = 1; m <= l; ++m) {
      const double sign = m % 2 ? -1.0 : 1.0;
      const double scale = 1.4142135623730950488 * sign * radial;
      phi[offset + l - m] = scale * y[l + m].real();
      phi[offset + l + m] = -scale * y[l + m].imag();
    }
  }
}

template <class DeviceType, class Density, class Output>
KOKKOS_INLINE_FUNCTION void evaluate_schedule(
    const YE3T_LAMMPS::MeanPropertyKokkosViews<DeviceType> &plan,
    const Density &density, const Output &output, int center, int central_species,
    int schedule, const double *phi)
{
  const int support = plan.schedule_support(schedule);
  for (int coefficient = plan.schedule_coefficient_offsets(schedule);
       coefficient < plan.schedule_coefficient_offsets(schedule + 1); ++coefficient) {
    const int term = plan.coefficient_term(coefficient);
    double value = plan.coefficient_value(coefficient);
    for (int factor = plan.term_factor_offsets(term);
         factor < plan.term_factor_offsets(term + 1); ++factor) {
      const int input = plan.term_input(factor);
      const int component = plan.input_component(input);
      const double source = plan.input_role(input) == support
          ? density(component, center) : phi[component];
      for (int power = 0; power < plan.term_exponent(factor); ++power) value *= source;
    }
    const int output_index = plan.coefficient_output(coefficient);
    for (int feature = 0; feature < plan.feature_count; ++feature) {
      if (plan.feature_schedule(feature) != schedule) continue;
      const int local = output_index - plan.feature_start(feature);
      if (local >= 0 && local < plan.width)
        output(center, local) += value *
            plan.readout(central_species * plan.feature_count + feature);
    }
  }
}

}  // namespace

template <class DeviceType>
ComputeYE3TPropertyAtomKokkos<DeviceType>::ComputeYE3TPropertyAtomKokkos(
    LAMMPS *lmp, int narg, char **arg) : ComputeYE3TPropertyAtom(lmp, narg, arg)
{
  kokkosable = 1;
  atomKK = static_cast<AtomKokkos *>(atom);
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;
  try {
    device_plan_.upload(property_model());
  } catch (const std::exception &exception) {
    error->all(FLERR, "Cannot upload YE3T property device plan: {}", exception.what());
  }
  const auto &mapping = property_type_map();
  type_to_species_ = IntArray("ye3t:property_type_map", mapping.size());
  auto host = Kokkos::create_mirror_view(type_to_species_);
  for (std::size_t index = 0; index < mapping.size(); ++index) host(index) = mapping[index];
  Kokkos::deep_copy(type_to_species_, host);
}

template <class DeviceType>
ComputeYE3TPropertyAtomKokkos<DeviceType>::~ComputeYE3TPropertyAtomKokkos()
{
  memory->destroy(host_output_);
}

template <class DeviceType> void ComputeYE3TPropertyAtomKokkos<DeviceType>::init()
{
  ComputeYE3TPropertyAtom::init();
  if (execution_space != Device)
    error->all(FLERR, "Compute ye3t/property/atom/kk requires a Kokkos device execution space");
  auto *request = neighbor->find_request(this);
  request->set_kokkos_host(false);
  request->set_kokkos_device(true);
}

template <class DeviceType>
void ComputeYE3TPropertyAtomKokkos<DeviceType>::compute_peratom()
{
  invoked_peratom = update->ntimestep;
  if (atom->nmax > device_nmax_) {
    memory->destroy(host_output_);
    device_nmax_ = atom->nmax;
    memory->create(host_output_, device_nmax_, size_peratom_cols,
                   "ye3t/property/atom/kk:host_output");
    array_atom = host_output_;
    output_ = RealArray("ye3t:property_output", device_nmax_, size_peratom_cols);
    density_ = RealScratch("ye3t:property_density", property_model().component_count(),
                           device_nmax_);
  }
  auto *list = static_cast<NeighListKokkos<DeviceType> *>(property_neighbor_list());
  if (!list) error->all(FLERR, "Compute ye3t/property/atom/kk has no device neighbor list");
  neighbor->build_one(list);
  atomKK->sync(execution_space, X_MASK | TYPE_MASK | MASK_MASK);
  auto positions = atomKK->k_x.template view<DeviceType>();
  auto types = atomKK->k_type.template view<DeviceType>();
  auto masks = atomKK->k_mask.template view<DeviceType>();
  const auto neighbors = list->d_neighbors;
  const auto ilist = list->d_ilist;
  const auto numneigh = list->d_numneigh;
  const auto type_map = type_to_species_;
  const auto plan = device_plan_.views();
  const auto density = density_;
  const auto output = output_;
  const int group = groupbit;
  const int nlocal = atom->nlocal;
  Kokkos::deep_copy(output, 0.0);
  Kokkos::deep_copy(density, 0.0);
  Kokkos::parallel_for("YE3T::property::density",
                       Kokkos::RangePolicy<DeviceType>(0, list->inum),
                       KOKKOS_LAMBDA(const int ii) {
    const int i = ilist(ii);
    if (i >= nlocal || !(masks(i) & group)) return;
    const int central = type_map(types(i));
    if (central < 0) return;
    for (int jj = 0; jj < numneigh(i); ++jj) {
      const int j = neighbors(i, jj) & NEIGHMASK;
      const int species = type_map(types(j));
      if (species < 0) continue;
      const double dx = positions(j, 0) - positions(i, 0);
      const double dy = positions(j, 1) - positions(i, 1);
      const double dz = positions(j, 2) - positions(i, 2);
      if (dx * dx + dy * dy + dz * dz >= plan.cutoff * plan.cutoff) continue;
      double phi[YE3T_LAMMPS::TAGGED_KOKKOS_MAX_COMPONENTS];
      edge_components(plan, central, species, dx, dy, dz, phi);
      for (int component = 0; component < plan.component_count; ++component)
        density(component, i) += phi[component];
    }
  });
  Kokkos::parallel_for("YE3T::property::readout",
                       Kokkos::RangePolicy<DeviceType>(0, list->inum),
                       KOKKOS_LAMBDA(const int ii) {
    const int i = ilist(ii);
    if (i >= nlocal || !(masks(i) & group)) return;
    const int central = type_map(types(i));
    if (central < 0) return;
    double phi[YE3T_LAMMPS::TAGGED_KOKKOS_MAX_COMPONENTS];
    for (int schedule = 0; schedule < plan.schedule_count; ++schedule)
      if (plan.schedule_support(schedule) == 0)
        evaluate_schedule(plan, density, output, i, central, schedule, phi);
    for (int jj = 0; jj < numneigh(i); ++jj) {
      const int j = neighbors(i, jj) & NEIGHMASK;
      const int species = type_map(types(j));
      if (species < 0) continue;
      const double dx = positions(j, 0) - positions(i, 0);
      const double dy = positions(j, 1) - positions(i, 1);
      const double dz = positions(j, 2) - positions(i, 2);
      if (dx * dx + dy * dy + dz * dz >= plan.cutoff * plan.cutoff) continue;
      edge_components(plan, central, species, dx, dy, dz, phi);
      for (int schedule = 0; schedule < plan.schedule_count; ++schedule)
        if (plan.schedule_support(schedule) != 0)
          evaluate_schedule(plan, density, output, i, central, schedule, phi);
    }
  });
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output);
  for (int i = 0; i < nlocal; ++i)
    for (int component = 0; component < size_peratom_cols; ++component)
      host_output_[i][component] = host(i, component);
}

template <class DeviceType>
double ComputeYE3TPropertyAtomKokkos<DeviceType>::memory_usage()
{
  return static_cast<double>(device_nmax_) *
         (2 * size_peratom_cols + property_model().component_count()) * sizeof(double);
}

template class ComputeYE3TPropertyAtomKokkos<LMPDeviceType>;
