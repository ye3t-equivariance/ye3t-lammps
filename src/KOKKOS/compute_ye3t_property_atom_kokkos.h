/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   Contributing author: James M. Goff (Sandia National Laboratories)
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(ye3t/property/atom/kk,ComputeYE3TPropertyAtomKokkos<LMPDeviceType>);
ComputeStyle(ye3t/property/atom/kk/device,ComputeYE3TPropertyAtomKokkos<LMPDeviceType>);
// clang-format on
#else

#ifndef LMP_COMPUTE_YE3T_PROPERTY_ATOM_KOKKOS_H
#define LMP_COMPUTE_YE3T_PROPERTY_ATOM_KOKKOS_H

#include "compute_ye3t_property_atom.h"
#include "kokkos_type.h"
#include "ye3t_mean_property_kokkos_plan.h"

namespace LAMMPS_NS {

template <class DeviceType>
class ComputeYE3TPropertyAtomKokkos : public ComputeYE3TPropertyAtom {
 public:
  using AT = ArrayTypes<DeviceType>;
  using RealArray = Kokkos::View<double **, Kokkos::LayoutRight, DeviceType>;
  using RealScratch = Kokkos::View<double **, Kokkos::LayoutRight, DeviceType>;
  using IntArray = Kokkos::View<int *, DeviceType>;

  ComputeYE3TPropertyAtomKokkos(LAMMPS *, int, char **);
  ~ComputeYE3TPropertyAtomKokkos() override;
  void init() override;
  void compute_peratom() override;
  double memory_usage() override;

 private:
  YE3T_LAMMPS::MeanPropertyKokkosPlan<DeviceType> device_plan_;
  IntArray type_to_species_;
  RealArray output_;
  RealScratch density_;
  double **host_output_ = nullptr;
  int device_nmax_ = 0;
};

}  // namespace LAMMPS_NS

#endif
#endif
