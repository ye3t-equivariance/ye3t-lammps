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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(ye3t/kk,PairYE3TKokkos<LMPDeviceType>);
PairStyle(ye3t/kk/device,PairYE3TKokkos<LMPDeviceType>);
// clang-format on
#else

#ifndef LMP_PAIR_YE3T_KOKKOS_H
#define LMP_PAIR_YE3T_KOKKOS_H

#include "pair_ye3t.h"

#include "kokkos_type.h"
#include "pair_kokkos.h"
#include "ye3t_kokkos_plan.h"
#include "ye3t_kokkos_step_state.h"
#include "ye3t_kokkos_types.h"
#include "ye3t_lifted_cauchy_kokkos_plan.h"
#include "ye3t_tagged_cauchy_kokkos_plan.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace LAMMPS_NS {

template <class DeviceType> class PairYE3TKokkos : public PairYE3T {
 public:
  using device_type = DeviceType;
  using AT = ArrayTypes<DeviceType>;
  using IntView = Kokkos::View<int *, DeviceType>;
  using OffsetView = Kokkos::View<std::int64_t *, DeviceType>;
  using RealView = Kokkos::View<double *, DeviceType>;
  using KKDeviceType = typename KKDevice<DeviceType>::value;
  using YE3TArrays = ye3t_kokkos::ArrayAliases<DeviceType>;
  using DupForceView = KKScatterView<ye3t_kokkos::force_float *[3],
                                     typename ye3t_kokkos::t_forces_device::array_layout,
                                     KKDeviceType, KKScatterSum, KKScatterDuplicated>;
  using NonDupForceView = KKScatterView<ye3t_kokkos::force_float *[3],
                                        typename ye3t_kokkos::t_forces_device::array_layout,
                                        KKDeviceType, KKScatterSum, KKScatterNonDuplicated>;
  using DupVirialView = KKScatterView<ye3t_kokkos::force_float *[6],
                                      typename ye3t_kokkos::t_vatom_device::array_layout,
                                      KKDeviceType, KKScatterSum, KKScatterDuplicated>;
  using NonDupVirialView = KKScatterView<ye3t_kokkos::force_float *[6],
                                         typename ye3t_kokkos::t_vatom_device::array_layout,
                                         KKDeviceType, KKScatterSum, KKScatterNonDuplicated>;

  explicit PairYE3TKokkos(LAMMPS *);
  ~PairYE3TKokkos() override;

  void compute(int, int) override;
  void init_style() override;
  double memory_usage() override;

 protected:
  void initialize_backend() override;
  void initialize_lifted_backend() override;
  YE3T_LAMMPS::YACEBlockPolicy model_load_policy() const override;
  bool supports_lifted_cauchy() const override { return true; }
  bool supports_tagged_cauchy() const override { return true; }
  void initialize_tagged_backend() override;

 private:
  friend void pair_virial_fdotr_compute<PairYE3TKokkos>(PairYE3TKokkos *);

  std::pair<std::size_t, std::size_t> device_memory_info() const;
  std::size_t allocation_budget() const;
  std::size_t center_storage_bytes(int) const;
  std::size_t edge_storage_bytes(int) const;
  bool allocation_preflight(std::size_t) const;
  int ensure_center_capacity(int);
  int ensure_lifted_center_capacity(int);
  int ensure_tagged_center_capacity(int);
  bool ensure_edge_capacity(int);
  int copy_device_status() const;
  void report_device_status(int) const;
  void initialize_rank_device_identity();
  void report_initial_capacity_state(int, int, int, int, int, int);
  void compute_lifted(int, int);
  void compute_tagged(int, int);

  YE3T_LAMMPS::YE3TKokkosDevicePlan<DeviceType> device_plan_;
  YE3T_LAMMPS::LiftedCauchyKokkosPlan<DeviceType> lifted_device_plan_;
  YE3T_LAMMPS::TaggedCauchyKokkosPlan<DeviceType> tagged_device_plan_;
  std::string flat_plan_hash_;
  std::string device_plan_hash_;
  std::string rank_device_map_hash_;
  std::string device_class_hash_;

  IntView type_to_species_;
  OffsetView center_counts_;
  OffsetView center_offsets_;
  IntView edge_centers_;
  IntView edge_neighbors_;
  IntView edge_bonds_;
  RealView edge_radius_;
  RealView edge_unit_;
  RealView edge_radial_derivative_;
  RealView edge_contracted_value_;
  RealView edge_contracted_derivative_;
  RealView source_real_;
  RealView source_imaginary_;
  RealView source_adjoint_real_;
  RealView source_adjoint_imaginary_;
  RealView lifted_workspace_;
  RealView dag_real_;
  RealView dag_imaginary_;
  RealView dag_adjoint_real_;
  RealView dag_adjoint_imaginary_;
  RealView total_density_real_;
  RealView total_density_imaginary_;
  RealView block_power_real_;
  RealView block_power_imaginary_;
  RealView block_output_real_;
  RealView block_output_imaginary_;
  RealView block_output_adjoint_real_;
  RealView block_output_adjoint_imaginary_;
  RealView block_monomial_real_;
  RealView block_monomial_imaginary_;
  RealView block_monomial_adjoint_real_;
  RealView block_monomial_adjoint_imaginary_;
  RealView block_route_density_real_;
  RealView block_route_density_imaginary_;
  RealView scalar_value_real_;
  RealView scalar_value_imaginary_;
  RealView scalar_adjoint_real_;
  RealView scalar_adjoint_imaginary_;
  RealView coupled_value_real_;
  RealView coupled_value_imaginary_;
  RealView coupled_adjoint_real_;
  RealView coupled_adjoint_imaginary_;
  RealView atomic_energies_;
  RealView edge_gradient_;
  IntView device_status_;
  YE3T_LAMMPS::KokkosStepState<DeviceType, EV_FLOAT> step_state_;

  // Tagged-Cauchy per-center scratch, "structure of arrays" over
  // center_capacity_ exactly like lifted_workspace_ (key index major, lane
  // == center-within-chunk minor). Per-edge phi_real itself is never
  // stored here -- it is recomputed into kernel-local registers within a
  // single edge iteration (see TaggedBuildDensityMoment/TaggedEdgeVJP) --
  // only these persist across the three compute_tagged() kernel launches
  // for one chunk.
  RealView tagged_density_;
  RealView tagged_moment_;
  RealView tagged_density_adjoint_;
  RealView tagged_moment_adjoint_;

  typename AT::t_neighbors_2d d_neighbors_;
  typename AT::t_int_1d_randomread d_ilist_;
  typename AT::t_int_1d_randomread d_numneigh_;
  typename YE3TArrays::t_positions_randomread x;
  typename YE3TArrays::t_forces f;
  typename AT::t_int_1d_randomread type_;
  ye3t_kokkos::tdual_eatom k_eatom_;
  ye3t_kokkos::tdual_vatom k_vatom_;
  typename YE3TArrays::t_eatom d_eatom_;
  typename YE3TArrays::t_vatom d_vatom_;
  DupForceView duplicated_force_;
  NonDupForceView atomic_force_;
  DupVirialView duplicated_vatom_;
  NonDupVirialView atomic_vatom_;

  int direct_resident_values_ = 0;
  int center_capacity_ = 0;
  int edge_capacity_ = 0;
  int reallocation_count_ = 0;
  int neighflag_ = 0;
  int need_dup_ = 0;
  int source_scratch_value_count_ = 0;
  int source_maximum_radial_base_count_ = 0;
  int source_maximum_contracted_width_ = 0;
  int source_maximum_angular_width_ = 0;
  int source_serial_work_count_ = 0;
  int edge_basis_cache_value_count_ = 0;
  int block_work_major_team_size_ = 0;
  std::size_t block_work_major_scratch_bytes_ = 0;
  std::size_t source_scratch_bytes_ = 0;
  std::size_t dynamic_bytes_ = 0;
  int recorded_chunk_limit_ = 0;
  bool use_neighbor_major_source_ = false;
  bool use_edge_team_source_ = false;
  bool use_block_program_ = false;
  bool use_work_major_block_ = false;
  bool use_scalar_program_ = false;
  bool use_coupled_program_ = false;
  bool capacity_state_reported_ = false;
  bool replay_workload_validated_ = false;
  bool require_exact_chunksize_ = false;
  bool device_plan_ready_ = false;
  bool lifted_device_plan_ready_ = false;
  bool tagged_device_plan_ready_ = false;
};

}    // namespace LAMMPS_NS

#endif
#endif
