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

/* ----------------------------------------------------------------------
   Kokkos precision/array type names used by pair_style ye3t/kk.

   LAMMPS renamed its Kokkos precision typedefs after the 22 Jul 2025 stable
   series: the legacy X_FLOAT / F_FLOAT / E_FLOAT scalars and the
   t_x_array / t_f_array / t_efloat_1d / t_virial_array views became the
   KK_FLOAT / KK_ACC_FLOAT scalars and the t_kkfloat_* / t_kkacc_* views, and
   the precision selector moved from LMP_PRECISION to the
   LMP_KOKKOS_{DOUBLE_DOUBLE,SINGLE_DOUBLE,SINGLE_SINGLE} macros.

   This header maps both spellings onto one set of names. On the stable
   series every alias is the identical legacy type, so the compiled kernels
   are unchanged; on development revisions the aliases resolve to the
   renamed types with the same element layout (positions read-only in
   LayoutRight, forces and per-atom energy/virial in the accumulator type).
------------------------------------------------------------------------- */

#ifndef LMP_YE3T_KOKKOS_TYPES_H
#define LMP_YE3T_KOKKOS_TYPES_H

#include "kokkos_type.h"

namespace LAMMPS_NS {
namespace ye3t_kokkos {

#if defined(LMP_KOKKOS_DOUBLE_DOUBLE) || defined(LMP_KOKKOS_SINGLE_DOUBLE) || \
    defined(LMP_KOKKOS_SINGLE_SINGLE)

  typedef KK_FLOAT position_float;
  typedef KK_ACC_FLOAT force_float;
  typedef KK_ACC_FLOAT energy_float;

  typedef DAT::ttransform_kkacc_1d tdual_eatom;
  typedef DAT::ttransform_kkacc_1d_6 tdual_vatom;
  typedef DAT::t_kkacc_1d_3 t_forces_device;
  typedef DAT::t_kkacc_1d_6 t_vatom_device;

  template <class DeviceType> struct ArrayAliases {
    typedef ArrayTypes<DeviceType> AT;
    typedef typename AT::t_kkfloat_1d_3_lr_randomread t_positions_randomread;
    typedef typename AT::t_kkacc_1d_3 t_forces;
    typedef typename AT::t_kkacc_1d t_eatom;
    typedef typename AT::t_kkacc_1d_6 t_vatom;
  };

#else

  typedef X_FLOAT position_float;
  typedef F_FLOAT force_float;
  typedef E_FLOAT energy_float;

  typedef DAT::tdual_efloat_1d tdual_eatom;
  typedef DAT::tdual_virial_array tdual_vatom;
  typedef DAT::t_f_array t_forces_device;
  typedef DAT::t_virial_array t_vatom_device;

  template <class DeviceType> struct ArrayAliases {
    typedef ArrayTypes<DeviceType> AT;
    typedef typename AT::t_x_array_randomread t_positions_randomread;
    typedef typename AT::t_f_array t_forces;
    typedef typename AT::t_efloat_1d t_eatom;
    typedef typename AT::t_virial_array t_vatom;
  };

#endif

}    // namespace ye3t_kokkos
}    // namespace LAMMPS_NS

#endif
