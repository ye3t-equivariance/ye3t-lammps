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

#ifndef LMP_YE3T_MEAN_PROPERTY_CPU_H
#define LMP_YE3T_MEAN_PROPERTY_CPU_H

#include <cstddef>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

namespace YE3T_LAMMPS {

// Forward-only evaluator for compiler-bound tagged or density full-M v2 artifacts.
// Edges are directed periodic occurrences in center-CSR order. Neighbor
// species indices are in species_order(), and a negative index means NULL.
// No pair energy, force, adjoint, or posterior covariance is calculated.
class YE3TMeanPropertyCPU {
 public:
  explicit YE3TMeanPropertyCPU(const std::string &path);

  int width() const { return width_; }
  double cutoff() const { return cutoff_; }
  const std::vector<std::string> &species_order() const { return species_order_; }

  // Validated, compiler-bound arrays for the device materialization. The
  // Kokkos evaluator uploads these records; it never reparses the artifact.
  struct Channel {
    int neighbor_species = -1;
    int radial_channel = 0;
    int l = 0;
    int offset = 0;
    double normalization = 1.0;
    std::vector<double> jacobi_coefficients;
  };
  struct Input {
    int component = 0;
    int role = 0;
  };
  struct Schedule {
    int tag_count = 0;
    int support_tag_count = 0;
    int output_dimension = 0;
    std::vector<Input> inputs;
    std::vector<int> term_offsets;
    std::vector<int> term_components;
    std::vector<int> term_exponents;
    std::vector<int> coefficient_terms;
    std::vector<int> coefficient_outputs;
    std::vector<double> coefficient_values;
  };
  struct Feature {
    int schedule_index = 0;
    int component_start = 0;
  };

  int component_count() const { return component_count_; }
  const std::vector<double> &pair_cutoffs() const { return pair_cutoffs_; }
  const std::vector<Channel> &channels() const { return channels_; }
  const std::vector<Schedule> &schedules() const { return schedules_; }
  const std::vector<Feature> &features() const { return features_; }
  const std::vector<double> &readout() const { return readout_; }
  bool tagged_model() const { return !density_model_; }

  void evaluate(int center_count, const int *central_species,
                const std::size_t *edge_offsets, const int *neighbor_species,
                const double *edge_vectors, const unsigned char *active_centers,
                double *output) const;

 private:
  struct DensityFactor {
    int n = 0;
    int l = 0;
    int mu = 0;
  };
  struct DensityTerm {
    std::vector<int> magnetic;
    std::complex<double> coefficient;
  };
  struct DensityFeature {
    int center_species = 0;
    std::vector<DensityFactor> factors;
    std::vector<DensityTerm> terms;
  };
  struct DensityBond {
    double cutoff = 0.0;
    std::int64_t intervals = 0;
    std::vector<double> spline;
  };

  bool density_model_ = false;
  int density_radial_count_ = 0;
  std::vector<std::vector<DensityFeature>> density_blocks_;
  std::vector<std::complex<double>> density_real_to_complex_;
  std::vector<double> density_beta_;
  std::vector<DensityBond> density_bonds_;
  std::vector<std::string> species_order_;
  std::vector<double> pair_cutoffs_;
  std::vector<Channel> channels_;
  std::vector<Schedule> schedules_;
  std::vector<Feature> features_;
  std::vector<double> readout_;
  double cutoff_ = 0.0;
  int component_count_ = 0;
  int width_ = 0;
};

}  // namespace YE3T_LAMMPS

#endif  // LMP_YE3T_MEAN_PROPERTY_CPU_H
