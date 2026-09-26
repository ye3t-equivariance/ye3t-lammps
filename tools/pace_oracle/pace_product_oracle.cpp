// SPDX-License-Identifier: GPL-2.0-or-later

#include "ace-evaluator/ace_evaluator.h"
#include "ace-evaluator/ace_radial.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef EXTRA_C_PROJECTIONS
#error "The PACE oracle requires EXTRA_C_PROJECTIONS"
#endif

#ifndef COMPUTE_B_GRAD
#error "The PACE oracle requires COMPUTE_B_GRAD"
#endif

#ifndef ML_YE3T_PACE_REVISION
#define ML_YE3T_PACE_REVISION "unrecorded"
#endif

namespace {

struct Environment {
  int central = -1;
  std::vector<std::array<double, 3>> positions;
  std::vector<SPECIES_TYPE> types;
  std::vector<int> neighbours;
};

class PACEProductOracleEvaluator : public ACECTildeEvaluator {
public:
  const Array2D<DOUBLE_TYPE> &rank_one_atomic_base() const { return A_rank1; }
  const Array4DLM<ACEComplex> &atomic_base() const { return A; }
};

Environment read_environment(const std::string &path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("Could not open environment: " + path);

  std::string token;
  input >> token;
  if (token != "pace_environment_v1") {
    throw std::runtime_error("Unsupported environment schema: " + token);
  }

  Environment environment;
  input >> token >> environment.central;
  if (token != "central") throw std::runtime_error("Expected central record");

  std::size_t atom_count = 0;
  input >> token >> atom_count;
  if (token != "atoms" || atom_count == 0) {
    throw std::runtime_error("Expected a nonempty atoms record");
  }
  environment.positions.resize(atom_count);
  environment.types.resize(atom_count);
  for (std::size_t atom = 0; atom < atom_count; ++atom) {
    int type = -1;
    input >> type >> environment.positions[atom][0] >> environment.positions[atom][1] >>
        environment.positions[atom][2];
    if (!input || type < 0) throw std::runtime_error("Invalid atom record");
    for (double coordinate : environment.positions[atom]) {
      if (!std::isfinite(coordinate)) throw std::runtime_error("Non-finite atom coordinate");
    }
    environment.types[atom] = static_cast<SPECIES_TYPE>(type);
  }

  std::size_t neighbour_count = 0;
  input >> token >> neighbour_count;
  if (token != "neighbors" || neighbour_count == 0) {
    throw std::runtime_error("Expected a nonempty neighbors record");
  }
  environment.neighbours.resize(neighbour_count);
  for (std::size_t edge = 0; edge < neighbour_count; ++edge) {
    input >> environment.neighbours[edge];
  }
  if (!input) throw std::runtime_error("Invalid neighbor list");
  if (environment.central < 0 ||
      static_cast<std::size_t>(environment.central) >= atom_count) {
    throw std::runtime_error("Central atom index is out of range");
  }
  for (int neighbour : environment.neighbours) {
    if (neighbour < 0 || static_cast<std::size_t>(neighbour) >= atom_count ||
        neighbour == environment.central) {
      throw std::runtime_error("Neighbor atom index is invalid");
    }
  }
  std::vector<int> unique_neighbours = environment.neighbours;
  std::sort(unique_neighbours.begin(), unique_neighbours.end());
  if (std::adjacent_find(unique_neighbours.begin(), unique_neighbours.end()) !=
      unique_neighbours.end()) {
    throw std::runtime_error("Neighbor atom indices must be unique");
  }
  if (input >> token) throw std::runtime_error("Unexpected trailing environment record");
  return environment;
}

double require_finite(double value, const char *field) {
  if (!std::isfinite(value)) throw std::runtime_error(std::string("Non-finite ") + field);
  return value;
}

void write_complex(std::ostream &output, const ACEComplex &value) {
  require_finite(value.real, "complex real component");
  require_finite(value.img, "complex imaginary component");
  output << '\t' << value.real << '\t' << value.img;
}

double norm(const std::array<double, 3> &value) {
  return std::sqrt(value[0] * value[0] + value[1] * value[1] + value[2] * value[2]);
}

void write_input(std::ostream &output, const ACECTildeBasisSet &basis,
                 const Environment &environment) {
  output << "meta\tschema\tpace_product_oracle_tsv_v1\n";
  output << "meta\tevaluator\tACECTildeEvaluator\n";
  output << "meta\tpace_revision\t" << ML_YE3T_PACE_REVISION << '\n';
  output << "meta\tnelements\t" << static_cast<int>(basis.nelements) << '\n';
  output << "meta\tnradbase\t" << static_cast<int>(basis.nradbase) << '\n';
  output << "meta\tnradmax\t" << static_cast<int>(basis.nradmax) << '\n';
  output << "meta\tlmax\t" << static_cast<int>(basis.lmax) << '\n';
  output << "meta\trankmax\t" << static_cast<int>(basis.rankmax) << '\n';
  output << "meta\tndensitymax\t" << static_cast<int>(basis.ndensitymax) << '\n';
  output << "meta\tdelta_spline_bins\t" << require_finite(basis.deltaSplineBins, "spline spacing")
         << '\n';
  output << "meta\tcentral\t" << environment.central << '\n';
  for (int mu = 0; mu < basis.nelements; ++mu) {
    output << "element\t" << mu << '\t' << basis.elements_name[mu] << '\n';
  }
  for (std::size_t atom = 0; atom < environment.positions.size(); ++atom) {
    output << "atom\t" << atom << '\t' << static_cast<int>(environment.types[atom]);
    for (double coordinate : environment.positions[atom]) {
      output << '\t' << require_finite(coordinate, "atom coordinate");
    }
    output << '\n';
  }
  for (std::size_t edge = 0; edge < environment.neighbours.size(); ++edge) {
    output << "neighbor\t" << edge << '\t' << environment.neighbours[edge] << '\n';
  }
}

void write_edge_sources(std::ostream &output, ACECTildeBasisSet &basis,
                        const Environment &environment) {
  auto *radial = dynamic_cast<ACERadialFunctions *>(basis.radial_functions);
  if (radial == nullptr) {
    throw std::runtime_error("The first oracle supports only ACERadialFunctions");
  }

  const int central = environment.central;
  const auto &center = environment.positions[central];
  const SPECIES_TYPE mu_i = environment.types[central];
  for (std::size_t edge = 0; edge < environment.neighbours.size(); ++edge) {
    const int neighbour = environment.neighbours[edge];
    const SPECIES_TYPE mu_j = environment.types[neighbour];
    std::array<double, 3> displacement{};
    for (int axis = 0; axis < 3; ++axis) {
      displacement[axis] = environment.positions[neighbour][axis] - center[axis];
    }
    const double radius = norm(displacement);
    if (!(radius > 0.0)) throw std::runtime_error("Zero-length neighbor displacement");
    std::array<double, 3> unit{};
    for (int axis = 0; axis < 3; ++axis) unit[axis] = displacement[axis] / radius;

    radial->radbase(radial->lambda(mu_i, mu_j), radial->cut(mu_i, mu_j),
                    radial->dcut(mu_i, mu_j), radial->radbasenameij(mu_i, mu_j), radius,
                    radial->cut_in(mu_i, mu_j), radial->dcut_in(mu_i, mu_j));
    radial->radfunc(mu_i, mu_j);
    for (int k = 0; k < basis.nradbase; ++k) {
      output << "radial_direct_g\t" << edge << '\t' << k << '\t'
             << require_finite(radial->gr(k), "direct radial value") << '\t'
             << require_finite(radial->dgr(k), "direct radial derivative") << '\n';
    }
    for (int n = 0; n < basis.nradmax; ++n) {
      for (int l = 0; l <= basis.lmax; ++l) {
        output << "radial_direct_R\t" << edge << '\t' << n << '\t' << l << '\t'
               << require_finite(radial->fr(n, l), "direct contracted radial value") << '\t'
               << require_finite(radial->dfr(n, l), "direct contracted radial derivative") << '\n';
      }
    }

    radial->evaluate(radius, basis.nradbase, basis.nradmax, mu_i, mu_j);
    for (int k = 0; k < basis.nradbase; ++k) {
      output << "radial_spline_g\t" << edge << '\t' << k << '\t'
             << require_finite(radial->gr(k), "spline radial value") << '\t'
             << require_finite(radial->dgr(k), "spline radial derivative") << '\n';
    }
    for (int n = 0; n < basis.nradmax; ++n) {
      for (int l = 0; l <= basis.lmax; ++l) {
        output << "radial_spline_R\t" << edge << '\t' << n << '\t' << l << '\t'
               << require_finite(radial->fr(n, l), "spline contracted radial value") << '\t'
               << require_finite(radial->dfr(n, l), "spline contracted radial derivative") << '\n';
      }
    }

    basis.spherical_harmonics.compute_ylm(unit[0], unit[1], unit[2], basis.lmax);
    for (int l = 0; l <= basis.lmax; ++l) {
      for (int m = 0; m <= l; ++m) {
        output << "ylm\t" << edge << '\t' << l << '\t' << m;
        write_complex(output, basis.spherical_harmonics.ylm(l, m));
        output << '\n';
        const ACEDYcomponent &derivative = basis.spherical_harmonics.dylm(l, m);
        for (int axis = 0; axis < 3; ++axis) {
          output << "dylm\t" << edge << '\t' << l << '\t' << m << '\t' << axis;
          write_complex(output, derivative.a[axis]);
          output << '\n';
        }
      }
    }
  }
}

void write_products(std::ostream &output, const ACECTildeBasisSet &basis,
                    const PACEProductOracleEvaluator &evaluator, SPECIES_TYPE mu_i) {
  const auto &rank_one = evaluator.rank_one_atomic_base();
  const auto &atomic_base = evaluator.atomic_base();
  for (int function = 0; function < basis.total_basis_size_rank1[mu_i]; ++function) {
    const ACECTildeBasisFunction &row = basis.basis_rank1[mu_i][function];
    output << "function_rank1\t" << function << '\t' << static_cast<int>(row.mus[0]) << '\t'
           << static_cast<int>(row.ns[0]) << '\t' << static_cast<int>(row.ls[0]) << '\n';
    const double product = require_finite(rank_one(row.mus[0], row.ns[0] - 1),
                                          "rank-one product");
    for (int density = 0; density < row.ndensity; ++density) {
      const double coefficient = require_finite(row.ctildes[density], "rank-one coefficient");
      output << "product_rank1\t" << function << '\t' << density << '\t' << product << '\t'
             << coefficient << '\t'
             << require_finite(product * coefficient, "rank-one contribution") << '\n';
    }
  }

  for (int function = 0; function < basis.total_basis_size[mu_i]; ++function) {
    const ACECTildeBasisFunction &row = basis.basis[mu_i][function];
    for (int combination = 0; combination < row.num_ms_combs; ++combination) {
      ACEComplex product{1.0, 0.0};
      output << "product_label\t" << function << '\t' << combination << '\t'
             << static_cast<int>(row.rank);
      for (int factor = 0; factor < row.rank; ++factor) {
        const int index = combination * row.rank + factor;
        output << '\t' << static_cast<int>(row.ms_combs[index]);
        product *= atomic_base(row.mus[factor], row.ns[factor] - 1, row.ls[factor],
                               row.ms_combs[index]);
      }
      output << '\n';
      for (int density = 0; density < row.ndensity; ++density) {
        const double coefficient = require_finite(
            row.ctildes[combination * row.ndensity + density], "C-tilde coefficient");
        output << "product\t" << function << '\t' << combination << '\t' << density;
        write_complex(output, product);
        output << '\t' << coefficient << '\t'
               << require_finite(product.real_part_product(coefficient), "C-tilde contribution")
               << '\n';
      }
    }
  }
}

double write_evaluator(std::ostream &output, const ACECTildeBasisSet &basis,
                       const Environment &environment,
                       const PACEProductOracleEvaluator &evaluator) {
  const SPECIES_TYPE mu_i = environment.types[environment.central];
  if (basis.map_embedding_specifications.at(mu_i).ndensity != 1) {
    throw std::runtime_error("The first oracle consistency checks require one density");
  }
  const auto &rank_one = evaluator.rank_one_atomic_base();
  const auto &atomic_base = evaluator.atomic_base();
  for (int mu = 0; mu < basis.nelements; ++mu) {
    for (int n = 0; n < basis.nradbase; ++n) {
      output << "A_rank1\t" << mu << '\t' << n << '\t'
             << require_finite(rank_one(mu, n), "rank-one atomic base") << '\n';
    }
    for (int n = 0; n < basis.nradmax; ++n) {
      for (int l = 0; l <= basis.lmax; ++l) {
        for (int m = -l; m <= l; ++m) {
          output << "A\t" << mu << '\t' << n << '\t' << l << '\t' << m;
          write_complex(output, atomic_base(mu, n, l, m));
          output << '\n';
        }
      }
    }
  }

  write_products(output, basis, evaluator, mu_i);

  double projection_sum = 0.0;
  const int rank_one_count = basis.total_basis_size_rank1[mu_i];
  const int higher_rank_count = basis.total_basis_size[mu_i];
  for (int function = 0; function < rank_one_count + higher_rank_count; ++function) {
    const double projection = require_finite(evaluator.projections(function), "projection");
    output << "projection\t" << function << '\t' << projection << '\n';
    projection_sum += projection;
  }
  for (int density = 0; density < basis.map_embedding_specifications.at(mu_i).ndensity;
       ++density) {
    output << "rho\t" << density << '\t' << require_finite(evaluator.rhos(density), "density")
           << '\n';
    output << "dF_drho\t" << density << '\t'
           << require_finite(evaluator.dF_drho(density), "embedding derivative") << '\n';
  }
  if (std::abs(evaluator.dF_drho(0) - 1.0) > 5.0e-14) {
    throw std::runtime_error("The first oracle consistency checks require identity embedding");
  }

  for (int function = 0; function < rank_one_count; ++function) {
    for (int mu = 0; mu < basis.nelements; ++mu) {
      for (int n = 0; n < basis.nradbase; ++n) {
        output << "root_rank1_dB\t" << function << '\t' << mu << '\t' << n << '\t'
               << require_finite(evaluator.weights_rank1_dB(function, mu, n),
                                 "rank-one descriptor root")
               << '\n';
      }
    }
  }
  for (int function = 0; function < higher_rank_count; ++function) {
    for (int mu = 0; mu < basis.nelements; ++mu) {
      for (int n = 0; n < basis.nradmax; ++n) {
        for (int l = 0; l <= basis.lmax; ++l) {
          for (int m = -l; m <= l; ++m) {
            output << "root_dB\t" << function << '\t' << mu << '\t' << n << '\t' << l
                   << '\t' << m;
            write_complex(output, evaluator.weights_dB(function, mu, n, l, m));
            output << '\n';
          }
        }
      }
    }
  }

  double maximum_force_residual = 0.0;
  for (std::size_t edge = 0; edge < environment.neighbours.size(); ++edge) {
    for (int axis = 0; axis < 3; ++axis) {
      double derivative_sum = 0.0;
      for (int function = 0; function < rank_one_count + higher_rank_count; ++function) {
        const double value = require_finite(evaluator.neighbours_dB(function, edge, axis),
                                            "edge descriptor derivative");
        output << "edge_dB\t" << function << '\t' << edge << '\t' << axis << '\t' << value
               << '\n';
        derivative_sum += value;
      }
      const double force = require_finite(evaluator.neighbours_forces(edge, axis),
                                          "neighbor force contribution");
      output << "central_force_contribution\t" << edge << '\t' << axis << '\t' << force
             << '\n';
      maximum_force_residual = std::max(maximum_force_residual, std::abs(force - derivative_sum));
    }
  }
  require_finite(projection_sum, "projection sum");
  require_finite(evaluator.e_atom, "atomic energy");
  require_finite(basis.E0vals(mu_i), "E0");
  output << "atomic_energy\t" << evaluator.e_atom << '\n';
  output << "check\tprojection_minus_rho\t" << projection_sum - evaluator.rhos(0) << '\n';
  output << "check\tedge_dB_minus_force_maxabs\t" << maximum_force_residual << '\n';
  output << "check\tprojection_plus_E0_minus_energy\t"
         << projection_sum + basis.E0vals(mu_i) - evaluator.e_atom << '\n';
  return std::max({std::abs(projection_sum - evaluator.rhos(0)), maximum_force_residual,
                   std::abs(projection_sum + basis.E0vals(mu_i) - evaluator.e_atom)});
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 4) {
      std::cerr << "usage: pace_product_oracle MODEL.yace ENVIRONMENT OUTPUT.tsv\n";
      return 2;
    }

    Environment environment = read_environment(argv[2]);
    ACECTildeBasisSet basis(argv[1]);
    for (SPECIES_TYPE type : environment.types) {
      if (type < 0 || type >= basis.nelements) {
        throw std::runtime_error("Environment atom type is not present in the model");
      }
    }

    std::ofstream output(argv[3]);
    if (!output) throw std::runtime_error("Could not open output: " + std::string(argv[3]));
    output << std::scientific << std::setprecision(17);
    write_input(output, basis, environment);
    write_edge_sources(output, basis, environment);

    std::vector<double *> position_rows(environment.positions.size());
    for (std::size_t atom = 0; atom < environment.positions.size(); ++atom) {
      position_rows[atom] = environment.positions[atom].data();
    }
    PACEProductOracleEvaluator evaluator;
    evaluator.set_basis(basis);
    evaluator.resize_neighbours_cache(static_cast<int>(environment.neighbours.size()));
    evaluator.compute_projections = true;
    evaluator.compute_b_grad = true;
    evaluator.compute_atom(environment.central, position_rows.data(), environment.types.data(),
                           static_cast<int>(environment.neighbours.size()),
                           environment.neighbours.data());
    const double residual = write_evaluator(output, basis, environment, evaluator);
    if (residual > 5.0e-12) {
      std::cerr << "PACE oracle internal consistency residual is " << residual << '\n';
      return 1;
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "PACE oracle failed: " << error.what() << '\n';
    return 1;
  }
}
