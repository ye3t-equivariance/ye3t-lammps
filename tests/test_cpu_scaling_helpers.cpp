// clang-format off
#include "ye3t_cpu_batching.h"
#include "ye3t_lifted_cauchy_cpu.h"
#include "reference/lifted_source_snapshot.h"
// clang-format on

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace YE3T_LAMMPS;
using Vec3 = std::array<double, 3>;

namespace {
void require(bool value, const std::string &message)
{
  if (!value) throw std::runtime_error(message);
}
void close(double x, double y, double tolerance, const std::string &where)
{
  require(std::isfinite(x) && std::isfinite(y) &&
              std::abs(x - y) <= tolerance * std::max({1.0, std::abs(x), std::abs(y)}),
          where + ": " + std::to_string(x) + " != " + std::to_string(y));
}
void close(const std::vector<double> &x, const std::vector<double> &y, double tolerance,
           const std::string &where)
{
  require(x.size() == y.size(), where + " size mismatch");
  for (std::size_t i = 0; i < x.size(); ++i) close(x[i], y[i], tolerance, where);
}
void close(const std::vector<Vec3> &x, const std::vector<Vec3> &y, double tolerance,
           const std::string &where)
{
  require(x.size() == y.size(), where + " size mismatch");
  for (std::size_t i = 0; i < x.size(); ++i)
    for (int k = 0; k < 3; ++k) close(x[i][k], y[i][k], tolerance, where);
}
template <class F> void throws(F f, const std::string &where)
{
  bool caught = false;
  try {
    f();
  } catch (const std::invalid_argument &) {
    caught = true;
  }
  require(caught, where + " did not reject invalid input");
}

struct SourceFixture {
  std::string name;
  LiftedCauchyModel model;
  std::vector<std::vector<LiftedCauchyEdge>> edges;
  std::vector<std::vector<double>> source, adjoint;
  std::vector<std::vector<Vec3>> gradients;
};
#include "lifted_source_fixtures_generated.h"

void test_buckets()
{
  CPUSpeciesBatches buckets;
  for (int n : {0, 1, 2, 7, 8, 9, 31, 32, 33, 255, 256, 257, 4096}) {
    std::vector<int> species(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) species[i] = (i * 7 + i / 3) % 5;
    buckets.build(n, 7, species.data());    // includes absent species
    require(buckets.offsets.back() == n, "bucket total");
    std::vector<int> seen(static_cast<std::size_t>(n));
    for (int s = 0; s < 7; ++s) {
      int previous = -1;
      for (int k = buckets.offsets[s]; k < buckets.offsets[s + 1]; ++k) {
        int atom = buckets.atoms[k];
        require(species[atom] == s && atom > previous, "unstable or wrong species bucket");
        ++seen[atom];
        previous = atom;
      }
    }
    require(std::all_of(seen.begin(), seen.end(),
                        [](int k) {
                          return k == 1;
                        }),
            "bucket atom coverage");
  }
  std::vector<int> species(256, 0);
  species[137] = 1;
  buckets.build(256, 2, species.data());
  require(buckets.offsets[1] == 255, "minority species destroyed majority batch");
  const int bad[] = {-1};
  throws(
      [&] {
        buckets.build(1, 2, bad);
      },
      "negative species");
  throws(
      [&] {
        buckets.build(1, 2, nullptr);
      },
      "null species");
  std::cout << "PASS stable species buckets, minority species, reuse, invalid inputs\n";
}

void test_archive_sources()
{
  for (const auto &fixture : archived_fixtures()) {
    LiftedCauchyCPUSource source(&fixture.model);
    testing::ReferenceLiftedSource reference(&fixture.model);
    std::vector<double> actual, old;
    std::vector<Vec3> gradient, old_gradient;
    for (std::size_t center = 0; center < fixture.edges.size(); ++center) {
      for (auto policy :
           {LiftedCauchySourcePolicy::DIRECT_Q, LiftedCauchySourcePolicy::FACTORIZED}) {
        source.accumulate(fixture.edges[center], policy, actual);
        reference.accumulate(fixture.edges[center], policy, old);
        close(actual, old, 2e-13, fixture.name + " source vs original");
        close(actual, fixture.source[center], 2e-11, fixture.name + " archived source");
        source.vjp(fixture.edges[center], fixture.adjoint[center], policy, gradient);
        reference.vjp(fixture.edges[center], fixture.adjoint[center], policy, old_gradient);
        close(gradient, old_gradient, 2e-13, fixture.name + " VJP vs original");
        close(gradient, fixture.gradients[center], 2e-10, fixture.name + " archived VJP");
      }
    }
    std::cout << "PASS archived source and VJP: " << fixture.name << "\n";
  }
}

LiftedCauchyModel synthetic_model()
{
  LiftedCauchyModel model;
  model.cutoff = 5.0;
  model.role_dimension = 2;
  model.real_component_count = 33;
  model.central_species_order = {"A", "B"};
  int offset = 0;
  for (int angular : {0, 1, 2, 3, 4, 5, 6, 7, 8, 16}) {
    LiftedCauchySourceGroup group;
    group.angular = angular;
    group.real_component_count = 2 * angular + 1;
    group.source_dimension = 2;
    group.neighbor_species_index = angular % 2;
    // Reversed and padded model offsets catch accidental compact/global mixing.
    group.q_source_variable_offsets = {offset + group.real_component_count, offset};
    group.factorized_radials = {{0, 0, 3}, {1, 1, 2}};
    group.transform_q_from_f = {1.0, 0.5, -0.25, 1.2};
    group.direct_q_polynomials = {{0, {1.0, -0.5}}, {1, {-0.25, 1.45}}};
    model.source_groups.push_back(group);
    offset += 2 * group.real_component_count + 3;
  }
  model.source_variable_count = offset;
  return model;
}

void test_random_source()
{
  auto model = synthetic_model();
  LiftedCauchyCPUSource source(&model);
  testing::ReferenceLiftedSource reference(&model);
  std::mt19937 random(190917);
  std::uniform_real_distribution<double> uniform(-2.4, 2.4);
  std::vector<LiftedCauchyEdge> edges;
  for (int i = 0; i < 73; ++i)
    edges.push_back({i % 2, {uniform(random), uniform(random), uniform(random)}});
  // Include origin, coordinate axes, exact/inside/outside cutoff.
  edges.insert(edges.end(),
               {{0, {0, 0, 0}},
                {1, {1, 0, 0}},
                {0, {0, -2, 0}},
                {1, {0, 0, 3}},
                {0, {5, 0, 0}},
                {1, {6, 0, 0}},
                {0, {std::nextafter(5.0, 0.0), 0, 0}}});
  std::vector<double> seed(static_cast<std::size_t>(model.source_variable_count));
  for (double &value : seed) value = uniform(random);
  std::vector<double> actual, old, direct;
  std::vector<Vec3> gradient, old_gradient, direct_gradient;
  for (auto policy : {LiftedCauchySourcePolicy::DIRECT_Q, LiftedCauchySourcePolicy::FACTORIZED}) {
    source.accumulate(edges, policy, actual);
    reference.accumulate(edges, policy, old);
    close(actual, old, 2e-13, "random source vs original");
    source.vjp(edges, seed, policy, gradient);
    reference.vjp(edges, seed, policy, old_gradient);
    close(gradient, old_gradient, 2e-13, "random VJP vs original");
    if (policy == LiftedCauchySourcePolicy::DIRECT_Q) {
      direct = actual;
      direct_gradient = gradient;
    } else {
      close(actual, direct, 2e-11, "random direct/factorized source");
      close(gradient, direct_gradient, 2e-10, "random direct/factorized VJP");
    }
    // Check actual coordinate derivatives, away from origin/cutoff.
    for (int edge = 0; edge < 6; ++edge) {
      for (int axis = 0; axis < 3; ++axis) {
        auto shifted = edges;
        const double eps = 1e-6;
        std::vector<double> plus, minus;
        shifted[edge].displacement[axis] += eps;
        source.accumulate(shifted, policy, plus);
        shifted[edge].displacement[axis] -= 2 * eps;
        source.accumulate(shifted, policy, minus);
        double fd = 0;
        for (std::size_t i = 0; i < seed.size(); ++i)
          fd += seed[i] * (plus[i] - minus[i]) / (2 * eps);
        close(fd, gradient[edge][axis], 2e-6, "coordinate finite difference");
      }
    }
    auto reordered = edges;
    std::reverse(reordered.begin(), reordered.end());
    source.accumulate(reordered, policy, old);
    close(old, actual, 2e-11, "neighbor reordering");
    source.vjp(reordered, seed, policy, old_gradient);
    std::reverse(old_gradient.begin(), old_gradient.end());
    close(old_gradient, gradient, 2e-13, "reordered VJP");
    source.accumulate({}, policy, old);
    require(std::all_of(old.begin(), old.end(),
                        [](double x) {
                          return x == 0;
                        }),
            "empty source");
    source.vjp({}, seed, policy, old_gradient);
    require(old_gradient.empty(), "empty VJP");
  }
  auto bad = edges;
  bad[0].neighbor_species_index = 2;
  throws(
      [&] {
        source.accumulate(bad, LiftedCauchySourcePolicy::DIRECT_Q, actual);
      },
      "invalid neighbor species");
  bad = edges;
  bad[0].displacement[0] = std::numeric_limits<double>::quiet_NaN();
  throws(
      [&] {
        source.accumulate(bad, LiftedCauchySourcePolicy::DIRECT_Q, actual);
      },
      "nonfinite geometry");
  throws(
      [&] {
        source.vjp(edges, {}, LiftedCauchySourcePolicy::DIRECT_Q, gradient);
      },
      "wrong adjoint width");
  std::cout << "PASS randomized mixed-l sources (l=0..8,16), compact/padded offsets, "
               "finite differences, permutation, boundaries, empty/invalid inputs\n";
}
}    // namespace

int main()
{
  try {
    test_buckets();
    test_archive_sources();
    test_random_source();
  } catch (const std::exception &e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
  std::cout << "Source/helper checks only; no YE3T-core, LAMMPS, MPI, or GPU validation.\n";
}
