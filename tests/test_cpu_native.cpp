#include "ye3t_cpu_evaluator.h"
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
namespace YE3T_LAMMPS {
YACEModel load_native_fixture(const std::string &);
YACEModel make_two_species_native_fixture(YACEModel);
struct YE3TCPUEvaluatorTestAccess {
  static void replay(YE3TCPUEvaluator &e, int cap)
  {
    e.reference_edge_tiling_ = true;
    e.source_tile_policy_.edge_capacity = cap;
  }
  static std::uint64_t edges(const YE3TCPUEvaluator &e) { return e.last_source_edges_evaluated_; }
  static int active_l(const YE3TCPUEvaluator &e) { return e.active_angular_maximum_; }
};
}    // namespace YE3T_LAMMPS
using namespace YE3T_LAMMPS;
void require(bool c, const std::string &m)
{
  if (!c) throw std::runtime_error(m);
}
struct Batch {
  std::vector<int> types, centers, neighbors;
  std::vector<double> xyz;
};
struct Values {
  std::vector<double> e, g;
};
Values eval(YE3TCPUEvaluator &e, const Batch &b)
{
  Values v;
  v.e.resize(b.types.size());
  v.g.resize(b.xyz.size());
  e.evaluate(b.types.size(), b.types.data(), b.centers.size(), b.centers.data(), b.neighbors.data(),
             b.xyz.data(), v.e.data(), v.g.data());
  return v;
}
void close(const std::vector<double> &a, const std::vector<double> &b, double tol, const char *what)
{
  require(a.size() == b.size(), "size");
  for (std::size_t i = 0; i < a.size(); ++i)
    require(std::isfinite(a[i]) && std::isfinite(b[i]) &&
                std::abs(a[i] - b[i]) <= tol * (1 + std::abs(a[i]) + std::abs(b[i])),
            what);
}
Batch make_batch(const YACEModel &m, int n)
{
  Batch b;
  b.types.resize(n);
  std::mt19937_64 rng(991);
  std::uniform_real_distribution<double> d(-.01, .01);
  const double a = 3.3161146998079496;
  std::vector<std::array<double, 3>> shell;
  for (int x = -2; x <= 2; ++x)
    for (int y = -2; y <= 2; ++y)
      for (int z = -2; z <= 2; ++z)
        for (int h = 0; h < 2; ++h) {
          std::array<double, 3> p{{a * (x + .5 * h), a * (y + .5 * h), a * (z + .5 * h)}};
          double rr = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
          if (rr > 1e-10 && rr < (m.maximum_cutoff() - .03) * (m.maximum_cutoff() - .03))
            shell.push_back(p);
        }
  for (int i = 0; i < n; ++i) {
    b.types[i] = i % m.species_count();
    int count = i % 11 == 0 ? 0 : int(shell.size()) - i % 3;
    for (int j = 0; j < count; ++j) {
      b.centers.push_back(i);
      b.neighbors.push_back((i + j) % m.species_count());
      for (int k = 0; k < 3; ++k) b.xyz.push_back(shell[j][k] + d(rng));
    }
    // Rejected neighbors exercise the runtime cutoff even when LAMMPS would omit them.
    if (i % 4 == 1) {
      b.centers.push_back(i);
      b.neighbors.push_back(i % m.species_count());
      b.xyz.insert(b.xyz.end(), {m.maximum_cutoff(), 0, 0});
    }
  }
  return b;
}
void test_model(YACEModel &m)
{
  YE3TCPUEvaluator automatic(&m), whole(&m), tiny(&m);
  YE3TCPUEvaluatorTestAccess::replay(whole, std::numeric_limits<int>::max());
  YE3TCPUEvaluatorTestAccess::replay(tiny, 7);
  for (int n : {0, 1, 7, 8, 9, 16, 17, 31, 129, 256}) {
    Batch b = make_batch(m, n);
    auto a = eval(automatic, b), r = eval(whole, b), t = eval(tiny, b);
    require(automatic.maximum_imaginary_density() <= 1.0e-10 &&
                whole.maximum_imaginary_density() <= 1.0e-10 &&
                tiny.maximum_imaginary_density() <= 1.0e-10,
            "native density has a material imaginary component");
    close(a.e, r.e, 2e-12, "whole energy");
    close(a.g, r.g, 2e-12, "whole gradient");
    close(a.e, t.e, 2e-12, "replay energy");
    close(a.g, t.g, 2e-12, "replay gradient");
    require(YE3TCPUEvaluatorTestAccess::edges(automatic) == b.centers.size(),
            "production replayed a source edge");
    std::vector<int> permutation(b.centers.size());
    std::iota(permutation.begin(), permutation.end(), 0);
    std::mt19937 rng(29);
    std::shuffle(permutation.begin(), permutation.end(), rng);
    Batch shuffled;
    shuffled.types = b.types;
    for (int e : permutation) {
      shuffled.centers.push_back(b.centers[e]);
      shuffled.neighbors.push_back(b.neighbors[e]);
      shuffled.xyz.insert(shuffled.xyz.end(), b.xyz.begin() + 3 * e, b.xyz.begin() + 3 * e + 3);
    }
    auto s = eval(automatic, shuffled);
    close(a.e, s.e, 5e-11, "shuffled energies");
    std::vector<double> unshuffled(a.g.size());
    for (std::size_t k = 0; k < permutation.size(); ++k)
      std::copy_n(s.g.data() + 3 * k, 3, unshuffled.data() + 3 * permutation[k]);
    close(a.g, unshuffled, 5e-11, "shuffled gradients");
    // Changing geometry on the same evaluator must not reuse previous-step data.
    for (auto &x : b.xyz) x *= 1.0003;
    auto changed = eval(automatic, b), changed_ref = eval(whole, b);
    close(changed.e, changed_ref.e, 2e-12, "changed energy");
    close(changed.g, changed_ref.g, 2e-12, "changed gradient");
  }
  Batch b = make_batch(m, 19);
  auto v = eval(automatic, b);
  int checked = 0;
  for (std::size_t coordinate = 0; coordinate < b.xyz.size() && checked < 15; coordinate += 13) {
    const std::size_t edge = coordinate / 3;
    double norm = 0;
    for (int a = 0; a < 3; ++a) norm += b.xyz[3 * edge + a] * b.xyz[3 * edge + a];
    if (std::abs(std::sqrt(norm) - m.maximum_cutoff()) < 1e-4) continue;
    const double old = b.xyz[coordinate], h = 2e-6;
    b.xyz[coordinate] = old + h;
    auto plus = eval(automatic, b);
    b.xyz[coordinate] = old - h;
    auto minus = eval(automatic, b);
    b.xyz[coordinate] = old;
    int center = b.centers[edge];
    double fd = (plus.e[center] - minus.e[center]) / (2 * h), g = v.g[coordinate];
    require(std::abs(fd - g) < 2e-6 * (1 + std::abs(g)), "energy coordinate finite difference");
    ++checked;
  }
  Batch empty;
  empty.types = {0, 0, 0};
  auto z = eval(automatic, empty);
  for (double x : z.e) require(x == m.species(0).reference_energy, "isolated energy");
  bool caught = false;
  Batch bad = make_batch(m, 2);
  bad.centers[0] = 2;
  try {
    eval(automatic, bad);
  } catch (const std::invalid_argument &) {
    caught = true;
  }
  require(caught, "invalid center not caught");
  bad = make_batch(m, 2);
  bad.xyz[0] = std::numeric_limits<double>::quiet_NaN();
  caught = false;
  try {
    eval(automatic, bad);
  } catch (const std::invalid_argument &) {
    caught = true;
  }
  require(caught, "invalid geometry not caught");
  std::cout << "PASS species=" << m.species_count()
            << " declared_l=" << m.maximum_angular_momentum()
            << " active_l=" << YE3TCPUEvaluatorTestAccess::active_l(automatic)
            << " finite_differences=" << checked << '\n';
}
int main(int argc, char **argv)
try {
  require(argc > 1, "need fixture paths");
  for (int i = 1; i < argc; ++i) {
    auto m = load_native_fixture(argv[i]);
    std::cout << argv[i] << '\n';
    test_model(m);
    if (i == 1) {
      auto mixed = make_two_species_native_fixture(m);
      test_model(mixed);
    }
  }
  std::cout << "native CPU scheduling tests PASS\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
