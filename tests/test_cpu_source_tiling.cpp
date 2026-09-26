#include "ye3t_cpu_source_tiling.h"

#include <array>
#include <cerrno>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace YE3T_LAMMPS;
using Complex = std::complex<double>;
using Vec3 = std::array<double, 3>;
namespace {
void require(bool ok, const char *message)
{
  if (!ok) throw std::runtime_error(message);
}
void close(double a, double b, double tol, const char *message)
{
  require(std::isfinite(a) && std::isfinite(b) &&
              std::abs(a - b) <= tol * std::max({1.0, std::abs(a), std::abs(b)}),
          message);
}
template <class F> void invalid(F f)
{
  bool caught = false;
  try {
    f();
  } catch (const std::invalid_argument &) {
    caught = true;
  }
  require(caught, "invalid dimensions not rejected");
}

void test_policy()
{
  for (std::size_t cache :
       {std::size_t(0), std::size_t(1), std::size_t(128 * 1024), std::size_t(512 * 1024),
        std::size_t(4 * 1024 * 1024), std::numeric_limits<std::size_t>::max()}) {
    for (int bonds : {1, 4, 25}) {
      for (int width : {1, 3, 45, 153, 100000}) {
        const auto p = make_cpu_source_tile_policy(11, 36, width, bonds, 2048, cache);
        require(p.edge_capacity > 0, "tile capacity must be positive");
        require(p.target_bytes >= 64 * 1024 && p.target_bytes <= 1024 * 1024,
                "cache-derived target outside clamp");
        if (p.edge_capacity > 1) {
          require(p.fixed_bytes + static_cast<std::size_t>(p.edge_capacity) * p.bytes_per_edge <=
                      p.target_bytes,
                  "estimated tile exceeds target");
        }
      }
    }
  }
  const auto single = make_cpu_source_tile_policy(11, 36, 45, 1, 0, 0);
  const auto mixed = make_cpu_source_tile_policy(11, 36, 45, 4, 0, 0);
  require(single.target_bytes == 256 * 1024, "fallback changed");
  require(mixed.bytes_per_edge > single.bytes_per_edge && mixed.fixed_bytes > 0,
          "mixed-bond gather scratch not counted");
  require(mixed.edge_capacity <= single.edge_capacity, "mixed capacity too large");
  require(make_cpu_source_tile_policy(1, 1, 1, 1, 8 * 1024 * 1024, 0).edge_capacity == 1,
          "fixed workspace must still permit one edge");
  invalid([] {
    make_cpu_source_tile_policy(-1, 1, 1, 1, 0, 0);
  });
  invalid([] {
    make_cpu_source_tile_policy(1, -1, 1, 1, 0, 0);
  });
  invalid([] {
    make_cpu_source_tile_policy(1, 1, 0, 1, 0, 0);
  });
  invalid([] {
    make_cpu_source_tile_policy(1, 1, 1, 0, 0, 0);
  });
  bool overflow = false;
  try {
    make_cpu_source_tile_policy(1, 1, 1, 4, std::numeric_limits<std::size_t>::max(), 0);
  } catch (const std::overflow_error &) {
    overflow = true;
  }
  require(overflow, "workspace addition overflow was not checked");
  errno = EDOM;
  const auto reported = cpu_reported_l2_cache_bytes();
  require(errno == EDOM, "cache query changed errno");
  std::cout << "PASS policy dimensions, gather accounting, fallback, overflow; reported L2="
            << reported << " bytes\n";
}

void test_schedule()
{
  for (int n : {0, 1, 2, 7, 8, 9, 31, 255, 256, 257, 2049}) {
    for (int capacity : {1, 2, 3, 8, 17, 256, 4096}) {
      std::vector<int> forward, backward(static_cast<std::size_t>(n), 0);
      int prepared_begin = -1, prepared_end = -1, prepares = 0, reads = 0;
      cpu_source_tiled_passes(
          n, capacity,
          [&](int a, int b) {
            require(b > a && b - a <= capacity, "invalid prepared interval");
            prepared_begin = a;
            prepared_end = b;
            ++prepares;
          },
          [&](int a, int b) {
            require(a == prepared_begin && b == prepared_end && reads == 0,
                    "forward used stale tables or incomplete readout");
            for (int i = a; i < b; ++i) forward.push_back(i);
          },
          [&] {
            require(forward.size() == static_cast<std::size_t>(n), "early readout");
            ++reads;
          },
          [&](int a, int b) {
            require(a == prepared_begin && b == prepared_end && reads == 1,
                    "VJP used stale tables or incomplete readout");
            for (int i = a; i < b; ++i) ++backward[i];
          });
      require(reads == 1, "readout batch was split");
      require(prepares == (n ? 2 * ((n - 1) / capacity + 1) - 1 : 0), "tile reuse/recompute count");
      for (int i = 0; i < n; ++i)
        require(forward[i] == i && backward[i] == 1, "edge lost, reordered or duplicated");
    }
  }
  auto no_edges = [](int, int) {
  };
  auto no_read = [] {
  };
  invalid([&] {
    cpu_source_tiled_passes(-1, 1, no_edges, no_edges, no_read, no_edges);
  });
  invalid([&] {
    cpu_source_tiled_passes(1, 0, no_edges, no_edges, no_read, no_edges);
  });
  // Arithmetic near INT_MAX without allocating or traversing billions of edges.
  for (int capacity : {std::numeric_limits<int>::max(), std::numeric_limits<int>::max() - 1}) {
    long long count = 0, reverse = 0;
    cpu_source_tiled_passes(
        std::numeric_limits<int>::max(), capacity, no_edges,
        [&](int a, int b) {
          count += static_cast<long long>(b) - a;
        },
        no_read,
        [&](int a, int b) {
          reverse += static_cast<long long>(b) - a;
        });
    require(count == std::numeric_limits<int>::max() && reverse == count,
            "schedule integer overflow");
  }
  std::cout
      << "PASS ordered edge coverage, single readout, checkpoint reuse, empty input, INT_MAX\n";
}

// Analytic test tables, deliberately NOT a substitute for the missing runtime's
// PACE spline/harmonic kernels. Exercise the production contraction arithmetic
// and checkpoint driver with independently differentiable inputs.
struct Tables {
  double radius;
  Vec3 direction;
  std::array<double, 2> radial, radial_derivative;
  std::array<double, 3> contracted, contracted_derivative;
  std::array<Complex, 3> angular;
  std::array<Complex, 9> angular_derivative;
};
Tables tables(const Vec3 &v)
{
  const double x = v[0], y = v[1], z = v[2];
  Tables t;
  const double r = std::sqrt(x * x + y * y + z * z);
  t.radius = r;
  t.direction = {x / r, y / r, z / r};
  t.radial = {r * r, std::exp(-r / 2)};
  t.radial_derivative = {2 * r, -0.5 * std::exp(-r / 2)};
  t.contracted = {r + 0.2, r * r * r, 1 / (1 + r)};
  t.contracted_derivative = {1, 3 * r * r, -1 / ((1 + r) * (1 + r))};
  t.angular = {Complex(0.8, 0.1), Complex(x + y, 2 * z - x), Complex(x * y, y * z)};
  t.angular_derivative = {Complex{},      Complex{},     Complex{},
                          Complex(1, -1), Complex(1, 0), Complex(0, 2),
                          Complex(y, 0),  Complex(x, z), Complex(0, y)};
  return t;
}
struct Edge {
  int center, neighbor;
  Vec3 vector;
};
struct Case {
  std::vector<int> species;
  std::vector<Edge> edges;
  std::array<YACEBond, 4> bonds;
};
Case make_case(int ncenters, int nedges)
{
  Case c;
  c.species.resize(ncenters);
  for (int i = 0; i < ncenters; ++i) c.species[i] = i % 2;
  for (int i = 0; i < 4; ++i) {
    auto &b = c.bonds[i];
    b.cutoff = 2.0 + 0.2 * i;
    b.radial_channel_outputs = {0, 2, 0};
    b.radial_channel_indices = {0, 1, 1};
    b.angular_channel_outputs = {1, 3, 1, 2};
    b.contracted_channel_indices = {0, 1, 2, 0};
    b.angular_channel_nonnegative_indices = {0, 1, 2, 1};
  }
  // Ragged species/bond maps and empty radial/angular maps are valid cases.
  c.bonds[1].radial_channel_outputs.clear();
  c.bonds[1].radial_channel_indices.clear();
  c.bonds[2].angular_channel_outputs.clear();
  c.bonds[2].contracted_channel_indices.clear();
  c.bonds[2].angular_channel_nonnegative_indices.clear();
  std::mt19937 gen(7309);
  std::uniform_real_distribution<double> dis(-1.5, 1.5);
  for (int e = 0; e < nedges; ++e) {
    // Deliberately interleaved centers, not merely contiguous environments.
    c.edges.push_back(
        {(e * 7 + e / 3) % std::max(1, ncenters), e % 2, {dis(gen), dis(gen), dis(gen)}});
  }
  return c;
}
struct Result {
  std::vector<Complex> source;
  std::vector<double> energy;
  std::vector<Vec3> gradient;
  int readouts = 0;
  std::size_t peak_tables = 0;
};
Result evaluate(const Case &c, int capacity, bool independent_reference = false)
{
  Result out;
  // Width varies by center species; offsets and unused components are retained.
  std::vector<std::size_t> offsets(c.species.size() + 1, 0);
  for (std::size_t i = 0; i < c.species.size(); ++i) offsets[i + 1] = offsets[i] + 4 + c.species[i];
  out.source.assign(offsets.back(), {});
  out.energy.assign(c.species.size(), 0.0);
  out.gradient.resize(c.edges.size());
  std::vector<Complex> root(offsets.back());
  std::vector<Tables> scratch;
  cpu_source_tiled_passes(
      static_cast<int>(c.edges.size()), capacity,
      [&](int begin, int end) {
        scratch.resize(end - begin);
        out.peak_tables = std::max(out.peak_tables, scratch.size());
        for (int e = begin; e < end; ++e) scratch[e - begin] = tables(c.edges[e].vector);
      },
      [&](int begin, int end) {
        for (int e = begin; e < end; ++e) {
          const auto &edge = c.edges[e];
          const auto &t = scratch[e - begin];
          const auto &b = c.bonds[c.species[edge.center] * 2 + edge.neighbor];
          if (t.radius >= b.cutoff) continue;
          auto *q = out.source.data() + offsets[edge.center];
          if (!independent_reference) {
            cpu_accumulate_ace_source_edge(b, t.radial.data(), t.contracted.data(),
                                           t.angular.data(), q);
          } else {
            // Independent direct map, not a call through the production helper.
            for (std::size_t k = 0; k < b.radial_channel_outputs.size(); ++k)
              q[b.radial_channel_outputs[k]] += t.radial[b.radial_channel_indices[k]];
            for (std::size_t k = 0; k < b.angular_channel_outputs.size(); ++k)
              q[b.angular_channel_outputs[k]] += t.contracted[b.contracted_channel_indices[k]] *
                  t.angular[b.angular_channel_nonnegative_indices[k]];
          }
        }
      },
      [&] {
        ++out.readouts;
        for (std::size_t i = 0; i < c.species.size(); ++i) {
          out.energy[i] = 0.7;    // isolated reference energy must also be evaluated
          for (std::size_t j = offsets[i]; j < offsets[i + 1]; ++j) {
            // Nonlinear analytic readout: performing this on partial source sums
            // would give a detectably incorrect energy and force.
            const double weight = 0.5 + 0.1 * (j - offsets[i]);
            out.energy[i] += 0.5 * weight * std::norm(out.source[j]) + 0.13 * out.source[j].real();
            root[j] = weight * out.source[j] + 0.13;
          }
        }
      },
      [&](int begin, int end) {
        for (int e = begin; e < end; ++e) {
          const auto &edge = c.edges[e];
          const auto &t = scratch[e - begin];
          const auto &b = c.bonds[c.species[edge.center] * 2 + edge.neighbor];
          if (t.radius >= b.cutoff) {
            out.gradient[e] = {0, 0, 0};
            continue;
          }
          const auto *q = root.data() + offsets[edge.center];
          if (!independent_reference) {
            cpu_pullback_ace_source_edge(b, t.direction.data(), t.radial_derivative.data(),
                                         t.contracted.data(), t.contracted_derivative.data(),
                                         t.angular.data(), t.angular_derivative.data(), q,
                                         out.gradient[e].data());
          } else {
            out.gradient[e] = {0, 0, 0};
            for (int d = 0; d < 3; ++d) {
              for (std::size_t k = 0; k < b.radial_channel_outputs.size(); ++k)
                out.gradient[e][d] += q[b.radial_channel_outputs[k]].real() *
                    t.radial_derivative[b.radial_channel_indices[k]] * t.direction[d];
              for (std::size_t k = 0; k < b.angular_channel_outputs.size(); ++k) {
                int a = b.angular_channel_nonnegative_indices[k],
                    r = b.contracted_channel_indices[k];
                Complex jac = t.contracted_derivative[r] * t.direction[d] * t.angular[a] +
                    t.contracted[r] * t.angular_derivative[3 * a + d];
                out.gradient[e][d] += (std::conj(q[b.angular_channel_outputs[k]]) * jac).real();
              }
            }
          }
        }
      });
  return out;
}
void compare(const Result &a, const Result &b, double tol)
{
  require(a.source.size() == b.source.size() && a.gradient.size() == b.gradient.size() &&
              a.energy.size() == b.energy.size(),
          "result shape mismatch");
  for (std::size_t i = 0; i < a.source.size(); ++i) {
    close(a.source[i].real(), b.source[i].real(), tol, "source real mismatch");
    close(a.source[i].imag(), b.source[i].imag(), tol, "source imaginary mismatch");
  }
  for (std::size_t i = 0; i < a.energy.size(); ++i)
    close(a.energy[i], b.energy[i], tol, "energy mismatch");
  for (std::size_t i = 0; i < a.gradient.size(); ++i)
    for (int d = 0; d < 3; ++d) close(a.gradient[i][d], b.gradient[i][d], tol, "VJP mismatch");
}
void test_contractions()
{
  for (int centers : {0, 1, 2, 17, 256}) {
    for (int edges : {0, 1, 7, 32, 513}) {
      if (centers == 0 && edges) continue;
      auto c = make_case(centers, edges);
      const auto baseline = evaluate(c, std::max(1, edges), true);
      for (int tile : {1, 2, 3, 8, 16, 31, 256, 4096}) {
        const auto result = evaluate(c, tile);
        compare(result, baseline, 5e-13);
        compare(result, evaluate(c, std::max(1, edges)), 0.0);    // identical accumulation order
        require(result.readouts == 1 && result.peak_tables <= static_cast<std::size_t>(tile),
                "source tiling shrank readout or exceeded scratch tile");
      }
    }
  }
  auto c = make_case(3, 23);
  const auto result = evaluate(c, 2);
  constexpr double h = 1e-6;
  for (std::size_t e = 0; e < c.edges.size(); ++e) {
    for (int d = 0; d < 3; ++d) {
      auto plus = c, minus = c;
      plus.edges[e].vector[d] += h;
      minus.edges[e].vector[d] -= h;
      auto ep = evaluate(plus, 2), em = evaluate(minus, 2);
      double derivative = (std::accumulate(ep.energy.begin(), ep.energy.end(), 0.0) -
                           std::accumulate(em.energy.begin(), em.energy.end(), 0.0)) /
          (2 * h);
      close(derivative, result.gradient[e][d], 2e-7, "coordinate finite difference");
    }
  }
  // Cutoff acceptance is fresh for each evaluation, not cached between calls.
  auto cross = make_case(1, 1);
  cross.edges[0].vector = {1.5, 0, 0};
  auto inside = evaluate(cross, 1);
  cross.edges[0].vector = {2.0, 0, 0};
  auto boundary = evaluate(cross, 1);
  cross.edges[0].vector = {2.5, 0, 0};
  auto outside = evaluate(cross, 1);
  require(inside.energy[0] != 0.7 && boundary.energy[0] == 0.7 && outside.energy[0] == 0.7,
          "cutoff acceptance mismatch");
  require(outside.gradient[0] == Vec3{0, 0, 0}, "excluded edge gradient not zeroed");
  std::cout << "PASS production source contractions/VJP: analytic tables, nonlinear readout, "
               "mixed/ragged maps, split centers, cutoff, finite differences\n";
}
}    // namespace
int main()
{
  try {
    test_policy();
    test_schedule();
    test_contractions();
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
