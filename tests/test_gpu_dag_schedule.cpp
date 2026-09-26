// clang-format off
#include "ye3t_gpu_dag_schedule.h"
#include "ye3t_gpu_block_schedule.h"
// clang-format on
#include <algorithm>
#include <complex>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#ifdef YE3T_TEST_NATIVE_DAG
#include "ye3t_yace_model.h"
namespace YE3T_LAMMPS {
YACEModel load_native_fixture(const std::string &);
}
#endif

using namespace YE3T_LAMMPS;
using Complex = std::complex<double>;
using Indices = std::vector<std::int64_t>;
template <class T> struct Span {
  const T *p;
  const T &operator()(int i) const { return p[i]; }
};
struct Plan {
  Span<int> dag_parent_offsets, dag_parents, dag_siblings;
  Span<double> dag_readout_seed;
};
void require(bool ok, const char *message)
{
  if (!ok) throw std::runtime_error(message);
}
void close(double a, double b, double tol = 2e-12)
{
  if (!std::isfinite(a) || !std::isfinite(b) ||
      std::abs(a - b) > tol * (1 + std::abs(a) + std::abs(b)))
    throw std::runtime_error("GPU DAG arithmetic mismatch: " + std::to_string(a) + " vs " +
                             std::to_string(b));
}
Complex cpow(Complex a, int n)
{
  Complex out = 1.;
  while (n-- > 0) out *= a;
  return out;
}
void exercise(int ns, const Indices &channels, const std::vector<int> &sources,
              const std::vector<int> &transforms, const Indices &exponents, const Indices &left,
              const Indices &right, const Indices &terminal,
              const std::vector<double> &coefficients, bool finite_difference)
{
  const auto schedule =
      make_gpu_dag_schedule(ns, channels, sources, left, right, terminal, coefficients);
  int np = channels.size(), nv = np + left.size(), stride = 19;
  require(schedule.parents.size() == 2 * left.size(), "each child occurrence must be retained");
  require(schedule.source_powers.size() == channels.size(), "source transpose lost a power");
  std::vector<int> seen(np, 0);
  for (int s = 0; s < ns; ++s)
    for (int e = schedule.source_power_offsets[s]; e < schedule.source_power_offsets[s + 1]; ++e) {
      int p = schedule.source_powers[e];
      require(sources[channels[p]] == s, "source transpose maps wrong channel");
      ++seen[p];
    }
  for (int n : seen) require(n == 1, "duplicate or omitted source power");
  Plan plan{{schedule.parent_offsets.data()},
            {schedule.parents.data()},
            {schedule.siblings.data()},
            {schedule.readout_seed.data()}};
  std::vector<double> vr(nv * stride, -99), vi(vr), ar(nv * stride, -77), ai(ar);
  std::mt19937 rng(332);
  std::uniform_real_distribution<double> d(-.3, .3);
  for (int lane = 0; lane < 17; ++lane) {
    std::vector<Complex> x(ns);
    for (auto &v : x) v = {d(rng), d(rng)};
    if (lane < ns) x[lane] = 0.;
    auto evaluate = [&](const std::vector<Complex> &input, std::vector<Complex> &v) {
      v.assign(nv, {});
      for (int p = 0; p < np; ++p) {
        int c = channels[p];
        Complex z = input[sources[c]];
        if (transforms[c]) z = double(transforms[c]) * std::conj(z);
        v[p] = cpow(z, exponents[p]);
      }
      for (std::size_t n = 0; n < left.size(); ++n) v[np + n] = v[left[n]] * v[right[n]];
      Complex rho{};
      for (std::size_t t = 0; t < terminal.size(); ++t)
        rho += coefficients[t] * (terminal[t] < 0 ? Complex(1) : v[terminal[t]]);
      return rho;
    };
    std::vector<Complex> v;
    auto rho = evaluate(x, v);
    (void) rho;
    std::vector<Complex> ref(nv);
    for (std::size_t t = 0; t < terminal.size(); ++t)
      if (terminal[t] >= 0) ref[terminal[t]] += coefficients[t];
    for (int n = int(left.size()) - 1; n >= 0; --n) {
      Complex seed = ref[np + n];
      ref[left[n]] += seed * std::conj(v[right[n]]);
      ref[right[n]] += seed * std::conj(v[left[n]]);
    }
    for (int i = 0; i < nv; ++i) {
      vr[i * stride + lane] = v[i].real();
      vi[i * stride + lane] = v[i].imag();
    }
    // The same gather helper called by the production Kokkos team kernel.
    // Reverse topological order is a legal sequential realization of its levels.
    for (int i = nv - 1; i >= 0; --i) {
      double rr, ii;
      gather_gpu_dag_adjoint(plan, Span<double>{vr.data()}, Span<double>{vi.data()},
                             Span<double>{ar.data()}, Span<double>{ai.data()}, 0, i, stride, lane,
                             rr, ii);
      ar[i * stride + lane] = rr;
      ai[i * stride + lane] = ii;
      close(rr, ref[i].real());
      close(ii, ref[i].imag());
    }
    // The new team-owned backing store and its resident slice use unit node
    // stride. Exercise both against the same independent reverse scatter.
    for (int pad : {0, 7}) {
      std::vector<double> cr(nv + 2 * pad, -333), ci(cr), car(cr), cai(cr);
      GpuStridedDoubles r{cr.data(), 1, static_cast<std::size_t>(pad)},
          im{ci.data(), 1, static_cast<std::size_t>(pad)},
          dr{car.data(), 1, static_cast<std::size_t>(pad)},
          di{cai.data(), 1, static_cast<std::size_t>(pad)};
      for (int i = 0; i < nv; ++i) {
        r(i) = v[i].real();
        im(i) = v[i].imag();
      }
      for (int i = nv - 1; i >= 0; --i) {
        double rr, ii;
        gather_gpu_dag_adjoint(plan, r, im, dr, di, 0, i, 1, 0, rr, ii);
        dr(i) = rr;
        di(i) = ii;
        close(rr, ref[i].real());
        close(ii, ref[i].imag());
      }
      for (int i = 0; i < pad; ++i)
        require(car[i] == -333 && car[nv + pad + i] == -333,
                "contiguous DAG escaped its center slice");
    }
    std::vector<Complex> source_ref(ns), source_gather(ns);
    auto term_gradient = [&](int p, Complex root) {
      int c = channels[p];
      Complex z = x[sources[c]];
      if (transforms[c]) z = double(transforms[c]) * std::conj(z);
      Complex g = root * std::conj(double(exponents[p]) * cpow(z, exponents[p] - 1));
      if (transforms[c]) g = double(transforms[c]) * std::conj(g);
      return g;
    };
    for (int p = 0; p < np; ++p) source_ref[sources[channels[p]]] += term_gradient(p, ref[p]);
    for (int s = 0; s < ns; ++s)
      for (int e = schedule.source_power_offsets[s]; e < schedule.source_power_offsets[s + 1];
           ++e) {
        int p = schedule.source_powers[e];
        source_gather[s] += term_gradient(p, {ar[p * stride + lane], ai[p * stride + lane]});
      }
    for (int s = 0; s < ns; ++s) {
      close(source_gather[s].real(), source_ref[s].real());
      close(source_gather[s].imag(), source_ref[s].imag());
    }
    if (finite_difference && lane < 3)
      for (int s = 0; s < ns; ++s)
        for (int axis = 0; axis < 2; ++axis) {
          const double h = 1e-6;
          auto xp = x, xm = x;
          Complex step = axis ? Complex(0, h) : Complex(h, 0);
          xp[s] += step;
          xm[s] -= step;
          std::vector<Complex> scratch;
          double fd = (evaluate(xp, scratch).real() - evaluate(xm, scratch).real()) / (2 * h);
          close(fd, axis ? source_gather[s].imag() : source_gather[s].real(), 2e-7);
        }
  }
  for (int i = 0; i < nv; ++i)
    for (int lane = 17; lane < stride; ++lane)
      require(ar[i * stride + lane] == -77 && ai[i * stride + lane] == -77,
              "wrote an inactive lane");
}
int main(int argc, char **argv)
try {
  int cases = 0;
  for (int nodes : {0, 1, 2, 15, 71, 240})
    for (int powers : {1, 3, 9}) {
      Indices channels, p, exponents, l, r, t;
      std::vector<int> src = {0, 0, 1, 1, 2}, tr = {0, 1, -1, 0, 0};
      std::vector<double> w;
      for (int i = 0; i < powers; ++i) {
        channels.push_back(i % src.size());
        exponents.push_back(1 + i % 4);
      }
      std::mt19937 rng(nodes + 117 * powers);
      for (int n = 0; n < nodes; ++n) {
        int a = rng() % (powers + n);
        l.push_back(a);
        r.push_back(n % 3 == 0 ? a : rng() % (powers + n));
      }
      t = {-1, 0, 0};
      w = {.9, .2, -.2};
      for (int i = 0; i < powers + nodes; ++i)
        if (i % 2 == 0 || i + 1 == powers + nodes) {
          t.push_back(i);
          w.push_back((i % 3 - 1) * .35 + .03);
        }
      exercise(4, channels, src, tr, exponents, l, r, t, w, true);
      ++cases;
    }
  exercise(2, {}, {0, 1}, {0, 0}, {}, {}, {}, {-1}, {.4}, true);
  ++cases;
  auto bad = [&](auto call) {
    bool raised = false;
    try {
      call();
    } catch (const std::exception &) {
      raised = true;
    }
    require(raised, "invalid plan was accepted");
  };
  bad([] {
    make_gpu_dag_schedule(1, Indices{0}, std::vector<int>{0}, Indices{1}, Indices{0}, Indices{0},
                          std::vector<double>{1});
  });
  bad([] {
    make_gpu_dag_schedule(1, Indices{0}, std::vector<int>{1}, Indices{}, Indices{}, Indices{0},
                          std::vector<double>{1});
  });
  bad([] {
    make_gpu_dag_schedule(1, Indices{0}, std::vector<int>{0}, Indices{}, Indices{}, Indices{3},
                          std::vector<double>{1});
  });
  bad([] {
    make_gpu_dag_schedule(1, Indices{0}, std::vector<int>{0}, Indices{}, Indices{}, Indices{0},
                          std::vector<double>{std::numeric_limits<double>::infinity()});
  });
#ifdef YE3T_TEST_NATIVE_DAG
  for (int f = 1; f < argc; ++f) {
    auto m = load_native_fixture(argv[f]);
    for (int s = 0; s < m.species_count(); ++s) {
      const auto &sp = m.species(s);
      const auto &p = sp.polynomial;
      const auto binary = lower_gpu_binary_dag(p);
      exercise(sp.source_channels.size(), p.power_channels, sp.full_channel_sources,
               sp.full_channel_transforms, p.power_exponents, binary.left, binary.right,
               binary.monomial_operands, p.monomial_coefficients, false);
      std::cout << "fixture " << argv[f] << " powers=" << p.power_channels.size()
                << " nodes=" << p.binary_node_left.size()
                << " readout_terms=" << p.monomial_nodes.size() << " PASS\n";
      ++cases;
    }
  }
#else
  (void) argc;
  (void) argv;
#endif
  std::cout << cases << " GPU DAG math cases PASS (host arithmetic, not GPU execution)\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
