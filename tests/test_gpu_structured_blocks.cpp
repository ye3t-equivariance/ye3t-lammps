#include "ye3t_gpu_block_schedule.h"
#include <complex>
#include <iostream>
#include <random>
#include <string>
using namespace YE3T_LAMMPS;
using C = std::complex<double>;
template <class T> struct Vec : std::vector<T> {
  using std::vector<T>::vector;
  Vec() = default;
  Vec(const std::vector<T> &v) : std::vector<T>(v) {}
  const T &operator()(int i) const { return this->at(i); }
};
struct Record {
  int monomial_begin = 0, monomial_count = 0, output_begin = 0, output_count = 0;
  int output_L = 0, flags = 0, input_begin = 0, output_storage_offset = 0;
};
struct Block {
  Vec<int> monomial_factor_offsets{0}, monomial_factor_components, monomial_factor_exponents;
  Vec<int> plan_input_power_offsets, plan_input_channels, output_coefficient_offsets{0},
      coefficient_terms;
  Vec<double> coefficient_real, coefficient_imaginary;
  Vec<int> plan_tile_offsets, tile_monomial_begin, tile_monomial_count, tile_output_offsets,
      tile_outputs, tile_coefficient_offsets, tile_coefficients, monomial_output_offsets,
      monomial_outputs, monomial_coefficients;
  void assign(const GpuBlockExecutionSchedule &s)
  {
#define COPY(x) x = s.x
    COPY(plan_tile_offsets);
    COPY(tile_monomial_begin);
    COPY(tile_monomial_count);
    COPY(tile_output_offsets);
    COPY(tile_outputs);
    COPY(tile_coefficient_offsets);
    COPY(tile_coefficients);
    COPY(monomial_output_offsets);
    COPY(monomial_outputs);
    COPY(monomial_coefficients);
#undef COPY
  }
};
void check(bool c, const char *s)
{
  if (!c) throw std::runtime_error(s);
}
void near(double a, double b, double tol = 3e-11)
{
  if (!std::isfinite(a) || !std::isfinite(b) ||
      std::abs(a - b) > tol * (1 + std::abs(a) + std::abs(b)))
    throw std::runtime_error("block numeric mismatch " + std::to_string(a) + " " +
                             std::to_string(b));
}
C power(C x, int n)
{
  C v = 1.;
  while (n-- > 0) v *= x;
  return v;
}

// Output-major independent reference, differentiating products by omission.
std::vector<C> reference(const Block &b, const Record &p, const std::vector<C> &x)
{
  std::vector<C> y(p.output_count);
  for (int c = 0; c < p.output_count; ++c) {
    if ((p.flags & 2) && c % (2 * p.output_L + 1) < p.output_L) continue;
    for (int j = b.output_coefficient_offsets[c]; j < b.output_coefficient_offsets[c + 1]; ++j) {
      int m = b.coefficient_terms[j];
      C v = 1.;
      for (int f = b.monomial_factor_offsets[m]; f < b.monomial_factor_offsets[m + 1]; ++f)
        v *= power(x[b.monomial_factor_components[f]], b.monomial_factor_exponents[f]);
      y[c] += C(b.coefficient_real[j], b.coefficient_imaginary[j]) * v;
    }
  }
  if (p.flags & 2)
    for (int copy = 0; copy < p.output_count / (2 * p.output_L + 1); ++copy)
      for (int m = 1; m <= p.output_L; ++m) {
        int a = copy * (2 * p.output_L + 1) + p.output_L - m, bb = a + 2 * m;
        y[a] = (m % 2 ? -1. : 1.) * std::conj(y[bb]);
      }
  return y;
}
void exercise(int support, int monomials, bool half, bool complex_coeff)
{
  std::mt19937 gen(712 + support + monomials);
  std::uniform_real_distribution<double> dist(-.6, .6);
  Block b;
  Record p;
  p.monomial_count = monomials;
  p.output_L = half ? 2 : 0;
  p.output_count = half ? 10 : 7;
  p.flags = (half ? 2 : 0) | (complex_coeff ? 0 : 1);
  for (int c = 0; c < support; ++c) {
    b.plan_input_channels.push_back(c);
    b.plan_input_power_offsets.push_back(c * 4);
  }
  for (int m = 0; m < monomials; ++m) {
    // Every factor support is distinct; repeated powers remain compressed.
    int count = m % 3 == 0 ? support : std::max(1, support / 2);
    for (int c = 0; c < count; ++c) {
      b.monomial_factor_components.push_back(c);
      b.monomial_factor_exponents.push_back(1 + int(gen() % 3));
    }
    b.monomial_factor_offsets.push_back(b.monomial_factor_components.size());
  }
  for (int c = 0; c < p.output_count; ++c) {
    for (int m = monomials - 1; m >= 0; --m)
      if ((m + c) % 3 || m == 0) {
        b.coefficient_terms.push_back(m);
        b.coefficient_real.push_back(dist(gen));
        b.coefficient_imaginary.push_back(complex_coeff ? dist(gen) : 0.);
        if (m == 0) {
          b.coefficient_terms.push_back(m);
          b.coefficient_real.push_back(.125);
          b.coefficient_imaginary.push_back(0.);
        }
      }
    b.output_coefficient_offsets.push_back(b.coefficient_terms.size());
  }
  b.assign(make_gpu_block_execution_schedule(std::vector<Record>{p}, b.output_coefficient_offsets,
                                             b.coefficient_terms, monomials, 4, 2));
  check(b.tile_monomial_count.empty() ||
            *std::max_element(b.tile_monomial_count.begin(), b.tile_monomial_count.end()) <= 16,
        "tile exceeds bound");
  constexpr int stride = 5, lane = 2;
  std::vector<double> pr(support * 4 * stride, -777), pi(pr), out_r(p.output_count * stride, -777),
      out_i(out_r), tmp_r(16 * stride, -777), tmp_i(tmp_r);
  GpuStridedDoubles vr{pr.data(), stride, lane}, vi{pi.data(), stride, lane},
      yr{out_r.data(), stride, lane}, yi{out_i.data(), stride, lane},
      tr{tmp_r.data(), stride, lane}, ti{tmp_i.data(), stride, lane};
  std::vector<C> x(support), roots(p.output_count);
  for (auto &r : roots) r = {dist(gen), dist(gen)};
  for (int trial = 0; trial < 4; ++trial) {
    for (auto &v : x) v = {dist(gen), dist(gen)};
    if (trial > 0) x[0] = 0.;
    if (trial > 1 && support > 1) x[1] = 0.;
    if (trial == 3)
      for (auto &v : x) v = C(1., 0.);
    for (int c = 0; c < support; ++c)
      for (int e = 0; e < 4; ++e) {
        C v = power(x[c], e);
        vr(c * 4 + e) = v.real();
        vi(c * 4 + e) = v.imag();
      }
    gpu_block_forward_tiles(b, p, 0, vr, vi, yr, yi, tr, ti, !complex_coeff);
    if (half)
      for (int copy = 0; copy < 2; ++copy)
        for (int m = 1; m <= 2; ++m) {
          int a = copy * 5 + 2 - m, bb = a + 2 * m;
          yr(a) = (m % 2 ? -1. : 1.) * yr(bb);
          yi(a) = -(m % 2 ? -1. : 1.) * yi(bb);
        }
    auto y = reference(b, p, x);
    for (int c = 0; c < p.output_count; ++c) {
      near(yr(c), y[c].real());
      near(yi(c), y[c].imag());
    }
    std::vector<double> rr(p.output_count), ri(rr);
    auto folded = roots;
    if (half)
      for (int copy = 0; copy < 2; ++copy)
        for (int m = 1; m <= 2; ++m) {
          int a = copy * 5 + 2 - m, bb = a + 2 * m;
          folded[bb] += (m % 2 ? -1. : 1.) * std::conj(folded[a]);
          folded[a] = 0.;
        }
    for (int c = 0; c < p.output_count; ++c) {
      rr[c] = folded[c].real();
      ri[c] = folded[c].imag();
    }
    GpuStridedDoubles rrv{rr.data(), 1, 0}, riv{ri.data(), 1, 0};
    std::vector<C> g(support), gr(support);
    const auto add = [&](int c, double r, double i) {
      g[c] += C(r, i);
    };
    gpu_block_adjoint_transpose(b, p, vr, vi, rrv, riv, add, .7, !complex_coeff);
    for (int c = 0; c < p.output_count; ++c)
      for (int j = b.output_coefficient_offsets[c]; j < b.output_coefficient_offsets[c + 1]; ++j) {
        int m = b.coefficient_terms[j], f0 = b.monomial_factor_offsets[m],
            f1 = b.monomial_factor_offsets[m + 1];
        for (int f = f0; f < f1; ++f) {
          int target = b.monomial_factor_components[f], e = b.monomial_factor_exponents[f];
          C d = double(e) * power(x[target], e - 1);
          for (int ff = f0; ff < f1; ++ff)
            if (ff != f)
              d *= power(x[b.monomial_factor_components[ff]], b.monomial_factor_exponents[ff]);
          gr[target] +=
              .7 * folded[c] * std::conj(C(b.coefficient_real[j], b.coefficient_imaginary[j]) * d);
        }
      }
    for (int c = 0; c < support; ++c) {
      near(g[c].real(), gr[c].real());
      near(g[c].imag(), gr[c].imag());
    }
    if (trial < 3 && support <= 8)
      for (int c = 0; c < support; ++c)
        for (int axis = 0; axis < 2; ++axis) {
          auto xp = x, xm = x;
          C h = axis ? C(0, 1e-6) : C(1e-6, 0);
          xp[c] += h;
          xm[c] -= h;
          auto yp = reference(b, p, xp), ym = reference(b, p, xm);
          double fd = 0;
          for (int o = 0; o < p.output_count; ++o)
            fd += .7 * (std::conj(roots[o]) * (yp[o] - ym[o])).real() / 2e-6;
          near(axis ? g[c].imag() : g[c].real(), fd, 2e-7);
        }
  }
  for (int v = 0; v < p.output_count; ++v)
    for (int ll = 0; ll < stride; ++ll)
      if (ll != lane)
        check(out_r[v * stride + ll] == -777 && out_i[v * stride + ll] == -777,
              "overwrote another center");
}
void exercise_offsets_and_constants()
{
  Block b;
  Record constant;
  constant.monomial_count = 1;
  constant.output_count = 1;
  constant.output_storage_offset = 3;
  Record mixed;
  mixed.monomial_begin = 1;
  mixed.monomial_count = 2;
  mixed.output_begin = 1;
  mixed.output_count = 2;
  mixed.output_storage_offset = 5;
  mixed.input_begin = 1;
  Record identity;
  identity.monomial_begin = 3;
  identity.output_begin = 3;
  identity.output_count = 1;
  identity.flags = 4;
  b.monomial_factor_offsets = {0, 0, 1, 3};
  b.monomial_factor_components = {0, 0, 1};
  b.monomial_factor_exponents = {2, 1, 1};
  b.plan_input_power_offsets = {0, 0, 3};
  b.plan_input_channels = {999, 3, 8};
  b.output_coefficient_offsets = {0, 1, 3, 4, 4};
  b.coefficient_terms = {0, 0, 1, 1};
  b.coefficient_real = {2., 3., -4., 5.};
  b.coefficient_imaginary = {0., .2, -.1, .4};
  b.assign(make_gpu_block_execution_schedule(std::vector<Record>{constant, mixed, identity},
                                             b.output_coefficient_offsets, b.coefficient_terms, 3,
                                             4, 2));
  double pr[6], pi[6], yr[8] = {}, yi[8] = {}, tr[16], ti[16];
  const C x(.3, .2), y(-.7, .1), a(3., .2), bc(-4., -.1), c(5., .4);
  for (int k = 0; k < 3; ++k) {
    const C xx = power(x, k), yy = power(y, k);
    pr[k] = xx.real();
    pi[k] = xx.imag();
    pr[3 + k] = yy.real();
    pi[3 + k] = yy.imag();
  }
  const GpuStridedDoubles r{pr, 1, 0}, i{pi, 1, 0}, out_r{yr, 1, 0}, out_i{yi, 1, 0},
      tmp_r{tr, 1, 0}, tmp_i{ti, 1, 0};
  gpu_block_forward_tiles(b, constant, 0, r, i, out_r, out_i, tmp_r, tmp_i, false);
  gpu_block_forward_tiles(b, mixed, 1, r, i, out_r, out_i, tmp_r, tmp_i, false);
  near(yr[3], 2.);
  near(yi[3], 0.);
  const C z0 = a * x * x + bc * x * y, z1 = c * x * y;
  near(yr[5], z0.real());
  near(yi[5], z0.imag());
  near(yr[6], z1.real());
  near(yi[6], z1.imag());
  double rr[8] = {}, ri[8] = {};
  rr[3] = 1.;
  rr[5] = .7;
  ri[5] = .1;
  rr[6] = -.2;
  ri[6] = .3;
  C g[10]{};
  const auto add = [&](int channel, double ar, double ai) {
    check(channel >= 0 && channel < 10, "source binding index");
    g[channel] += C(ar, ai);
  };
  const GpuStridedDoubles root_r{rr, 1, 0}, root_i{ri, 1, 0};
  gpu_block_adjoint_transpose(b, constant, r, i, root_r, root_i, add, 1., false);
  gpu_block_adjoint_transpose(b, mixed, r, i, root_r, root_i, add, 1., false);
  const C gx = C(.7, .1) * std::conj(2. * a * x + bc * y) + C(-.2, .3) * std::conj(c * y);
  const C gy = C(.7, .1) * std::conj(bc * x) + C(-.2, .3) * std::conj(c * x);
  for (int k = 0; k < 10; ++k) {
    const C expected = k == 3 ? gx : (k == 8 ? gy : C{});
    near(g[k].real(), expected.real());
    near(g[k].imag(), expected.imag());
  }
}

int main()
try {
  exercise_offsets_and_constants();
  int n = 0;
  for (int support : {1, 2, 4, 8, 17, 32, 33, 65})
    for (int m : {0, 1, 15, 16, 17, 67})
      for (bool half : {false, true})
        for (bool complex_coeff : {false, true}) {
          exercise(support, m, half, complex_coeff);
          ++n;
        }
  Record bad;
  bad.output_count = 1;
  bad.monomial_count = 1;
  bool caught = false;
  try {
    make_gpu_block_execution_schedule(std::vector<Record>{bad}, std::vector<int>{0, 1},
                                      std::vector<int>{2}, 1, 4, 2);
  } catch (const std::invalid_argument &) {
    caught = true;
  }
  check(caught, "invalid monomial accepted");
  // Identity plans and disjoint plan ranges do not acquire fabricated tiles.
  Record id;
  id.flags = 4;
  id.output_count = 1;
  auto identity = make_gpu_block_execution_schedule(std::vector<Record>{id}, std::vector<int>{0, 0},
                                                    std::vector<int>{}, 0, 4, 2);
  check(identity.plan_tile_offsets == std::vector<int>({0, 0}), "identity acquired work");
  std::cout << n
            << " structured block cases PASS; forward, transpose, zero-safe VJP, finite "
               "differences, tails, >32-support fallback\n";
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
