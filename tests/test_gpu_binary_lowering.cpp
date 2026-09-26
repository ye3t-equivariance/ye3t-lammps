// Host regression for the production upload lowering and owner-gather arithmetic.
// This is not a CUDA/HIP or LAMMPS integration test.
#include "ye3t_gpu_dag_schedule.h"
#include <algorithm>
#include <complex>
#include <iostream>
#include <numeric>
#include <random>
#include <string>

using namespace YE3T_LAMMPS;
using C = std::complex<double>;
using I = std::int64_t;
#ifdef YE3T_TEST_NATIVE_DAG
namespace YE3T_LAMMPS {
YACEModel load_native_fixture(const std::string &);
YACESparsePolynomial compile_native_test_polynomial(YACESparsePolynomial);
YACESparsePolynomial native_test_without_descriptors(const YACESparsePolynomial &,
                                                     const std::vector<int> &);
}    // namespace YE3T_LAMMPS
#endif
void require(bool condition, const char *message)
{
  if (!condition) throw std::runtime_error(message);
}
void near(C a, C b, double tolerance = 3e-12)
{
  if (!std::isfinite(std::abs(a)) || !std::isfinite(std::abs(b)) ||
      std::abs(a - b) > tolerance * (1 + std::abs(a) + std::abs(b)))
    throw std::runtime_error("prefix/binary value or adjoint mismatch");
}
C power(C z, I n)
{
  C value = 1.;
  while (n-- > 0) value *= z;
  return value;
}
template <class T> struct View {
  const T *p;
  const T &operator()(int i) const { return p[i]; }
};
struct ReverseViews {
  View<int> dag_parent_offsets, dag_parents, dag_siblings;
  View<double> dag_readout_seed;
};
struct Evaluation {
  C value{};
  std::vector<C> gradient;
};
Evaluation evaluate(const YACESparsePolynomial &p, const GpuBinaryDag *gpu,
                    const std::vector<C> &input, const std::vector<int> &sources,
                    const std::vector<int> &transforms)
{
  std::vector<C> base, powers, power_gradient(p.power_channels.size());
  for (std::size_t k = 0; k < p.power_channels.size(); ++k) {
    const auto channel = p.power_channels[k];
    C z = input[sources[channel]];
    if (transforms[channel]) z = double(transforms[channel]) * std::conj(z);
    base.push_back(z);
    powers.push_back(power(z, p.power_exponents[k]));
  }
  Evaluation out;
  out.gradient.assign(input.size(), {});
  if (!gpu && !p.binary_dag) {
    // Independent prefix execution, including its explicit unit node.
    std::vector<C> values(p.dag_node_parents.size()), adjoint(values.size());
    if (!values.empty()) values[0] = 1.;
    for (std::size_t node = 1; node < values.size(); ++node)
      values[node] = values[p.dag_node_parents[node]] * powers[p.dag_node_powers[node]];
    for (std::size_t term = 0; term < p.monomial_nodes.size(); ++term) {
      const auto node = p.monomial_nodes[term];
      out.value += p.monomial_coefficients[term] * values[node];
      adjoint[node] += p.monomial_coefficients[term];
    }
    for (std::size_t node = values.size(); node > 1;) {
      --node;
      const auto parent = p.dag_node_parents[node], k = p.dag_node_powers[node];
      power_gradient[k] += adjoint[node] * std::conj(values[parent]);
      adjoint[parent] += adjoint[node] * std::conj(powers[k]);
    }
  } else {
    const auto &left = gpu ? gpu->left : p.binary_node_left;
    const auto &right = gpu ? gpu->right : p.binary_node_right;
    const auto &terminal = gpu ? gpu->monomial_operands : p.monomial_nodes;
    const int np = static_cast<int>(powers.size()), nv = np + static_cast<int>(left.size());
    std::vector<C> values = powers, adjoint(nv);
    for (std::size_t n = 0; n < left.size(); ++n)
      values.push_back(values[left[n]] * values[right[n]]);
    for (std::size_t t = 0; t < terminal.size(); ++t) {
      const auto op = terminal[t];
      out.value += p.monomial_coefficients[t] * (op < 0 ? C(1) : values[op]);
      if (op >= 0) adjoint[op] += p.monomial_coefficients[t];
    }
    if (gpu) {
      const auto reverse =
          make_gpu_dag_schedule(static_cast<int>(input.size()), p.power_channels, sources, left,
                                right, terminal, p.monomial_coefficients);
      ReverseViews plan{{reverse.parent_offsets.data()},
                        {reverse.parents.data()},
                        {reverse.siblings.data()},
                        {reverse.readout_seed.data()}};
      std::vector<double> vr(nv), vi(nv), ar(nv, -999), ai(nv, -999);
      for (int i = 0; i < nv; ++i) {
        vr[i] = values[i].real();
        vi[i] = values[i].imag();
      }
      for (int i = nv; i-- > 0;) {
        double real, imaginary;
        gather_gpu_dag_adjoint(plan, View<double>{vr.data()}, View<double>{vi.data()},
                               View<double>{ar.data()}, View<double>{ai.data()}, 0, i, 1, 0, real,
                               imaginary);
        ar[i] = real;
        ai[i] = imaginary;
      }
      for (int k = 0; k < np; ++k) power_gradient[k] = {ar[k], ai[k]};
    } else {
      for (int n = static_cast<int>(left.size()); n-- > 0;) {
        adjoint[left[n]] += adjoint[np + n] * std::conj(values[right[n]]);
        adjoint[right[n]] += adjoint[np + n] * std::conj(values[left[n]]);
      }
      std::copy_n(adjoint.begin(), np, power_gradient.begin());
    }
  }
  for (std::size_t k = 0; k < powers.size(); ++k) {
    const auto channel = p.power_channels[k];
    C g = power_gradient[k] *
        std::conj(double(p.power_exponents[k]) * power(base[k], p.power_exponents[k] - 1));
    if (transforms[channel]) g = double(transforms[channel]) * std::conj(g);
    out.gradient[sources[channel]] += g;
  }
  return out;
}
void exercise(const YACESparsePolynomial &p, int source_count, const std::vector<int> &sources,
              const std::vector<int> &transforms, bool finite_difference = true)
{
  const auto before = p;
  const auto gpu = lower_gpu_binary_dag(p);
  // Lowering is const and affects only operand tables. Input powers, coefficients,
  // descriptor ownership and the prefix-vs-binary CPU choice stay unchanged.
  require(p.binary_dag == before.binary_dag && p.monomial_nodes == before.monomial_nodes &&
              p.dag_node_parents == before.dag_node_parents &&
              p.dag_node_powers == before.dag_node_powers &&
              p.binary_node_left == before.binary_node_left &&
              p.binary_node_right == before.binary_node_right &&
              p.power_channels == before.power_channels &&
              p.power_exponents == before.power_exponents &&
              p.factor_offsets == before.factor_offsets &&
              p.factor_indices == before.factor_indices &&
              p.factor_exponents == before.factor_exponents &&
              p.descriptor_offsets == before.descriptor_offsets &&
              p.descriptor_terms == before.descriptor_terms &&
              p.descriptor_coefficients == before.descriptor_coefficients &&
              p.monomial_coefficients == before.monomial_coefficients,
          "lowering mutated the model");
  if (p.binary_dag)
    require(gpu.left == p.binary_node_left && gpu.right == p.binary_node_right &&
                gpu.monomial_operands == p.monomial_nodes,
            "existing binary DAG changed");
  else if (!p.dag_node_parents.empty())
    require(gpu.left.size() == p.dag_node_parents.size() - 1 - p.dag_root_node_count,
            "lowering inflated the product count");
  std::mt19937 rng(1257);
  std::uniform_real_distribution<double> dist(-.45, .45);
  for (int sample = 0; sample < 8; ++sample) {
    std::vector<C> input(source_count);
    for (auto &v : input) v = {dist(rng), dist(rng)};
    if (source_count && sample < 2) input[sample % source_count] = 0.;
    const auto reference = evaluate(p, nullptr, input, sources, transforms);
    const auto actual = evaluate(p, &gpu, input, sources, transforms);
    near(reference.value, actual.value);
    for (int s = 0; s < source_count; ++s) near(reference.gradient[s], actual.gradient[s]);
    if (finite_difference && sample < 2)
      for (int s = 0; s < source_count; ++s)
        for (int axis = 0; axis < 2; ++axis) {
          const double h = 1e-6;
          auto plus = input, minus = input;
          C step = axis ? C(0, h) : C(h, 0);
          plus[s] += step;
          minus[s] -= step;
          const double fd = (evaluate(p, nullptr, plus, sources, transforms).value.real() -
                             evaluate(p, nullptr, minus, sources, transforms).value.real()) /
              (2 * h);
          near(fd, axis ? actual.gradient[s].imag() : actual.gradient[s].real(), 3e-7);
        }
  }
}
YACESparsePolynomial make_prefix(int powers, int products)
{
  YACESparsePolynomial p;
  p.dag_node_parents = {-1};
  p.dag_node_powers = {-1};
  p.dag_root_node_count = powers;
  for (int k = 0; k < powers; ++k) {
    p.power_channels.push_back(k % 6);
    p.power_exponents.push_back(1 + k % 5);
    p.dag_node_parents.push_back(0);
    p.dag_node_powers.push_back(k);
  }
  std::mt19937 rng(powers * 91 + products);
  for (int n = 0; n < products; ++n) {
    p.dag_node_parents.push_back(1 + rng() % (p.dag_node_parents.size() - 1));
    p.dag_node_powers.push_back(rng() % powers);
  }
  for (std::size_t n = 0; n < p.dag_node_parents.size(); ++n) {
    p.monomial_nodes.push_back(n);
    p.monomial_coefficients.push_back(.03 * (int(n % 9) - 4));
  }
  p.monomial_nodes.push_back(0);
  p.monomial_coefficients.push_back(.75);
  return p;
}
int main(int argc, char **argv)
try {
  const std::vector<int> sources{0, 0, 1, 1, 2, 2}, transforms{0, 1, 0, -1, 0, -1};
  int cases = 0;
  for (int np : {1, 3, 11})
    for (int nn : {0, 1, 2, 12, 89}) {
      auto p = make_prefix(np, nn);
      exercise(p, 3, sources, transforms);
      ++cases;
      const auto b = lower_gpu_binary_dag(p);
      p.binary_dag = true;
      p.binary_node_left = b.left;
      p.binary_node_right = b.right;
      p.monomial_nodes = b.monomial_operands;
      exercise(p, 3, sources, transforms);
      ++cases;
    }
  exercise(YACESparsePolynomial{}, 3, sources, transforms);
  ++cases;    // empty DIRECT residual
  auto constant = make_prefix(0, 0);
  exercise(constant, 3, sources, transforms);
  ++cases;
  auto repeated = make_prefix(1, 0);
  repeated.dag_node_parents = {-1, 0, 0, 1, 3};
  repeated.dag_node_powers = {-1, 0, 0, 0, 0};
  repeated.dag_root_node_count = 2;
  repeated.monomial_nodes = {0, 1, 2, 3, 4, 4};
  repeated.monomial_coefficients = {.4, .2, .3, -.6, .7, -.1};
  exercise(repeated, 3, sources, transforms);
  ++cases;
  auto high = make_prefix(1, 0);
  high.power_exponents = {32};
  exercise(high, 3, sources, transforms);
  ++cases;
  int bad_cases = 0;
  auto bad = [&](YACESparsePolynomial p) {
    bool rejected = false;
    try {
      lower_gpu_binary_dag(p);
    } catch (const std::exception &) {
      rejected = true;
    }
    require(rejected, "malformed DAG was accepted");
    ++bad_cases;
  };
  auto p = make_prefix(2, 2), q = p;
  q.dag_node_parents[0] = 0;
  bad(q);
  q = p;
  q.dag_node_powers[0] = 0;
  bad(q);
  q = p;
  q.dag_node_parents[3] = 3;
  bad(q);
  q = p;
  q.dag_node_parents[3] = -1;
  bad(q);
  q = p;
  q.dag_node_powers[3] = 2;
  bad(q);
  q = p;
  q.dag_node_powers[3] = -1;
  bad(q);
  q = p;
  q.dag_root_node_count = 1;
  bad(q);
  q = p;
  q.dag_root_node_count = -1;
  bad(q);
  q = p;
  q.dag_root_node_count = 100;
  bad(q);
  q = p;
  q.monomial_nodes[0] = 99;
  bad(q);
  q = p;
  q.monomial_nodes[0] = -1;
  bad(q);
  q = p;
  q.monomial_nodes.pop_back();
  bad(q);
  q = p;
  q.dag_node_powers.pop_back();
  bad(q);
  q = p;
  q.power_exponents.pop_back();
  bad(q);
  q = p;
  q.binary_node_left = {0};
  bad(q);
  q = p;
  q.dag_node_parents.clear();
  q.dag_node_powers.clear();
  bad(q);
  const auto binary = lower_gpu_binary_dag(p);
  p.binary_dag = true;
  p.binary_node_left = binary.left;
  p.binary_node_right = binary.right;
  p.monomial_nodes = binary.monomial_operands;
  q = p;
  q.binary_node_left[0] = 2;
  bad(q);
  q = p;
  q.binary_node_right.pop_back();
  bad(q);
  q = p;
  q.monomial_nodes[0] = -2;
  bad(q);
  q = p;
  q.monomial_nodes[0] = 99;
  bad(q);
#ifdef YE3T_TEST_NATIVE_DAG
  // The actual production compiler chooses PREFIX in these equal-product-count
  // cases. The old Kokkos admission check would reject each nonempty plan.
  for (int rank : {1, 2, 3, 8, 32}) {
    YACESparsePolynomial raw;
    raw.maximum_rank = rank;
    raw.factor_offsets = {0, 0};
    raw.monomial_coefficients = {.3};    // constant
    for (int c = 0; c < 3; ++c) {
      raw.factor_indices.push_back(c);
      raw.factor_exponents.push_back(rank);
      raw.factor_offsets.push_back(raw.factor_indices.size());
      raw.monomial_coefficients.push_back(.2 * (c + 1));
    }
    if (rank >= 2) {
      raw.factor_indices.insert(raw.factor_indices.end(), {0, 1});
      raw.factor_exponents.insert(raw.factor_exponents.end(), {rank - 1, 1});
      raw.factor_offsets.push_back(raw.factor_indices.size());
      raw.monomial_coefficients.push_back(-.4);
    }
    raw.descriptor_offsets = {0, static_cast<I>(raw.monomial_coefficients.size())};
    raw.descriptor_coefficients = raw.monomial_coefficients;
    raw.descriptor_terms.resize(raw.monomial_coefficients.size());
    std::iota(raw.descriptor_terms.begin(), raw.descriptor_terms.end(), 0);
    auto compiled = compile_native_test_polynomial(raw);
    require(!compiled.binary_dag, "native tie fixture no longer selects prefix");
    exercise(compiled, 3, sources, transforms);
    ++cases;
    std::cout << "production compiler rank " << rank << ": prefix -> GPU binary PASS\n";
    auto residual = native_test_without_descriptors(compiled, {});
    require(!residual.binary_dag, "residual no longer selects prefix");
    exercise(residual, 3, sources, transforms);
    ++cases;
    auto empty = native_test_without_descriptors(compiled, {0});
    require(empty.monomial_coefficients.empty(), "fully covered residual not empty");
    exercise(empty, 3, sources, transforms);
    ++cases;
  }
  // A genuine partial descriptor removal: one descriptor moves to a specialized
  // route while the nonempty prefix residual retains only the other coefficients.
  YACESparsePolynomial raw;
  raw.factor_offsets = {0, 2, 4};
  raw.factor_indices = {0, 1, 0, 2};
  raw.factor_exponents = {1, 1, 1, 1};
  raw.monomial_coefficients = {.2, .5};
  raw.descriptor_offsets = {0, 1, 2};
  raw.descriptor_terms = {0, 1};
  raw.descriptor_coefficients = {.2, .5};
  auto full = compile_native_test_polynomial(raw);
  auto residual = native_test_without_descriptors(full, {0});
  require(!residual.binary_dag && residual.monomial_coefficients == std::vector<double>{.5},
          "partial residual fixture");
  exercise(residual, 3, sources, transforms);
  ++cases;
  std::cout << "production partial descriptor residual: prefix -> GPU binary PASS\n";
  for (int file = 1; file < argc; ++file) {
    const auto m = load_native_fixture(argv[file]);
    for (int s = 0; s < m.species_count(); ++s) {
      const auto &sp = m.species(s);
      exercise(sp.polynomial, sp.source_channels.size(), sp.full_channel_sources,
               sp.full_channel_transforms, false);
      ++cases;
      std::cout << "native fixture " << argv[file] << ": "
                << (sp.polynomial.binary_dag ? "binary unchanged" : "prefix lowered") << " PASS\n";
    }
  }
#else
  (void) argc;
  (void) argv;
#endif
  std::cout << cases << " prefix/binary lowering arithmetic cases and " << bad_cases
            << " malformed-table cases PASS (host only)\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
