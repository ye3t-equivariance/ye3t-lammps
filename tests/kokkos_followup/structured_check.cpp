// Real Kokkos test, NOT a mock backend. Build alongside the existing opt-in
// checks with the same Kokkos/backend/compiler as the LAMMPS executable.
// clang-format off
#include <Kokkos_Core.hpp>
#include "ye3t_kokkos_plan.h"
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <vector>
// clang-format on

using Exec = Kokkos::DefaultExecutionSpace;
using namespace YE3T_LAMMPS;
using C = std::complex<double>;
template <class T> Kokkos::View<T *, Exec> upload(const std::vector<T> &values)
{
  Kokkos::View<T *, Exec> result("structured:test", values.size());
  auto host = Kokkos::create_mirror_view(result);
  for (std::size_t i = 0; i < values.size(); ++i) host(i) = values[i];
  Kokkos::deep_copy(result, host);
  return result;
}
void near(double a, double b)
{
  if (!std::isfinite(a) || !std::isfinite(b) ||
      std::abs(a - b) > 1e-10 * (1 + std::abs(a) + std::abs(b)))
    throw std::runtime_error("structured Kokkos numerical mismatch");
}
void block_tiles()
{
  constexpr int count = 37, capacity = 43, monomials = 35;
  YE3TKokkosBlockPlanRecord p;
  p.input_count = 2;
  p.monomial_count = monomials;
  p.output_count = 2;
  p.output_storage_offset = 3;
  std::vector<int> offsets{0}, components, exponents, terms, output_offsets{0};
  std::vector<double> cr, ci;
  C total[2]{};
  for (int m = 0; m < monomials; ++m) {
    components.insert(components.end(), {0, 1});
    exponents.insert(exponents.end(), {1, 1});
    offsets.push_back(components.size());
  }
  for (int o = 0; o < 2; ++o) {
    for (int m = monomials - 1; m >= 0; --m) {
      terms.push_back(m);
      cr.push_back(.01 * (m + 1));
      ci.push_back(.02 * (o + 1));
      total[o] += C(cr.back(), ci.back());
    }
    output_offsets.push_back(terms.size());
  }
  const auto schedule = make_gpu_block_execution_schedule(
      std::vector<YE3TKokkosBlockPlanRecord>{p}, output_offsets, terms, monomials,
      YE3T_BLOCK_DIRECT_INPUT, YE3T_BLOCK_CONJUGATE_HALF_OUTPUT);
  // The actual production view record, not a lookalike, is passed to helpers.
  YE3TKokkosBlockViews<Exec> b;
#define SET(field, values) b.field = upload(values)
  SET(monomial_factor_offsets, offsets);
  SET(monomial_factor_components, components);
  SET(monomial_factor_exponents, exponents);
  SET(coefficient_terms, terms);
  SET(coefficient_real, cr);
  SET(coefficient_imaginary, ci);
  SET(output_coefficient_offsets, output_offsets);
  SET(plan_input_channels, std::vector<int>({0, 1}));
  SET(plan_input_power_offsets, std::vector<int>({0, 2}));
#define S(field) SET(field, schedule.field)
  S(plan_tile_offsets);
  S(tile_monomial_begin);
  S(tile_monomial_count);
  S(tile_output_offsets);
  S(tile_outputs);
  S(tile_coefficient_offsets);
  S(tile_coefficients);
  S(monomial_output_offsets);
  S(monomial_outputs);
  S(monomial_coefficients);
#undef S
#undef SET
  Kokkos::View<double *, Exec> powers_r("powers:r", 4 * capacity),
      powers_i("powers:i", 4 * capacity);
  Kokkos::View<double *, Exec> outputs_r("outputs:r", 5 * capacity),
      outputs_i("outputs:i", 5 * capacity);
  Kokkos::View<double *, Exec> gradients_r("g:r", 2 * capacity), gradients_i("g:i", 2 * capacity);
  Kokkos::deep_copy(outputs_r, -777.);
  Kokkos::deep_copy(outputs_i, -777.);
  auto pr = Kokkos::create_mirror_view(powers_r), pi = Kokkos::create_mirror_view(powers_i);
  for (int lane = 0; lane < count; ++lane) {
    const C x = lane % 3 == 0 ? C{} : C(.1 + .01 * lane, -.2),
            y = lane % 5 == 0 ? C{} : C(-.3, .12);
    const C powers[4] = {C(1.), x, C(1.), y};
    for (int j = 0; j < 4; ++j) {
      pr(j * capacity + lane) = powers[j].real();
      pi(j * capacity + lane) = powers[j].imag();
    }
  }
  Kokkos::deep_copy(powers_r, pr);
  Kokkos::deep_copy(powers_i, pi);
  using Member = typename Kokkos::TeamPolicy<Exec>::member_type;
  constexpr bool host =
      Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename Exec::memory_space>::accessible;
  const int team_size = host ? 1 : 32;
  auto policy = Kokkos::TeamPolicy<Exec>((count + team_size - 1) / team_size, team_size);
  policy = policy.set_scratch_size(
      0, Kokkos::PerTeam(2 * GPU_BLOCK_MONOMIAL_TILE * team_size * sizeof(double)));
  Kokkos::parallel_for(
      "structured_block_tiles", policy, KOKKOS_LAMBDA(const Member &team) {
        double *scratch = static_cast<double *>(team.team_shmem().get_shmem(
            2 * GPU_BLOCK_MONOMIAL_TILE * team.team_size() * sizeof(double)));
        const int lane = team.league_rank() * team.team_size() + team.team_rank();
        if (lane >= count) return;
        const GpuConstStridedDoubles r{powers_r.data(), capacity, static_cast<std::size_t>(lane)};
        const GpuConstStridedDoubles i{powers_i.data(), capacity, static_cast<std::size_t>(lane)};
        const GpuStridedDoubles yr{outputs_r.data(), capacity, static_cast<std::size_t>(lane)};
        const GpuStridedDoubles yi{outputs_i.data(), capacity, static_cast<std::size_t>(lane)};
        const GpuStridedDoubles tr{scratch, static_cast<std::size_t>(team.team_size()),
                                   static_cast<std::size_t>(team.team_rank())};
        const GpuStridedDoubles ti{scratch + GPU_BLOCK_MONOMIAL_TILE * team.team_size(),
                                   static_cast<std::size_t>(team.team_size()),
                                   static_cast<std::size_t>(team.team_rank())};
        gpu_block_forward_tiles(b, p, 0, r, i, yr, yi, tr, ti, false);
        // Roots for the two output components, including a complex seed.
        double roots_r[5] = {0., 0., 0., .7, -.2}, roots_i[5] = {0., 0., 0., .1, .3};
        const GpuStridedDoubles rr{roots_r, 1, 0}, ri{roots_i, 1, 0};
        double gr[2] = {}, gi[2] = {};
        const auto add = [&](int c, double ar, double ai) {
          gr[c] += ar;
          gi[c] += ai;
        };
        gpu_block_adjoint_transpose(b, p, r, i, rr, ri, add, 1., false);
        for (int c = 0; c < 2; ++c) {
          gradients_r(c * capacity + lane) = gr[c];
          gradients_i(c * capacity + lane) = gi[c];
        }
      });
  const auto yr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), outputs_r);
  const auto yi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), outputs_i);
  const auto gr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), gradients_r);
  const auto gi = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), gradients_i);
  const C root = C(.7, .1) * std::conj(total[0]) + C(-.2, .3) * std::conj(total[1]);
  for (int lane = 0; lane < count; ++lane) {
    const C x(pr(capacity + lane), pi(capacity + lane)),
        y(pr(3 * capacity + lane), pi(3 * capacity + lane));
    for (int o = 0; o < 2; ++o) {
      const C z = total[o] * x * y;
      near(yr((3 + o) * capacity + lane), z.real());
      near(yi((3 + o) * capacity + lane), z.imag());
    }
    const C g[2] = {root * std::conj(y), root * std::conj(x)};
    for (int c = 0; c < 2; ++c) {
      near(gr(c * capacity + lane), g[c].real());
      near(gi(c * capacity + lane), g[c].imag());
    }
  }
  for (int o = 0; o < 5; ++o)
    for (int lane = 0; lane < capacity; ++lane)
      if (o < 3 || lane >= count) {
        near(yr(o * capacity + lane), -777.);
        near(yi(o * capacity + lane), -777.);
      }
}
void resident_dag()
{
  const auto s =
      make_gpu_dag_schedule(1, std::vector<int>{0}, std::vector<int>{0}, std::vector<int>{0, 1},
                            std::vector<int>{0, 0}, std::vector<int>{2}, std::vector<double>{1.});
  YE3TKokkosSpeciesViews<Exec> b;
  b.dag_parent_offsets = upload(s.parent_offsets);
  b.dag_parents = upload(s.parents);
  b.dag_siblings = upload(s.siblings);
  b.dag_readout_seed = upload(s.readout_seed);
  Kokkos::View<double *, Exec> output("resident:output", 37);
  using Member = typename Kokkos::TeamPolicy<Exec>::member_type;
  auto policy = Kokkos::TeamPolicy<Exec>(37, Kokkos::AUTO);
  policy = policy.set_scratch_size(0, Kokkos::PerTeam(12 * sizeof(double)));
  Kokkos::parallel_for(
      "resident_dag", policy, KOKKOS_LAMBDA(const Member &team) {
        auto *p = static_cast<double *>(team.team_shmem().get_shmem(12 * sizeof(double)));
        GpuStridedDoubles r{p, 1, 0}, i{p + 3, 1, 0}, ar{p + 6, 1, 0}, ai{p + 9, 1, 0};
        const double x = .01 * team.league_rank();
        if (team.team_rank() == 0) {
          r(0) = x;
          r(1) = x * x;
          r(2) = x * x * x;
          for (int j = 0; j < 3; ++j) i(j) = 0.;
        }
        team.team_barrier();
        for (int j = 2; j >= 0; --j) {
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 1), [&](int) {
            double rr, ii;
            gather_gpu_dag_adjoint(b, r, i, ar, ai, 0, j, 1, 0, rr, ii);
            ar(j) = rr;
            ai(j) = ii;
          });
          team.team_barrier();
        }
        if (team.team_rank() == 0) output(team.league_rank()) = ar(0);
      });
  auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output);
  for (int i = 0; i < 37; ++i) near(host(i), 3 * (.01 * i) * (.01 * i));
}
void harmonic_stream()
{
  // Normalization Y_00=1 matches the pair style's stored source convention.
  const auto plan = upload(std::vector<double>{1., 0., 0., std::sqrt(3.), 0., -std::sqrt(1.5), 0.});
  Kokkos::View<double *, Exec> result("harmonic:result", 18);
  Kokkos::parallel_for(
      "structured_harmonic", Kokkos::RangePolicy<Exec>(0, 1), KOKKOS_LAMBDA(int) {
        const double direction[3] = {0., 0., 1.};
        GpuHarmonicStream<true, decltype(plan)> stream(plan, 1, direction, 2.);
        const auto y0 = stream.advance(1, 0), y1 = stream.advance(1, 1);
        GpuHarmonicStream<false, decltype(plan)> values(plan, 1, direction);
        const auto v0 = values.advance(1, 0), v1 = values.advance(1, 1);
        result(0) = y0.real;
        result(1) = y0.imaginary;
        result(2) = y1.real;
        result(3) = y1.imaginary;
        for (int a = 0; a < 3; ++a) {
          result(4 + a) = y0.gradient_real[a];
          result(7 + a) = y0.gradient_imaginary[a];
          result(10 + a) = y1.gradient_real[a];
          result(13 + a) = y1.gradient_imaginary[a];
        }
        result(16) = v0.real;
        result(17) = v1.imaginary;
      });
  const auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
  for (int j = 0; j < 18; ++j) {
    double expected = 0.;
    if (j == 0 || j == 16) expected = std::sqrt(3.);
    if (j == 10 || j == 14) expected = -std::sqrt(1.5) / 2.;
    near(h(j), expected);
  }
}
int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  int code = 0;
  try {
    block_tiles();
    resident_dag();
    harmonic_stream();
    std::cout << "PASS structured blocks/resident DAG on " << Exec::name() << "\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    code = 1;
  }
  Kokkos::finalize();
  return code;
}
