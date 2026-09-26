// Opt-in Kokkos compilation and execution checks. They are not full
// LAMMPS/MPI validation.
// clang-format off
#include <Kokkos_Core.hpp>
#include "ye3t_kokkos_step_state.h"
#include "ye3t_tagged_cauchy_kokkos_plan.h"
#include "ye3t_gpu_dag_schedule.h"
#include "gpu_tagged_test_support.h"
#include "gpu_tagged_source_reference.h"
#include <array>
#include <iostream>
#include <stdexcept>
// clang-format on

using Exec = Kokkos::DefaultExecutionSpace;
void check(bool condition, const char *message)
{
  if (!condition) throw std::runtime_error(message);
}
void near(double a, double b)
{
  check(std::isfinite(a) && std::isfinite(b) &&
            std::abs(a - b) <= 2e-10 * (1 + std::abs(a) + std::abs(b)),
        "Kokkos numeric mismatch");
}
struct Tally {
  double energy = 0.0;
  double virial[6] = {};
  KOKKOS_INLINE_FUNCTION void operator+=(const Tally &other)
  {
    energy += other.energy;
    for (int i = 0; i < 6; ++i) virial[i] += other.virial[i];
  }
};
struct ReduceTally {
  using value_type = Tally;
  KOKKOS_INLINE_FUNCTION void init(Tally &value) const { value = Tally{}; }
  KOKKOS_INLINE_FUNCTION void join(Tally &a, const Tally &b) const { a += b; }
  KOKKOS_INLINE_FUNCTION void operator()(int i, Tally &value) const
  {
    value.energy += i + .25;
    for (int axis = 0; axis < 6; ++axis) value.virial[axis] += (axis + 1) * (i + .5);
  }
};
void check_state()
{
  KokkosStepState<Exec, Tally> state;
  Kokkos::View<std::int64_t *, Exec> counts("counts", 37), offsets("offsets", 38);
  Kokkos::View<int *, Exec> status("status", 1);
  for (int step = 0; step < 3; ++step) {
    state.reset();
    double expected_energy = 0.0, expected_virial[6] = {}, expected_imag = 0.0;
    for (int chunk = 0; chunk < 5; ++chunk) {
      const int n = chunk == 4 ? 0 : 5 + 7 * chunk, code = (chunk % 2) * 8;
      const bool global = (chunk % 2) == 0;
      Kokkos::deep_copy(Exec{}, status, code);
      Kokkos::parallel_reduce(
          "counts", Kokkos::RangePolicy<Exec>(0, n),
          KOKKOS_LAMBDA(int i, int &maximum) {
            const int count = (i + chunk) % 7;
            counts(i) = count;
            if (count > maximum) maximum = count;
          },
          Kokkos::Max<int, typename Exec::memory_space>(state.maximum_edges));
      Kokkos::parallel_scan("scan", Kokkos::RangePolicy<Exec>(0, n + 1),
                            KokkosScanEdgesSummary<Exec>{counts, offsets, n, state.summary_view(),
                                                         status, state.maximum_edges, n > 0});
      const auto summary = state.read_edge_summary();
      int total = 0, maximum = 0;
      for (int i = 0; i < n; ++i) {
        const int v = (i + chunk) % 7;
        total += v;
        maximum = std::max(maximum, v);
      }
      check(summary.edge_count == total && summary.maximum_edges == maximum &&
                summary.status == code,
            "combined edge summary mismatch");
      if (global) {
        Kokkos::parallel_reduce("ev", Kokkos::RangePolicy<Exec>(0, n), ReduceTally{},
                                state.ev_chunk);
        for (int i = 0; i < n; ++i) {
          expected_energy += i + .25;
          for (int j = 0; j < 6; ++j) expected_virial[j] += (j + 1) * (i + .5);
        }
      }
      const double imag = .125 * (chunk + 1);
      Kokkos::deep_copy(Exec{}, state.imaginary_chunk, imag);
      state.accumulate(global, true);
      expected_imag = std::max(expected_imag, imag);
    }
    auto totals = state.read_totals();
    near(totals.ev.energy, expected_energy);
    near(totals.maximum_imaginary, expected_imag);
    for (int j = 0; j < 6; ++j) near(totals.ev.virial[j], expected_virial[j]);
  }
  state.reset();
  state.accumulate(false, false);
  auto empty = state.read_totals();
  near(empty.ev.energy, 0.0);
  near(empty.maximum_imaginary, 0.0);
}
struct DagViews {
  Kokkos::View<int *, Exec> dag_parent_offsets, dag_parents, dag_siblings;
  Kokkos::View<double *, Exec> dag_readout_seed;
};
template <class T> Kokkos::View<T *, Exec> upload(const std::vector<T> &data)
{
  Kokkos::View<T *, Exec> view("test", data.size());
  auto host = Kokkos::create_mirror_view(view);
  for (std::size_t i = 0; i < data.size(); ++i) host(i) = data[i];
  Kokkos::deep_copy(view, host);
  return view;
}
void check_dag()
{
  // x -> x*x -> x*x*x, with two distinct occurrences of the first child.
  const auto schedule =
      make_gpu_dag_schedule(1, std::vector<int>{0}, std::vector<int>{0}, std::vector<int>{0, 1},
                            std::vector<int>{0, 0}, std::vector<int>{2}, std::vector<double>{1.0});
  const DagViews plan{upload(schedule.parent_offsets), upload(schedule.parents),
                      upload(schedule.siblings), upload(schedule.readout_seed)};
  constexpr int stride = 23, lanes = 17;
  Kokkos::View<double *, Exec> real("real", 3 * stride), imaginary("imag", 3 * stride),
      ar("ar", 3 * stride), ai("ai", 3 * stride);
  Kokkos::deep_copy(real, 2.0);
  Kokkos::deep_copy(imaginary, 0.0);
  Kokkos::deep_copy(ar, -77.0);
  Kokkos::deep_copy(ai, -77.0);
  using Member = typename Kokkos::TeamPolicy<Exec>::member_type;
  Kokkos::parallel_for(
      "owner_gather", Kokkos::TeamPolicy<Exec>(lanes, Kokkos::AUTO),
      KOKKOS_LAMBDA(const Member &team) {
        const int lane = team.league_rank();
        if (team.team_rank() == 0) {
          real(stride + lane) = 4.0;
          real(2 * stride + lane) = 8.0;
        }
        team.team_barrier();
        for (int value = 2; value >= 0; --value) {
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 1), [&](int) {
            double rr, ii;
            gather_gpu_dag_adjoint(plan, real, imaginary, ar, ai, 0, value, stride, lane, rr, ii);
            ar(value * stride + lane) = rr;
            ai(value * stride + lane) = ii;
          });
          team.team_barrier();
        }
      });
  const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ar);
  for (int lane = 0; lane < lanes; ++lane) near(host(lane), 12.0);
  for (int lane = lanes; lane < stride; ++lane) near(host(lane), -77.0);
}
void check_tagged()
{
  for (bool v3 : {false, true})
    for (int l : {0, 1, 4, 8}) {
      const auto model = make_model(l, v3);
      TaggedCauchyKokkosPlan<Exec> uploaded;
      uploaded.upload(model, 1U << 20);
      const auto plan = uploaded.views();
      const HostSource host_plan(model);
      constexpr int edges = 31;
      const int n = plan.total_component_count;
      Kokkos::View<double **, Kokkos::LayoutRight, Exec> output("output", edges, 5 * n + 3);
      Kokkos::parallel_for(
          "tagged_source", Kokkos::RangePolicy<Exec>(0, edges), KOKKOS_LAMBDA(int edge) {
            const int species = edge % 3;
            const double dx = .31 + .13 * edge, dy = -.49, dz = .87;
            double values[TAGGED_KOKKOS_MAX_COMPONENTS], gx[TAGGED_KOKKOS_MAX_COMPONENTS],
                gy[TAGGED_KOKKOS_MAX_COMPONENTS], gz[TAGGED_KOKKOS_MAX_COMPONENTS];
            tagged_kokkos_edge_components(plan, species, dx, dy, dz, true, values, gx, gy, gz);
            for (int i = 0; i < n; ++i) {
              output(edge, i) = values[i];
              output(edge, n + i) = gx[i];
              output(edge, 2 * n + i) = gy[i];
              output(edge, 3 * n + i) = gz[i];
            }
            // Exercise the no-gradient specialization too, not just its C++ parsing.
            tagged_kokkos_edge_components(plan, species, dx, dy, dz, false, values, nullptr,
                                          nullptr, nullptr);
            for (int i = 0; i < n; ++i) output(edge, 4 * n + i) = output(edge, i) - values[i];
            if (plan.physical_image_v3) {
              for (int i = 0; i < n; ++i) values[i] = .17 * (i % 7 - 3);
              double g[3];
              gpu_tagged_edge_vjp<TaggedKokkosMath>(plan, species, dx, dy, dz, values, g);
              for (int axis = 0; axis < 3; ++axis) output(edge, 5 * n + axis) = g[axis];
            }
          });
      auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, output);
      std::vector<double> value(n), gx(n), gy(n), gz(n);
      for (int edge = 0; edge < edges; ++edge) {
        tagged_reference::tagged_kokkos_edge_components(host_plan, edge % 3, .31 + .13 * edge, -.49,
                                                        .87, true, value.data(), gx.data(),
                                                        gy.data(), gz.data());
        double gradient[3] = {};
        for (int i = 0; i < n; ++i) {
          near(host(edge, i), value[i]);
          near(host(edge, 4 * n + i), 0.0);
          near(host(edge, n + i), gx[i]);
          near(host(edge, 2 * n + i), gy[i]);
          near(host(edge, 3 * n + i), gz[i]);
          const double seed = .17 * (i % 7 - 3);
          gradient[0] += seed * gx[i];
          gradient[1] += seed * gy[i];
          gradient[2] += seed * gz[i];
        }
        if (v3)
          for (int axis = 0; axis < 3; ++axis) near(host(edge, 5 * n + axis), gradient[axis]);
      }
    }
}
int main(int argc, char **argv)
{
  Kokkos::initialize(argc, argv);
  int result = 0;
  try {
    check_state();
    check_dag();
    check_tagged();
    std::cout << "Kokkos device checks PASS on " << Exec::name() << '\n';
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    result = 1;
  }
  Kokkos::finalize();
  return result;
}
