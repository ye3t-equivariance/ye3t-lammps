// clang-format off
#include "ye3t_gpu_tagged_source.h"
#include "ye3t_shifted_jacobi.h"
#include "gpu_tagged_source_reference.h"
// clang-format on
#include <array>
#include <complex>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include "gpu_tagged_test_support.h"
void require(bool ok, const char *message)
{
  if (!ok) throw std::runtime_error(message);
}
double maximum_error = 0.0;
void close(double a, double b, double tolerance = 3e-12)
{
  const double error = std::abs(a - b) / (1 + std::abs(a) + std::abs(b));
  if (!std::isfinite(a) || !std::isfinite(b) || error > tolerance)
    throw std::runtime_error("tagged source mismatch: " + std::to_string(a) + " / " +
                             std::to_string(b));
  maximum_error = std::max(maximum_error, error);
}
void exercise(const TaggedCauchyModel &model, const std::string &name)
{
  HostSource plan(model);
  std::mt19937 generator(554);
  std::uniform_real_distribution<double> random(-1, 1);
  std::vector<std::array<double, 3>> positions = {{0, 0, 0},
                                                  {1e-14, 0, 0},
                                                  {0, 0, .8},
                                                  {0, 0, -.8},
                                                  {0, 1.3, 0},
                                                  {1.2, 0, 0},
                                                  {plan.cutoff, 0, 0},
                                                  {1.01 * plan.cutoff, 0, 0},
                                                  {plan.cutoff - 1e-9, 0, 0},
                                                  {plan.cutoff - plan.radial_cutoff_width, 0, 0}};
  for (int trial = 0; trial < 24; ++trial)
    positions.push_back(
        {2.7 * random(generator), 2.7 * random(generator), 2.7 * random(generator)});
  const int n = plan.total_component_count;
  std::vector<double> v(n), gx(n), gy(n), gz(n), reference(n), rx(n), ry(n), rz(n), forward(n),
      vp(n), vm(n);
  for (int species = 0; species < static_cast<int>(model.species_order.size()); ++species) {
    for (std::size_t trial = 0; trial < positions.size(); ++trial) {
      const auto xyz = positions[trial];
      gpu_tagged_edge_components<true, HostMath>(plan, species, xyz[0], xyz[1], xyz[2], v.data(),
                                                 gx.data(), gy.data(), gz.data());
      // Reference per-channel implementation, not the grouped code.
      tagged_reference::tagged_kokkos_edge_components(plan, species, xyz[0], xyz[1], xyz[2], true,
                                                      reference.data(), rx.data(), ry.data(),
                                                      rz.data());
      gpu_tagged_edge_components<false, HostMath>(plan, species, xyz[0], xyz[1], xyz[2],
                                                  forward.data(), nullptr, nullptr, nullptr);
      for (int a = 0; a < n; ++a) {
        close(v[a], reference[a]);
        close(gx[a], rx[a]);
        close(gy[a], ry[a]);
        close(gz[a], rz[a]);
        close(forward[a], v[a]);
      }
      // Production selects the streamed contraction only for V3 with no
      // moments. Legacy keeps its original flat-component contraction order:
      // epsilon-normalized near-origin harmonics can be ill-conditioned there.
      if (plan.physical_image_v3) {
        std::vector<double> seeds(n);
        double expected[3] = {0.0, 0.0, 0.0}, streamed[3];
        for (int a = 0; a < n; ++a) {
          seeds[a] = (a % 5 == 0) ? 0.0 : .17 * (a % 7 - 3);
          expected[0] += seeds[a] * gx[a];
          expected[1] += seeds[a] * gy[a];
          expected[2] += seeds[a] * gz[a];
        }
        gpu_tagged_edge_vjp<HostMath>(plan, species, xyz[0], xyz[1], xyz[2], seeds.data(),
                                      streamed);
        for (int axis = 0; axis < 3; ++axis) close(streamed[axis], expected[axis], 1e-10);
      }
      // Exclude the declared epsilon normalization kink and cutoff finite-difference
      // stencil crossings. The exact points above are still compared to baseline.
      if (trial >= 10) {
        for (int axis = 0; axis < 3; ++axis) {
          const double h = 1e-6;
          auto plus = xyz, minus = xyz;
          plus[axis] += h;
          minus[axis] -= h;
          gpu_tagged_edge_components<false, HostMath>(plan, species, plus[0], plus[1], plus[2],
                                                      vp.data(), nullptr, nullptr, nullptr);
          gpu_tagged_edge_components<false, HostMath>(plan, species, minus[0], minus[1], minus[2],
                                                      vm.data(), nullptr, nullptr, nullptr);
          const auto &gradient = axis == 0 ? gx : axis == 1 ? gy : gz;
          for (int a = 0; a < n; ++a) close((vp[a] - vm[a]) / (2 * h), gradient[a], 5e-6);
        }
      }
    }
  }
  // Reordering channel records must not reorder output components.
  auto shuffled = model;
  std::reverse(shuffled.channels.begin(), shuffled.channels.end());
  const HostSource other(shuffled);
  gpu_tagged_edge_components<true, HostMath>(plan, 0, .31, -.49, .87, v.data(), gx.data(),
                                             gy.data(), gz.data());
  gpu_tagged_edge_components<true, HostMath>(other, 0, .31, -.49, .87, reference.data(), rx.data(),
                                             ry.data(), rz.data());
  for (int a = 0; a < n; ++a) {
    close(v[a], reference[a]);
    close(gx[a], rx[a]);
    close(gy[a], ry[a]);
    close(gz[a], rz[a]);
  }
  std::cout << name << " channels=" << plan.channel_count
            << " angular/real-form groups=" << plan.source_group_count << " PASS\n";
}
#ifdef YE3T_GPU_TAGGED_FIXTURES
#include "gpu_tagged_fixtures_generated.h"
#endif
int main()
try {
  for (int l = 0; l <= TAGGED_KOKKOS_MAX_L; ++l) {
    exercise(make_model(l, false), "legacy-l" + std::to_string(l));
    exercise(make_model(l, true), "v3-l" + std::to_string(l));
  }
  auto reject = [](const TaggedCauchyModel &m) {
    bool caught = false;
    try {
      make_gpu_tagged_source_groups(m);
    } catch (const std::exception &) {
      caught = true;
    }
    require(caught, "invalid tagged layout accepted");
  };
  auto bad = make_model(1, false);
  bad.channels[0].radial_channel = -1;
  reject(bad);
  bad = make_model(1, false);
  bad.channels[0].real_form_index = 99;
  reject(bad);
  bad = make_model(1, false);
  bad.channels[0].neighbor_species_index = -1;
  reject(bad);
  bad = make_model(1, false);
  bad.channels[0].component_offset = bad.channels[1].component_offset;
  reject(bad);
  bad = make_model(1, false);
  bad.channels[0].radial_channel = 16;
  reject(bad);
#ifdef YE3T_GPU_TAGGED_FIXTURES
  for (const auto &fixture : gpu_tagged_fixtures()) exercise(fixture.second, fixture.first);
#endif
  std::cout << "Tagged GPU source HOST math tests PASS; maximum scaled error (including finite "
               "differences)="
            << maximum_error << '\n';
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
