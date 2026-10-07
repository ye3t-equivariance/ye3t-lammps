// Standalone unit test for the native tagged-Cauchy CPU loader/evaluator.
// Loads a `ye3t_tagged_cauchy_slice_v2` model plus a
// fixture JSON produced by `tests/generate_tagged_cauchy_fixture.py` (one
// owned center's hand-specified edge list, evaluated in Python via
// `ye3t_methods.atomistic.tagged_cauchy_linear.RealMomentEvaluator`), evaluates the same
// center/edges through `YE3T_LAMMPS::TaggedCauchyCPUEvaluator`, and compares
// energy and per-edge dE/d(displacement) gradients to 1e-10 absolute.
//
// Usage: ml_ye3t_tagged_cauchy_cpu_test <fixture.json> [<fixture.json> ...]

#include "ye3t_tagged_cauchy_cpu.h"
#include "ye3t_tagged_cauchy_model.h"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr double TOLERANCE = 1.0e-10;

int species_index(const std::vector<std::string> &species_order, const std::string &name)
{
  for (std::size_t index = 0; index < species_order.size(); ++index)
    if (species_order[index] == name) return static_cast<int>(index);
  return -1;
}

bool check_close(double expected, double actual, const std::string &label, bool &all_ok)
{
  const double error = std::abs(expected - actual);
  const bool ok = error <= TOLERANCE;
  if (!ok) {
    std::cerr << "FAIL " << label << ": expected " << expected << " got " << actual << " |error| "
              << error << " > " << TOLERANCE << "\n";
    all_ok = false;
  }
  return ok;
}

int run_fixture(const std::string &fixture_path)
{
  std::cout << "=== " << fixture_path << " ===\n";
  const YAML::Node fixture = YAML::LoadFile(fixture_path);
  std::filesystem::path model_path = fixture["model_path"].as<std::string>();
  if (model_path.is_relative())
    model_path = std::filesystem::path(fixture_path).parent_path() / model_path;
  const std::string expected_model_hash = fixture["model_self_hash"].as<std::string>();

  YE3T_LAMMPS::TaggedCauchyModel model = YE3T_LAMMPS::TaggedCauchyModel::load(model_path.string());
  if (model.self_hash != expected_model_hash) {
    std::cerr << "FAIL model self_hash: fixture says " << expected_model_hash << " loader computed "
              << model.self_hash << "\n";
    return 1;
  }
  for (const auto &form : model.real_forms) {
    if (form.inverse_row_offsets.size() != static_cast<std::size_t>(form.width + 1) ||
        form.inverse_columns.size() != form.inverse_values.size() ||
        form.inverse_row_offsets.back() != static_cast<int>(form.inverse_values.size())) {
      std::cerr << "FAIL: sparse real-form inverse has inconsistent sizes\n";
      return 1;
    }
    for (int row = 0; row < form.width; ++row) {
      std::vector<std::complex<double>> reconstructed(static_cast<std::size_t>(form.width), 0.0);
      for (int entry = form.inverse_row_offsets[static_cast<std::size_t>(row)];
           entry < form.inverse_row_offsets[static_cast<std::size_t>(row) + 1]; ++entry)
        reconstructed[static_cast<std::size_t>(
            form.inverse_columns[static_cast<std::size_t>(entry)])] =
            form.inverse_values[static_cast<std::size_t>(entry)];
      for (int col = 0; col < form.width; ++col)
        if (reconstructed[static_cast<std::size_t>(col)] !=
            form.inverse[static_cast<std::size_t>(row) * form.width + col]) {
          std::cerr << "FAIL: sparse real-form inverse changed a coefficient\n";
          return 1;
        }
    }
  }

  const std::string central_species = fixture["central_species"].as<std::string>();
  const int central_index = species_index(model.species_order, central_species);
  if (central_index < 0) {
    std::cerr << "FAIL: central species '" << central_species
              << "' is absent from the model's species_order\n";
    return 1;
  }

  const YAML::Node edges = fixture["edges"];
  const int edge_count = static_cast<int>(edges.size());
  std::vector<int> edge_neighbor_species(static_cast<std::size_t>(edge_count));
  std::vector<double> edge_vectors(static_cast<std::size_t>(edge_count) * 3);
  for (int edge = 0; edge < edge_count; ++edge) {
    const YAML::Node entry = edges[static_cast<std::size_t>(edge)];
    const std::string neighbor_species = entry["neighbor_species"].as<std::string>();
    const int neighbor_index = species_index(model.species_order, neighbor_species);
    if (neighbor_index < 0) {
      std::cerr << "FAIL: edge neighbor species '" << neighbor_species
                << "' is absent from the model's species_order\n";
      return 1;
    }
    edge_neighbor_species[static_cast<std::size_t>(edge)] = neighbor_index;
    const YAML::Node displacement = entry["displacement_A"];
    for (int axis = 0; axis < 3; ++axis)
      edge_vectors[static_cast<std::size_t>(edge) * 3 + static_cast<std::size_t>(axis)] =
          displacement[static_cast<std::size_t>(axis)].as<double>();
  }

  YE3T_LAMMPS::TaggedCauchyCPUEvaluator evaluator(&model);
  const double evaluator_bytes_before = evaluator.memory_usage();
  const std::vector<std::size_t> edge_offsets = {0, static_cast<std::size_t>(edge_count)};
  const std::vector<int> central_species_indices = {central_index};
  double atomic_energy = 0.0;
  std::vector<double> edge_gradients(static_cast<std::size_t>(edge_count) * 3);
  evaluator.evaluate(1, central_species_indices.data(), edge_offsets.data(),
                     edge_neighbor_species.data(), edge_vectors.data(), &atomic_energy,
                     edge_gradients.data());

  bool ok = true;
  if (model.is_physical_image()) {
    const std::vector<std::size_t> overlap_offsets = {0, 1};
    const std::vector<int> overlap_neighbor_species = {central_index};
    const std::vector<double> overlap_vector = {0.0, 0.0, 0.0};
    double overlap_energy = 0.0;
    std::vector<double> overlap_gradient(3, 0.0);
    bool rejected = false;
    try {
      evaluator.evaluate(1, central_species_indices.data(), overlap_offsets.data(),
                         overlap_neighbor_species.data(), overlap_vector.data(), &overlap_energy,
                         overlap_gradient.data());
    } catch (const std::exception &) {
      rejected = true;
    }
    if (!rejected) {
      std::cerr << "FAIL: tagged V3 did not reject an exact overlap before "
                   "direction evaluation\n";
      ok = false;
    }
  }
  const double evaluator_bytes_after = evaluator.memory_usage();
  if (!(model.memory_usage() > 0.0 && evaluator_bytes_after >= evaluator_bytes_before &&
        (edge_count == 0 || evaluator_bytes_after > evaluator_bytes_before))) {
    std::cerr << "FAIL: tagged memory accounting did not include grown scratch\n";
    ok = false;
  }
  const double expected_energy = fixture["energy_eV"].as<double>();
  check_close(expected_energy, atomic_energy, "energy_eV", ok);

  const YAML::Node expected_gradients = fixture["edge_gradients_dE_dd_eV_per_A"];
  if (static_cast<int>(expected_gradients.size()) != edge_count) {
    std::cerr << "FAIL: fixture edge_gradients length mismatch\n";
    return 1;
  }
  double maximum_absolute_error = 0.0;
  for (int edge = 0; edge < edge_count; ++edge) {
    const YAML::Node row = expected_gradients[static_cast<std::size_t>(edge)];
    for (int axis = 0; axis < 3; ++axis) {
      const double expected = row[static_cast<std::size_t>(axis)].as<double>();
      const double actual =
          edge_gradients[static_cast<std::size_t>(edge) * 3 + static_cast<std::size_t>(axis)];
      maximum_absolute_error = std::max(maximum_absolute_error, std::abs(expected - actual));
      check_close(expected, actual,
                  "edge_gradients[" + std::to_string(edge) + "][" + std::to_string(axis) + "]", ok);
    }
  }

  // Exercise the bounded multi-center source batch with uneven edge counts,
  // including one empty center and more centers than one source batch.  The
  // reference evaluates each center separately, so any lost center boundary,
  // local edge offset, or central-species lookup is detected independently of
  // the Python fixture values above.
  if (edge_count > 0) {
    constexpr int batch_center_count = 10;
    std::vector<int> batch_central_species(batch_center_count);
    std::vector<std::size_t> batch_offsets(batch_center_count + 1, 0);
    std::vector<int> batch_neighbor_species;
    std::vector<double> batch_vectors;
    for (int center = 0; center < batch_center_count; ++center) {
      batch_central_species[static_cast<std::size_t>(center)] =
          center % static_cast<int>(model.species_order.size());
      const int count = center % (edge_count + 1);
      for (int edge = 0; edge < count; ++edge) {
        batch_neighbor_species.push_back(edge_neighbor_species[static_cast<std::size_t>(edge)]);
        for (int axis = 0; axis < 3; ++axis)
          batch_vectors.push_back(edge_vectors[static_cast<std::size_t>(edge) * 3 + axis]);
      }
      batch_offsets[static_cast<std::size_t>(center) + 1] = batch_neighbor_species.size();
    }

    YE3T_LAMMPS::TaggedCauchyCPUEvaluator batch_evaluator(&model);
    std::vector<double> batch_energies(batch_center_count, 0.0);
    std::vector<double> batch_gradients(batch_neighbor_species.size() * 3, 0.0);
    batch_evaluator.evaluate(batch_center_count, batch_central_species.data(), batch_offsets.data(),
                             batch_neighbor_species.data(), batch_vectors.data(),
                             batch_energies.data(), batch_gradients.data());

    YE3T_LAMMPS::TaggedCauchyCPUEvaluator single_evaluator(&model);
    for (int center = 0; center < batch_center_count; ++center) {
      const std::size_t begin = batch_offsets[static_cast<std::size_t>(center)];
      const std::size_t end = batch_offsets[static_cast<std::size_t>(center) + 1];
      const std::vector<std::size_t> single_offsets = {0, end - begin};
      const int single_species = batch_central_species[static_cast<std::size_t>(center)];
      double single_energy = 0.0;
      std::vector<double> single_gradients((end - begin) * 3, 0.0);
      single_evaluator.evaluate(
          1, &single_species, single_offsets.data(), batch_neighbor_species.data() + begin,
          batch_vectors.data() + begin * 3, &single_energy, single_gradients.data());
      check_close(single_energy, batch_energies[static_cast<std::size_t>(center)],
                  "source batch energy[" + std::to_string(center) + "]", ok);
      for (std::size_t index = 0; index < single_gradients.size(); ++index)
        check_close(single_gradients[index], batch_gradients[begin * 3 + index],
                    "source batch gradient[" + std::to_string(center) + "][" +
                        std::to_string(index) + "]",
                    ok);
    }
  }

  if (model.execution_portfolio.present) {
    const std::vector<YE3T_LAMMPS::TaggedCauchyExecutionPolicy> policies = {
        YE3T_LAMMPS::TaggedCauchyExecutionPolicy::GENERIC_DAG,
        YE3T_LAMMPS::TaggedCauchyExecutionPolicy::SYMMETRIC_POWER,
        YE3T_LAMMPS::TaggedCauchyExecutionPolicy::BLOCK,
        YE3T_LAMMPS::TaggedCauchyExecutionPolicy::AUTO,
    };
    for (const auto policy : policies) {
      YE3T_LAMMPS::TaggedCauchyCPUEvaluator alternate(&model, policy);
      if (policy == YE3T_LAMMPS::TaggedCauchyExecutionPolicy::AUTO) {
        bool rejected_uncalibrated = false;
        try {
          double rejected_energy = 0.0;
          std::vector<double> rejected_gradients(static_cast<std::size_t>(edge_count) * 3, 0.0);
          alternate.evaluate(1, central_species_indices.data(), edge_offsets.data(),
                             edge_neighbor_species.data(), edge_vectors.data(), &rejected_energy,
                             rejected_gradients.data());
        } catch (const std::runtime_error &) {
          rejected_uncalibrated = true;
        }
        if (!rejected_uncalibrated) {
          std::cerr << "FAIL: uncalibrated AUTO did not fail closed\n";
          ok = false;
        }
        const auto calibration = alternate.calibrate_auto(8, 8, 7);
        const std::vector<YE3T_LAMMPS::TaggedCauchyExecutionPolicy> calibration_policies = {
            YE3T_LAMMPS::TaggedCauchyExecutionPolicy::COMPILED_DIRECT,
            YE3T_LAMMPS::TaggedCauchyExecutionPolicy::GENERIC_DAG,
            YE3T_LAMMPS::TaggedCauchyExecutionPolicy::SYMMETRIC_POWER,
            YE3T_LAMMPS::TaggedCauchyExecutionPolicy::BLOCK,
        };
        int fastest = 0;
        for (int candidate = 1; candidate < 4; ++candidate)
          if (calibration[static_cast<std::size_t>(candidate)] <
              calibration[static_cast<std::size_t>(fastest)])
            fastest = candidate;
        int selected = 0;
        if (fastest != 0 &&
            alternate.auto_candidate_confidently_faster(
                calibration_policies[static_cast<std::size_t>(fastest)], 0.92))
          selected = fastest;
        alternate.freeze_auto_policy(calibration_policies[static_cast<std::size_t>(selected)]);
      }
      double alternate_energy = 0.0;
      std::vector<double> alternate_gradients(static_cast<std::size_t>(edge_count) * 3, 0.0);
      alternate.evaluate(1, central_species_indices.data(), edge_offsets.data(),
                         edge_neighbor_species.data(), edge_vectors.data(), &alternate_energy,
                         alternate_gradients.data());
      const std::string route = alternate.selected_evaluator_name();
      if (route == "invalid" ||
          (policy != YE3T_LAMMPS::TaggedCauchyExecutionPolicy::AUTO &&
           route == "compiled_direct")) {
        std::cerr << "FAIL: portfolio policy selected forbidden route " << route << "\n";
        ok = false;
      }
      check_close(atomic_energy, alternate_energy, "portfolio " + route + " energy", ok);
      for (std::size_t index = 0; index < alternate_gradients.size(); ++index)
        check_close(edge_gradients[index], alternate_gradients[index],
                    "portfolio " + route + " gradient[" + std::to_string(index) + "]", ok);
    }
  }

  // With two explicit tags and only one physical neighbor, the exact
  // collision correction makes every tagged feature vanish. This is a
  // small but important regression check for the optimized free-count and
  // division-free reverse schedules: the result must be the elemental
  // offset with a zero edge gradient, including when repeated factors are
  // present in the compiled monomials.
  if (model.deployment_kind == YE3T_LAMMPS::TaggedCauchyDeploymentKind::LegacyMomentV2 &&
      model.tag_count == 2 && edge_count > 0) {
    const std::vector<std::size_t> one_edge_offsets = {0, 1};
    double one_edge_energy = 0.0;
    std::vector<double> one_edge_gradient(3, 0.0);
    evaluator.evaluate(1, central_species_indices.data(), one_edge_offsets.data(),
                       edge_neighbor_species.data(), edge_vectors.data(), &one_edge_energy,
                       one_edge_gradient.data());
    check_close(model.offsets[static_cast<std::size_t>(central_index)], one_edge_energy,
                "one-neighbor collision energy", ok);
    for (int axis = 0; axis < 3; ++axis)
      check_close(0.0, one_edge_gradient[static_cast<std::size_t>(axis)],
                  "one-neighbor collision gradient[" + std::to_string(axis) + "]", ok);
  }

  std::cout << "energy: expected=" << expected_energy << " actual=" << atomic_energy
            << " |error|=" << std::abs(expected_energy - atomic_energy) << "\n";
  std::cout << "max |edge gradient error| = " << maximum_absolute_error << "\n";
  std::cout << (ok ? "PASS" : "FAIL") << " " << fixture_path << "\n";
  return ok ? 0 : 1;
}

}    // namespace

int main(int argc, char **argv)
{
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <fixture.json> [<fixture.json> ...]\n";
    return 2;
  }
  int status = 0;
  for (int index = 1; index < argc; ++index) {
    try {
      status |= run_fixture(argv[index]);
    } catch (const std::exception &exception) {
      std::cerr << "FAIL " << argv[index] << ": exception: " << exception.what() << "\n";
      status = 1;
    }
  }
  return status;
}
