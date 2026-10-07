#include "ye3t_mean_property_cpu.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void compare(const std::vector<double> &actual, const YAML::Node &expected,
             int width, double tolerance)
{
  if (!expected.IsSequence() || actual.size() != expected.size() * width)
    throw std::runtime_error("mean-property fixture shape changed");
  double maximum = 0.0;
  for (std::size_t center = 0; center < expected.size(); ++center) {
    if (!expected[center].IsSequence() || expected[center].size() != static_cast<std::size_t>(width))
      throw std::runtime_error("mean-property fixture row width changed");
    for (int component = 0; component < width; ++component) {
      const double reference = expected[center][component].as<double>();
      const double observed = actual[center * width + component];
      maximum = std::max(maximum, std::abs(observed - reference));
      if (!std::isfinite(observed) ||
          std::abs(observed - reference) > tolerance * std::max(1.0, std::abs(reference))) {
        std::cerr << "center " << center << " component " << component
                  << " observed " << observed << " expected " << reference << '\n';
        throw std::runtime_error("mean-property CPU result differs from Python reference");
      }
    }
  }
  std::cout << "maximum_absolute_error " << maximum << '\n';
}

}  // namespace

int main(int argc, char **argv)
{
  try {
    if (argc == 3 && std::string(argv[1]) == "--accept-model") {
      YE3T_LAMMPS::YE3TMeanPropertyCPU evaluator(argv[2]);
      if (evaluator.width() != 3 && evaluator.width() != 5)
        throw std::runtime_error("accepted mean property has invalid width");
      return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--reject-model") {
      bool rejected = false;
      try {
        YE3T_LAMMPS::YE3TMeanPropertyCPU evaluator(argv[2]);
      } catch (const std::exception &error) {
        rejected = true;
        std::cout << error.what() << '\n';
      }
      if (!rejected) throw std::runtime_error("invalid mean property model was accepted");
      return 0;
    }
    if (argc != 2) throw std::runtime_error("provide one generated fixture JSON path");
    const std::filesystem::path fixture_path(argv[1]);
    const YAML::Node fixture = YAML::LoadFile(fixture_path.string());
    if (fixture["schema"].as<std::string>() != "ye3t_mean_property_cpu_fixture_v1")
      throw std::runtime_error("unknown mean-property fixture");
    const std::filesystem::path model_path = fixture_path.parent_path() /
        fixture["model"].as<std::string>();
    YE3T_LAMMPS::YE3TMeanPropertyCPU evaluator(model_path.string());
    std::vector<int> central, neighbor;
    std::vector<std::size_t> offsets;
    std::vector<double> vectors;
    for (const auto &value : fixture["central_species"]) central.push_back(value.as<int>());
    for (const auto &value : fixture["neighbor_species"]) neighbor.push_back(value.as<int>());
    for (const auto &value : fixture["edge_offsets"])
      offsets.push_back(value.as<std::size_t>());
    for (const auto &value : fixture["edge_vectors"]) vectors.push_back(value.as<double>());
    if (offsets.size() != central.size() + 1 || vectors.size() != 3 * neighbor.size() ||
        offsets.back() != neighbor.size())
      throw std::runtime_error("fixture edge CSR is invalid");
    std::vector<double> result(central.size() * evaluator.width());
    evaluator.evaluate(static_cast<int>(central.size()), central.data(), offsets.data(),
                       neighbor.data(), vectors.data(), nullptr, result.data());
    compare(result, fixture["expected"], evaluator.width(),
            fixture["tolerance"].as<double>());

    std::vector<double> rotated_vectors;
    for (const auto &value : fixture["rotated_edge_vectors"])
      rotated_vectors.push_back(value.as<double>());
    if (rotated_vectors.size() != vectors.size())
      throw std::runtime_error("rotated edge vector count changed");
    evaluator.evaluate(static_cast<int>(central.size()), central.data(), offsets.data(),
                       neighbor.data(), rotated_vectors.data(), nullptr, result.data());
    compare(result, fixture["rotated_expected"], evaluator.width(),
            fixture["tolerance"].as<double>());

    for (double &value : vectors) value = -value;
    evaluator.evaluate(static_cast<int>(central.size()), central.data(), offsets.data(),
                       neighbor.data(), vectors.data(), nullptr, result.data());
    const double parity = fixture["case"]["L"].as<int>() % 2 ? -1.0 : 1.0;
    std::vector<double> inverted_expected;
    for (const auto &row : fixture["expected"])
      for (const auto &value : row) inverted_expected.push_back(parity * value.as<double>());
    for (std::size_t index = 0; index < result.size(); ++index)
      if (std::abs(result[index] - inverted_expected[index]) >
          fixture["tolerance"].as<double>() *
              std::max(1.0, std::abs(inverted_expected[index])))
        throw std::runtime_error("inversion parity changed");
    for (double &value : vectors) value = -value;

    for (std::size_t center = 0; center < central.size(); ++center) {
      const std::size_t first = offsets[center], last = offsets[center + 1];
      std::reverse(neighbor.begin() + first, neighbor.begin() + last);
      for (std::size_t left = first; left < first + (last - first) / 2; ++left)
        for (int component = 0; component < 3; ++component)
          std::swap(vectors[3 * left + component],
                    vectors[3 * (last - 1 - (left - first)) + component]);
    }
    evaluator.evaluate(static_cast<int>(central.size()), central.data(), offsets.data(),
                       neighbor.data(), vectors.data(), nullptr, result.data());
    compare(result, fixture["expected"], evaluator.width(),
            fixture["tolerance"].as<double>());

    std::vector<unsigned char> active(central.size(), 1);
    active.front() = 0;
    std::vector<double> masked(result.size(), -1.0);
    evaluator.evaluate(static_cast<int>(central.size()), central.data(), offsets.data(),
                       neighbor.data(), vectors.data(), active.data(), masked.data());
    for (int component = 0; component < evaluator.width(); ++component)
      if (masked[component] != 0.0) throw std::runtime_error("masked center is nonzero");
    for (std::size_t index = evaluator.width(); index < result.size(); ++index)
      if (masked[index] != result[index])
        throw std::runtime_error("group masking changed another center's result");
    central.front() = -1;
    evaluator.evaluate(static_cast<int>(central.size()), central.data(), offsets.data(),
                       neighbor.data(), vectors.data(), nullptr, masked.data());
    for (int component = 0; component < evaluator.width(); ++component)
      if (masked[component] != 0.0) throw std::runtime_error("NULL center is nonzero");
    std::cout << "group_and_null_center_passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
