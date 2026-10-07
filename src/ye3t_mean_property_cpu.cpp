/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: James M. Goff (Sandia National Laboratories)
------------------------------------------------------------------------- */

#include "ye3t_mean_property_cpu.h"

#include "ye3t_canonical_json_hash.h"
#include "ye3t_runtime_core.h"
#include "ye3t_sha256.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace YE3T_LAMMPS {
namespace {

[[noreturn]] void invalid(const std::string &message)
{
  throw std::runtime_error("YE3T mean property: " + message);
}

YAML::Node member(const YAML::Node &node, const char *key)
{
  if (!node || !node.IsMap() || !node[key]) invalid(std::string("missing ") + key);
  return node[key];
}

std::string string_value(const YAML::Node &node)
{
  if (!node || !node.IsScalar()) invalid("expected a string");
  return node.as<std::string>();
}

int integer_value(const YAML::Node &node, int minimum = 0)
{
  if (!node || !node.IsScalar()) invalid("expected an integer");
  const long long value = node.as<long long>();
  if (value < minimum || value > std::numeric_limits<int>::max())
    invalid("integer exceeds the supported range");
  return static_cast<int>(value);
}

double finite_value(const YAML::Node &node)
{
  if (!node || !node.IsScalar()) invalid("expected a finite number");
  const double value = node.as<double>();
  if (!std::isfinite(value)) invalid("nonfinite number");
  return value;
}

long double positive_integer_magnitude(const YAML::Node &node)
{
  const std::string digits = string_value(node);
  if (digits.empty() || digits.find_first_not_of("0123456789") != std::string::npos ||
      digits.find_first_not_of('0') == std::string::npos)
    invalid("normalization ratio is not a positive integer");
  const long double value = std::stold(digits);
  if (!std::isfinite(value) || value <= 0.0L)
    invalid("normalization ratio exceeds the native numeric range");
  return value;
}

std::vector<int> integers(const YAML::Node &node, int minimum = 0)
{
  if (!node || !node.IsSequence()) invalid("expected an integer sequence");
  std::vector<int> values;
  values.reserve(node.size());
  for (const auto &item : node) values.push_back(integer_value(item, minimum));
  return values;
}

std::vector<double> numbers(const YAML::Node &node)
{
  if (!node || !node.IsSequence()) invalid("expected a number sequence");
  std::vector<double> values;
  values.reserve(node.size());
  for (const auto &item : node) values.push_back(finite_value(item));
  return values;
}

std::tuple<std::string, int, int, std::string> channel_key(const YAML::Node &channel)
{
  return {string_value(member(channel, "neighbor_species")),
          integer_value(member(channel, "radial_channel")),
          integer_value(member(channel, "l")),
          string_value(member(channel, "source_family_id"))};
}

double polynomial_value(const std::vector<double> &coefficients, double x)
{
  double result = 0.0;
  for (auto it = coefficients.rbegin(); it != coefficients.rend(); ++it)
    result = result * x + *it;
  return result;
}

}  // namespace

YE3TMeanPropertyCPU::YE3TMeanPropertyCPU(const std::string &path)
{
  if (std::filesystem::file_size(path) > 256ULL * 1024ULL * 1024ULL)
    invalid("artifact exceeds the 256 MiB reader limit");
  const YAML::Node root = YAML::LoadFile(path);
  const std::string schema = string_value(member(root, "schema"));
  if (schema == "ye3t_methods_density_full_m_per_atom_v2") {
    density_model_ = true;
    if (string_value(member(root, "self_hash")) !=
        canonical_json_hash_without_root_member(path, "self_hash"))
      invalid("density artifact hash mismatch");
    const std::string plan_json = canonical_json_root_member_value(path, "native_property_plan");
    const YAML::Node plan = YAML::Load(plan_json);
    const YAML::Node compiled = member(root, "compiled_density");
    const YAML::Node source = member(plan, "source");
    const YAML::Node source_basis = member(source, "site_basis");
    const std::string compiled_json = canonical_json_root_member_value(path, "compiled_density");
    if (string_value(member(plan, "schema")) !=
            "ye3t_density_full_m_native_property_plan_v1" ||
        string_value(member(plan, "self_hash")) !=
            canonical_json_value_hash_without_root_member(plan_json, "self_hash") ||
        string_value(member(compiled, "schema")) != "ye3t_density_full_m_compiler_v1" ||
        sha256_string(compiled_json) != string_value(member(root, "compiler_hash")) ||
        string_value(member(plan, "compiler_hash")) !=
            string_value(member(root, "compiler_hash")) ||
        string_value(member(plan, "coefficient_sha256")) !=
            string_value(member(compiled, "coefficient_sha256")) ||
        string_value(member(plan, "real_form_sha256")) !=
            string_value(member(root, "real_form_sha256")) ||
        string_value(member(plan, "provenance")) != "ye3t.couplings.compile")
      invalid("density compiler or native plan binding mismatch");
    if (canonical_json_value_root_member(compiled_json, "site_basis") !=
        canonical_json_value_root_member(canonical_json_value_root_member(plan_json, "source"),
                                         "site_basis"))
      invalid("density source differs from the saved compiler source");
    const YAML::Node fit = member(root, "fit");
    const YAML::Node readout = member(plan, "readout");
    const YAML::Node fit_metadata = member(fit, "fit_metadata");
    const YAML::Node penalty_metric = member(fit_metadata, "coordinate_penalty_metric");
    if (string_value(member(readout, "binding")) !=
            "compiled_feature_shared_over_M" ||
        string_value(member(readout, "fit_sha256")) !=
            sha256_string(canonical_json_root_member_value(path, "fit")) ||
        string_value(member(fit, "kind")) != "density_full_m_per_atom" ||
        string_value(member(fit_metadata, "design_column_order")) !=
            "compiled_center_and_channel_multiplet_shared_over_M" ||
        string_value(member(penalty_metric, "schema")) !=
            "selected_compiler_coordinate_euclidean_v1" ||
        string_value(member(penalty_metric, "physical_image_plan_hash")) !=
            string_value(member(root, "compiler_hash")) ||
        string_value(member(penalty_metric, "interpretation")) !=
            "coefficient_norm_in_saved_selected_coordinates")
      invalid("density readout fit binding mismatch");
    const YAML::Node target = member(plan, "target");
    const int L = integer_value(member(target, "L"), 1);
    if ((L != 1 && L != 2) ||
        string_value(member(target, "group")) != "O3" ||
        string_value(member(target, "basis")) != "real_tesseral_tensor_components" ||
        string_value(member(target, "axis_order")) !=
            "cos_L_to_cos_1_zero_sin_1_to_sin_L" ||
        string_value(member(target, "parity")) != (L == 1 ? "odd" : "even"))
      invalid("unsupported density target convention");
    width_ = 2 * L + 1;
    const auto target_m = integers(member(target, "M_values"), -L);
    if (target_m.size() != static_cast<std::size_t>(width_))
      invalid("density magnetic axis count changed");
    for (int index = 0; index < width_; ++index)
      if (target_m[index] != index - L) invalid("density magnetic axis order changed");
    const YAML::Node root_convention = member(root, "coordinate_convention");
    if (integer_value(member(root_convention, "L")) != L ||
        string_value(member(root_convention, "group")) != "O3" ||
        string_value(member(root_convention, "basis")) !=
            "real_tesseral_tensor_components" ||
        string_value(member(root_convention, "axis_order")) !=
            "cos_L_to_cos_1_zero_sin_1_to_sin_L" ||
        string_value(member(root_convention, "parity")) != (L == 1 ? "odd" : "even") ||
        integers(member(root_convention, "M_values"), -L) != target_m)
      invalid("density fitted coordinate convention changed");
    const YAML::Node matrix = member(target, "real_to_complex_matrix");
    if (!matrix.IsSequence() || matrix.size() != static_cast<std::size_t>(width_))
      invalid("density real-form matrix shape changed");
    density_real_to_complex_.resize(static_cast<std::size_t>(width_ * width_));
    for (int axis = 0; axis < width_; ++axis) {
      if (!matrix[axis].IsSequence() || matrix[axis].size() != static_cast<std::size_t>(width_))
        invalid("density real-form matrix width changed");
      for (int magnetic = 0; magnetic < width_; ++magnetic) {
        const YAML::Node pair = matrix[axis][magnetic];
        if (!pair.IsSequence() || pair.size() != 2)
          invalid("density real-form matrix entry changed");
        const std::complex<double> actual(finite_value(pair[0]), finite_value(pair[1]));
        std::complex<double> expected(0.0, 0.0);
        if (axis == L && magnetic == L) expected = {1.0, 0.0};
        for (int m = 1; m <= L; ++m) {
          const double scale = std::sqrt(0.5);
          const int cosine_axis = L - m;
          const int sine_axis = L + m;
          if (axis == cosine_axis && magnetic == L - m) expected = {scale, 0.0};
          if (axis == cosine_axis && magnetic == L + m)
            expected = {m % 2 ? -scale : scale, 0.0};
          if (axis == sine_axis && magnetic == L - m) expected = {0.0, scale};
          if (axis == sine_axis && magnetic == L + m)
            expected = {0.0, m % 2 ? scale : -scale};
        }
        if (std::abs(actual - expected) > 1e-14)
          invalid("density real-form matrix is not the declared tesseral convention");
        density_real_to_complex_[static_cast<std::size_t>(axis * width_ + magnetic)] = actual;
      }
    }

    const YAML::Node names = member(source, "species_order");
    if (!names.IsSequence() || names.size() == 0 || names.size() > 32)
      invalid("density species order is empty or too large");
    std::set<std::string> distinct_species;
    for (const auto &name : names) {
      const std::string value = string_value(name);
      if (!distinct_species.insert(value).second)
        invalid("density species order contains a duplicate");
      species_order_.push_back(value);
    }
    const int species_count = static_cast<int>(species_order_.size());
    if (string_value(member(source_basis, "radial_basis")) != "PACE_ChebExpCos" ||
        string_value(member(source_basis, "pace_crad_policy")) != "identity" ||
        string_value(member(source_basis, "chemical_basis")) != "delta" ||
        string_value(member(source_basis, "dtype")) != "float64" ||
        string_value(member(source_basis, "spherical_backend")) != "complex" ||
        string_value(member(source_basis, "spherical_normalization")) != "pace_y00_one" ||
        string_value(member(source_basis, "factor_normalization")) != "none" ||
        string_value(member(source_basis, "atomic_base_normalization")) != "none" ||
        string_value(member(source_basis, "charge_mode")) != "none")
      invalid("unsupported density physical source convention");
    const auto possible_types = integers(member(source_basis, "possible_types"));
    if (possible_types.size() != static_cast<std::size_t>(species_count))
      invalid("density source type map differs from species order");
    for (int index = 0; index < species_count; ++index)
      if (possible_types[index] != index)
        invalid("density source type map differs from species order");
    density_radial_count_ = integer_value(member(source_basis, "nradmax"), 1);
    const int lmax = integer_value(member(source_basis, "lmax"));
    if (density_radial_count_ > 128 || lmax > 16)
      invalid("density source exceeds the bounded native reader limits");
    const YAML::Node eta_by_center = member(source, "physical_eta_by_center");
    std::vector<std::map<int, std::pair<int, int>>> bound_content(
        static_cast<std::size_t>(species_count));
    for (int center = 0; center < species_count; ++center) {
      const YAML::Node entries = member(eta_by_center, species_order_[center].c_str());
      if (!entries.IsSequence()) invalid("density physical channel inventory changed");
      for (const auto &entry : entries) {
        const YAML::Node chemical = member(entry, "chemical");
        const int mu = integer_value(member(chemical, "chemical_index"));
        const int n = integer_value(member(entry, "native_pace_n"), 1);
        const int content_id = integer_value(member(entry, "compiler_content_id"), 1);
        if (mu >= species_count || n > density_radial_count_ ||
            string_value(member(entry, "central_species")) != species_order_[center] ||
            string_value(member(chemical, "kind")) != "explicit" ||
            string_value(member(chemical, "neighbor_species")) != species_order_[mu] ||
            integer_value(member(entry, "radial_index")) != n - 1 ||
            !bound_content[center].emplace(content_id, std::make_pair(n, mu)).second)
          invalid("density physical channel differs from compiler content");
      }
    }
    const auto rc = numbers(member(source_basis, "rc"));
    const auto widths = numbers(member(source_basis, "pace_cutoff_width"));
    const auto lambdas = numbers(member(source_basis, "lmbda"));
    const auto spacings = numbers(member(source_basis, "pace_spline_spacing"));
    const auto inner = numbers(member(source_basis, "pace_inner_cutoff"));
    const auto inner_width = numbers(member(source_basis, "pace_inner_cutoff_width"));
    const std::size_t bond_count = static_cast<std::size_t>(species_count * species_count);
    if (rc.size() != bond_count || widths.size() != bond_count ||
        lambdas.size() != bond_count || spacings.size() != bond_count ||
        inner.size() != bond_count || inner_width.size() != bond_count)
      invalid("density directed bond parameter count changed");
    const YAML::Node saved_cutoffs = member(source, "pair_cutoffs_A");
    pair_cutoffs_.resize(bond_count);
    density_bonds_.resize(bond_count);
    for (int center = 0; center < species_count; ++center)
      for (int neighbor = 0; neighbor < species_count; ++neighbor) {
        const std::size_t bond = static_cast<std::size_t>(center * species_count + neighbor);
        const std::string key = species_order_[center] + "-" + species_order_[neighbor];
        if (!saved_cutoffs[key] || finite_value(saved_cutoffs[key]) != rc[bond] ||
            rc[bond] <= 0.0 || lambdas[bond] <= 0.0 || widths[bond] < 0.0 ||
            widths[bond] > rc[bond] || spacings[bond] <= 0.0 ||
            rc[bond] / spacings[bond] < 2.0 ||
            !std::isfinite(rc[bond] / spacings[bond]) ||
            rc[bond] / spacings[bond] >
                static_cast<double>(2000000 / density_radial_count_) ||
            inner[bond] != 0.0 || inner_width[bond] != 0.0)
          invalid("unsupported density directed bond parameters");
        pair_cutoffs_[bond] = rc[bond];
        cutoff_ = std::max(cutoff_, rc[bond]);
        DensityBond &record = density_bonds_[bond];
        record.cutoff = rc[bond];
        record.intervals = ye3t::runtime::pace_uniform_spline_interval_count<double>(
            spacings[bond], rc[bond]);
        if (record.intervals < 2 ||
            record.intervals > 2000000 / density_radial_count_)
          invalid("density spline table exceeds the bounded native reader limit");
        const std::size_t nodes = static_cast<std::size_t>(record.intervals);
        std::vector<double> radii(nodes), cutoff_values(nodes, rc[bond]);
        std::vector<double> width_values(nodes, widths[bond]);
        std::vector<double> lambda_values(nodes, lambdas[bond]);
        const double spacing = rc[bond] / static_cast<double>(record.intervals);
        for (std::size_t node = 0; node < nodes; ++node)
          radii[node] = spacing * static_cast<double>(node + 1);
        std::vector<double> values(nodes * density_radial_count_);
        std::vector<double> derivatives(values.size());
        ye3t::runtime::pace_cheb_exp_cos_radial_table_with_derivative<double>(
            radii.data(), cutoff_values.data(), width_values.data(), lambda_values.data(),
            record.intervals, density_radial_count_, values.data(), derivatives.data());
        record.spline.resize((nodes + 1) * density_radial_count_ * 4);
        ye3t::runtime::pace_uniform_cubic_spline_build<double>(
            values.data(), derivatives.data(), record.intervals, density_radial_count_,
            rc[bond], record.spline.data());
      }

    const YAML::Node feature_rows = member(plan, "features");
    const YAML::Node blocks = member(compiled, "specs_by_M");
    const YAML::Node fit_beta = member(fit, "beta");
    const YAML::Node plan_beta = member(readout, "coefficients");
    const YAML::Node plan_ids = member(plan, "selected_coordinate_ids");
    const YAML::Node root_ids = member(root, "selected_coordinate_ids");
    const YAML::Node metric_columns = member(penalty_metric, "columns");
    const YAML::Node metric_diagonal = member(penalty_metric, "diagonal");
    if (!feature_rows.IsSequence() || feature_rows.size() == 0 ||
        !fit_beta.IsSequence() || !plan_beta.IsSequence() ||
        !plan_ids.IsSequence() || !root_ids.IsSequence() ||
        !blocks.IsSequence() || blocks.size() != static_cast<std::size_t>(width_) ||
        fit_beta.size() != feature_rows.size() || plan_beta.size() != feature_rows.size() ||
        plan_ids.size() != feature_rows.size() || root_ids.size() != feature_rows.size() ||
        !metric_columns.IsSequence() || !metric_diagonal.IsSequence() ||
        metric_columns.size() != feature_rows.size() ||
        metric_diagonal.size() != feature_rows.size() ||
        integer_value(member(fit_metadata, "n_cols")) !=
            static_cast<int>(feature_rows.size()))
      invalid("density fitted feature or magnetic block axes changed");
    for (std::size_t feature = 0; feature < feature_rows.size(); ++feature) {
      if (integer_value(member(feature_rows[feature], "compiler_feature_index")) !=
              static_cast<int>(feature) ||
          string_value(member(feature_rows[feature], "coordinate_id")) !=
              string_value(plan_ids[feature]) ||
          string_value(root_ids[feature]) != string_value(plan_ids[feature]) ||
          string_value(member(metric_columns[feature], "coordinate_id")) !=
              string_value(plan_ids[feature]) ||
          finite_value(metric_diagonal[feature]) != 1.0 ||
          finite_value(plan_beta[feature]) != finite_value(fit_beta[feature]))
        invalid("density selected feature order or readout differs from the fit");
      density_beta_.push_back(finite_value(plan_beta[feature]));
    }
    SHA256Builder coefficient_hash;
    auto hash_u64 = [&](std::uint64_t value) {
      unsigned char bytes[8];
      for (int index = 0; index < 8; ++index)
        bytes[index] = static_cast<unsigned char>((value >> (8 * index)) & 0xffU);
      coefficient_hash.update(bytes, 8);
    };
    auto hash_double = [&](double value) {
      std::uint64_t bits = 0;
      std::memcpy(&bits, &value, sizeof(bits));
      hash_u64(bits);
    };
    density_blocks_.resize(width_);
    for (int magnetic = 0; magnetic < width_; ++magnetic) {
      const YAML::Node block = blocks[magnetic];
      if (!block.IsSequence() || block.size() != feature_rows.size())
        invalid("density compiler block width differs from selected features");
      hash_u64(block.size());
      for (std::size_t feature = 0; feature < block.size(); ++feature) {
        const YAML::Node row = block[feature];
        const YAML::Node feature_record = feature_rows[feature];
        const YAML::Node keys = member(feature_record, "spec_keys_by_M");
        if (!keys.IsSequence() || keys.size() != static_cast<std::size_t>(width_) ||
            string_value(keys[magnetic]) != string_value(member(row, "key")) ||
            integer_value(member(row, "M_R"), -L) != magnetic - L ||
            integer_value(member(row, "L_R")) != L)
          invalid("density compiler magnetic row differs from native plan");
        DensityFeature record;
        record.center_species = integer_value(member(feature_record, "center_type"));
        if (record.center_species >= species_count) invalid("density center species is invalid");
        const YAML::Node factors = member(row, "channels");
        const YAML::Node label = member(row, "label");
        const YAML::Node combinations = member(row, "ms_combinations");
        const YAML::Node coefficients = member(row, "coeffs");
        if (!factors.IsSequence() || factors.size() == 0 ||
            !combinations.IsSequence() || !coefficients.IsSequence() ||
            combinations.size() != coefficients.size())
          invalid("density compiler polynomial shape changed");
        const auto content_ids = integers(member(label, "n_tuple"), 1);
        if (content_ids.size() != factors.size())
          invalid("density compiler content and factor counts differ");
        for (std::size_t slot = 0; slot < factors.size(); ++slot) {
          const YAML::Node factor = factors[slot];
          DensityFactor channel;
          channel.n = integer_value(member(factor, "n"), 1);
          channel.l = integer_value(member(factor, "l"));
          channel.mu = integer_value(member(factor, "mu"));
          if (channel.n > density_radial_count_ || channel.l > lmax ||
              channel.mu >= species_count ||
              integer_value(member(factor, "mu0")) != record.center_species ||
              integer_value(member(factor, "kappa0")) != 0 ||
              integer_value(member(factor, "kappa")) != 0 ||
              integer_value(member(factor, "m")) != 0 ||
              !member(factor, "eta").IsNull() ||
              !member(factor, "l_aux").IsNull() ||
              !member(factor, "m_aux").IsNull())
            invalid("unsupported density compiler source channel");
          const auto bound = bound_content[record.center_species].find(content_ids[slot]);
          if (bound == bound_content[record.center_species].end() ||
              bound->second != std::make_pair(channel.n, channel.mu))
            invalid("density compiler channel differs from physical content");
          record.factors.push_back(channel);
        }
        int angular_degree_sum = 0;
        for (const DensityFactor &factor : record.factors)
          angular_degree_sum += factor.l;
        if ((angular_degree_sum - L) % 2 != 0)
          invalid("density compiler parity differs from target");
        if (magnetic > 0) {
          const DensityFeature &reference = density_blocks_[0][feature];
          if (reference.center_species != record.center_species ||
              reference.factors.size() != record.factors.size())
            invalid("density compiler factors differ across magnetic blocks");
          for (std::size_t slot = 0; slot < record.factors.size(); ++slot)
            if (reference.factors[slot].n != record.factors[slot].n ||
                reference.factors[slot].l != record.factors[slot].l ||
                reference.factors[slot].mu != record.factors[slot].mu)
              invalid("density compiler factors differ across magnetic blocks");
        }
        hash_u64(coefficients.size());
        std::vector<double> real_parts, imag_parts;
        for (std::size_t term = 0; term < coefficients.size(); ++term) {
          const auto orders = integers(combinations[term], -lmax);
          const YAML::Node pair = coefficients[term];
          if (orders.size() != record.factors.size() ||
              !pair.IsSequence() || pair.size() != 2)
            invalid("density compiler coefficient axis changed");
          for (std::size_t slot = 0; slot < orders.size(); ++slot)
            if (orders[slot] < -record.factors[slot].l ||
                orders[slot] > record.factors[slot].l)
              invalid("density source magnetic index is out of range");
          int total_magnetic = 0;
          for (int order : orders) total_magnetic += order;
          if (total_magnetic != magnetic - L)
            invalid("density compiler term has the wrong total magnetic index");
          const double real = finite_value(pair[0]);
          const double imag = finite_value(pair[1]);
          real_parts.push_back(real);
          imag_parts.push_back(imag);
          record.terms.push_back({orders, {real, imag}});
        }
        for (double value : real_parts) hash_double(value);
        for (double value : imag_parts) hash_double(value);
        density_blocks_[magnetic].push_back(std::move(record));
      }
    }
    if (coefficient_hash.finish() != string_value(member(compiled, "coefficient_sha256")))
      invalid("density compiler coefficient hash mismatch");
    return;
  }
  if (schema != "ye3t_methods_tagged_full_m_per_atom_v2")
    invalid("expected a tagged full-M v2 artifact");
  if (string_value(member(root, "self_hash")) !=
      canonical_json_hash_without_root_member(path, "self_hash"))
    invalid("artifact hash mismatch");
  const std::string plan_json = canonical_json_root_member_value(path, "native_property_plan");
  const YAML::Node plan = YAML::Load(plan_json);
  if (string_value(member(plan, "schema")) != "ye3t_tagged_full_m_native_property_plan_v1" ||
      string_value(member(plan, "self_hash")) !=
          canonical_json_value_hash_without_root_member(plan_json, "self_hash"))
    invalid("native property plan schema or hash mismatch");
  if (string_value(member(plan, "compiler_hash")) != string_value(member(root, "compiler_hash")) ||
      string_value(member(plan, "physical_image_plan_hash")) !=
          string_value(member(root, "physical_image_plan_hash")))
    invalid("native property plan compiler binding mismatch");
  const std::string compiled_json = canonical_json_root_member_value(path, "compiled_catalogue");
  const YAML::Node compiled = member(root, "compiled_catalogue");
  if (string_value(member(compiled, "self_hash")) !=
          canonical_json_value_hash_without_root_member(compiled_json, "self_hash") ||
      string_value(member(compiled, "self_hash")) !=
          string_value(member(plan, "compiler_hash")))
    invalid("compiled catalogue hash mismatch");
  const YAML::Node compiled_sources = member(compiled, "sources");
  const auto source_json = canonical_json_value_array_items(
      canonical_json_value_root_member(compiled_json, "sources"));
  if (!compiled_sources.IsSequence() || compiled_sources.size() != source_json.size())
    invalid("compiled source inventory changed");
  std::set<std::string> compiled_source_hashes;
  for (std::size_t index = 0; index < source_json.size(); ++index) {
    const std::string hash = string_value(member(compiled_sources[index], "self_hash"));
    if (hash != canonical_json_value_hash_without_root_member(source_json[index], "self_hash") ||
        !compiled_source_hashes.insert(hash).second)
      invalid("compiled source hash or uniqueness changed");
  }
  const YAML::Node fit = member(root, "fit");
  const YAML::Node readout = member(plan, "readout");
  if (string_value(member(readout, "fit_sha256")) !=
      sha256_string(canonical_json_root_member_value(path, "fit")) ||
      string_value(member(readout, "binding")) !=
          "central_species_then_selected_multiplet_shared_over_M")
    invalid("native property plan fit binding mismatch");

  const YAML::Node target = member(plan, "target");
  const int L = integer_value(member(target, "L"), 1);
  if ((L != 1 && L != 2) ||
      string_value(member(plan, "provenance")) !=
          "ye3t.couplings.tagged_cauchy_carrier_schedule" ||
      string_value(member(target, "group")) != "O3" ||
      string_value(member(target, "basis")) != "real_tesseral_tensor_components" ||
      string_value(member(target, "axis_order")) != "cos_L_to_cos_1_zero_sin_1_to_sin_L")
    invalid("only real-tesseral L1/L2 property targets are supported");
  width_ = 2 * L + 1;
  const std::vector<int> target_m = integers(member(target, "M_values"), -L);
  if (target_m.size() != static_cast<std::size_t>(width_))
    invalid("target magnetic axis count changed");
  for (int index = 0; index < width_; ++index)
    if (target_m[index] != index - L) invalid("target magnetic axis order changed");
  const std::string expected_parity = L == 1 ? "odd" : "even";
  if (string_value(member(target, "parity")) != expected_parity)
    invalid("target natural parity changed");
  const YAML::Node target_root = member(root, "coordinate_convention");
  if (integer_value(member(target_root, "L")) != L ||
      string_value(member(target_root, "group")) != "O3" ||
      string_value(member(target_root, "basis")) != "real_tesseral_tensor_components" ||
      integers(member(target_root, "M_values"), -L) != target_m ||
      string_value(member(target_root, "parity")) != string_value(member(target, "parity")) ||
      string_value(member(target_root, "axis_order")) != string_value(member(target, "axis_order")))
    invalid("property target convention differs from model");
  const YAML::Node source = member(plan, "source");
  cutoff_ = finite_value(member(source, "cutoff_A"));
  const YAML::Node radial = member(source, "radial");
  if (cutoff_ <= 0.0 || string_value(member(radial, "family")) != "shifted_jacobi" ||
      finite_value(member(radial, "cutoff_A")) != cutoff_ ||
      string_value(member(radial, "units")) != "Angstrom" ||
      string_value(member(radial, "source_family_id")) !=
          "orthogonal_shifted_jacobi_origin_regular_v1" ||
      integer_value(member(radial, "public_radial_index_origin")) != 0 ||
      string_value(member(member(source, "chemical"), "kind")) != "explicit")
    invalid("unsupported tagged source family or cutoff");
  const YAML::Node species = member(source, "species_order");
  if (!species.IsSequence() || species.size() == 0) invalid("species order is empty");
  for (const auto &item : species) species_order_.push_back(string_value(item));
  if (std::set<std::string>(species_order_.begin(), species_order_.end()).size() !=
      species_order_.size()) invalid("duplicate species in property plan");
  const YAML::Node pair_cutoffs = member(source, "pair_cutoffs_A");
  for (const auto &left : species_order_)
    for (const auto &right : species_order_) {
      const double value = finite_value(member(pair_cutoffs, (left + "-" + right).c_str()));
      if (value <= 0.0 || value > cutoff_) invalid("invalid directed pair cutoff");
      pair_cutoffs_.push_back(value);
    }

  std::map<std::tuple<std::string, int, int, std::string>, int> channel_indices;
  const YAML::Node source_channels = member(source, "channels");
  if (!source_channels.IsSequence()) invalid("source channels are invalid");
  for (const auto &record : source_channels) {
    const YAML::Node channel = member(record, "channel");
    const auto key = channel_key(channel);
    const auto &[neighbor_name, q, l, family] = key;
    if (family != "orthogonal_shifted_jacobi_origin_regular_v1" ||
        channel_indices.count(key)) invalid("unknown or duplicate tagged source channel");
    const auto found = std::find(species_order_.begin(), species_order_.end(), neighbor_name);
    if (found == species_order_.end()) invalid("source channel has an unknown neighbor species");
    if (integer_value(member(record, "jacobi_alpha")) != 4 ||
        integer_value(member(record, "jacobi_beta")) != 2 * l + 2 ||
        string_value(member(record, "angular_convention")) != "compiler_ordered_regular_solid_v1" ||
        string_value(member(record, "real_axis_order")) !=
            "cos_l_to_cos_1_zero_sin_1_to_sin_l")
      invalid("tagged primitive convention changed");
    Channel parsed;
    parsed.neighbor_species = static_cast<int>(found - species_order_.begin());
    parsed.radial_channel = q;
    parsed.l = l;
    parsed.offset = component_count_;
    parsed.jacobi_coefficients = numbers(member(record, "shifted_jacobi_power_coefficients"));
    if (parsed.jacobi_coefficients.size() != static_cast<std::size_t>(q + 1))
      invalid("Jacobi polynomial width changed");
    const YAML::Node squared = member(record, "normalization_squared");
    const long double ratio = positive_integer_magnitude(member(squared, "numerator")) /
                              positive_integer_magnitude(member(squared, "denominator"));
    parsed.normalization = finite_value(member(record, "binary64_normalization"));
    if (parsed.normalization <= 0.0 ||
        std::abs(parsed.normalization - static_cast<double>(std::sqrt(ratio))) >
            4e-15 * std::max(1.0, parsed.normalization))
      invalid("tagged primitive normalization changed");
    if (l > (std::numeric_limits<int>::max() - component_count_ - 1) / 2)
      invalid("tagged primitive width overflows");
    component_count_ += 2 * l + 1;
    channel_indices[key] = static_cast<int>(channels_.size());
    channels_.push_back(std::move(parsed));
  }

  const YAML::Node schedule_rows = member(plan, "schedules");
  if (!schedule_rows.IsSequence()) invalid("property schedules are invalid");
  const auto schedule_json = canonical_json_value_array_items(
      canonical_json_value_root_member(plan_json, "schedules"));
  if (schedule_json.size() != schedule_rows.size()) invalid("schedule JSON inventory changed");
  std::vector<std::tuple<std::string, int, int, int>> inventory;
  for (std::size_t schedule_index = 0; schedule_index < schedule_json.size(); ++schedule_index) {
    const YAML::Node record = schedule_rows[schedule_index];
    if (string_value(member(record, "self_hash")) !=
        canonical_json_value_hash_without_root_member(schedule_json[schedule_index], "self_hash"))
      invalid("compiler schedule hash mismatch");
    const YAML::Node source_hashes = member(record, "source_hashes");
    if (!source_hashes.IsSequence() || source_hashes.size() == 0)
      invalid("compiler schedule has no bound source hashes");
    std::set<std::string> schedule_source_hashes;
    for (const auto &item : source_hashes) {
      const std::string hash = string_value(item);
      if (!compiled_source_hashes.count(hash) || !schedule_source_hashes.insert(hash).second)
        invalid("compiler schedule source hash differs from catalogue");
    }
    Schedule schedule;
    schedule.tag_count = integer_value(member(record, "tag_count"));
    schedule.support_tag_count = integer_value(member(record, "support_tag_count"));
    schedule.output_dimension = integer_value(member(record, "output_dimension"));
    if (string_value(member(record, "schema")) !=
            "ye3t_tagged_carrier_sparse_schedule_v1" ||
        string_value(member(record, "provenance")) != "ye3t.couplings.compile" ||
        string_value(member(record, "coefficient_precision")) != "binary64" ||
        schedule.tag_count > 2 ||
        schedule.support_tag_count != (schedule.tag_count == 0 ? 0 : 1) ||
        string_value(member(record, "support_realization")) !=
            (schedule.tag_count == 2 ? "edge_marginal" : "ordered_tags"))
      invalid("unsupported tagged occurrence schedule");
    const YAML::Node local_channels = member(record, "channels");
    if (!local_channels.IsSequence()) invalid("schedule channels are invalid");
    std::vector<int> local_offsets;
    for (const auto &local : local_channels) {
      const auto found = channel_indices.find(channel_key(local));
      if (found == channel_indices.end()) invalid("schedule channel is absent from source plan");
      local_offsets.push_back(channels_[found->second].offset);
    }
    const YAML::Node coordinates = member(record, "input_coordinates");
    if (!coordinates.IsSequence() ||
        integer_value(member(record, "input_dimension")) != static_cast<int>(coordinates.size()))
      invalid("schedule input dimension changed");
    for (const auto &coordinate : coordinates) {
      const auto triple = integers(coordinate);
      if (triple.size() != 3 || triple[0] >= static_cast<int>(local_offsets.size()) ||
          triple[1] > schedule.support_tag_count ||
          triple[2] >= 2 * channels_[channel_indices.at(channel_key(local_channels[triple[0]]))].l + 1)
        invalid("schedule input coordinate is out of range");
      schedule.inputs.push_back({local_offsets[triple[0]] + triple[2], triple[1]});
    }
    schedule.term_offsets = integers(member(record, "term_offsets"));
    schedule.term_components = integers(member(record, "term_components"));
    schedule.term_exponents = integers(member(record, "term_exponents"), 1);
    schedule.coefficient_terms = integers(member(record, "coefficient_terms"));
    schedule.coefficient_outputs = integers(member(record, "coefficient_outputs"));
    schedule.coefficient_values = numbers(member(record, "coefficient_values"));
    if (schedule.term_offsets.empty() || schedule.term_offsets.front() != 0 ||
        schedule.term_offsets.back() != static_cast<int>(schedule.term_components.size()) ||
        schedule.term_components.size() != schedule.term_exponents.size() ||
        schedule.coefficient_terms.size() != schedule.coefficient_outputs.size() ||
        schedule.coefficient_terms.size() != schedule.coefficient_values.size())
      invalid("schedule sparse array sizes differ");
    for (std::size_t index = 1; index < schedule.term_offsets.size(); ++index)
      if (schedule.term_offsets[index] < schedule.term_offsets[index - 1])
        invalid("schedule term offsets are not monotone");
    for (int component : schedule.term_components)
      if (component >= static_cast<int>(schedule.inputs.size()))
        invalid("schedule term input is out of range");
    if (schedule.support_tag_count > 0)
      for (std::size_t term = 0; term + 1 < schedule.term_offsets.size(); ++term) {
        bool has_explicit_factor = false;
        for (int factor = schedule.term_offsets[term];
             factor < schedule.term_offsets[term + 1]; ++factor)
          if (schedule.inputs[schedule.term_components[factor]].role <
              schedule.support_tag_count)
            has_explicit_factor = true;
        if (!has_explicit_factor)
          invalid("tagged support term has no explicit edge factor");
      }
    for (std::size_t index = 0; index < schedule.coefficient_terms.size(); ++index)
      if (schedule.coefficient_terms[index] >=
              static_cast<int>(schedule.term_offsets.size()) - 1 ||
          schedule.coefficient_outputs[index] >= schedule.output_dimension)
        invalid("schedule coefficient index is out of range");
    const YAML::Node labels = member(record, "inventory");
    if (!labels.IsSequence()) invalid("schedule inventory is invalid");
    std::vector<std::string> local_ids;
    for (const auto &item : labels) {
      const auto span = integers(member(item, "component_slice"));
      if (span.size() != 2 || span[1] - span[0] != width_ ||
          span[1] > schedule.output_dimension)
        invalid("schedule multiplet component span changed");
      const std::string coordinate = string_value(member(member(item, "label"), "coordinate_id"));
      local_ids.push_back(coordinate);
      inventory.emplace_back(coordinate,
                             static_cast<int>(schedules_.size()), span[0], schedule.tag_count);
    }
    const YAML::Node certificate = record["marginal_image_certificate"];
    if (schedule.tag_count == 2) {
      if (!certificate || !certificate.IsMap() ||
          integer_value(member(certificate, "analytic_tag_count")) != 2 ||
          integer_value(member(certificate, "explicit_support_tag_count")) != 1 ||
          !member(certificate, "exact_all_M_reconstruction").as<bool>() ||
          string_value(member(certificate, "identity")) !=
              "sum_k_distinct_B(j,k)=B(j,A)-B(j,j)" ||
          string_value(member(certificate, "orthogonalization")) != "none" ||
          string_value(member(certificate, "physical_collision_lowering")) !=
              "exact_species_jacobi_racah_real_tesseral" ||
          string_value(member(certificate, "inherited_tag_action")) !=
              "right_S2_character_exact_by_linearity" ||
          string_value(member(certificate, "physical_image_scope")) !=
              "arbitrary_coordination_with_independent_untagged_neighbor_moments")
        invalid("unsupported edge-marginal certificate");
      const YAML::Node available = member(certificate, "available_coordinate_ids");
      const YAML::Node selected = member(certificate, "selected_coordinate_ids");
      const YAML::Node reconstruction = member(certificate, "reconstruction");
      if (!available.IsSequence() || !selected.IsSequence() || !reconstruction.IsSequence() ||
          available.size() != local_ids.size() || selected.size() != local_ids.size() ||
          reconstruction.size() != local_ids.size())
        invalid("edge-marginal coordinate inventory changed");
      for (std::size_t index = 0; index < local_ids.size(); ++index) {
        if (string_value(available[index]) != local_ids[index] ||
            string_value(selected[index]) != local_ids[index] ||
            !reconstruction[index].IsSequence() || reconstruction[index].size() != 1 ||
            string_value(member(reconstruction[index][0], "coordinate_id")) != local_ids[index])
          invalid("edge-marginal reconstruction is not identity");
        const YAML::Node coefficient = member(reconstruction[index][0], "coefficient");
        const auto binary = numbers(member(coefficient, "binary64"));
        const YAML::Node real = member(coefficient, "real");
        const YAML::Node imag = member(coefficient, "imag");
        if (binary != std::vector<double>{1.0, 0.0} || !real.IsSequence() ||
            real.size() != 1 || !imag.IsSequence() || imag.size() != 0 ||
            integer_value(member(real[0], "coefficient_numerator")) != 1 ||
            integer_value(member(real[0], "coefficient_denominator")) != 1 ||
            integer_value(member(real[0], "radicand_numerator")) != 1 ||
            integer_value(member(real[0], "radicand_denominator")) != 1)
          invalid("edge-marginal reconstruction coefficient is not one");
      }
    } else if (certificate && !certificate.IsNull()) {
      invalid("unexpected marginal certificate for ordered-tag schedule");
    }
    schedules_.push_back(std::move(schedule));
  }

  const YAML::Node feature_rows = member(plan, "feature_slices");
  const YAML::Node plan_ids = member(plan, "selected_coordinate_ids");
  const YAML::Node root_ids = member(root, "selected_coordinate_ids");
  if (!feature_rows.IsSequence() || !plan_ids.IsSequence() || !root_ids.IsSequence() ||
      feature_rows.size() != inventory.size() || plan_ids.size() != inventory.size() ||
      root_ids.size() != inventory.size())
    invalid("feature inventory width changed");
  for (std::size_t index = 0; index < inventory.size(); ++index) {
    const YAML::Node row = feature_rows[index];
    const auto span = integers(member(row, "component_slice"));
    if (span.size() != 2 ||
        string_value(member(row, "coordinate_id")) != std::get<0>(inventory[index]) ||
        string_value(plan_ids[index]) != std::get<0>(inventory[index]) ||
        string_value(root_ids[index]) != std::get<0>(inventory[index]) ||
        integer_value(member(row, "schedule_index")) != std::get<1>(inventory[index]) ||
        span[0] != std::get<2>(inventory[index]) || span[1] - span[0] != width_ ||
        integer_value(member(row, "tag_count")) != std::get<3>(inventory[index]))
      invalid("selected feature order differs from compiler inventory");
    features_.push_back({std::get<1>(inventory[index]), span[0]});
  }
  const YAML::Node coefficients = member(readout, "coefficients_by_species");
  const YAML::Node fitted_beta = member(fit, "beta");
  if (!coefficients.IsSequence() || coefficients.size() != species_order_.size() ||
      !fitted_beta.IsSequence() || fitted_beta.size() != species_order_.size() * features_.size())
    invalid("property readout width differs from fit");
  for (std::size_t species_index = 0; species_index < species_order_.size(); ++species_index) {
    const YAML::Node row = coefficients[species_index];
    if (!row.IsSequence() || row.size() != features_.size()) invalid("readout row width changed");
    for (std::size_t feature = 0; feature < features_.size(); ++feature) {
      const double value = finite_value(row[feature]);
      if (value != finite_value(fitted_beta[species_index * features_.size() + feature]))
        invalid("native readout differs from fitted coefficients");
      readout_.push_back(value);
    }
  }
}

void YE3TMeanPropertyCPU::evaluate(int center_count, const int *central_species,
                                   const std::size_t *edge_offsets,
                                   const int *neighbor_species,
                                   const double *edge_vectors,
                                   const unsigned char *active_centers,
                                   double *output) const
{
  if (center_count < 0 || !central_species || !edge_offsets || !output)
    invalid("missing center arrays");
  if (density_model_) {
    const int species_count = static_cast<int>(species_order_.size());
    const double angular_scale = std::sqrt(4.0 * std::acos(-1.0));
    for (int center = 0; center < center_count; ++center) {
      double *row = output + static_cast<std::size_t>(center) * width_;
      std::fill(row, row + width_, 0.0);
      if (edge_offsets[center + 1] < edge_offsets[center]) invalid("edge offsets decrease");
      const int central = central_species[center];
      if ((active_centers && !active_centers[center]) || central < 0) continue;
      if (central >= species_count) invalid("center species is out of range");
      const std::size_t first = edge_offsets[center];
      const std::size_t count = edge_offsets[center + 1] - first;
      if (count && (!neighbor_species || !edge_vectors)) invalid("missing edge arrays");
      using Key = std::tuple<int, int, int>;
      std::map<Key, std::vector<std::complex<double>>> density;
      for (const auto &block : density_blocks_)
        for (const DensityFeature &feature : block)
          if (feature.center_species == central)
            for (const DensityFactor &factor : feature.factors)
              density.try_emplace({factor.n, factor.l, factor.mu},
                                  static_cast<std::size_t>(2 * factor.l + 1),
                                  std::complex<double>(0.0, 0.0));
      for (std::size_t local = 0; local < count; ++local) {
        const std::size_t edge = first + local;
        const int neighbor = neighbor_species[edge];
        if (neighbor < 0) continue;
        if (neighbor >= species_count) invalid("neighbor species is out of range");
        const std::size_t bond_index =
            static_cast<std::size_t>(central * species_count + neighbor);
        const DensityBond &bond = density_bonds_[bond_index];
        const double *vector = edge_vectors + 3 * edge;
        const double radius = std::hypot(std::hypot(vector[0], vector[1]), vector[2]);
        if (radius <= 0.0 || radius >= bond.cutoff) continue;
        std::vector<double> radial(static_cast<std::size_t>(density_radial_count_));
        std::vector<double> radial_derivative(radial.size());
        ye3t::runtime::pace_uniform_cubic_spline_evaluate_with_derivative<double>(
            &radius, bond.spline.data(), 1, density_radial_count_, bond.intervals,
            bond.cutoff, radial.data(), radial_derivative.data());
        std::map<int, std::vector<std::complex<double>>> harmonics;
        for (auto &entry : density) {
          const int n = std::get<0>(entry.first);
          const int l = std::get<1>(entry.first);
          const int mu = std::get<2>(entry.first);
          if (mu != neighbor) continue;
          if (!harmonics.count(l)) {
            std::vector<std::complex<double>> values(static_cast<std::size_t>(2 * l + 1));
            std::vector<std::complex<double>> derivatives(values.size() * 3);
            ye3t::runtime::complex_spherical_harmonics_with_derivative<double>(
                vector, 1, l, 1e-12, values.data(), derivatives.data());
            harmonics.emplace(l, std::move(values));
          }
          const auto &angular = harmonics.at(l);
          for (std::size_t component = 0; component < angular.size(); ++component)
            entry.second[component] += radial[static_cast<std::size_t>(n - 1)] *
                                       angular_scale * angular[component];
        }
      }
      std::vector<std::complex<double>> magnetic_values(static_cast<std::size_t>(width_));
      for (int magnetic = 0; magnetic < width_; ++magnetic)
        for (std::size_t feature = 0; feature < density_blocks_[magnetic].size(); ++feature) {
          const DensityFeature &record = density_blocks_[magnetic][feature];
          if (record.center_species != central) continue;
          std::complex<double> value(0.0, 0.0);
          for (const DensityTerm &term : record.terms) {
            std::complex<double> product = term.coefficient;
            for (std::size_t slot = 0; slot < record.factors.size(); ++slot) {
              const DensityFactor &factor = record.factors[slot];
              const Key key(factor.n, factor.l, factor.mu);
              product *= density.at(key)[static_cast<std::size_t>(term.magnetic[slot] + factor.l)];
            }
            value += product;
          }
          magnetic_values[magnetic] += density_beta_[feature] * value;
        }
      for (int axis = 0; axis < width_; ++axis) {
        std::complex<double> value(0.0, 0.0);
        for (int magnetic = 0; magnetic < width_; ++magnetic)
          value += magnetic_values[magnetic] * std::conj(
              density_real_to_complex_[static_cast<std::size_t>(axis * width_ + magnetic)]);
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag()) ||
            std::abs(value.imag()) > 1e-10 * std::max(1.0, std::abs(value.real())))
          invalid("density compiler mean has an imaginary tesseral residue");
        row[axis] = value.real();
      }
    }
    return;
  }
  const int species_count = static_cast<int>(species_order_.size());
  const double pi = std::acos(-1.0);
  int max_l = 0;
  for (const Channel &channel : channels_) max_l = std::max(max_l, channel.l);
  for (int center = 0; center < center_count; ++center) {
    std::fill(output + static_cast<std::size_t>(center) * width_,
              output + static_cast<std::size_t>(center + 1) * width_, 0.0);
    if (edge_offsets[center + 1] < edge_offsets[center]) invalid("edge offsets decrease");
    const int central = central_species[center];
    if ((active_centers && !active_centers[center]) || central < 0) continue;
    if (central >= species_count) invalid("center species is out of range");
    const std::size_t first = edge_offsets[center];
    const std::size_t count = edge_offsets[center + 1] - first;
    if (count && (!neighbor_species || !edge_vectors)) invalid("missing edge arrays");
    std::vector<double> edge_values(count * static_cast<std::size_t>(component_count_), 0.0);
    std::vector<double> density(static_cast<std::size_t>(component_count_), 0.0);
    std::vector<std::vector<double>> angular(static_cast<std::size_t>(max_l + 1));
    std::vector<std::vector<double>> angular_derivative(static_cast<std::size_t>(max_l + 1));
    std::vector<unsigned char> angular_ready(static_cast<std::size_t>(max_l + 1));
    for (std::size_t local = 0; local < count; ++local) {
      const std::size_t edge = first + local;
      const int neighbor = neighbor_species[edge];
      if (neighbor < 0) continue;
      if (neighbor >= species_count) invalid("neighbor species is out of range");
      const double *vector = edge_vectors + 3 * edge;
      const double radius = std::hypot(std::hypot(vector[0], vector[1]), vector[2]);
      const double pair_cutoff = pair_cutoffs_[static_cast<std::size_t>(central) * species_count + neighbor];
      if (radius <= 0.0 || radius >= pair_cutoff) continue;
      const double x = radius / pair_cutoff;
      std::fill(angular_ready.begin(), angular_ready.end(), 0);
      for (const Channel &channel : channels_) {
        if (channel.neighbor_species != neighbor) continue;
        const int angular_width = 2 * channel.l + 1;
        if (!angular_ready[channel.l]) {
          angular[channel.l].resize(static_cast<std::size_t>(angular_width));
          angular_derivative[channel.l].resize(static_cast<std::size_t>(angular_width) * 3);
          ye3t::runtime::real_spherical_harmonics_with_derivative<double>(
              vector, 1, channel.l, 1e-12, angular[channel.l].data(),
              angular_derivative[channel.l].data());
          angular_ready[channel.l] = 1;
        }
        double x_l = 1.0;
        for (int power = 0; power < channel.l; ++power) x_l *= x;
        const double radial = channel.normalization * x_l *
                              (1.0 - x) * (1.0 - x) *
                              polynomial_value(channel.jacobi_coefficients, x) *
                              std::sqrt(4.0 * pi / angular_width);
        for (int component = 0; component < angular_width; ++component) {
          const int offset = channel.offset + component;
          const double value = radial * angular[channel.l][angular_width - 1 - component];
          edge_values[local * component_count_ + offset] = value;
          density[offset] += value;
        }
      }
    }
    std::vector<std::vector<double>> pooled;
    pooled.reserve(schedules_.size());
    for (const Schedule &schedule : schedules_) {
      std::vector<double> values(static_cast<std::size_t>(schedule.output_dimension), 0.0);
      const std::size_t support_count = schedule.support_tag_count == 0 ? 1 : count;
      std::vector<double> inputs(schedule.inputs.size());
      std::vector<double> monomials(schedule.term_offsets.size() - 1);
      for (std::size_t support = 0; support < support_count; ++support) {
        for (std::size_t index = 0; index < schedule.inputs.size(); ++index) {
          const Input &input = schedule.inputs[index];
          inputs[index] = input.role == schedule.support_tag_count ?
              density[input.component] : edge_values[support * component_count_ + input.component];
        }
        for (std::size_t term = 0; term < monomials.size(); ++term) {
          double product = 1.0;
          for (int factor = schedule.term_offsets[term];
               factor < schedule.term_offsets[term + 1]; ++factor)
            for (int power = 0; power < schedule.term_exponents[factor]; ++power)
              product *= inputs[schedule.term_components[factor]];
          monomials[term] = product;
        }
        for (std::size_t term = 0; term < schedule.coefficient_values.size(); ++term)
          values[schedule.coefficient_outputs[term]] +=
              schedule.coefficient_values[term] * monomials[schedule.coefficient_terms[term]];
      }
      pooled.push_back(std::move(values));
    }
    for (std::size_t feature = 0; feature < features_.size(); ++feature) {
      const Feature &record = features_[feature];
      const double coefficient = readout_[static_cast<std::size_t>(central) * features_.size() + feature];
      const double *source = pooled[record.schedule_index].data() + record.component_start;
      for (int component = 0; component < width_; ++component)
        output[static_cast<std::size_t>(center) * width_ + component] +=
            coefficient * source[component];
    }
  }
}

}  // namespace YE3T_LAMMPS
