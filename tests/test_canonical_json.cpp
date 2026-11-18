#include "ye3t_canonical_json_hash.h"
#include "ye3t_sha256.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using YE3T_LAMMPS::CanonicalJSONDocument;

namespace {
void require(bool condition, const std::string &message)
{
  if (!condition) throw std::runtime_error(message);
}
}

int main(int argc, char **argv)
{
  if (argc != 2) return 2;
  const std::filesystem::path path(argv[1]);
  try {
    const std::string source =
        R"({ "a": [1, {"b":"comma, brace } and quote \""}], "nested": {"x": -1.20e+03}, "self_hash":"test" })";
    {
      std::ofstream stream(path);
      stream << source;
    }
    const CanonicalJSONDocument document(path.string());
    // Subsequent lookups must use the captured document, including before its
    // first index lookup; neither the model file nor ignored fields are reread.
    std::filesystem::remove(path);
    const auto expected_hash = YE3T_LAMMPS::sha256_string(
        R"({"a":[1,{"b":"comma, brace } and quote \""}],"nested":{"x":-1.20e+03}})");
    for (int repeat = 0; repeat < 4; ++repeat) {
      require(document.nested_member_value("nested", "x") == "-1.20e+03",
              "numeric spelling changed");
      require(document.hash_without_root_member("self_hash") == expected_hash,
              "cached canonical hash changed");
      require(document.root_member_names() == std::vector<std::string>({"a", "nested", "self_hash"}),
              "member inventory changed");
      const auto values = YE3T_LAMMPS::canonical_json_array_values(document.root_member_value("a"));
      require(values == YE3T_LAMMPS::canonical_json_value_array_items(document.root_member_value("a")),
              "mean-property array reader compatibility changed");
      require(values.size() == 2 && values[0] == "1" &&
                  values[1] == R"({"b":"comma, brace } and quote \""})",
              "streamed array records changed");
    }
    require(YE3T_LAMMPS::canonical_json_array_values("[]").empty(), "empty array");
    const std::string large_value(1024 * 1024, 'x');
    const std::string payload = "{\"data\":\"" + large_value + "\"}";
    const auto payload_hash = YE3T_LAMMPS::sha256_string(payload);
    require(CanonicalJSONDocument("{\"data\":\"" + large_value +
                "\",\"self_hash\":\"test\"}", true).hash_without_root_member("self_hash") == payload_hash,
            "streamed large-payload hash changed");
    require(CanonicalJSONDocument(R"({"self_hash":"only"})", true)
                .hash_without_root_member("self_hash") == YE3T_LAMMPS::sha256_string("{}"),
            "streamed empty-payload hash changed");
    bool missing_rejected = false;
    try { document.root_member_value("missing"); }
    catch (const std::runtime_error &) { missing_rejected = true; }
    require(missing_rejected, "missing member accepted");
    // Even an unrequested/offline subtree must be syntactically validated.
    for (const auto &invalid : {
             R"({"a":1,"a":2})", R"({"b":1,"a":2})",
             R"({"a":1,"z":{"x":1,"x":2}})", R"({"a":1,"z":[1,]})",
             R"({"a":1,"z":01})", R"({"a":1} trailing)"}) {
      bool rejected = false;
      try { CanonicalJSONDocument(invalid, true).root_member_value("a"); }
      catch (const std::runtime_error &) { rejected = true; }
      require(rejected, "malformed or noncanonical document accepted");
    }
  } catch (const std::exception &error) {
    std::filesystem::remove(path);
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "Canonical document reuse, record extraction and rejection tests passed.\n";
  return 0;
}
