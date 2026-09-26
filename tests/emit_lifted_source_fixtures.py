#!/usr/bin/env python3
"""Emit C++ source-only test data from existing JSON fixtures (no YAML/core).

This bypasses the deployment loader, deliberately. It validates the source
kernel against the archived source/adjoint data, not schema/hash validation.
"""
import argparse
import json
from pathlib import Path


def cpp(value):
    if isinstance(value, list):
        return "{" + ", ".join(cpp(item) for item in value) + "}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return repr(value)
    raise TypeError(type(value))


def generate(root: Path, output: Path):
    lines = ["// Generated from tests/fixtures; do not edit.",
             "std::vector<SourceFixture> archived_fixtures() {",
             "  std::vector<SourceFixture> result;"]
    for name in ("lifted_cauchy_ta_v1", "lifted_cauchy_mixed_l_v2", "lifted_cauchy_ta_w_v1"):
        folder = root / "tests" / "fixtures" / name
        native = json.loads((folder / "native_runtime.ye3t.json").read_text())
        ref = json.loads((folder / "reference.json").read_text())
        lines += ["  {", "    SourceFixture fixture;", f"    fixture.name = {cpp(name)};",
                  "    auto &model = fixture.model;"]
        for field, value in {
            "cutoff": native["cutoff_A"],
            "role_dimension": native["role_dimension"],
            "real_component_count": native["real_component_count"],
            "source_variable_count": native["source_variable_count"],
            "central_species_order": native["central_species_order"],
            "exclude_zero_separation": native["schema"].endswith("_v2"),
        }.items():
            lines.append(f"    model.{field} = {cpp(value)};")
        for group in native["source_groups"]:
            lines += ["    {", "      LiftedCauchySourceGroup group;"]
            for field in ("neighbor_species_index", "real_component_count", "source_dimension", "q_source_variable_offsets"):
                value = group.get(field, native["real_component_count"]) if field == "real_component_count" else group[field]
                lines.append(f"      group.{field} = {cpp(value)};")
            lines.append(f"      group.angular = {group['l']};")
            lines.append("      group.transform_q_from_f = " + cpp([v for row in group["transform_q_from_f"] for v in row]) + ";")
            for poly in group["direct_q_polynomials"]:
                lines.append(f"      group.direct_q_polynomials.push_back({{{poly['q']}, {cpp(poly['coefficients'])}}});")
            for radial in group["factorized_radials"]:
                lines.append(f"      group.factorized_radials.push_back({{{radial['q']}, {radial['x_power']}, {radial['envelope_power']}}});")
            lines += ["      model.source_groups.push_back(std::move(group));", "    }"]
        species = ([native["central_species_order"].index(symbol) for symbol in ref["symbols"]]
                   if "symbols" in ref else ref["atom_types"])
        ncenters = len(ref["canonical_source_values_q"])
        for center in range(ncenters):
            selected = [e for e, c in enumerate(ref["directed_edge_centers"]) if c == center]
            edges = ", ".join("{" + str(species[ref["directed_edge_neighbors"][e]]) + ", " +
                              cpp(ref["directed_edge_displacements_A"][e]) + "}" for e in selected)
            lines.append("    fixture.edges.push_back({" + edges + "});")
            lines.append("    fixture.source.push_back(" + cpp(ref["canonical_source_values_q"][center]) + ");")
            lines.append("    fixture.adjoint.push_back(" + cpp(ref["canonical_source_adjoints_dE_dq"][center]) + ");")
            lines.append("    fixture.gradients.push_back(" + cpp([ref["directed_edge_gradients_dE_dd_eV_per_A"][e] for e in selected]) + ");")
        lines += ["    result.push_back(std::move(fixture));", "  }"]
    lines += ["  return result;", "}"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generate(args.root, args.output)
