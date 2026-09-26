"""Generate the bounded tagged-Cauchy physical-image native fixture.

This is a deterministic compiler/runtime parity fixture, not a fitted
publication potential.  Run it from installed/source checkouts with both
``ye3t`` and ``ye3t-ace`` importable.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ye3t.couplings import compile as compile_coupling
from ye3t.couplings import plan as plan_coupling
from ye3t.couplings import racah_harmonic_product_plan
from ye3t.couplings import tagged_cauchy_image_request
from ye3t.couplings import (
    ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY,
    build_radial_species_product_record,
)
from ye3t_ace.tagged_cauchy_image import (
    TaggedCauchyImageEvaluator,
    TaggedCauchyImageLinearModel,
    export_tagged_cauchy_image_model,
    realify_tagged_cauchy_image,
    tagged_cauchy_source_plan,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    angular = racah_harmonic_product_plan(
        (1,), maximum_collision_arity=2
    )
    source_key = {
        "neighbor_species": "Ta",
        "q": 0,
        "l": 1,
        "source_family_id": ORTHOGONAL_SHIFTED_JACOBI_SOURCE_FAMILY,
    }
    source_product = build_radial_species_product_record(
        (source_key,), angular, maximum_collision_arity=2
    )
    request = tagged_cauchy_image_request(
        source_product, angular, source_key=source_key
    )
    compiled = compile_coupling(plan_coupling(request))
    program = realify_tagged_cauchy_image(compiled)
    source_plan = tagged_cauchy_source_plan(compiled, 4.8, program)
    evaluator = TaggedCauchyImageEvaluator(compiled, source_plan, program)
    beta_reference = torch.tensor([0.7, -0.2], dtype=torch.float64)
    offset_reference = -0.3

    fit_clusters = (
        [[1.1, 0.2, -0.1], [-0.4, 1.3, 0.5], [0.3, -0.7, 1.5]],
        [[0.8, -0.5, 0.4], [1.4, 0.3, -0.2], [-0.6, 1.0, 0.7]],
        [[1.7, 0.1, 0.5], [-0.2, 1.8, -0.4], [0.4, -0.9, 1.2]],
        [[1.2, 0.7, 0.2], [-1.1, 0.3, 0.8], [0.5, -0.4, 1.7]],
    )
    design_rows = []
    targets = []
    for cluster in fit_clusters:
        fit_displacements = torch.tensor(cluster, dtype=torch.float64)
        fit_edge_index = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long)
        fit_types = torch.zeros(3, dtype=torch.long)
        fit_features, fit_derivative = evaluator.evaluate_edge_list(
            fit_edge_index, fit_displacements, fit_types, 4
        )
        design_rows.append(np.concatenate((fit_features[0].numpy(), [1.0])))
        targets.append(
            float(offset_reference + torch.dot(fit_features[0], beta_reference))
        )
        for row, target in zip(
            fit_derivative.permute(0, 2, 1)
            .reshape(-1, evaluator.feature_count)
            .numpy(),
            torch.einsum("f,efd->ed", beta_reference, fit_derivative)
            .reshape(-1)
            .numpy(),
            strict=True,
        ):
            design_rows.append(np.concatenate((row, [0.0])))
            targets.append(float(target))
    design = np.asarray(design_rows, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    penalty = np.diag([1.0] * evaluator.feature_count + [0.0])
    alpha = 1.0e-12
    fitted = np.linalg.solve(design.T @ design + alpha * penalty, design.T @ target)
    beta = torch.as_tensor(fitted[:-1], dtype=torch.float64)
    offset = float(fitted[-1])
    model = TaggedCauchyImageLinearModel(
        evaluator, {"Ta": beta}, {"Ta": offset}
    )

    displacements = torch.tensor(
        [[1.1, 0.2, -0.1], [-0.4, 1.3, 0.5], [0.3, -0.7, 1.5]],
        dtype=torch.float64,
    )
    edge_index = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long)
    neighbor_types = torch.zeros(3, dtype=torch.long)
    features, feature_derivative = evaluator.evaluate_edge_list(
        edge_index, displacements, neighbor_types, 4
    )
    energy = offset + torch.dot(features[0], beta)
    edge_gradient = torch.einsum("f,efd->ed", beta, feature_derivative)

    model_path = args.output_dir / "tagged_cauchy_physical_image_v3.json"
    payload = export_tagged_cauchy_image_model(
        model_path,
        model,
        fit_metadata={
            "purpose": "deterministic_native_parity_fixture",
            "not_a_fitted_publication_model": True,
            "fit_kind": "tiny_materialized_ridge_runtime_gate",
            "ridge_alpha": alpha,
            "maximum_training_residual": float(
                np.max(np.abs(design @ fitted - target))
            ),
        },
    )
    fixture = {
        "schema": "ye3t_tagged_cauchy_native_fixture_v2",
        "model_path": model_path.name,
        "model_self_hash": payload["self_hash"],
        "central_species": "Ta",
        "edges": [
            {
                "neighbor_species": "Ta",
                "displacement_A": row,
            }
            for row in displacements.tolist()
        ],
        "energy_eV": float(energy),
        "edge_gradients_dE_dd_eV_per_A": edge_gradient.tolist(),
    }
    fixture_path = (
        args.output_dir / "tagged_cauchy_physical_image_v3.fixture.json"
    )
    fixture_path.write_text(
        json.dumps(fixture, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(model_path)
    print(fixture_path)


if __name__ == "__main__":
    main()
