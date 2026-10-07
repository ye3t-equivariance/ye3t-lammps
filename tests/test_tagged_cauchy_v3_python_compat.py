from pathlib import Path

import pytest

pytest.importorskip("ye3t", reason="requires the ye3t package")
pytest.importorskip("ye3t_methods.atomistic", reason="requires ye3t-methods")

from ye3t.couplings import compile as compile_coupling
from ye3t_methods.atomistic.tagged_cauchy_image import load_tagged_cauchy_image_model


def test_committed_preselector_v3_plan_recompiles():
    model_path = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "tagged_cauchy_physical_image_v3.json"
    )
    model = load_tagged_cauchy_image_model(model_path)
    replay = compile_coupling(model.evaluator.compiled.plan)
    assert replay.payload["image_rows"] == model.evaluator.compiled.payload[
        "image_rows"
    ]
    assert replay.payload["raw_rows"] == model.evaluator.compiled.payload[
        "raw_rows"
    ]
