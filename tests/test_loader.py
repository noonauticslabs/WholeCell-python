from pathlib import Path

import numpy as np

from wholecell_py.loader import (
    DataConfig,
    category_registry,
    load_model_parameters,
)


def test_load_extracted_model_parameters_preserves_mat_shapes_and_types():
    root = Path(__file__).resolve().parents[1] / "reference"
    model = load_model_parameters(
        DataConfig.from_extracted_data(
            root, "6cdee6b355aa0f5ff2953b1ab356eea049108e07"
        )
    )
    assert set(model.values) == {
        "fittedConstants",
        "fixedConstants",
        "options",
        "parameters",
    }
    assert model.source_commit == "6cdee6b355aa0f5ff2953b1ab356eea049108e07"
    for name, value in model.values.items():
        assert isinstance(value, np.ndarray)
        assert value.shape == (1, 1)
        assert value.dtype == object
    assert model["parameters"][0, 0].__class__.__name__ == "mat_struct"


def test_registry_exposes_known_categories():
    registry = category_registry()
    assert "Metabolite" in registry["states"]
    assert "Metabolism" in registry["processes"]
    assert len(registry["states"]) == 16
    assert len(registry["processes"]) == 28
