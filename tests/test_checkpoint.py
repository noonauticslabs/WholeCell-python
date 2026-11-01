from pathlib import Path
import numpy as np
from wholecell_py.runner import load_model
from wholecell_py.checkpoint import save_checkpoint, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def test_native_resume_preserves_coupled_state_identity_and_rng(tmp_path):
    original = load_model(ROOT / "reference/snapshot_t0.json", seed=1)
    path = tmp_path / "native.pkl"
    metadata = save_checkpoint(original, path)
    restored = load_checkpoint(path)
    path.unlink()
    assert metadata["time_s"] == 0
    assert (
        restored.processes["Translation"].ribosome
        is restored.states["Ribosome"]
    )
    assert restored.states["Ribosome"] is not original.states["Ribosome"]
    original.simulation.evolveState()
    restored.simulation.evolveState()
    for name in original.states:
        for field in (
            "counts",
            "states",
            "positionStrands",
            "boundMRNAs",
            "nascentMonomerLengths",
        ):
            actual = getattr(restored.states[name], field, None)
            expected = getattr(original.states[name], field, None)
            if isinstance(actual, np.ndarray) and actual.dtype.kind in "biuf":
                np.testing.assert_array_equal(actual, expected)
    for name in ("polymerizedRegions", "linkingNumbers", "complexBoundSites"):
        ac, av = getattr(restored.states["Chromosome"], name).find()
        ec, ev = getattr(original.states["Chromosome"], name).find()
        np.testing.assert_array_equal(ac, ec)
        np.testing.assert_array_equal(av, ev)
    for key in original.processes:
        np.testing.assert_array_equal(
            original.processes[key].randStream.rand(5, 1),
            restored.processes[key].randStream.rand(5, 1),
        )
    np.testing.assert_array_equal(
        original.simulation.randStream.rand(5, 1),
        restored.simulation.randStream.rand(5, 1),
    )


def test_checkpoint_rejects_changed_runtime_source_unless_explicit(tmp_path):
    import pytest
    from types import SimpleNamespace

    graph = SimpleNamespace(
        states={"Time": SimpleNamespace(values=np.array([[7.0]]))},
        runtime_source_sha256="previous-source",
    )
    path = tmp_path / "mismatched.pkl"
    save_checkpoint(graph, path)
    with pytest.raises(ValueError, match="Runtime source differs"):
        load_checkpoint(path)
    restored = load_checkpoint(path, allow_code_change=True)
    assert restored.checkpoint_provenance["code_changed_on_resume"]
    assert restored.checkpoint_provenance["source_sha256"] == "previous-source"
