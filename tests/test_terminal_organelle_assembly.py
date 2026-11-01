import json
from pathlib import Path

import numpy as np

from wholecell_py.processes.terminal_organelle_assembly import (
    TerminalOrganelleAssembly,
)
from wholecell_py.snapshot import load_snapshot
from wholecell_py.runtime_registry import implemented_registry

ROOT = Path(__file__).parents[1]


def test_snapshot_t0_resource_requirements_match_matlab_reference():
    registry = implemented_registry()
    registry[
        "edu.stanford.covert.cell.sim.process.TerminalOrganelleAssembly"
    ] = TerminalOrganelleAssembly
    graph = load_snapshot(
        ROOT / "reference" / "snapshot_t0.json", class_registry=registry
    )
    process = graph.processes["TerminalOrganelleAssembly"]
    expected = json.loads(
        (ROOT / "reference" / "requirements_t0.json").read_text()
    )["TerminalOrganelleAssembly"]
    np.testing.assert_array_equal(
        process.calcResourceRequirements_Current(),
        np.asarray(expected["data"]).reshape(expected["shape"], order="F"),
    )


def test_iterative_localization_and_no_progress_termination():
    process = TerminalOrganelleAssembly(
        substrates=np.array([[0, 2], [0, 3], [1, 0]], dtype=int),
        localizationReactions=np.array(
            [[[0, 0], [0, 0], [1, 0]], [[1, 0], [0, 1], [0, 0]]]
        ),
        localizationSubstrates=np.array([[1, 0, 0], [0, 1, 0]]),
        localizationThreshold=np.array([1, 1]),
    )
    process.evolveState()
    np.testing.assert_array_equal(process.substrates, [[2, 0], [3, 0], [1, 0]])


def test_lifecycle_requirements_and_explicit_kb_initialization_limit():
    process = TerminalOrganelleAssembly(
        substrates=np.zeros((8, 2)),
        enzymeWholeCellModelIDs=["a", "b", "c", "d"],
        enzymeIndexs_HMW1=1,
        enzymeIndexs_HMW2=2,
        enzymeIndexs_HMW3=3,
        enzymeIndexs_P32=4,
    )
    bm, by, minimum, maximum = process.calcResourceRequirements_LifeCycle()
    assert bm.shape == by.shape == (8, 1)
    np.testing.assert_array_equal(minimum.ravel(), [1, 1, 1, 1])
    assert np.isinf(maximum).all()
    with np.testing.assert_raises_regex(
        NotImplementedError, "knowledgeBase and simulation"
    ):
        process.initializeConstants()
