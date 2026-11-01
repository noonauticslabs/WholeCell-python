import numpy as np
import pytest

from wholecell_py.states.ftsz_ring import FtsZRing


def test_allocation_initialization_and_snapshot_copy():
    state = FtsZRing(
        snapshot={
            "numFtsZSubunitsPerNm": 0.23,
            "numFtsZSubunitsPerFilament": 9,
        }
    )
    state.initializeConstants(None)
    assert state.filamentLengthInNm == pytest.approx(9 / 0.23)
    state.allocateMemory(3)
    for name in state.stateNames:
        assert getattr(state, name).shape == (1, 1, 3)
        np.testing.assert_array_equal(getattr(state, name), 0)
    state.initialize()
    assert state.numResidualBent.shape == (1, 1, 1)

    arr = np.ones((1, 1, 2))
    restored = FtsZRing(snapshot={"numEdgesTwoBent": arr})
    assert restored.numEdgesTwoBent is not arr


def test_store_reference_and_dependent_num_edges():
    class Simulation:
        def state(self, name):
            assert name == "Geometry"
            return type("Geometry", (), {"pinchedDiameter": 1e-6})()

    state = FtsZRing(filamentLengthInNm=40)
    state.storeObjectReferences(Simulation())
    assert state.numEdges == np.floor(np.pi / np.arcsin(40e-9 / 1e-6))
    state.geometry.pinchedDiameter = 0
    assert state.numEdges == 0
    assert FtsZRing.calcNumEdges(1e-6, 40) == np.floor(
        np.pi / np.arcsin(40e-9 / 1e-6)
    )


def test_release_warns_and_returns_input_value():
    values = np.array([[1], [0]])
    with pytest.warns(RuntimeWarning, match="FtsZ not decayed"):
        result = FtsZRing.releaseFtsZ(values)
    np.testing.assert_array_equal(result, values)
    assert result is not values
    np.testing.assert_array_equal(
        FtsZRing.releaseFtsZ(np.zeros((2, 1))), np.zeros((2, 1))
    )


def test_allocate_validates_time_count():
    with pytest.raises(ValueError, match="non-negative"):
        FtsZRing().allocateMemory(-1)
    with pytest.raises(TypeError, match="integer"):
        FtsZRing().allocateMemory(1.5)
