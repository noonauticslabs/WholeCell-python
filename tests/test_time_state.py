import numpy as np
import pytest

from wholecell_py.states import Time


def test_time_metadata_and_default_fields():
    state = Time("ecoli", "time")
    assert state.wholeCellModelID == "ecoli"
    assert state.name == "time"
    assert state.cellCycleLength is None
    assert state.values is None
    assert Time.optionNames == ("verbosity", "seed")
    assert Time.fixedConstantNames == (
        "cellCycleLength",
        "replicationDuration",
        "cytokinesisDuration",
    )
    assert Time.fittedConstantNames == ("replicationInitiationDuration",)
    assert Time.stateNames == ("values",)
    assert Time.dependentStateNames == ()
    assert Time.dryWeight == 0


def test_allocate_memory_preserves_matlab_shape_and_reinitializes():
    state = Time()
    state.allocateMemory(4)
    assert state.values.shape == (1, 1, 4)
    np.testing.assert_array_equal(state.values, np.zeros((1, 1, 4)))
    state.values[...] = 8
    state.initialize()
    assert state.values.shape == (1, 1, 1)
    np.testing.assert_array_equal(state.values, np.zeros((1, 1, 1)))


def test_snapshot_attributes_and_arrays_are_restored_without_reshape():
    values = np.arange(3).reshape((1, 1, 3))
    state = Time(
        snapshot={"cellCycleLength": 120, "values": values, "verbosity": 2}
    )
    assert state.cellCycleLength == 120
    assert state.verbosity == 2
    assert state.values is values
    assert state.values.shape == (1, 1, 3)


@pytest.mark.parametrize("count", [-1, 1.5, "2"])
def test_allocate_memory_rejects_invalid_time_point_count(count):
    with pytest.raises((TypeError, ValueError)):
        Time().allocateMemory(count)
