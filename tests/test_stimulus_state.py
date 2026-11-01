import numpy as np
import pytest

from wholecell_py.states.stimulus import Stimulus


def test_metadata_snapshot_arrays_and_constants():
    values = np.arange(6).reshape((1, 2, 3))
    state = Stimulus(
        "cell",
        "stimulus",
        snapshot={
            "wholeCellModelIDs": ["s1"],
            "names": ["pulse"],
            "setValues": np.zeros((0, 6)),
            "values": values,
            "verbosity": 2,
        },
    )
    assert state.wholeCellModelID == "cell"
    assert state.name == "stimulus"
    assert state.verbosity == 2
    assert state.values is values
    assert state.values.shape == (1, 2, 3)
    assert Stimulus.optionNames == ("verbosity", "seed")
    assert Stimulus.fixedConstantNames == ()
    assert Stimulus.fittedConstantNames == ()
    assert Stimulus.stateNames == ("values",)
    assert Stimulus.dependentStateNames == ()
    assert Stimulus.dryWeight == 0


def test_constant_initialization_and_reference_storage():
    kb = {
        "numStimulis": 2,
        "numCompartments": 3,
        "stimulis": [
            {
                "wholeCellModelID": "s1",
                "name": "pulse",
                "idx": [1, 1],
                "compartments": [{"idx": 1}, {"idx": 3}],
                "values": [4.5, 7],
                "initialTimes": [0, 8],
                "finalTimes": [3, 12],
            },
            {
                "wholeCellModelID": "s2",
                "name": "step",
                "idx": [2],
                "compartments": [{"idx": 2}],
                "values": [9],
                "initialTimes": [1],
                "finalTimes": [5],
            },
        ],
    }
    state = Stimulus()
    state.initializeConstants(kb)
    assert state.wholeCellModelIDs == ["s1", "s2"]
    assert state.names == ["pulse", "step"]
    np.testing.assert_array_equal(
        state.setValues,
        [
            [1, 1, 4.5, 0, 3, 1],
            [1, 3, 7, 8, 12, 5],
            [2, 2, 9, 1, 5, 4],
        ],
    )
    compartment = type("Compartment", (), {"count": 3})()
    simulation = type("Simulation", (), {"compartment": compartment})()
    state.storeObjectReferences(simulation)
    state.allocateMemory(4)
    assert state.values.shape == (2, 3, 4)
    state.initialize()
    assert state.values.shape == (2, 3, 1)
    assert state.compartment is compartment


def test_get_indexs_uses_matlab_one_based_indices_and_zero_for_missing():
    state = Stimulus(wholeCellModelIDs=["a", "b", "c"])
    np.testing.assert_array_equal(state.getIndexs(["c", "x", "a"]), [3, 0, 1])
    assert state.getIndexs("b") == 2


def test_scalar_record_index_broadcasts_and_drives_linear_index():
    state = Stimulus()
    state.initializeConstants(
        {
            "numStimulis": 3,
            "numCompartments": 2,
            "stimulis": [
                {
                    "wholeCellModelID": "s",
                    "name": "s",
                    "idx": 2,
                    "compartments": [{"idx": 1}, {"idx": 2}],
                    "values": [4, 5],
                    "initialTimes": [0, 0],
                    "finalTimes": [9, 9],
                }
            ],
        }
    )
    np.testing.assert_array_equal(state.setValues[:, [0, 5]], [[2, 2], [2, 5]])


@pytest.mark.parametrize("count", [-1, 1.5, "2"])
def test_allocate_memory_rejects_invalid_time_point_count(count):
    with pytest.raises((TypeError, ValueError)):
        Stimulus(
            wholeCellModelIDs=[], compartment=type("C", (), {"count": 0})()
        ).allocateMemory(count)
