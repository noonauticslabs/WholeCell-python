import numpy as np
import pytest

from wholecell_py.states.metabolic_reaction import MetabolicReaction


def test_initialize_constants_and_allocate_matlab_shapes():
    state = MetabolicReaction("State_MetabolicReaction", "Metabolic reaction")
    kb = {
        "processes": [
            {"wholeCellModelID": "Process_Other", "reactions": []},
            {
                "wholeCellModelID": "Process_Metabolism",
                "reactions": [
                    {"wholeCellModelID": "RXN_A"},
                    {"wholeCellModelID": "RXN_B"},
                ],
            },
        ]
    }
    state.initializeConstants(kb)
    state.allocateMemory(4)
    state.initialize()
    assert state.reactionWholeCellModelIDs == ["RXN_A", "RXN_B"]
    assert state.growth.shape == (1, 1, 4)
    assert state.fluxs.shape == (2, 1, 4)
    np.testing.assert_array_equal(state.growth, 0)
    np.testing.assert_array_equal(state.fluxs, 0)


def test_doubling_time_is_reciprocal_growth():
    state = MetabolicReaction(snapshot={"growth": np.array([[[0.5, 0.25]]])})
    np.testing.assert_allclose(state.doublingTime, [[[2.0, 4.0]]])


def test_snapshot_and_keyword_metadata_are_preserved_and_arrays_copied():
    growth = np.ones((1, 1, 2))
    state = MetabolicReaction(
        snapshot={"growth": growth, "growth0": 0.1}, meanInitialGrowthRate=0.2
    )
    assert state.growth is not growth
    assert state.growth0 == 0.1
    assert state.meanInitialGrowthRate == 0.2
    assert state.stateNames == ("growth", "fluxs")
    assert state.dependentStateNames == ("doublingTime",)


def test_allocate_validates_count_and_initialized_reactions():
    with pytest.raises(ValueError, match="reactionWholeCellModelIDs"):
        MetabolicReaction().allocateMemory(1)
    state = MetabolicReaction(snapshot={"reactionWholeCellModelIDs": []})
    with pytest.raises(ValueError, match="non-negative"):
        state.allocateMemory(-1)
