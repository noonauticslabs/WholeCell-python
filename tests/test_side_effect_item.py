import warnings

import numpy as np

from wholecell_py.side_effect_item import SimulationStateSideEffectItem


class State:
    def __init__(self):
        self.values = np.zeros((3, 2), dtype=float)
        self.localIndexs = np.array([3, 1])
        self.molecularWeights = np.array([2.0, 5.0, 7.0])


class Sim:
    def __init__(self):
        self.states = {"metabolite": State()}


def test_indexed_update_uses_matlab_one_based_component_and_compartment():
    sim = Sim()
    item = SimulationStateSideEffectItem(
        "Metabolite", "values", "localIndexs", 1, 2, 4
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        returned = item.updateSimulationState(sim)

    assert returned is sim
    np.testing.assert_array_equal(
        sim.states["metabolite"].values, [[0, 0], [0, 0], [0, 4]]
    )
    assert len(caught) == 1
    assert "preserve mass" in str(caught[0].message)


def test_direct_component_update_and_balanced_no_warning():
    sim = Sim()
    sim.states["metabolite"].values[:] = 2
    first = SimulationStateSideEffectItem("Metabolite", "values", "", 1, 1, 3)
    second = SimulationStateSideEffectItem(
        "Metabolite", "values", "", 1, 1, -3
    )

    # An item array is mass balanced as a whole, matching the MATLAB method.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SimulationStateSideEffectItem.updateSimulationState(
            [first, second], sim
        )

    assert sim.states["metabolite"].values[0, 0] == 2
    assert not caught


def test_no_argument_construction_allows_snapshot_hydration():
    item = SimulationStateSideEffectItem()
    item.stateName = "Metabolite"
    item.propertyName = "values"
    item.indexName = ""
    item.componentIndex = 2
    item.compartmentIndex = 1
    item.deltaValue = 1
    sim = Sim()
    item.updateSimulationState(sim)
    assert sim.states["metabolite"].values[1, 0] == 1


def test_multitime_side_effect_uses_matlab_collapsed_second_subscript():
    sim = Sim()
    state = sim.states["metabolite"]
    state.values = np.zeros((3, 2, 3), dtype=float)
    item = SimulationStateSideEffectItem("Metabolite", "values", "", 2, 2, 7)

    item.updateSimulationState(sim)

    # MATLAB A(2, 2) selects row 2, compartment 2, first timepoint.
    expected = np.zeros((3, 2, 3), dtype=float)
    expected[1, 1, 0] = 7
    np.testing.assert_array_equal(state.values, expected)


def test_multitime_side_effect_can_address_folded_later_time_column():
    sim = Sim()
    state = sim.states["metabolite"]
    state.values = np.zeros((3, 2, 3), dtype=float)
    # In MATLAB A(1, 3) on a 3x2x3 array maps to column 1, timepoint 2.
    item = SimulationStateSideEffectItem("Metabolite", "values", "", 1, 3, 5)

    item.updateSimulationState(sim)

    expected = np.zeros((3, 2, 3), dtype=float)
    expected[0, 0, 1] = 5
    np.testing.assert_array_equal(state.values, expected)
