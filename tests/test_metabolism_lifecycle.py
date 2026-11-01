"""Source-shaped tests for Metabolism life-cycle resource initialization."""

from types import SimpleNamespace

import numpy as np
import pytest

from wholecell_py.processes.metabolism import Metabolism


@pytest.mark.parametrize(
    "initialization,growth_factor,expected_growth",
    [("expected", 3.0, 0.4), ("measured", 3.0, 1.2)],
)
def test_lifecycle_assembles_initial_substrates_and_enzymes_by_matlab_maps(
    initialization, growth_factor, expected_growth
):
    process = Metabolism()
    # Six substrate rows across two compartments make MATLAB's linear
    # indexing distinguishable from row-wise or per-compartment indexing.
    original_substrates = np.arange(1.0, 13.0).reshape((6, 2), order="F")
    process.substrates = original_substrates.copy()
    process.substrateWholeCellModelIDs = np.arange(1, 7).reshape(-1, 1)
    process.substrateMetaboliteLocalIndexs = np.array([1, 6])
    process.substrateIndexs_internalExchangedLimitedMetabolites = np.array([2])
    process.metabolismRecyclingProduction = np.arange(101.0, 113.0).reshape(
        (6, 2), order="F"
    )

    # Local rows 3/4 are monomers and row 5 is a complex. The second global
    # map column is deliberately different: MATLAB uses column one here.
    process.substrateMonomerLocalIndexs = np.array([3, 4])
    process.substrateMonomerGlobalIndexs = np.array([[5, 1], [3, 2]])
    process.substrateComplexLocalIndexs = np.array([5])
    process.substrateComplexGlobalIndexs = np.array([[4, 1]])
    process.enzymeWholeCellModelIDs = np.array(
        ["e1", "e2", "e3"], dtype=object
    )
    process.enzymeMonomerLocalIndexs = np.array([1, 3])
    process.enzymeMonomerGlobalIndexs = np.array([6, 2])
    process.enzymeComplexLocalIndexs = np.array([2])
    process.enzymeComplexGlobalIndexs = np.array([4])

    process.metabolicReaction = SimpleNamespace(growth0=0.4)
    process.macromoleculeStateInitialization = initialization
    process.macromoleculeStateInitializationGrowthFactor = growth_factor
    states = {
        "monomers0": np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
        "complexs0": np.array([70.0, 80.0, 90.0, 100.0]),
    }

    observed = {}

    def fit(growth, substrates, enzymes):
        observed["fit_growth"] = growth
        observed["fit_substrates"] = substrates.copy()
        observed["fit_enzymes"] = enzymes.copy()
        return enzymes + np.array([0.5, 1.5, 2.5])

    def minmax(substrates, enzymes):
        observed["minmax_substrates"] = substrates.copy()
        observed["minmax_enzymes"] = enzymes.copy()
        return enzymes - 0.25, enzymes + 0.75

    # Instrument the two downstream stages so this test checks lifecycle
    # assembly independently of the GLPK fitting implementation.
    process.fitEnzymes = fit
    process.calcMinMaxEnzymes = minmax

    bm, by, minimum, maximum = process.calcResourceRequirements_LifeCycle(
        states=states
    )

    expected_substrates = np.zeros((6, 2))
    expected_substrates[[0, 5], :] = original_substrates[[0, 5], :]
    # MATLAB linear index 2 is row 2, compartment 1 in column-major layout.
    expected_substrates[1, 0] = (
        -expected_growth * process.metabolismRecyclingProduction[1, 0]
    )
    expected_substrates[2, :] = states["monomers0"][4]
    expected_substrates[3, :] = states["monomers0"][2]
    expected_substrates[4, :] = states["complexs0"][3]

    expected_enzymes = np.array([60.0, 100.0, 20.0])
    fitted = expected_enzymes + np.array([0.5, 1.5, 2.5])
    assert observed["fit_growth"] == pytest.approx(expected_growth)
    np.testing.assert_allclose(observed["fit_substrates"], expected_substrates)
    np.testing.assert_array_equal(observed["fit_enzymes"], expected_enzymes)
    np.testing.assert_allclose(
        observed["minmax_substrates"], expected_substrates
    )
    np.testing.assert_array_equal(observed["minmax_enzymes"], fitted)

    np.testing.assert_array_equal(bm, np.zeros((6, 1)))
    np.testing.assert_array_equal(by, np.zeros((6, 1)))
    np.testing.assert_array_equal(minimum, (fitted - 0.25).reshape(-1, 1))
    np.testing.assert_array_equal(maximum, (fitted + 0.75).reshape(-1, 1))
    # Lifecycle fitting must not mutate the current simulation substrate state.
    np.testing.assert_array_equal(process.substrates, original_substrates)
