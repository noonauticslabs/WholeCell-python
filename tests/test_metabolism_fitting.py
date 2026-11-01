"""Mass-preserving fitting through a small real GLPK network, plus API edges."""

from types import SimpleNamespace
import numpy as np
import pytest
from wholecell_py.processes.metabolism import Metabolism


class SmallNetwork(Metabolism):
    def __init__(self):
        super().__init__()
        self.fbaObjective = np.array([[0.0], [1.0]])
        self.fbaReactionStoichiometryMatrix = np.array([[1.0, -1.0]])
        self.fbaReactionBounds = np.array([[0.0, 10.0], [0.0, 10.0]])
        self.fbaEnzymeBounds = np.array([[0.0, 1.0], [-np.inf, np.inf]])
        self.fbaReactionCatalysisMatrix = np.array([[1.0, 0.0], [0.0, 0.0]])
        self.fbaRightHandSide = np.array([0.0])
        self.fbaReactionIndexs_biomassProduction = np.array([2])
        self.fbaReactionIndexs_metabolicConversion = np.array([1])
        self.reactionIndexs_fba = np.array([1])
        self.reactionStoichiometryMatrix = np.array([[1.0]])
        self.fbaSubstrateIndexs_substrates = np.array([1])
        self.substrateIndexs_fba = np.array([1])
        self.substrates = np.array([[100.0]])
        self.enzymeMolecularWeights = np.array([[1.0], [1.0]])
        self.enzymeMonomerLocalIndexs = np.array([1, 2])
        self.enzymeComplexLocalIndexs = np.array([], dtype=int)
        self.monomer = self.complex = SimpleNamespace(
            minimumAverageExpression=0.1
        )
        self.realmax = 1e10
        self.tolerance = 1e-8
        self.linearProgrammingOptions = {"backend": "glpk444"}

    def calcEffectiveFBANetwork(self, n=4):
        return (
            self.fbaObjective.copy(),
            self.fbaReactionStoichiometryMatrix.copy(),
            self.fbaReactionBounds.copy(),
        )

    def calcFluxBounds(
        self,
        substrates,
        enzymes,
        bounds,
        kinetics,
        applyEnzymeKineticBounds=True,
    ):
        result = bounds.copy()
        if applyEnzymeKineticBounds:
            result[0, 1] = min(result[0, 1], np.asarray(enzymes).ravel()[0])
        return result


@pytest.mark.parametrize(
    "initial,target,expected",
    [([1.0, 9.0], 2.0, [2.0, 8.0]), ([2.0, 8.0], 1.0, [1.0, 9.0])],
)
def test_fitting_changes_rate_and_preserves_total_protein_mass(
    initial, target, expected
):
    process = SmallNetwork()
    initial = np.array(initial)[:, None]
    fitted = process.fitEnzymes(target, process.substrates, initial)
    np.testing.assert_allclose(fitted.ravel(), expected, atol=1e-9)
    assert fitted.shape == initial.shape
    assert fitted.sum() == pytest.approx(initial.sum())
    bounds = process.calcFluxBounds(
        process.substrates,
        fitted,
        process.fbaReactionBounds,
        process.fbaEnzymeBounds,
    )
    assert process.calcGrowthRate(bounds)[0] == pytest.approx(target)
    np.testing.assert_array_equal(
        initial.ravel(), [1, 9] if target == 2 else [2, 8]
    )


def test_fitting_rejects_target_above_nonkinetic_capacity():
    process = SmallNetwork()
    with pytest.raises(RuntimeError, match="Cannot fit enzyme expression"):
        process.fitEnzymes(11.0, process.substrates, np.array([1.0, 9.0]))


def test_minimum_and_maximum_include_unconstrained_enzyme_and_literal_zero_factor():
    process = SmallNetwork()
    minimum, maximum = process.calcMinMaxEnzymes(
        process.substrates, np.array([[2.0], [8.0]])
    )
    np.testing.assert_allclose(minimum.ravel(), [2.0, 1.0])
    assert (
        maximum[0, 0] == 2
    )  # MATLAB literal 0*min means no increased constrained enzyme.
    assert maximum[1, 0] == np.finfo(float).max


def test_effective_network_replaces_only_biomass_and_internal_exchange_without_mutation():
    process = Metabolism(
        fbaObjective=np.array([[4.0], [2.0], [8.0]]),
        fbaReactionStoichiometryMatrix=np.arange(6.0).reshape(2, 3),
        fbaReactionBounds=np.ones((3, 2)),
        fbaReactionIndexs_biomassProduction=[3],
        fbaSubstrateIndexs_substrates=[2, 1],
        substrateIndexs_fba=[1, 2],
        metabolismNewProduction=np.array([[2.0], [3.0]]),
        metabolismRecyclingProduction=np.array([[5.0], [7.0]]),
        fbaReactionIndexs_metaboliteInternalExchange=[2],
    )
    objective, matrix, bounds = process.calcEffectiveFBANetwork()
    np.testing.assert_array_equal(objective.ravel(), [0, 0, 1])
    np.testing.assert_array_equal(matrix[:, 2], [-31, -22])
    np.testing.assert_array_equal(bounds, [[1, 1], [0, 0], [1, 1]])
    np.testing.assert_array_equal(
        process.fbaReactionStoichiometryMatrix, np.arange(6.0).reshape(2, 3)
    )
