import numpy as np

from wholecell_py.processes import metabolism_fit_sparse as sparse_fit
from wholecell_py.processes.metabolism import Metabolism


class GrowthModel:
    tolerance = 1e-9
    fbaEnzymeBounds = np.empty((0, 2))

    def __init__(self, growth):
        self.growth = growth
        self.evaluated = []

    def calcEffectiveFBANetwork(self):
        return "objective", "stoichiometry", "reaction bounds"

    def calcFluxBounds(
        self, substrates, enzymes, reaction_bounds, enzyme_bounds
    ):
        self.evaluated.append(np.asarray(enzymes).copy())
        return np.asarray(enzymes).copy()

    def calcGrowthRate(self, flux_bounds, objective, stoichiometry):
        return self.growth(np.asarray(flux_bounds)), np.empty(0)


def test_recursive_helper_keeps_only_source_subset_that_preserves_growth():
    model = GrowthModel(lambda enzymes: float(np.all(enzymes[:2] > 0)))
    enzymes = np.array([10, 20, 30, 40])
    initial = np.zeros(4)

    selected = sparse_fit.adjustEnzymeExpressionSparselyHelper(
        model, 1, np.empty(0), enzymes, initial, np.ones(4, dtype=bool)
    )

    np.testing.assert_array_equal(selected, [False, False, True, True])
    np.testing.assert_array_equal(model.evaluated[0], [0, 0, 0, 0])


def test_outer_search_checks_singletons_then_combinations_by_descending_size(
    monkeypatch,
):
    model = GrowthModel(
        lambda enzymes: float(enzymes[0] == 0 and enzymes[1] == 0)
    )
    enzymes = np.array([5, 6, 7])
    initial = np.zeros(3)
    monkeypatch.setattr(
        sparse_fit,
        "adjustEnzymeExpressionSparselyHelper",
        lambda self, growth0, substrates, enzymes, init, tfs: np.array(
            [True, False, False]
        ),
    )

    result = sparse_fit.adjustEnzymeExpressionSparsely(
        model, 1, np.empty(0), enzymes, initial
    )

    np.testing.assert_array_equal(result, [0, 0, 0])
    # The first singleton restores enzyme 2 and passes; the combination loop
    # then tests the largest subset first and restores enzyme 3 as well.
    np.testing.assert_array_equal(model.evaluated[0], [0, 0, 7])
    np.testing.assert_array_equal(model.evaluated[1], [0, 0, 0])


def test_metabolism_wrapper_dispatches_recursive_helper_through_instance():
    model = Metabolism()
    model.tolerance = 1e-9
    model.fbaEnzymeBounds = np.empty((0, 2))
    model.calcEffectiveFBANetwork = lambda: (
        "objective",
        "stoichiometry",
        "bounds",
    )
    model.calcFluxBounds = (
        lambda substrates, enzymes, bounds, enzyme_bounds: np.asarray(enzymes)
    )
    model.calcGrowthRate = lambda flux, objective, stoich: (
        float(np.all(np.asarray(flux)[:2] > 0)),
        np.empty(0),
    )
    enzymes = np.array([10, 20, 30, 40])

    selected = model.adjustEnzymeExpressionSparselyHelper(
        1, np.empty(0), enzymes, np.zeros(4), np.ones(4, dtype=bool)
    )

    np.testing.assert_array_equal(selected, [False, False, True, True])
