from pathlib import Path

import numpy as np

from wholecell_py.processes.metabolism import Metabolism
from wholecell_py.runtime_registry import implemented_registry
from wholecell_py.snapshot import load_snapshot


ROOT = Path(__file__).resolve().parents[1]


def test_snapshot_metabolism_resource_requirements_matches_reference():
    registry = implemented_registry()
    registry["edu.stanford.covert.cell.sim.process.Metabolism"] = Metabolism
    graph = load_snapshot(
        ROOT / "reference/snapshot_t0.json", class_registry=registry
    )
    process = graph.processes["Metabolism"]
    import json

    with (ROOT / "reference/requirements_t0.json").open() as stream:
        values = json.load(stream)["Metabolism"]
    actual = process.calcResourceRequirements_Current()
    np.testing.assert_allclose(
        actual,
        np.asarray(values["data"]).reshape(values["shape"], order="F"),
        rtol=1e-12,
        atol=1e-12,
    )


def test_snapshot_metabolism_bounds_growth_and_fluxes_match_matlab():
    registry = implemented_registry()
    registry["edu.stanford.covert.cell.sim.process.Metabolism"] = Metabolism
    graph = load_snapshot(
        ROOT / "reference/snapshot_t0.json", class_registry=registry
    )
    process = graph.processes["Metabolism"]
    import json

    with (ROOT / "reference/metabolism_t0.json").open() as stream:
        expected = json.load(stream)

    bounds = process.calcFluxBounds(
        process.substrates,
        process.enzymes,
        process.fbaReactionBounds,
        process.fbaEnzymeBounds,
    )
    realmax = float(np.asarray(process.realmax).item())
    bounds[:, 0] = np.maximum(bounds[:, 0], -realmax)
    bounds[:, 1] = np.minimum(bounds[:, 1], realmax)
    np.testing.assert_allclose(
        bounds, np.asarray(expected["bounds"]), rtol=0, atol=0
    )

    growth, reaction_fluxes, fba_fluxes, *_ = process.calcGrowthRate(
        bounds, return_full=True
    )
    assert growth == expected["growth"]
    np.testing.assert_allclose(
        reaction_fluxes, expected["reactionFluxs"], rtol=0, atol=0
    )
    np.testing.assert_allclose(
        fba_fluxes, expected["fbaFluxs"], rtol=0, atol=0
    )


def test_calc_flux_bounds_applies_availability_and_directionality():
    p = Metabolism()
    p.stepSizeSec = 2
    p.fbaReactionStoichiometryMatrix = np.zeros((1, 3))
    p.fbaReactionCatalysisMatrix = np.array([[1.0], [0.0], [0.0]])
    p.fbaReactionBounds = np.array([[-4, 4], [-5, 5], [0, 3.0]])
    p.fbaEnzymeBounds = np.array([[-2, 3], [0, 0], [0, 0.0]])
    p.fbaReactionIndexs_metabolicConversion = np.array([[1]])
    p.fbaReactionIndexs_metaboliteInternalExchange = np.array([[2]])
    p.fbaReactionIndexs_biomassExchange = np.array([], dtype=int)
    p.fbaReactionIndexs_biomassProduction = np.array([[3]])
    p.fbaReactionIndexs_metaboliteExternalExchange = np.array([[2]])
    p.fbaReactionIndexs_metaboliteInternalLimitedExchange = np.array([[2]])
    p.substrateIndexs_externalExchangedMetabolites = np.array([[1]])
    p.compartmentIndexs_extracellular = np.array([[1]])
    p.substrateIndexs_internalExchangedLimitedMetabolites = np.array([[1]])
    p.mass = type("Mass", (), {"cellDry": np.array([[2.0]])})()
    p.substrateIndexs_limitableProteins = np.array([], dtype=int)
    p.proteinLimitableProteinComposition = np.empty((0, 0))
    p.substrateMonomerLocalIndexs = p.substrateComplexLocalIndexs = np.array(
        [], dtype=int
    )
    result = p.calcFluxBounds(
        np.array([[6.0]]),
        np.array([[2.0]]),
        p.fbaReactionBounds,
        p.fbaEnzymeBounds,
        applyProteinBounds=False,
        applyEnzymeBounds=False,
    )
    np.testing.assert_allclose(result, [[-4, 4], [0, 0], [0, 0]])


def test_internal_metabolite_bound_uses_matlab_column_major_linear_indexing():
    p = Metabolism()
    p.stepSizeSec = np.array([[2.0]])
    p.fbaReactionStoichiometryMatrix = np.zeros((1, 1))
    p.fbaReactionCatalysisMatrix = np.zeros((1, 0))
    p.fbaReactionBounds = np.array([[-10.0, 10.0]])
    p.fbaEnzymeBounds = np.array([[-np.inf, np.inf]])
    p.fbaReactionIndexs_metabolicConversion = np.array([], dtype=int)
    p.fbaReactionIndexs_metaboliteInternalExchange = np.array([], dtype=int)
    p.fbaReactionIndexs_biomassExchange = np.array([], dtype=int)
    p.fbaReactionIndexs_biomassProduction = np.array([], dtype=int)
    p.fbaReactionIndexs_metaboliteExternalExchange = np.array([], dtype=int)
    p.fbaReactionIndexs_metaboliteInternalLimitedExchange = np.array([[1]])
    p.substrateIndexs_externalExchangedMetabolites = np.array([], dtype=int)
    p.compartmentIndexs_extracellular = np.array([[1]])
    p.substrateIndexs_internalExchangedLimitedMetabolites = np.array([[2]])
    p.mass = type("Mass", (), {"cellDry": np.array([[1.0]])})()
    p.substrateIndexs_limitableProteins = np.array([], dtype=int)
    p.proteinLimitableProteinComposition = np.empty((0, 0))
    p.substrateMonomerLocalIndexs = p.substrateComplexLocalIndexs = np.array(
        [], dtype=int
    )

    # MATLAB linear index 2 selects row 2, column 1 in column-major order.
    substrates = np.array([[100.0, 0.0], [6.0, 1.0]])
    bounds = p.calcFluxBounds(
        substrates,
        np.empty((0, 1)),
        p.fbaReactionBounds,
        p.fbaEnzymeBounds,
        applyEnzymeKineticBounds=False,
        applyDirectionalityBounds=False,
        applyExternalMetaboliteBounds=False,
        applyProteinBounds=False,
    )
    np.testing.assert_allclose(bounds, [[-3.0, np.inf]])


def test_protein_bounds_match_any_nonzero_compartment_stoichiometry():
    p = Metabolism()
    p.stepSizeSec = 1
    p.fbaReactionStoichiometryMatrix = np.zeros((1, 1))
    p.fbaReactionCatalysisMatrix = np.zeros((1, 0))
    p.reactionStoichiometryMatrix = np.array([[[2.0, 0.0]]])
    p.reactionIndexs_fba = np.array([[1]])
    p.substrateMonomerLocalIndexs = np.array([[1]])
    p.substrateComplexLocalIndexs = np.array([], dtype=int)
    p.substrateIndexs_limitableProteins = np.array([[1]])
    p.proteinLimitableProteinComposition = np.array([[1.0]])
    p.fbaReactionBounds = np.array([[-np.inf, np.inf]])
    p.fbaEnzymeBounds = np.array([[-np.inf, np.inf]])
    p.fbaReactionIndexs_metabolicConversion = np.array([[1]])
    p.fbaReactionIndexs_metaboliteInternalExchange = np.array([], dtype=int)
    p.fbaReactionIndexs_biomassExchange = np.array([], dtype=int)
    p.fbaReactionIndexs_biomassProduction = np.array([], dtype=int)
    p.fbaReactionIndexs_metaboliteExternalExchange = np.array([], dtype=int)
    p.fbaReactionIndexs_metaboliteInternalLimitedExchange = np.array(
        [], dtype=int
    )
    p.substrateIndexs_externalExchangedMetabolites = np.array([], dtype=int)
    p.compartmentIndexs_extracellular = np.array([[1]])
    p.substrateIndexs_internalExchangedLimitedMetabolites = np.array(
        [], dtype=int
    )
    p.mass = type("Mass", (), {"cellDry": np.array([[1.0]])})()

    bounds = p.calcFluxBounds(
        np.array([[0.0, 5.0]]),
        np.empty((0, 1)),
        p.fbaReactionBounds,
        p.fbaEnzymeBounds,
        applyEnzymeKineticBounds=False,
        applyDirectionalityBounds=False,
        applyExternalMetaboliteBounds=False,
        applyInternalMetaboliteBounds=False,
    )
    np.testing.assert_allclose(bounds, [[0.0, 0.0]])


def test_growth_solver_unwraps_matlab_options_and_warns_on_fallback(
    monkeypatch,
):
    import wholecell_py.processes.metabolism as metabolism_module

    p = Metabolism()
    p.realmax = np.array([[1e6]])
    p.fbaObjective = np.array([[1.0]])
    p.fbaReactionStoichiometryMatrix = np.zeros((1, 1))
    p.fbaRightHandSide = np.array([[0.0]])
    p.fbaReactionIndexs_biomassProduction = np.array([[1]])
    p.reactionStoichiometryMatrix = np.zeros((1, 1))
    p.reactionIndexs_fba = np.array([[1]])
    p.fbaReactionIndexs_metabolicConversion = np.array([[1]])
    p.linearProgrammingOptions = np.array(
        [[{"backend": "glpk444", "solverOptions": {"glpk": {"msglev": 0}}}]],
        dtype=object,
    )
    captured = {}

    def failed_lp(args):
        captured.update(args)
        return {
            "x": np.array([np.nan]),
            "errorFlag": True,
            "errorMsg": "fixture failure",
            "lambda": {
                "reducedCosts": np.array([np.nan]),
                "shadowPrices": np.array([np.nan]),
            },
        }

    monkeypatch.setattr(metabolism_module, "solve_lp", failed_lp)
    with np.testing.assert_warns(RuntimeWarning):
        growth, fluxes = p.calcGrowthRate(np.array([[-np.inf, np.inf]]))
    assert growth == 0
    np.testing.assert_array_equal(fluxes, [0])
    assert captured["backend"] == "glpk444"
    assert captured["glpk_options"] == {"msglev": 0}


def test_missing_solver_is_fatal_not_zero_metabolism(monkeypatch):
    import pytest
    from wholecell_py import solvers
    from wholecell_py.runner import load_model

    graph = load_model(ROOT / "reference/snapshot_t60.json", seed=101)
    process = graph.processes["Metabolism"]

    def missing(*args, **kwargs):
        raise FileNotFoundError("test missing GLPK executable")

    monkeypatch.setattr(solvers, "_solve_glpk444", missing)
    with pytest.raises(
        RuntimeError, match="solver infrastructure failure.*missing GLPK"
    ):
        process.calcGrowthRate(process.fbaReactionBounds)
