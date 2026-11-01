"""Flux balance metabolism process, ported from MATLAB Metabolism.m.

Includes source knowledge-base initialization and FBA network construction.
Full model-wide fitting remains separate from snapshot runtime evolution.
"""

from __future__ import annotations

import numpy as np
import warnings

from ..process_base import Process
from ..reaction_process import ReactionProcess
from ..solvers import solve_lp


def _dense(value):
    return value.full() if hasattr(value, "full") else np.asarray(value)


def _idx(value):
    return np.asarray(value, dtype=int).reshape(-1) - 1


def _matlab_options(value):
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.item()
    return value if isinstance(value, dict) else {}


class Metabolism(ReactionProcess):
    def __init__(
        self,
        wholeCellModelID="Process_Metabolism",
        name="Metabolism",
        **properties,
    ):
        super().__init__(wholeCellModelID, name, **properties)

    def storeObjectReferences(self, simulation):
        super().storeObjectReferences(simulation)
        self.mass = simulation.state("Mass")
        self.metabolicReaction = simulation.state("MetabolicReaction")
        self.states.extend((self.mass, self.metabolicReaction))

    def initializeConstants(self, knowledgeBase, simulation, options=None):
        from ..process_base import _get, _cell_text

        ReactionProcess.initializeConstants(
            self,
            knowledgeBase,
            simulation,
            {"retainSubstrateCompartments": True},
        )
        self.cellCycleLength = simulation.state("Time").cellCycleLength
        (
            self.compartmentIndexs_cytosol,
            self.compartmentIndexs_extracellular,
            self.compartmentIndexs_membrane,
        ) = (1, 2, 3)
        compartments = (
            np.asarray(
                [
                    np.asarray(getattr(self.compartment, name)).item()
                    for name in (
                        "cytosolIndexs",
                        "extracellularIndexs",
                        "membraneIndexs",
                    )
                ],
                dtype=int,
            )
            - 1
        )
        for category in (
            "Stimulus",
            "Metabolite",
            "RNA",
            "Monomer",
            "Complex",
        ):
            for suffix in (
                "GlobalIndexs",
                "CompartmentIndexs",
                "GlobalCompartmentIndexs",
            ):
                key = "substrate" + category + suffix
                setattr(
                    self,
                    key,
                    np.asarray(getattr(self, key))[:, compartments].copy(),
                )
        self.reactionStoichiometryMatrix = _dense(
            self.reactionStoichiometryMatrix
        )[:, :, compartments]
        self.reactionCoenzymeMatrix = _dense(self.reactionCoenzymeMatrix)[
            :, :, compartments
        ]
        for suffix, identifier in (
            ("amp", "AMP"),
            ("adp", "ADP"),
            ("atp", "ATP"),
            ("diphosphate", "PPI"),
            ("phosphate", "PI"),
            ("water", "H2O"),
            ("hydrogen", "H"),
        ):
            setattr(
                self,
                "substrateIndexs_" + suffix,
                self.substrateIndexs([identifier]),
            )
        self.substrateIndexs_atpHydrolysis = self.substrateIndexs(
            ["ATP", "H2O", "ADP", "PI", "H"]
        )
        self.substrateIndexs_energy = np.sort(
            self.substrateIndexs(
                [
                    "AMP",
                    "CMP",
                    "GMP",
                    "UMP",
                    "ADP",
                    "CDP",
                    "GDP",
                    "UDP",
                    "ATP",
                    "CTP",
                    "GTP",
                    "UTP",
                    "PI",
                    "PPI",
                    "H",
                ]
            ).ravel()
        ).reshape(-1, 1)
        types = np.asarray(
            [
                _cell_text(v)
                for v in np.asarray(self.reactionTypes, dtype=object).ravel()
            ]
        )
        self.reactionIndexs_chemical = (
            np.flatnonzero(types == "chemical") + 1
        ).reshape(-1, 1)
        self.reactionIndexs_transport = (
            np.flatnonzero(types == "transport") + 1
        ).reshape(-1, 1)
        mon_local, cmp_local = _idx(self.substrateMonomerLocalIndexs), _idx(
            self.substrateComplexLocalIndexs
        )
        mon_global = _idx(self.substrateMonomerGlobalIndexs[:, 0])
        cmp_global = _idx(self.substrateComplexGlobalIndexs[:, 0])
        complexes = np.asarray(
            _get(knowledgeBase, "proteinComplexs"), dtype=object
        ).ravel()
        limit_complex = [
            j
            for j, k in enumerate(cmp_global)
            if _cell_text(
                _get(
                    _get(complexes[k], "complexFormationProcess"),
                    "wholeCellModelID",
                )
            )
            == "Process_MacromolecularComplexation"
        ]
        limited = np.r_[mon_local, cmp_local[limit_complex]]
        self.substrateIndexs_limitableProteins = (limited + 1).reshape(-1, 1)
        composition = np.zeros((np.size(self.substrateWholeCellModelIDs),) * 2)
        mrna = _idx(self.gene.mRNAIndexs)[mon_global]
        mon_comp = _dense(
            _get(knowledgeBase, "proteinComplexMonomerComposition")
        )
        cmp_comp = _dense(
            _get(knowledgeBase, "proteinComplexComplexComposition")
        )
        composition[np.ix_(cmp_local, mon_local)] = np.any(
            mon_comp[mrna[:, None], cmp_global[None, :], :], axis=2
        ).T
        composition[np.ix_(cmp_local, cmp_local)] = np.any(
            cmp_comp[cmp_global[:, None], cmp_global[None, :], :], axis=2
        ).T
        composition[:, np.setdiff1d(np.arange(len(composition)), limited)] = 0
        composition[limited, :] = 0
        composition[limited, limited] = 1
        self.proteinLimitableProteinComposition = composition[
            np.ix_(np.r_[mon_local, cmp_local], limited)
        ]
        self.substrateExternalExchangeBounds = np.zeros(
            (np.size(self.substrateWholeCellModelIDs), 2)
        )
        metabolites = np.asarray(
            _get(knowledgeBase, "metabolites"), dtype=object
        ).ravel()
        for local, global_index in zip(
            _idx(self.substrateMetaboliteLocalIndexs),
            _idx(self.substrateMetaboliteGlobalIndexs[:, 0]),
        ):
            metabolite = metabolites[global_index]
            formula = _get(metabolite, "empiricalFormula")
            carbon = (
                formula.get("C", 0)
                if isinstance(formula, dict)
                else getattr(formula, "C", 0)
            )
            limit = float(
                np.asarray(
                    self.exchangeRateUpperBound_carbon
                    if carbon > 0
                    else self.exchangeRateUpperBound_noncarbon
                ).item()
            )
            self.substrateExternalExchangeBounds[local] = [
                max(
                    -limit,
                    np.asarray(_get(metabolite, "exchangeLowerBound")).item(),
                ),
                min(
                    limit,
                    np.asarray(_get(metabolite, "exchangeUpperBound")).item(),
                ),
            ]
        self.substrateExternalExchangeBounds *= 6.0221417930e23 / (1000 * 3600)
        self.monomerCompartments = np.array(
            _get(knowledgeBase, "proteinMonomerCompartments"), copy=True
        )

    def formulateFBA(
        self,
        metabolismProduction=None,
        unaccountedEnergyConsumption=None,
        simplifyNetwork=True,
    ):
        from .metabolism_fba import formulate_fba

        return formulate_fba(
            self,
            metabolismProduction,
            unaccountedEnergyConsumption,
            simplifyNetwork,
        )

    def calcEffectiveFBANetwork(self, n=4):
        from .metabolism_fitting import effective_network

        return effective_network(self, n)

    def calcEnzymeKineticallyLimitedReactions(
        self, substrates, enzymes, fbaObj, fbaSMat, fbaRxnBounds
    ):
        from .metabolism_fitting import limited_reactions

        return limited_reactions(
            self, substrates, enzymes, fbaObj, fbaSMat, fbaRxnBounds
        )

    def fitEnzymes(self, growth0, substrates, enzymes):
        from .metabolism_fitting import fit_enzymes

        return fit_enzymes(self, growth0, substrates, enzymes)

    def calcMinMaxEnzymes(self, substrates, enzymes):
        from .metabolism_fitting import min_max_enzymes

        return min_max_enzymes(self, substrates, enzymes)

    def adjustEnzymeExpressionSparsely(
        self, growth0, substrates, enzymes, initEnzymes
    ):
        from .metabolism_fit_sparse import adjust_enzyme_expression_sparsely

        return adjust_enzyme_expression_sparsely(
            self, growth0, substrates, enzymes, initEnzymes
        )

    def adjustEnzymeExpressionSparselyHelper(
        self, growth0, substrates, enzymes, initEnzymes, tfs
    ):
        from .metabolism_fit_sparse import (
            adjust_enzyme_expression_sparsely_helper,
        )

        return adjust_enzyme_expression_sparsely_helper(
            self, growth0, substrates, enzymes, initEnzymes, tfs
        )

    def calcResourceRequirements_LifeCycle(self, constants=None, states=None):
        from ..process_base import _get

        if states is None:
            raise ValueError(
                "Metabolism life-cycle requirements need fitted initial states"
            )
        growth = np.asarray(self.metabolicReaction.growth0).item()
        if self.macromoleculeStateInitialization != "expected":
            growth *= np.asarray(
                self.macromoleculeStateInitializationGrowthFactor
            ).item()
        substrates = np.zeros_like(self.substrates, dtype=float)
        ix = _idx(self.substrateMetaboliteLocalIndexs)
        substrates[ix, :] = self.substrates[ix, :]
        flat = substrates.ravel(order="F").copy()
        ix = _idx(self.substrateIndexs_internalExchangedLimitedMetabolites)
        flat[ix] = (
            -growth
            * np.asarray(self.metabolismRecyclingProduction).ravel(order="F")[
                ix
            ]
        )
        substrates = flat.reshape(substrates.shape, order="F")
        enzymes = np.zeros(np.size(self.enzymeWholeCellModelIDs))
        for kind, key in (("Monomer", "monomers0"), ("Complex", "complexs0")):
            initial = np.asarray(_get(states, key)).ravel(order="F")
            local = _idx(getattr(self, "substrate" + kind + "LocalIndexs"))
            global_index = _idx(
                np.asarray(getattr(self, "substrate" + kind + "GlobalIndexs"))[
                    :, 0
                ]
            )
            substrates[local, :] = initial[global_index, None]
            enzymes[_idx(getattr(self, "enzyme" + kind + "LocalIndexs"))] = (
                initial[_idx(getattr(self, "enzyme" + kind + "GlobalIndexs"))]
            )
        enzymes = self.fitEnzymes(growth, substrates, enzymes)
        minimum, maximum = self.calcMinMaxEnzymes(substrates, enzymes)
        zero = np.zeros(np.asarray(self.substrateWholeCellModelIDs).shape)
        return (
            zero,
            zero.copy(),
            minimum.reshape(-1, 1),
            maximum.reshape(-1, 1),
        )

    def calcResourceRequirements_Current(self):
        result = np.zeros_like(np.asarray(self.substrates), dtype=float)
        exchanged = _idx(self.substrateIndexs_externalExchangedMetabolites)
        extracellular = _idx(self.compartmentIndexs_extracellular)
        rxns = _idx(self.fbaReactionIndexs_metaboliteExternalExchange)
        dry = np.asarray(self.mass.cellDry).sum()
        result[np.ix_(exchanged, extracellular)] = (
            np.asarray(self.fbaReactionBounds)[rxns, 1, None]
            * dry
            * float(np.asarray(self.stepSizeSec).item())
        )
        growth = np.asarray(self.metabolicReaction.growth).item()
        recycle = _dense(self.metabolismRecyclingProduction)
        factor = 2 * max(
            dry
            / np.asarray(self.mass.cellInitialDryWeight).item()
            * np.log(2)
            / np.asarray(self.cellCycleLength).item(),
            growth,
        )
        return result - recycle * factor

    def initializeState(self):
        bounds = self.calcFluxBounds(
            self.substrates,
            self.enzymes,
            self.fbaReactionBounds,
            self.fbaEnzymeBounds,
        )
        growth, fluxes = self.calcGrowthRate(bounds)
        self.metabolicReaction.growth = np.asarray([[growth]])
        self.metabolicReaction.fluxs = fluxes

    def calcGrowthRate(
        self, fluxBounds, fbaObj=None, fbaSMat=None, return_full=False
    ):
        fluxBounds = np.asarray(fluxBounds, dtype=float)
        lo, hi = np.maximum(
            fluxBounds[:, 0], -float(np.asarray(self.realmax).item())
        ), np.minimum(fluxBounds[:, 1], float(np.asarray(self.realmax).item()))
        objective = np.asarray(
            self.fbaObjective if fbaObj is None else fbaObj, dtype=float
        ).reshape(-1)
        stoich = _dense(
            self.fbaReactionStoichiometryMatrix if fbaSMat is None else fbaSMat
        )
        opts = _matlab_options(getattr(self, "linearProgrammingOptions", {}))
        solver_opts = _matlab_options(opts.get("solverOptions", {}))
        glpk_opts = _matlab_options(solver_opts.get("glpk", {}))
        backend = opts.get("backend", "glpk444")
        result = solve_lp(
            {
                "maximize": True,
                "f": objective,
                "A": stoich,
                "b": np.asarray(self.fbaRightHandSide).reshape(-1),
                "lb": lo,
                "ub": hi,
                "constraint_types": "S",
                "variable_types": "C",
                "backend": backend,
                "glpk_options": glpk_opts,
            }
        )
        if result.get("status") == "solver_error":
            raise RuntimeError(
                f"Metabolism solver infrastructure failure: {result.get('errorMsg', 'unknown solver error')}"
            )
        x = np.asarray(result["x"], dtype=float)
        if result["errorFlag"]:
            warnings.warn(
                f"Metabolism FBA failed ({result.get('errorMsg', 'solver error')}); using zero fluxes",
                RuntimeWarning,
            )
            x = np.zeros_like(lo)
        x = np.maximum(np.minimum(x, hi), lo)
        growth = x[_idx(self.fbaReactionIndexs_biomassProduction)]
        reaction_fluxes = np.zeros(
            _dense(self.reactionStoichiometryMatrix).shape[1]
        )
        reaction_fluxes[_idx(self.reactionIndexs_fba)] = x[
            _idx(self.fbaReactionIndexs_metabolicConversion)
        ]
        if return_full:
            fba_reduced = np.asarray(
                result["lambda"]["reducedCosts"], dtype=float
            )
            reduced = np.zeros(
                _dense(self.reactionStoichiometryMatrix).shape[1]
            )
            reduced[_idx(self.reactionIndexs_fba)] = fba_reduced[
                _idx(self.fbaReactionIndexs_metabolicConversion)
            ]
            fba_shadow = np.asarray(
                result["lambda"]["shadowPrices"], dtype=float
            )
            shadow = np.zeros(np.asarray(self.substrates).size)
            shadow[_idx(self.substrateIndexs_fba)] = fba_shadow[
                _idx(self.fbaSubstrateIndexs_substrates)
            ]
            shadow = shadow.reshape(
                np.asarray(self.substrates).shape, order="F"
            )
            return (
                float(growth[0]),
                reaction_fluxes,
                x,
                reduced,
                fba_reduced,
                shadow,
                fba_shadow,
            )
        return float(growth[0]), reaction_fluxes

    def calcFluxBounds(
        self,
        substrates,
        enzymes,
        fbaReactionBounds,
        fbaEnzymeBounds,
        applyEnzymeKineticBounds=True,
        applyEnzymeBounds=True,
        applyDirectionalityBounds=True,
        applyExternalMetaboliteBounds=True,
        applyInternalMetaboliteBounds=True,
        applyProteinBounds=True,
    ):
        cat = _dense(self.fbaReactionCatalysisMatrix)
        rxn_enzymes = cat @ np.asarray(enzymes)
        n = _dense(self.fbaReactionStoichiometryMatrix).shape[1]
        lower, upper = np.full(n, -np.inf), np.full(n, np.inf)
        eb, rb = np.asarray(fbaEnzymeBounds), np.asarray(fbaReactionBounds)
        if applyEnzymeKineticBounds:
            with np.errstate(invalid="ignore"):
                lower = np.fmax(lower, eb[:, 0] * rxn_enzymes.reshape(-1))
                upper = np.fmin(upper, eb[:, 1] * rxn_enzymes.reshape(-1))
        if applyEnzymeBounds:
            no_enzyme = np.any(cat != 0, axis=1) & (
                rxn_enzymes.reshape(-1) <= 0
            )
            lower[no_enzyme] = upper[no_enzyme] = 0
        if applyDirectionalityBounds:
            for name in (
                "fbaReactionIndexs_metabolicConversion",
                "fbaReactionIndexs_metaboliteInternalExchange",
                "fbaReactionIndexs_biomassExchange",
                "fbaReactionIndexs_biomassProduction",
            ):
                ix = _idx(getattr(self, name))
                lower[ix] = np.fmax(lower[ix], rb[ix, 0])
                upper[ix] = np.fmin(upper[ix], rb[ix, 1])
        if applyExternalMetaboliteBounds:
            ix = _idx(self.fbaReactionIndexs_metaboliteExternalExchange)
            si, ci = _idx(
                self.substrateIndexs_externalExchangedMetabolites
            ), _idx(self.compartmentIndexs_extracellular)
            dry = np.asarray(self.mass.cellDry).sum()
            lower[ix] = np.fmax(lower[ix], rb[ix, 0] * dry)
            upper[ix] = np.fmin(upper[ix], rb[ix, 1] * dry)
            upper[ix] = np.fmin(
                upper[ix],
                np.asarray(substrates)[np.ix_(si, ci)].reshape(-1)
                / float(np.asarray(self.stepSizeSec).item()),
            )
        if applyInternalMetaboliteBounds:
            ix = _idx(self.fbaReactionIndexs_metaboliteInternalLimitedExchange)
            si = _idx(self.substrateIndexs_internalExchangedLimitedMetabolites)
            lower[ix] = np.fmax(
                lower[ix],
                -np.asarray(substrates).ravel(order="F")[si]
                / float(np.asarray(self.stepSizeSec).item()),
            )
        if applyProteinBounds:
            # Determine exactly which metabolic reactions consume a limiting protein.
            rxns = _idx(self.reactionIndexs_fba)
            mon = _idx(self.substrateMonomerLocalIndexs)
            com = _idx(self.substrateComplexLocalIndexs)
            protein_rows = np.r_[mon, com]
            if protein_rows.size:
                sto = _dense(self.reactionStoichiometryMatrix)
                sto = np.take(np.take(sto, protein_rows, axis=0), rxns, axis=1)
                needs_protein = (
                    np.any(sto != 0, axis=tuple(range(2, sto.ndim)))
                    if sto.ndim > 2
                    else sto != 0
                )
                available = (
                    np.asarray(self.proteinLimitableProteinComposition)
                    @ np.asarray(substrates)[
                        _idx(self.substrateIndexs_limitableProteins), :
                    ]
                )
                if sto.ndim > 2:
                    limited = np.any(
                        np.any(
                            (sto != 0) & (available[:, None, :] <= 0), axis=2
                        ),
                        axis=0,
                    )
                else:
                    limited = np.any(
                        needs_protein & (available.reshape(-1, 1) <= 0), axis=0
                    )
                ix = _idx(self.fbaReactionIndexs_metabolicConversion)[limited]
                lower[ix] = upper[ix] = 0
        return np.column_stack((lower, upper))

    def evolveState(self):
        bounds = self.calcFluxBounds(
            self.substrates,
            self.enzymes,
            self.fbaReactionBounds,
            self.fbaEnzymeBounds,
        )
        growth, fluxes, fba_fluxes, _, _, _, _ = self.calcGrowthRate(
            bounds, return_full=True
        )
        self.metabolicReaction.growth = np.asarray([[growth]])
        fba_fluxes = np.asarray(fba_fluxes, dtype=float)
        stream, substrates = self.randStream, np.asarray(self.substrates)
        ext, extracellular = _idx(
            self.substrateIndexs_externalExchangedMetabolites
        ), _idx(self.compartmentIndexs_extracellular)
        exchange = _idx(self.fbaReactionIndexs_metaboliteExternalExchange)
        substrates[np.ix_(ext, extracellular)] -= stream.stochasticRound(
            fba_fluxes[exchange, None]
            * float(np.asarray(self.stepSizeSec).item())
        )
        internal = _idx(self.substrateIndexs_internalExchangedMetabolites)
        int_rxn = _idx(self.fbaReactionIndexs_metaboliteInternalExchange)
        subs_linear = substrates.ravel(order="F")
        subs_linear[internal] += stream.stochasticRound(fba_fluxes[int_rxn])
        substrates = subs_linear.reshape(substrates.shape, order="F")
        substrates += stream.stochasticRound(
            _dense(self.metabolismNewProduction)
            * growth
            * float(np.asarray(self.stepSizeSec).item())
        )
        atp = _idx(self.substrateIndexs_atpHydrolysis)
        energy = np.asarray(
            stream.stochasticRound(
                np.asarray(self.unaccountedEnergyConsumption)
                * growth
                * float(np.asarray(self.stepSizeSec).item())
            )
        ).item()
        subs_linear = substrates.ravel(order="F")
        subs_linear[atp] += np.asarray([-1, -1, 1, 1, 1]).reshape(-1) * energy
        substrates = subs_linear.reshape(substrates.shape, order="F")
        metabolites = _idx(self.substrateMetaboliteLocalIndexs[:, 0])
        substrates[metabolites, :] = np.maximum(0, substrates[metabolites, :])
        self.substrates = substrates
        self.metabolicReaction.fluxs = fluxes
        self.mass.calcMass()
        self.geometry.calculateVolume()

    def initializeFittedConstants(self, *args, **kwargs):
        raise NotImplementedError(
            "Metabolism fitted constants require MATLAB fitting data"
        )
