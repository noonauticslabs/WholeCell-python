"""tRNA and tmRNA aminoacylation process (MATLAB tRNAAminoacylation)."""

from __future__ import annotations

import numpy as np
from ..process_base import Process, _get, _set
from ..reaction_process import ReactionProcess


class TRNAAminoacylation(ReactionProcess):
    """Greedy resource-limited aminoacylation with MATLAB one-based indexes."""

    def __init__(
        self,
        wholeCellModelID="Process_tRNAAminoacylation",
        name="tRNA aminoacylation",
        **properties,
    ):
        super().__init__(wholeCellModelID, name, **properties)

    def initializeConstants(
        self, knowledgeBase=None, simulation=None, options=None, **kwargs
    ):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if knowledgeBase is not None or simulation is not None:
            if knowledgeBase is None or simulation is None:
                raise ValueError(
                    "knowledgeBase and simulation must be supplied together"
                )
            ReactionProcess.initializeConstants(
                self, knowledgeBase, simulation, options or {}
            )
            self.compartmentWholeCellModelIDs = np.asarray(
                _get(self.compartment, "wholeCellModelIDs"), dtype=object
            ).reshape(-1)[
                int(_get(self.compartment, "cytosolIndexs"))
                - 1 : int(_get(self.compartment, "cytosolIndexs"))
            ]
            self.compartmentIndexs_cytosol = 1

            # The source adds GLN, which cancels in KB reactions but is needed
            # by the edited glutamyl-amidotransferase transfer reaction.
            self.substrateWholeCellModelIDs = np.vstack(
                (
                    np.asarray(
                        self.substrateWholeCellModelIDs, dtype=object
                    ).reshape(-1, 1),
                    [["GLN"]],
                )
            )
            self.reactionStoichiometryMatrix = np.vstack(
                (
                    np.asarray(self.reactionStoichiometryMatrix),
                    np.zeros(
                        (
                            1,
                            np.asarray(self.reactionStoichiometryMatrix).shape[
                                1
                            ],
                        )
                    ),
                )
            )
            self.reactionCoenzymeMatrix = np.column_stack(
                (
                    np.asarray(self.reactionCoenzymeMatrix),
                    np.zeros(np.asarray(self.reactionCoenzymeMatrix).shape[0]),
                )
            )

            for name, ids in {
                "aminoAcids": [
                    "ALA",
                    "ARG",
                    "ASN",
                    "ASP",
                    "CYS",
                    "GLN",
                    "GLU",
                    "GLY",
                    "HIS",
                    "ILE",
                    "LEU",
                    "LYS",
                    "MET",
                    "PHE",
                    "PRO",
                    "SER",
                    "THR",
                    "TRP",
                    "TYR",
                    "VAL",
                ],
                "glutamate": ["GLU"],
                "glutamine": ["GLN"],
                "methionine": ["MET"],
                "fmethionine": ["FMET"],
                "atp": ["ATP"],
                "amp": ["AMP"],
                "adp": ["ADP"],
                "diphosphate": ["PPI"],
                "phosphate": ["PI"],
                "water": ["H2O"],
                "hydrogen": ["H"],
                "fthf10": ["FTHF10"],
                "thf": ["THF"],
            }.items():
                setattr(
                    self, f"substrateIndexs_{name}", self.substrateIndexs(ids)
                )

            rna_ids = np.asarray(
                _get(self.rna, "wholeCellModelIDs"), dtype=object
            ).ravel(order="F")
            mature = np.asarray(
                _get(self.rna, "matureIndexs"), dtype=int
            ).ravel(order="F")
            mature_trna = np.asarray(
                _get(self.rna, "matureTRNAIndexs"), dtype=int
            ).ravel(order="F")
            trna_ids = rna_ids[mature[mature_trna - 1] - 1]
            self.enzymeWholeCellModelIDs = np.vstack(
                (
                    np.asarray(
                        self.enzymeWholeCellModelIDs, dtype=object
                    ).reshape(-1, 1),
                    trna_ids.reshape(-1, 1),
                    [["MG_0004"]],
                )
            )
            ids_by_name = {
                "tRNASynthetases": [
                    "MG_292_TETRAMER",
                    "MG_378_MONOMER",
                    "MG_036_DIMER",
                    "MG_113_DIMER",
                    "MG_253_MONOMER",
                    "MG_462_MONOMER",
                    "MG_251_DIMER",
                    "MG_035_DIMER",
                    "MG_345_MONOMER",
                    "MG_266_MONOMER",
                    "MG_136_DIMER",
                    "MG_021_DIMER",
                    "MG_194_195_TETRAMER",
                    "MG_283_DIMER",
                    "MG_005_DIMER",
                    "MG_375_DIMER",
                    "MG_126_DIMER",
                    "MG_455_DIMER",
                    "MG_334_MONOMER",
                ],
                "tRNATransferases": [
                    "MG_098_099_100_TRIMER",
                    "MG_365_MONOMER",
                ],
                "tRNAGlutamylSynthetase": ["MG_462_MONOMER"],
                "tRNAMethionylSynthetase": ["MG_021_DIMER"],
                "tRNAGlutamylAmidotransferase": ["MG_098_099_100_TRIMER"],
                "tRNAMethionylFormyltransferase": ["MG_365_MONOMER"],
            }
            for name, ids in ids_by_name.items():
                setattr(self, f"enzymeIndexs_{name}", self.enzymeIndexs(ids))
            self.enzymeWholeCellModelIDs = (
                np.vstack(
                    (
                        np.asarray(self.enzymeWholeCellModelIDs, dtype=object),
                        [["MG_0004"]],
                    )
                )
                if "MG_0004"
                not in np.asarray(self.enzymeWholeCellModelIDs).ravel()[-1:]
                else self.enzymeWholeCellModelIDs
            )
            self.enzymeIndexs_tRNAs = self.enzymeIndexs(trna_ids)
            self.enzymeIndexs_tmRNA = self.enzymeIndexs(["MG_0004"])

            modification = np.asarray(self.reactionModificationMatrix).copy()
            modification[np.isnan(modification)] = 1
            aa_genes = np.any(modification, axis=0)
            composition = np.asarray(
                _get(self.rna, "matureRNAGeneComposition")
            )
            inverse = np.diag(1 / np.sum(composition, axis=0)) @ composition.T
            aa_rnas = inverse @ aa_genes
            self.aminoacylatedRNAGlobalIndexs = np.flatnonzero(aa_rnas) + 1
            self.aminoacylatedRNAtRNAIndexs = np.intersect1d(
                self.aminoacylatedRNAGlobalIndexs, mature_trna
            )
            self.freeRNAWholeCellModelIDs = rna_ids[
                mature[self.aminoacylatedRNAGlobalIndexs - 1] - 1
            ].reshape(-1, 1)
            aminoacylated_indices = np.asarray(
                _get(self.rna, "aminoacylatedIndexs"), dtype=int
            ).ravel(order="F")
            self.aminoacylatedRNAWholeCellModelIDs = rna_ids[
                aminoacylated_indices[self.aminoacylatedRNAGlobalIndexs - 1]
                - 1
            ].reshape(-1, 1)
            self.reactionModificationMatrix = modification[:, aa_genes]

            reaction_types = np.asarray(
                self.reactionTypes, dtype=object
            ).ravel(order="F")
            self.reactionIndexs_aminoacylation = (
                np.flatnonzero(reaction_types == "aminoacylation") + 1
            )
            self.reactionIndexs_transfer = (
                np.flatnonzero(reaction_types == "transfer") + 1
            )
            self.reactionIndexs_glutamylamidotransfer = self.reactionIndexs(
                ["MG502_Amidotransferase"]
            )
            self.reactionIndexs_methionylformyltransfer = self.reactionIndexs(
                ["MG488_Formyltransferase"]
            )
            self.reactionStoichiometryMatrix[
                np.asarray(self.substrateIndexs_glutamine).ravel() - 1,
                np.asarray(self.reactionIndexs_glutamylamidotransfer).ravel()
                - 1,
            ] = -1
            self.reactionStoichiometryMatrix[
                np.asarray(self.substrateIndexs_glutamate).ravel() - 1,
                np.asarray(self.reactionIndexs_glutamylamidotransfer).ravel()
                - 1,
            ] = 1
            self.reactionStoichiometryMatrix[
                np.asarray(self.substrateIndexs_methionine).ravel() - 1,
                np.asarray(self.reactionIndexs_methionylformyltransfer).ravel()
                - 1,
            ] = 0
            self.reactionStoichiometryMatrix[
                np.asarray(self.substrateIndexs_fmethionine).ravel() - 1,
                np.asarray(self.reactionIndexs_methionylformyltransfer).ravel()
                - 1,
            ] = 0

            Process.initializeConstants(
                self, knowledgeBase, simulation, options or {}
            )
            self.initializeSpeciesNetwork()
            self.monomerTRNACounts = np.asarray(
                _get(knowledgeBase, "proteinMonomerTRNACounts")
            )
            return
        required = (
            "reactionStoichiometryMatrix",
            "reactionCatalysisMatrix",
            "reactionModificationMatrix",
            "enzymeBounds",
            "speciesReactantMatrix",
            "speciesReactantByproductMatrix",
            "speciesIndexs_enzymes",
        )
        missing = [k for k in required if not hasattr(self, k)]
        if missing:
            raise NotImplementedError(
                "tRNAAminoacylation snapshot lacks initialized network fields: "
                + ", ".join(missing)
            )

    def initializeSpeciesNetwork(self):
        sto = np.asarray(self.reactionStoichiometryMatrix)
        cat = np.asarray(self.reactionCatalysisMatrix)
        mod = np.asarray(self.reactionModificationMatrix)
        num_metabolites, num_enzymes, num_rnas = (
            sto.shape[0],
            cat.shape[1],
            mod.shape[1],
        )
        scaled_cat = cat / (
            np.asarray(self.enzymeBounds)[:, 1:2]
            * float(np.asarray(self.stepSizeSec).item())
        )
        base = np.concatenate((-sto.T, scaled_cat), axis=1)
        self.speciesReactantByproductMatrix = np.concatenate(
            (mod.T @ base, np.eye(num_rnas)), axis=1
        )
        self.speciesReactantMatrix = np.concatenate(
            (mod.T @ np.maximum(0, base), np.eye(num_rnas)), axis=1
        )
        self.speciesIndexs_enzymes = (
            np.arange(num_enzymes, dtype=int).reshape(-1, 1)
            + num_metabolites
            + 1
        )

    def _evolution_constants(self):
        """Prepare model arrays and derived masks for one evolve call.

        These arrays are public NumPy attributes and may be edited in place,
        so derived values are deliberately not cached across calls.
        """
        sto = np.asarray(self.reactionStoichiometryMatrix)
        cat = np.asarray(self.reactionCatalysisMatrix)
        mod = np.asarray(self.reactionModificationMatrix)
        need = np.asarray(self.speciesReactantMatrix)
        net = np.asarray(self.speciesReactantByproductMatrix)
        enzix = np.asarray(self.speciesIndexs_enzymes, dtype=int).ravel() - 1
        water = int(np.asarray(self.substrateIndexs_water).ravel()[0]) - 1
        hydrogen = (
            int(np.asarray(self.substrateIndexs_hydrogen).ravel()[0]) - 1
        )
        other = np.r_[0 : enzix[0], enzix[-1] + 1 : net.shape[1]]
        # Reused in every greedy selection round of this step, which may
        # select hundreds of reactions.
        net_consumption = np.maximum(0, net)
        return (
            sto,
            cat,
            mod,
            need,
            net,
            enzix,
            water,
            hydrogen,
            other,
            net_consumption,
        )

    def copyFromState(self):
        super().copyFromState()
        ix = (
            np.asarray(self.aminoacylatedRNAGlobalIndexs, dtype=int).ravel()
            - 1
        )
        cyt = (
            np.asarray(self.compartment.cytosolIndexs, dtype=int).ravel()[0]
            - 1
        )
        counts = np.asarray(_get(self.rna, "counts"))
        if counts.ndim == 2:
            counts = counts[:, :, None]
        mature = (
            np.asarray(_get(self.rna, "matureIndexs"), dtype=int).ravel()[ix]
            - 1
        )
        aa = (
            np.asarray(
                _get(self.rna, "aminoacylatedIndexs"), dtype=int
            ).ravel()[ix]
            - 1
        )
        self.freeRNAs = counts[mature, cyt, :].reshape((len(ix), 1, -1))
        self.aminoacylatedRNAs = counts[aa, cyt, :].reshape((len(ix), 1, -1))
        if counts.shape[2] == 1:
            self.freeRNAs = self.freeRNAs[:, :, 0]
            self.aminoacylatedRNAs = self.aminoacylatedRNAs[:, :, 0]

    def copyToState(self):
        super().copyToState()
        ix = (
            np.asarray(self.aminoacylatedRNAGlobalIndexs, dtype=int).ravel()
            - 1
        )
        cyt = (
            np.asarray(self.compartment.cytosolIndexs, dtype=int).ravel()[0]
            - 1
        )
        counts = np.asarray(_get(self.rna, "counts"))
        if counts.ndim == 2:
            counts = counts[:, :, None]
        mature = (
            np.asarray(_get(self.rna, "matureIndexs"), dtype=int).ravel()[ix]
            - 1
        )
        aa = (
            np.asarray(
                _get(self.rna, "aminoacylatedIndexs"), dtype=int
            ).ravel()[ix]
            - 1
        )
        free = np.asarray(self.freeRNAs).reshape((len(ix), -1))
        charged = np.asarray(self.aminoacylatedRNAs).reshape((len(ix), -1))
        counts[mature, cyt, :] = free
        counts[aa, cyt, :] = charged
        _set(self.rna, "counts", counts)

    def allocateMemoryForState(self, numTimePoints=1):
        n = len(np.asarray(self.aminoacylatedRNAGlobalIndexs).ravel())
        shape = (n, 1) if numTimePoints == 1 else (n, 1, numTimePoints)
        self.freeRNAs = np.zeros(shape)
        self.aminoacylatedRNAs = np.zeros(shape)

    def initializeState(self):
        total = np.asarray(self.freeRNAs) + np.asarray(self.aminoacylatedRNAs)
        self.aminoacylatedRNAs = np.ceil(2 * total / 3)
        self.freeRNAs = total - self.aminoacylatedRNAs

    def calcResourceRequirements_Current(self):
        mat = np.asarray(self.reactionModificationMatrix)
        cat = np.asarray(self.reactionCatalysisMatrix)
        demand = np.minimum(
            mat
            @ (
                np.asarray(self.freeRNAs).reshape(-1)
                + np.asarray(self.aminoacylatedRNAs).reshape(-1)
                + 1
            ),
            cat
            @ np.asarray(self.enzymes).reshape(-1)[: cat.shape[1]]
            * float(np.asarray(self.stepSizeSec).item()),
        )
        return (
            np.maximum(0, -np.asarray(self.reactionStoichiometryMatrix))
            @ demand
        )

    def evolveState(self):
        free = np.asarray(self.freeRNAs).reshape(-1).copy()
        charged = np.asarray(self.aminoacylatedRNAs).reshape(-1).copy()
        if not np.any(free):
            return
        n = free.size
        (
            sto,
            cat,
            mod,
            need,
            net,
            enzix,
            water,
            hydrogen,
            other,
            net_consumption,
        ) = self._evolution_constants()
        nenz = cat.shape[1]
        species = np.r_[
            np.asarray(self.substrates).reshape(-1),
            np.asarray(self.enzymes).reshape(-1)[:nenz],
            free,
        ]
        limits = np.divide(
            species[None, :],
            need,
            out=np.full((n, need.shape[1]), np.inf),
            where=need != 0,
        )
        limits[:, [water, hydrogen]] = np.nan
        initial_limits = np.nanmin(limits, axis=1)
        # MATLAB first replaces Inf, NaN, and negative limits with zero, then
        # marks nonpositive reactions inactive. In particular, +Inf is inactive.
        inactive = ~np.isfinite(initial_limits) | (initial_limits <= 0)
        flux = np.zeros(n, dtype=float)
        while True:
            lim = np.divide(
                species[None, :],
                net_consumption,
                out=np.full((n, net.shape[1]), np.inf),
                where=net_consumption != 0,
            )
            lim[:, [water, hydrogen]] = np.nan
            with np.errstate(all="ignore"):
                enzyme = np.nanmin(lim[:, enzix], axis=1)
                rest = np.nanmin(lim[:, other], axis=1)
            r = np.minimum(
                self.randStream.stochasticRound(
                    np.asarray(enzyme).reshape(-1, 1)
                ).reshape(-1),
                rest,
            )
            r[inactive | ~np.isfinite(r) | (r < 1)] = 0
            if not np.any(r):
                break
            weights = r / r.sum()
            edges = np.minimum(np.r_[0, np.cumsum(weights)], 1)
            n_reactions = int(np.min(r[r > 0]))
            if n_reactions <= 1:
                draw = float(self.randStream.rand(1, 1).item())
                # MATLAB histc bins [edge[i], edge[i+1]); an exact interior
                # edge belongs to the bin on its right.
                selected = int(np.searchsorted(edges, draw, side="right") - 1)
                flux[selected] += 1
                species -= net[selected]
            else:
                # Source batches up to the smallest positive reaction capacity
                # using one histc call, then applies the reaction matrix once.
                edges[-1] = 1.1
                draws = np.asarray(
                    self.randStream.rand(n_reactions, 1)
                ).reshape(-1)
                selected_bins = np.searchsorted(edges, draws, side="right") - 1
                selected = np.bincount(selected_bins, minlength=n).astype(
                    float
                )
                flux += selected
                species -= selected @ net
        if flux.any():
            self.substrates = np.asarray(self.substrates) + (
                sto @ mod @ flux
            ).reshape(np.asarray(self.substrates).shape)
            free -= flux
            charged += flux
            self.freeRNAs = free.reshape(np.asarray(self.freeRNAs).shape)
            self.aminoacylatedRNAs = charged.reshape(
                np.asarray(self.aminoacylatedRNAs).shape
            )


# MATLAB runtime class spelling used by the snapshot registry.
tRNAAminoacylation = TRNAAminoacylation
