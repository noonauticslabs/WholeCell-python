"""RNA maturation by sequential, resource-limited modification reactions."""

from __future__ import annotations

import numpy as np
from ..process_base import _get, _set
from ..reaction_process import ReactionProcess


class RNAModification(ReactionProcess):
    """MATLAB RNAModification port; public index vectors remain one-based."""

    def __init__(
        self,
        wholeCellModelID="Process_RNAModification",
        name="RNA Modification",
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
            super().initializeConstants(
                knowledgeBase, simulation, options or {}
            )
            self._initialize_from_knowledge_base(knowledgeBase)
            return
        required = (
            "reactionStoichiometryMatrix",
            "reactionCatalysisMatrix",
            "reactionModificationMatrix",
            "enzymeBounds",
            "speciesReactantMatrix",
            "speciesReactantByproductMatrix",
            "speciesIndexs_enzymes",
            "rnaMask_modified",
            "rnaMask_unmodified",
            "rnaIndexs_modified",
            "rnaIndexs_unmodified",
        )
        missing = [k for k in required if not hasattr(self, k)]
        if missing:
            raise NotImplementedError(
                "RNAModification snapshot lacks initialized network fields: "
                + ", ".join(missing)
            )

    def _initialize_from_knowledge_base(self, knowledgeBase):
        for name, ids in {
            "amet": ["AMET"],
            "fthf5": ["FTHF5"],
            "cys": ["CYS"],
            "gly": ["GLY"],
            "lys": ["LYS"],
            "atp": ["ATP"],
            "water": ["H2O"],
            "hydrogen": ["H"],
            "nmp": ["AMP", "CMP", "GMP", "UMP"],
        }.items():
            setattr(self, f"substrateIndexs_{name}", self.substrateIndexs(ids))
        metabolites = np.asarray(
            _get(knowledgeBase, "metabolites"), dtype=object
        ).ravel(order="F")
        substrate_metabolites = np.asarray(
            self.substrateMetaboliteGlobalIndexs, dtype=int
        ).ravel(order="F")
        categories = np.asarray(
            [
                _get(metabolites[i - 1], "category")
                for i in substrate_metabolites
            ],
            dtype=object,
        )
        self.substrateIndexs_modifiedNMP = (
            np.flatnonzero(
                categories == "modified ribonucleotide monophosphate"
            ).reshape(-1, 1)
            + 1
        )

        enzyme_ids = {
            "rRNA16SDimethyladenosineTransferase": "MG_463_MONOMER",
            "rRNA16SMethyltransferaseGidB": "MG_380_MONOMER",
            "rRNA23SMethyltransferaseI": "MG_252_DIMER",
            "rRNA23SMethyltransferaseII": "MG_346_DIMER",
            "rRNA23SPseudouridineSynthaseI": "MG_209_MONOMER",
            "rRNA23SPseudouridineSynthaseII": "MG_370_MONOMER",
            "tRNAGuanineN1Methyltransferase": "MG_445_DIMER",
            "tRNAGuanineN7Methyltransferase": "MG_347_DIMER",
            "tRNALysidineSynthetase": "MG_084_TETRAMER",
            "tRNAPseudouridineSynthase": "MG_182_DIMER",
            "tRNAUracil2Sulfurtransferase": "MG_295_MONOMER",
            "tRNAUracil4Sulfurtransferase": "MG_372_DIMER",
            "tRNAUracil5Carboxymethylaminomethyl": "MG_008_379_TETRAMER",
        }
        for name, identifier in enzyme_ids.items():
            setattr(
                self, f"enzymeIndexs_{name}", self.enzymeIndexs([identifier])
            )

        rna_ids = np.asarray(
            _get(self.rna, "wholeCellModelIDs"), dtype=object
        ).ravel(order="F")
        processed = np.asarray(
            _get(self.rna, "processedIndexs"), dtype=int
        ).ravel(order="F")
        mature = np.asarray(_get(self.rna, "matureIndexs"), dtype=int).ravel(
            order="F"
        )
        self.unmodifiedRNAWholeCellModelIDs = rna_ids[processed - 1].reshape(
            -1, 1
        )
        self.modifiedRNAWholeCellModelIDs = rna_ids[mature - 1].reshape(-1, 1)

        nmp = np.asarray(self.substrateIndexs_nmp, dtype=int).ravel() - 1
        modified_nmp = (
            np.asarray(self.substrateIndexs_modifiedNMP, dtype=int).ravel() - 1
        )
        sto = np.asarray(self.reactionStoichiometryMatrix)
        sto[nmp] = np.maximum(sto[nmp], 0)
        sto[modified_nmp] = 0
        composition = np.asarray(_get(self.rna, "matureRNAGeneComposition"))
        self.reactionModificationMatrix = (
            np.asarray(self.reactionModificationMatrix) > 0
        ) @ composition
        self.rnaMask_modified = np.any(self.reactionModificationMatrix, axis=0)
        self.rnaMask_unmodified = ~self.rnaMask_modified
        self.rnaIndexs_modified = np.flatnonzero(self.rnaMask_modified) + 1
        self.rnaIndexs_unmodified = np.flatnonzero(self.rnaMask_unmodified) + 1
        self.initializeSpeciesNetwork()

    def initializeSpeciesNetwork(self):
        num_metabolites = np.asarray(self.reactionStoichiometryMatrix).shape[0]
        num_enzymes = np.asarray(self.reactionCatalysisMatrix).shape[1]
        modified = np.flatnonzero(
            np.asarray(self.rnaMask_modified, dtype=bool)
        )
        self.speciesIndexs_enzymes = (
            np.arange(num_enzymes, dtype=int).reshape(-1, 1)
            + num_metabolites
            + 1
        )
        cat = np.asarray(self.reactionCatalysisMatrix)
        sto = np.asarray(self.reactionStoichiometryMatrix)
        mod = np.asarray(self.reactionModificationMatrix)[:, modified]
        scaled_cat = cat / (
            np.asarray(self.enzymeBounds)[:, 1:2]
            * float(np.asarray(self.stepSizeSec).item())
        )
        base = np.concatenate((-sto.T, scaled_cat), axis=1)
        self.speciesReactantByproductMatrix = np.concatenate(
            (mod.T @ base, np.eye(modified.size)), axis=1
        )
        self.speciesReactantMatrix = np.concatenate(
            (mod.T @ np.maximum(0, base), np.eye(modified.size)), axis=1
        )

    def copyFromState(self):
        super().copyFromState()
        counts = np.asarray(_get(self.rna, "counts"))
        if counts.ndim == 2:
            counts = counts[:, :, None]
        cyt = int(np.asarray(self.compartment.cytosolIndexs).ravel()[0]) - 1
        proc = (
            np.asarray(_get(self.rna, "processedIndexs"), dtype=int).ravel()
            - 1
        )
        mature = (
            np.asarray(_get(self.rna, "matureIndexs"), dtype=int).ravel() - 1
        )
        self.unmodifiedRNAs = counts[proc, cyt, :].reshape((len(proc), 1, -1))
        self.modifiedRNAs = counts[mature, cyt, :].reshape(
            (len(mature), 1, -1)
        )
        if counts.shape[2] == 1:
            self.unmodifiedRNAs = self.unmodifiedRNAs[:, :, 0]
            self.modifiedRNAs = self.modifiedRNAs[:, :, 0]

    def copyToState(self):
        super().copyToState()
        counts = np.asarray(_get(self.rna, "counts"))
        if counts.ndim == 2:
            counts = counts[:, :, None]
        cyt = int(np.asarray(self.compartment.cytosolIndexs).ravel()[0]) - 1
        proc = (
            np.asarray(_get(self.rna, "processedIndexs"), dtype=int).ravel()
            - 1
        )
        mature = (
            np.asarray(_get(self.rna, "matureIndexs"), dtype=int).ravel() - 1
        )
        # Keep the RNA-by-time columns explicit.  NumPy's mixed advanced
        # indexing can reorder/broadcast axes when ``proc``/``mature`` arrive
        # as column vectors, while MATLAB's indexed assignment preserves the
        # (RNA, 1, time) shape.
        unmodified = np.asarray(self.unmodifiedRNAs).reshape((len(proc), -1))
        modified = np.asarray(self.modifiedRNAs).reshape((len(mature), -1))
        for t in range(counts.shape[2]):
            counts[proc, cyt, t] = unmodified[:, t]
            counts[mature, cyt, t] = modified[:, t]
        _set(self.rna, "counts", counts)

    def allocateMemoryForState(self, numTimePoints=1):
        n = len(np.asarray(self.unmodifiedRNAWholeCellModelIDs).ravel())
        shape = (n, 1) if numTimePoints == 1 else (n, 1, numTimePoints)
        self.unmodifiedRNAs = np.zeros(shape)
        self.modifiedRNAs = np.zeros(shape)

    def calcResourceRequirements_Current(self):
        # MATLAB limits each modification by both available RNA substrates and
        # the enzymes that can turn it over during this step.
        enzyme_limit = np.ceil(
            np.asarray(self.reactionCatalysisMatrix)
            @ np.asarray(self.enzymes).reshape(-1)
            * float(np.asarray(self.stepSizeSec).item())
        )
        rna_limit = np.asarray(self.reactionModificationMatrix) @ np.asarray(
            self.unmodifiedRNAs
        ).reshape(-1)
        n_rxn = np.minimum(enzyme_limit, rna_limit)
        return (
            np.maximum(0, -np.asarray(self.reactionStoichiometryMatrix))
            @ n_rxn
        )

    def calcResourceRequirements_LifeCycle(self, states):
        mod = np.asarray(self.reactionModificationMatrix)
        sto = np.asarray(self.reactionStoichiometryMatrix)
        cat = np.asarray(self.reactionCatalysisMatrix)
        prod = np.asarray(_get(states, "rnaProductions")).reshape(-1)
        prod0 = np.asarray(_get(states, "rnaProductions0")).reshape(-1)
        return (
            np.maximum(0, -sto) @ (mod @ prod),
            np.maximum(0, sto) @ (mod @ prod),
            2 * cat.T @ ((mod @ prod0) / np.asarray(self.enzymeBounds)[:, 1]),
        )

    def initializeState(self):
        pass

    def evolveState(self):
        modified = np.asarray(self.modifiedRNAs).reshape(-1).copy()
        unmodified = np.asarray(self.unmodifiedRNAs).reshape(-1).copy()
        mask_m = np.asarray(self.rnaMask_modified, dtype=bool).ravel()
        mask_u = np.asarray(self.rnaMask_unmodified, dtype=bool).ravel()
        modified[mask_u] += unmodified[mask_u]
        unmodified[mask_u] = 0
        if not np.any(unmodified):
            self.modifiedRNAs = modified.reshape(
                np.asarray(self.modifiedRNAs).shape
            )
            self.unmodifiedRNAs = unmodified.reshape(
                np.asarray(self.unmodifiedRNAs).shape
            )
            return
        n = int(mask_m.sum())
        net = np.asarray(self.speciesReactantByproductMatrix)
        need = np.asarray(self.speciesReactantMatrix)
        species = np.r_[
            np.asarray(self.substrates).ravel(),
            np.asarray(self.enzymes).ravel(),
            unmodified[mask_m],
        ].astype(float)
        enz = np.asarray(self.speciesIndexs_enzymes, dtype=int).ravel() - 1
        water = int(np.asarray(self.substrateIndexs_water).ravel()[0]) - 1
        hydrogen = (
            int(np.asarray(self.substrateIndexs_hydrogen).ravel()[0]) - 1
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            limits = np.divide(
                species[None, :],
                need,
                out=np.full((n, need.shape[1]), np.inf),
                where=need != 0,
            )
            limits[:, [water, hydrogen]] = np.nan
            inactive = (
                np.nan_to_num(np.nanmin(limits, axis=1), nan=0, posinf=np.inf)
                <= 0
            )
        flux = np.zeros(n, dtype=np.int64)
        other = np.r_[0 : enz[0], enz[-1] + 1 : net.shape[1]]
        while True:
            with np.errstate(divide="ignore", invalid="ignore"):
                lim = np.divide(
                    species[None, :],
                    np.maximum(0, net),
                    out=np.full((n, net.shape[1]), np.inf),
                    where=np.maximum(0, net) != 0,
                )
                lim[:, [water, hydrogen]] = np.nan
                enzyme_limit = np.nanmin(lim[:, enz], axis=1)
                resource_limit = np.nanmin(lim[:, other], axis=1)
            rnd = np.asarray(
                self.randStream.stochasticRound(enzyme_limit)
            ).reshape(-1)
            r = np.minimum(rnd, resource_limit)
            r[inactive | ~np.isfinite(r) | (r < 1)] = 0
            if not np.any(r):
                break
            selected = (
                int(
                    np.asarray(
                        self.randStream.randsample(n, 1, True, r)
                    ).reshape(-1)[0]
                )
                - 1
            )
            flux[selected] += 1
            species -= net[selected]
        if flux.any():
            substrate_change = (
                np.asarray(self.reactionStoichiometryMatrix)
                @ np.asarray(self.reactionModificationMatrix)[:, mask_m]
                @ flux
            )
            self.substrates = np.asarray(
                self.substrates
            ) + substrate_change.reshape(np.asarray(self.substrates).shape)
            unmodified[mask_m] -= flux
            modified[mask_m] += flux
        self.unmodifiedRNAs = unmodified.reshape(
            np.asarray(self.unmodifiedRNAs).shape
        )
        self.modifiedRNAs = modified.reshape(
            np.asarray(self.modifiedRNAs).shape
        )


RNAmodification = RNAModification
