"""Protein adduction and ligation maturation (MATLAB ProteinModification)."""

from __future__ import annotations

import numpy as np
from ..process_base import _get, _set
from ..reaction_process import ReactionProcess


class ProteinModification(ReactionProcess):
    def __init__(
        self,
        wholeCellModelID="Process_ProteinModification",
        name="Protein Modification",
        **properties,
    ):
        super().__init__(wholeCellModelID, name, **properties)

    def initializeConstants(
        self, knowledgeBase=None, simulation=None, options=None, **kwargs
    ):
        # A cached snapshot may supply the initialized network directly.
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
            "monomerIndexs_modified",
            "monomerIndexs_notmodified",
            "monomerCompartments",
        )
        missing = [k for k in required if not hasattr(self, k)]
        if missing:
            raise NotImplementedError(
                "ProteinModification initialized snapshot is missing: "
                + ", ".join(missing)
            )
        if not hasattr(self, "speciesReactantMatrix") or not hasattr(
            self, "speciesReactantByproductMatrix"
        ):
            self.initializeSpeciesNetwork()

    def _initialize_from_knowledge_base(self, knowledgeBase):
        self.substrateIndexs_aminoAcids = self.substrateIndexs(
            ["GLU", "LYS", "SER", "THR", "TYR"]
        )
        self.substrateIndexs_modifiedAminoAcids = self.substrateIndexs(
            ["LIPOYLLYS", "pSER", "pTHR", "pTYR"]
        )
        for name, ids in {
            "atp": ["ATP"],
            "adp": ["ADP"],
            "amp": ["AMP"],
            "hydrogen": ["H"],
            "phosphate": ["PI"],
            "glutamate": ["GLU"],
            "lipoylAmp": ["LIPOYLAMP"],
            "lipoylLys": ["LIPOYLLYS"],
        }.items():
            setattr(self, f"substrateIndexs_{name}", self.substrateIndexs(ids))
        for name, ids in {
            "serineThreonineKinase": ["MG_109_DIMER"],
            "lipoylTransferase": ["MG_270_MONOMER"],
            "glutamateLigase": ["MG_012_MONOMER"],
        }.items():
            setattr(self, f"enzymeIndexs_{name}", self.enzymeIndexs(ids))

        monomer_ids = np.asarray(
            _get(self.monomer, "wholeCellModelIDs"), dtype=object
        ).ravel(order="F")
        folded = np.asarray(
            _get(self.monomer, "foldedIndexs"), dtype=int
        ).ravel(order="F")
        mature = np.asarray(
            _get(self.monomer, "matureIndexs"), dtype=int
        ).ravel(order="F")
        self.unmodifiedMonomerWholeCellModelIDs = monomer_ids[
            folded - 1
        ].reshape(-1, 1)
        self.modifiedMonomerWholeCellModelIDs = monomer_ids[
            mature - 1
        ].reshape(-1, 1)
        types = np.asarray(self.reactionTypes, dtype=object).ravel(order="F")
        self.reactionIndexs_adduction = (
            np.flatnonzero(types == "adduction") + 1
        )
        self.reactionIndexs_ligation = np.flatnonzero(types == "ligation") + 1
        adduction = self.reactionIndexs_adduction - 1
        aa = np.asarray(self.substrateIndexs_aminoAcids, dtype=int).ravel() - 1
        modified_aa = (
            np.asarray(
                self.substrateIndexs_modifiedAminoAcids, dtype=int
            ).ravel()
            - 1
        )
        self.reactionStoichiometryMatrix[np.ix_(aa, adduction)] = 0
        self.reactionStoichiometryMatrix[np.ix_(modified_aa, adduction)] = 0

        gene_columns = (
            np.asarray(_get(self.gene, "mRNAIndexs"), dtype=int).ravel(
                order="F"
            )
            - 1
        )
        modification = np.asarray(self.reactionModificationMatrix)
        self.reactionModificationMatrix = modification[:, gene_columns] > 0
        self.monomerIndexs_modified = (
            np.flatnonzero(np.any(self.reactionModificationMatrix, axis=0)) + 1
        )
        self.monomerIndexs_notmodified = (
            np.flatnonzero(~np.any(self.reactionModificationMatrix, axis=0))
            + 1
        )
        self.monomerCompartments = np.asarray(
            _get(knowledgeBase, "proteinMonomerCompartments"), dtype=int
        ).copy()
        cyt = int(np.asarray(_get(self.compartment, "cytosolIndexs")).item())
        mem = int(np.asarray(_get(self.compartment, "membraneIndexs")).item())
        toc = int(
            np.asarray(
                _get(self.compartment, "terminalOrganelleCytosolIndexs")
            ).item()
        )
        tom = int(
            np.asarray(
                _get(self.compartment, "terminalOrganelleMembraneIndexs")
            ).item()
        )
        self.monomerCompartments[self.monomerCompartments == toc] = cyt
        self.monomerCompartments[self.monomerCompartments == tom] = mem
        self.initializeSpeciesNetwork()

    def initializeSpeciesNetwork(self):
        mod = np.asarray(self.reactionModificationMatrix, dtype=bool)
        sto = np.asarray(self.reactionStoichiometryMatrix)
        cat = np.asarray(self.reactionCatalysisMatrix)
        bounds = np.asarray(self.enzymeBounds)
        idx = np.asarray(self.monomerIndexs_modified, dtype=int).ravel() - 1
        enz = cat / (
            bounds[:, 1:2] * float(np.asarray(self.stepSizeSec).item())
        )
        base = np.concatenate((-sto.T, enz), axis=1)
        # reactionModificationMatrix is reactions × monomers. MATLAB selects
        # modified monomer columns and transposes them into monomers × reactions.
        modified_reactions = mod[:, idx].T.astype(float)
        self.speciesReactantByproductMatrix = np.concatenate(
            (modified_reactions @ base, np.eye(idx.size)), axis=1
        )
        self.speciesReactantMatrix = np.concatenate(
            (modified_reactions @ np.maximum(0, base), np.eye(idx.size)),
            axis=1,
        )
        self.speciesIndexs_enzymes = (
            np.arange(cat.shape[1], dtype=int) + sto.shape[0] + 1
        )

    def _monomer_comps(self):
        idx = np.asarray(self.monomer.foldedIndexs, dtype=int).ravel() - 1
        comps = (
            np.asarray(self.monomer.compartments, dtype=int)
            .ravel()[idx]
            .copy()
        )
        cyt = int(np.asarray(self.compartment.cytosolIndexs).ravel()[0])
        mem = int(np.asarray(self.compartment.membraneIndexs).ravel()[0])
        toc = int(
            np.asarray(
                self.compartment.terminalOrganelleCytosolIndexs
            ).ravel()[0]
        )
        tom = int(
            np.asarray(
                self.compartment.terminalOrganelleMembraneIndexs
            ).ravel()[0]
        )
        comps[comps == toc] = cyt
        comps[comps == tom] = mem
        return comps

    def _copy_monomers(self, to_state):
        counts = np.asarray(_get(self.monomer, "counts"))
        folded = np.asarray(self.monomer.foldedIndexs, dtype=int).ravel()
        mature = np.asarray(self.monomer.matureIndexs, dtype=int).ravel()
        comps = self._monomer_comps()
        nt = (
            np.asarray(self.unmodifiedMonomers).shape[2]
            if np.asarray(self.unmodifiedMonomers).ndim > 2
            else 1
        )
        for ix, comp, local in (
            (folded, comps, "unmodifiedMonomers"),
            (mature, comps, "modifiedMonomers"),
        ):
            vals = np.asarray(getattr(self, local))
            destix = ix - 1
            ci = comp - 1
            if counts.ndim == 2:
                counts = counts[:, :, None]
            for t in range(nt):
                if to_state:
                    counts[destix, ci, t] = vals.reshape(
                        vals.shape[0], -1, order="F"
                    )[:, t]
                else:
                    vals.reshape(vals.shape[0], -1, order="F")[:, t] = counts[
                        destix, ci, t
                    ]
            if not to_state:
                setattr(self, local, vals)
        _set(self.monomer, "counts", counts)

    def copyFromState(self):
        super().copyFromState()
        self._copy_monomers(False)

    def copyToState(self):
        super().copyToState()
        self._copy_monomers(True)

    def allocateMemoryForState(self, numTimePoints=1):
        n = len(self.unmodifiedMonomerWholeCellModelIDs)
        m = len(self.modifiedMonomerWholeCellModelIDs)
        shape = lambda z: (
            (z, 1) if numTimePoints == 1 else (z, 1, numTimePoints)
        )
        self.unmodifiedMonomers = np.zeros(shape(n))
        self.modifiedMonomers = np.zeros(shape(m))

    def calcResourceRequirements_LifeCycle(self, *args):
        states = args[-1] if args else self.states
        prod = np.asarray(_get(states, "monomerProductions"))
        prod0 = np.asarray(_get(states, "monomerProductions0"))
        n = np.asarray(self.reactionModificationMatrix) @ prod
        bm = np.maximum(0, -np.asarray(self.reactionStoichiometryMatrix)) @ n
        by = np.maximum(0, np.asarray(self.reactionStoichiometryMatrix)) @ n
        n0 = np.asarray(self.reactionModificationMatrix) @ prod0
        emin = (
            2
            * np.asarray(self.reactionCatalysisMatrix).T
            @ (n0 / np.asarray(self.enzymeBounds)[:, 1])
        )
        return (
            bm,
            by,
            emin,
            np.full(np.asarray(self.enzymeWholeCellModelIDs).size, np.inf),
        )

    def initializeState(self):
        return None

    def calcResourceRequirements_Current(self):
        possible = np.minimum(
            np.ceil(
                np.asarray(self.reactionCatalysisMatrix)
                @ np.asarray(self.enzymes).reshape(-1)
                * float(np.asarray(self.stepSizeSec).item())
            ),
            np.asarray(self.reactionModificationMatrix)
            @ np.asarray(self.unmodifiedMonomers).reshape(-1),
        )
        return (
            np.maximum(0, -np.asarray(self.reactionStoichiometryMatrix))
            @ possible
        )

    def evolveState(self):
        un = np.asarray(self.unmodifiedMonomers).reshape(-1).copy()
        mod = np.asarray(self.modifiedMonomers).reshape(-1).copy()
        modified = (
            np.asarray(self.monomerIndexs_modified, dtype=int).ravel() - 1
        )
        plain = (
            np.asarray(self.monomerIndexs_notmodified, dtype=int).ravel() - 1
        )
        mod[plain] += un[plain]
        un[plain] = 0
        self.unmodifiedMonomers = un.reshape(
            np.asarray(self.unmodifiedMonomers).shape
        )
        self.modifiedMonomers = mod.reshape(
            np.asarray(self.modifiedMonomers).shape
        )
        if not np.any(un):
            return
        n = modified.size
        species = np.concatenate(
            (
                np.asarray(self.substrates).reshape(-1),
                np.asarray(self.enzymes).reshape(-1),
                un[modified],
            )
        )
        demand = np.asarray(self.speciesReactantMatrix)
        net = np.asarray(self.speciesReactantByproductMatrix)
        limits = np.divide(
            species[None, :],
            demand,
            out=np.full((n, demand.shape[1]), np.inf),
            where=demand != 0,
        )
        inactive = np.min(limits, axis=1) <= 0
        flux = np.zeros(n, dtype=int)
        enzix = np.asarray(self.speciesIndexs_enzymes, dtype=int).ravel() - 1
        other = np.r_[0 : enzix[0], enzix[-1] + 1 : net.shape[1]]
        while True:
            lim = np.divide(
                species[None, :],
                np.maximum(0, net),
                out=np.full((n, net.shape[1]), np.inf),
                where=np.maximum(0, net) != 0,
            )
            e = np.min(lim[:, enzix], axis=1)
            o = (
                np.min(lim[:, other], axis=1)
                if other.size
                else np.full(n, np.inf)
            )
            rl = np.asarray(
                np.minimum(self.randStream.stochasticRound(e), o)
            ).reshape(-1)
            rl[inactive | ~np.isfinite(rl) | (rl < 1)] = 0
            if not np.any(rl):
                break
            selected = (
                int(
                    np.asarray(
                        self.randStream.randsample(n, 1, True, rl)
                    ).ravel()[0]
                )
                - 1
            )
            flux[selected] += 1
            species -= net[selected]
        if flux.any():
            sto = np.asarray(self.reactionStoichiometryMatrix)
            mat = np.asarray(self.reactionModificationMatrix)[:, modified]
            self.substrates = np.asarray(self.substrates) + (
                sto @ mat @ flux
            ).reshape(np.asarray(self.substrates).shape)
            un[modified] -= flux
            mod[modified] += flux
            self.unmodifiedMonomers = un.reshape(
                np.asarray(self.unmodifiedMonomers).shape
            )
            self.modifiedMonomers = mod.reshape(
                np.asarray(self.modifiedMonomers).shape
            )

    def getDryWeight(self):
        base = np.asarray(super().getDryWeight())
        na = float(getattr(self, "nAvogadro", 6.0221417930e23))
        weights = np.asarray(self.monomer.molecularWeights).ravel(order="F")
        fi = np.asarray(self.monomer.foldedIndexs, dtype=int).ravel() - 1
        mi = np.asarray(self.monomer.matureIndexs, dtype=int).ravel() - 1
        u = np.asarray(self.unmodifiedMonomers).reshape(len(fi), -1, order="F")
        m = np.asarray(self.modifiedMonomers).reshape(len(mi), -1, order="F")
        extra = (weights[fi] @ u + weights[mi] @ m) / na
        extra = (
            extra.reshape(1, 1) if extra.size == 1 else extra.reshape(1, 1, -1)
        )
        return base + extra
