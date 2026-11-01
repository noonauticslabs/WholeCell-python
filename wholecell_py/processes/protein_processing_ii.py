"""ProteinProcessingII lipoprotein and secreted protein maturation.

Port of WholeCell/src/+edu/+stanford/+covert/+cell/+sim/+process/ProteinProcessingII.m.
Exported indices remain MATLAB one based; local vectors are indexed only after
conversion at the point of use.
"""

from __future__ import annotations

import numpy as np

from ..process_base import Process, _get


class ProteinProcessingII(Process):
    optionNames__ = ()
    fixedConstantNames__ = (
        "lipoproteinSignalPeptidaseSpecificRate",
        "lipoproteinDiacylglycerylTransferaseSpecificRate",
    )
    fittedConstantNames__ = ()
    localStateNames__ = (
        "unprocessedMonomers",
        "processedMonomers",
        "signalSequenceMonomers",
    )
    stimuliWholeCellModelIDs = ()
    substrateWholeCellModelIDs = (
        "H2O",
        "diacylglycerolCys",
        "PG160",
        "SNGLYP",
        "H",
    )
    substrateIndexs_water = 1
    substrateIndexs_diacylglycerylCys = 2
    substrateIndexs_PG160 = 3
    substrateIndexs_SNGLYP = 4
    substrateIndexs_hydrogen = 5
    enzymeWholeCellModelIDs = ("MG_210_MONOMER", "MG_086_MONOMER")
    enzymeIndexs_signalPeptidase = 1
    enzymeIndexs_diacylglycerylTransferase = 2
    _N_AVOGADRO = 6.0221417930e23

    def __init__(
        self,
        wholeCellModelID="Process_ProteinProcessingII",
        name="Protein Processing II",
        **properties,
    ):
        super().__init__(wholeCellModelID, name, **properties)

    def initializeConstants(
        self, knowledgeBase=None, simulation=None, **kwargs
    ):
        """Initialize source-derived protein classes when a KB is available."""
        if knowledgeBase is not None or simulation is not None:
            if knowledgeBase is None or simulation is None:
                raise ValueError(
                    "knowledgeBase and simulation must be supplied together"
                )
            super().initializeConstants(
                knowledgeBase, simulation, kwargs.get("options", {})
            )
            monomer = self.monomer
            ids = np.asarray(
                _get(monomer, "wholeCellModelIDs"), dtype=object
            ).ravel(order="F")
            processed_i = np.asarray(
                _get(monomer, "processedIIndexs"), dtype=int
            ).ravel(order="F")
            processed_ii = np.asarray(
                _get(monomer, "processedIIIndexs"), dtype=int
            ).ravel(order="F")
            signal = np.asarray(
                _get(monomer, "signalSequenceIndexs"), dtype=int
            ).ravel(order="F")
            self.unprocessedMonomerWholeCellModelIDs = ids[
                processed_i - 1
            ].reshape(-1, 1)
            self.processedMonomerWholeCellModelIDs = ids[
                processed_ii - 1
            ].reshape(-1, 1)
            self.signalSequenceMonomerWholeCellModelIDs = ids[
                signal - 1
            ].reshape(-1, 1)
            comps = np.asarray(
                _get(knowledgeBase, "proteinMonomerCompartments"), dtype=int
            ).copy()
            cyt = int(
                np.asarray(_get(self.compartment, "cytosolIndexs")).item()
            )
            mem = int(
                np.asarray(_get(self.compartment, "membraneIndexs")).item()
            )
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
            comps[comps == toc] = cyt
            comps[comps == tom] = mem
            self.monomerCompartments = comps
            monomers = np.asarray(
                _get(knowledgeBase, "proteinMonomers"), dtype=object
            ).ravel(order="F")
            signal_types = np.asarray(
                [_get(record, "signalSequenceType") for record in monomers],
                dtype=object,
            )
            self.lipoproteinMonomerIndexs = (
                np.flatnonzero(signal_types == "lipoprotein") + 1
            ).reshape(-1, 1)
            self.secretedMonomerIndexs = (
                np.flatnonzero(signal_types == "secretory") + 1
            ).reshape(-1, 1)
            all_indices = np.arange(
                1, int(_get(knowledgeBase, "numProteinMonomers")) + 1
            )
            excluded = np.r_[
                self.lipoproteinMonomerIndexs.ravel(),
                self.secretedMonomerIndexs.ravel(),
            ]
            self.unprocessedMonomerIndexs = np.setdiff1d(
                all_indices, excluded
            ).reshape(-1, 1)
            return
        needed = (
            "monomerCompartments",
            "lipoproteinMonomerIndexs",
            "secretedMonomerIndexs",
            "unprocessedMonomerIndexs",
            "unprocessedMonomerWholeCellModelIDs",
            "processedMonomerWholeCellModelIDs",
            "signalSequenceMonomerWholeCellModelIDs",
            "lipoproteinSignalPeptidaseSpecificRate",
            "lipoproteinDiacylglycerylTransferaseSpecificRate",
        )
        missing = [name for name in needed if not hasattr(self, name)]
        if missing:
            raise NotImplementedError(
                "initializeConstants requires initialized knowledge-base/simulation snapshot fields: "
                + ", ".join(missing)
            )

    def allocateMemoryForState(self, numTimePoints=1):
        ncomp = int(
            getattr(
                self.compartment, "count", getattr(self, "numCompartments", 1)
            )
        )
        shape = (len(self.unprocessedMonomerWholeCellModelIDs), ncomp)
        pshape = (len(self.processedMonomerWholeCellModelIDs), ncomp)
        sshape = (len(self.signalSequenceMonomerWholeCellModelIDs), 1)
        if numTimePoints == 1:
            self.unprocessedMonomers = np.zeros(shape)
            self.processedMonomers = np.zeros(pshape)
            self.signalSequenceMonomers = np.zeros(sshape)
        else:
            self.unprocessedMonomers = np.zeros((*shape, numTimePoints))
            self.processedMonomers = np.zeros((*pshape, numTimePoints))
            self.signalSequenceMonomers = np.zeros((*sshape, numTimePoints))

    def copyFromState(self):
        super().copyFromState()
        counts = np.asarray(
            self.monomer["counts"]
            if isinstance(self.monomer, dict)
            else self.monomer.counts
        )
        comps = np.asarray(self.monomerCompartments, dtype=int).ravel() - 1
        pi = np.asarray(self.monomer.processedIIndexs, dtype=int).ravel() - 1
        pii = np.asarray(self.monomer.processedIIIndexs, dtype=int).ravel() - 1
        si = (
            np.asarray(self.monomer.signalSequenceIndexs, dtype=int).ravel()
            - 1
        )
        cyt = int(self.compartment.cytosolIndexs) - 1
        if counts.ndim == 2:
            self.unprocessedMonomers = counts[pi, comps].reshape((-1, 1))
            self.processedMonomers = counts[pii, comps].reshape((-1, 1))
            self.signalSequenceMonomers = counts[si, cyt].reshape((-1, 1))
        else:
            self.unprocessedMonomers = counts[pi, comps, :]
            self.processedMonomers = counts[pii, comps, :]
            self.signalSequenceMonomers = counts[si, cyt, :].reshape(
                (len(si), 1, -1)
            )

    def copyToState(self):
        super().copyToState()
        state = self.monomer
        counts = np.asarray(
            state["counts"] if isinstance(state, dict) else state.counts
        )
        comps = np.asarray(self.monomerCompartments, dtype=int).ravel() - 1
        pi = (
            np.asarray(
                (
                    state["processedIIndexs"]
                    if isinstance(state, dict)
                    else state.processedIIndexs
                ),
                dtype=int,
            ).ravel()
            - 1
        )
        pii = (
            np.asarray(
                (
                    state["processedIIIndexs"]
                    if isinstance(state, dict)
                    else state.processedIIIndexs
                ),
                dtype=int,
            ).ravel()
            - 1
        )
        si = (
            np.asarray(
                (
                    state["signalSequenceIndexs"]
                    if isinstance(state, dict)
                    else state.signalSequenceIndexs
                ),
                dtype=int,
            ).ravel()
            - 1
        )
        cyt = int(self.compartment.cytosolIndexs) - 1
        if counts.ndim == 2:
            counts[pi, comps] = np.asarray(self.unprocessedMonomers).reshape(
                -1
            )
            counts[pii, comps] = np.asarray(self.processedMonomers).reshape(-1)
            counts[si, cyt] = np.asarray(self.signalSequenceMonomers).reshape(
                -1
            )
        else:
            counts[pi, comps, :] = self.unprocessedMonomers
            counts[pii, comps, :] = self.processedMonomers
            counts[si, cyt, :] = np.asarray(
                self.signalSequenceMonomers
            ).reshape((len(si), -1))
        if isinstance(state, dict):
            state["counts"] = counts
        else:
            state.counts = counts

    def _indices(self, name):
        return np.asarray(getattr(self, name), dtype=int).ravel() - 1

    def calcResourceRequirements_LifeCycle(self, *args):
        states = args[-1] if args else self.states
        prod = np.asarray(states.monomerProductions).ravel(order="F")
        prod0 = np.asarray(states.monomerProductions0).ravel(order="F")
        lip, sec = self._indices("lipoproteinMonomerIndexs"), self._indices(
            "secretedMonomerIndexs"
        )
        bm, by = np.zeros(5), np.zeros(5)
        lo, hi = np.zeros(2), np.full(2, np.inf)
        cleavages, transfers = prod[np.r_[lip, sec]].sum(), prod[lip].sum()
        bm[0], bm[2] = cleavages, transfers
        by[3], by[4] = transfers, transfers
        lo[0] = (
            2
            * prod0[np.r_[lip, sec]].sum()
            / self.lipoproteinSignalPeptidaseSpecificRate
        )
        lo[1] = (
            2
            * prod0[lip].sum()
            / self.lipoproteinDiacylglycerylTransferaseSpecificRate
        )
        return bm, by, lo, hi

    def initializeState(self):
        return None

    def calcResourceRequirements_Current(self):
        subs = np.asarray(self.substrates)
        enz = np.asarray(self.enzymes).ravel(order="F")
        unp = np.asarray(self.unprocessedMonomers).ravel(order="F")
        pept = int(
            np.ceil(
                enz[0]
                * self.lipoproteinSignalPeptidaseSpecificRate
                * self.stepSizeSec
            )
        )
        trans = int(
            np.ceil(
                enz[1]
                * self.lipoproteinDiacylglycerylTransferaseSpecificRate
                * self.stepSizeSec
            )
        )
        out = np.zeros_like(subs)
        lip, sec = self._indices("lipoproteinMonomerIndexs"), self._indices(
            "secretedMonomerIndexs"
        )
        out[0] = min(pept, unp[np.r_[lip, sec]].sum())
        out[2] = min(trans, unp[lip].sum())
        return out

    def evolveState(self):
        # MATLAB collapses trailing singleton compartment/time dimensions when
        # indexing a single runtime state. Snapshot arrays can retain them, so
        # use one-dimensional views for the species-indexed updates below.
        # These reshapes are views for the contiguous snapshot state arrays;
        # updates therefore retain the original writeback shapes.
        unp = np.asarray(self.unprocessedMonomers).reshape(-1)
        done = np.asarray(self.processedMonomers).reshape(-1)
        sig = np.asarray(self.signalSequenceMonomers).reshape(-1)
        subs = np.asarray(self.substrates)
        enz = np.asarray(self.enzymes).ravel(order="F")
        lip, sec, other = (
            self._indices(k)
            for k in (
                "lipoproteinMonomerIndexs",
                "secretedMonomerIndexs",
                "unprocessedMonomerIndexs",
            )
        )
        peptidx = np.r_[lip, sec]
        done[other] += unp[other]
        unp[other] = 0
        if not np.any(unp[peptidx]):
            return
        p_limit = float(
            np.asarray(
                enz[0]
                * self.lipoproteinSignalPeptidaseSpecificRate
                * self.stepSizeSec
            ).item()
        )

        def apply(trans, transfer):
            nonlocal p_limit
            trans = np.asarray(
                self.randStream.stochasticRound(trans.reshape((-1, 1)))
            ).reshape(trans.shape)
            if trans[peptidx].sum() > subs[0]:
                trans[peptidx] = np.minimum(
                    trans[peptidx],
                    self.randStream.mnrnd(
                        subs[0], trans[peptidx] / trans[peptidx].sum()
                    ).ravel(),
                )
            if transfer and trans[lip].sum() > subs[2]:
                trans[lip] = np.minimum(
                    trans[lip],
                    self.randStream.mnrnd(
                        subs[2], trans[lip] / trans[lip].sum()
                    ).ravel(),
                )
            sig[peptidx] += trans[peptidx]
            done[...] += trans
            unp[...] -= trans
            p_limit -= trans[peptidx].sum()
            subs[0] -= trans[peptidx].sum()
            if transfer:
                total = trans[lip].sum()
                subs[2] -= total
                subs[3] += total
                subs[4] += total

        if np.any(unp[lip]):
            t_limit = float(
                np.asarray(
                    enz[1]
                    * self.lipoproteinDiacylglycerylTransferaseSpecificRate
                    * self.stepSizeSec
                ).item()
            )
            tr = unp.copy()
            tr[peptidx] *= min(1, p_limit / tr[peptidx].sum())
            tr[lip] *= min(1, t_limit / tr[lip].sum())
            apply(tr, True)
        tr = unp.copy()
        if tr[peptidx].sum():
            tr[peptidx] *= min(1, p_limit / tr[peptidx].sum())
        tr[lip] = 0
        if np.any(tr):
            apply(tr, False)

    def getDryWeight(self):
        na = float(getattr(self, "nAvogadro", self._N_AVOGADRO))
        base = super().getDryWeight()
        mw = np.asarray(self.monomer.molecularWeights)
        pi, pii, si = (
            np.asarray(getattr(self.monomer, key), dtype=int).ravel() - 1
            for key in (
                "processedIIndexs",
                "processedIIIndexs",
                "signalSequenceIndexs",
            )
        )
        return (
            base
            + (
                mw[pi] @ np.asarray(self.unprocessedMonomers)
                + mw[pii] @ np.asarray(self.processedMonomers)
                + mw[si] @ np.asarray(self.signalSequenceMonomers)
            )
            / na
        )
