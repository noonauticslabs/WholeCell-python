"""Protein N-terminal deformylation and methionine cleavage process.

Port of WholeCell ``ProteinProcessingI.m``. Snapshot indexes remain one based;
conversion to NumPy indexes is confined to state access.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from ..process_base import Process, _arr, _get, _set


class ProteinProcessingI(Process):
    def __init__(
        self,
        wholeCellModelID="Process_ProteinProcessingI",
        name="Protein Processing I",
        **properties,
    ):
        super().__init__(wholeCellModelID, name, **properties)
        self.stimuliWholeCellModelIDs = []
        self.substrateWholeCellModelIDs = ["H2O", "H", "MET", "FOR"]
        self.substrateIndexs_water, self.substrateIndexs_hydrogen = 1, 2
        self.substrateIndexs_methionine, self.substrateIndexs_formate = 3, 4
        self.enzymeWholeCellModelIDs = ["MG_106_DIMER", "MG_172_MONOMER"]
        self.enzymeIndexs_deformylase = 1
        self.enzymeIndexs_methionineAminoPeptidase = 2

    def initializeConstants(
        self, knowledgeBase=None, simulation=None, **kwargs
    ):
        """Load cleavage flags when a KB object is supplied; snapshots can inject all fields."""
        if knowledgeBase is not None:
            if simulation is None:
                raise ValueError(
                    "simulation is required with knowledgeBase initialization"
                )
            super().initializeConstants(
                knowledgeBase, simulation, kwargs.get("options", {})
            )
            kb_flags = _get(
                knowledgeBase, "proteinMonomerNTerminalMethionineCleavages"
            )
            self.nascentMonomerNTerminalMethionineCleavages = np.asarray(
                kb_flags, dtype=bool
            ).reshape(-1)
            if self.monomer is None and simulation is not None:
                self.storeObjectReferences(simulation)
        if self.monomer is not None:
            ids = list(_get(self.monomer, "wholeCellModelIDs"))
            nascent = np.asarray(
                _get(self.monomer, "nascentIndexs"), dtype=int
            ).ravel()
            processed = np.asarray(
                _get(self.monomer, "processedIIndexs"), dtype=int
            ).ravel()
            self.unprocessedMonomerWholeCellModelIDs = [
                ids[i - 1] for i in nascent
            ]
            self.processedMonomerWholeCellModelIDs = [
                ids[i - 1] for i in processed
            ]
        if not hasattr(self, "nascentMonomerNTerminalMethionineCleavages"):
            raise NotImplementedError(
                "proteinMonomerNTerminalMethionineCleavages requires initialized knowledge-base data"
            )
        if not hasattr(self, "unprocessedMonomerWholeCellModelIDs"):
            raise NotImplementedError(
                "monomer nascent/processed indexes require initialized simulation state"
            )

    def copyFromState(self):
        super().copyFromState()
        nascent = np.asarray(self.monomer.nascentIndexs, dtype=int).ravel() - 1
        mature = (
            np.asarray(self.monomer.processedIIndexs, dtype=int).ravel() - 1
        )
        counts = np.asarray(_get(self.monomer, "counts"))
        cyt = np.asarray(self.compartment.cytosolIndexs, dtype=int).ravel() - 1
        # MATLAB indexes the third (time) dimension explicitly. Keep it even
        # when it has length one; NumPy's paired advanced indexing otherwise
        # collapses the singleton compartment/time axes.
        if counts.ndim == 2:
            counts = counts.reshape((*counts.shape, 1))
        self.unprocessedMonomers = counts[nascent[:, None], cyt[None, :], :]
        self.processedMonomers = counts[mature[:, None], cyt[None, :], :]

    def copyToState(self):
        super().copyToState()
        counts = np.asarray(_get(self.monomer, "counts"))
        nascent = np.asarray(self.monomer.nascentIndexs, dtype=int).ravel() - 1
        mature = (
            np.asarray(self.monomer.processedIIndexs, dtype=int).ravel() - 1
        )
        cyt = np.asarray(self.compartment.cytosolIndexs, dtype=int).ravel() - 1
        if counts.ndim == 2:
            counts = counts.reshape((*counts.shape, 1))
        counts[nascent[:, None], cyt[None, :], :] = np.asarray(
            self.unprocessedMonomers
        ).reshape((nascent.size, cyt.size, counts.shape[2]))
        counts[mature[:, None], cyt[None, :], :] = np.asarray(
            self.processedMonomers
        ).reshape((mature.size, cyt.size, counts.shape[2]))
        _set(self.monomer, "counts", counts)

    def allocateMemoryForState(self, numTimePoints=1):
        n, m = len(self.unprocessedMonomerWholeCellModelIDs), len(
            self.processedMonomerWholeCellModelIDs
        )
        shape = (n, 1) if numTimePoints == 1 else (n, 1, numTimePoints)
        shape2 = (m, 1) if numTimePoints == 1 else (m, 1, numTimePoints)
        self.unprocessedMonomers, self.processedMonomers = np.zeros(
            shape
        ), np.zeros(shape2)

    def calcResourceRequirements_LifeCycle(self, *args):
        states = args[-1] if args else self.states
        prod = np.asarray(_get(states, "monomerProductions")).reshape(-1)
        prod0 = np.asarray(_get(states, "monomerProductions0")).reshape(-1)
        flags = np.asarray(
            self.nascentMonomerNTerminalMethionineCleavages, dtype=bool
        ).reshape(-1)
        bm = np.zeros(4)
        by = np.zeros(4)
        enzmin = np.zeros(2)
        enzmax = np.full(2, np.inf)
        total = prod.sum()
        cleave = prod @ flags
        bm[0] = total + cleave
        by[3] = total
        by[1] = total
        by[2] = cleave
        enzmin[0] = 2 * prod0.sum() / self.deformylaseSpecificRate
        enzmin[1] = (
            2 * (prod0 @ flags) / self.methionineAminoPeptidaseSpecificRate
        )
        return bm, by, enzmin, enzmax

    def initializeState(self):
        return None

    def calcResourceRequirements_Current(self):
        self._require_single_timepoint()
        flags = np.asarray(
            self.nascentMonomerNTerminalMethionineCleavages, dtype=bool
        ).reshape(-1)
        u = np.asarray(self.unprocessedMonomers).reshape(-1)
        dl = np.ceil(
            self.enzymes[0] * self.deformylaseSpecificRate * self.stepSizeSec
        )
        cl = np.ceil(
            self.enzymes[1]
            * self.methionineAminoPeptidaseSpecificRate
            * self.stepSizeSec
        )
        available = min(dl, u.sum()) + min(cl, u[flags].sum())
        out = np.zeros_like(np.asarray(self.substrates))
        out[0] = min(np.asarray(self.substrates).reshape(-1)[0], available)
        return out

    def evolveState(self):
        self._require_single_timepoint()
        flags = np.asarray(
            self.nascentMonomerNTerminalMethionineCleavages, dtype=bool
        ).reshape(-1)
        u = np.asarray(self.unprocessedMonomers).reshape(-1).copy()
        p = np.asarray(self.processedMonomers).reshape(-1).copy()
        subs = np.asarray(self.substrates).reshape(-1)
        stream = self.randStream
        dl = self.enzymes[0] * self.deformylaseSpecificRate * self.stepSizeSec
        cl = (
            self.enzymes[1]
            * self.methionineAminoPeptidaseSpecificRate
            * self.stepSizeSec
        )
        if cl > 0 and np.any(u[flags]):
            x = u.astype(float).copy()
            x *= min(1, dl / x.sum()) if x.sum() else 0
            if x[flags].sum():
                x[flags] *= min(1, cl / x[flags].sum())
            x = np.asarray(stream.stochasticRound(x)).reshape(-1).astype(int)
            if x.sum() + x[flags].sum() > subs[0]:
                wc = int(
                    stream.stochasticRound(
                        subs[0]
                        * 2
                        * x[flags].sum()
                        / (x.sum() + x[flags].sum())
                    )
                )
                wd = subs[0] - wc
                if x[flags].any():
                    x[flags] = np.minimum(
                        x[flags],
                        stream.mnrnd(
                            wc // 2, x[flags] / x[flags].sum()
                        ).reshape(-1),
                    )
                if x[~flags].any():
                    x[~flags] = np.minimum(
                        x[~flags],
                        stream.mnrnd(wd, x[~flags] / x[~flags].sum()).reshape(
                            -1
                        ),
                    )
            u -= x
            p += x
            dl = max(0, dl - x.sum())
            subs[[0, 3, 1]] += np.array([-1, 1, 1]) * x.sum()
            subs[[0, 2]] += np.array([-1, 1]) * x[flags].sum()
        x = u.copy()
        x[flags] = 0
        if dl > 0 and x.any():
            x = x.astype(float) * min(1, dl / x.sum())
            x = np.asarray(stream.stochasticRound(x)).reshape(-1).astype(int)
            if x.sum() > subs[0]:
                x = np.minimum(
                    x, stream.mnrnd(int(subs[0]), x / x.sum()).reshape(-1)
                )
            u -= x
            p += x
            subs[[0, 3, 1]] += np.array([-1, 1, 1]) * x.sum()
        self.unprocessedMonomers = u.reshape(
            np.asarray(self.unprocessedMonomers).shape
        )
        self.processedMonomers = p.reshape(
            np.asarray(self.processedMonomers).shape
        )

    def _require_single_timepoint(self):
        """Runtime mutation methods operate on one simulation step at a time."""
        for name in (
            "unprocessedMonomers",
            "processedMonomers",
            "substrates",
            "enzymes",
        ):
            value = np.asarray(getattr(self, name))
            if value.ndim >= 3 and value.shape[2] != 1:
                raise ValueError(
                    f"{name} contains {value.shape[2]} time points; runtime processing requires one"
                )

    def getDryWeight(self):
        na = float(getattr(self, "nAvogadro", 6.0221417930e23))
        weights = np.asarray(_get(self.monomer, "molecularWeights")).reshape(
            -1
        )
        ni = np.asarray(self.monomer.nascentIndexs, dtype=int).ravel() - 1
        pi = np.asarray(self.monomer.processedIIndexs, dtype=int).ravel() - 1
        u = np.asarray(self.unprocessedMonomers).reshape(len(ni), -1)
        p = np.asarray(self.processedMonomers).reshape(len(pi), -1)
        protein_weight = (weights[ni] @ u + weights[pi] @ p) / na
        base = np.asarray(super().getDryWeight()).reshape(-1)
        if base.size == 1 and protein_weight.size > 1:
            base = np.repeat(base, protein_weight.size)
        total = base + protein_weight
        return total.reshape((1, 1) if total.size == 1 else (1, 1, total.size))
