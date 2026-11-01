"""RNA transcript cleavage process translated from WholeCell RNAProcessing.m."""

from __future__ import annotations
import numpy as np
from ..computation import invert_composition_matrix
from ..process_base import Process, _get, _set


class RNAProcessing(Process):
    def __init__(
        self,
        wholeCellModelID="Process_RNAProcessing",
        name="RNA Processing",
        **properties,
    ):
        super().__init__(wholeCellModelID, name, **properties)
        self.stimuliWholeCellModelIDs = []
        self.substrateWholeCellModelIDs = [
            "ATP",
            "GTP",
            "ADP",
            "GDP",
            "PI",
            "H2O",
            "H",
        ]
        self.substrateIndexs_NTPs = np.array([1, 2])
        self.substrateIndexs_ATP = 1
        self.substrateIndexs_GTP = 2
        self.substrateIndexs_NDPs = np.array([3, 4])
        self.substrateIndexs_ADP = 3
        self.substrateIndexs_GDP = 4
        self.substrateIndexs_PI = 5
        self.substrateIndexs_H2O = 6
        self.substrateIndexs_H = 7
        self.enzymeWholeCellModelIDs = [
            "MG_367_DIMER",
            "MG_139_DIMER",
            "MG_0003_465",
            "MG_110_MONOMER",
            "MG_425_DIMER",
        ]
        self.enzymeIndexs_RNAseIII = 1
        self.enzymeIndexs_RNAseJ = 2
        self.enzymeIndexs_RNAseP = 3
        self.enzymeIndexs_RsgA = 4
        self.enzymeIndexs_DeaD = 5
        self.enzymeSpecificRate_DeaD = 1.48
        self.enzymeSpecificRate_RsgA = 0.2917
        self.enzymeSpecificRate_RNAseP = 6.0
        self.enzymeSpecificRate_RNAseIII = 7.7
        self.enzymeSpecificRate_RNAseJ = 0.37
        self.enzymeEnergyCost_DeaD = 2.37
        self.enzymeEnergyCost_RsgA = 1.0

    def initializeConstants(
        self, knowledgeBase=None, simulation=None, **kwargs
    ):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if simulation is not None and self.rna is None:
            self.storeObjectReferences(simulation)
        if self.rna is not None:
            r = self.rna
            ids = list(_get(r, "wholeCellModelIDs"))
            nas = np.asarray(_get(r, "nascentIndexs"), int).ravel()
            pro = np.asarray(_get(r, "processedIndexs"), int).ravel()
            self.unprocessedRNAWholeCellModelIDs = [ids[i - 1] for i in nas]
            self.processedRNAWholeCellModelIDs = [ids[i - 1] for i in pro]
            self.intergenicRNAWholeCellModelIDs = [
                ids[i - 1]
                for i in np.asarray(_get(r, "intergenicIndexs"), int).ravel()
            ]
            self.processedRNAIndexs_mRNA = np.asarray(
                r.matureMRNAIndexs, int
            ).ravel()
            self.processedRNAIndexs_rRNA = np.asarray(
                r.matureRRNAIndexs, int
            ).ravel()
            self.processedRNAIndexs_sRNA = np.asarray(
                r.matureSRNAIndexs, int
            ).ravel()
            self.processedRNAIndexs_tRNA = np.asarray(
                r.matureTRNAIndexs, int
            ).ravel()
            self.processedRNAIndexs_scRNA = (
                np.flatnonzero(
                    np.asarray(
                        [str(x) for x in self.processedRNAWholeCellModelIDs]
                    )
                    == "MG_0001"
                )
                + 1
            )
            self.processedRNAIndexs_tmRNA = (
                np.flatnonzero(
                    np.asarray(
                        [str(x) for x in self.processedRNAWholeCellModelIDs]
                    )
                    == "MG_0004"
                )
                + 1
            )
            comp = np.asarray(_get(r, "nascentRNAMatureRNAComposition"))
            inter = np.asarray(_get(r, "intergenicRNAMatrix"))
            for typ, ix in [
                ("mRNA", self.processedRNAIndexs_mRNA),
                ("rRNA", self.processedRNAIndexs_rRNA),
                ("sRNA", self.processedRNAIndexs_sRNA),
                ("tRNA", self.processedRNAIndexs_tRNA),
                ("scRNA", self.processedRNAIndexs_scRNA),
                ("tmRNA", self.processedRNAIndexs_tmRNA),
            ]:
                setattr(
                    self,
                    "unprocessedRNAIndexs_" + typ,
                    np.flatnonzero(np.sum(comp[ix - 1, :], axis=0)) + 1,
                )
            mat = np.zeros((7, len(nas)))
            nr = np.sum(comp, axis=0) + np.sum(inter, axis=0) - 1
            mat[5, :] = -nr
            mat[6, :] = nr
            rr = self.unprocessedRNAIndexs_rRNA - 1
            mat[np.ix_([0, 2, 4, 5, 6], rr)] += np.array([-1, 1, 1, -1, 1])[
                :, None
            ]
            # Original uses RsgA GTP cost=1 and DeaD ATP cost=round(2.37)=2.
            mat[np.ix_([1, 3, 4, 5, 6], rr)] += np.array([-1, 1, 1, -1, 1])[
                :, None
            ] * round(float(np.asarray(self.enzymeEnergyCost_DeaD).item()))
            self.reactantByproductMatrix = mat
            cat = np.zeros((5, len(nas)))
            rr = self.unprocessedRNAIndexs_rRNA - 1
            sc = self.unprocessedRNAIndexs_scRNA - 1
            tm = self.unprocessedRNAIndexs_tmRNA - 1
            tr = self.unprocessedRNAIndexs_tRNA - 1
            cat[4, rr] = 1
            cat[1, rr] = 2
            cat[0, rr] = 2
            cat[3, rr] = 1
            cat[0, sc] = 2
            cat[2, tm] = 1
            cat[0, tr] = 1
            cat[2, tr] = 1
            rates = np.array(
                [
                    float(np.asarray(x).item())
                    for x in (
                        self.enzymeSpecificRate_RNAseIII,
                        self.enzymeSpecificRate_RNAseJ,
                        self.enzymeSpecificRate_RNAseP,
                        self.enzymeSpecificRate_RsgA,
                        self.enzymeSpecificRate_DeaD,
                    )
                ]
            )
            self.catalysisMatrix = cat / rates[:, None]
        required = (
            "unprocessedRNAWholeCellModelIDs",
            "processedRNAWholeCellModelIDs",
            "intergenicRNAWholeCellModelIDs",
            "reactantByproductMatrix",
            "catalysisMatrix",
        )
        missing = [k for k in required if not hasattr(self, k)]
        if missing:
            raise NotImplementedError(
                "RNAProcessing requires initialized RNA knowledge-base matrices/indexes: "
                + ", ".join(missing)
            )

    def copyFromState(self):
        super().copyFromState()
        r = self.rna
        c = np.asarray(_get(r, "counts"))
        c = c[:, :, None] if c.ndim == 2 else c
        cyt = np.asarray(self.compartment.cytosolIndexs, int).ravel() - 1
        self.unprocessedRNAs = c[np.asarray(r.nascentIndexs, int).ravel() - 1][
            :, cyt, :
        ]
        self.processedRNAs = c[np.asarray(r.processedIndexs, int).ravel() - 1][
            :, cyt, :
        ]
        self.intergenicRNAs = c[
            np.asarray(r.intergenicIndexs, int).ravel() - 1
        ][:, cyt, :]
        if c.shape[2] == 1:
            self.unprocessedRNAs = self.unprocessedRNAs[:, :, 0]
            self.processedRNAs = self.processedRNAs[:, :, 0]
            self.intergenicRNAs = self.intergenicRNAs[:, :, 0]

    def copyToState(self):
        super().copyToState()
        r = self.rna
        c = np.asarray(_get(r, "counts"))
        c = c[:, :, None] if c.ndim == 2 else c
        cyt = np.asarray(self.compartment.cytosolIndexs, int).ravel() - 1
        for inds, vals in (
            (r.nascentIndexs, self.unprocessedRNAs),
            (r.processedIndexs, self.processedRNAs),
            (r.intergenicIndexs, self.intergenicRNAs),
        ):
            ix = np.asarray(inds, int).ravel() - 1
            c[np.ix_(ix, cyt, np.arange(c.shape[2]))] = np.asarray(
                vals
            ).reshape((len(ix), len(cyt), c.shape[2]))
        _set(r, "counts", c)

    def allocateMemoryForState(self, numTimePoints=1):
        for attr, n in [
            ("unprocessedRNAs", len(self.unprocessedRNAWholeCellModelIDs)),
            ("processedRNAs", len(self.processedRNAWholeCellModelIDs)),
            ("intergenicRNAs", len(self.intergenicRNAWholeCellModelIDs)),
        ]:
            setattr(
                self,
                attr,
                np.zeros(
                    (n, 1) if numTimePoints == 1 else (n, 1, numTimePoints)
                ),
            )

    def initializeState(self):
        return None

    def calcResourceRequirements_Current(self):
        self._require_single_timepoint()
        limits = np.ceil(
            np.min(
                np.divide(
                    np.asarray(self.enzymes).reshape(-1, 1)
                    * float(np.asarray(self.stepSizeSec).item()),
                    self.catalysisMatrix,
                    out=np.full_like(self.catalysisMatrix, np.inf),
                    where=self.catalysisMatrix != 0,
                ),
                axis=0,
            )
        )
        rxn = np.minimum(limits, np.asarray(self.unprocessedRNAs).reshape(-1))
        return np.maximum(0, -self.reactantByproductMatrix) @ rxn

    def calcResourceRequirements_LifeCycle(self, *args):
        states = args[-1]
        prod = np.asarray(_get(states, "rnaProductions"))
        prod0 = np.asarray(_get(states, "rnaProductions0"))
        inv = invert_composition_matrix(
            self.rna.nascentRNAMatureRNAComposition
        )
        bm = np.maximum(0, -self.reactantByproductMatrix) @ inv @ prod
        by = np.maximum(0, self.reactantByproductMatrix) @ inv @ prod
        return (
            bm,
            by,
            2 * self.catalysisMatrix @ inv @ prod0,
            np.full((len(self.enzymeWholeCellModelIDs), 1), np.inf),
        )

    def evolveState(self):
        self._require_single_timepoint()
        for typ in [
            ["mRNA", "rRNA", "sRNA", "tRNA"][i - 1]
            for i in np.asarray(self.randStream.randperm(4)).ravel()
        ]:
            self.evolveState_Helper(typ)

    def evolveState_Helper(self, rnaType):
        ix = (
            np.asarray(
                getattr(self, "unprocessedRNAIndexs_" + rnaType), int
            ).ravel()
            - 1
        )
        total = int(np.sum(self.unprocessedRNAs[ix]))
        if not total:
            return
        # Keep the state arrays' MATLAB-like column-vector shapes when writing
        # the updated values back.  Flattening is useful for the calculations,
        # but replacing these attributes with 1-D arrays breaks copyToState's
        # dimensional contract.
        substrate_shape = np.asarray(self.substrates).shape
        processed_shape = np.asarray(self.processedRNAs).shape
        intergenic_shape = np.asarray(self.intergenicRNAs).shape
        unprocessed_shape = np.asarray(self.unprocessedRNAs).shape
        denom = np.maximum(
            0, -np.min(self.reactantByproductMatrix[:, ix], axis=1)
        )
        subs = np.asarray(self.substrates).reshape(-1)
        with_sub = np.divide(
            subs, denom, out=np.full(denom.shape, np.inf), where=denom > 0
        )
        enzyme = (
            np.asarray(self.enzymes).reshape(-1)
            * self.stepSizeSec
            / self.catalysisMatrix[:, ix[0]]
        )
        n = int(
            np.floor(
                min(
                    total,
                    float(np.min(self.randStream.stochasticRound(enzyme))),
                    float(np.min(with_sub)),
                )
            )
        )
        if n <= 0:
            return
        draws = np.asarray(
            self.randStream.randCounts(
                np.asarray(self.unprocessedRNAs).reshape(-1)[ix], n
            )
        ).reshape(-1)
        comp = np.asarray(self.rna.nascentRNAMatureRNAComposition)
        inter = np.asarray(self.rna.intergenicRNAMatrix)
        self.processedRNAs = (
            np.asarray(self.processedRNAs).reshape(-1) + comp[:, ix] @ draws
        ).reshape(processed_shape)
        self.intergenicRNAs = (
            np.asarray(self.intergenicRNAs).reshape(-1) + inter[:, ix] @ draws
        ).reshape(intergenic_shape)
        u = np.asarray(self.unprocessedRNAs).reshape(-1).copy()
        u[ix] -= draws
        self.unprocessedRNAs = u.reshape(unprocessed_shape)
        self.substrates = (
            subs + self.reactantByproductMatrix[:, ix] @ draws
        ).reshape(substrate_shape)

    def _require_single_timepoint(self):
        for n in (
            "unprocessedRNAs",
            "processedRNAs",
            "intergenicRNAs",
            "substrates",
            "enzymes",
        ):
            x = np.asarray(getattr(self, n))
            if x.ndim > 2 and x.shape[2] != 1:
                raise ValueError(
                    f"{n} runtime processing requires one time point"
                )


RNAProcessingProcess = RNAProcessing
