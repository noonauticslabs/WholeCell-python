"""Protein translocation into the membrane and extracellular compartments."""

from __future__ import annotations

import numpy as np

from ..process_base import Process, _get, _set


class ProteinTranslocation(Process):
    optionNames__ = ()
    fixedConstantNames__ = (
        "translocaseSpecificRate",
        "preproteinTranslocase_aaTranslocatedPerATP",
        "SRP_GTPUsedPerMonomer",
        "monomerLengths",
        "monomerCompartments",
        "monomerSRPPathways",
    )
    fittedConstantNames__ = ()
    localStateNames__ = ("monomers",)
    stimuliWholeCellModelIDs = ()
    substrateWholeCellModelIDs = ("ATP", "GTP", "ADP", "GDP", "H", "H2O", "PI")
    enzymeWholeCellModelIDs = (
        "MG_0001_048",
        "MG_297_MONOMER",
        "MG_072_DIMER",
        "MG_055_170_277_464_476_20MER",
    )
    _N_AVOGADRO = 6.0221417930e23

    def __init__(
        self,
        wholeCellModelID="Process_ProteinTranslocation",
        name="Protein Translocation",
        **properties,
    ):
        super().__init__(wholeCellModelID, name, **properties)
        self.substrateIndexs_atp, self.substrateIndexs_gtp = 1, 2
        self.substrateIndexs_adp, self.substrateIndexs_gdp = 3, 4
        (
            self.substrateIndexs_hydrogen,
            self.substrateIndexs_water,
            self.substrateIndexs_phosphate,
        ) = (5, 6, 7)
        self.enzymeIndexs_signalRecognitionParticle = 1
        self.enzymeIndexs_signalRecognitionParticleReceptor = 2
        self.enzymeIndexs_translocaseATPase = 3
        self.enzymeIndexs_translocasePore = 4

    @staticmethod
    def _idx(values):
        return np.asarray(values, dtype=int).ravel() - 1

    @staticmethod
    def _scalar(value):
        return np.asarray(value).item()

    def initializeConstants(
        self, knowledgeBase=None, simulation=None, options=None, **kwargs
    ):
        if knowledgeBase is not None:
            Process.initializeConstants(
                self, knowledgeBase, simulation, options
            )
            rows = self._idx(_get(self.monomer, "processedIIndexs"))
            self.monomerWholeCellModelIDs = (
                np.asarray(
                    _get(self.monomer, "wholeCellModelIDs"), dtype=object
                )
                .ravel(order="F")[rows]
                .reshape(-1, 1)
            )
            self.monomerLengths = (
                np.asarray(_get(self.monomer, "lengths"))
                .ravel(order="F")[rows]
                .reshape(-1, 1)
            )
            comps = np.array(
                _get(knowledgeBase, "proteinMonomerCompartments"),
                dtype=int,
                copy=True,
            ).reshape(-1, 1)
            c = self.compartment
            cyt = int(np.asarray(c.cytosolIndexs).item())
            comps[
                comps
                == int(np.asarray(c.terminalOrganelleCytosolIndexs).item())
            ] = cyt
            comps[
                comps
                == int(np.asarray(c.terminalOrganelleMembraneIndexs).item())
            ] = int(np.asarray(c.membraneIndexs).item())
            self.monomerCompartments = comps
            self.monomerIndexs_translocating = (
                np.flatnonzero(comps.ravel() != cyt) + 1
            ).reshape(-1, 1)
            self.monomerIndexs_nontranslocating = (
                np.flatnonzero(comps.ravel() == cyt) + 1
            ).reshape(-1, 1)
            monomers = _get(knowledgeBase, "proteinMonomers")
            self.monomerSRPPathways = np.asarray(
                [
                    _get(m, "signalSequenceType")
                    in ("lipoprotein", "secretory")
                    for m in monomers
                ],
                dtype=bool,
            ).reshape(-1, 1)
        required = (
            "monomerWholeCellModelIDs",
            "monomerLengths",
            "monomerCompartments",
            "monomerIndexs_translocating",
            "monomerIndexs_nontranslocating",
            "monomerSRPPathways",
            "translocaseSpecificRate",
            "preproteinTranslocase_aaTranslocatedPerATP",
            "SRP_GTPUsedPerMonomer",
        )
        missing = [x for x in required if not hasattr(self, x)]
        if missing:
            raise NotImplementedError(
                "initializeConstants requires initialized knowledge-base/simulation snapshot fields: "
                + ", ".join(missing)
            )

    def allocateMemoryForState(self, numTimePoints=1):
        n = len(self.monomerWholeCellModelIDs)
        nc = (
            int(_get(self.compartment, "count"))
            if self.compartment is not None
            else int(getattr(self, "numCompartments", 1))
        )
        shape = (n, nc) if numTimePoints == 1 else (n, nc, numTimePoints)
        self.monomers = np.zeros(shape)

    def copyFromState(self):
        super().copyFromState()
        counts = np.asarray(_get(self.monomer, "counts"))
        rows = self._idx(_get(self.monomer, "processedIIndexs"))
        self.monomers = counts[rows, ...]

    def copyToState(self):
        super().copyToState()
        counts = np.asarray(_get(self.monomer, "counts"))
        rows = self._idx(_get(self.monomer, "processedIIndexs"))
        target = counts[rows, ...]
        counts[rows, ...] = np.asarray(self.monomers).reshape(target.shape)
        _set(self.monomer, "counts", counts)

    def calcResourceRequirements_LifeCycle(self, *args):
        states = args[-1] if args else self.states
        prod = np.asarray(_get(states, "monomerProductions")).reshape(
            -1, order="F"
        )
        prod0 = np.asarray(_get(states, "monomerProductions0")).reshape(
            -1, order="F"
        )
        trans, srp = (
            self._idx(self.monomerIndexs_translocating),
            np.asarray(self.monomerSRPPathways, dtype=bool).ravel(),
        )
        aa_per_atp = self._scalar(
            self.preproteinTranslocase_aaTranslocatedPerATP
        )
        te = (
            prod[trans]
            @ np.ceil(
                np.asarray(self.monomerLengths).ravel()[trans] / aa_per_atp
            )
            if trans.size
            else 0
        )
        se = self._scalar(self.SRP_GTPUsedPerMonomer) * (srp @ prod)
        bm, by = np.zeros(7), np.zeros(7)
        bm[0] += te
        bm[5] += te
        by[2] += te
        by[6] += te
        by[4] += te
        bm[1] += se
        bm[5] += se
        by[3] += se
        by[6] += se
        by[4] += se
        lo, hi = np.zeros(4), np.full(4, np.inf)
        enz_need = (
            2
            * (
                prod0[trans]
                @ np.ceil(
                    np.asarray(self.monomerLengths).ravel()[trans] / aa_per_atp
                )
                if trans.size
                else 0
            )
            / self._scalar(self.translocaseSpecificRate)
        )
        lo[[2, 3]] = enz_need
        srp_need = 2 * (srp @ prod0)
        lo[[0, 1]] = srp_need
        return bm, by, lo, hi

    def initializeState(self):
        return None

    def calcResourceRequirements_Current(self):
        out = np.zeros_like(np.asarray(self.substrates))
        subs, enz = (
            np.asarray(self.substrates).ravel(),
            np.asarray(self.enzymes).ravel(),
        )
        trans, srp = (
            self._idx(self.monomerIndexs_translocating),
            np.asarray(self.monomerSRPPathways, dtype=bool).ravel(),
        )
        cyt = int(_get(self.compartment, "cytosolIndexs")) - 1
        counts = np.asarray(self.monomers)
        if counts.ndim == 3:
            counts = counts[:, :, 0]
        rate = self._scalar(self.translocaseSpecificRate)
        aa_per_atp = self._scalar(
            self.preproteinTranslocase_aaTranslocatedPerATP
        )
        gtp_cost = self._scalar(self.SRP_GTPUsedPerMonomer)
        step = self._scalar(self.stepSizeSec)
        nt = int(np.ceil(min(enz[[2, 3]]) * rate * step))
        ns = int(np.ceil(min(enz[[0, 1]]) * step))
        costs = np.ceil(
            np.asarray(self.monomerLengths).ravel()[trans] / aa_per_atp
        )
        out.flat[0] = min(nt, counts[trans, cyt] @ costs)
        out.flat[1] = gtp_cost * min(ns, srp @ counts[:, cyt])
        out.flat[5] = out.flat[0] + out.flat[1]
        return out

    def evolveState(self):
        m = np.asarray(self.monomers)
        if m.ndim == 3:
            if m.shape[2] != 1:
                raise ValueError(
                    "evolveState supports the current single-time state only"
                )
            m = m[:, :, 0]
        trans = self._idx(self.monomerIndexs_translocating)
        cyt = int(_get(self.compartment, "cytosolIndexs")) - 1
        cumulative = np.cumsum(m[trans, cyt])
        if not cumulative.size or cumulative[-1] == 0:
            return
        subs, enz = (
            np.asarray(self.substrates).ravel(),
            np.asarray(self.enzymes).ravel(),
        )
        atp, gtp, water = subs[0], subs[1], subs[5]
        rate = self._scalar(self.translocaseSpecificRate)
        aa_per_atp = self._scalar(
            self.preproteinTranslocase_aaTranslocatedPerATP
        )
        gtp_cost = self._scalar(self.SRP_GTPUsedPerMonomer)
        step = self._scalar(self.stepSizeSec)
        capacity = min(enz[[2, 3]]) * rate * step / aa_per_atp
        srpcap = min(enz[[0, 1]]) * step
        order = np.asarray(
            self.randStream.randperm(int(cumulative[-1]))
        ).ravel()
        for draw in order:
            k = int(np.searchsorted(cumulative, draw, side="left"))
            row = trans[k]
            tc = int(
                np.ceil(
                    np.asarray(self.monomerLengths).ravel()[row] / aa_per_atp
                )
            )
            sc = int(bool(np.asarray(self.monomerSRPPathways).ravel()[row]))
            gc, wc = sc * gtp_cost, tc + sc * gtp_cost
            if (
                capacity < tc
                or srpcap < sc
                or atp < tc
                or gtp < gc
                or water < wc
            ):
                break
            dest = int(np.asarray(self.monomerCompartments).ravel()[row]) - 1
            m[row, dest] += 1
            m[row, cyt] -= 1
            atp -= tc
            gtp -= gc
            water -= wc
            capacity -= tc
            srpcap -= sc
            subs[0] -= tc
            subs[1] -= gc
            subs[5] -= wc
            subs[2] += tc
            subs[3] += gc
            subs[6] += wc
            subs[4] += wc

    def getDryWeight(self):
        base = super().getDryWeight()
        rows = self._idx(_get(self.monomer, "processedIIndexs"))
        weights = np.asarray(_get(self.monomer, "molecularWeights")).ravel()[
            rows
        ]
        monomers = np.asarray(self.monomers)
        if monomers.ndim == 2:
            amount = monomers.sum(axis=1)
        else:
            amount = monomers.sum(axis=1)
        extra = np.tensordot(weights, amount, axes=(0, 0)) / float(
            getattr(self, "nAvogadro", self._N_AVOGADRO)
        )
        return base + extra.reshape(
            (1, 1) if np.ndim(extra) == 0 else (1, 1, np.size(extra))
        )
