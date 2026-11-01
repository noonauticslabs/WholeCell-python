"""Enzyme-limited stochastic RNA decay (MATLAB RNADecay port)."""

from __future__ import annotations

import numpy as np
from ..computation import invert_composition_matrix
from ..process_base import Process, _get, _set


class RNADecay(Process):
    optionNames__ = []
    fixedConstantNames__ = [
        "peptidylTRNAHydrolaseSpecificRate",
        "ribonucleaseRFragmentLength",
        "decayReactions",
    ]
    fittedConstantNames__ = []
    localStateNames__ = ["RNAs"]
    enzymeWholeCellModelIDs = ["MG_104_MONOMER", "MG_083_MONOMER"]
    enzymeIndexs_ribonucleaseR = 1
    enzymeIndexs_peptidylTRNAHydrolase = 2

    def __init__(
        self,
        wholeCellModelID="Process_RNADecay",
        name="RNA Decay",
        **properties,
    ):
        super().__init__(wholeCellModelID, name, **properties)
        self.peptidylTRNAHydrolaseSpecificRate = getattr(
            self, "peptidylTRNAHydrolaseSpecificRate", 0.7
        )
        self.ribonucleaseRFragmentLength = getattr(
            self, "ribonucleaseRFragmentLength", 5
        )
        self.transcripts = getattr(self, "transcripts", None)

    def storeObjectReferences(self, simulation):
        super().storeObjectReferences(simulation)
        fn = getattr(simulation, "state", None)
        self.transcripts = (
            fn("Transcript")
            if callable(fn)
            else _get(simulation, "states")["Transcript"]
        )
        self.states = self.states + [self.transcripts]

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
            names = (
                "nascentRNAs",
                "processedRNAs",
                "intergenicRNAs",
                "matureRNAs",
                "aminoacylatedRNAs",
            )
            decay_sets = [
                np.asarray(_get(_get(knowledgeBase, name), "decayReactions"))
                for name in names
            ]
            metabolite_ids = np.asarray(
                _get(self.metabolite, "wholeCellModelIDs"), dtype=object
            ).ravel(order="F")
            active = np.zeros(len(metabolite_ids), dtype=bool)
            for reactions in decay_sets:
                active |= np.any(reactions != 0, axis=0)
            fixed = [
                "H",
                "H2O",
                "NH3",
                "FOR",
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
                "FMET",
            ]
            included = set(metabolite_ids[active].tolist()) | set(fixed)
            self.substrateWholeCellModelIDs = np.asarray(
                sorted(included, key=str), dtype=object
            ).reshape(-1, 1)
            Process.initializeConstants(
                self, knowledgeBase, simulation, options or {}
            )

            self.substrateIndexs_hydrogen = self.substrateIndexs(["H"])
            self.substrateIndexs_water = self.substrateIndexs(["H2O"])
            for name, ids in {
                "methionine": ["MET"],
                "fmethionine": ["FMET"],
                "glutamate": ["GLU"],
                "glutamine": ["GLN"],
                "ammonia": ["NH3"],
                "formate": ["FOR"],
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
                    "FMET",
                ],
                "nmps": ["AMP", "CMP", "GMP", "UMP"],
            }.items():
                setattr(
                    self, f"substrateIndexs_{name}", self.substrateIndexs(ids)
                )

            composition = np.asarray(
                _get(self.rna, "matureRNAGeneComposition")
            )
            trna_genes = (
                np.asarray(_get(self.gene, "tRNAIndexs"), dtype=int).ravel(
                    order="F"
                )
                - 1
            )
            self.matureTRNAIndexs = (
                np.flatnonzero(np.any(composition[trna_genes, :] != 0, axis=0))
                + 1
            ).reshape(-1, 1)
            self.matureTMRNAIndexs = np.asarray(
                _get(self.rna, "getIndexs")(["MG_0004"]), dtype=int
            ).reshape(-1, 1)

            n_rna = len(
                np.asarray(
                    _get(self.rna, "wholeCellModelIDs"), dtype=object
                ).ravel(order="F")
            )
            n_sub = len(
                np.asarray(self.substrateWholeCellModelIDs).ravel(order="F")
            )
            self.decayReactions = np.zeros((n_rna, n_sub))
            substrate_rows = (
                np.asarray(
                    self.substrateMetaboliteGlobalIndexs, dtype=int
                ).ravel(order="F")
                - 1
            )
            state_indexes = (
                _get(self.rna, "nascentIndexs"),
                _get(self.rna, "processedIndexs"),
                _get(self.rna, "intergenicIndexs"),
                _get(self.rna, "matureIndexs"),
                _get(self.rna, "aminoacylatedIndexs"),
            )
            for reactions, state_rows in zip(decay_sets, state_indexes):
                rows = np.asarray(state_rows, dtype=int).ravel(order="F") - 1
                self.decayReactions[rows, :] = reactions[:, substrate_rows]
            return
        required = (
            "decayReactions",
            "substrateIndexs_water",
            "substrateIndexs_nmps",
            "matureTRNAIndexs",
            "matureTMRNAIndexs",
        )
        missing = [k for k in required if not hasattr(self, k)]
        if missing:
            raise NotImplementedError(
                "RNADecay snapshot lacks initialized network fields: "
                + ", ".join(missing)
            )

    def copyFromState(self):
        super().copyFromState()
        c = np.asarray(_get(self.rna, "counts"))
        c = c[:, :, None] if c.ndim == 2 else c
        cyt = int(np.asarray(self.compartment.cytosolIndexs).ravel()[0]) - 1
        self.RNAs = c[:, cyt, :]
        if c.shape[2] == 1:
            self.RNAs = self.RNAs[:, None]

    def copyToState(self):
        super().copyToState()
        c = np.asarray(_get(self.rna, "counts"))
        c = c[:, :, None] if c.ndim == 2 else c
        cyt = int(np.asarray(self.compartment.cytosolIndexs).ravel()[0]) - 1
        c[:, cyt, :] = np.asarray(self.RNAs).reshape((c.shape[0], -1))
        _set(self.rna, "counts", c)

    def allocateMemoryForState(self, numTimePoints=1):
        c = np.asarray(_get(self.rna, "counts"))
        n = c.shape[0]
        self.RNAs = np.zeros(
            (n, 1) if numTimePoints == 1 else (n, 1, numTimePoints)
        )

    def initializeState(self):
        pass

    def calcResourceRequirements_Current(self):
        if (
            np.asarray(self.enzymes).reshape(-1)[
                int(np.asarray(self.enzymeIndexs_ribonucleaseR).reshape(-1)[0])
                - 1
            ]
            == 0
        ):
            return np.zeros_like(np.asarray(self.substrates))
        out = np.maximum(
            0,
            -np.asarray(self.decayReactions).T
            @ (
                np.minimum(1, np.asarray(self.rna.decayRates).reshape(-1))
                * np.asarray(self.RNAs).reshape(-1)
            ),
        )
        aborted = (
            _get(self.transcripts, "abortedTranscripts")
            if self.transcripts is not None
            else []
        )
        if np.asarray(aborted).size:
            out[
                int(np.asarray(self.substrateIndexs_water).ravel()[0]) - 1
            ] += np.asarray(aborted)[:, 1].sum() - len(aborted)
        return out

    def calcResourceRequirements_LifeCycle(self, *args):
        states = args[-1]
        rna = self.rna
        inv = np.asarray(
            _get(rna, "intergenicRNAMatrix")
        ) @ invert_composition_matrix(
            _get(rna, "nascentRNAMatureRNAComposition")
        )
        aa = np.asarray(self.decayReactions)[
            np.asarray(rna.aminoacylatedIndexs, dtype=int).ravel() - 1
        ]
        inter = np.asarray(self.decayReactions)[
            np.asarray(rna.intergenicIndexs, dtype=int).ravel() - 1
        ]
        interdec = inv @ np.asarray(_get(states, "rnaProductions")).reshape(-1)
        dec = np.asarray(_get(states, "rnaDecays")).reshape(-1)
        bm = np.maximum(0, -aa).T @ dec + np.maximum(0, -inter).T @ interdec
        by = np.maximum(0, aa).T @ dec + np.maximum(0, inter).T @ interdec
        dec0 = np.asarray(_get(states, "rnaDecays0")).reshape(-1)
        inter0 = inv @ np.asarray(_get(states, "rnaProductions0")).reshape(-1)
        tri = (
            np.r_[
                np.asarray(
                    rna.matureTRNAIndexs
                    if hasattr(rna, "matureTRNAIndexs")
                    else self.matureTRNAIndexs
                ).ravel(),
                np.asarray(self.matureTMRNAIndexs).ravel(),
            ].astype(int)
            - 1
        )
        enzyme = np.zeros(2)
        enzyme[0] = 2 * (dec0.sum() + inter0.sum())
        enzyme[1] = (
            2
            * dec0[tri].sum()
            / float(np.asarray(self.peptidylTRNAHydrolaseSpecificRate))
        )
        return bm, by, enzyme, np.full(2, np.inf)

    def evolveState(self):
        substrate_shape = np.asarray(self.substrates).shape
        enz = np.asarray(self.enzymes).reshape(-1)
        rr = enz[0]
        hyd = float(
            np.asarray(
                self.randStream.stochasticRound(
                    enz[1]
                    * float(np.asarray(self.peptidylTRNAHydrolaseSpecificRate))
                    * float(np.asarray(self.stepSizeSec))
                )
            ).reshape(-1)[0]
        )
        if rr == 0:
            return
        # Aborted transcript sequence degradation uses the same ssRNA chemistry as MATLAB.
        seqs = _get(self.transcripts, "abortedSequences")
        aborted = np.asarray(_get(self.transcripts, "abortedTranscripts"))
        keep = np.ones(len(seqs), dtype=bool)
        for i, seq in enumerate(seqs):
            if not isinstance(seq, str):
                seq = "".join(np.asarray(seq).ravel(order="F"))
            cost = np.zeros(np.asarray(self.substrates).size)
            cost[
                np.asarray(self.substrateIndexs_nmps, dtype=int).ravel() - 1
            ] = [seq.count(base) for base in "ACGU"]
            bonds = max(0, len(seq) - 1)
            cost[
                int(np.asarray(self.substrateIndexs_water).item()) - 1
            ] -= bonds
            cost[
                int(np.asarray(self.substrateIndexs_hydrogen).item()) - 1
            ] += bonds
            if np.any(np.asarray(self.substrates).reshape(-1) < -cost):
                break
            keep[i] = False
            self.substrates = (
                np.asarray(self.substrates).reshape(-1) + cost
            ).reshape(substrate_shape)
        if aborted.size:
            _set(self.transcripts, "abortedTranscripts", aborted[keep, :])
        rates = np.minimum(
            1e6,
            np.asarray(self.rna.decayRates).reshape(-1)
            * float(np.asarray(self.stepSizeSec)),
        )
        decay = np.minimum(
            self.randStream.random(
                "poisson", np.asarray(self.RNAs).reshape(-1) * rates
            ).reshape(-1),
            np.asarray(self.RNAs).reshape(-1),
        ).astype(int)
        if not np.any(decay):
            return
        aa = np.asarray(self.rna.aminoacylatedIndexs, dtype=int).ravel() - 1
        tmp = decay[aa].copy()
        accepted = np.zeros_like(tmp)
        while tmp.any() and hyd > 0:
            ix = (
                int(
                    np.asarray(
                        self.randStream.randsample(len(tmp), 1, True, tmp)
                    ).reshape(-1)[0]
                )
                - 1
            )
            tmp[ix] -= 1
            accepted[ix] += 1
            hyd -= 1
        decay[aa] = accepted
        wateridx = int(np.asarray(self.substrateIndexs_water).ravel()[0]) - 1
        water = float(np.asarray(self.substrates).reshape(-1)[wateridx])
        req = np.maximum(0, -np.asarray(self.decayReactions)[:, wateridx])
        tmp = decay.copy()
        accepted = np.zeros_like(tmp)
        while tmp.any():
            ix = (
                int(
                    np.asarray(
                        self.randStream.randsample(len(tmp), 1, True, tmp)
                    ).reshape(-1)[0]
                )
                - 1
            )
            if water < req[ix]:
                break
            water -= req[ix]
            tmp[ix] -= 1
            accepted[ix] += 1
        decay = accepted
        self.RNAs = np.asarray(self.RNAs).reshape(-1) - decay
        self.substrates = (
            np.asarray(self.substrates).reshape(-1)
            + np.asarray(self.decayReactions).T @ decay
        ).reshape(substrate_shape)
        c = np.asarray(_get(self.rna, "counts"))
        c = c[:, :, None] if c.ndim == 2 else c
        cyt = int(np.asarray(self.compartment.cytosolIndexs).ravel()[0]) - 1
        c[:, cyt, 0] = self.RNAs
        _set(self.rna, "counts", c)
        delta = np.zeros((c.shape[0], c.shape[1]))
        delta[:, cyt] = -decay
        if np.any(self.rna.updateExternalState(delta, True)):
            raise RuntimeError("All RNAs should have been degraded")
        self.RNAs = np.asarray(self.rna.counts)[:, cyt, :]

    def getDryWeight(self):
        counts = np.asarray(self.RNAs)
        weights = np.asarray(self.rna.molecularWeights).ravel(order="F")
        extra = (
            weights
            @ counts.reshape(weights.size, -1, order="F")
            / 6.0221417930e23
        )
        return super().getDryWeight() + extra.reshape(
            (1, 1) if extra.size == 1 else (1, 1, -1)
        )


RNADecay = RNADecay
