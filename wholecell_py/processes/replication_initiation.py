"""DnaA binding, polymerization, hydrolysis, and release at oriC.

Indices exposed by this process retain the MATLAB source's one-based
convention.  Conversion is performed only at NumPy indexing boundaries.
"""

from __future__ import annotations

import numpy as np

from ..process_base import Process, _get
from ..chromosome_process_aspect import ChromosomeProcessAspect


class ReplicationInitiation(Process, ChromosomeProcessAspect):
    substrateWholeCellModelIDs = ("ATP", "ADP", "PI", "H2O", "H")
    substrateIndexs_atp, substrateIndexs_adp, substrateIndexs_phosphate = (
        1,
        2,
        3,
    )
    substrateIndexs_water, substrateIndexs_hydrogen = 4, 5
    enzymeWholeCellModelIDs = (
        "MG_469_1MER_ADP",
        "MG_469_1MER_ATP",
        "MG_469_2MER_1ATP_ADP",
        "MG_469_2MER_ATP",
        "MG_469_3MER_2ATP_ADP",
        "MG_469_3MER_ATP",
        "MG_469_4MER_3ATP_ADP",
        "MG_469_4MER_ATP",
        "MG_469_5MER_4ATP_ADP",
        "MG_469_5MER_ATP",
        "MG_469_6MER_5ATP_ADP",
        "MG_469_6MER_ATP",
        "MG_469_7MER_6ATP_ADP",
        "MG_469_7MER_ATP",
        "MG_469_MONOMER",
    )
    enzymeIndexs_DnaA = 15
    enzymeIndexs_DnaA_1mer_ADP, enzymeIndexs_DnaA_1mer_ATP = 1, 2
    enzymeIndexs_DnaA_Nmer_ATP = np.arange(2, 15, 2)
    enzymeIndexs_DnaA_Nmer_ADP = np.arange(1, 14, 2)
    enzymeIndexs_DnaA_polymer_ATP = np.arange(4, 15, 2)
    enzymeIndexs_DnaA_polymer_ADP = np.arange(3, 14, 2)
    dnaABoxStatus_NotExist, dnaABoxStatus_NotBound = -1, 0
    dnaABoxStatus_DnaAATPBound, dnaABoxStatus_DnaAADPBound = 1, 2

    def __init__(
        self,
        wholeCellModelID="Process_ReplicationInitiation",
        name="Replication Initiation",
        **properties,
    ):
        super().__init__(wholeCellModelID, name, **properties)
        self.chromosome = getattr(self, "chromosome", None)

    def storeObjectReferences(self, simulation):
        super().storeObjectReferences(simulation)
        ChromosomeProcessAspect.storeObjectReferences(self, simulation)
        statefn = getattr(simulation, "state", None)
        if callable(statefn):
            self.mass = statefn("Mass")
        else:
            states = _get(simulation, "states")
            self.mass = {str(k).lower(): v for k, v in states.items()}["mass"]
        self.states = [*self.states, self.mass]

    def initializeConstants(
        self, knowledgeBase=None, simulation=None, **kwargs
    ):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if knowledgeBase is not None or simulation is not None:
            ChromosomeProcessAspect.initializeConstants(
                self, knowledgeBase, simulation
            )
        if knowledgeBase is not None:
            boxes = [
                x for x in knowledgeBase.genomeFeatures if x.type == "DnaA box"
            ]
            names = [x.name for x in boxes]
            self.dnaABoxStartPositions = np.ceil(
                np.array(
                    [x.startCoordinate + x.sequenceLength / 2 for x in boxes]
                )
                - self.enzymeDNAFootprints[1] / 2
            ).astype(int)
            subtype = [x.subtype for x in boxes]
            self.dnaABoxIndexs_7mer = (
                np.flatnonzero(np.array(subtype) == "7mer") + 1
            )
            self.dnaABoxIndexs_8mer = (
                np.flatnonzero(np.array(subtype) == "8mer") + 1
            )
            self.dnaABoxIndexs_9mer = (
                np.flatnonzero(np.array(subtype) == "9mer") + 1
            )
            self.dnaABoxIndexs_R12345 = np.array(
                [
                    names.index("Functional box R%d" % i) + 1
                    for i in range(1, 6)
                ]
            )
            self.dnaABoxIndexs_R1234 = self.dnaABoxIndexs_R12345[:4]
            self.dnaABoxIndexs_R5 = self.dnaABoxIndexs_R12345[4:]
            self.calcDnaAxpReleaseReactions()

    @staticmethod
    def _idx(indices):
        return np.asarray(indices, dtype=int).reshape(-1) - 1

    def calcDnaAxpReleaseReactions(self):
        self.dnaARelease_remainingDnaAIndexs = np.zeros(
            len(self.enzymeComplexGlobalIndexs), dtype=int
        )
        self.dnaARelease_remainingDnaAIndexs[
            self._idx(self.enzymeIndexs_DnaA_Nmer_ATP[1:])
        ] = self.enzymeIndexs_DnaA_Nmer_ATP[:-1]
        self.dnaARelease_remainingDnaAIndexs[
            self._idx(self.enzymeIndexs_DnaA_Nmer_ADP[1:])
        ] = self.enzymeIndexs_DnaA_Nmer_ATP[:-1]
        n = len(self.enzymeWholeCellModelIDs)
        self.dnaARelease_remainingDnaAMatrix = np.zeros((n, n))
        self.dnaARelease_releasedDnaAMatrix = np.zeros((n, n))
        for remaining, bound in zip(
            self.enzymeIndexs_DnaA_Nmer_ATP[:-1],
            self.enzymeIndexs_DnaA_Nmer_ATP[1:],
        ):
            self.dnaARelease_remainingDnaAMatrix[remaining - 1, bound - 1] = 1
        for remaining, bound in zip(
            self.enzymeIndexs_DnaA_Nmer_ATP[:-1],
            self.enzymeIndexs_DnaA_Nmer_ADP[1:],
        ):
            self.dnaARelease_remainingDnaAMatrix[remaining - 1, bound - 1] = 1
        self.dnaARelease_releasedDnaAIndexs = np.zeros(
            len(self.enzymeComplexGlobalIndexs), dtype=int
        )
        self.dnaARelease_releasedDnaAIndexs[
            self._idx(self.enzymeIndexs_DnaA_Nmer_ATP)
        ] = self.enzymeIndexs_DnaA_1mer_ATP
        self.dnaARelease_releasedDnaAIndexs[
            self._idx(self.enzymeIndexs_DnaA_Nmer_ADP)
        ] = self.enzymeIndexs_DnaA_1mer_ADP
        self.dnaARelease_releasedDnaAIndexs[0:2] = [
            self.enzymeIndexs_DnaA_1mer_ADP,
            self.enzymeIndexs_DnaA_1mer_ATP,
        ]
        self.dnaARelease_releasedDnaAMatrix[0, 0] = 1
        self.dnaARelease_releasedDnaAMatrix[1, 1] = 1
        for i in range(0, 14, 2):
            # Released ATP-monomer products come from ATP Nmers (local
            # indices 4, 6, ..., 14); ADP-monomer products come from ADP
            # Nmers (3, 5, ..., 13), matching MATLAB's indexed assignments.
            self.dnaARelease_releasedDnaAMatrix[1, i + 1] = 1
            self.dnaARelease_releasedDnaAMatrix[0, i] = 1

    def sampleDnaABoxes(self, nBoxes):
        nBoxes = int(nBoxes)
        if not (
            len(self.dnaABoxIndexs_R12345)
            <= nBoxes
            <= len(self.dnaABoxStartPositions)
        ):
            raise ValueError(
                "nBoxes must retain all five functional boxes and not exceed available boxes"
            )
        reserved = set(np.asarray(self.dnaABoxIndexs_R12345, int).tolist())
        others = np.array(
            [
                i
                for i in range(1, len(self.dnaABoxStartPositions) + 1)
                if i not in reserved
            ]
        )
        selected = np.sort(
            np.r_[
                self.randStream.randomlySelectNRows(
                    others, nBoxes - 5
                ).reshape(-1),
                self.dnaABoxIndexs_R12345,
            ]
        ).astype(int)
        # The source evaluates dnaABoxStartPositions(idxs) without assignment;
        # the coordinate vector is deliberately left intact while category
        # indexes are remapped into the sampled box list below.
        for key in (
            "dnaABoxIndexs_7mer",
            "dnaABoxIndexs_8mer",
            "dnaABoxIndexs_9mer",
            "dnaABoxIndexs_R12345",
            "dnaABoxIndexs_R1234",
            "dnaABoxIndexs_R5",
        ):
            old = np.asarray(getattr(self, key), int).reshape(-1)
            setattr(self, key, np.flatnonzero(np.isin(selected, old)) + 1)

    def bindDnaAATP(self, a, d, maxBinding, positionsStrands, bindingRates):
        if maxBinding == 0:
            return a, d, 0
        tfs, _, _, nbound, _ = self.bindProteinToChromosome(
            positionsStrands,
            2,
            maxBinding,
            bindingRates,
            True,
            False,
            1,
            False,
            [],
            True,
        )
        a[
            np.asarray(tfs, bool).ravel()[self._idx(self.dnaABoxIndexs_R1234)],
            0,
        ] = 1
        return a, d, nbound

    def bindDnaAADP(self, a, d, maxBinding, positionsStrands, bindingRates):
        if maxBinding == 0:
            return a, d, 0
        tfs, _, _, nbound, _ = self.bindProteinToChromosome(
            positionsStrands,
            1,
            maxBinding,
            bindingRates,
            True,
            False,
            1,
            False,
            [],
            True,
        )
        d[
            np.asarray(tfs, bool).ravel()[self._idx(self.dnaABoxIndexs_R1234)],
            0,
        ] = 1
        return a, d, nbound

    def calcResourceRequirements_Current(self):
        out = np.zeros_like(np.asarray(self.substrates), dtype=float)
        out[0] = self.enzymes[14] + self.enzymes[0] + self.boundEnzymes[0]
        out[3] = np.dot(
            np.arange(2, 8),
            self.enzymes[self._idx(self.enzymeIndexs_DnaA_polymer_ATP)],
        )
        return out

    def calcResourceRequirements_LifeCycle(self, constants, states):
        bm = np.zeros(5)
        by = np.zeros(5)
        ni = np.arange(1, 8)
        n_atp = np.dot(
            ni,
            np.asarray(_get(states, "complexProductions"))[
                self._idx(
                    self.enzymeGlobalIndexs[
                        self._idx(self.enzymeIndexs_DnaA_Nmer_ATP)
                    ]
                )
            ].reshape(-1),
        )
        n_adp = np.dot(
            ni,
            np.asarray(_get(states, "complexProductions"))[
                self._idx(
                    self.enzymeGlobalIndexs[
                        self._idx(self.enzymeIndexs_DnaA_Nmer_ADP)
                    ]
                )
            ].reshape(-1),
        )
        hydrolysis = 4 * len(self.enzymeIndexs_DnaA_Nmer_ATP)
        bm[0], bm[3] = hydrolysis + n_atp, hydrolysis
        by[1:] = [hydrolysis - n_adp, hydrolysis, hydrolysis, hydrolysis]
        return bm, by, np.asarray(self.enzymes).copy(), np.full(15, np.inf)

    def initializeState(self):
        atp, adp = self._idx(self.enzymeIndexs_DnaA_Nmer_ATP), self._idx(
            self.enzymeIndexs_DnaA_Nmer_ADP
        )
        self.enzymes[atp] += self.boundEnzymes[atp]
        self.enzymes[adp] += self.boundEnzymes[adp]
        self.boundEnzymes[atp] = 0
        self.boundEnzymes[adp] = 0
        self.substrates[0] += np.dot(np.arange(1, 8), self.enzymes[atp])
        self.substrates[1] += np.dot(np.arange(1, 8), self.enzymes[adp])
        self.enzymes[14] += np.dot(
            np.arange(1, 8), self.enzymes[atp]
        ) + np.dot(np.arange(1, 8), self.enzymes[adp])
        self.enzymes[atp] = 0
        self.enzymes[adp] = 0
        self.initializeStateBasedOnFinalConditions()

    def initializeStateBasedOnFinalConditions(self):
        total = self.enzymes[14]
        formed = (total >= 50) or bool(
            self.randStream.stochasticRound(0.5 * (total >= 47))
        )
        available = total - 29 * formed
        f = (
            self.kb1ATP
            * len(self.dnaABoxIndexs_9mer)
            / (
                self.kb1ATP * len(self.dnaABoxIndexs_9mer)
                + self.kb2ATP * len(self.dnaABoxIndexs_8mer)
            )
        )
        nine = self.randStream.stochasticRound(f * available)
        eight = available - nine
        self.enzymes[:] = 0
        self.enzymes[1] = available + formed
        self.enzymes[13] = 4 * formed
        self.boundEnzymes[:] = 0
        tfs7, _, _, _, _ = self.bindProteinToChromosome(
            np.c_[
                self.dnaABoxStartPositions[
                    self._idx(self.dnaABoxIndexs_R1234)
                ],
                np.ones(4),
            ],
            14,
            4 * formed,
            [],
            True,
            False,
            1,
            False,
            [],
            True,
        )
        tfs1, _, _, _, _ = self.bindProteinToChromosome(
            np.c_[
                self.dnaABoxStartPositions[self._idx(self.dnaABoxIndexs_R5)],
                np.ones(len(self.dnaABoxIndexs_R5)),
            ],
            2,
            formed,
            [],
            True,
            False,
            1,
            False,
            [],
            True,
        )
        if (
            np.count_nonzero(tfs7) != 4 * formed
            or np.count_nonzero(tfs1) != formed
        ):
            raise RuntimeError("DnaA-ATP should be bound to oriC sites")
        tfs9, _, _, _, _ = self.bindProteinToChromosome(
            np.c_[
                self.dnaABoxStartPositions[self._idx(self.dnaABoxIndexs_9mer)],
                np.ones(len(self.dnaABoxIndexs_9mer)),
            ],
            2,
            nine,
            [],
            True,
            False,
            1,
            False,
            [],
            True,
        )
        tfs8, _, _, _, _ = self.bindProteinToChromosome(
            np.c_[
                self.dnaABoxStartPositions[self._idx(self.dnaABoxIndexs_8mer)],
                np.ones(len(self.dnaABoxIndexs_8mer)),
            ],
            2,
            eight,
            [],
            True,
            False,
            1,
            False,
            [],
            True,
        )
        if np.count_nonzero(tfs9) != nine or np.count_nonzero(tfs8) != eight:
            raise RuntimeError(
                "DnaA-ATP should be bound to 8- and 9-mer sites"
            )

    def evolveState(self):
        a, d = self.calculateDnaAR1234Polymerization()
        self.activateFreeDnaA()
        self.inactivateFreeDnaAATP()
        if self.chromosome.supercoiled.nnz():
            a, d = self.bindAndPolymerizeDnaAATP(a, d)
            a, d = self.bindAndPolymerizeDnaAADP(a, d)
        self.releaseDnaAAxP(a, d)
        self.reactivateFreeDnaAADP()

    def activateFreeDnaA(self):
        n = min(
            float(np.asarray(self.substrates[0]).item()),
            float(np.asarray(self.enzymes[14]).item()),
        )
        self.substrates[0] -= n
        self.enzymes[14] -= n
        self.enzymes[1] += n

    def inactivateFreeDnaAATP(self):
        polymers = self.enzymes[
            self._idx(self.enzymeIndexs_DnaA_polymer_ATP)
        ].copy()
        while True:
            monomers = np.dot(np.arange(2, 8), polymers)
            if monomers == 0:
                return
            if self.substrates[3] >= monomers:
                break
            j = self.randStream.randsample(6, 1, False, polymers)
            polymers[np.asarray(j, dtype=int).reshape(-1)[0] - 1] -= 1
        self.substrates[3] -= monomers
        self.substrates[2] += monomers
        self.substrates[4] += monomers
        self.enzymes[0] += monomers
        self.enzymes[self._idx(self.enzymeIndexs_DnaA_polymer_ATP)] -= polymers

    def calculateDnaAR1234ComplexSize(self, polATPs, polADPs):
        return np.min(np.maximum(polATPs, np.asarray(polADPs) - 1), axis=0)

    def calculateDnaAR1234Polymerization(self):
        a = np.zeros((4, 2), int)
        d = np.zeros((4, 2), int)
        c = self.chromosome
        sites, vals = c.complexBoundSites.find()
        for col, strand in enumerate((1, 3)):
            for row, box in enumerate(self._idx(self.dnaABoxIndexs_R1234)):
                at = self.dnaABoxStartPositions[box]
                found = np.flatnonzero(
                    (sites[:, 0] == at) & (sites[:, 1] == strand)
                )
                if found.size:
                    v = vals[found[0]]
                    q = np.flatnonzero(self.enzymeComplexGlobalIndexs == v)
                    if q.size:
                        loc = int(q[0]) + 1
                        hit = np.flatnonzero(
                            np.asarray(self.enzymeIndexs_DnaA_Nmer_ATP) == loc
                        )
                        if hit.size:
                            a[row, col] = hit[0] + 1
                        hit = np.flatnonzero(
                            np.asarray(self.enzymeIndexs_DnaA_Nmer_ADP) == loc
                        )
                        if hit.size:
                            d[row, col] = hit[0] + 1
        return a, d

    def calculateDnaABoxStatus(self):
        boxes = np.asarray(self.dnaABoxStartPositions).reshape(-1)
        sites = np.c_[
            np.tile(boxes, 2),
            np.repeat([1, 3], len(boxes)),
        ]
        status = np.zeros((len(boxes), 2), int)
        pr = self.chromosome.polymerizedRegions
        starts, lengths = pr.find()
        starts = np.asarray(starts, dtype=int).reshape(-1, 2)
        lengths = np.asarray(lengths).reshape(-1)
        starts1_mask = starts[:, 1] == 3
        starts2_mask = starts[:, 1] == 4
        starts1 = starts[starts1_mask, 0]
        lengths1 = lengths[starts1_mask]
        starts2 = starts[starts2_mask, 0]
        if not starts1.size:
            if not starts2.size:
                status[:, 1] = -1
            else:
                status[boxes < starts2[-1], 1] = -1
        elif not starts2.size:
            status[boxes > starts1[0], 1] = -1
        else:
            status[
                (boxes > starts1[0] + lengths1[0]) & (boxes < starts2[-1]),
                1,
            ] = -1
        for idx, pos in enumerate(self.dnaABoxStartPositions):
            for col, strand in enumerate((1, 3)):
                q = np.flatnonzero(
                    (sites[:, 0] == pos) & (sites[:, 1] == strand)
                )
                if q.size:
                    v = self.chromosome.complexBoundSites[
                        sites[q[0], 0], sites[q[0], 1]
                    ]
                    if (
                        v
                        in self.enzymeComplexGlobalIndexs[
                            self._idx(self.enzymeIndexs_DnaA_Nmer_ATP)
                        ]
                    ):
                        status[idx, col] = 1
                    if (
                        v
                        in self.enzymeComplexGlobalIndexs[
                            self._idx(self.enzymeIndexs_DnaA_Nmer_ADP)
                        ]
                    ):
                        status[idx, col] = 2
        return status

    def calculateIsDnaAORIComplexAssembled(self):
        a, d = self.calculateDnaAR1234Polymerization()
        r5 = np.asarray(self.calcuateIsDnaAR5Occupied(), dtype=bool).reshape(
            -1
        )
        # The MATLAB method returns one result per chromosome copy. Its
        # single-output call to calculateDnaAR1234Polymerization uses ATP
        # polymerization only; ADP-bound layers do not satisfy assembly.
        return np.all(np.asarray(a) >= 7, axis=0) & r5

    def calculateDnaAR1234ATPPolymerizationCooperativity(self, a, d, p):
        p = np.asarray(p).reshape(1, -1)
        out = self.siteCooperativity * np.vstack(
            (
                a[3, :] > p[0],
                np.all(a[[0, 3], :] > p, axis=0),
                np.all(a[[0, 3], :] > p, axis=0),
                np.any(a > p, axis=0),
            )
        )
        out[3, :] += float(np.asarray(self.stateCooperativity).item()) * p[0]
        return np.maximum(1, ((a == p) & (d == 0)) * out)

    def _binding_rates(self, a, d, atp=True):
        k1 = self.kb1ATP if atp else self.kb1ADP
        k2 = self.kb2ATP if atp else self.kb2ADP
        scale = (
            1e9
            / 3600
            / 6.0221417930e23
            / self.geometry.volume
            * self.stepSizeSec
        )
        r9, r8 = k1 * scale, k2 * scale
        n = len(self.dnaABoxStartPositions)
        avg = (
            r9 * len(self.dnaABoxIndexs_9mer)
            + r8 * len(self.dnaABoxIndexs_8mer)
        ) / n
        rates = np.zeros(n)
        rates[self._idx(self.dnaABoxIndexs_9mer)] = r9
        rates[self._idx(self.dnaABoxIndexs_8mer)] = r8
        positions = np.c_[self.dnaABoxStartPositions, np.ones(n, dtype=int)]
        # MATLAB collapses all polymerized-region entries to count replicated bases.
        pr = self.chromosome.polymerizedRegions
        replicated = (
            float(pr.collapse(None))
            if hasattr(pr, "collapse")
            else float(np.asarray(pr).sum())
        )
        seqLen = self.chromosome.sequenceLen
        second_keep = None
        if replicated > 2 * seqLen:
            rates2 = rates.copy()
            positions2 = np.c_[
                self.dnaABoxStartPositions, np.full(n, 3, dtype=int)
            ]
            if replicated < 4 * seqLen:
                coords, lengths = pr.find()
                mask = coords[:, 1] == 4
                starts = coords[mask, 0]
                lens = np.asarray(lengths).reshape(-1)[mask]
                keep = (
                    np.asarray(self.dnaABoxStartPositions).ravel()
                    < starts[0] + lens[0]
                ) | (
                    np.asarray(self.dnaABoxStartPositions).ravel() > starts[-1]
                )
                # Keep the full second-copy rate vector until cooperativity
                # has been mapped to R1-R4 in source (unfiltered) box order.
                second_keep = keep
            rates = np.r_[rates, rates2]
            positions = np.vstack((positions, positions2))
        return positions, rates, avg, second_keep

    @staticmethod
    def _filter_second_chromosome(positions, rates, second_keep):
        if second_keep is None:
            return positions, rates
        n = len(rates) - len(second_keep)
        return (
            np.vstack((positions[:n], positions[n:][second_keep])),
            np.r_[rates[:n], rates[n:][second_keep]],
        )

    def calculateDnaAATPBindingRates(self, a, d):
        pos, r, avg, second_keep = self._binding_rates(a, d, True)
        p = self.calculateDnaAR1234ComplexSize(a, d)
        n = len(self.dnaABoxStartPositions)
        r[self._idx(self.dnaABoxIndexs_R5)] = (p[0] == 7) * np.finfo(float).max
        if len(r) > n:
            r[n + self._idx(self.dnaABoxIndexs_R5)] = (p[1] == 7) * np.finfo(
                float
            ).max
        coop = self.calculateDnaAR1234ATPPolymerizationCooperativity(a, d, p)
        r[self._idx(self.dnaABoxIndexs_R1234)] *= coop[:, 0]
        if len(r) > n:
            r[n + self._idx(self.dnaABoxIndexs_R1234)] *= coop[:, 1]
        pos, r = self._filter_second_chromosome(pos, r, second_keep)
        return pos, r, avg

    def calculateDnaAADPBindingRates(self, a, d):
        pos, r, avg, second_keep = self._binding_rates(a, d, False)
        p = self.calculateDnaAR1234ComplexSize(a, d)
        n = len(self.dnaABoxStartPositions)
        r[self._idx(self.dnaABoxIndexs_R5)] = (p[0] == 7) * np.finfo(float).max
        if len(r) > n:
            r[n + self._idx(self.dnaABoxIndexs_R5)] = (p[1] == 7) * np.finfo(
                float
            ).max
        pos, r = self._filter_second_chromosome(pos, r, second_keep)
        return pos, r, avg

    def calculateDnaAR1234ATPPolymerizationRates(self, a, d):
        p = self.calculateDnaAR1234ComplexSize(a, d)
        rates = np.zeros((4, 2))
        if np.any(p):
            pr = np.clip(p, 1, 6)
            scale = (
                1e9
                / 3600
                / 6.0221417930e23
                / self.geometry.volume
                * self.stepSizeSec
            )
            rates[:3, :] = self.kb2ATP * scale
            rates[3, :] = self.kb1ATP * scale
            rates *= a == pr[None, :]
            rates *= self.calculateDnaAR1234ATPPolymerizationCooperativity(
                a, d, p
            )
        return rates

    def calculateDnaAR1234ADPPolymerizationRates(self, a, d):
        p = self.calculateDnaAR1234ComplexSize(a, d)
        rates = np.zeros((4, 2))
        if np.any(p):
            pr = np.clip(p, 1, 6)
            rates[:3, :] = (
                self.kb2ADP
                * 1e9
                / 3600
                / 6.0221417930e23
                / self.geometry.volume
                * self.stepSizeSec
            )
            rates[3, :] = (
                self.kb1ADP
                * 1e9
                / 3600
                / 6.0221417930e23
                / self.geometry.volume
                * self.stepSizeSec
            )
            rates *= a == pr[None, :]
        return rates

    def reactivateFreeDnaAADP(self):
        n = float(np.asarray(self.enzymes[0]).item())
        if n == 0:
            return
        membrane = (
            self.mass.metaboliteWt[
                0,
                int(np.asarray(self.mass.compartment.membraneIndexs).item())
                - 1,
            ]
            / self.geometry.volume
        )
        n = float(
            np.asarray(
                min(
                    self.substrates[0],
                    n,
                    self.randStream.stochasticRound(
                        n
                        * (self.k_Regen / 3600 * membrane)
                        / (self.K_Regen_P4 + membrane)
                        * self.stepSizeSec
                    ),
                )
            ).item()
        )
        self.enzymes[0] -= n
        self.enzymes[1] += n
        self.substrates[0] -= n
        self.substrates[1] += n

    def bindAndPolymerizeDnaAATP(self, a, d):
        return self._bind_polymerize(a, d, True)

    def bindAndPolymerizeDnaAADP(self, a, d):
        return self._bind_polymerize(a, d, False)

    def _bind_polymerize(self, a, d, atp):
        free = float(np.asarray(self.enzymes[1 if atp else 0]).item())
        if free == 0:
            return a, d
        pos, br, avg = (
            self.calculateDnaAATPBindingRates(a, d)
            if atp
            else self.calculateDnaAADPBindingRates(a, d)
        )
        _, vals = self.chromosome.complexBoundSites.find()
        monomerGlobals = np.asarray(self.enzymeComplexGlobalIndexs)[
            self._idx(
                [
                    self.enzymeIndexs_DnaA_1mer_ATP,
                    self.enzymeIndexs_DnaA_1mer_ADP,
                ]
            )
        ]
        numfree = len(br) - np.count_nonzero(np.isin(vals, monomerGlobals))
        n7 = int(np.count_nonzero(br[self._idx(self.dnaABoxIndexs_7mer)]))
        maxbind = self.randStream.stochasticRound(
            min(avg * numfree * free + n7, numfree, free)
        )
        rates = (
            self.calculateDnaAR1234ATPPolymerizationRates(a, d)
            if atp
            else self.calculateDnaAR1234ADPPolymerizationRates(a, d)
        )
        maxpol = 0
        if np.any(a):
            nsites = int(np.count_nonzero(rates))
            if nsites:
                maxpol = self.randStream.stochasticRound(
                    min(rates.sum() * free, nsites, free)
                )
        total = avg * numfree + rates.sum()
        if total:
            maxbind = min(
                maxbind,
                self.randStream.stochasticRound(
                    free * (avg * numfree) / total
                ),
            )
        if atp:
            a, d, nb = self.bindDnaAATP(a, d, maxbind, pos, br)
        else:
            a, d, nb = self.bindDnaAADP(a, d, maxbind, pos, br)
        n_poly = min(maxpol, free - nb)
        if n_poly == 0:
            return a, d
        idx = np.asarray(
            self.randStream.randsample(
                rates.size, n_poly, False, rates.reshape(-1, order="F")
            ),
            dtype=int,
        ).reshape(-1)
        rows = (idx - 1) % 4
        cols = (idx - 1) // 4
        oldATP = a[rows, cols].copy()
        if atp:
            a[rows, cols] += 1
            new = a[rows, cols]
            pos2 = np.c_[
                self.dnaABoxStartPositions[
                    self._idx(self.dnaABoxIndexs_R1234)
                ][rows],
                2 * cols + 1,
            ]
            self.modifyProteinOnChromosome(
                pos2, self.enzymeIndexs_DnaA_Nmer_ATP[new - 1]
            )
            counts = np.bincount(new, minlength=8)[2:8]
            self.enzymes[
                self._idx(self.enzymeIndexs_DnaA_Nmer_ATP[:6])
            ] -= counts[:, None]
            self.enzymes[
                self._idx(self.enzymeIndexs_DnaA_Nmer_ATP[1:])
            ] += counts[:, None]
            self.enzymes[1] -= n_poly
        else:
            d[rows, cols] = oldATP + 1
            a[rows, cols] = 0
            new = d[rows, cols]
            pos2 = np.c_[
                self.dnaABoxStartPositions[
                    self._idx(self.dnaABoxIndexs_R1234)
                ][rows],
                2 * cols + 1,
            ]
            self.modifyProteinOnChromosome(
                pos2, self.enzymeIndexs_DnaA_Nmer_ADP[new - 1]
            )
            counts = np.bincount(new, minlength=8)[2:8]
            self.enzymes[
                self._idx(self.enzymeIndexs_DnaA_Nmer_ATP[:6])
            ] -= counts[:, None]
            self.enzymes[
                self._idx(self.enzymeIndexs_DnaA_Nmer_ADP[1:])
            ] += counts[:, None]
            self.enzymes[0] -= n_poly
        return a, d

    def releaseDnaAAxP(self, a, d):
        # Find the bound DnaA complexes and their local enzyme indices.
        sites, vals = self.chromosome.complexBoundSites.find()
        complex_ids = np.asarray(self.enzymeComplexGlobalIndexs).reshape(-1)
        vals = np.asarray(vals).ravel()
        lookup = {int(value): i for i, value in enumerate(complex_ids)}
        local = np.array(
            [lookup.get(int(value), -1) for value in vals], dtype=int
        )
        is_dnaa = local >= 0

        # The minimum completed polymerized layer at R1-R4 is protected from
        # release.  At seven or more subunits, R5 is protected as well.
        min_pol = self.calculateDnaAR1234ComplexSize(a, d)
        box_pos = np.asarray(self.dnaABoxStartPositions).reshape(-1)
        r1234 = np.asarray(self.dnaABoxIndexs_R1234, dtype=int).reshape(-1) - 1
        r5 = np.asarray(self.dnaABoxIndexs_R5, dtype=int).reshape(-1) - 1
        for strand, column in ((1, 0), (3, 1)):
            protected = np.empty(0, dtype=int)
            if min_pol[column] > 0:
                protected = box_pos[
                    r1234[np.asarray(a)[:, column] == min_pol[column]]
                ]
                if min_pol[column] >= 7:
                    protected = np.r_[protected, box_pos[r5]]
            if protected.size:
                is_dnaa &= ~(
                    (sites[:, 1] == strand) & np.isin(sites[:, 0], protected)
                )

        # Release the remaining eligible complexes stochastically.
        chosen = np.zeros(len(sites), dtype=bool)
        chosen[is_dnaa] = (
            np.asarray(
                self.randStream.rand(np.count_nonzero(is_dnaa), 1)
            ).reshape(-1)
            < float(np.asarray(self.kd1ATP).item()) / 3600
        )
        if not np.any(chosen):
            return a, d
        local = local[chosen] + 1  # MATLAB enzyme-local indices are one based.
        positions = sites[chosen]
        remaining = np.asarray(
            self.dnaARelease_remainingDnaAIndexs, dtype=int
        ).ravel()[local - 1]

        # Monomers leave the chromosome; larger complexes shrink by one layer.
        self.releaseProteinFromSites(positions[remaining == 0], False)
        keep = remaining != 0
        if np.any(keep):
            remaining = remaining[keep]
            local = local[keep]
            self.modifyProteinOnChromosome(positions[keep], remaining)
            released = np.asarray(
                self.dnaARelease_releasedDnaAIndexs, dtype=int
            ).ravel()[local - 1]
            # Account for the consumed complex and the two products.
            np.add.at(self.enzymes, local - 1, -1)
            np.add.at(self.enzymes, remaining - 1, 1)
            np.add.at(self.enzymes, released - 1, 1)
        return a, d

    def calcuateIsDnaAR5Occupied(self):
        p = self.dnaABoxStartPositions[self._idx(self.dnaABoxIndexs_R5)]
        return np.asarray(
            [
                self.chromosome.complexBoundSites[x, s]
                == self.enzymeComplexGlobalIndexs[1]
                for x in p
                for s in (1, 3)
            ]
        )
