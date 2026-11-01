"""Faithful numerical/runtime port of WholeCell DNASupercoiling.m."""

from __future__ import annotations

import numpy as np

from ..process_base import Process, _get
from ..chromosome_process_aspect import ChromosomeProcessAspect
from ..circular_sparsemat import CircularSparseMat


class DNASupercoiling(ChromosomeProcessAspect, Process):
    optionNames__ = ()
    fixedConstantNames__ = (
        "gyraseMeanDwellTime",
        "gyraseSigmaLimit",
        "topoISigmaLimit",
        "topoIVSigmaLimit",
        "gyraseDeltaLK",
        "topoIDeltaLK",
        "topoIVDeltaLK",
        "gyraseActivityRate",
        "topoIActivityRate",
        "topoIVActivityRate",
        "gyraseATPCost",
        "topoIATPCost",
        "topoIVATPCost",
        "foldChangeSlopes",
        "foldChangeIntercepts",
        "foldChangeLowerSigmaLimit",
        "foldChangeUpperSigmaLimit",
        "numTranscriptionUnits",
        "tuIndexs",
        "tuCoordinates",
    )
    fittedConstantNames__ = ()
    localStateNames__ = ()
    stimuliWholeCellModelIDs = ()
    substrateWholeCellModelIDs = ("ATP", "ADP", "PI", "H2O", "H")
    (
        substrateIndexs_atp,
        substrateIndexs_adp,
        substrateIndexs_phosphate,
        substrateIndexs_water,
        substrateIndexs_hydrogen,
    ) = (1, 2, 3, 4, 5)
    enzymeWholeCellModelIDs = (
        "DNA_GYRASE",
        "MG_203_204_TETRAMER",
        "MG_122_MONOMER",
    )
    enzymeIndexs_gyrase, enzymeIndexs_topoIV, enzymeIndexs_topoI = 1, 2, 3

    def __init__(
        self,
        wholeCellModelID="Process_DNASupercoiling",
        name="DNA Supercoiling",
        **properties,
    ):
        super().__init__(wholeCellModelID, name, **properties)
        self.chromosome = getattr(self, "chromosome", None)
        self.rnaPolymerase = getattr(self, "rnaPolymerase", None)

    def storeObjectReferences(self, simulation):
        Process.storeObjectReferences(self, simulation)
        statefn = getattr(simulation, "state", None)
        if callable(statefn):
            self.chromosome, self.rnaPolymerase = statefn(
                "Chromosome"
            ), statefn("RNAPolymerase")
        else:
            states = _get(simulation, "states")
            lookup = {str(k).lower(): v for k, v in states.items()}
            self.chromosome, self.rnaPolymerase = (
                lookup["chromosome"],
                lookup["rnapolymerase"],
            )
        self.states += [self.chromosome, self.rnaPolymerase]

    def initializeConstants(
        self, knowledgeBase=None, simulation=None, options=None, **kwargs
    ):
        for key, value in kwargs.items():
            setattr(self, key, value)
        if knowledgeBase is not None or simulation is not None:
            if knowledgeBase is None or simulation is None:
                raise ValueError(
                    "knowledgeBase and simulation must be supplied together"
                )
            Process.initializeConstants(
                self, knowledgeBase, simulation, options or {}
            )
            ChromosomeProcessAspect.initializeConstants(
                self, knowledgeBase, simulation
            )
        # Values are fitted/model data and must be supplied by the snapshot.
        required = (
            "gyraseMeanDwellTime",
            "gyraseSigmaLimit",
            "topoISigmaLimit",
            "topoIVSigmaLimit",
            "gyraseDeltaLK",
            "topoIDeltaLK",
            "topoIVDeltaLK",
            "gyraseActivityRate",
            "topoIActivityRate",
            "topoIVActivityRate",
            "gyraseATPCost",
            "topoIATPCost",
            "topoIVATPCost",
            "topoILogisiticConst",
            "gyrLogisiticConst",
            "foldChangeSlopes",
            "foldChangeIntercepts",
            "foldChangeLowerSigmaLimit",
            "foldChangeUpperSigmaLimit",
            "enzymeDNAFootprints",
            "enzymeGlobalIndexs",
            "enzymeMonomerLocalIndexs",
            "enzymeComplexLocalIndexs",
        )
        missing = [x for x in required if not hasattr(self, x)]
        if missing:
            raise NotImplementedError(
                "initializeConstants requires source snapshot constants: "
                + ", ".join(missing)
            )
        c = self.chromosome
        self.enzymeProperties = self.buildEnzymeProperties()
        for prop in self.enzymeProperties:
            for key in (
                "sigmaLimit",
                "atpCost",
                "deltaLK",
                "activityRate",
                "meanDwellTime",
            ):
                prop[key] = float(np.asarray(prop[key]).item())
        ids = ["MG_003", "MG_203", "MG_122"]
        geneids = list(np.asarray(self.gene.wholeCellModelIDs).reshape(-1))
        gi = [geneids.index(x) + 1 for x in ids]
        comp = np.asarray(self.rna.nascentRNAGeneComposition)
        self.tuIndexs = (
            np.flatnonzero(np.any(comp[np.asarray(gi) - 1, :] != 0, axis=0))
            + 1
        )
        self.numTranscriptionUnits = len(
            np.asarray(c.transcriptionUnitStartCoordinates).reshape(-1)
        )
        self.tuCoordinates = np.asarray(
            c.transcriptionUnitStartCoordinates
        ).reshape(-1)[self.tuIndexs - 1]

    def _gyrase_activity(self, s, c):
        return 1 / (
            1
            + np.exp(
                self.gyrLogisiticConst
                * (np.asarray(s) - c.equilibriumSuperhelicalDensity)
            )
        )

    def _topoiv_activity(self, s, c):
        return np.ones_like(np.asarray(s, dtype=float))

    def _topoi_activity(self, s, c):
        return 1 / (
            1
            + np.exp(
                self.topoILogisiticConst
                * (np.asarray(s) - c.equilibriumSuperhelicalDensity)
            )
        )

    def _topoi_accessible_space(
        self, start, strand, length, monpos, mons, cmppos, cmps, dmg
    ):
        """Mirror the source footprint lookup for fragmented-region Topo I."""
        c = self.chromosome
        mend = (
            (monpos[:, 0] >= start)
            & (monpos[:, 0] < start + length)
            & (np.ceil(monpos[:, 1] / 2) == strand)
        )
        cend = (
            (cmppos[:, 0] >= start)
            & (cmppos[:, 0] < start + length)
            & (np.ceil(cmppos[:, 1] / 2) == strand)
        )
        dend = (
            (dmg[:, 0] >= start)
            & (dmg[:, 0] < start + length)
            & (np.ceil(dmg[:, 1] / 2) == strand)
            if dmg.size
            else np.zeros(0, bool)
        )
        # MATLAB lines 446-454 compare ceil(bound compartment / 2) to the raw
        # ds strand, then index monomerDNAFootprints for complex-bound IDs.
        # Preserve both source behaviors; full-chromosome branches use complex
        # footprints and group compartment columns 1-2 vs 3-4.
        return (
            length
            - np.sum(c.monomerDNAFootprints[mons[mend] - 1])
            - np.sum(c.monomerDNAFootprints[cmps[cend] - 1])
            - np.sum(dend)
        )

    def _transient_topoi_binding_complete_chromosomes(
        self, lengths, enzyme_index
    ):
        """Allocate transient Topo I using source full-chromosome cases."""
        c = self.chromosome
        monpos, mons = c.monomerBoundSites.find()
        cmppos, cmps = c.complexBoundSites.find()
        dmgpos, _ = c.damagedSites.find()
        monpos, mons = np.asarray(monpos), np.asarray(mons).reshape(-1).astype(
            int
        )
        cmppos, cmps = np.asarray(cmppos), np.asarray(cmps).reshape(-1).astype(
            int
        )
        dmgpos = np.asarray(dmgpos)
        space = np.asarray(lengths, dtype=float).reshape(-1).copy()

        if len(space) == 1:
            space[0] -= np.sum(c.monomerDNAFootprints[mons - 1])
            space[0] -= np.sum(c.complexDNAFootprints[cmps - 1])
            space[0] -= len(dmgpos)
        else:
            for copy, start, stop in ((0, 1, 3), (1, 3, 5)):
                mon_mask = (monpos[:, 1] >= start) & (monpos[:, 1] < stop)
                cmp_mask = (cmppos[:, 1] >= start) & (cmppos[:, 1] < stop)
                dmg_mask = (
                    (dmgpos[:, 1] >= start) & (dmgpos[:, 1] < stop)
                    if dmgpos.size
                    else np.zeros(0, dtype=bool)
                )
                space[copy] -= np.sum(
                    c.monomerDNAFootprints[mons[mon_mask] - 1]
                )
                space[copy] -= np.sum(
                    c.complexDNAFootprints[cmps[cmp_mask] - 1]
                )
                space[copy] -= np.count_nonzero(dmg_mask)

        footprint = float(
            np.asarray(self.enzymeDNAFootprints).reshape(-1)[enzyme_index]
        )
        enzyme_count = float(np.asarray(self.enzymes)[enzyme_index, 0])
        if len(space) == 1:
            allocated = np.asarray([min(enzyme_count, space[0] / footprint)])
        else:
            with np.errstate(all="ignore"):
                allocated = (
                    min(enzyme_count, np.sum(space) / footprint)
                    * space
                    / np.sum(space)
                )

        if len(space) == 2:
            # MATLAB round() is half away from zero; roundHalfDown follows
            # floor(val) + (rem(val, 1) > .5), including negative values.
            half_up = lambda value: np.sign(value) * np.floor(abs(value) + 0.5)
            half_down = lambda value: np.floor(value) + (
                np.fmod(value, 1) > 0.5
            )
            draw = float(np.asarray(self.randStream.rand(1, 1)).item())
            if draw < 0.5:
                allocated[0] = half_up(allocated[0])
                allocated[1] = half_down(allocated[1])
            else:
                allocated[1] = half_up(allocated[1])
                allocated[0] = half_down(allocated[0])
        return allocated

    def buildEnzymeProperties(self):
        return [
            dict(
                idx=1,
                sigmaLimit=self.gyraseSigmaLimit,
                atpCost=self.gyraseATPCost,
                deltaLK=self.gyraseDeltaLK,
                activityRate=self.gyraseActivityRate,
                meanDwellTime=self.gyraseMeanDwellTime,
                probOfActivityFunc=self._gyrase_activity,
            ),
            dict(
                idx=2,
                sigmaLimit=self.topoIVSigmaLimit,
                atpCost=self.topoIVATPCost,
                deltaLK=self.topoIVDeltaLK,
                activityRate=self.topoIVActivityRate,
                meanDwellTime=np.nan,
                probOfActivityFunc=self._topoiv_activity,
            ),
            dict(
                idx=3,
                sigmaLimit=self.topoISigmaLimit,
                atpCost=self.topoIATPCost,
                deltaLK=self.topoIDeltaLK,
                activityRate=self.topoIActivityRate,
                meanDwellTime=0,
                probOfActivityFunc=self._topoi_activity,
            ),
        ]

    def restoreSnapshotRuntime(self):
        # MATLAB anonymous functions are data tags in the portable snapshot.
        # Rebuild the source-defined functions from the stored model constants.
        self.enzymeProperties = self.buildEnzymeProperties()
        for prop in self.enzymeProperties:
            for key in (
                "sigmaLimit",
                "atpCost",
                "deltaLK",
                "activityRate",
                "meanDwellTime",
            ):
                prop[key] = float(np.asarray(prop[key]).item())

    def calcResourceRequirements_LifeCycle(self, constants, states):
        bm = np.zeros(5)
        by = np.zeros(5)
        mn = np.zeros(3)
        mx = np.full(3, np.inf)
        c = self.chromosome
        t = constants.states.Time
        lk0 = float(c.sequenceLen) / float(c.relaxedBasesPerTurn)
        lkss = (1 + float(c.equilibriumSuperhelicalDensity)) * lk0
        nr = (-lkss - 2 * (lk0 - lkss)) / float(self.gyraseDeltaLK)
        top = self.enzymeProperties[2]
        gyr = self.enzymeProperties[0]
        nr += 0.25 * abs(
            float(
                np.asarray(states.monomers).ravel()[
                    int(np.asarray(self.enzymeGlobalIndexs).ravel()[2]) - 1
                ]
            )
            * top["activityRate"]
            * top["deltaLK"]
            / gyr["deltaLK"]
        )
        n_atp = nr * gyr["atpCost"]
        bm[0] = bm[3] = n_atp
        by[1] = by[2] = by[4] = n_atp
        mn[0] = (
            nr
            / float(t.replicationDuration)
            / float(self.gyraseActivityRate)
            / np.exp(
                np.log(2)
                * float(t.replicationInitiationDuration)
                / float(t.cellCycleLength)
            )
        )
        return bm, by, mn, mx

    def calcResourceRequirements_Current(self):
        c = self.chromosome
        density = c.superhelicalDensity
        _, sigma = (
            density.find() if hasattr(density, "find") else (None, density)
        )
        sigma = np.asarray(sigma).reshape(-1)
        sigma = sigma[np.isfinite(sigma)]
        if not sigma.size:
            sigma = np.array([0.0])
        p = self.enzymeProperties[0]
        n = p["atpCost"] * np.ceil(
            (
                np.asarray(self.enzymes)[0, 0]
                + np.asarray(self.boundEnzymes)[0, 0]
            )
            * p["activityRate"]
            * np.max(p["probOfActivityFunc"](sigma, c))
        )
        out = np.zeros_like(np.asarray(self.substrates), dtype=float)
        out[0, 0] = out[3, 0] = n
        return out

    def calcRNAPolymeraseBindingProbFoldChange(
        self, dsPosStrands=None, lengths=None, linkingNumbers=None
    ):
        c = self.chromosome
        if dsPosStrands is None:
            pos, lens = c.doubleStrandedRegions.find()
            pos = np.asarray(pos)
            lens = np.asarray(lens).reshape(-1)
            ix = np.flatnonzero(pos[:, 1] % 2 == 1)
            dsPosStrands = pos[ix]
            lengths = lens[ix]
            _, ln = c.linkingNumbers.find()
            linkingNumbers = np.asarray(ln).reshape(-1)[ix]
        lengths = np.asarray(lengths).reshape(-1)
        ln = np.asarray(linkingNumbers).reshape(-1)
        sig = (ln - lengths / float(c.relaxedBasesPerTurn)) / (
            lengths / float(c.relaxedBasesPerTurn)
        )
        out = np.ones(
            (
                len(
                    np.asarray(c.transcriptionUnitStartCoordinates).reshape(-1)
                ),
                2,
            )
        )
        for i, coord in enumerate(np.asarray(self.tuCoordinates).reshape(-1)):
            for k in np.flatnonzero(
                (np.asarray(dsPosStrands)[:, 0] <= coord)
                & (np.asarray(dsPosStrands)[:, 0] + lengths - 1 >= coord)
            ):
                s = np.clip(
                    sig[k],
                    self.foldChangeLowerSigmaLimit,
                    self.foldChangeUpperSigmaLimit,
                )
                out[
                    int(self.tuIndexs[i]) - 1,
                    int((dsPosStrands[k, 1] + 1) // 2) - 1,
                ] = (
                    self.foldChangeIntercepts[i] + self.foldChangeSlopes[i] * s
                )
        return out

    def initializeState(self):
        n = int(
            np.asarray(
                self.randStream.stochasticRound(
                    np.asarray(self.enzymes)[self.enzymeIndexs_gyrase - 1]
                    * (1 - 1 / self.gyraseMeanDwellTime / self.stepSizeSec)
                )
            ).item()
        )
        self.bindProteinToChromosomeStochastically(self.enzymeIndexs_gyrase, n)
        self.rnaPolymerase.supercoilingBindingProbFoldChange = (
            self.calcRNAPolymeraseBindingProbFoldChange()
        )

    def evolveState(self):
        c, props = self.chromosome, self.enzymeProperties
        pos, lens = c.doubleStrandedRegions.find()
        pos, lens = np.asarray(pos), np.asarray(lens).reshape(-1)
        ix = np.flatnonzero(pos[:, 1] % 2 == 1)
        ds, lens = pos[ix], lens[ix]
        _, lk = c.linkingNumbers.find()
        lk = np.asarray(lk).reshape(-1)[ix].astype(float)
        turn = float(np.asarray(c.relaxedBasesPerTurn).item())
        sigma = (lk - lens / turn) / (lens / turn)
        legal = np.column_stack(
            (
                sigma > float(np.asarray(self.gyraseSigmaLimit).item()),
                sigma > float(np.asarray(self.topoIVSigmaLimit).item()),
                sigma < float(np.asarray(self.topoISigmaLimit).item()),
            )
        )
        self.releaseProteinFromChromosome(
            self.enzymeIndexs_topoIV,
            np.inf,
            ds[legal[:, self.enzymeIndexs_topoIV - 1]],
            lens[legal[:, self.enzymeIndexs_topoIV - 1]],
        )
        self.releaseProteinFromChromosome(
            self.enzymeIndexs_gyrase, 1 / self.gyraseMeanDwellTime
        )
        transient = np.zeros(legal.shape)
        order = (
            np.asarray(self.randStream.randperm(len(self.enzymes)))
            .reshape(-1)
            .astype(int)
        )
        # randperm follows MATLAB's one-based index contract.
        for j1 in order:
            j = int(j1) - 1
            if props[j]["meanDwellTime"] == 0:
                if len(lens) in (1, 2) and np.all(lens == c.sequenceLen):
                    transient[:, j] = (
                        self._transient_topoi_binding_complete_chromosomes(
                            lens, j
                        )
                    )
                    continue
                # Topo I binds transiently and is limited by accessible DNA space.
                monpos, mons = c.monomerBoundSites.find()
                cmppos, cmps = c.complexBoundSites.find()
                dmg = c.damagedSites.find()[0]
                monpos, mons = np.asarray(monpos), np.asarray(mons).reshape(
                    -1
                ).astype(int)
                cmppos, cmps = np.asarray(cmppos), np.asarray(cmps).reshape(
                    -1
                ).astype(int)
                dmg = np.asarray(dmg)
                space = np.zeros(len(lens))
                for k, (start, strand) in enumerate(ds):
                    if not legal[k, j]:
                        continue
                    space[k] = self._topoi_accessible_space(
                        start, strand, lens[k], monpos, mons, cmppos, cmps, dmg
                    )
                fp = float(np.asarray(self.enzymeDNAFootprints).reshape(-1)[j])
                if len(lens) == 1:
                    transient[:, j] = min(
                        float(self.enzymes[j, 0]), space / fp
                    )
                elif space.sum():
                    transient[:, j] = (
                        space
                        / space.sum()
                        * min(float(self.enzymes[j, 0]), space.sum() / fp)
                    )
            elif np.any(legal[:, j]):
                self.bindProteinToChromosomeStochastically(
                    j + 1,
                    positionsStrands=ds[legal[:, j]],
                    lengths=lens[legal[:, j]],
                )
        for i in range(len(ds)):
            for prop in (
                props[k]
                for k in np.asarray(self.randStream.randperm(len(props)))
                .reshape(-1)
                .astype(int)
                - 1
            ):
                j = prop["idx"] - 1
                if not legal[i, j]:
                    continue
                nbound = (
                    transient[i, j]
                    if prop["meanDwellTime"] == 0
                    else len(
                        self.findProteinInRegion(
                            ds[i, 0], ds[i, 1], lens[i], prop["idx"]
                        )
                    )
                )
                n = int(
                    np.asarray(
                        self.randStream.stochasticRound(
                            nbound
                            * prop["activityRate"]
                            * prop["probOfActivityFunc"](sigma[i], c)
                        )
                    ).item()
                )
                if prop["atpCost"]:
                    n = min(
                        n,
                        int(
                            np.floor(
                                min(
                                    self.substrates[
                                        self.substrateIndexs_atp - 1
                                    ],
                                    self.substrates[
                                        self.substrateIndexs_water - 1
                                    ],
                                )
                                / prop["atpCost"]
                            )
                        ),
                    )
                lk[i] += prop["deltaLK"] * n
                atp = n * prop["atpCost"]
                self.substrates[self.substrateIndexs_atp - 1] -= atp
                self.substrates[self.substrateIndexs_water - 1] -= atp
                self.substrates[self.substrateIndexs_adp - 1] += atp
                self.substrates[self.substrateIndexs_phosphate - 1] += atp
                self.substrates[self.substrateIndexs_hydrogen - 1] += atp
        c.linkingNumbers = CircularSparseMat(
            np.vstack((ds, np.column_stack((ds[:, 0], ds[:, 1] + 1)))),
            np.r_[lk, lk],
            [c.sequenceLen, c.nCompartments],
            [1],
        )
        self.rnaPolymerase.supercoilingBindingProbFoldChange = (
            self.calcRNAPolymeraseBindingProbFoldChange(ds, lens, lk)
        )
