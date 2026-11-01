"""Process-side chromosome binding, release, and query helpers.

MATLAB-facing indices and coordinates stay one based at this boundary.
"""

from __future__ import annotations

import numpy as np

from .process_base import _get


def _v(value, dtype=None):
    return np.asarray(value, dtype=dtype).reshape(-1)


class ChromosomeProcessAspect:
    """Mixin port of ``ChromosomeProcessAspect.m`` for process classes."""

    def storeObjectReferences(self, simulation):
        statefn = getattr(simulation, "state", None)
        if callable(statefn):
            self.chromosome = statefn("Chromosome")
        else:
            states = _get(simulation, "states")
            self.chromosome = {str(k).lower(): v for k, v in states.items()}[
                "chromosome"
            ]
        self.states = [*(getattr(self, "states", []) or []), self.chromosome]

    def initializeConstants(
        self, knowledgeBase=None, simulation=None, **kwargs
    ):
        for key, value in kwargs.items():
            setattr(self, key, value)
        # MATLAB deliberately ignores these two inherited initializer
        # arguments. The enzyme ID/index arrays are established by the
        # enclosing Process initializer before this aspect derives footprints.
        required = (
            "enzymeWholeCellModelIDs",
            "enzymeMonomerLocalIndexs",
            "enzymeMonomerGlobalIndexs",
            "enzymeComplexLocalIndexs",
            "enzymeComplexGlobalIndexs",
        )
        missing = [x for x in required if not hasattr(self, x)]
        if missing:
            raise NotImplementedError(
                "ChromosomeProcessAspect snapshot lacks initialized fields: "
                + ", ".join(missing)
            )
        c = self.chromosome
        self.enzymeDNAFootprints = np.zeros(
            np.asarray(self.enzymeWholeCellModelIDs).size
        )
        ml, cl = _v(self.enzymeMonomerLocalIndexs, int), _v(
            self.enzymeComplexLocalIndexs, int
        )
        if ml.size:
            self.enzymeDNAFootprints[ml - 1] = _v(c.monomerDNAFootprints)[
                _v(self.enzymeMonomerGlobalIndexs, int) - 1
            ]
        if cl.size:
            self.enzymeDNAFootprints[cl - 1] = _v(c.complexDNAFootprints)[
                _v(self.enzymeComplexGlobalIndexs, int) - 1
            ]
        self.enzymeDNAFootprints3Prime, self.enzymeDNAFootprints5Prime = (
            c.calculateFootprintOverhangs(self.enzymeDNAFootprints)
        )

    def bindProteinToChromosome(
        self,
        positionsStrands,
        proteinIndexs,
        maxBindings=None,
        weights=None,
        isBindingStable=None,
        isPositionsStrandFootprintCentroid=None,
        processivityLengths=None,
        isBindingProcessive=None,
        ignoreDamageFilter=None,
        checkRegionSupercoiled=False,
    ):
        p = np.asarray(positionsStrands, dtype=int).reshape((-1, 2))
        maximum = (
            len(p) if maxBindings is None else min(int(maxBindings), len(p))
        )
        if maximum == 0:
            return (
                np.zeros(0, bool),
                np.zeros(0, int),
                np.empty((0, 2), int),
                0,
                np.zeros(0),
            )
        ids = np.asarray(proteinIndexs)
        if ids.dtype == bool:
            ids = np.flatnonzero(ids.reshape(-1)) + 1
        ids = _v(ids, int)
        ml, mg = _v(self.enzymeMonomerLocalIndexs, int), _v(
            self.enzymeMonomerGlobalIndexs, int
        )
        cl, cg = _v(self.enzymeComplexLocalIndexs, int), _v(
            self.enzymeComplexGlobalIndexs, int
        )
        mon = mg[np.isin(ml, ids)]
        comp = cg[np.isin(cl, ids)]
        w = np.ones(len(p)) if weights is None else weights
        stable = True if isBindingStable is None else isBindingStable
        centroid = (
            True
            if isPositionsStrandFootprintCentroid is None
            else isPositionsStrandFootprintCentroid
        )
        proc = False if isBindingProcessive is None else isBindingProcessive
        lengths = 1 if processivityLengths is None else processivityLengths
        result = self.chromosome.setSiteProteinBound(
            p,
            maximum,
            w,
            mon,
            comp,
            mg,
            cg,
            stable,
            centroid,
            lengths,
            proc,
            ignoreDamageFilter,
            checkRegionSupercoiled,
        )
        released_m, released_c, effects, tfs, idxs, positions, proc_lengths = (
            result
        )
        self._adjust_bound_counts(released_m, released_c)
        self._append_side_effects(effects)
        return tfs, idxs, positions, len(idxs), proc_lengths

    def _adjust_bound_counts(self, released_m, released_c):
        ml, cl = _v(self.enzymeMonomerLocalIndexs, int), _v(
            self.enzymeComplexLocalIndexs, int
        )
        rm, rc = _v(released_m), _v(released_c)
        for indices, values in ((ml, rm), (cl, rc)):
            if indices.size:
                shape = np.asarray(self.boundEnzymes)[indices - 1].shape
                self.boundEnzymes[indices - 1] -= values.reshape(shape)
                self.enzymes[indices - 1] += values.reshape(
                    np.asarray(self.enzymes)[indices - 1].shape
                )

    def _append_side_effects(self, effects):
        if np.size(effects):
            old = getattr(
                self, "simulationStateSideEffects", np.empty((0, 1), object)
            )
            self.simulationStateSideEffects = np.concatenate(
                (np.asarray(old).reshape(-1), np.asarray(effects).reshape(-1))
            ).reshape((-1, 1))

    def modifyProteinOnChromosome(self, positionsStrands, newProteinIndexs):
        p = np.asarray(positionsStrands, dtype=int).reshape((-1, 2))
        if not len(p):
            return
        ids = np.asarray(newProteinIndexs)
        ml, mg = _v(self.enzymeMonomerLocalIndexs, int), _v(
            self.enzymeMonomerGlobalIndexs, int
        )
        cl, cg = _v(self.enzymeComplexLocalIndexs, int), _v(
            self.enzymeComplexGlobalIndexs, int
        )
        if ids.size == 1:
            i = int(ids.item())
            nm = mg[ml == i]
            nc = cg[cl == i]
            if nm.size:
                nc = np.array([0])
            elif nc.size:
                nm = np.array([0])
            else:
                nm, nc = np.array([0]), np.array([0])
        else:
            ids = _v(ids, int)
            nm = np.zeros(len(p), int)
            nc = np.zeros(len(p), int)
            for i, x in enumerate(ids):
                a = mg[ml == x]
                b = cg[cl == x]
                if a.size:
                    nm[i] = a[0]
                if b.size:
                    nc[i] = b[0]
        rm, rc, effects = self.chromosome.modifyBoundProtein(p, nm, nc, mg, cg)
        self._append_side_effects(effects)
        self._adjust_bound_counts(rm, rc)

    def releaseProteinFromChromosome(
        self,
        proteinLclIdx,
        rate,
        protectedPositionsStrands=None,
        protectedLengths=None,
    ):
        if proteinLclIdx is None or np.size(proteinLclIdx) == 0 or rate == 0:
            return np.empty((0, 2), int)
        i = int(proteinLclIdx)
        ml = _v(self.enzymeMonomerLocalIndexs, int)
        gi = _v(self.enzymeGlobalIndexs, int)[i - 1]
        mon, comp = (gi, None) if np.any(ml == i) else (None, gi)
        p, effects = self.chromosome.stochasticallySetProteinUnbound(
            mon,
            comp,
            rate * self.stepSizeSec,
            protectedPositionsStrands,
            protectedLengths,
            False,
            False,
        )
        self._append_side_effects(effects)
        n = len(p)
        if n:
            self.enzymes[i - 1] += n
            self.boundEnzymes[i - 1] -= n
        return p

    def releaseProteinFromSites(
        self,
        positionsStrands,
        bothStrands=None,
        proteinLclIdx=None,
        isPositionsStrandFootprintCentroid=True,
        suspendExternalStateUpdating=False,
    ):
        released = np.zeros_like(np.asarray(self.enzymes))
        p = np.asarray(positionsStrands, dtype=int).reshape((-1, 2)).copy()
        if not len(p):
            return released
        both = True if bothStrands is None else bothStrands
        if proteinLclIdx is not None and isPositionsStrandFootprintCentroid:
            i = int(proteinLclIdx) - 1
            odd = p[:, 1] % 2 == 1
            p[odd, 0] -= int(_v(self.enzymeDNAFootprints5Prime)[i])
            p[~odd, 0] -= int(_v(self.enzymeDNAFootprints3Prime)[i])
        rm, rc, effects = self.chromosome.setRegionProteinUnbound(
            p,
            1,
            self.enzymeMonomerGlobalIndexs,
            self.enzymeComplexGlobalIndexs,
            both,
            both,
            suspendExternalStateUpdating,
            False,
        )
        ml, cl = _v(self.enzymeMonomerLocalIndexs, int), _v(
            self.enzymeComplexLocalIndexs, int
        )
        self._adjust_bound_counts(rm, rc)
        if ml.size:
            released[ml - 1] = _v(rm).reshape(released[ml - 1].shape)
        if cl.size:
            released[cl - 1] = _v(rc).reshape(released[cl - 1].shape)
        self._append_side_effects(effects)
        return released

    def isDnaBound(self, positionStrands, enzymeIndexs):
        p = np.asarray(positionStrands, dtype=int).reshape((-1, 2))
        ids = _v(enzymeIndexs, int)
        out = np.zeros(len(p), bool)
        ml = _v(self.enzymeMonomerLocalIndexs, int)
        for j, (pos, idx) in enumerate(zip(p, ids)):
            mono = np.any(ml == idx)
            gi = int(_v(self.enzymeGlobalIndexs, int)[idx - 1])
            matrix = (
                self.chromosome.monomerBoundSites
                if mono
                else self.chromosome.complexBoundSites
            )
            subs, vals = matrix.find()
            out[j] = np.any(
                np.all(subs == pos, axis=1) & (vals.reshape(-1) == gi)
            )
        return out

    def findProteinInRegion(self, position, strand, regionLength, enzymeIndex):
        if regionLength == 0:
            return np.empty((0, 2), int)
        ml = _v(self.enzymeMonomerLocalIndexs, int)
        gi = int(_v(self.enzymeGlobalIndexs, int)[int(enzymeIndex) - 1])
        matrix = (
            self.chromosome.monomerBoundSites
            if np.any(ml == enzymeIndex)
            else self.chromosome.complexBoundSites
        )
        subs, vals = matrix.find()
        mask = (
            (vals.reshape(-1) == gi)
            & (subs[:, 1] == strand)
            & (subs[:, 0] >= position)
            & (subs[:, 0] < position + regionLength)
        )
        return subs[mask]

    def bindProteinToChromosomeStochastically(
        self,
        enzymeIndex,
        nProteins=None,
        positionsStrands=None,
        lengths=None,
        calcRegionWeightsFun=None,
        calcBindingPositionFun=None,
        calcNewRegionsFun=None,
        checkRegionSupercoiled=False,
    ):
        if nProteins is None:
            nProteins = int(np.asarray(self.enzymes)[int(enzymeIndex) - 1])
        if nProteins == 0:
            return 0, np.empty((0, 2), int)
        c = self.chromosome
        fp = float(_v(self.enzymeDNAFootprints)[int(enzymeIndex) - 1])
        supplied = positionsStrands is not None and not (
            np.asarray(positionsStrands).size == 1
            and np.isnan(np.asarray(positionsStrands)).item()
        )
        if not supplied:
            positionsStrands, lengths = c.polymerizedRegions.find()
        if calcRegionWeightsFun is None:
            calcRegionWeightsFun = lambda L: np.maximum(
                0, np.asarray(L) - fp + 1
            )
        if calcBindingPositionFun is None:
            calcBindingPositionFun = lambda r, L: int(
                np.ceil(float(np.asarray(r).item()) * (L - fp + 1))
            )
        if calcNewRegionsFun is None:

            def calcNewRegionsFun(pos, lens, probs, idx, offset):
                pos = np.asarray(pos)
                lens = np.asarray(lens).copy()
                probs = np.asarray(probs).copy()
                pos = np.vstack(
                    (pos, [pos[idx, 0] + offset + fp, pos[idx, 1]])
                )
                lens = np.r_[lens, lens[idx] - offset - fp]
                lens[idx] = offset
                probs[idx] = calcRegionWeightsFun(lens[idx])
                probs = np.r_[probs, calcRegionWeightsFun(lens[-1])]
                return pos, lens, probs

        accessible = c.getAccessibleRegions(
            _v(self.enzymeMonomerGlobalIndexs, int)[
                np.isin(_v(self.enzymeMonomerLocalIndexs, int), [enzymeIndex])
            ],
            _v(self.enzymeComplexGlobalIndexs, int)[
                np.isin(_v(self.enzymeComplexLocalIndexs, int), [enzymeIndex])
            ],
        )
        pos, lens = c.intersectRegions(*accessible, positionsStrands, lengths)
        probs = np.asarray(calcRegionWeightsFun(lens), float).reshape(-1)
        selected = []
        for _ in range(int(nProteins)):
            if not np.any(probs):
                break
            idx = (
                int(self.randStream.randsample(len(probs), 1, True, probs)[0])
                - 1
            )
            offset = (
                int(calcBindingPositionFun(self.randStream.rand(), lens[idx]))
                - 1
            )
            selected.append([int(pos[idx, 0] + offset), int(pos[idx, 1])])
            pos, lens, probs = calcNewRegionsFun(pos, lens, probs, idx, offset)
        selected = np.asarray(selected, dtype=int).reshape((-1, 2))
        if not supplied and len(selected):
            selected[:, 0] = (selected[:, 0] - 1) % int(c.sequenceLen) + 1
        if len(selected):
            bound = self.bindProteinToChromosome(
                selected,
                enzymeIndex,
                len(selected),
                None,
                None,
                False,
                1,
                False,
                None,
                checkRegionSupercoiled,
            )
            if not np.all(bound[0]):
                raise RuntimeError(
                    "Region should be accessible: selected protein binding failed"
                )
        return len(selected), selected
