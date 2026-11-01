"""Protein binding and release updates ported from Chromosome.m."""

import numpy as np

from ..circular_sparsemat import CircularSparseMat
from ..side_effect import SimulationStateSideEffect
from ..side_effect_item import SimulationStateSideEffectItem


class ChromosomeBindingUpdates:
    """Mixin for chromosome protein occupancy changes.

    The host supplies footprint and strandedness metadata plus its RNAP state.
    """

    @staticmethod
    def _bound_rows(matrix):
        return matrix.subs.copy(), matrix.vals.reshape(-1).copy()

    def setRegionProteinUnbound(
        self,
        positionsStrands,
        lengths,
        mainEffects_monomerIndexs,
        mainEffects_complexIndexs,
        regionBothStrands,
        bindingBothStrands,
        suspendExternalStateUpdating,
        proteinIsDegraded,
    ):
        p = np.asarray(positionsStrands, dtype=int).reshape((-1, 2))
        lengths = np.asarray(lengths, dtype=int).reshape(-1)
        if len(lengths) == 1:
            lengths = np.repeat(lengths, len(p))
        if len(p) != len(lengths):
            raise ValueError(
                "positionsStrands and lengths must have equal row counts"
            )
        n = int(np.asarray(self.sequenceLen).item())
        starts = (p[:, 0] + np.minimum(0, lengths + 1) - 1) % n + 1
        ends = starts + np.abs(lengths) - 1
        regions = []
        for i, (start, end, strand) in enumerate(zip(starts, ends, p[:, 1])):
            if end > n:
                regions.extend(
                    (
                        (int(start), n, int(strand)),
                        (1, int(end - n), int(strand)),
                    )
                )
            else:
                regions.append((int(start), int(end), int(strand)))
        if bindingBothStrands:
            regions = [(a, b, (s + 1) // 2) for a, b, s in regions]

        outputs, effects = [], []
        for kind in ("monomer", "complex"):
            field = kind + "BoundSites"
            matrix = getattr(self, field)
            subs, vals = self._bound_rows(matrix)
            footprints = np.asarray(getattr(self, kind + "DNAFootprints"))[
                vals.astype(int) - 1
            ].reshape(-1)
            binding_strand = getattr(
                self, kind + "DNAFootprintBindingStrandedness", None
            )
            ds = getattr(self, "dnaStrandedness_dsDNA", None)
            remove = np.zeros(len(vals), dtype=bool)
            for i, ((start, strand), footprint) in enumerate(
                zip(subs, footprints)
            ):
                end = int(start + footprint - 1)
                pieces = [(int(start), min(end, n))]
                if end > n:
                    pieces.append((1, end - n))
                strands = [int(strand)]
                if bindingBothStrands:
                    strands = [(int(strand) + 1) // 2]
                elif (
                    regionBothStrands
                    and binding_strand is not None
                    and ds is not None
                ):
                    if np.asarray(binding_strand)[int(vals[i]) - 1] == ds:
                        strands = [
                            2 * ((int(strand) + 1) // 2) - 1,
                            2 * ((int(strand) + 1) // 2),
                        ]
                for ra, rb, rs in regions:
                    for ps in strands:
                        if rs == ps and any(
                            a <= rb and b >= ra for a, b in pieces
                        ):
                            remove[i] = True
            main = np.asarray(
                (
                    mainEffects_monomerIndexs
                    if kind == "monomer"
                    else mainEffects_complexIndexs
                ),
                dtype=int,
            ).reshape(-1)
            released = np.zeros((main.size, 1), dtype=float)
            for value in vals[remove]:
                idx = np.flatnonzero(main == value)
                if idx.size:
                    released[idx[0], 0] += 1
                elif value != 0:
                    state = (
                        "ProteinMonomer"
                        if kind == "monomer"
                        else "ProteinComplex"
                    )
                    effects.append(
                        SimulationStateSideEffect(
                            [
                                SimulationStateSideEffectItem(
                                    state,
                                    "counts",
                                    "matureIndexs",
                                    int(value),
                                    int(self.compartment.cytosolIndexs),
                                    1,
                                ),
                                SimulationStateSideEffectItem(
                                    state,
                                    "counts",
                                    "boundIndexs",
                                    int(value),
                                    int(self.compartment.cytosolIndexs),
                                    -1,
                                ),
                            ]
                        )
                    )
            outputs.append(released)
            setattr(
                self,
                field,
                CircularSparseMat(subs[~remove], vals[~remove], matrix.siz, 1),
            )
            if kind == "monomer":
                monomer_subs, monomer_vals = subs[remove], vals[remove]
            else:
                complex_subs, complex_vals = subs[remove], vals[remove]
        if not suspendExternalStateUpdating:
            self.updateExternalState(
                monomer_subs,
                monomer_vals,
                complex_subs,
                complex_vals,
                proteinIsDegraded,
            )
        return (
            outputs[0],
            outputs[1],
            np.asarray(effects, dtype=object).reshape((-1, 1)),
        )

    def stochasticallySetProteinUnbound(
        self,
        monomerIndex,
        complexIndex,
        rate,
        protectedPositionsStrands,
        protectedLengths,
        suspendExternalStateUpdating,
        proteinIsDegraded,
    ):
        if (
            monomerIndex is not None
            and np.size(monomerIndex)
            and complexIndex is not None
            and np.size(complexIndex)
        ):
            raise ValueError("Can only unbind one protein at a time")
        kind = (
            "monomer"
            if monomerIndex is not None and np.size(monomerIndex)
            else (
                "complex"
                if complexIndex is not None and np.size(complexIndex)
                else None
            )
        )
        if kind is None:
            return np.empty((0, 2), dtype=int), np.empty((0, 1), dtype=object)
        target = int(
            np.asarray(
                monomerIndex if kind == "monomer" else complexIndex
            ).item()
        )
        matrix = getattr(self, kind + "BoundSites")
        subs, vals = self._bound_rows(matrix)
        eligible = vals == target
        if not np.any(eligible):
            return np.empty((0, 2), dtype=int), np.empty((0, 1), dtype=object)
        rate = float(np.asarray(rate).item())
        selected = eligible.copy()
        if np.isfinite(rate):
            selected[eligible] = (
                np.asarray(
                    self.randStream.rand(int(eligible.sum()), 1)
                ).reshape(-1)
                < rate
            )
        protected = np.asarray(
            (
                []
                if protectedPositionsStrands is None
                else protectedPositionsStrands
            ),
            dtype=int,
        ).reshape((-1, 2))
        plen = np.asarray(
            [] if protectedLengths is None else protectedLengths, dtype=int
        ).reshape(-1)
        if len(protected):
            footprint = int(
                self.getDNAFootprint(monomerIndex, complexIndex)[0]
            )
            for (pos, strand), length in zip(protected, plen):
                overlap = (
                    (subs[:, 0] <= pos + length - 1)
                    & (subs[:, 0] + footprint - 1 >= pos)
                    & (subs[:, 1] == strand)
                )
                selected &= ~overlap
        removed = selected
        setattr(
            self,
            kind + "BoundSites",
            CircularSparseMat(subs[~removed], vals[~removed], matrix.siz, 1),
        )
        if not suspendExternalStateUpdating:
            empty_subs, empty_vals = np.empty((0, 2), int), np.empty(0, int)
            if kind == "monomer":
                self.updateExternalState(
                    subs[removed],
                    vals[removed],
                    empty_subs,
                    empty_vals,
                    proteinIsDegraded,
                )
            else:
                self.updateExternalState(
                    empty_subs,
                    empty_vals,
                    subs[removed],
                    vals[removed],
                    proteinIsDegraded,
                )
        return subs[removed], np.empty((0, 1), dtype=object)

    def updateExternalState(
        self,
        unbindingMonomerSubs,
        unbindingMonomerVals,
        unbindingComplexSubs,
        unbindingComplexVals,
        proteinIsDegraded,
    ):
        values = np.asarray(unbindingComplexVals).reshape(-1)
        positions = np.asarray(unbindingComplexSubs, dtype=int).reshape(
            (-1, 2)
        )
        rnap = np.asarray(self.complexIndexs_rnaPolymerase).reshape(-1)
        positions = positions[np.isin(values, rnap)]
        if len(positions):
            _, fp3, fp5, _, _ = self.getDNAFootprint(
                [], self.complexIndexs_rnaPolymerase
            )
            fp3, fp5 = int(np.asarray(fp3).reshape(-1)[0]), int(
                np.asarray(fp5).reshape(-1)[0]
            )
            positions = positions.copy()
            odd = positions[:, 1] % 2 == 1
            positions[odd, 0] += fp5
            positions[~odd, 0] += fp3
            self.rnaPolymerase.releasePolymerase(positions, proteinIsDegraded)
