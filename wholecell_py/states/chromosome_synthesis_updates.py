"""Chromosome synthesis state mutations translated from ``Chromosome.m``."""

import numpy as np

from wholecell_py.circular_sparsemat import CircularSparseMat
from wholecell_py.side_effect import SimulationStateSideEffect


class ChromosomeSynthesisUpdates:
    """Mutations for DNA unwinding and polymerization (one-based coordinates)."""

    @staticmethod
    def shiftStrandToNewChromosome(spmat, pos, length, oldStrand, newStrand):
        subs, vals = spmat.find()
        subs = np.asarray(subs, dtype=int).reshape((-1, len(spmat.siz))).copy()
        vals = np.asarray(vals).reshape(-1)
        move = (
            (
                (subs[:, 0] >= pos)
                & (subs[:, 0] < pos + length)
                & (subs[:, 1] == oldStrand)
            )
            if len(subs)
            else np.zeros(0, bool)
        )
        subs[move, 1] = newStrand
        return CircularSparseMat(
            subs, vals, spmat.siz, getattr(spmat, "circularDims", [1])
        )

    def _coords(self, matrix, pos_strands):
        p = np.asarray(pos_strands, dtype=int).reshape((-1, 2))
        if len(matrix.siz) > 2:
            p = np.column_stack((p, np.ones(len(p), dtype=int)))
        return p

    def _set_regions(self, matrix, coords, values):
        return matrix.set_coords(self._coords(matrix, coords), values)

    def mergeOwnAdjacentRegions(self):
        regions = self.polymerizedRegions
        if regions.nnz() <= 1:
            return
        ds, lengths = self.doubleStrandedRegions.find()
        ds = np.asarray(ds, dtype=int).reshape(
            (-1, len(self.linkingNumbers.siz))
        )
        lengths = np.asarray(lengths).reshape(-1)
        lkpos, lkvals = self.linkingNumbers.find()
        lkpos = np.asarray(lkpos, dtype=int).reshape(
            (-1, len(self.linkingNumbers.siz))
        )
        lkvals = np.asarray(lkvals).reshape(-1)
        sums = np.zeros(len(lengths))
        for coord, value in zip(lkpos, lkvals):
            matches = (
                (
                    (coord[0] >= ds[:, 0])
                    & (coord[0] <= ds[:, 0] + lengths)
                    & np.all(ds[:, 1:] == coord[1:], axis=1)
                )
                if len(ds)
                else np.zeros(0, bool)
            )
            sums[matches] += value
        self.linkingNumbers = CircularSparseMat(
            ds, sums, self.linkingNumbers.siz, [1]
        )
        self.polymerizedRegions = self.mergeAdjacentRegions(regions)

    def mergeAdjacentRegions(self, regions):
        p, lengths = regions.find()
        p = np.asarray(p, dtype=int).reshape((-1, len(regions.siz)))
        lengths = np.asarray(lengths).reshape(-1).astype(int)
        if len(lengths) <= 1:
            return regions
        order = np.lexsort(tuple(p[:, i] for i in range(p.shape[1])))
        p, lengths = p[order], lengths[order]
        if np.any(lengths < 0) or any(
            np.array_equal(p[i + 1, 1:], p[i, 1:])
            and p[i + 1, 0] - p[i, 0] < lengths[i]
            for i in range(len(p) - 1)
        ):
            raise ValueError("polymerizedRegions is corrupt")
        outp, outl = [], []
        for coord, length in zip(p, lengths):
            if (
                outp
                and np.array_equal(coord[1:], outp[-1][1:])
                and coord[0] == outp[-1][0] + outl[-1]
            ):
                outl[-1] += int(length)
            else:
                outp.append(coord.copy())
                outl.append(int(length))
        return CircularSparseMat(
            np.asarray(outp).reshape(-1, len(regions.siz)),
            outl,
            regions.siz,
            [1],
        )

    def setRegionPolymerized(self, positionsStrands, lengths):
        p = np.asarray(positionsStrands, dtype=int)
        lens = np.asarray(lengths, dtype=int).reshape(-1)
        if p.ndim == 1:
            p = p.reshape(1, -1)
        if p.ndim != 2 or p.shape[1] != 2:
            raise ValueError("positionsStrands must have 2 columns")
        if len(p) != len(lens):
            raise ValueError(
                "positionsStrands and lengths must have same number of rows"
            )
        L = int(np.asarray(self.sequenceLen).item())
        if np.any((p[:, 1] != 1) & (p[:, 1] != 2)):
            raise ValueError(
                "positionsStrands must be a valid template strand (eg. 1 or 2)"
            )
        if np.any(p[:, 0] + lens - 1 > L):
            raise ValueError("positionsStrands cannot wrap ORI")
        p = p.copy()
        p[:, 0] += np.minimum(0, lens + 1)
        lens = np.abs(lens)
        for (pos, template), length in zip(p, lens):
            if length == 0:
                continue
            template_strand = int(
                np.asarray(self.strandIndexs_template).reshape(-1)[
                    template - 1
                ]
            )
            product_strand = int(
                np.asarray(self.strandIndexs_nonTemplate).reshape(-1)[
                    template - 1
                ]
            )
            if not self.isRegionPolymerized(
                [[pos, template_strand]], [length], False
            )[0][0]:
                raise ValueError(
                    "cannot polymerize a region without a template"
                )
            if not self.isRegionNotPolymerized(
                [[pos, product_strand]], [length], False
            )[0][0]:
                raise ValueError(
                    "cannot polymerize a region that's already been polymerized"
                )
            self.polymerizedRegions = self._set_regions(
                self.polymerizedRegions, [[pos, product_strand]], [length]
            )
            self.linkingNumbers = self._set_regions(
                self.linkingNumbers,
                [[pos, template_strand], [pos, product_strand]],
                [length / float(self.relaxedBasesPerTurn)] * 2,
            )
            self.mergeOwnAdjacentRegions()
        return SimulationStateSideEffect()

    def setRegionUnwound(self, positions, lengths):
        p = np.asarray(positions, dtype=int)
        lens = np.asarray(lengths, dtype=int)
        if p.ndim <= 1:
            p = p.reshape(-1, 1)
        if lens.ndim <= 1:
            lens = lens.reshape(-1, 1)
        if p.shape[0] != lens.shape[0]:
            raise ValueError(
                "positions and lengths must have same number of rows"
            )
        if p.shape[1] != 1:
            raise ValueError("positions must have 1 columns")
        if lens.shape[1] != 1:
            raise ValueError("lengths must have 1 column")
        if not np.any(lens):
            return SimulationStateSideEffect()
        L = int(np.asarray(self.sequenceLen).item())
        if np.any(p > L):
            raise ValueError("positions cannot wrap ORI")
        keep = lens[:, 0] != 0
        p = p[keep, 0]
        lens = lens[keep, 0]
        n = len(p)
        if not np.all(
            self.isRegionDoubleStranded(
                np.column_stack((p, np.ones(n, int))), lens, False
            )[0]
        ):
            raise ValueError("regions must be double-stranded")
        adjacent = (
            (p == 1)
            | (p == L)
            | self.isRegionSingleStranded(
                np.column_stack((np.maximum(1, p - 1), np.ones(n, int))),
                np.ones(n, int),
                False,
            )[0]
            | self.isRegionSingleStranded(
                np.column_stack((np.minimum(L, p + 1), np.ones(n, int))),
                np.ones(n, int),
                False,
            )[0]
        )
        if not np.all(adjacent):
            raise ValueError(
                "unwinding must begin at either end of dsDNA or continue where it left off"
            )
        if not np.all(
            self.isRegionNotPolymerized(
                np.vstack(
                    (
                        np.column_stack((p, np.full(n, 3))),
                        np.column_stack((p, np.full(n, 4))),
                    )
                ),
                np.r_[lens, lens],
                False,
            )[0]
        ):
            raise ValueError("chromosome 2 region cannot be polymerized")
        old = int(np.asarray(self.strandIndexs_ch1).reshape(-1)[1])
        new = int(np.asarray(self.strandIndexs_ch2).reshape(-1)[1])
        directions = np.sign(lens)
        p = p + np.minimum(0, lens + 1)
        lens = np.abs(lens)
        fields = (
            "monomerBoundSites",
            "complexBoundSites",
            "gapSites",
            "abasicSites",
            "damagedSugarPhosphates",
            "damagedBases",
            "intrastrandCrossLinks",
            "strandBreaks",
            "hollidayJunctions",
        )
        for pos, length, direction in zip(p, lens, directions):
            for field in fields:
                setattr(
                    self,
                    field,
                    self.shiftStrandToNewChromosome(
                        getattr(self, field), pos, length, old, new
                    ),
                )
            starts, reglengths = self.polymerizedRegions.find()
            starts = np.asarray(starts).reshape(
                (-1, len(self.polymerizedRegions.siz))
            )
            which = np.flatnonzero(
                (starts[:, 1] == old)
                & (starts[:, 0] <= pos)
                & (starts[:, 0] + np.asarray(reglengths).reshape(-1) > pos)
            )
            if len(which) != 1:
                raise ValueError(
                    "unwinding region does not identify exactly one polymerized region"
                )
            idx = which[0]
            start = int(starts[idx, 0])
            region_len = int(np.asarray(reglengths).reshape(-1)[idx])
            if direction == 1 and pos != start:
                raise ValueError("programmer error: unwinding bad region")
            if direction == 1:
                lk = self.linkingNumbers.get_coords(
                    self._coords(self.linkingNumbers, [[pos, old]])
                ).reshape(-1)[0]
                self.linkingNumbers = self._set_regions(
                    self.linkingNumbers,
                    [[pos + length, 1], [pos + length, 2]],
                    [lk, lk],
                )
                self.linkingNumbers = self._set_regions(
                    self.linkingNumbers, [[pos, 1], [pos, 2]], [0, 0]
                )
            if (direction == 1 and length == region_len) or (
                direction == -1 and pos == start
            ):
                lk = self.linkingNumbers.get_coords(
                    self._coords(self.linkingNumbers, [[pos, old]])
                ).reshape(-1)[0]
                if abs(lk) > 1e-6:
                    raise ValueError(
                        "cannot completely unwind region with nonzero linking number"
                    )
            remainder = region_len - (pos - start) - length
            self.polymerizedRegions = self._set_regions(
                self.polymerizedRegions, [[start, old]], [pos - start]
            )
            if remainder != 0:
                self.polymerizedRegions = self._set_regions(
                    self.polymerizedRegions, [[pos + length, old]], [remainder]
                )
            self.polymerizedRegions = self._set_regions(
                self.polymerizedRegions, [[pos, new]], [length]
            )
            self.mergeOwnAdjacentRegions()
        return SimulationStateSideEffect()
