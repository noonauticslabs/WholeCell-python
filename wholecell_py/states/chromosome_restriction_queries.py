"""Restriction modification site status queries for chromosome."""

from __future__ import annotations

import numpy as np

from ..circular_sparsemat import CircularSparseMat
from ..sparsemat import SparseMat


def _lookup(mat, coords):
    coords = np.asarray(coords, dtype=int)
    if coords.size == 0:
        return np.zeros(0, dtype=int)
    return np.asarray(mat.get_coords(coords)).reshape(-1)


class ChromosomeRestrictionQueries:
    """MunI and general restriction site status queries (MATLAB coordinates)."""

    def calcRestrictableMunIRMSites(self):
        dr = self.dnaRepair
        sites = np.asarray(dr.RM_MunI_RecognitionSites, dtype=int)
        n = len(sites)
        mpos = np.asarray(dr.RM_MunI_MethylatedPositions, dtype=int).reshape(
            -1
        )
        rpos = np.asarray(dr.RM_MunI_RestrictionPositions, dtype=int).reshape(
            -1
        )
        methyl = np.concatenate(
            [
                np.column_stack((sites[:, mpos[0] - 1], np.full(n, s)))
                for s in (1, 3)
            ]
            + [
                np.column_stack((sites[:, mpos[1] - 1], np.full(n, s)))
                for s in (2, 4)
            ]
        )
        methylated = _lookup(self.damagedBases, methyl).reshape(
            (2 * n, 2), order="F"
        ) == int(np.asarray(self.metabolite.m6ADIndexs).item())
        rcoords = np.concatenate(
            [
                np.column_stack((sites[:, rpos[0] - 1], np.full(n, strand)))
                for strand in (1, 3)
            ]
            + [
                np.column_stack((sites[:, rpos[1] - 1], np.full(n, strand)))
                for strand in (2, 4)
            ]
        )
        width = sites.shape[1]

        def damage(strand):
            coords = np.column_stack(
                (sites.ravel(order="F"), np.full(n * width, strand))
            )
            return _lookup(self.damagedSites_shifted_incm6AD, coords).reshape(
                (n, width), order="F"
            )

        damaged = np.concatenate(
            (
                np.stack((damage(1), damage(2)), axis=2),
                np.stack((damage(3), damage(4)), axis=2),
            ),
            axis=0,
        )

        def breaks(strand, pos):
            return _lookup(
                self.strandBreaks,
                np.column_stack((sites[:, pos - 1], np.full(n, strand))),
            )

        # Preserve MATLAB's logical *linear* indexing here, including its short
        # mask against the larger recognition-position tensor.
        mask = np.stack(
            (
                np.column_stack((breaks(1, rpos[0]), breaks(2, rpos[1]))),
                np.column_stack((breaks(3, rpos[0]), breaks(4, rpos[1]))),
            ),
            axis=2,
        )
        flat = damaged.ravel(order="F").copy()
        flat[np.flatnonzero(mask.ravel(order="F"))] = 0
        damaged = flat.reshape(damaged.shape, order="F")
        clean = ~np.any(damaged != 0, axis=(1, 2))
        unmethylated = ~methylated.any(axis=1)
        polymerized = np.asarray(
            self.isRegionPolymerized(rcoords, 1, False, False, False)[0], bool
        ).ravel()
        selected = np.tile(unmethylated & clean, 2) & polymerized
        return CircularSparseMat(
            rcoords[selected],
            True,
            [int(self.sequenceLen), int(self.nCompartments)],
            1,
        )

    def calcHemiunmethylatedMunIRMSites(self):
        dr = self.dnaRepair
        sites = np.asarray(dr.RM_MunI_RecognitionSites, dtype=int)
        n = len(sites)
        mp = np.asarray(dr.RM_MunI_MethylatedPositions, dtype=int).reshape(-1)
        coords = np.concatenate(
            [
                np.column_stack((sites[:, mp[0] - 1], np.full(n, s)))
                for s in (1, 3)
            ]
            + [
                np.column_stack((sites[:, mp[1] - 1], np.full(n, s)))
                for s in (2, 4)
            ]
        )
        methyl = _lookup(self.damagedBases, coords).reshape(
            (2 * n, 2), order="F"
        ) == int(np.asarray(self.metabolite.m6ADIndexs).item())
        hemi = methyl.any(axis=1) & ~methyl.all(axis=1)
        selected = np.tile(hemi, 2) & ~methyl.ravel(order="F")
        return CircularSparseMat(
            coords[selected],
            1,
            [int(self.sequenceLen), int(self.nCompartments)],
            1,
        )

    def rmStatus(
        self,
        sites,
        methylatedPositions,
        restrictionPositions,
        bindingMonomers=None,
        bindingComplexs=None,
    ):
        # MATLAB's damagedRegions expression is intentionally expanded here:
        # each term has strand/position exceptions that a generic damagedSites
        # query would lose.  Keep time as an explicit third sparse coordinate.
        sites = np.asarray(sites, dtype=int)
        mp = np.asarray(methylatedPositions, dtype=int).reshape(-1)
        rp = np.asarray(restrictionPositions, dtype=int).reshape(-1)
        n, width = sites.shape
        nchrom = int(np.asarray(self.nCompartments).item()) // 2
        nt = (
            int(self.damagedBases.siz[2])
            if len(self.damagedBases.siz) > 2
            else 1
        )
        m6ad = int(np.asarray(self.metabolite.m6ADIndexs).item())

        def at(mat, pos, strand, t):
            pos = np.asarray(pos, dtype=int).reshape(-1)
            if not len(pos):
                return np.zeros((n, 0), dtype=int)
            coords = np.column_stack(
                (pos, np.full(len(pos), strand), np.full(len(pos), t))
            )
            return _lookup(mat, coords).reshape(n, -1)

        # Each chromosome occupies an adjacent odd/even strand pair.
        damaged = np.zeros((n, nchrom, nt), dtype=bool)
        methylated = np.zeros_like(damaged)
        hemi = np.zeros_like(damaged)
        unmethylated = np.zeros_like(damaged)
        cleaved = np.zeros_like(damaged)
        nonmeth1 = np.delete(np.arange(width), mp[0] - 1)
        nonmeth2 = np.delete(np.arange(width), mp[1] - 1)
        nonrest1 = np.delete(np.arange(width), rp[0] - 1)
        nonrest2 = np.delete(np.arange(width), rp[1] - 1)
        # The MATLAB break ranges are offset by one at their respective ends.
        br1, br2 = nonrest1[:-1], nonrest2[1:]
        for c in range(nchrom):
            s1, s2 = 2 * c + 1, 2 * c + 2
            for t in range(1, nt + 1):
                a = at(self.damagedBases, sites[:, mp[0] - 1], s1, t)[:, 0]
                b = at(self.damagedBases, sites[:, mp[1] - 1], s2, t)[:, 0]
                methylated[:, c, t - 1] = (a == m6ad) & (b == m6ad)
                hemi[:, c, t - 1] = (a == m6ad) ^ (b == m6ad)
                unmethylated[:, c, t - 1] = (a != m6ad) & (b != m6ad)
                cb1 = (
                    at(self.strandBreaks, sites[:, rp[0] - 1], s1, t)[:, 0]
                    != 0
                )
                cb2 = (
                    at(self.strandBreaks, sites[:, rp[1] - 1], s2, t)[:, 0]
                    != 0
                )
                cleaved[:, c, t - 1] = cb1 & cb2
                d = (
                    ((a != m6ad) & (a != 0))
                    | ((b != m6ad) & (b != 0))
                    | (cb1 ^ cb2)
                )
                d |= np.any(
                    at(self.damagedBases, sites[:, nonmeth1], s1, t) != 0,
                    axis=1,
                )
                d |= np.any(
                    at(self.damagedBases, sites[:, nonmeth2], s2, t) != 0,
                    axis=1,
                )
                d |= (
                    at(self.damagedBases, sites[:, mp[0] - 1], s2, t)[:, 0]
                    != 0
                )
                d |= (
                    at(self.damagedBases, sites[:, mp[1] - 1], s1, t)[:, 0]
                    != 0
                )
                d |= np.any(
                    at(self.strandBreaks, sites[:, br1], s1, t) != 0, axis=1
                )
                d |= np.any(
                    at(self.strandBreaks, sites[:, br2], s2, t) != 0, axis=1
                )
                d |= (
                    at(self.strandBreaks, sites[:, rp[0] - 1], s2, t)[:, 0]
                    != 0
                )
                d |= (
                    at(self.strandBreaks, sites[:, rp[1] - 1], s1, t)[:, 0]
                    != 0
                )
                for mat in (
                    self.intrastrandCrossLinks,
                    self.damagedSugarPhosphates,
                    self.abasicSites,
                    self.gapSites,
                ):
                    d |= np.any(at(mat, sites, s1, t) != 0, axis=1)
                    d |= np.any(at(mat, sites, s2, t) != 0, axis=1)
                d |= np.any(
                    at(self.hollidayJunctions, sites[:, :-1], s1, t) != 0,
                    axis=1,
                )
                d |= np.any(
                    at(self.hollidayJunctions, sites[:, 1:], s2, t) != 0,
                    axis=1,
                )
                damaged[:, c, t - 1] = d

        inaccessible = damaged | (cleaved & ~unmethylated)
        # Accessibility uses the same paired-strand start coordinates as the
        # source. Its damage mask is supplied above explicitly for each time.
        site_strands = np.vstack(
            (
                np.column_stack((sites[:, 0], np.ones(n, int))),
                np.column_stack((sites[:, 0], np.full(n, 3, int))),
            )
        )
        access = np.asarray(
            self.isRegionAccessible(
                site_strands,
                width,
                bindingMonomers,
                bindingComplexs,
                True,
                np.arange(1, width + 1),
                False,
                True,
            )[0],
            dtype=bool,
        ).reshape(-1)
        accessible = access.reshape((n, nchrom), order="F")
        inaccessible |= (~accessible)[:, :, None]
        methylated &= ~inaccessible
        hemi &= ~inaccessible
        unmethylated &= ~inaccessible & ~cleaved
        cleaved &= ~inaccessible
        shape = [n, nchrom, nt]

        def sparse(mask):
            subs = np.argwhere(mask).astype(int) + 1
            return SparseMat(subs, np.ones(len(subs), dtype=bool), shape)

        return tuple(
            sparse(x)
            for x in (unmethylated, hemi, methylated, cleaved, inaccessible)
        )
