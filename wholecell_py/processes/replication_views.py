"""Live derived views for the chromosome replication process.

Coordinates exposed here follow the MATLAB source and remain one-based.
"""

from __future__ import annotations

import numpy as np


class ReplicationViews:
    @property
    def isDnaAORIComplexAssembled(self):
        c = self.chromosome
        ps, complexes = c.complexBoundSites.find()
        seven = np.flatnonzero(
            complexes.reshape(-1) == self.complexIndexs_DnaA_7mer_ATP
        )
        if len(seven) < 4:
            return False
        r5 = np.asarray(self.dnaAFunctionalBoxStartPositions).ravel()[
            np.asarray(self.dnaAFunctionalBoxIndexs_R5, int).ravel() - 1
        ]
        one = np.flatnonzero(
            (ps[:, 0] == r5)
            & (complexes.reshape(-1) == self.complexIndexs_DnaA_1mer_ATP)
            & (ps[:, 1] == 1)
        )
        if not len(one):
            return False
        r1234 = (
            np.asarray(self.dnaAFunctionalBoxStartPositions)
            .ravel()[
                np.asarray(self.dnaAFunctionalBoxIndexs_R1234, int).ravel() - 1
            ]
            .reshape(-1)
        )
        bound = ps[seven[ps[seven, 1] == 1], 0]
        return bool(np.all(np.isin(r1234, bound)))

    @property
    def isAnyHelicaseBound(self):
        return bool(
            np.any(
                self.chromosome.complexBoundSites.find()[1].reshape(-1)
                == self.enzymeGlobalIndexs[self.enzymeIndexs_helicase - 1]
            )
        )

    @property
    def isAnyPolymeraseBound(self):
        _, vals = self.chromosome.complexBoundSites.find()
        vals = vals.reshape(-1)
        ix = (
            np.asarray(
                [
                    self.enzymeIndexs_2coreBetaClampGammaComplexPrimase,
                    self.enzymeIndexs_coreBetaClampGammaComplex,
                    self.enzymeIndexs_coreBetaClampPrimase,
                ],
                int,
            )
            - 1
        )
        return bool(
            np.any(np.isin(vals, np.asarray(self.enzymeGlobalIndexs)[ix]))
        )

    @property
    def leadingStrandElongating(self):
        return (np.asarray(self.helicasePosition) != 0) & (
            np.asarray(self.leadingPolymerasePosition) != 0
        )

    @property
    def laggingStrandElongating(self):
        return np.asarray(self.laggingPolymerasePosition) != 0

    def _region_starts_ends(self, strand):
        c = self.chromosome
        coords, vals = c.polymerizedRegions.find()
        m = coords[coords[:, 1] == strand]
        return (
            m[:, 0],
            vals.reshape(-1)[np.flatnonzero(coords[:, 1] == strand)],
        )

    @property
    def leadingStrandPolymerized(self):
        c = self.chromosome
        out = np.zeros(2, dtype=bool)
        for j, s in enumerate((2, 4)):
            p, l = self._region_starts_ends(s)
            if j == 0:
                a = (p <= self.terCPosition + 1) & (p + l - 1 == c.sequenceLen)
            else:
                b = (p <= self.terCPosition + 1) & (p + l - 1 == c.sequenceLen)
            if j == 0:
                one = a
            else:
                two = b
        out[0] = np.any(one) and np.any(two)
        out[1] = np.all(
            c.polymerizedRegions.get_coords([[1, 1], [1, 3]]).reshape(-1)
            >= self.terCPosition
        )
        return out

    @property
    def laggingStrandPolymerized(self):
        c = self.chromosome
        out = np.zeros(2, dtype=bool)
        for j, s in enumerate((1, 3)):
            p, l = self._region_starts_ends(s)
            hit = (p <= self.terCPosition + 1) & (p + l - 1 == c.sequenceLen)
            if j == 0:
                one = hit
            else:
                two = hit
        out[0] = np.any(one) and np.any(two)
        out[1] = np.all(
            c.polymerizedRegions.get_coords([[1, 2], [1, 4]]).reshape(-1)
            >= self.terCPosition
        )
        return out

    @property
    def strandPolymerized(self):
        return self.leadingStrandPolymerized & self.laggingStrandPolymerized

    @property
    def numLigations(self):
        f = np.asarray(self.okazakiFragmentIndex)
        prog = np.asarray(self.okazakiFragmentProgress)
        pos = [[], []]
        for i in range(2):
            locations = np.asarray(self.primaseBindingLocations[i]).reshape(-1)
            if f[i] > 0:
                stop = max(0, f[i] - 2)
                pos[i].extend(
                    (locations[:stop] - (1 if i == 0 else 0)).tolist()
                )
            if f[i] > 1:
                pos[i].append(0)
            if f[i] == len(locations) and prog[i] > 0:
                pos[i].append(self.terCPosition)
        sp = self.strandPolymerized
        lead = np.asarray(self.leadingPosition)
        if sp[0]:
            pos[0] = [0] + (
                np.asarray(self.primaseBindingLocations[0]).reshape(-1) - 1
            ).tolist()
        if sp[1] and (sp[0] or np.any(lead)):
            pos[1] = [0] + np.asarray(self.primaseBindingLocations[1]).reshape(
                -1
            ).tolist()
        return np.asarray(
            [
                (
                    len(pos[0])
                    - np.count_nonzero(
                        self.chromosome.strandBreaks.get_coords(
                            np.column_stack((pos[0], np.full(len(pos[0]), 3)))
                        )
                    )
                    if pos[0]
                    else 0
                ),
                (
                    len(pos[1])
                    - np.count_nonzero(
                        self.chromosome.strandBreaks.get_coords(
                            np.column_stack((pos[1], np.full(len(pos[1]), 2)))
                        )
                    )
                    if pos[1]
                    else 0
                ),
            ]
        )

    @property
    def strandLigated(self):
        return self.numLigations == np.asarray(
            [
                len(self.primaseBindingLocations[0]) + 1,
                len(self.primaseBindingLocations[1]) + 1,
            ]
        )

    @property
    def strandDuplicated(self):
        return (
            self.strandPolymerized
            & self.strandLigated
            & (np.asarray(self.helicasePosition) == 0)
            & (np.asarray(self.leadingPosition) == 0)
            & (np.asarray(self.laggingPosition) == 0)
            & (np.asarray(self.laggingBackupBetaClampPosition) == 0)
        )

    def _bound_positions(self, enzyme_indices):
        coords, vals = self.chromosome.complexBoundSites.find()
        ix = np.asarray(enzyme_indices, int) - 1
        glob = np.asarray(self.enzymeGlobalIndexs)[ix].reshape(-1)
        return coords[np.isin(vals.reshape(-1), glob)]

    @property
    def helicasePosition(self):
        p = self._bound_positions([self.enzymeIndexs_helicase])
        out = np.zeros(2, dtype=int)
        for i, s in enumerate(self.leadingStrandIndexs):
            rows = p[p[:, 1] == s]
            if len(rows) > 1:
                raise ValueError("More than 1 active replisome")
            if len(rows):
                out[i] = rows[0, 0]
        return out

    @property
    def leadingPolymerasePosition(self):
        p = self._bound_positions(
            [
                self.enzymeIndexs_coreBetaClampGammaComplex,
                self.enzymeIndexs_2coreBetaClampGammaComplexPrimase,
            ]
        )
        out = np.zeros(2, dtype=int)
        for i, s in enumerate(self.leadingStrandIndexs):
            rows = p[p[:, 1] == s]
            if len(rows) > 1:
                raise ValueError("More than 1 active replisome")
            if len(rows):
                out[i] = rows[0, 0]
        return out

    @property
    def laggingPolymerasePosition(self):
        p = self._bound_positions([self.enzymeIndexs_coreBetaClampPrimase])
        out = np.zeros(2, dtype=int)
        for i, s in enumerate(self.laggingStrandIndexs):
            rows = p[p[:, 1] == s, 0]
            if len(rows):
                out[i] = np.max(rows) if i == 0 else np.min(rows)
            if len(rows) > 2:
                raise ValueError("More than 1 active replisome")
        return out

    def _poly_positions(self, leading):
        out = np.asarray(
            self.leadingPolymerasePosition
            if leading
            else self.laggingPolymerasePosition
        ).copy()
        hol = self.enzymeDNAFootprints[
            self.enzymeIndexs_coreBetaClampGammaComplex - 1
        ]
        cor = self.enzymeDNAFootprints5Prime[self.enzymeIndexs_core - 1]
        if leading:
            out[0] += cor if out[0] else 0
            out[1] += hol - cor - 1 if out[1] else 0
        else:
            out[0] += hol - cor - 1 if out[0] else 0
            out[1] += cor if out[1] else 0
        nz = out != 0
        out[nz] = (out[nz] - 1) % int(
            np.asarray(self.chromosome.sequenceLen).item()
        ) + 1
        return out

    @property
    def leadingPosition(self):
        return self._poly_positions(True)

    @property
    def laggingPosition(self):
        return self._poly_positions(False)

    @property
    def laggingBackupBetaClampPosition(self):
        p = self._bound_positions([self.enzymeIndexs_betaClamp])
        out = np.zeros(2, dtype=int)
        for i, s in enumerate(self.laggingStrandIndexs):
            rows = p[p[:, 1] == s, 0]
            if len(rows):
                out[i] = np.max(rows) if i == 0 else np.min(rows)
            if len(rows) > 1:
                raise ValueError("More than 1 active replisome")
        return out

    def _adjust_lagging(self):
        p = np.asarray(self.laggingPosition).copy()
        n = int(np.asarray(self.chromosome.sequenceLen).item())
        if p[0] and p[0] < self.terCPosition / 2:
            p[0] += n
        if p[1] and p[1] > 3 * self.terCPosition / 2:
            p[1] -= n
        return p

    @property
    def okazakiFragmentIndex(self):
        c = self.chromosome
        lag = np.asarray(self.laggingPosition)
        adj = self._adjust_lagging()
        loc = [np.asarray(x).reshape(-1) for x in self.primaseBindingLocations]
        out = np.zeros(2, dtype=int)
        # Keep the chromosome's source query and argument order, including one-based coordinates.
        a = (
            bool(c.isRegionPolymerized([adj[0] - 1, 3], 1, False)[0][0])
            and adj[0] > self.terCPosition + 1
            if lag[0]
            else False
        )
        b = (
            bool(c.isRegionPolymerized([adj[1] + 1, 2], 1, False)[0][0])
            and adj[1] < self.terCPosition
            if lag[1]
            else False
        )
        q1 = loc[0] <= adj[0] - a
        q2 = loc[1] >= adj[1] + b
        ids = np.flatnonzero(q1)
        jds = np.flatnonzero(q2)
        if lag[0] and len(ids):
            out[0] = ids[0] + 1
        if lag[1] and len(jds):
            out[1] = jds[0] + 1
        return out

    @property
    def okazakiFragmentPosition(self):
        f = self.okazakiFragmentIndex
        out = np.zeros(2, dtype=int)
        for i in range(2):
            if f[i]:
                out[i] = np.asarray(self.primaseBindingLocations[i]).reshape(
                    -1
                )[f[i] - 1]
        return out

    @property
    def okazakiFragmentLength(self):
        f = self.okazakiFragmentIndex
        starts = np.zeros(2, dtype=int)
        ends = np.zeros(2, dtype=int)
        for i in range(2):
            loc = np.asarray(self.primaseBindingLocations[i]).reshape(-1)
            if f[i]:
                starts[i] = loc[f[i] - 1]
            if f[i] > 1:
                ends[i] = loc[f[i] - 2] + (-1 if i == 0 else 1)
            elif f[i] == 1:
                ends[i] = (
                    int(np.asarray(self.chromosome.sequenceLen).item())
                    if i == 0
                    else 1
                )
        out = np.abs(ends - starts) + 1
        out[f == 0] = 0
        return out

    @property
    def okazakiFragmentProgress(self):
        lag = np.asarray(self.laggingPosition)
        adj = self._adjust_lagging()
        f = self.okazakiFragmentIndex
        out = np.zeros(2, dtype=int)
        for i in range(2):
            if lag[i] and f[i]:
                loc = np.asarray(self.primaseBindingLocations[i]).reshape(-1)[
                    f[i] - 1
                ]
                out[i] = adj[i] - loc if i == 0 else loc - adj[i]
        return out

    def _bound_ssbs(self, leading):
        from ..circular_sparsemat import CircularSparseMat

        hp = np.asarray(self.helicasePosition)
        pp = np.asarray(self.leadingPolymerasePosition)
        p = self._bound_positions([self.enzymeIndexs_ssb8mer])
        coords = []
        f = np.asarray(self.okazakiFragmentIndex) if not leading else None
        starts = None
        if not leading:
            starts = np.array(
                [
                    (
                        int(np.asarray(self.chromosome.sequenceLen).item())
                        if not f[0]
                        else np.asarray(
                            self.primaseBindingLocations[0]
                        ).reshape(-1)[f[0] - 1]
                    ),
                    (
                        1
                        if not f[1]
                        else np.asarray(
                            self.primaseBindingLocations[1]
                        ).reshape(-1)[f[1] - 1]
                    ),
                ]
            )
        for pos, strand in p:
            if leading:
                if pp[0] and strand == 1 and pos < pp[0] and pos > hp[0]:
                    coords.append((pos, 1))
                if pp[1] and strand == 4 and pos > pp[1] and pos < hp[1]:
                    coords.append((pos, 2))
            else:
                if strand == 4 and pos < starts[0] and pos > hp[0]:
                    coords.append((pos, 1))
                if strand == 1 and pos > starts[1] and pos < hp[1]:
                    coords.append((pos, 2))
        return CircularSparseMat(
            np.asarray(coords, dtype=int).reshape(-1, 2),
            np.ones(len(coords)),
            [int(np.asarray(self.chromosome.sequenceLen).item()), 2],
            1,
        )

    @property
    def leadingStrandBoundSSBs(self):
        return self._bound_ssbs(True)

    @property
    def laggingStrandBoundSSBs(self):
        return self._bound_ssbs(False)

    @property
    def numLeadingTemplateBoundSSBs(self):
        return np.asarray(
            self.leadingStrandBoundSSBs.full().sum(axis=0)
        ).reshape(-1)

    @property
    def numLaggingTemplateBoundSSBs(self):
        return np.asarray(
            self.laggingStrandBoundSSBs.full().sum(axis=0)
        ).reshape(-1)

    @property
    def areLaggingStrandSSBSitesBound(self):
        n = np.asarray(self.numLaggingTemplateBoundSSBs)
        ssb = self.enzymeDNAFootprints[self.enzymeIndexs_ssb8mer - 1]
        spacing = self.ssbComplexSpacing
        hel = np.asarray(self.helicasePosition)
        f = self.okazakiFragmentIndex
        lead = self.enzymeDNAFootprints[
            self.enzymeIndexs_coreBetaClampGammaComplex - 1
        ]
        lag = self.enzymeDNAFootprints[
            self.enzymeIndexs_coreBetaClampPrimase - 1
        ]
        starts = np.array(
            [
                (
                    int(np.asarray(self.chromosome.sequenceLen).item())
                    if not f[0]
                    else np.asarray(self.primaseBindingLocations[0]).reshape(
                        -1
                    )[f[0] - 1]
                ),
                (
                    1
                    if not f[1]
                    else np.asarray(self.primaseBindingLocations[1]).reshape(
                        -1
                    )[f[1] - 1]
                ),
            ]
        )
        out = np.zeros(2, dtype=bool)
        if hel[0]:
            out[0] = n[0] >= np.floor(
                (starts[0] - hel[0] - lead - lag) / (ssb + spacing) - 2
            )
        if hel[1]:
            out[1] = n[1] >= np.floor(
                (hel[1] - starts[1] - lead - lag) / (ssb + spacing) - 2
            )
        return out
