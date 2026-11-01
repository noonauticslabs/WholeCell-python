"""Runtime replication updates translated from Replication.m (616-1277).

The process stores MATLAB-facing local indices and chromosome coordinates as
one-based values; conversion is left to state APIs.
"""

from __future__ import annotations

import numpy as np


class ReplicationUpdates:
    """Runtime update methods mixed into the Replication process."""

    def initiateReplication(self):
        c = self.chromosome
        chrLen = int(np.asarray(c.sequenceLen).item())
        holFtpt = np.asarray(self.enzymeDNAFootprints).ravel()[
            self.enzymeIndexs_2coreBetaClampGammaComplexPrimase - 1
        ]
        helFtpt3 = np.asarray(self.enzymeDNAFootprints3Prime).ravel()[
            self.enzymeIndexs_helicase - 1
        ]
        helFtpt5 = np.asarray(self.enzymeDNAFootprints5Prime).ravel()[
            self.enzymeIndexs_helicase - 1
        ]
        corFtpt5 = np.asarray(self.enzymeDNAFootprints5Prime).ravel()[
            self.enzymeIndexs_core - 1
        ]
        spacing = helFtpt3 + corFtpt5 + 1
        if not self.isDnaAORIComplexAssembled or np.any(
            self.leadingPolymerasePosition
        ):
            return
        if float(c.polymerizedRegions.collapse(None)) > 2 * chrLen:
            return
        helPos = np.asarray(self.helicasePosition)
        maxSteps = float(
            np.floor(
                np.min(
                    self.substrates[
                        [
                            self.substrateIndexs_atp - 1,
                            self.substrateIndexs_water - 1,
                        ]
                    ]
                )
                / 2
            )
        )
        if np.all(helPos):
            old = (-(1 - helFtpt3 - helPos[1]) - 1) % chrLen + 1
            length = min(spacing - old, maxSteps)
            cost = 2 * length
            if length <= 0:
                return
            self.releaseProteinFromSites(
                np.column_stack((helPos, self.leadingStrandIndexs)), False
            )
        elif not np.any(helPos):
            length = min(spacing, maxSteps - 1)
            old = 0
            if length < 0 or self.enzymes[self.enzymeIndexs_helicase - 1] < 2:
                return
            cost = 2 * (1 + length)
        else:
            raise ValueError("All or no helicases should be bound")
        c.setRegionUnwound(chrLen - old, -length)
        c.setRegionUnwound(1 + old, length)
        if not np.all(
            self.bindProteinToChromosome(
                np.array(
                    [
                        [chrLen - helFtpt5 - old, self.leadingStrandIndexs[0]],
                        [1 - helFtpt3 + old, self.leadingStrandIndexs[1]],
                    ]
                ),
                self.enzymeIndexs_helicase,
                None,
                None,
                True,
                False,
                [-length - 1, length + 1],
                True,
                [],
            )[0]
        ):
            raise RuntimeError("Both helicase must bind")
        for idx, delta in (
            (self.substrateIndexs_atp, -cost),
            (self.substrateIndexs_water, -cost),
            (self.substrateIndexs_adp, cost),
            (self.substrateIndexs_phosphate, cost),
            (self.substrateIndexs_hydrogen, cost),
        ):
            self.substrates[idx - 1] += delta
        comp = (
            2
            * self.enzymeComposition[
                :, self.enzymeIndexs_2coreBetaClampGammaComplexPrimase - 1
            ]
        ).reshape(self.enzymes.shape)
        if old + length == spacing and np.all(self.enzymes >= comp):
            self.enzymes -= comp
            self.enzymes[
                self.enzymeIndexs_2coreBetaClampGammaComplexPrimase - 1
            ] += 2
            p = np.array(
                [
                    [chrLen - corFtpt5, self.leadingStrandIndexs[0]],
                    [1 + corFtpt5 - holFtpt + 1, self.leadingStrandIndexs[1]],
                ]
            )
            if not np.all(
                self.bindProteinToChromosome(
                    p,
                    self.enzymeIndexs_2coreBetaClampGammaComplexPrimase,
                    None,
                    None,
                    True,
                    False,
                    1,
                    True,
                    [],
                )[0]
            ):
                raise RuntimeError("Both polymerases must bind")

    def unwindAndPolymerizeDNA(self):
        """Advance helicases and DNA polymerases, following Replication.m."""
        from ..polymerize import polymerize

        c = self.chromosome
        if not self.isAnyHelicaseBound or not np.all(
            self.leadingStrandElongating
        ):
            return
        helIdx = int(self.enzymeIndexs_helicase)
        leadIdx = np.asarray(
            [
                self.enzymeIndexs_2coreBetaClampGammaComplexPrimase,
                self.enzymeIndexs_coreBetaClampGammaComplex,
            ],
            dtype=int,
        )
        leadGlobal = np.asarray(self.enzymeGlobalIndexs)[leadIdx - 1]
        lagGlobal = np.asarray(self.enzymeGlobalIndexs)[
            int(self.enzymeIndexs_coreBetaClampPrimase) - 1
        ]
        hf = float(np.asarray(self.enzymeDNAFootprints).ravel()[helIdx - 1])
        h3 = float(
            np.asarray(self.enzymeDNAFootprints3Prime).ravel()[helIdx - 1]
        )
        h5 = float(
            np.asarray(self.enzymeDNAFootprints5Prime).ravel()[helIdx - 1]
        )
        hol = float(
            np.asarray(self.enzymeDNAFootprints).ravel()[
                self.enzymeIndexs_coreBetaClampGammaComplex - 1
            ]
        )
        cor = float(
            np.asarray(self.enzymeDNAFootprints).ravel()[
                self.enzymeIndexs_core - 1
            ]
        )
        c3 = float(
            np.asarray(self.enzymeDNAFootprints3Prime).ravel()[
                self.enzymeIndexs_core - 1
            ]
        )
        c5 = float(
            np.asarray(self.enzymeDNAFootprints5Prime).ravel()[
                self.enzymeIndexs_core - 1
            ]
        )
        bf = float(
            np.asarray(self.enzymeDNAFootprints).ravel()[
                self.enzymeIndexs_betaClamp - 1
            ]
        )
        L = int(c.sequenceLen)
        leadStr = np.asarray(self.leadingStrandIndexs, dtype=int).ravel()
        lagStr = np.asarray(self.laggingStrandIndexs, dtype=int).ravel()
        first = np.asarray(
            [
                self.primaseBindingLocations[0][0] - c3 - bf,
                self.primaseBindingLocations[1][0] + c3 + 1,
            ],
            dtype=int,
        )
        lp = (
            np.asarray(self.leadingPolymerasePosition, dtype=int)
            .ravel()
            .copy()
        )
        kp = (
            np.asarray(self.laggingPolymerasePosition, dtype=int)
            .ravel()
            .copy()
        )
        tfs = (kp == 0) & (
            np.asarray(self.laggingBackupBetaClampPosition).ravel() == first
        )
        # A first lagging clamp can be converted only when its leading complex is bound.
        for i in range(2):
            if tfs[i]:
                sites = c.complexBoundSites
                vals = (
                    np.asarray(sites.get_coords([[lp[i], leadStr[i]]])).ravel()
                    if hasattr(sites, "get_coords")
                    else np.asarray([])
                )
                if vals.size and vals[0] != leadGlobal[0]:
                    tfs[i] = False
        kp[tfs] = first[tfs]
        if tfs[1]:
            kp[1] -= cor
        n = int(np.count_nonzero(tfs))
        if n:
            self.releaseProteinFromSites(
                np.column_stack((lp[tfs], leadStr[tfs])), False
            )
            self.releaseProteinFromSites(
                np.column_stack((first[tfs], lagStr[tfs])), False
            )
            for idx, d in (
                (self.enzymeIndexs_2coreBetaClampGammaComplexPrimase, -n),
                (self.enzymeIndexs_coreBetaClampGammaComplex, n),
                (self.enzymeIndexs_core, n),
                (self.enzymeIndexs_primase, n),
                (self.enzymeIndexs_betaClamp, -n),
                (self.enzymeIndexs_betaClampMonomer, 2 * n),
                (self.enzymeIndexs_betaClampMonomer, -2 * n),
                (self.enzymeIndexs_core, -n),
                (self.enzymeIndexs_primase, -n),
                (self.enzymeIndexs_coreBetaClampPrimase, n),
            ):
                self.enzymes[idx - 1] += d
            if not np.all(
                self.bindProteinToChromosome(
                    np.column_stack((lp[tfs], leadStr[tfs])),
                    self.enzymeIndexs_coreBetaClampGammaComplex,
                    None,
                    None,
                    True,
                    False,
                    1,
                    True,
                    [],
                )[0]
            ):
                raise RuntimeError("All polymerases should bind.")
            if not np.all(
                self.bindProteinToChromosome(
                    np.column_stack((kp[tfs], lagStr[tfs])),
                    self.enzymeIndexs_coreBetaClampPrimase,
                    None,
                    None,
                    True,
                    False,
                    1,
                    True,
                    [],
                )[0]
            ):
                raise RuntimeError("All polymerases should bind.")
        hel = np.asarray(self.helicasePosition, dtype=int).ravel()
        lp = np.asarray(self.leadingPolymerasePosition, dtype=int).ravel()
        kp = np.asarray(self.laggingPolymerasePosition, dtype=int).ravel()
        lead = np.asarray(self.leadingPosition, dtype=int).ravel()
        lag = np.asarray(self.laggingPosition, dtype=int).ravel()
        fp = np.asarray(self.okazakiFragmentPosition, dtype=int).ravel()
        pr = np.asarray(self.okazakiFragmentProgress, dtype=int).ravel()
        fl = np.asarray(self.okazakiFragmentLength, dtype=int).ravel()
        lim = np.zeros((2, 2), int)
        lim[0] = [
            self.primerLength - (L - lead[0]),
            self.primerLength - (lead[1] - 1),
        ]
        lim[1] = [
            self.primerLength - (lag[0] - fp[0]),
            self.primerLength - (fp[1] - lag[1]),
        ]
        lim[lim <= 0] = int(self.dnaPolymeraseElongationRate)
        lim[0] *= lead != 0
        lim[1] *= lag != 0
        lim[0] *= np.asarray(
            self.areLaggingStrandSSBSitesBound, dtype=int
        ).ravel()
        self.releaseProteinFromSites(np.column_stack((hel, leadStr)), False)
        _, _, _, ext = c.isRegionAccessible(
            np.column_stack((hel + np.asarray([0, hf - 1]), leadStr)),
            np.asarray([-1, 1]) * (lim[0] + hf + 1),
            [],
            self.enzymeGlobalIndexs[helIdx - 1],
            False,
            [],
            True,
            True,
        )
        lim[0] = np.maximum(0, np.abs(np.asarray(ext).ravel()) - hf - 1)
        bound = np.asarray(
            c.complexBoundSites.get_coords(np.column_stack((lp, leadStr)))
        ).ravel()
        bnd = np.array(
            [
                np.flatnonzero(np.asarray(leadGlobal).ravel() == value).item()
                for value in bound
            ],
            dtype=int,
        )
        self.releaseProteinFromSites(np.column_stack((lp, leadStr)), False)
        _, _, _, ext = c.isRegionAccessible(
            np.column_stack((lp + np.asarray([0, hol - 1]), leadStr)),
            np.asarray([-1, 1]) * (lim[0] + hol + 1),
            [],
            leadGlobal[0],
            False,
            [],
            True,
            True,
        )
        lim[0] = np.maximum(0, np.abs(np.asarray(ext).ravel()) - hol - 1)
        for i, start, length in (
            (0, kp[0], lim[1, 0] + hol + 1),
            (1, kp[1] + hol - 1, -(lim[1, 1] + hol + 1)),
        ):
            if lag[i] != 0:
                self.releaseProteinFromSites([[kp[i], lagStr[i]]], False)
                _, _, _, ex = c.isRegionAccessible(
                    [[start, lagStr[i]]],
                    length,
                    [],
                    lagGlobal,
                    False,
                    [],
                    True,
                    True,
                )
                lim[1, i] = max(
                    0, abs(int(np.asarray(ex).ravel()[0])) - hol - 1
                )
        lim[1] = np.minimum(lim[1], fl - pr)
        tmp = fp.copy()
        if fp[0] == 0:
            tmp[0] = L
        if fp[1] == 0:
            tmp[1] = 1
        lim[0, 0] *= (
            tmp[0] - hel[0] - hf
        ) < 2 * self.okazakiFragmentMeanLength
        lim[0, 1] *= (hel[1] - tmp[1]) < 2 * self.okazakiFragmentMeanLength
        ter = int(np.asarray(c.terCPosition).item())
        cross_left = (
            hel[0] + h5 >= ter + 1 and hel[0] + h5 - lim[0, 0] < ter + 1
        )
        cross_right = (
            hel[1] + h5 - 1 <= ter and hel[1] + h5 - 1 + lim[0, 1] > ter
        )
        lkpos = int(min(ter + 1, hel[1] + h5 - 1))
        lkval = float(c.linkingNumbers.get_coords([[lkpos, 1]]).item())
        if cross_left and cross_right and abs(lkval) > 1e-6:
            lim[0, int(np.ceil(2 * self.randStream.rand().item())) - 1] = 0
        elif (
            cross_left
            and not cross_right
            and abs(lkval) > 1e-6
            and not c.isRegionPolymerized([[ter, 2]], 1, False)[0][0]
        ):
            lim[0, 0] = 0
        elif (
            cross_right
            and not cross_left
            and abs(lkval) > 1e-6
            and not c.isRegionPolymerized([[ter + 1, 2]], 1, False)[0][0]
        ):
            lim[0, 1] = 0
        pos, vals = c.complexBoundSites.find()
        pos = np.asarray(pos, dtype=int).reshape(-1, 2)
        vals = np.asarray(vals).ravel()
        rna = np.isin(
            vals, np.asarray(self.complex.rnaPolymeraseIndexs).ravel()
        )
        if np.any(rna):
            pos = pos[rna]
            rnaft = c.getDNAFootprint(
                None, np.asarray(self.complex.rnaPolymeraseIndexs).ravel()[0]
            )[0]
            tmp = np.zeros((2, 2), int)
            queries = [
                (
                    (pos[:, 0] < hel[0]) & (pos[:, 1] % 2 == 1),
                    hel[0] - (pos[:, 0] + rnaft),
                ),
                (
                    (pos[:, 0] > hel[1]) & (pos[:, 1] % 2 == 0),
                    pos[:, 0] - (hel[1] + hf),
                ),
                (
                    (pos[:, 0] > kp[0]) & (pos[:, 1] % 2 == 0),
                    pos[:, 0] - (kp[0] + hol),
                ),
                (
                    (pos[:, 0] < kp[1]) & (pos[:, 1] % 2 == 1),
                    kp[1] - (pos[:, 0] + rnaft),
                ),
            ]
            for target, (mask, distances) in enumerate(queries):
                if np.any(mask):
                    tmp.ravel()[target] = np.min(distances[mask])
            stalled = (
                np.asarray(
                    self.randStream.random(
                        "poisson",
                        1 / self.rnaPolymeraseCollisionMeanDwellTime,
                        (2, 2),
                    )
                )
                == 0
            )
            lim[stalled] = np.minimum(
                lim[stalled], np.maximum(0, tmp[stalled])
            )
        pol = lim.copy()
        pol[0, 0] = min(lim[0, 0], max(0, lead[0] - self.terCPosition))
        pol[0, 1] = min(lim[0, 1], max(0, self.terCPosition - lead[1] + 1))
        over = c5 + bf + c5 + h3
        lim[0, 0] = min(
            lim[0, 0], max(0, lead[0] - (self.terCPosition - over))
        )
        lim[0, 1] = min(
            lim[0, 1], max(0, self.terCPosition + 1 + over - lead[1])
        )
        seq = np.full((4, int(np.max(pol, initial=0))), " ", dtype="<U1")
        for i, (p, d, s) in enumerate(
            (
                (lead[0], -1, 2),
                (lead[1], 1, 3),
                (lag[0], 1, 3),
                (lag[1], -1, 2),
            )
        ):
            if pol.ravel()[i]:
                # MATLAB permits a column selection on the RHS of a row-vector
                # assignment. Flatten these characters explicitly at that boundary.
                seq[i, : pol.ravel()[i]] = np.asarray(
                    c.sequence[
                        np.asarray([p + d * k for k in range(pol.ravel()[i])]),
                        s,
                    ]
                ).ravel(order="F")
        polflat = (
            polymerize(
                seq,
                self.substrates[self.substrateIndexs_dntp - 1],
                "ACGT",
                " ",
                0,
                0,
                self.randStream,
            )[0]
            .ravel()
            .reshape(2, 2)
        )
        previous_pol = pol.copy()
        pol = np.minimum(pol, polflat)
        lim[pol != previous_pol] = pol[pol != previous_pol]
        unw = np.minimum(
            pol[0],
            np.maximum(
                0,
                [
                    hel[0] + h5 - self.terCPosition,
                    self.terCPosition + 1 - (hel[1] + hf - h5 - 1),
                ],
            ),
        )
        total = int(np.sum(unw))
        energy = float(
            np.min(
                self.substrates[
                    [
                        self.substrateIndexs_atp - 1,
                        self.substrateIndexs_water - 1,
                    ]
                ]
            )
        )
        unlim = (
            np.floor(unw / total * energy).astype(int)
            if total
            else np.zeros(2, int)
        )
        unlim = np.minimum(unw, unlim)
        pol[0, unw != unlim] = unlim[unw != unlim]
        lim[0, unw != unlim] = unlim[unw != unlim]
        if not np.all(
            self.bindProteinToChromosome(
                np.column_stack((hel, leadStr)),
                helIdx,
                None,
                None,
                True,
                False,
                np.asarray([-1, 1]) * (lim[0] + 1),
                True,
                [],
            )[0]
        ):
            raise RuntimeError("helicases must bind")
        c.setRegionUnwound(hel[0] + h5, -unlim[0])
        c.setRegionUnwound(hel[1] + hf - h5 - 1, unlim[1])
        n = int(np.sum(unlim))
        for idx, d in (
            (self.substrateIndexs_atp, -n),
            (self.substrateIndexs_water, -n),
            (self.substrateIndexs_adp, n),
            (self.substrateIndexs_hydrogen, n),
            (self.substrateIndexs_phosphate, n),
        ):
            self.substrates[idx - 1] += d
        for i in range(2):
            if not self.bindProteinToChromosome(
                [[lp[i], leadStr[i]]],
                leadIdx[bnd[i]],
                None,
                None,
                True,
                False,
                (-1 if i == 0 else 1) * (lim[0, i] + 1),
                True,
                [],
            )[0][0]:
                raise RuntimeError("polymerases must bind")
            if (
                lag[i] != 0
                and not self.bindProteinToChromosome(
                    [[kp[i], lagStr[i]]],
                    self.enzymeIndexs_coreBetaClampPrimase,
                    None,
                    None,
                    True,
                    False,
                    (1 if i == 0 else -1) * (lim[1, i] + 1),
                    True,
                    [],
                )[0][0]
            ):
                raise RuntimeError("polymerases must bind")
        c.setRegionPolymerized(
            np.column_stack((lead, [1, 2])), np.asarray([-1, 1]) * pol[0]
        )
        c.setRegionPolymerized(
            np.column_stack((lag, [2, 1])), np.asarray([1, -1]) * pol[1]
        )
        used = sum(
            (
                c.sequence.subsequenceBaseCounts(
                    [p + d * k for k in range(int(n))], s
                )
                for p, d, s, n in (
                    (lead[0], -1, 2, pol[0, 0]),
                    (lead[1], 1, 3, pol[0, 1]),
                    (lag[0], 1, 3, pol[1, 0]),
                    (lag[1], -1, 2, pol[1, 1]),
                )
            ),
            np.zeros_like(self.substrates[self.substrateIndexs_dntp - 1]),
        )
        self.substrates[self.substrateIndexs_dntp - 1] -= used
        self.substrates[self.substrateIndexs_diphosphate - 1] += np.sum(used)

    def dissociateFreeSSBComplexes(self):
        i = self.enzymeIndexs_ssb4mer - 1
        j = self.enzymeIndexs_ssb8mer - 1
        self.enzymes[i] += 2 * self.enzymes[j]
        self.enzymes[j] = 0

    def freeAndBindSSBs(self):
        positions = self.releaseProteinFromChromosome(
            self.enzymeIndexs_ssb8mer, self.ssbDissociationRate, None, None
        )
        self.enzymes[self.enzymeIndexs_ssb4mer - 1] += 2 * len(positions)
        self.enzymes[self.enzymeIndexs_ssb8mer - 1] -= len(positions)
        available = self.enzymes[self.enzymeIndexs_ssb4mer - 1] / 2
        if available < 1:
            return
        c = self.chromosome
        globalidx = self.enzymeGlobalIndexs[self.enzymeIndexs_ssb8mer - 1]
        pos, lens = c.getAccessibleRegions(None, globalidx)
        if len(pos) == 0:
            return
        ftpt = int(
            np.asarray(self.enzymeDNAFootprints).ravel()[
                self.enzymeIndexs_ssb8mer - 1
            ]
        )
        spacing = int(self.ssbComplexSpacing)
        ssb, ssb_values = c.complexBoundSites.find()
        ssb = ssb[
            np.asarray(ssb_values).ravel() == np.asarray(globalidx).item()
        ]
        pos, lens = c.excludeRegions(
            pos,
            lens,
            (
                np.column_stack((ssb[:, 0] - spacing, ssb[:, 1]))
                if len(ssb)
                else np.empty((0, 2))
            ),
            ftpt + 2 * spacing,
        )
        if len(pos) == 0:
            return
        split = np.flatnonzero(
            (pos[:, 0] <= self.terCPosition)
            & (pos[:, 0] + lens - 1 > self.terCPosition)
        )
        if len(split):
            pos = np.vstack(
                (
                    pos,
                    np.column_stack(
                        (
                            np.full(len(split), self.terCPosition + 1),
                            pos[split, 1],
                        )
                    ),
                )
            )
            lens = np.r_[
                lens, lens[split] - (self.terCPosition - pos[split, 0] + 1)
            ]
            lens[split] = self.terCPosition - pos[split, 0] + 1
        far = pos[:, 0] > self.terCPosition
        pos[far, 0] += lens[far] - ftpt
        lens[far] = -lens[far]
        one = np.abs(lens) == ftpt
        many = np.abs(lens) > ftpt
        pos = np.vstack(
            (
                c.splitRegions(pos[many], lens[many], ftpt + spacing)[0],
                pos[one],
            )
        )
        if not len(pos):
            return
        n = int(min(len(pos), np.floor(available)))
        self.enzymes[self.enzymeIndexs_ssb4mer - 1] -= 2 * n
        self.enzymes[self.enzymeIndexs_ssb8mer - 1] += n
        result = self.bindProteinToChromosome(
            pos,
            self.enzymeIndexs_ssb8mer,
            n,
            None,
            None,
            False,
            1,
            False,
            None,
        )
        if n != result[3]:
            raise RuntimeError("SSBs must bind")

    def initiateOkazakiFragment(self):
        if not self.isAnyHelicaseBound or not np.all(
            self.leadingStrandElongating
        ):
            return
        h3 = np.asarray(self.enzymeDNAFootprints3Prime).ravel()[
            self.enzymeIndexs_helicase - 1
        ]
        h5 = np.asarray(self.enzymeDNAFootprints5Prime).ravel()[
            self.enzymeIndexs_helicase - 1
        ]
        c3 = np.asarray(self.enzymeDNAFootprints3Prime).ravel()[
            self.enzymeIndexs_core - 1
        ]
        b = np.asarray(self.enzymeDNAFootprints).ravel()[
            self.enzymeIndexs_betaClamp - 1
        ]
        fi = np.asarray(self.okazakiFragmentIndex)
        pr = np.asarray(self.okazakiFragmentProgress)
        hp = np.asarray(self.helicasePosition)
        bp = np.asarray(self.laggingBackupBetaClampPosition)
        done = np.asarray(self.strandPolymerized)
        p = []
        if (
            fi[0] < len(self.primaseBindingLocations[0])
            and (fi[0] == 0 or pr[0] >= self.laggingBackupClampReloadingLength)
            and hp[0] + h5 + 1
            < self.primaseBindingLocations[0][fi[0]] - c3 - b
            and bp[0] != self.primaseBindingLocations[0][fi[0]] - c3 - b
            and not done[0]
        ):
            p.append(
                [
                    self.primaseBindingLocations[0][fi[0]] - c3 - b,
                    self.laggingStrandIndexs[0],
                ]
            )
        if (
            fi[1] < len(self.primaseBindingLocations[1])
            and (fi[1] == 0 or pr[1] >= self.laggingBackupClampReloadingLength)
            and hp[1] + h3
            > self.primaseBindingLocations[1][fi[1]] + c3 + 1 + b
            and bp[1] != self.primaseBindingLocations[1][fi[1]] + c3 + 1
            and not done[1]
        ):
            p.append(
                [
                    self.primaseBindingLocations[1][fi[1]] + c3 + 1,
                    self.laggingStrandIndexs[1],
                ]
            )
        p = np.asarray(p, int).reshape((-1, 2))
        n = int(
            np.floor(
                min(
                    len(p),
                    self.substrates[self.substrateIndexs_atp - 1],
                    self.substrates[self.substrateIndexs_water - 1],
                    self.enzymes[self.enzymeIndexs_betaClampMonomer - 1] / 2,
                )
            )
        )
        out = self.bindProteinToChromosome(
            p, self.enzymeIndexs_betaClamp, n, None, True, False, 1, False, []
        )
        if n != out[3]:
            raise RuntimeError("Beta clamps must bind")
        self.enzymes[self.enzymeIndexs_betaClampMonomer - 1] -= 2 * n
        self.enzymes[self.enzymeIndexs_betaClamp - 1] += n
        for idx, d in (
            (self.substrateIndexs_atp, -n),
            (self.substrateIndexs_water, -n),
            (self.substrateIndexs_adp, n),
            (self.substrateIndexs_phosphate, n),
            (self.substrateIndexs_hydrogen, n),
        ):
            self.substrates[idx - 1] += d

    def terminateOkazakiFragment(self):
        if not self.isAnyHelicaseBound or not np.all(
            self.leadingStrandElongating
        ):
            return
        c = self.chromosome
        L = c.sequenceLen
        fi = np.asarray(self.okazakiFragmentIndex)
        pr = np.asarray(self.okazakiFragmentProgress)
        fl = np.asarray(self.okazakiFragmentLength)
        hp = np.asarray(self.helicasePosition)
        hf = np.asarray(self.enzymeDNAFootprints).ravel()[
            self.enzymeIndexs_helicase - 1
        ]
        hol = np.asarray(self.enzymeDNAFootprints).ravel()[
            self.enzymeIndexs_coreBetaClampPrimase - 1
        ]
        cf = np.asarray(self.enzymeDNAFootprints).ravel()[
            self.enzymeIndexs_core - 1
        ]
        c3 = np.asarray(self.enzymeDNAFootprints3Prime).ravel()[
            self.enzymeIndexs_core - 1
        ]
        c5 = np.asarray(self.enzymeDNAFootprints5Prime).ravel()[
            self.enzymeIndexs_core - 1
        ]
        ssb = np.asarray(self.areLaggingStrandSSBSitesBound)
        bp = np.asarray(self.laggingBackupBetaClampPosition)
        lp = np.asarray(self.leadingPolymerasePosition)
        kp = np.asarray(self.laggingPolymerasePosition)
        release = []
        clamps = []
        bind = []
        term = []
        for i in range(2):
            loc = self.primaseBindingLocations[i]
            ok = fi[i] > 0 and pr[i] == fl[i] and ssb[i]
            if i == 0:
                ok = ok and (
                    fi[i] == len(loc)
                    or (
                        (
                            fi[i] == len(loc) - 1
                            or loc[fi[i]] - (hp[i] + hf)
                            > self.startingOkazakiLoopLength
                        )
                        and bp[i] == loc[fi[i]] - (hol - c5) + 1
                    )
                )
            else:
                ok = ok and (
                    fi[i] == len(loc)
                    or (
                        (
                            fi[i] == len(loc) - 1
                            or hp[i] - loc[fi[i]]
                            > self.startingOkazakiLoopLength
                        )
                        and bp[i] == loc[fi[i]] + c3 + 1
                    )
                )
            if not ok:
                continue
            release.append([kp[i], self.laggingStrandIndexs[i]])
            if fi[i] < len(loc):
                clamps.append([bp[i], self.laggingStrandIndexs[i]])
                bind.append(
                    [bp[i] - (cf if i else 0), self.laggingStrandIndexs[i]]
                )
            else:
                term.append([lp[i], self.leadingStrandIndexs[i]])
            # CircularSparseMat coordinates are MATLAB one-based, unlike loc.
            if fi[i] == 1:
                c.strandBreaks[L, 3 if i == 0 else 2] = 1
            else:
                c.strandBreaks[
                    loc[fi[i] - 2] - (1 if i == 0 else 0), 3 if i == 0 else 2
                ] = 1
        release = np.asarray(release, int).reshape((-1, 2))
        clamps = np.asarray(clamps, int).reshape((-1, 2))
        bind = np.asarray(bind, int).reshape((-1, 2))
        term = np.asarray(term, int).reshape((-1, 2))
        self.releaseProteinFromSites(release, False)
        n = len(release)
        for idx, d in (
            (self.enzymeIndexs_coreBetaClampPrimase, -n),
            (self.enzymeIndexs_primase, n),
            (self.enzymeIndexs_core, n),
            (self.enzymeIndexs_betaClampMonomer, 2 * n),
        ):
            self.enzymes[idx - 1] += d
        self.releaseProteinFromSites(clamps, False)
        n = len(clamps)
        self.enzymes[self.enzymeIndexs_betaClamp - 1] -= n
        self.enzymes[self.enzymeIndexs_betaClampMonomer - 1] += 2 * n
        out = self.bindProteinToChromosome(
            bind,
            self.enzymeIndexs_coreBetaClampPrimase,
            None,
            None,
            True,
            False,
            1,
            False,
            [],
        )
        if not np.all(out[0]):
            raise RuntimeError("lagging strand polymerases must bind")
        n = len(bind)
        for idx, d in (
            (self.enzymeIndexs_primase, -n),
            (self.enzymeIndexs_core, -n),
            (self.enzymeIndexs_betaClampMonomer, -2 * n),
            (self.enzymeIndexs_coreBetaClampPrimase, n),
        ):
            self.enzymes[idx - 1] += d
        if len(term):
            n = len(term)
            self.modifyProteinOnChromosome(
                term,
                np.full(
                    n, self.enzymeIndexs_2coreBetaClampGammaComplexPrimase
                ),
            )
            for idx, d in (
                (self.enzymeIndexs_coreBetaClampGammaComplex, -n),
                (self.enzymeIndexs_core, n),
                (self.enzymeIndexs_betaClampMonomer, 2 * n),
                (self.enzymeIndexs_gammaComplex, n),
                (self.enzymeIndexs_2coreBetaClampGammaComplexPrimase, n),
                (self.enzymeIndexs_primase, -n),
                (self.enzymeIndexs_core, -2 * n),
                (self.enzymeIndexs_betaClampMonomer, -2 * n),
                (self.enzymeIndexs_gammaComplex, -n),
            ):
                self.enzymes[idx - 1] += d

    def ligateDNA(self):
        c = self.chromosome
        pos = c.singleStrandBreaks.find()[0]
        if len(pos) == 0:
            return
        n = max(
            0,
            int(
                np.floor(
                    min(
                        len(pos),
                        self.randStream.stochasticRound(
                            self.enzymes[self.enzymeIndexs_ligase - 1]
                            * self.stepSizeSec
                            * self.ligaseRate
                        ),
                        self.substrates[self.substrateIndexs_nad - 1],
                    )
                )
            ),
        )
        if n == 0:
            return
        out = self.bindProteinToChromosome(
            pos,
            self.enzymeIndexs_ligase,
            n,
            None,
            False,
            True,
            1,
            False,
            [1, 2],
        )
        pos = out[2]
        n = out[3]
        if n == 0:
            return
        c.strandBreaks = c.strandBreaks.set_coords(pos, 0)
        for idx, d in (
            (self.substrateIndexs_nad, -n),
            (self.substrateIndexs_nmn, n),
            (self.substrateIndexs_amp, n),
            (self.substrateIndexs_hydrogen, n),
        ):
            self.substrates[idx - 1] += d

    def terminateReplication(self):
        if (
            not np.all(self.leadingStrandElongating)
            or not np.all(self.strandPolymerized)
            or np.any(self.laggingStrandElongating)
        ):
            return
        ter = int(np.asarray(self.terCPosition).item())
        self.chromosome.strandBreaks = self.chromosome.strandBreaks.set_coords(
            np.array([[ter, 2], [ter, 3]], dtype=int), 1
        )
        if (
            len(
                self.releaseProteinFromChromosome(
                    self.enzymeIndexs_helicase, np.inf, None, None
                )
            )
            != 2
        ):
            raise RuntimeError("Two helicases must be released")
        if (
            len(
                self.releaseProteinFromChromosome(
                    self.enzymeIndexs_2coreBetaClampGammaComplexPrimase,
                    np.inf,
                    None,
                    None,
                )
            )
            != 2
        ):
            raise RuntimeError("Two polymerase complexes must be released")
        self.enzymes += (
            2
            * self.enzymeComposition[
                :, self.enzymeIndexs_2coreBetaClampGammaComplexPrimase - 1
            ]
        ).reshape(self.enzymes.shape)
        self.enzymes[
            self.enzymeIndexs_2coreBetaClampGammaComplexPrimase - 1
        ] -= 2
