"""Sparse chromosome damage views and strand-relative coordinate shifts."""

from __future__ import annotations

import numpy as np

from ..circular_sparsemat import CircularSparseMat
from ..sparsemat import SparseMat


class ChromosomeDamageViews:
    """Queries derived from the chromosome's sparse damage matrices."""

    def getDamagedSites(
        self,
        includeBases,
        includeBonds,
        includeBase5Prime,
        includeBond5Prime,
        includeBase3Prime,
        includeBond3Prime,
        includeM6AD,
    ):
        flags = tuple(
            bool(x)
            for x in (
                includeBases,
                includeBonds,
                includeBase5Prime,
                includeBond5Prime,
                includeBase3Prime,
                includeBond3Prime,
                includeM6AD,
            )
        )
        if getattr(self, "_disable_damage_cache", False):
            return self._computeDamagedSites(*flags)
        # Exact content keys detect both replacement and in-place sparse edits.
        # Keep only the current input version; no trajectory data accumulates.
        parts = []
        names = (
            "gapSites",
            "abasicSites",
            "damagedSugarPhosphates",
            "damagedBases",
            "intrastrandCrossLinks",
            "strandBreaks",
            "hollidayJunctions",
        )
        # Standalone fixtures may supply shifted views as stored values rather
        # than the runtime's derived getters; include those explicit inputs too.
        names += tuple(
            name
            for name in (
                "intrastrandCrossLinks5",
                "intrastrandCrossLinks3",
                "strandBreaks5",
                "strandBreaks3",
                "hollidayJunctions5",
                "hollidayJunctions3",
            )
            if name in self.__dict__
        )
        for name in names:
            value = getattr(self, name, None)
            if value is None:
                parts.append(None)
            elif isinstance(value, SparseMat):
                parts.append(
                    (
                        value.siz.tobytes(),
                        value.subs.tobytes(),
                        value.vals.dtype.str,
                        value.vals.tobytes(),
                        tuple(value.__dict__.get("circularDims", ())),
                    )
                )
            else:
                return self._computeDamagedSites(*flags)
        metabolite = getattr(self, "metabolite", None)
        methyl = getattr(metabolite, "m6ADIndexs", None)
        parts.append(
            None
            if methyl is None
            else (
                np.asarray(methyl).dtype.str,
                np.asarray(methyl).shape,
                np.asarray(methyl).tobytes(),
            )
        )
        token = tuple(parts)
        if token != getattr(self, "_damage_cache_token", None):
            self._damage_cache_token = token
            self._damage_cache = {}
        cache = self._damage_cache
        if flags not in cache:
            cache[flags] = self._computeDamagedSites(*flags)
        return cache[flags].copy()

    def _computeDamagedSites(
        self,
        includeBases,
        includeBonds,
        includeBase5Prime,
        includeBond5Prime,
        includeBase3Prime,
        includeBond3Prime,
        includeM6AD,
    ):
        if not includeBases:
            value = CircularSparseMat([], [], self.gapSites.siz, 1)
        else:
            damaged = self.damagedBases
            if not includeM6AD:
                coords, vals = damaged.find()
                m6ad = int(np.asarray(self.metabolite.m6ADIndexs).item())
                keep = vals[:, 0] != m6ad
                value = CircularSparseMat(
                    coords[keep], vals[keep], damaged.siz, 1
                )
            else:
                value = CircularSparseMat(damaged, 1)
            for name in ("gapSites", "abasicSites", "damagedSugarPhosphates"):
                value = value + getattr(self, name)
            value = value + self.intrastrandCrossLinks
            if includeBase5Prime:
                value = value + self.intrastrandCrossLinks5
            if includeBase3Prime:
                value = value + self.intrastrandCrossLinks3
        if includeBonds:
            value = value + self.strandBreaks
            if includeBond5Prime:
                value = value + self.strandBreaks5
            if includeBond3Prime:
                value = value + self.strandBreaks3
            value = value + self.hollidayJunctions
            if includeBond5Prime:
                value = value + self.hollidayJunctions5
            if includeBond3Prime:
                value = value + self.hollidayJunctions3
        return CircularSparseMat(value.subs, value.vals, value.siz, 1)

    def calcDamagedSites(self):
        return self.getDamagedSites(True, True, True, True, False, True, False)

    def calcDamagedSites_shifted_incm6AD(self):
        return self.getDamagedSites(True, True, True, True, False, True, True)

    def calcDamagedSites_nonRedundant(self):
        return self.getDamagedSites(
            True, True, False, False, False, False, True
        )

    def calcDamagedSites_excm6AD(self):
        return self.getDamagedSites(
            True, True, False, False, False, False, False
        )

    def _shift_damage_view(self, name, shift):
        return (
            self.shiftCircularSparseMatBase5Prime(getattr(self, name))
            if shift == 3
            else self.shiftCircularSparseMatBase3Prime(getattr(self, name))
        )

    def calcGapSites3(self):
        return self._shift_damage_view("gapSites", 3)

    def calcGapSites5(self):
        return self._shift_damage_view("gapSites", 5)

    def calcAbasicSites3(self):
        return self._shift_damage_view("abasicSites", 3)

    def calcAbasicSites5(self):
        return self._shift_damage_view("abasicSites", 5)

    def calcDamagedSugarPhosphates3(self):
        return self._shift_damage_view("damagedSugarPhosphates", 3)

    def calcDamagedSugarPhosphates5(self):
        return self._shift_damage_view("damagedSugarPhosphates", 5)

    def calcDamagedBases3(self):
        return self._shift_damage_view("damagedBases", 3)

    def calcDamagedBases5(self):
        return self._shift_damage_view("damagedBases", 5)

    def calcIntrastrandCrossLinks3(self):
        return self._shift_damage_view("intrastrandCrossLinks", 3)

    def calcIntrastrandCrossLinks5(self):
        return self._shift_damage_view("intrastrandCrossLinks", 5)

    def calcStrandBreaks3(self):
        return self.unshiftCircularSparseMatBond3Prime(self.strandBreaks)

    def calcStrandBreaks5(self):
        return self.unshiftCircularSparseMatBond5Prime(self.strandBreaks)

    def calcHollidayJunctions3(self):
        return self.shiftCircularSparseMatBond5Prime(self.hollidayJunctions)

    def calcHollidayJunctions5(self):
        return self.shiftCircularSparseMatBond3Prime(self.hollidayJunctions)

    @staticmethod
    def _shift_circular(sparseMat, fn, *args):
        if not isinstance(
            sparseMat, CircularSparseMat
        ) or not sparseMat.isDimCircular(1):
            raise ValueError(
                "sparseMat must be a circular sparse mat with a circular first dimension"
            )
        coords, vals = sparseMat.find()
        coords = fn(coords, *args)
        return CircularSparseMat(coords, vals, sparseMat.siz, 1)

    @staticmethod
    def shiftCircularSparseMatBase3Prime(sparseMat, *args):
        return ChromosomeDamageViews._shift_circular(
            sparseMat,
            ChromosomeDamageViews.shiftPositionsStrandsBase3Prime,
            *args,
        )

    @staticmethod
    def shiftCircularSparseMatBase5Prime(sparseMat, *args):
        return ChromosomeDamageViews._shift_circular(
            sparseMat,
            ChromosomeDamageViews.shiftPositionsStrandsBase5Prime,
            *args,
        )

    @staticmethod
    def shiftCircularSparseMatBond3Prime(sparseMat, *args):
        return ChromosomeDamageViews._shift_circular(
            sparseMat,
            ChromosomeDamageViews.shiftPositionsStrandsBond3Prime,
            *args,
        )

    @staticmethod
    def unshiftCircularSparseMatBond3Prime(sparseMat, *args):
        return ChromosomeDamageViews._shift_circular(
            sparseMat,
            ChromosomeDamageViews.unshiftPositionsStrandsBond3Prime,
            *args,
        )

    @staticmethod
    def shiftCircularSparseMatBond5Prime(sparseMat, *args):
        return ChromosomeDamageViews._shift_circular(
            sparseMat,
            ChromosomeDamageViews.shiftPositionsStrandsBond5Prime,
            *args,
        )

    @staticmethod
    def unshiftCircularSparseMatBond5Prime(sparseMat, *args):
        return ChromosomeDamageViews._shift_circular(
            sparseMat,
            ChromosomeDamageViews.unshiftPositionsStrandsBond5Prime,
            *args,
        )

    @staticmethod
    def _positions(positionsStrands, lengths, delta):
        p = np.asarray(positionsStrands, dtype=np.int64).copy()
        if p.size == 0:
            return p if p.ndim == 2 else p.reshape(0, 2)
        if p.ndim == 1:
            p = p.reshape(1, -1)
        if p.shape[1] < 2:
            raise ValueError(
                "positionsStrands must include position and strand columns"
            )
        shifts = np.asarray(
            1 if lengths is None else lengths, dtype=np.int64
        ).reshape(-1)
        for parity in (1, 0):
            mask = p[:, 1] % 2 == parity
            if not np.any(mask):
                continue
            direction = delta(p[mask, 1])
            if not np.any(direction):
                # Bond-shift source methods only assign one strand parity;
                # the untouched parity imposes no `lengths` shape constraint.
                continue
            if shifts.size == 1:
                amount = shifts.item()
            elif shifts.size == np.count_nonzero(mask):
                # MATLAB applies the entire `lengths` vector to each parity
                # subset independently; it does not index lengths by source row.
                amount = shifts
            else:
                raise ValueError(
                    "lengths must be scalar or match each nonempty strand-parity subset"
                )
            p[mask, 0] += direction * amount
        return p

    @staticmethod
    def shiftPositionsStrandsBase3Prime(p, lengths=None):
        return ChromosomeDamageViews._positions(
            p, lengths, lambda s: np.where(s % 2 == 1, 1, -1)
        )

    @staticmethod
    def shiftPositionsStrandsBase5Prime(p, lengths=None):
        return ChromosomeDamageViews._positions(
            p, lengths, lambda s: np.where(s % 2 == 1, -1, 1)
        )

    @staticmethod
    def shiftPositionsStrandsBond3Prime(p, lengths=None):
        return ChromosomeDamageViews._positions(
            p, lengths, lambda s: np.where(s % 2 == 0, -1, 0)
        )

    @staticmethod
    def unshiftPositionsStrandsBond3Prime(p, lengths=None):
        return ChromosomeDamageViews._positions(
            p, lengths, lambda s: np.where(s % 2 == 0, 1, 0)
        )

    @staticmethod
    def shiftPositionsStrandsBond5Prime(p, lengths=None):
        return ChromosomeDamageViews._positions(
            p, lengths, lambda s: np.where(s % 2 == 1, -1, 0)
        )

    @staticmethod
    def unshiftPositionsStrandsBond5Prime(p, lengths=None):
        return ChromosomeDamageViews._positions(
            p, lengths, lambda s: np.where(s % 2 == 1, 1, 0)
        )

    @staticmethod
    def calculateFootprintOverhangs(footprint):
        f5 = (np.asarray(footprint) - 1 + 1) // 2
        f3 = np.asarray(footprint) - 1 - f5
        if np.ndim(footprint) == 0:
            return int(f3), int(f5)
        return f3, f5
