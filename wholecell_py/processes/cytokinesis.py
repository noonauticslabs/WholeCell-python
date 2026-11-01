"""Cytokinesis process translated from WholeCell's Cytokinesis.m."""

from __future__ import annotations

import numpy as np

from ..process_base import Process, _get, _set
from ..chromosome_process_aspect import ChromosomeProcessAspect
from ..states.ftsz_ring import FtsZRing


def _scalar(value):
    """Unbox MATLAB 1-by-1 snapshot arrays while preserving larger arrays."""
    array = np.asarray(value)
    return array.item() if array.size == 1 else value


class Cytokinesis(ChromosomeProcessAspect, Process):
    optionNames__ = ()
    fixedConstantNames__ = (
        "rateFtsZGtpHydrolysis",
        "rateFilamentBindingMembrane",
        "rateFilamentDissociation",
    )
    fittedConstantNames__ = ()
    localStateNames__ = ()

    stimuliWholeCellModelIDs = ()
    substrateWholeCellModelIDs = ("PI", "H2O", "H")
    substrateIndexs_phosphate = 1
    substrateIndexs_water = 2
    substrateIndexs_hydrogen = 3
    enzymeWholeCellModelIDs = (
        "MG_224_9MER_GTP",
        "MG_224_9MER_GDP",
        "MG_224_MONOMER_GDP",
        "MG_224_MONOMER_GTP",
    )
    enzymeIndexs_ftsZ_GTP_polymer = 1
    enzymeIndexs_ftsZ_GDP_polymer = 2
    enzymeIndexs_ftsZ_GDP = 3
    enzymeIndexs_ftsZ_GTP = 4

    def __init__(
        self,
        wholeCellModelID="Process_Cytokinesis",
        name="Cytokinesis",
        **properties,
    ):
        super().__init__(wholeCellModelID, name, **properties)
        self.ftsZRing = getattr(self, "ftsZRing", None)
        self.chromosome = getattr(self, "chromosome", None)

    def storeObjectReferences(self, simulation):
        super().storeObjectReferences(simulation)
        statefn = getattr(simulation, "state", None)
        if callable(statefn):
            self.chromosome = statefn("Chromosome")
            self.ftsZRing = statefn("FtsZRing")
        else:
            states = _get(simulation, "states")
            normalized = {str(k).lower(): v for k, v in states.items()}
            self.chromosome = normalized["chromosome"]
            self.ftsZRing = normalized["ftszring"]
        self.states = [*self.states, self.chromosome, self.ftsZRing]

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

    def initializeState(self, *args, **kwargs):
        pass

    def calcResourceRequirements_LifeCycle(self, *args, **kwargs):
        n_substrates = len(self.substrateWholeCellModelIDs)
        n_enzymes = len(self.enzymeWholeCellModelIDs)
        bmProd = np.zeros(n_substrates)
        byProd = np.zeros(n_substrates)
        minEnzExp = np.zeros(n_enzymes)
        maxEnzExp = np.full(n_enzymes, np.inf)
        _, gtpCost = self.calcRequiredPinchingCycles(
            self.geometry.width,
            self.ftsZRing.numFtsZSubunitsPerFilament,
            self.ftsZRing.filamentLengthInNm,
        )
        bmProd[int(self.substrateIndexs_water) - 1] = gtpCost
        byProd[int(self.substrateIndexs_phosphate) - 1] = gtpCost
        byProd[int(self.substrateIndexs_hydrogen) - 1] = gtpCost
        minEnzExp[int(self.enzymeIndexs_ftsZ_GTP_polymer) - 1] = (
            4
            * self.ftsZRing.calcNumEdges(
                self.geometry.width, self.ftsZRing.filamentLengthInNm
            )
        )
        return bmProd, byProd, minEnzExp, maxEnzExp

    def calcResourceRequirements_Current(self):
        result = np.zeros_like(np.asarray(self.substrates), dtype=float)
        # The MATLAB expression uses linear one-based indexing, so with
        # multiple compartments it addresses the first compartment only.
        flat_enzymes = np.asarray(self.enzymes).ravel(order="F")
        result_index = np.unravel_index(
            int(_scalar(self.substrateIndexs_water)) - 1,
            result.shape,
            order="F",
        )
        result[result_index] = (
            _scalar(self.ftsZRing.numFtsZSubunitsPerFilament)
            * flat_enzymes[
                int(_scalar(self.enzymeIndexs_ftsZ_GTP_polymer)) - 1
            ]
        )
        return result

    @staticmethod
    def _rand(stream):
        if stream is None or not callable(getattr(stream, "rand", None)):
            raise NotImplementedError(
                "Cytokinesis.evolveState requires a runtime random stream with rand()"
            )
        return stream.rand()

    def evolveState(self):
        if not _get(self.chromosome, "segregated"):
            return
        ring = self.ftsZRing
        # MATLAB exports this scalar as a 1-by-1 array. Unbox it before
        # in-place arithmetic on the one-dimensional compartment row.
        subunits_per_filament = _scalar(ring.numFtsZSubunitsPerFilament)
        gtp_poly = int(self.enzymeIndexs_ftsZ_GTP_polymer) - 1
        gdp_poly = int(self.enzymeIndexs_ftsZ_GDP_polymer) - 1
        gdp = int(self.enzymeIndexs_ftsZ_GDP) - 1
        water = int(self.substrateIndexs_water) - 1
        phosphate = int(self.substrateIndexs_phosphate) - 1
        hydrogen = int(self.substrateIndexs_hydrogen) - 1

        if ring.numEdgesTwoBent == 0 and not self.geometry.pinched:
            for _ in range(2):
                for _ in range(
                    int(
                        ring.numEdges
                        - ring.numEdgesOneStraight
                        - ring.numEdgesTwoStraight
                    )
                ):
                    if (
                        self._rand(self.randStream)
                        <= self.rateFilamentBindingMembrane
                        and self.enzymes[gtp_poly] >= 1
                    ):
                        self.enzymes[gtp_poly] -= 1
                        self.boundEnzymes[gtp_poly] += 1
                        ring.numEdgesOneStraight += 1
            for _ in range(int(ring.numEdgesOneStraight)):
                if (
                    self._rand(self.randStream)
                    <= self.rateFilamentBindingMembrane
                    and self.enzymes[gtp_poly] >= 1
                ):
                    self.enzymes[gtp_poly] -= 1
                    self.boundEnzymes[gtp_poly] += 1
                    ring.numEdgesOneStraight -= 1
                    ring.numEdgesTwoStraight += 1

        if (
            ring.numEdgesOneStraight + ring.numEdgesTwoStraight
            == ring.numEdges
            or self.geometry.pinched
        ):
            for _ in range(int(ring.numResidualBent)):
                if (
                    self._rand(self.randStream)
                    <= self.rateFilamentDissociation
                ):
                    ring.numResidualBent -= 1
                    self.boundEnzymes[gdp_poly] -= 1
                    self.enzymes[gdp] += subunits_per_filament

        if (
            ring.numEdgesTwoBent + ring.numEdgesTwoStraight == ring.numEdges
            and ring.numEdgesTwoStraight > 0
            and ring.numResidualBent == 0
        ):
            for _ in range(int(ring.numEdgesTwoStraight)):
                n = 2 * subunits_per_filament
                if (
                    self._rand(self.randStream) <= self.rateFtsZGtpHydrolysis
                    and self.substrates[water] >= n
                ):
                    ring.numEdgesTwoStraight -= 1
                    ring.numEdgesTwoBent += 1
                    self.substrates[water] -= n
                    self.substrates[phosphate] += n
                    self.substrates[hydrogen] += n
                    self.boundEnzymes[gtp_poly] -= 2
                    self.boundEnzymes[gdp_poly] += 2
            if ring.numEdgesTwoStraight == 0 and not self.geometry.pinched:
                self.geometry.pinchedDiameter = self.calcNextPinchedDiameter(
                    self.geometry.pinchedDiameter, ring.filamentLengthInNm
                )

        if ring.numEdgesTwoStraight == 0:
            for _ in range(int(ring.numEdgesTwoBent)):
                if (
                    self._rand(self.randStream)
                    <= self.rateFilamentDissociation
                ):
                    ring.numEdgesTwoBent -= 1
                    ring.numResidualBent += 1
                    self.boundEnzymes[gdp_poly] -= 1
                    self.enzymes[gdp] += subunits_per_filament

    @staticmethod
    def calcNextPinchedDiameter(pinchedDiameter, filamentLengthInNm):
        numEdges = FtsZRing.calcNumEdges(pinchedDiameter, filamentLengthInNm)
        flooredDiameter = filamentLengthInNm * 1e-9 / np.sin(np.pi / numEdges)
        newDiameter = filamentLengthInNm * 1e-9 * numEdges / np.pi
        result = newDiameter + (pinchedDiameter - flooredDiameter)
        if result <= filamentLengthInNm * 1e-9:
            result = 0
        return result

    @staticmethod
    def calcRequiredPinchingCycles(
        diameter, numFtsZSubunitsPerFilament, filamentLengthInNm
    ):
        nCycles, nGTP = 0, 0
        while diameter > 0:
            nCycles += 1
            nGTP += (
                2
                * numFtsZSubunitsPerFilament
                * FtsZRing.calcNumEdges(diameter, filamentLengthInNm)
            )
            diameter = Cytokinesis.calcNextPinchedDiameter(
                diameter, filamentLengthInNm
            )
        return nCycles, nGTP
