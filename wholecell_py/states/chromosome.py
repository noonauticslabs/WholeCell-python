"""Chromosome state assembled from faithful source-method ports.

Derived getters use live state, with content-validated caches for three region
views and recomputation elsewhere. Exported getter values are
retained for inspection only, never replayed as dynamic results. Unported
calculations fail explicitly; this class is not yet a complete chromosome.
"""

import numpy as np
from ..circular_sparsemat import CircularSparseMat
from .chromosome_regions import ChromosomeRegions
from .chromosome_mass_view import ChromosomeMassView
from .chromosome_derived import ChromosomeDerived
from .chromosome_polymerized_queries import ChromosomePolymerizedQueries
from .chromosome_accessibility import ChromosomeAccessibility
from .chromosome_damage_views import ChromosomeDamageViews
from .chromosome_synthesis_updates import ChromosomeSynthesisUpdates
from .chromosome_binding_updates import ChromosomeBindingUpdates
from .chromosome_binding_creation import ChromosomeBindingCreation
from .chromosome_damage_updates import ChromosomeDamageUpdates
from .chromosome_breaks import ChromosomeBreaks
from .chromosome_sampling import ChromosomeSampling
from .chromosome_restriction_queries import ChromosomeRestrictionQueries


class Chromosome(
    ChromosomeRegions,
    ChromosomeMassView,
    ChromosomeDerived,
    ChromosomePolymerizedQueries,
    ChromosomeAccessibility,
    ChromosomeDamageViews,
    ChromosomeSynthesisUpdates,
    ChromosomeBindingUpdates,
    ChromosomeBindingCreation,
    ChromosomeSampling,
    ChromosomeRestrictionQueries,
    ChromosomeBreaks,
    ChromosomeDamageUpdates,
):
    def __init__(
        self, wholeCellModelID=None, name=None, snapshot=None, **fields
    ):
        self.wholeCellModelID = wholeCellModelID
        self.name = name
        for key, value in {**(snapshot or {}), **fields}.items():
            setattr(self, key, value)

    def storeObjectReferences(self, simulation):
        self.compartment, self.gene = simulation.compartment, simulation.gene
        self.metabolite = simulation.state("Metabolite")
        self.transcript = simulation.state("Transcript")
        self.rnaPolymerase = simulation.state("RNAPolymerase")
        self.dnaRepair = simulation.process("DNARepair")

    def allocateMemory(self, numTimePoints):
        shape = [
            int(np.asarray(self.sequenceLen).item()),
            int(np.asarray(self.nCompartments).item()),
            int(numTimePoints),
        ]
        bools = {
            "gapSites",
            "abasicSites",
            "strandBreaks",
            "hollidayJunctions",
        }
        for name in (
            "polymerizedRegions",
            "linkingNumbers",
            "monomerBoundSites",
            "complexBoundSites",
            "gapSites",
            "abasicSites",
            "damagedSugarPhosphates",
            "damagedBases",
            "intrastrandCrossLinks",
            "strandBreaks",
            "hollidayJunctions",
        ):
            setattr(
                self,
                name,
                CircularSparseMat(
                    [],
                    np.empty(0, dtype=bool if name in bools else float),
                    shape,
                    1,
                ),
            )
        self.segregated = np.zeros((1, 1, int(numTimePoints)), dtype=bool)
        self.invalidate()

    def initialize(self):
        self.allocateMemory(1)
        strands = np.asarray(self.strandIndexs_ch1).ravel().astype(int)
        coords = np.column_stack((np.ones(strands.size, dtype=int), strands))
        length = float(np.asarray(self.sequenceLen).item())
        self.polymerizedRegions = self.polymerizedRegions.set_coords(
            coords, length
        )
        self.linkingNumbers = self.linkingNumbers.set_coords(
            coords,
            length
            / float(np.asarray(self.relaxedBasesPerTurn).item())
            * (
                1
                + float(np.asarray(self.equilibriumSuperhelicalDensity).item())
            ),
        )
        self.invalidate()

    def invalidate(self):
        # Keep the snapshot-era counter; the three sparse derived getters use
        # content signatures so assignment and in-place mutation are detected.
        self.validated = (
            int(np.asarray(getattr(self, "validated", 0)).item()) + 1
        )


_CACHED_SPARSE_DERIVED = {
    "unpolymerizedRegions",
    "singleStrandedRegions",
    "doubleStrandedRegions",
}


def _array_signature(value):
    array = np.asarray(value)
    return array.dtype.str, array.shape, array.tobytes(order="A")


def _method_identity(method):
    return getattr(method, "__func__", method), getattr(
        method, "__self__", None
    )


def _chromosome_regions_signature(state, calculation):
    source = state.polymerizedRegions
    sparse_fields = tuple(
        _array_signature(getattr(source, field))
        for field in ("subs", "vals", "siz", "circularDims")
    )
    return (
        sparse_fields,
        _array_signature(state.sequenceLen),
        _array_signature(state.nCompartments),
        _method_identity(calculation),
        _method_identity(state.mergeAdjacentRegions),
    )


def _derived(name, method):
    def read(self):
        calculation = getattr(self, method, None)
        if calculation is None:
            raise NotImplementedError(
                f"Chromosome.{method} is not implemented"
            )
        if name not in _CACHED_SPARSE_DERIVED:
            return calculation()

        cache = self.__dict__.get("_derived_sparse_cache")
        if not isinstance(cache, dict):
            cache = {}
            self.__dict__["_derived_sparse_cache"] = cache
        key = _chromosome_regions_signature(self, calculation)
        entry = cache.get(name)
        if entry is not None and entry[0] == key:
            # Derived properties historically returned fresh sparse objects;
            # keep callers from mutating the cached value through the result.
            return entry[1].copy()

        value = calculation()
        cache[name] = (key, value.copy())
        return value

    def restore(self, value):
        self.__dict__["_exported_" + name] = value

    return property(read, restore)


_DERIVED = {
    "unpolymerizedRegions": "calcUnpolymerizedRegions",
    "singleStrandedRegions": "calcSingleStrandedRegions",
    "doubleStrandedRegions": "calcDoubleStrandedRegions",
    "geneCopyNumbers": "calcGeneCopyNumbers",
    "ploidy": "calcPloidy",
    "polymerizedGenes": "calcPolymerizedGenes",
    "transcriptionUnitCopyNumbers": "getTranscriptionUnitCopyNumbers",
    "polymerizedTranscriptionUnits": "calcPolymerizedTranscriptionUnits",
    "geneCopyNumbers_Accessible": "calcCopyNumbers_Accessible",
    "transcriptionUnitCopyNumbers_Accessible": "calcTranscriptionUnitCopyNumbers_Accessible",
    "accessibleGenes": "calcAccessibleGenes",
    "accessibleTranscriptionUnits": "calcAccessibleTranscriptionUnits",
    "strandBreakClassification": "calcStrandBreakClassification",
    "linkingNumbers_minFreeEnergy": "calcLinkingNumbers_minFreeEnergy",
    "supercoils": "calcSupercoils",
    "superhelicalDensity": "calcSuperhelicalDensity",
    "supercoiled": "calcSupercoiled",
    "damagedSites": "calcDamagedSites",
    "damagedSites_shifted_incm6AD": "calcDamagedSites_shifted_incm6AD",
    "damagedSites_nonRedundant": "calcDamagedSites_nonRedundant",
    "damagedSites_excm6AD": "calcDamagedSites_excm6AD",
    "gapSites3": "calcGapSites3",
    "gapSites5": "calcGapSites5",
    "abasicSites3": "calcAbasicSites3",
    "abasicSites5": "calcAbasicSites5",
    "damagedSugarPhosphates3": "calcDamagedSugarPhosphates3",
    "damagedSugarPhosphates5": "calcDamagedSugarPhosphates5",
    "damagedBases3": "calcDamagedBases3",
    "damagedBases5": "calcDamagedBases5",
    "intrastrandCrossLinks3": "calcIntrastrandCrossLinks3",
    "intrastrandCrossLinks5": "calcIntrastrandCrossLinks5",
    "strandBreaks3": "calcStrandBreaks3",
    "strandBreaks5": "calcStrandBreaks5",
    "hollidayJunctions3": "calcHollidayJunctions3",
    "hollidayJunctions5": "calcHollidayJunctions5",
    "singleStrandBreaks": "calcSingleStrandBreaks",
    "doubleStrandBreaks": "calcDoubleStrandBreaks",
    "restrictableMunIRMSites": "calcRestrictableMunIRMSites",
    "hemiunmethylatedMunIRMSites": "calcHemiunmethylatedMunIRMSites",
    "dryWeight": "calcDryWeight",
}
for _name, _method in _DERIVED.items():
    setattr(Chromosome, _name, _derived(_name, _method))
