"""ChromosomeSegregation.evolveState deterministic port.

Source: WholeCell/src/+edu/+stanford/+covert/+cell/+sim/+process/ChromosomeSegregation.m
"""

from __future__ import annotations

import numpy as np

from ..process_base import Process, _get, _set
from ..sparsemat import SparseMat
from ..chromosome_process_aspect import ChromosomeProcessAspect


class ChromosomeSegregation(Process, ChromosomeProcessAspect):
    """Process that segregates a fully replicated, supercoiled chromosome.

    The indices stored on the process follow the source MATLAB class and are
    one-based. Snapshot fields can be passed as keyword arguments, just as the
    snapshot loader assigns them after constructing a process.
    """

    substrateWholeCellModelIDs = ("GTP", "GDP", "H", "H2O", "PI")
    enzymeWholeCellModelIDs = (
        "MG_470_MONOMER",
        "MG_221_OCTAMER",
        "MG_387_MONOMER",
        "MG_384_MONOMER",
        "MG_203_204_TETRAMER",
    )
    substrateIndexs_gtp = 1
    substrateIndexs_gdp = 2
    substrateIndexs_hydrogen = 3
    substrateIndexs_water = 4
    substrateIndexs_phosphate = 5
    enzymeIndexs_cobQ = 1
    enzymeIndexs_mraZ = 2
    enzymeIndexs_obg = 3
    enzymeIndexs_era = 4
    enzymeIndexs_topoIV = 5

    def __init__(
        self,
        wholeCellModelID: str = "Process_ChromosomeSegregation",
        name: str = "Chromosome Segregation",
        **properties,
    ):
        properties.setdefault("gtpCost", 1)
        super().__init__(wholeCellModelID, name, **properties)
        self.chromosome = getattr(self, "chromosome", None)

    def storeObjectReferences(self, simulation):
        """Keep the base state references and the shared chromosome state."""
        super().storeObjectReferences(simulation)
        statefn = getattr(simulation, "state", None)
        if callable(statefn):
            self.chromosome = statefn("Chromosome")
        else:
            states = _get(simulation, "states")
            normalized = {str(k).lower(): v for k, v in states.items()}
            self.chromosome = normalized["chromosome"]
        self.states = [*self.states, self.chromosome]

    def initializeConstants(
        self, knowledgeBase=None, simulation=None, options=None, **kwargs
    ):
        for key, value in kwargs.items():
            setattr(self, key, value)
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
        """Chromosome state initializes itself; this process has no local state."""

    @staticmethod
    def _collapse(value):
        if isinstance(value, SparseMat):
            # SparseMat.collapse(None) reduces the stored entries directly and
            # avoids materializing a potentially chromosome-sized dense array.
            return float(value.collapse(None))
        if hasattr(value, "collapse") and callable(value.collapse):
            return float(value.collapse(None))
        return float(np.asarray(value).sum())

    def _is_replicated(self):
        c = self.chromosome
        return self._collapse(_get(c, "polymerizedRegions")) == _get(
            c, "nCompartments"
        ) * _get(c, "sequenceLen")

    def calcResourceRequirements_Current(self):
        """Return GTP and water demand when replication and enzymes permit it."""
        c = self.chromosome
        result = np.zeros_like(np.asarray(self.substrates), dtype=float)
        if (
            not bool(_get(c, "segregated"))
            and self._is_replicated()
            and bool(np.all(self.enzymes))
        ):
            result[int(self.substrateIndexs_gtp) - 1] = self.gtpCost
            result[int(self.substrateIndexs_water) - 1] = self.gtpCost
        return result

    def calcResourceRequirements_LifeCycle(self, *args, **kwargs):
        """Return biomass/byproduct and enzyme-expression requirements."""
        n_substrates = len(self.substrateWholeCellModelIDs)
        n_enzymes = len(self.enzymeWholeCellModelIDs)
        biomass = np.zeros(n_substrates)
        byproducts = np.zeros(n_substrates)
        minimum_enzyme_expression = np.zeros(n_enzymes)
        maximum_enzyme_expression = np.full(n_enzymes, np.inf)
        for index in (self.substrateIndexs_gtp, self.substrateIndexs_water):
            biomass[index - 1] = self.gtpCost
        for index in (
            self.substrateIndexs_gdp,
            self.substrateIndexs_phosphate,
            self.substrateIndexs_hydrogen,
        ):
            byproducts[index - 1] = self.gtpCost
        minimum_enzyme_expression[:] = 2
        return (
            biomass,
            byproducts,
            minimum_enzyme_expression,
            maximum_enzyme_expression,
        )

    def evolveState(self):
        """Segregate and hydrolyze GTP once all source guards pass."""
        c = self.chromosome
        if (
            bool(_get(c, "segregated"))
            or not self._is_replicated()
            or self._collapse(_get(c, "supercoiled"))
            != _get(c, "nCompartments")
            or not bool(np.all(self.enzymes))
        ):
            return

        gtp = int(np.asarray(self.substrateIndexs_gtp).item()) - 1
        water = int(np.asarray(self.substrateIndexs_water).item()) - 1
        cost = np.asarray(self.gtpCost).item()
        # MATLAB's single subscript is column-major linear indexing, and a
        # serialized 1x1 cost is still scalar for arithmetic/assignment.
        values = np.asarray(self.substrates).ravel(order="F").copy()
        if values[gtp] < cost or values[water] < cost:
            return

        _set(c, "segregated", True)
        values[gtp] -= cost
        values[water] -= cost
        for index in (
            self.substrateIndexs_gdp,
            self.substrateIndexs_phosphate,
            self.substrateIndexs_hydrogen,
        ):
            values[int(np.asarray(index).item()) - 1] += cost
        self.substrates[...] = values.reshape(
            np.shape(self.substrates), order="F"
        )


def run_case(inputs: dict, parameters: dict) -> dict:
    """Apply chromosome segregation and GTP hydrolysis if all source guards pass.

    chromosome polymerized_regions and supercoiled are summed as MATLAB
    collapse(A) does. Enzymes are tested elementwise for truth, as all(enzymes).
    """
    c = inputs["chromosome"]
    polymerized = float(np.asarray(c["polymerized_regions"]).sum())
    supercoiled = float(np.asarray(c["supercoiled"]).sum())
    n_comp = int(c["n_compartments"])
    seq_len = int(c["sequence_len"])
    substrates = np.asarray(inputs["substrates"], dtype=float).copy()
    enzymes = np.asarray(inputs["enzymes"])
    cost = float(parameters["gtp_cost"])
    # MATLAB indices: GTP,GDP,H,H2O,PI -> Python indices 0..4.
    if substrates.ndim != 1 or substrates.size != 5:
        raise ValueError(
            "substrates must have shape (5,) in GTP,GDP,H,H2O,PI order"
        )
    segregated = bool(c["segregated"])
    do_segregate = (
        not segregated
        and polymerized == n_comp * seq_len
        and supercoiled == n_comp
        and bool(np.all(enzymes))
        and substrates[0] >= cost
        and substrates[3] >= cost
    )
    if do_segregate:
        segregated = True
        substrates[0] -= cost
        substrates[3] -= cost
        substrates[1] += cost
        substrates[4] += cost
        substrates[2] += cost
    return {"segregated": segregated, "substrates": substrates}
