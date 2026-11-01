"""Protein monomer state translated from WholeCell's ``ProteinMonomer``."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from wholecell_py.molecule_count_state import MoleculeCountState, _field


class ProteinMonomer(MoleculeCountState):
    optionNames = ("verbosity", "seed")
    fixedConstantNames = (
        "molecularWeights",
        "baseCounts",
        "lengths",
        "halfLives",
        "compartments",
        "minimumAverageExpression",
        "macromoleculeStateInitializationVariation",
    )
    fittedConstantNames: tuple[str, ...] = ()
    stateNames = ("counts",)
    dependentStateNames: tuple[str, ...] = ()

    _index_names = (
        "nascentIndexs",
        "processedIIndexs",
        "processedIIIndexs",
        "signalSequenceIndexs",
        "foldedIndexs",
        "matureIndexs",
        "inactivatedIndexs",
        "boundIndexs",
        "misfoldedIndexs",
        "damagedIndexs",
        "translationFactorIndexs",
    )
    _snapshot_names = (
        MoleculeCountState._snapshot_names
        + _index_names
        + (
            "minimumAverageExpression",
            "macromoleculeStateInitializationVariation",
        )
    )

    def __init__(
        self,
        wholeCellModelID: Any = None,
        name: Any = None,
        snapshot: Mapping[str, Any] | None = None,
        **attributes: Any,
    ) -> None:
        super().__init__(wholeCellModelID, name, snapshot, **attributes)
        for field in self._index_names:
            if not hasattr(self, field):
                setattr(self, field, None)
        if not hasattr(self, "minimumAverageExpression"):
            self.minimumAverageExpression = None
        if not hasattr(self, "macromoleculeStateInitializationVariation"):
            self.macromoleculeStateInitializationVariation = None
        self.chromosome = getattr(self, "chromosome", None)
        self.ribosome = getattr(self, "ribosome", None)

    def storeObjectReferences(self, simulation: Any) -> None:
        super().storeObjectReferences(simulation)
        self.chromosome = _simulation_state(simulation, "Chromosome")
        self.ribosome = _simulation_state(simulation, "Ribosome")

    def initializeConstants(
        self, knowledgeBase: Any, simulation: Any = None
    ) -> None:
        """Derive the ten monomer species blocks from MATLAB KB record fields."""
        super().initializeConstants(knowledgeBase, simulation)
        if not hasattr(knowledgeBase, "proteinMonomers") and not (
            isinstance(knowledgeBase, Mapping)
            and "proteinMonomers" in knowledgeBase
        ):
            raise NotImplementedError(
                "ProteinMonomer constant derivation requires knowledge-base proteinMonomers records"
            )
        monomers = list(_field(knowledgeBase, "proteinMonomers"))
        n = int(_field(knowledgeBase, "numProteinMonomers"))
        if len(monomers) != n:
            raise ValueError(
                "numProteinMonomers does not match proteinMonomers records"
            )

        ids = [_field(record, "wholeCellModelID") for record in monomers]
        names = [_field(record, "name") for record in monomers]
        self.nascentIndexs = np.arange(1, n + 1, dtype=int).reshape(-1, 1)
        self.processedIIndexs = self.nascentIndexs + n
        self.processedIIIndexs = self.processedIIndexs + n
        self.signalSequenceIndexs = self.processedIIIndexs + n
        self.foldedIndexs = self.signalSequenceIndexs + n
        self.matureIndexs = self.foldedIndexs + n
        self.inactivatedIndexs = self.matureIndexs + n
        self.boundIndexs = self.inactivatedIndexs + n
        self.misfoldedIndexs = self.boundIndexs + n
        self.damagedIndexs = self.misfoldedIndexs + n

        self.wholeCellModelIDs = ids * 10
        self.names = names * 10
        mw_fields = (
            "molecularWeight",
            "processedISequenceMolecularWeight",
            "processedIISequenceMolecularWeight",
            "signalSequenceMolecularWeight",
            "foldedSequenceMolecularWeight",
            "matureSequenceMolecularWeight",
        )
        weights = [
            np.asarray([_field(r, key) for r in monomers]).reshape(-1)
            for key in mw_fields
        ]
        self.molecularWeights = np.concatenate(
            weights[:5] + [weights[5]] * 5
        ).reshape(-1, 1)

        count_fields = (
            "baseCount",
            "processedISequenceBaseCount",
            "processedIISequenceBaseCount",
            "signalSequenceBaseCount",
            "foldedSequenceBaseCount",
            "matureSequenceBaseCount",
        )
        count_blocks = [
            np.vstack(
                [np.asarray(_field(r, key)).reshape(1, -1) for r in monomers]
            )
            for key in count_fields
        ]
        self.baseCounts = np.vstack(count_blocks[:5] + [count_blocks[5]] * 5)

        length_fields = (
            "sequenceLength",
            "processedISequenceLength",
            "processedIISequenceLength",
            "signalSequenceLength",
            "foldedSequenceLength",
            "matureSequenceLength",
        )
        length_blocks = [
            np.asarray(
                [_field(r, key) for r in monomers], dtype=float
            ).reshape(-1)
            for key in length_fields
        ]
        self.lengths = np.concatenate(
            length_blocks[:5] + [length_blocks[5]] * 5
        ).reshape(-1, 1)

        compartments = np.asarray(
            _field(knowledgeBase, "proteinMonomerCompartments"), dtype=float
        ).reshape(-1)
        if compartments.size != n:
            raise ValueError(
                "proteinMonomerCompartments must align with proteinMonomers"
            )
        self.compartments = np.tile(compartments, 10).reshape(-1, 1)
        self.compartments[self.signalSequenceIndexs - 1, 0] = (
            _compartment_index(self.compartment, "cytosolIndexs")
        )

        half_life_fields = (
            "halfLife",
            "processedISequenceHalfLife",
            "processedIISequenceHalfLife",
            "signalSequenceHalfLife",
            "foldedSequenceHalfLife",
            "matureSequenceHalfLife",
        )
        half_life_blocks = [
            np.asarray(
                [_field(r, key) for r in monomers], dtype=float
            ).reshape(-1)
            for key in half_life_fields
        ]
        half_lives = np.concatenate(
            half_life_blocks[:5]
            + [half_life_blocks[5]] * 3
            + [np.zeros(n), np.zeros(n)]
        )
        compartment = self.compartment
        allowed = np.r_[
            np.asarray(
                _field(compartment, "cytosolIndexs"), dtype=int
            ).reshape(-1),
            np.asarray(
                _field(compartment, "terminalOrganelleCytosolIndexs"),
                dtype=int,
            ).reshape(-1),
        ]
        half_lives[~np.isin(self.compartments.reshape(-1), allowed)] = np.inf
        half_lives[self.signalSequenceIndexs.reshape(-1) - 1] = 0
        self.halfLives = half_lives.reshape(-1, 1)
        self.translationFactorIndexs = np.asarray(
            self.getIndexs("MG_026_MONOMER"), dtype=int
        ).reshape(-1, 1)

    def updateExternalState(
        self, deltaProteins: Any, proteinIsDegraded: Any
    ) -> np.ndarray:
        """Release chromosome-bound monomers corresponding to decreased bound counts."""
        chromosome = _require(self, "chromosome")
        stream = _require(self, "randStream")
        if not hasattr(chromosome, "setRegionProteinUnbound") or not hasattr(
            stream, "randomlySelectNRows"
        ):
            raise NotImplementedError(
                "updateExternalState requires chromosome unbinding and runtime random-selection methods"
            )
        delta = np.asarray(deltaProteins)
        if delta.ndim != 2:
            raise ValueError(
                "deltaProteins must have molecule-by-compartment dimensions"
            )
        compartments = _require(self, "compartment")
        bound_idxs = np.asarray(self.boundIndexs, dtype=int).reshape(-1)
        cytosol = int(np.asarray(compartments.cytosolIndexs).reshape(-1)[0])
        delta_bound = -delta[bound_idxs - 1, cytosol - 1].copy()
        translation = np.asarray(
            self.translationFactorIndexs, dtype=int
        ).reshape(-1)
        if translation.size:
            delta_bound[translation - 1] = 0

        sites = chromosome.monomerBoundSites
        if not hasattr(sites, "find"):
            raise NotImplementedError(
                "updateExternalState requires monomerBoundSites.find()"
            )
        coordinates, values = sites.find()
        coordinates = np.asarray(coordinates, dtype=int)
        proteins = np.asarray(values).reshape(-1).astype(int)
        releases: list[np.ndarray] = []
        for protein_idx in np.flatnonzero(delta_bound):
            candidates = coordinates[proteins == protein_idx + 1, :2]
            count = int(delta_bound[protein_idx])
            if count < 0 or len(candidates) < count:
                raise RuntimeError("Error updating external state")
            selected, _ = stream.randomlySelectNRows(candidates, count)
            selected = np.asarray(selected, dtype=int).reshape((-1, 2))
            releases.append(selected)
        positions = (
            np.concatenate(releases, axis=0)
            if releases
            else np.empty((0, 2), dtype=int)
        )
        released, _, _ = chromosome.setRegionProteinUnbound(
            positions,
            1,
            np.flatnonzero(delta_bound) + 1,
            [],
            False,
            False,
            False,
            proteinIsDegraded,
        )
        released = np.asarray(released).reshape(-1)
        expected = delta_bound[np.flatnonzero(delta_bound)]
        if not np.array_equal(expected, released):
            raise RuntimeError(
                "Chromosomally bound proteins improperly released"
            )
        return np.zeros_like(delta)

    def getIndexs(self, wholeCellModelIDs: Any) -> Any:
        """Return one-based monomer indices for mature-form IDs."""
        if self.wholeCellModelIDs is None or self.matureIndexs is None:
            raise ValueError(
                "wholeCellModelIDs and matureIndexs must be initialized"
            )
        ids = np.asarray(self.wholeCellModelIDs, dtype=object).reshape(
            -1, order="F"
        )
        mature = np.asarray(self.matureIndexs, dtype=int).reshape(
            -1, order="F"
        )
        requested = np.asarray(wholeCellModelIDs, dtype=object)
        flat = requested.reshape(-1, order="F")
        result = []
        mature_ids = ids[mature - 1]
        for item in flat:
            match = np.flatnonzero(mature_ids == item)
            result.append(int(match[0] + 1) if match.size else 0)
        values = np.asarray(result, dtype=int)
        if requested.ndim == 0:
            return int(values[0])
        return values.reshape(requested.shape, order="F")


def _require(obj: Any, field: str) -> Any:
    value = getattr(obj, field, None)
    if value is None:
        raise RuntimeError(
            f"ProteinMonomer dependency {field!r} has not been assigned"
        )
    return value


def _simulation_state(simulation: Any, name: str) -> Any:
    if isinstance(simulation, Mapping):
        states = simulation.get("states", {})
        if name in states:
            return states[name]
    getter = getattr(simulation, "state", None)
    if getter is None:
        raise RuntimeError(f"simulation cannot provide state {name!r}")
    return getter(name)


def _compartment_index(compartment: Any, name: str) -> int:
    values = np.asarray(_field(compartment, name), dtype=int).reshape(-1)
    if values.size != 1:
        raise ValueError(f"compartment.{name} must identify one compartment")
    return int(values[0])
