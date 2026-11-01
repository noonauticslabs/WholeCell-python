"""Protein complex state translated from WholeCell's ``ProteinComplex``."""

from __future__ import annotations

from typing import Any, Mapping
import numpy as np

from wholecell_py.molecule_count_state import MoleculeCountState, _field


class ProteinComplex(MoleculeCountState):
    optionNames = ("verbosity", "seed")
    fixedConstantNames = (
        "molecularWeights",
        "baseCounts",
        "lengths",
        "halfLives",
        "compartments",
        "proteinComplexComposition",
        "minimumAverageExpression",
    )
    fittedConstantNames: tuple[str, ...] = ()
    stateNames = ("counts",)
    dependentStateNames: tuple[str, ...] = ()

    _index_names = (
        "nascentIndexs",
        "matureIndexs",
        "inactivatedIndexs",
        "boundIndexs",
        "misfoldedIndexs",
        "damagedIndexs",
        "ribosome30SIndexs",
        "ribosome50SIndexs",
        "ribosome70SIndexs",
        "translationFactorIndexs",
        "ftsZGTPIndexs",
        "ftsZGDPIndexs",
        "rnaPolymeraseIndexs",
        "replisomeIndexs",
        "dnaPolymeraseIndexs",
        "dnaAPolymerIndexs",
    )
    _snapshot_names = (
        MoleculeCountState._snapshot_names
        + _index_names
        + (
            "proteinComplexComposition",
            "formationProcesses",
            "minimumAverageExpression",
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
        for field in (
            "proteinComplexComposition",
            "formationProcesses",
            "minimumAverageExpression",
        ):
            if not hasattr(self, field):
                setattr(self, field, None)
        for field in ("chromosome", "rnaPolymerase", "ribosome", "ftsZRing"):
            setattr(self, field, getattr(self, field, None))

    def storeObjectReferences(self, simulation: Any) -> None:
        super().storeObjectReferences(simulation)
        self.chromosome = _simulation_state(simulation, "Chromosome")
        self.rnaPolymerase = _simulation_state(simulation, "RNAPolymerase")
        self.ribosome = _simulation_state(simulation, "Ribosome")
        self.ftsZRing = _simulation_state(simulation, "FtsZRing")

    def initializeConstants(
        self, knowledgeBase: Any, simulation: Any = None
    ) -> None:
        super().initializeConstants(knowledgeBase, simulation)
        if not _has_field(knowledgeBase, "proteinComplexs"):
            raise NotImplementedError(
                "ProteinComplex constant derivation requires proteinComplexs records"
            )
        complexes = list(_field(knowledgeBase, "proteinComplexs"))
        n = int(_field(knowledgeBase, "numProteinComplexs"))
        if len(complexes) != n:
            raise ValueError(
                "numProteinComplexs does not match proteinComplexs records"
            )

        ids = [_field(record, "wholeCellModelID") for record in complexes]
        names = [_field(record, "name") for record in complexes]
        self.nascentIndexs = np.arange(1, n + 1, dtype=int).reshape(-1, 1)
        self.matureIndexs = self.nascentIndexs + n
        self.inactivatedIndexs = self.matureIndexs + n
        self.boundIndexs = self.inactivatedIndexs + n
        self.misfoldedIndexs = self.boundIndexs + n
        self.damagedIndexs = self.misfoldedIndexs + n
        self.wholeCellModelIDs = ids * 6
        self.names = names * 6

        self.baseCounts = np.vstack(
            [
                np.vstack(
                    [
                        np.asarray(_field(r, "baseCount")).reshape(1, -1)
                        for r in complexes
                    ]
                )
            ]
            + [
                np.vstack(
                    [
                        np.asarray(_field(r, "matureBaseCount")).reshape(1, -1)
                        for r in complexes
                    ]
                )
            ]
            * 5
        )
        weights = np.asarray(
            [_field(r, "molecularWeight") for r in complexes], dtype=float
        )
        mature_weights = np.asarray(
            [_field(r, "matureMolecularWeight") for r in complexes],
            dtype=float,
        )
        self.molecularWeights = np.concatenate(
            [weights] + [mature_weights] * 5
        ).reshape(-1, 1)
        lengths = np.asarray(
            [_field(r, "sequenceLength") for r in complexes], dtype=float
        )
        mature_lengths = np.asarray(
            [_field(r, "matureSequenceLength") for r in complexes], dtype=float
        )
        self.lengths = np.concatenate(
            [lengths] + [mature_lengths] * 5
        ).reshape(-1, 1)

        compartments = np.asarray(
            _field(knowledgeBase, "proteinComplexCompartments"), dtype=float
        ).reshape(-1)
        if compartments.size != n:
            raise ValueError(
                "proteinComplexCompartments must align with proteinComplexs"
            )
        self.compartments = np.tile(compartments, 6).reshape(-1, 1)

        half_lives = np.asarray(
            [_field(r, "halfLife") for r in complexes], dtype=float
        )
        mature_half_lives = np.asarray(
            [_field(r, "matureHalfLife") for r in complexes], dtype=float
        )
        self.halfLives = np.concatenate(
            [half_lives] + [mature_half_lives] * 3 + [np.zeros(n), np.zeros(n)]
        ).reshape(-1, 1)
        allowed = np.r_[
            _indices(_field(self.compartment, "cytosolIndexs")),
            _indices(
                _field(self.compartment, "terminalOrganelleCytosolIndexs")
            ),
        ]
        self.halfLives[~np.isin(self.compartments.reshape(-1), allowed), 0] = (
            np.inf
        )

        composition = np.array(
            _field(knowledgeBase, "proteinComplexAllRNAComposition"), copy=True
        )
        monomer_composition = np.asarray(
            _field(knowledgeBase, "proteinComplexAllMonomerComposition")
        )
        mrna_rows = np.asarray(
            [
                _field(gene, "idx")
                for gene in _field(knowledgeBase, "mRNAGenes")
            ],
            dtype=int,
        ).reshape(-1)
        if mrna_rows.size:
            composition[mrna_rows - 1, ...] = monomer_composition
        self.proteinComplexComposition = composition

        formation = [
            _field(_field(record, "complexFormationProcess"), "idx")
            for record in complexes
        ]
        self.formationProcesses = np.tile(
            np.asarray(formation, dtype=int), 6
        ).reshape(-1, 1)

        self.ribosome30SIndexs = self.getIndexs("RIBOSOME_30S")
        self.ribosome50SIndexs = self.getIndexs("RIBOSOME_50S")
        self.ribosome70SIndexs = self.getIndexs("RIBOSOME_70S")
        self.translationFactorIndexs = self.getIndexs(
            ["MG_089_DIMER", "MG_433_DIMER", "MG_451_DIMER"]
        )
        self.ftsZGTPIndexs = self.getIndexs("MG_224_9MER_GTP")
        self.ftsZGDPIndexs = self.getIndexs("MG_224_9MER_GDP")
        self.rnaPolymeraseIndexs = self.getIndexs(
            ["RNA_POLYMERASE", "RNA_POLYMERASE_HOLOENZYME"]
        )
        self.replisomeIndexs = self.getIndexs(
            [
                "MG_094_HEXAMER",
                "DNA_POLYMERASE_2CORE_BETA_CLAMP_GAMMA_COMPLEX_PRIMASE",
                "DNA_POLYMERASE_CORE_BETA_CLAMP_GAMMA_COMPLEX",
                "DNA_POLYMERASE_CORE_BETA_CLAMP_PRIMASE",
            ]
        )
        self.dnaPolymeraseIndexs = self.getIndexs(
            [
                "DNA_POLYMERASE_2CORE_BETA_CLAMP_GAMMA_COMPLEX_PRIMASE",
                "DNA_POLYMERASE_CORE_BETA_CLAMP_GAMMA_COMPLEX",
                "DNA_POLYMERASE_CORE_BETA_CLAMP_PRIMASE",
            ]
        )
        dna_a_ids = []
        for mers in range(2, 8):
            dna_a_ids.extend(
                (
                    f"MG_469_{mers}MER_{mers - 1}ATP_ADP",
                    f"MG_469_{mers}MER_ATP",
                )
            )
        self.dnaAPolymerIndexs = self.getIndexs(dna_a_ids)

    def updateExternalState(
        self, deltaProteins: Any, proteinIsDegraded: Any
    ) -> np.ndarray:
        """Release affected complex-bound proteins and coordinate dependent states."""
        delta = np.asarray(deltaProteins)
        if delta.ndim != 2:
            raise ValueError(
                "deltaProteins must have molecule-by-compartment dimensions"
            )
        compartment = _require(self, "compartment")
        cytosol = int(np.asarray(compartment.cytosolIndexs).reshape(-1)[0]) - 1
        mature = _indices(self.matureIndexs)
        bound = _indices(self.boundIndexs)
        free_delta = delta[mature - 1, cytosol]
        bound_delta = delta[bound - 1, cytosol].copy()
        not_updating = np.zeros_like(delta)

        ribosome70 = _indices(self.ribosome70SIndexs)
        if ribosome70.size:
            self.ribosome.releaseRibosome(-bound_delta[ribosome70 - 1], 0, 0)
            bound_delta[ribosome70 - 1] = 0
        translation = _indices(self.translationFactorIndexs)
        if translation.size:
            bound_delta[translation - 1] = 0
        ftsz = np.concatenate(
            (_indices(self.ftsZGTPIndexs), _indices(self.ftsZGDPIndexs))
        )
        if ftsz.size:
            released = self.ftsZRing.releaseFtsZ(-bound_delta[ftsz - 1])
            not_updating[bound[ftsz - 1] - 1, cytosol] = np.asarray(
                released
            ).reshape(-1)
            bound_delta[ftsz - 1] = 0
        for protected_field, message in (
            ("dnaAPolymerIndexs", "DnaA complex not decayed"),
            ("replisomeIndexs", "DNA polymerase not decayed"),
        ):
            protected = _indices(getattr(self, protected_field))
            changed = protected[bound_delta[protected - 1] < 0]
            if changed.size:
                import warnings

                warnings.warn(message, RuntimeWarning)
                not_updating[bound[protected - 1] - 1, cytosol] = -bound_delta[
                    protected - 1
                ]
                bound_delta[protected - 1] = 0

        chromosome_changes = np.flatnonzero(bound_delta < 0)
        if chromosome_changes.size:
            chromosome = _require(self, "chromosome")
            stream = _require(self, "randStream")
            sites = chromosome.complexBoundSites
            if hasattr(sites, "find"):
                positions, proteins = sites.find()
            else:
                arr = np.asarray(sites)
                linear = np.flatnonzero(arr.ravel(order="F"))
                positions = np.column_stack(
                    np.unravel_index(linear, arr.shape, order="F")
                )
                proteins = arr[tuple(positions.T)]
                positions = positions + 1
            positions, proteins = np.asarray(positions, dtype=int), np.asarray(
                proteins
            ).reshape(-1).astype(int)
            selected_positions = []
            for idx in chromosome_changes:
                candidates = positions[proteins == idx + 1, :2]
                count = int(-bound_delta[idx])
                if len(candidates) < count:
                    raise RuntimeError("Error updating external state")
                selected, _ = stream.randomlySelectNRows(candidates, count)
                selected_positions.append(
                    np.asarray(selected, dtype=int).reshape((-1, 2))
                )
            release_positions = (
                np.concatenate(selected_positions, axis=0)
                if selected_positions
                else np.empty((0, 2), int)
            )
            _, released_complexes, _ = chromosome.setRegionProteinUnbound(
                release_positions,
                1,
                [],
                chromosome_changes + 1,
                False,
                False,
                False,
                proteinIsDegraded,
            )
            if not np.array_equal(
                -bound_delta[chromosome_changes],
                np.asarray(released_complexes).reshape(-1),
            ):
                raise RuntimeError(
                    "Chromosomally bound proteins improperly released"
                )
        rnap = _indices(self.rnaPolymeraseIndexs)
        if rnap.size and free_delta[rnap[0] - 1] and proteinIsDegraded:
            self.rnaPolymerase.degradeFreePolymerase(-free_delta[rnap[0] - 1])
        return not_updating

    def getIndexs(self, wholeCellModelIDs: Any) -> Any:
        if self.wholeCellModelIDs is None or self.matureIndexs is None:
            raise ValueError(
                "wholeCellModelIDs and matureIndexs must be initialized"
            )
        ids = np.asarray(self.wholeCellModelIDs, dtype=object).reshape(
            -1, order="F"
        )
        mature = _indices(self.matureIndexs)
        requested = np.asarray(wholeCellModelIDs, dtype=object)
        values = np.asarray(
            [
                next(
                    (
                        i + 1
                        for i, val in enumerate(ids[mature - 1])
                        if val == x
                    ),
                    0,
                )
                for x in requested.reshape(-1, order="F")
            ],
            dtype=int,
        )
        return (
            int(values[0])
            if requested.ndim == 0
            else values.reshape(requested.shape, order="F")
        )


def _indices(value: Any) -> np.ndarray:
    return np.asarray(value if value is not None else [], dtype=int).reshape(
        -1, order="F"
    )


def _require(obj: Any, field: str) -> Any:
    value = getattr(obj, field, None)
    if value is None:
        raise RuntimeError(
            f"ProteinComplex dependency {field!r} has not been assigned"
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


def _has_field(value: Any, name: str) -> bool:
    return (
        name in value if isinstance(value, Mapping) else hasattr(value, name)
    )
