"""Metabolite state translated from WholeCell's ``Metabolite`` class."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from wholecell_py.molecule_count_state import MoleculeCountState, _field
from wholecell_py.sparsemat import SparseMat


class Metabolite(MoleculeCountState):
    optionNames = ("verbosity", "seed")
    fixedConstantNames = (
        "molecularWeights",
        "baseCounts",
        "lengths",
        "halfLives",
        "compartments",
        "meanNTPConcentration",
        "meanNDPConcentration",
        "meanNMPConcentration",
    )
    fittedConstantNames = (
        "biomassComposition",
        "biomassProduction",
        "byproducts",
    )
    stateNames = (
        "counts",
        "processRequirements",
        "processAllocations",
        "processUsages",
    )
    dependentStateNames: tuple[str, ...] = ()

    _metabolite_snapshot_names = (
        "hydrophobicIndexs",
        "aminoAcidIndexs",
        "dnmpIndexs",
        "dndpIndexs",
        "dntpIndexs",
        "nmpIndexs",
        "ndpIndexs",
        "ntpIndexs",
        "ampIndexs",
        "adpIndexs",
        "atpIndexs",
        "phosphateIndexs",
        "diphosphateIndexs",
        "waterIndexs",
        "hydrogenIndexs",
        "dr5pIndexs",
        "unmodifiedBaseIndexs",
        "m6ADIndexs",
        "lipidIndexs",
        "polyamineIndexs",
        "carbohydrateIndexs",
        "vitaminIndexs",
        "ionIndexs",
        "antibioticIndexs",
        "setCounts",
        "experimentalBiomassComposition",
        "experimentalBiomassCompositionWeightFractions",
        "experimentalBiomassCompositionMolFractions",
        "biomassComposition",
        "biomassProduction",
        "byproducts",
        "processBiomassProduction",
        "processByproduct",
        "meanNTPConcentration",
        "meanNDPConcentration",
        "meanNMPConcentration",
        "processWholeCellModelIDs",
        "processRequirements",
        "processAllocations",
        "processUsages",
    )
    _snapshot_names = tuple(
        dict.fromkeys(
            MoleculeCountState._snapshot_names + _metabolite_snapshot_names
        )
    )

    def __init__(
        self,
        wholeCellModelID: Any = None,
        name: Any = None,
        snapshot: Mapping[str, Any] | None = None,
        **attributes: Any,
    ) -> None:
        # Merge before delegating because the common base restores snapshots
        # after its positional ID/name values. Explicit constructor arguments
        # and keyword attributes therefore take precedence over snapshot data.
        restored = dict(snapshot or {})
        restored.update(attributes)
        if wholeCellModelID is not None:
            restored["wholeCellModelID"] = wholeCellModelID
        if name is not None:
            restored["name"] = name
        super().__init__(wholeCellModelID, name, restored)
        # Keep absent KB/snapshot values explicitly unavailable.
        for field in self._metabolite_snapshot_names:
            if not hasattr(self, field):
                setattr(self, field, None)

    def initializeConstants(
        self, knowledgeBase: Any, simulation: Any = None
    ) -> None:
        super().initializeConstants(knowledgeBase, simulation)
        metabolites = list(_field(knowledgeBase, "metabolites"))
        self.wholeCellModelIDs = [
            _field(m, "wholeCellModelID") for m in metabolites
        ]
        self.names = [_field(m, "name") for m in metabolites]
        self.molecularWeights = np.asarray(
            [_field(m, "molecularWeight") for m in metabolites]
        ).reshape(-1, 1)

        self.hydrophobicIndexs = _find(
            _field(m, "hydrophobic", False) for m in metabolites
        )
        group_ids = {
            "aminoAcidIndexs": (
                "ALA",
                "ARG",
                "ASN",
                "ASP",
                "CYS",
                "GLN",
                "GLU",
                "GLY",
                "HIS",
                "ILE",
                "LEU",
                "LYS",
                "MET",
                "PHE",
                "PRO",
                "SER",
                "THR",
                "TRP",
                "TYR",
                "VAL",
                "FMET",
            ),
            "dnmpIndexs": ("DAMP", "DCMP", "DGMP", "DTMP"),
            "dndpIndexs": ("DADP", "DCDP", "DGDP", "DTDP"),
            "dntpIndexs": ("DATP", "DCTP", "DGTP", "DTTP"),
            "nmpIndexs": ("AMP", "CMP", "GMP", "UMP"),
            "ndpIndexs": ("ADP", "CDP", "GDP", "UDP"),
            "ntpIndexs": ("ATP", "CTP", "GTP", "UTP"),
            "ampIndexs": ("AMP",),
            "adpIndexs": ("ADP",),
            "atpIndexs": ("ATP",),
            "phosphateIndexs": ("PI",),
            "diphosphateIndexs": ("PPI",),
            "waterIndexs": ("H2O",),
            "hydrogenIndexs": ("H",),
            "dr5pIndexs": ("DR5P",),
            "unmodifiedBaseIndexs": ("AD", "CSN", "GN", "THY"),
            "m6ADIndexs": ("m6AD",),
        }
        for attr, ids in group_ids.items():
            setattr(
                self,
                attr,
                np.asarray(self.getIndexs(ids), dtype=int).reshape(-1, 1),
            )
        for attr, category in (
            ("lipidIndexs", "lipid"),
            ("polyamineIndexs", "polyamine"),
            ("carbohydrateIndexs", "carbohydrate"),
            ("vitaminIndexs", "vitamin"),
            ("ionIndexs", "ion"),
            ("antibioticIndexs", "antibiotic"),
        ):
            setattr(
                self,
                attr,
                _find(
                    _field(m, "category", None) == category
                    for m in metabolites
                ),
            )

        self.setCounts = self._media_conditions(
            metabolites, knowledgeBase, simulation
        )
        experimental = -np.asarray(_field(knowledgeBase, "biomassComposition"))
        self.experimentalBiomassComposition = experimental
        mol = np.sum(experimental, axis=1, keepdims=True) / np.sum(
            experimental
        )
        self.experimentalBiomassCompositionMolFractions = mol
        self.experimentalBiomassCompositionWeightFractions = (
            mol * self.molecularWeights / (mol.T @ self.molecularWeights)
        )
        self.processWholeCellModelIDs = [
            _field(process, "wholeCellModelID")
            for process in _field(knowledgeBase, "processes")
        ]

    def _media_conditions(
        self, metabolites: list[Any], knowledgeBase: Any, simulation: Any
    ) -> np.ndarray:
        rows: list[list[float]] = []
        n_compartments = int(_field(knowledgeBase, "numCompartments"))
        for metabolite in metabolites:
            media = _field(metabolite, "mediaCompartments", [])
            if media is None:
                continue
            for j, compartment in enumerate(media):
                metabolite_index = _field(metabolite, "idx")
                compartment_index = _field(compartment, "idx")
                value = _field(metabolite, "mediaConcentrations")[j]
                initial = _field(metabolite, "mediaInitialTimes")[j]
                final = _field(metabolite, "mediaFinalTimes")[j]
                linear = (
                    (metabolite_index - 1)
                    + (compartment_index - 1) * len(metabolites)
                    + 1
                )
                rows.append(
                    [
                        metabolite_index,
                        compartment_index,
                        value,
                        initial,
                        final,
                        linear,
                    ]
                )
        if not rows:
            return np.zeros((0, 6))
        result = np.asarray(rows, dtype=float)
        if simulation is None:
            raise ValueError(
                "simulation is required to convert media concentrations to molecule counts"
            )
        geometry = (
            simulation.state("Geometry")
            if callable(getattr(simulation, "state", None))
            else _field(simulation, "geometry")
        )
        chamber_volume = _field(geometry, "chamberVolume")
        molecules = (
            result[:, 2] * MoleculeCountState.nAvogadro / 1000 * chamber_volume
        )
        result[:, 2] = np.copysign(
            np.floor(np.abs(molecules) + 0.5), molecules
        )
        return result

    def allocateMemory(self, numTimePoints: int) -> None:
        super().allocateMemory(numTimePoints)
        if self.processWholeCellModelIDs is None:
            raise ValueError(
                "processWholeCellModelIDs must be initialized before allocating process state"
            )
        if self.compartment is None:
            raise ValueError(
                "compartment reference must be initialized before allocating process state"
            )
        shape = (
            len(self.wholeCellModelIDs) * int(self.compartment.count),
            len(self.processWholeCellModelIDs),
            int(numTimePoints),
        )
        self.processRequirements = SparseMat([], [], shape)
        self.processAllocations = SparseMat([], [], shape)
        self.processUsages = SparseMat([], [], shape)

    @property
    def metabolismProduction(self) -> Any:
        if self.biomassProduction is None or self.byproducts is None:
            raise ValueError(
                "biomassProduction and byproducts must be initialized"
            )
        return np.asarray(self.biomassProduction) - np.asarray(self.byproducts)

    @property
    def experimentalNMPComposition(self) -> np.ndarray:
        return _normalize(
            self._required("experimentalBiomassCompositionMolFractions")[
                np.asarray(self._required("nmpIndexs"), dtype=int).reshape(-1)
                - 1
            ]
        )

    @property
    def experimentalAAComposition(self) -> np.ndarray:
        return _normalize(
            self._required("experimentalBiomassCompositionMolFractions")[
                np.asarray(
                    self._required("aminoAcidIndexs"), dtype=int
                ).reshape(-1)
                - 1
            ]
        )

    @property
    def nmpComposition(self) -> np.ndarray:
        return _normalize(
            self._required("biomassComposition")[
                np.asarray(self._required("nmpIndexs"), dtype=int).reshape(-1)
                - 1,
                :,
            ]
        )

    @property
    def aaComposition(self) -> np.ndarray:
        values = np.sum(
            self._required("biomassComposition")[
                np.asarray(
                    self._required("aminoAcidIndexs"), dtype=int
                ).reshape(-1)
                - 1,
                :,
            ],
            axis=1,
        )
        return _normalize(values).reshape(-1, 1)

    @property
    def wetWeight(self) -> np.ndarray:
        counts = self._required("counts")
        water = np.asarray(self._required("waterIndexs"), dtype=int).reshape(
            -1
        )
        weights = np.asarray(self._required("molecularWeights")).reshape(-1)
        if len(water) != 1 or water[0] < 1 or water[0] > len(weights):
            raise ValueError("waterIndexs must identify one valid metabolite")
        return (
            np.asarray(counts)[water[0] - 1 : water[0], ...]
            * weights[water[0] - 1]
            / MoleculeCountState.nAvogadro
        )

    def calcDryWeight(self) -> np.ndarray:
        return np.asarray(super().calcDryWeight()) - self.wetWeight

    def _required(self, name: str) -> np.ndarray:
        value = getattr(self, name, None)
        if value is None:
            raise ValueError(f"{name} must be initialized")
        return np.asarray(value)


def _find(values: Any) -> np.ndarray:
    return (
        np.flatnonzero(np.asarray(list(values), dtype=bool))
        .astype(int)
        .reshape(-1, 1)
        + 1
    )


def _normalize(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    total = np.sum(result)
    with np.errstate(divide="ignore", invalid="ignore"):
        return result / total
