"""Cell mass state translated from WholeCell's ``CellMass`` state."""

from __future__ import annotations

from typing import Any, Mapping
import numpy as np


class Mass:
    optionNames = ("verbosity", "seed")
    fixedConstantNames = (
        "fractionWetWeight",
        "initialBiomassConcentration",
        "cellInitialDryWeight",
        "initialFractionNTPsInRNAs",
        "initialFractionAAsInMonomers",
        "timeAveragedCellWeight",
        "cellInitialMassVariation",
    )
    fittedConstantNames = (
        "dryWeightFractionDNA",
        "dryWeightFractionRNA",
        "dryWeightFractionProtein",
        "dryWeightFractionLipid",
        "dryWeightFractionPolyamine",
        "dryWeightFractionCarbohydrate",
        "dryWeightFractionVitamin",
        "dryWeightFractionIon",
        "dryWeightFractionNucleotide",
    )
    stateNames = ()
    dependentStateNames = (
        "total",
        "cell",
        "cellDry",
        "media",
        "waterWt",
        "metaboliteWt",
        "dnaWt",
        "rnaWt",
        "proteinWt",
    )
    dryWeight = 0
    _snapshot_names = (
        "wholeCellModelID",
        "name",
        *fixedConstantNames,
        *fittedConstantNames,
        *dependentStateNames,
    )

    def __init__(
        self,
        wholeCellModelID: Any = None,
        name: Any = None,
        snapshot: Mapping[str, Any] | None = None,
        **attributes: Any,
    ) -> None:
        self.wholeCellModelID, self.name = wholeCellModelID, name
        for key in (
            *self.fixedConstantNames,
            *self.fittedConstantNames,
            *self.dependentStateNames,
        ):
            setattr(self, key, None)
        self.compartment = self.states = None
        self.metabolite = self.rna = self.monomer = self.complex = (
            self.chromosome
        ) = None
        restored = dict(snapshot or {})
        restored.update(attributes)
        for key, value in restored.items():
            setattr(
                self,
                key,
                (
                    np.array(value, copy=True)
                    if isinstance(value, np.ndarray)
                    else value
                ),
            )

    def storeObjectReferences(self, simulation: Any) -> None:
        self.compartment = simulation.compartment
        self.states = simulation.states
        self.metabolite = simulation.state("Metabolite")
        self.rna = simulation.state("Rna")
        self.monomer = simulation.state("ProteinMonomer")
        self.complex = simulation.state("ProteinComplex")
        self.chromosome = simulation.state("Chromosome")

    def allocateMemory(self, numTimePoints: int) -> None:
        # The source state has no independent time-series storage.
        return None

    def initialize(self) -> None:
        self.calcMass()

    def calcMass(self) -> None:
        """Recompute mass components from live state getters, preserving (1,C,T) shape."""
        original_water = np.asarray(self.metabolite.wetWeight)

        def shaped(value):
            value = np.asarray(value)
            if value.ndim == 0:
                return value.reshape(1, 1, 1)
            if value.ndim == 2:
                return value[:, :, None]
            if value.ndim != 3:
                raise ValueError(
                    "mass components require scalar, 2D, or 3D MATLAB arrays"
                )
            return value

        water = shaped(original_water)
        self.waterWt = water
        self.total = np.array(water, copy=True)
        self.proteinWt = np.zeros_like(
            water, dtype=np.result_type(water, float)
        )
        self.metaboliteWt = self.dnaWt = self.rnaWt = None
        states = (
            self.states.values()
            if isinstance(self.states, dict)
            else np.asarray(self.states, dtype=object).ravel(order="F")
        )
        for state in states:
            dry = shaped(state.dryWeight)
            if dry.shape[1] == 1:
                idx = (
                    np.asarray(
                        self.compartment.cytosolIndexs, dtype=int
                    ).reshape(-1)
                    - 1
                )
                self.total[:, idx, :] += dry
            else:
                self.total += dry
            if state is self.metabolite:
                self.metaboliteWt = dry
            elif state is self.chromosome:
                self.dnaWt = np.zeros_like(
                    water, dtype=np.result_type(water, dry)
                )
                idx = (
                    np.asarray(
                        self.compartment.cytosolIndexs, dtype=int
                    ).reshape(-1)
                    - 1
                )
                self.dnaWt[:, idx, :] = dry
            elif state is self.rna:
                self.rnaWt = dry
            elif state is self.monomer or state is self.complex:
                self.proteinWt += dry
        self.cell = np.array(self.total, copy=True)
        ext = (
            np.asarray(
                self.compartment.extracellularIndexs, dtype=int
            ).reshape(-1)
            - 1
        )
        self.cell[:, ext, :] = 0
        self.cellDry = np.zeros_like(self.cell)
        cellular = (
            np.asarray(self.compartment.cellularIndexs, dtype=int).reshape(-1)
            - 1
        )
        self.cellDry[:, cellular, :] = (
            self.cell[:, cellular, :] - self.waterWt[:, cellular, :]
        )
        self.media = np.array(self.total, copy=True)
        self.media[:, cellular, :] = 0
        if original_water.ndim == 2:
            for name in Mass.dependentStateNames:
                value = getattr(self, name)
                if isinstance(value, np.ndarray) and value.ndim == 3:
                    setattr(self, name, value[:, :, 0])
