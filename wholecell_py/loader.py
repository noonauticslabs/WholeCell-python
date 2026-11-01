"""MAT-file data access without pretending to reconstruct MATLAB objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import scipy.io


STATE_NAMES = (
    "CellGeometry",
    "CellMass",
    "Chromosome",
    "FtsZRing",
    "Host",
    "MetabolicReaction",
    "Metabolite",
    "Polypeptide",
    "ProteinComplex",
    "ProteinMonomer",
    "Ribosome",
    "RNAPolymerase",
    "Rna",
    "Stimulus",
    "Time",
    "Transcript",
)
PROCESS_NAMES = (
    "ChromosomeCondensation",
    "ChromosomeSegregation",
    "Cytokinesis",
    "DNADamage",
    "DNARepair",
    "DNASupercoiling",
    "FtsZPolymerization",
    "HostInteraction",
    "MacromolecularComplexation",
    "Metabolism",
    "ProteinActivation",
    "ProteinDecay",
    "ProteinFolding",
    "ProteinModification",
    "ProteinProcessingI",
    "ProteinProcessingII",
    "ProteinTranslocation",
    "Replication",
    "ReplicationInitiation",
    "RibosomeAssembly",
    "RNADecay",
    "RNAModification",
    "RNAProcessing",
    "tRNAAminoacylation",
    "TerminalOrganelleAssembly",
    "Transcription",
    "TranscriptionalRegulation",
    "Translation",
)


@dataclass(frozen=True)
class DataConfig:
    """Explicit paths and provenance for a model-data bundle."""

    model_parameters: Path
    source_commit: str | None = None

    @classmethod
    def from_extracted_data(
        cls, root: str | Path, source_commit: str | None = None
    ) -> "DataConfig":
        return cls(Path(root) / "model_parameters_plain.mat", source_commit)


@dataclass
class ModelParameters:
    """Variables as stored by SciPy; MATLAB structs remain structured arrays."""

    values: dict[str, Any]
    source_id: str
    source_commit: str | None

    def __getitem__(self, key: str) -> Any:
        return self.values[key]


def load_model_parameters(config: DataConfig) -> ModelParameters:
    """Load variables with MATLAB dimensions/types retained by ``loadmat``."""
    path = Path(config.model_parameters)
    if not path.is_file():
        raise FileNotFoundError(path)
    # Keep squeeze_me=False: singleton dimensions and MATLAB struct wrappers
    # carry meaningful shape information for downstream model translations.
    values = scipy.io.loadmat(path, squeeze_me=False, struct_as_record=False)
    values = {
        key: value for key, value in values.items() if not key.startswith("__")
    }
    return ModelParameters(
        values,
        source_id=str(path.resolve()),
        source_commit=config.source_commit,
    )


def category_registry() -> dict[str, tuple[str, ...]]:
    """List implemented source categories; names are registry metadata, not stubs."""
    return {"states": STATE_NAMES, "processes": PROCESS_NAMES}
