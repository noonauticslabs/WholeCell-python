"""Explicit registry of implemented classes; missing classes remain visible.

This registry alone is not a runnable whole-cell model. ``load_snapshot`` with
``strict_runtime=True`` rejects unimplemented classes rather than substituting
generic data containers for executable biology.
"""

from importlib import import_module


STATE_MODULES = {
    "CellMass": ("mass", "Mass"),
    "CellGeometry": ("geometry", "Geometry"),
    "Time": ("time", "Time"),
    "Stimulus": ("stimulus", "Stimulus"),
    "MetabolicReaction": ("metabolic_reaction", "MetabolicReaction"),
    "Host": ("host", "Host"),
    "FtsZRing": ("ftsz_ring", "FtsZRing"),
    "Polypeptide": ("polypeptide", "Polypeptide"),
    "RNAPolymerase": ("rna_polymerase", "RNAPolymerase"),
    "Ribosome": ("ribosome", "Ribosome"),
    "Metabolite": ("metabolite", "Metabolite"),
    "ProteinMonomer": ("protein_monomer", "ProteinMonomer"),
    "ProteinComplex": ("protein_complex", "ProteinComplex"),
    "Rna": ("rna", "Rna"),
    "Transcript": ("transcript", "Transcript"),
}
PROCESS_MODULES = {
    "DNARepair": "dna_repair",
    "DNADamage": "dna_damage",
    "ProteinDecay": "protein_decay",
    "ReplicationInitiation": "replication_initiation",
    "Replication": "replication",
    "Transcription": "transcription",
    "Translation": "translation",
    "ProteinActivation": "protein_activation",
    "ChromosomeSegregation": "chromosome_segregation",
    "Cytokinesis": "cytokinesis",
    "HostInteraction": "host_interaction",
    "ProteinProcessingI": "protein_processing_i",
    "ProteinProcessingII": "protein_processing_ii",
    "ProteinFolding": "protein_folding",
    "ProteinModification": "protein_modification",
    "ProteinTranslocation": "protein_translocation",
    "Metabolism": "metabolism",
    "RibosomeAssembly": "ribosome_assembly",
    "RNAModification": "rna_modification",
    "RNAProcessing": "rna_processing",
    "RNADecay": "rna_decay",
    "FtsZPolymerization": "ftsz_polymerization",
    "ChromosomeCondensation": "chromosome_condensation",
    "DNASupercoiling": "dna_supercoiling",
    "TerminalOrganelleAssembly": "terminal_organelle_assembly",
    "MacromolecularComplexation": "macromolecular_complexation",
    "TranscriptionalRegulation": "transcriptional_regulation",
    "tRNAAminoacylation": "trna_aminoacylation",
}


def implemented_registry():
    from .side_effect import SimulationStateSideEffect
    from .side_effect_item import SimulationStateSideEffectItem
    from .gene import Gene
    from .compartment import Compartment

    registry = {
        "edu.stanford.covert.cell.sim.constant.Gene": Gene,
        "edu.stanford.covert.cell.sim.constant.Compartment": Compartment,
        "edu.stanford.covert.cell.sim.SimulationStateSideEffect": SimulationStateSideEffect,
        "edu.stanford.covert.cell.sim.SimulationStateSideEffectItem": SimulationStateSideEffectItem,
    }
    for source, (module, name) in STATE_MODULES.items():
        registry["edu.stanford.covert.cell.sim.state." + source] = getattr(
            import_module("wholecell_py.states." + module), name
        )
    for name, module in PROCESS_MODULES.items():
        registry["edu.stanford.covert.cell.sim.process." + name] = getattr(
            import_module("wholecell_py.processes." + module),
            "TRNAAminoacylation" if name == "tRNAAminoacylation" else name,
        )
    return registry
