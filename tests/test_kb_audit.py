from pathlib import Path
from types import SimpleNamespace

from wholecell_py.kb_audit import load_kb_audit
from wholecell_py.states.metabolite import Metabolite


def test_kb_audit_decoder_preserves_records_and_incomplete_getters():
    path = (
        Path(__file__).parents[1]
        / "reference"
        / "kb_audit_export_20261101.json"
    )
    kb = load_kb_audit(path)

    assert not kb.complete
    assert len(kb.proteinMonomers) == 482
    assert len(kb.proteinComplexs) == 201
    assert len(kb.metabolites) == 722
    assert kb.proteinMonomers[0]["wholeCellModelID"] == "MG_001_MONOMER"
    assert kb.errors[0]["record_set"] == "proteinMonomers"
    assert kb.errors[0]["record_index"] == 132
    assert kb.errors[0]["stack"][0]["name"] == "ProteinMonomer.get.sequence"
    assert kb.source_sha256 == (
        "55402852c350c05d2c19fe23042a9d64124417789a64ef062329380220c5d151"
    )


def test_authentic_kb_metabolite_initializer_accepts_exported_records():
    path = (
        Path(__file__).parents[1]
        / "reference"
        / "kb_audit_export_20261101.json"
    )
    kb = load_kb_audit(path)
    state = Metabolite("State_Metabolite")
    simulation = SimpleNamespace(
        verbosity=0, seed=1, geometry=SimpleNamespace(chamberVolume=1.0)
    )

    state.initializeConstants(kb, simulation)

    assert state.molecularWeights.shape == (722, 1)
    assert state.setCounts.shape == (83, 6)
