from types import SimpleNamespace

import numpy as np
import pytest

from wholecell_py.states.polypeptide import Polypeptide
from wholecell_py.states.transcript import Transcript
from wholecell_py.states.metabolite import Metabolite, _normalize


@pytest.mark.parametrize("kind", ["protein", "rna"])
def test_nonzero_aborted_id_counts_in_source_mass_even_at_zero_length(kind):
    empty = np.empty((0, 1))
    if kind == "protein":
        state = Polypeptide(
            boundMRNAs=empty,
            nascentMonomerLengths=empty,
            proteolysisTagLengths=empty,
            monomerAASequences=["M"],
            proteolysisTagAASequence="A",
            aminoAcidMolecularWeights=np.ones(20),
            abortedPolypeptides=np.array([[1, 0, 0], [0, 0, 0]]),
        )
        bond_mass = 1.0079 + 2 * 15.9994
    else:
        state = Transcript(
            boundTranscriptionUnits=empty,
            boundTranscriptProgress=empty,
            transcriptionUnitSequences=["A"],
            nmpMolecularWeights=np.ones(4),
            abortedTranscripts=np.array([[1, 0], [0, 0]]),
        )
        bond_mass = 1.0079 + 15.9994
    # Preserve the source formula, including its treatment of empty aborted rows.
    assert state.dryWeight == pytest.approx(bond_mass / 6.0221417930e23, abs=0)


def test_media_half_integer_counts_round_away_from_zero():
    state = Metabolite()
    metabolite = dict(
        idx=1,
        mediaCompartments=[dict(idx=1), dict(idx=2)],
        mediaConcentrations=[2.5, -2.5],
        mediaInitialTimes=[0, 0],
        mediaFinalTimes=[1, 1],
    )
    simulation = SimpleNamespace(
        state=lambda _: SimpleNamespace(chamberVolume=1.0)
    )
    from unittest.mock import patch

    # Isolate rounding from inexact units conversion.
    with (
        patch.object(Metabolite, "nAvogadro", 1000.0),
        patch(
            "wholecell_py.states.metabolite.MoleculeCountState.nAvogadro",
            1000.0,
        ),
    ):
        got = state._media_conditions(
            [metabolite], dict(numCompartments=2), simulation
        )
    np.testing.assert_array_equal(got[:, 2], [3, -3])


def test_zero_composition_preserves_matlab_nan_result():
    assert np.isnan(_normalize(np.zeros((2, 1)))).all()


def test_empty_aborted_peptides_follow_source_formyl_methionine_conversion():
    empty = np.empty((0, 1))
    state = Polypeptide(
        boundMRNAs=empty,
        nascentMonomerLengths=empty,
        proteolysisTagLengths=empty,
        monomerAASequences=["M"],
        proteolysisTagAASequence="A",
        abortedPolypeptides=np.array([[1, 0, 0], [0, 0, 0]]),
    )
    # ProteinMonomer.computeBaseCount subtracts M/adds fM unconditionally
    # when its flag is true; Polypeptide calls it for EVERY aborted row.
    counts = state.totalBaseCounts.ravel()
    assert counts[12] == -2
    assert counts[20] == 2
    assert counts.sum() == 0
