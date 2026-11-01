import numpy as np

from wholecell_py.states.polypeptide import Polypeptide


def test_snapshot_allocation_and_initialize():
    state = Polypeptide(
        snapshot={
            "monomerLengths": np.array([4]),
            "boundMRNAs": np.ones((2, 1, 3)),
        }
    )
    assert state.monomerLengths.tolist() == [4]
    assert state.boundMRNAs.flags.f_contiguous
    state.allocateMemory(5)
    assert state.boundMRNAs.shape == (0, 1, 5)
    assert state.abortedPolypeptides.shape == (0, 3, 5)
    state.initialize()
    assert state.abortedPolypeptides.shape == (0, 3)


def test_dependent_state_properties_with_matlab_indices():
    state = Polypeptide(
        monomerLengths=[4, 3],
        monomerAASequences=["ACDE", "FGH"],
        proteolysisTagAASequence="WY",
        aminoAcidMolecularWeights=np.full(20, 100.0),
        boundMRNAs=np.array([[[1]], [[2]], [[0]]]),
        nascentMonomerLengths=np.array([[[2]], [[1]], [[0]]]),
        proteolysisTagLengths=np.array([[[1]], [[0]], [[0]]]),
        abortedPolypeptides=np.array([[2, 2, 1], [0, 0, 0]]),
    )
    assert state.mRNAMaxNascentMonomerLength[:, 0, 0].tolist() == [2, 1]
    assert state.abortedSequences == ["FGW", ""]
    assert state.abortedSequenceLengths.ravel().tolist() == [3, 0]
    assert state.totalBaseCounts.sum() == 7
    assert state.dryWeight > 0


def test_mass_and_base_counts_use_matlab_first_time_slice():
    state = Polypeptide(
        monomerLengths=[1, 1],
        monomerAASequences=["A", "W"],
        proteolysisTagAASequence="Y",
        aminoAcidMolecularWeights=np.full(20, 100.0),
        boundMRNAs=np.array([[[1, 2, 1]]]),
        nascentMonomerLengths=np.array([[[1, 1, 1]]]),
        proteolysisTagLengths=np.zeros((1, 1, 3), dtype=int),
        abortedPolypeptides=np.array(
            [[[1, 2, 1], [1, 1, 1], [0, 0, 0]]], dtype=int
        ),
    )

    # MATLAB's mass/count getters use two-subscript reads and therefore read
    # timepoint one; only the explicitly time-looping max-length getter spans
    # every allocated timepoint.
    assert state._nascent_sequences() == ["A"]
    assert state.abortedSequences == ["A"]
    assert state.abortedSequenceLengths.ravel().tolist() == [1]
    counts = state.totalBaseCounts.ravel()
    assert counts[0] == 2  # Nascent A and aborted A; later slices are ignored.
    assert state.mRNAMaxNascentMonomerLength[0, 0, :].tolist() == [1, 0, 1]
    assert state.mRNAMaxNascentMonomerLength[1, 0, :].tolist() == [0, 1, 0]


def test_kb_initialization_derives_monomer_and_tmRNA_tag_sequences():
    metabolites = [
        {"wholeCellModelID": f"AA{i}", "molecularWeight": float(100 + i)}
        for i in range(1, 22)
    ]
    tm_gene = {
        "wholeCellModelID": "MG_0004",
        "aminoAcid": {"idx": 1},
        "features": [{"type": "tmRNA proteolysis tag", "sequence": "ATGCCC"}],
    }
    protein_gene = {
        "wholeCellModelID": "MG_PROT",
        "dnaSequence": "CCCATGAAACCC",
    }
    kb = {
        "states": [{"wholeCellModelID": "Polypeptide", "parameters": []}],
        "aminoAcidIndexs": np.arange(1, 22),
        "metabolites": metabolites,
        "proteinMonomers": [
            {
                "wholeCellModelID": "MG_PROT_MONOMER",
                "sequenceLength": 1,
                "gene": protein_gene,
                "sequence": "M",
            }
        ],
        "genes": [tm_gene, protein_gene],
        "geneticCodeCodons": ["GGG", "CAT", "AAA", "CCC"],
        "geneticCodeTRNAs": [1, 2, 3, 4],
        "geneticCodeAminoAcids": ["GLY", "HIS", "LYS", "PRO"],
        "geneticCodeStartCodons": ["ATG"],
        "geneticCodeStartTRNAs": [5],
    }
    state = Polypeptide("Polypeptide", "Polypeptides")

    state.initializeConstants(kb)

    assert state.monomerWholeCellModelIDs == ["MG_PROT_MONOMER"]
    assert state.monomerLengths.ravel().tolist() == [1]
    np.testing.assert_array_equal(
        state.monomerTRNASequences[0].ravel(), [5, 3, 4]
    )
    assert state.monomerAASequences == ["M"]
    assert state.proteolysisTagTRNASequence.ravel().tolist() == [0, 1, 2]
    assert state.proteolysisTagLength == 3
    assert state.proteolysisTagAASequence == "AGH"
    aa_weights = np.arange(101.0, 122.0)
    expected_counts = np.array(
        [1 if aa in "AGH" else 0 for aa in "ARNDCQEGHILKMFPSTWYV"] + [0]
    )
    expected_mass = expected_counts @ aa_weights - (1.0079 + 2 * 15.9994) * 2
    assert state.proteolysisTagMolecularWeight == expected_mass


def test_snapshot_character_arrays_and_matlab_dry_weight_constants():
    # MATLAB snapshots encode character vectors as arrays, rather than Python
    # strings. The source weight order is ProteinMonomer.bases(1:20).
    aa = np.array([list("ACDM")], dtype="U1")
    tag = np.array([list("WY")], dtype="U1")
    state = Polypeptide(
        monomerAASequences=[aa],
        proteolysisTagAASequence=tag,
        aminoAcidMolecularWeights=np.arange(1.0, 22.0),
        boundMRNAs=np.array([[[1]]]),
        nascentMonomerLengths=np.array([[[3]]]),
        proteolysisTagLengths=np.array([[[1]]]),
        abortedPolypeptides=np.array([[1, 1, 1]]),
    )
    assert state._nascent_sequences() == ["ACDW"]
    assert state.abortedSequences == ["AW"]
    # A,C,D,W,A,W; molecular weight indices are A=1,C=5,D=4,W=18.
    sequence_mass = 1 + 5 + 4 + 18 + 1 + 18
    water = 1.0079 + 2 * 15.9994
    expected = (sequence_mass - (6 - 2) * water) / 6.0221417930e23
    assert np.isclose(state.dryWeight, expected, rtol=1e-14, atol=0)
    counts = state.totalBaseCounts.ravel()
    assert counts[12] == -2  # two N-terminal M residues become formyl-M
    assert counts[20] == 2


def test_real_snapshot_dry_weight_matches_matlab_reference():
    import json
    from pathlib import Path
    import numpy as np
    from wholecell_py.snapshot import load_snapshot
    from wholecell_py.states.polypeptide import Polypeptide

    root = Path(__file__).resolve().parents[1]
    graph = load_snapshot(root / "reference/snapshot_t0.json")
    state = Polypeptide(snapshot=graph.states["Polypeptide"].fields)
    reference = json.loads(
        (root / "reference/state_weights_t0.json").read_text()
    )
    np.testing.assert_allclose(
        state.dryWeight, reference["Polypeptide"]["data"], rtol=1e-14, atol=0
    )
