"""Polypeptide translation state from WholeCell's ``Polypeptide``."""

from __future__ import annotations

from typing import Any, Mapping
import numpy as np

from wholecell_py.molecule_count_state import MoleculeCountState, _field


class Polypeptide:
    optionNames = ("verbosity", "seed")
    fixedConstantNames = (
        "monomerLengths",
        "monomerTRNASequences",
        "monomerAASequences",
        "proteolysisTagLength",
        "proteolysisTagMolecularWeight",
        "proteolysisTagTRNASequence",
        "proteolysisTagAASequence",
    )
    fittedConstantNames: tuple[str, ...] = ()
    stateNames = (
        "boundMRNAs",
        "nascentMonomerLengths",
        "proteolysisTagLengths",
        "abortedPolypeptides",
    )
    dependentStateNames: tuple[str, ...] = ()
    _snapshot_names = (
        "wholeCellModelID",
        "name",
        "aminoAcidMolecularWeights",
        "monomerWholeCellModelIDs",
        "monomerLengths",
        "monomerTRNASequences",
        "monomerAASequences",
        "proteolysisTagLength",
        "proteolysisTagMolecularWeight",
        "proteolysisTagTRNASequence",
        "proteolysisTagAASequence",
        "boundMRNAs",
        "nascentMonomerLengths",
        "proteolysisTagLengths",
        "abortedPolypeptides",
    )

    def __init__(
        self,
        wholeCellModelID: Any = None,
        name: Any = None,
        snapshot: Mapping[str, Any] | None = None,
        **attributes: Any,
    ) -> None:
        self.wholeCellModelID, self.name = wholeCellModelID, name
        self.verbosity, self.seed = 0, None
        for key in self._snapshot_names[2:]:
            setattr(self, key, None)
        restored = dict(snapshot or {})
        restored.update(attributes)
        for key, value in restored.items():
            setattr(
                self,
                key,
                (
                    np.array(value, copy=True, order="F")
                    if isinstance(value, np.ndarray)
                    else value
                ),
            )

    def storeObjectReferences(self, *args: Any) -> None:
        return None

    def initializeConstants(
        self, knowledgeBase: Any, simulation: Any = None
    ) -> None:
        MoleculeCountState.initializeConstants(self, knowledgeBase, simulation)
        aa_indices = _field(knowledgeBase, "aminoAcidIndexs")
        metabolites = list(_field(knowledgeBase, "metabolites"))
        self.aminoAcidMolecularWeights = np.asarray(
            [
                _field(metabolites[int(index) - 1], "molecularWeight")
                for index in np.asarray(aa_indices, dtype=int).reshape(-1)
            ],
            dtype=float,
        ).reshape(-1, 1)

        monomers = list(_field(knowledgeBase, "proteinMonomers"))
        self.monomerWholeCellModelIDs = [
            _field(record, "wholeCellModelID") for record in monomers
        ]
        self.monomerLengths = np.asarray(
            [_field(record, "sequenceLength") for record in monomers],
            dtype=int,
        ).reshape(-1, 1)
        cached_trna = _field(
            knowledgeBase, "proteinMonomerTRNASequences", None
        )
        if cached_trna is not None and np.size(cached_trna):
            self.monomerTRNASequences = [
                np.asarray(value, dtype=int).reshape(-1, 1)
                for value in cached_trna
            ]
        else:
            codons = _cell_strings(_field(knowledgeBase, "geneticCodeCodons"))
            trna_indices = np.asarray(
                _field(knowledgeBase, "geneticCodeTRNAs"), dtype=int
            ).reshape(-1)
            code = dict(zip(codons, trna_indices))
            start_code = dict(
                zip(
                    _cell_strings(
                        _field(knowledgeBase, "geneticCodeStartCodons")
                    ),
                    np.asarray(
                        _field(knowledgeBase, "geneticCodeStartTRNAs"),
                        dtype=int,
                    ).reshape(-1),
                )
            )
            sequences = []
            for record in monomers:
                dna = _field(_field(record, "gene"), "dnaSequence")
                sequences.append(
                    _translate_dna_to_trna(
                        _sequence(dna), code, True, start_code
                    )
                )
            self.monomerTRNASequences = sequences
        self.monomerAASequences = [
            _sequence(_field(record, "sequence")) for record in monomers
        ]

        tm_gene = next(
            (
                gene
                for gene in _field(knowledgeBase, "genes")
                if _as_text(_field(gene, "wholeCellModelID")) == "MG_0004"
            ),
            None,
        )
        if tm_gene is None:
            raise ValueError("knowledgeBase has no MG_0004 tmRNA gene")
        tag = next(
            (
                feature
                for feature in _field(tm_gene, "features")
                if _as_text(_field(feature, "type")) == "tmRNA proteolysis tag"
            ),
            None,
        )
        if tag is None:
            raise ValueError("MG_0004 has no tmRNA proteolysis tag feature")
        tag_dna = _reverse_complement(_sequence(_field(tag, "sequence")))
        codons = _cell_strings(_field(knowledgeBase, "geneticCodeCodons"))
        trna_code = dict(
            zip(
                codons,
                np.asarray(
                    _field(knowledgeBase, "geneticCodeTRNAs"), dtype=int
                ).reshape(-1),
            )
        )
        tag_trna = _translate_dna_to_trna(tag_dna, trna_code, False)
        self.proteolysisTagTRNASequence = np.r_[0, tag_trna].reshape(-1, 1)
        self.proteolysisTagLength = len(self.proteolysisTagTRNASequence)

        amino_acid_idx = int(_field(_field(tm_gene, "aminoAcid"), "idx"))
        aa_indices_flat = np.asarray(aa_indices, dtype=int).reshape(-1)
        aa_position = np.flatnonzero(aa_indices_flat == amino_acid_idx)
        if not aa_position.size:
            raise ValueError(
                "MG_0004 amino acid is absent from aminoAcidIndexs"
            )
        prefix = "ARNDCQEGHILKMFPSTWYV"[int(aa_position[0])]
        self.proteolysisTagAASequence = prefix + _translate_dna_to_amino_acids(
            tag_dna,
            codons,
            _cell_strings(_field(knowledgeBase, "geneticCodeAminoAcids")),
        )
        counts = _aa_counts(self.proteolysisTagAASequence)
        self.proteolysisTagMolecularWeight = float(
            counts @ self.aminoAcidMolecularWeights.reshape(-1)
            - (1.0079 + 2 * 15.9994) * max(0, self.proteolysisTagLength - 1)
        )

    def allocateMemory(self, numTimePoints: int) -> None:
        if (
            not isinstance(numTimePoints, (int, np.integer))
            or numTimePoints < 0
        ):
            raise ValueError("numTimePoints must be a non-negative integer")
        shape = (0, 1, int(numTimePoints))
        self.boundMRNAs = np.zeros(shape, order="F")
        self.nascentMonomerLengths = np.zeros(shape, order="F")
        self.proteolysisTagLengths = np.zeros(shape, order="F")
        self.abortedPolypeptides = np.zeros(
            (0, 3, int(numTimePoints)), order="F"
        )

    def initialize(self) -> None:
        self.abortedPolypeptides = np.zeros((0, 3), order="F")

    @property
    def totalBaseCounts(self) -> np.ndarray:
        result = np.zeros(21)
        for seq in self._nascent_sequences():
            if seq:
                result += _aa_counts(seq, n_terminal_formyl_methionine=True)
        for seq in self.abortedSequences:
            result += _aa_counts(seq, n_terminal_formyl_methionine=True)
        return result.reshape((1, 21))

    @property
    def mRNAMaxNascentMonomerLength(self) -> np.ndarray:
        bound = np.asarray(self.boundMRNAs)
        lengths = np.asarray(self.nascentMonomerLengths)
        out = np.zeros(
            (
                len(self.monomerLengths),
                1,
                bound.shape[2] if bound.ndim > 2 else 1,
            )
        )
        if bound.ndim == 2:
            bound, lengths = bound[:, :, None], lengths[:, :, None]
        for i in range(bound.shape[0]):
            for k in range(bound.shape[2]):
                idx = int(bound[i, 0, k])
                if idx > 0:
                    out[idx - 1, 0, k] = max(
                        out[idx - 1, 0, k], lengths[i, 0, k]
                    )
        return out

    @property
    def abortedSequences(self) -> list[str]:
        rows = _first_timepoint(self.abortedPolypeptides)
        values = []
        for monomer, n, tag in rows:
            seq = ""
            if monomer:
                seq = (
                    _cell_sequence(self.monomerAASequences, int(monomer) - 1)[
                        : int(n)
                    ]
                    + _sequence(self.proteolysisTagAASequence)[: int(tag)]
                )
            values.append(seq)
        return values

    @property
    def abortedSequenceLengths(self) -> np.ndarray:
        return np.sum(
            _first_timepoint(self.abortedPolypeptides)[:, 1:3],
            axis=1,
            keepdims=True,
        )

    @property
    def dryWeight(self) -> float:
        seqs = [s for s in self._nascent_sequences() if s] + [
            s
            for row, s in zip(
                _first_timepoint(self.abortedPolypeptides),
                self.abortedSequences,
            )
            if row[0] != 0
        ]
        seq = "".join(seqs)
        # MATLAB Polypeptide uses ProteinMonomer.bases(1:20), in this order.
        order = "ARNDCQEGHILKMFPSTWYV"
        weights = np.asarray(self.aminoAcidMolecularWeights).reshape(-1)[:20]
        mass = sum(seq.count(aa) * weights[i] for i, aa in enumerate(order))
        # Preserve the source formula H + 2*O exactly, including its unusual
        # bond correction; changing its chemistry would change the model.
        water = 1.0079 + 2 * 15.9994
        return float((mass - (len(seq) - len(seqs)) * water) / 6.0221417930e23)

    def _nascent_sequences(self) -> list[str]:
        bound = np.asarray(self.boundMRNAs)
        lengths = np.asarray(self.nascentMonomerLengths)
        tags = np.asarray(self.proteolysisTagLengths)
        if bound.ndim == 2:
            bound, lengths, tags = (
                bound[:, :, None],
                lengths[:, :, None],
                tags[:, :, None],
            )
        result = []
        # MATLAB totalBaseCounts and dryWeight use two-subscript accesses such
        # as boundMRNAs(i), which linear-index the first time slice. Keep the
        # all-timepoint behavior in mRNAMaxNascentMonomerLength, whose source
        # explicitly loops over k.
        k = 0
        for i in range(bound.shape[0]):
            idx = int(bound[i, 0, k])
            seq = ""
            if int(lengths[i, 0, k]) > 0:
                seq += _cell_sequence(self.monomerAASequences, idx - 1)[
                    : int(lengths[i, 0, k])
                ]
            if int(tags[i, 0, k]) > 0:
                seq += _sequence(self.proteolysisTagAASequence)[
                    : int(tags[i, 0, k])
                ]
            result.append(seq)
        return result


def _first_timepoint(values):
    array = np.asarray(values)
    if array.ndim >= 3:
        return array[:, :, 0]
    return array


def _aa_counts(
    sequence: str, n_terminal_formyl_methionine: bool = False
) -> np.ndarray:
    # ProteinMonomer.bases has twenty amino acids plus formyl-methionine.
    alphabet = "ARNDCQEGHILKMFPSTWYV"
    counts = np.array(
        [sequence.count(aa) for aa in alphabet] + [0.0], dtype=float
    )
    if n_terminal_formyl_methionine:
        counts[12] -= 1  # N-terminal methionine is formylated.
        counts[20] += 1
    return counts


def _sequence(value: Any) -> str:
    """Join character-array snapshots into the original amino-acid sequence."""
    if isinstance(value, str):
        return value
    array = np.asarray(value)
    if array.dtype.kind in "US":
        return "".join(array.astype(str).reshape(-1).tolist())
    return str(value)


def _cell_sequence(values, index):
    value = (
        values[index]
        if isinstance(values, (list, tuple))
        else np.asarray(values, dtype=object).ravel(order="F")[index]
    )
    return _sequence(value)


def _cell_strings(values) -> list[str]:
    if isinstance(values, str):
        return [values]
    return [
        _sequence(value)
        for value in np.asarray(values, dtype=object).reshape(-1, order="F")
    ]


def _as_text(value: Any) -> str:
    return _sequence(value).strip()


def _reverse_complement(sequence: str) -> str:
    table = str.maketrans("ACGTacgt", "TGCAtgca")
    return sequence.translate(table)[::-1]


def _translate_dna_to_trna(
    sequence: str,
    codon_to_trna: Mapping[str, int],
    find_start: bool,
    start_codon_to_trna: Mapping[str, int] | None = None,
) -> np.ndarray:
    start = 0
    if find_start:
        start = next(
            (
                i
                for i in range(max(0, len(sequence) - 2))
                if sequence[i : i + 3] in (start_codon_to_trna or {})
            ),
            -1,
        )
        if start < 0:
            raise ValueError("protein-coding sequence has no start codon")
        trna = [int((start_codon_to_trna or {})[sequence[start : start + 3]])]
        codon_start = start + 3
    else:
        trna = []
        codon_start = 0
    for i in range(codon_start, len(sequence) - 2, 3):
        value = codon_to_trna.get(sequence[i : i + 3])
        if value is None:
            break
        trna.append(int(value))
    return np.asarray(trna, dtype=int)


def _translate_dna_to_amino_acids(
    sequence: str, codons: list[str], amino_acids: list[str]
) -> str:
    by_codon = dict(zip(codons, amino_acids))
    one_letter = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "FMET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }
    result = []
    for i in range(0, len(sequence) - 2, 3):
        codon = sequence[i : i + 3]
        aa = by_codon.get(codon)
        if aa is None:
            raise ValueError(
                f"codon {codon!r} has no amino-acid mapping in the KB"
            )
        aa = _as_text(aa)
        if aa not in one_letter:
            raise ValueError(f"amino acid {aa!r} is not a protein monomer")
        result.append(one_letter[aa])
    return "".join(result)
