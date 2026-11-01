"""RNA molecule-count state translated from WholeCell's ``Rna``."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from wholecell_py.molecule_count_state import MoleculeCountState, _field


class Rna(MoleculeCountState):
    optionNames = ("verbosity", "seed")
    fixedConstantNames = (
        "molecularWeights",
        "baseCounts",
        "lengths",
        "halfLives",
        "compartments",
        "expectedGeneExpression",
        "expectedGeneHalfLives",
        "weightFractionMRNA",
        "weightFractionRRNA5S",
        "weightFractionRRNA16S",
        "weightFractionRRNA23S",
        "weightFractionTRNA",
        "weightFractionSRNA",
        "nascentRNAGeneComposition",
        "nascentRNAMatureRNAComposition",
        "matureRNAGeneComposition",
        "intergenicRNAMatrix",
        "geneExpressionRobustness",
        "minTRnaCnt",
    )
    fittedConstantNames = ("expression",)
    stateNames = ("counts",)
    dependentStateNames: tuple[str, ...] = ()

    mRNAWeightFractionIndexs = 1
    rRNAWeightFractionIndexs = np.arange(2, 5).reshape(-1, 1)
    sRNAWeightFractionIndexs = 5
    tRNAWeightFractionIndexs = 6
    rRNA5SWeightFractionIndexs = 2
    rRNA16SWeightFractionIndexs = 3
    rRNA23SWeightFractionIndexs = 4

    _index_names = (
        "nascentIndexs",
        "processedIndexs",
        "intergenicIndexs",
        "matureIndexs",
        "boundIndexs",
        "misfoldedIndexs",
        "damagedIndexs",
        "aminoacylatedIndexs",
        "matureMRNAIndexs",
        "matureRRNAIndexs",
        "matureSRNAIndexs",
        "matureTRNAIndexs",
        "matureRibosomalRRNAIndexs",
        "matureTMRNAIndexs",
    )
    _rna_names = (
        "expression",
        "expectedGeneExpression",
        "expectedGeneHalfLives",
        "weightFractionMRNA",
        "weightFractionRRNA5S",
        "weightFractionRRNA16S",
        "weightFractionRRNA23S",
        "weightFractionTRNA",
        "weightFractionSRNA",
        "nascentRNAGeneComposition",
        "nascentRNAMatureRNAComposition",
        "matureRNAGeneComposition",
        "intergenicRNAMatrix",
        "geneExpressionRobustness",
        "minTRnaCnt",
    )
    _snapshot_names = (
        MoleculeCountState._snapshot_names + _index_names + _rna_names
    )

    def __init__(
        self,
        wholeCellModelID: Any = None,
        name: Any = None,
        snapshot: Mapping[str, Any] | None = None,
        **attributes: Any,
    ) -> None:
        super().__init__(wholeCellModelID, name, snapshot, **attributes)
        for field in self._index_names + self._rna_names:
            if not hasattr(self, field):
                setattr(self, field, None)
        self.ribosome = getattr(self, "ribosome", None)

    def storeObjectReferences(self, simulation: Any) -> None:
        super().storeObjectReferences(simulation)
        self.ribosome = _simulation_state(simulation, "Ribosome")

    def initializeConstants(
        self, knowledgeBase: Any, simulation: Any = None
    ) -> None:
        super().initializeConstants(knowledgeBase, simulation)
        genes = list(_field(knowledgeBase, "genes"))
        n_genes = int(_field(knowledgeBase, "numGenes"))
        if len(genes) != n_genes:
            raise ValueError("numGenes does not match genes records")
        self.expectedGeneHalfLives = (
            np.asarray(
                [_field(gene, "halfLife") for gene in genes], dtype=float
            ).reshape(-1, 1)
            * 60
        )
        expression_rows = [
            np.asarray(_field(gene, "expression"), dtype=float).reshape(
                -1, order="F"
            )
            for gene in genes
        ]
        if expression_rows:
            n_conditions = len(expression_rows[0])
            if any(len(row) != n_conditions for row in expression_rows):
                raise ValueError(
                    "gene expression rows must have equal lengths"
                )
            expected = np.vstack(expression_rows)
            expected = expected / np.sum(expected, axis=0, keepdims=True)
        else:
            expected = np.zeros((0, 0), dtype=float)
        self.expectedGeneExpression = expected

        self.nascentRNAGeneComposition = _sum_third(
            _field(knowledgeBase, "transcriptionUnitComposition_Genes")
        )
        self.nascentRNAMatureRNAComposition = _sum_third(
            _field(knowledgeBase, "transcriptionUnitComposition_ProcessedRNAs")
        )
        self.matureRNAGeneComposition = _sum_third(
            _field(knowledgeBase, "processedRNAComposition_Genes")
        )
        self.intergenicRNAMatrix = _sum_third(
            _field(
                knowledgeBase, "transcriptionUnitComposition_IntergenicRNAs"
            )
        )

        nascent = _field(knowledgeBase, "nascentRNAs")
        processed = _field(knowledgeBase, "processedRNAs")
        intergenic = _field(knowledgeBase, "intergenicRNAs")
        mature = _field(knowledgeBase, "matureRNAs")
        aminoacylated = _field(knowledgeBase, "aminoacylatedRNAs")
        nascent_ids = _vector(_field(nascent, "wholeCellModelIDs"), object)
        processed_ids = _vector(_field(processed, "wholeCellModelIDs"), object)
        intergenic_ids = _vector(
            _field(intergenic, "wholeCellModelIDs"), object
        )
        mature_ids = _vector(_field(mature, "wholeCellModelIDs"), object)
        aminoacylated_ids = _vector(
            _field(aminoacylated, "wholeCellModelIDs"), object
        )
        n_nascent, n_processed, n_intergenic, n_mature = (
            len(nascent_ids),
            len(processed_ids),
            len(intergenic_ids),
            len(mature_ids),
        )
        self.nascentIndexs = np.arange(1, n_nascent + 1, dtype=int).reshape(
            -1, 1
        )
        self.processedIndexs = np.arange(
            n_nascent + 1, n_nascent + n_processed + 1, dtype=int
        ).reshape(-1, 1)
        self.intergenicIndexs = np.arange(
            n_nascent + n_processed + 1,
            n_nascent + n_processed + n_intergenic + 1,
            dtype=int,
        ).reshape(-1, 1)
        first_mature = n_nascent + n_processed + n_intergenic + 1
        self.matureIndexs = np.arange(
            first_mature, first_mature + n_mature, dtype=int
        ).reshape(-1, 1)
        self.boundIndexs = self.matureIndexs + n_mature
        self.misfoldedIndexs = self.boundIndexs + n_mature
        self.damagedIndexs = self.misfoldedIndexs + n_mature
        aminoacylated_start = first_mature + 4 * n_mature
        self.aminoacylatedIndexs = np.arange(
            aminoacylated_start,
            aminoacylated_start + len(aminoacylated_ids),
            dtype=int,
        ).reshape(-1, 1)

        self.wholeCellModelIDs = (
            list(nascent_ids)
            + list(processed_ids)
            + list(intergenic_ids)
            + list(mature_ids) * 4
            + list(aminoacylated_ids)
        )
        self.names = (
            list(_vector(_field(nascent, "names"), object))
            + list(_vector(_field(processed, "names"), object))
            + list(_vector(_field(intergenic, "names"), object))
            + list(_vector(_field(mature, "names"), object)) * 4
            + list(_vector(_field(aminoacylated, "names"), object))
        )
        self.lengths = np.concatenate(
            [
                _vector(_field(nascent, "lengths"), float),
                _vector(_field(processed, "lengths"), float),
                _vector(_field(intergenic, "lengths"), float),
                np.tile(_vector(_field(mature, "lengths"), float), 4),
                _vector(_field(aminoacylated, "lengths"), float),
            ]
        ).reshape(-1, 1)
        self.baseCounts = np.vstack(
            [
                _matrix(_field(nascent, "baseCounts")),
                _matrix(_field(processed, "baseCounts")),
                _matrix(_field(intergenic, "baseCounts")),
                np.tile(_matrix(_field(mature, "baseCounts")), (4, 1)),
                _matrix(_field(aminoacylated, "baseCounts")),
            ]
        )
        self.molecularWeights = np.concatenate(
            [
                _vector(_field(nascent, "molecularWeights"), float),
                _vector(_field(processed, "molecularWeights"), float),
                _vector(_field(intergenic, "molecularWeights"), float),
                np.tile(_vector(_field(mature, "molecularWeights"), float), 4),
                _vector(_field(aminoacylated, "molecularWeights"), float),
            ]
        ).reshape(-1, 1)

        # This is the literal source initializer. It is a precursor expression
        # vector; the Rna fitted constant is replaced by FitConstants before
        # execution. Preserve the source's omission of an intergenic zero block.
        mature_expression = (
            self.matureRNAGeneComposition.T @ self.expectedGeneExpression[:, 0]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            mature_expression = mature_expression / np.linalg.norm(
                mature_expression, ord=1
            )
        self.expression = np.concatenate(
            [
                np.zeros(n_nascent + n_processed),
                mature_expression,
                np.zeros(3 * n_mature + len(aminoacylated_ids)),
            ]
        ).reshape(-1, 1)

        gene_groups = (
            ("matureMRNAIndexs", "mRNAGenes"),
            ("matureRRNAIndexs", "rRNAGenes"),
            ("matureSRNAIndexs", "sRNAGenes"),
            ("matureTRNAIndexs", "tRNAGenes"),
        )
        for target, source in gene_groups:
            gene_rows = (
                np.asarray(
                    [
                        _field(gene, "idx")
                        for gene in _field(knowledgeBase, source)
                    ],
                    dtype=int,
                ).reshape(-1)
                - 1
            )
            totals = np.sum(
                self.matureRNAGeneComposition[gene_rows, :], axis=0
            )
            setattr(self, target, (np.flatnonzero(totals) + 1).reshape(-1, 1))

        ribosomal_gene_indices = _vector(
            _field(knowledgeBase, "ribosomalRRNAIndexs"), int
        )
        ribosomal_rna_indices = []
        for gene_index in ribosomal_gene_indices:
            matches = np.flatnonzero(
                self.matureRNAGeneComposition[int(gene_index) - 1, :]
            )
            if not matches.size:
                raise ValueError(
                    f"ribosomal gene index {gene_index} has no mature RNA"
                )
            ribosomal_rna_indices.append(int(matches[0]) + 1)
        self.matureRibosomalRRNAIndexs = np.asarray(
            ribosomal_rna_indices, dtype=int
        ).reshape(-1, 1)
        self.matureTMRNAIndexs = self.getIndexs("MG_0004")

    def updateExternalState(
        self, deltaRnas: Any, rnaIsDegraded: Any = None
    ) -> np.ndarray:
        del rnaIsDegraded
        ribosome = _require(self, "ribosome")
        compartment = _require(self, "compartment")
        delta = np.asarray(deltaRnas)
        cytosol = _indices(compartment.cytosolIndexs)
        mature_delta = delta[_indices(self.matureIndexs), cytosol]
        bound_delta = delta[_indices(self.boundIndexs), cytosol]
        mrna = _indices(self.matureMRNAIndexs)
        tmrna = _indices(self.matureTMRNAIndexs)
        if np.any(mature_delta[mrna]) or np.any(bound_delta[tmrna] < 0):
            ribosome.releaseRibosome(
                0, -bound_delta[tmrna], -mature_delta[mrna]
            )
        return np.zeros_like(delta)

    def getIndexs(self, wholeCellModelIDs: Any) -> Any:
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
        mature_ids = ids[mature - 1]
        requested = np.asarray(wholeCellModelIDs, dtype=object)
        result = [
            (
                int(match[0] + 1)
                if (match := np.flatnonzero(mature_ids == item)).size
                else 0
            )
            for item in requested.reshape(-1, order="F")
        ]
        values = np.asarray(result, dtype=int)
        return (
            int(values[0])
            if requested.ndim == 0
            else values.reshape(requested.shape, order="F")
        )

    @property
    def geneHalfLives(self) -> np.ndarray:
        return (
            np.asarray(self.matureRNAGeneComposition)
            @ np.asarray(self.halfLives)[_indices(self.matureIndexs)]
        )

    @property
    def geneDecayRates(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(2) / self.geneHalfLives

    @property
    def expectedGeneDecayRates(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(2) / np.asarray(self.expectedGeneHalfLives)

    @property
    def weightFractions(self) -> np.ndarray:
        weights = np.asarray(self.molecularWeights).reshape(-1, order="F")
        expression = np.asarray(self.expression).reshape(-1, order="F")
        mature = _indices(self.matureIndexs)
        rrna = np.asarray(self.matureRibosomalRRNAIndexs, dtype=int).reshape(
            -1, order="F"
        )
        groups = (
            self.matureMRNAIndexs,
            rrna[:1],
            rrna[1:2],
            rrna[2:3],
            self.matureSRNAIndexs,
            self.matureTRNAIndexs,
        )
        values = []
        for group in groups:
            idx = mature[_indices(group)]
            values.append(np.sum(weights[idx] * expression[idx]))
        value = np.asarray(values, dtype=float)
        return value / np.sum(value)

    @property
    def expectedWeightFractions(self) -> np.ndarray:
        value = np.asarray(
            [
                self.weightFractionMRNA,
                self.weightFractionRRNA5S,
                self.weightFractionRRNA16S,
                self.weightFractionRRNA23S,
                self.weightFractionSRNA,
                self.weightFractionTRNA,
            ],
            dtype=float,
        )
        return (value / np.sum(value)).reshape(-1, 1)

    @property
    def geneExpression(self) -> np.ndarray:
        value = (
            np.asarray(self.matureRNAGeneComposition)
            @ np.asarray(self.expression)[_indices(self.matureIndexs)]
        )
        return value / np.sum(value)


def _indices(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=int).reshape(-1, order="F") - 1


def _require(obj: Any, field: str) -> Any:
    value = getattr(obj, field, None)
    if value is None:
        raise RuntimeError(f"Rna dependency {field!r} has not been assigned")
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


def _vector(value: Any, dtype: Any) -> np.ndarray:
    return np.asarray(value, dtype=dtype).reshape(-1, order="F")


def _matrix(value: Any) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim == 1:
        return result.reshape(1, -1)
    return result.reshape((result.shape[0], -1), order="F")


def _sum_third(value: Any) -> np.ndarray:
    array = np.asarray(value)
    return np.sum(array, axis=2) if array.ndim >= 3 else array.copy()
