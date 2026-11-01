"""Runtime port of MATLAB MacromolecularComplexation."""

from __future__ import annotations
import numpy as np
from ..process_base import Process


class MacromolecularComplexation(Process):
    stimuliWholeCellModelIDs = ()
    substrateWholeCellModelIDs = ()
    enzymeWholeCellModelIDs = ()
    complexWholeCellModelIDs = ()
    optionNames__ = ()
    fixedConstantNames__ = (
        "complexComposition",
        "complexNetworks",
        "substrates2complexNetworks",
        "complexs2complexNetworks",
    )
    fittedConstantNames__ = ()
    localStateNames__ = ("complexs",)

    def __init__(
        self,
        wholeCellModelID="Process_MacromolecularComplexation",
        name="Macromolecular complexation",
        **properties,
    ):
        super().__init__(wholeCellModelID, name, **properties)

    @staticmethod
    def _idx(x):
        return np.asarray(x, dtype=int).ravel() - 1

    @staticmethod
    def _get(o, k):
        return o[k] if isinstance(o, dict) else getattr(o, k)

    @staticmethod
    def _set(o, k, v):
        if isinstance(o, dict):
            o[k] = v
        else:
            setattr(o, k, v)

    def initializeConstants(
        self, knowledgeBase=None, simulation=None, *args, **kwargs
    ):
        options = kwargs.pop("options", None)
        if knowledgeBase is not None or simulation is not None:
            if knowledgeBase is None or simulation is None:
                raise ValueError(
                    "knowledgeBase and simulation must be provided together"
                )
            genes = np.asarray(
                self._get(self.gene, "wholeCellModelIDs"), dtype=object
            ).ravel(order="F")
            substrate_ids = np.empty(len(genes), dtype=object)
            substrate_ids[:] = ""
            for gene_field, state_name, state_index in (
                ("mRNAIndexs", "monomer", "matureIndexs"),
                ("rRNAIndexs", "rna", "matureRRNAIndexs"),
                ("sRNAIndexs", "rna", "matureSRNAIndexs"),
                ("tRNAIndexs", "rna", "matureTRNAIndexs"),
            ):
                gene_indices = np.asarray(
                    self._get(self.gene, gene_field), dtype=int
                ).ravel(order="F")
                state = getattr(self, state_name)
                whole_ids = np.asarray(
                    self._get(state, "wholeCellModelIDs"), dtype=object
                ).ravel(order="F")
                mature = np.asarray(
                    self._get(state, state_index), dtype=int
                ).ravel(order="F")
                state_local_indices = np.arange(len(gene_indices), dtype=int)
                substrate_ids[gene_indices - 1] = whole_ids[
                    mature[state_local_indices] - 1
                ]
            self.substrateWholeCellModelIDs = substrate_ids.reshape(-1, 1)

            records = np.asarray(
                self._get(knowledgeBase, "processes"), dtype=object
            ).ravel(order="F")
            from ..process_base import _cell_text

            record = next(
                (
                    p
                    for p in records
                    if _cell_text(self._get(p, "wholeCellModelID"))
                    == _cell_text(self.wholeCellModelID)
                ),
                None,
            )
            if record is None:
                raise ValueError(
                    f"MacromolecularComplexation: {self.wholeCellModelID!r} missing from knowledge base"
                )
            complex_records = np.asarray(
                self._get(knowledgeBase, "proteinComplexs"), dtype=object
            ).ravel(order="F")
            process_idx = (
                self._get(record, "idx") if hasattr(record, "idx") else None
            )
            selected = []
            for i, complex_record in enumerate(complex_records):
                formation = self._get(
                    complex_record, "complexFormationProcess"
                )
                if isinstance(formation, str):
                    matches = _cell_text(formation) == _cell_text(
                        self.wholeCellModelID
                    )
                elif hasattr(formation, "wholeCellModelID"):
                    matches = _cell_text(
                        self._get(formation, "wholeCellModelID")
                    ) == _cell_text(self.wholeCellModelID)
                else:
                    formation_idx = (
                        self._get(formation, "idx")
                        if hasattr(formation, "idx")
                        else formation
                    )
                    matches = (
                        process_idx is not None
                        and formation_idx == process_idx
                    )
                if matches:
                    selected.append(i + 1)
            complex_indices = np.asarray(selected, dtype=int)
            monomer_nascent = np.asarray(
                self._get(self.monomer, "nascentIndexs"), dtype=int
            ).ravel(order="F")
            self.complexGlobalIndexs = monomer_nascent[
                complex_indices - 1
            ].reshape(-1, 1)
            self.complexCompartmentIndexs = (
                np.asarray(self._get(self.complex, "compartments"), dtype=int)
                .ravel(order="F")[self.complexGlobalIndexs.ravel() - 1]
                .reshape(-1, 1)
            )
            complex_ids = np.asarray(
                self._get(self.complex, "wholeCellModelIDs"), dtype=object
            ).ravel(order="F")
            self.complexWholeCellModelIDs = complex_ids[
                self.complexGlobalIndexs.ravel() - 1
            ].reshape(-1, 1)
            n_complex_state = len(complex_ids)
            self.complexGlobalCompartmentIndexs = (
                self.complexGlobalIndexs.ravel()
                + (self.complexCompartmentIndexs.ravel() - 1) * n_complex_state
            ).reshape(-1, 1)

            rna_comp = np.asarray(
                self._get(knowledgeBase, "proteinComplexRNAComposition")
            )[:, complex_indices - 1, :]
            mon_comp = np.asarray(
                self._get(knowledgeBase, "proteinComplexMonomerComposition")
            )[:, complex_indices - 1, :]
            self.complexComposition = np.sum(rna_comp + mon_comp, axis=2)
            self.initializeComplexNetworks()
            from ..process_base import Process

            Process.initializeConstants(
                self, knowledgeBase, simulation, options or {}
            )
            return

        required = (
            "complexComposition",
            "complexNetworks",
            "substrates2complexNetworks",
            "complexs2complexNetworks",
            "complexGlobalIndexs",
            "complexCompartmentIndexs",
            "complexGlobalCompartmentIndexs",
        )
        missing = [x for x in required if not hasattr(self, x)]
        if missing:
            raise NotImplementedError(
                "initializeConstants requires initialized knowledge-base/simulation network fields: "
                + ", ".join(missing)
            )

    def initializeComplexNetworks(self):
        matrix = np.asarray(self.complexComposition)
        if matrix.ndim != 2:
            raise ValueError("complexComposition must be a 2-D matrix")
        nr, nc = matrix.shape
        row_assignment = np.zeros(nr, dtype=int)
        col_assignment = np.zeros(nc, dtype=int)

        # MATLAB findNonInteractingRowsAndColumns puts all empty rows and
        # columns into block 1 before it walks the nonzero connected blocks.
        empty_rows = ~np.any(matrix != 0, axis=1)
        empty_cols = ~np.any(matrix != 0, axis=0)
        if np.any(empty_rows) or np.any(empty_cols):
            row_assignment[empty_rows] = 1
            col_assignment[empty_cols] = 1
            nblocks = 1
        else:
            nblocks = 0

        def visit_row(row, block):
            row_assignment[row] = block
            pending_cols = np.flatnonzero(
                (matrix[row, :] != 0) & (col_assignment == 0)
            )
            for col in pending_cols:
                visit_col(int(col), block)

        def visit_col(col, block):
            col_assignment[col] = block
            pending_rows = np.flatnonzero(
                (matrix[:, col] != 0) & (row_assignment == 0)
            )
            for row in pending_rows:
                visit_row(int(row), block)

        # This traversal order matches the MATLAB utility: first unassigned
        # rows in order, then unassigned columns in order.
        for row in range(nr):
            if row_assignment[row] == 0:
                nblocks += 1
                visit_row(row, nblocks)
        for col in range(nc):
            if col_assignment[col] == 0:
                nblocks += 1
                visit_col(col, nblocks)

        source_blocks = [
            matrix[np.ix_(row_assignment == i, col_assignment == i)]
            for i in range(1, nblocks + 1)
        ]
        substrates2networks = np.zeros(nr, dtype=int)
        complexes2networks = np.zeros(nc, dtype=int)
        networks = [None]
        network_index = 1
        for i, block in enumerate(source_blocks, start=1):
            ncols = block.shape[1]
            if ncols == 0:
                continue
            if ncols == 1:
                substrates2networks[row_assignment == i] = 1
                complexes2networks[col_assignment == i] = 1
            else:
                network_index += 1
                substrates2networks[row_assignment == i] = network_index
                complexes2networks[col_assignment == i] = network_index
                networks.append(block)

        # MATLAB merges every single-column block into network 1 and removes
        # rows/columns assigned to empty, zero-column blocks.
        networks[0] = matrix[
            np.ix_(substrates2networks == 1, complexes2networks == 1)
        ]
        keep_substrates = substrates2networks > 0
        keep_complexes = complexes2networks > 0
        self.substrates2complexNetworks = substrates2networks[
            keep_substrates
        ].reshape(-1, 1)
        self.complexs2complexNetworks = complexes2networks[
            keep_complexes
        ].reshape(-1, 1)
        self.complexNetworks = networks
        self.complexComposition = matrix[
            np.ix_(keep_substrates, keep_complexes)
        ]

        for name, keep in (
            ("substrateWholeCellModelIDs", keep_substrates),
            ("complexWholeCellModelIDs", keep_complexes),
            ("complexGlobalIndexs", keep_complexes),
            ("complexCompartmentIndexs", keep_complexes),
            ("complexGlobalCompartmentIndexs", keep_complexes),
        ):
            if hasattr(self, name):
                values = np.asarray(getattr(self, name))
                if values.shape[0] == keep.size:
                    setattr(self, name, values[keep])

    def allocateMemoryForState(self, numTimePoints=1):
        super().allocateMemoryForState(numTimePoints)
        n = len(np.asarray(self.complexWholeCellModelIDs).reshape(-1))
        self.complexs = np.zeros(
            (n, 1) if numTimePoints == 1 else (n, 1, numTimePoints)
        )

    def copyFromState(self):
        super().copyFromState()
        counts = np.asarray(self._get(self.complex, "counts"))
        gi = self._idx(self.complexGlobalIndexs)
        ci = self._idx(self.complexCompartmentIndexs)
        if counts.ndim < 3 or counts.shape[2] == 1:
            self.complexs = counts.ravel(order="F")[
                gi + ci * counts.shape[0]
            ].reshape(-1, 1)
        else:
            self.complexs = counts[gi, ci, :].reshape(
                (len(gi), 1, counts.shape[2])
            )

    def copyToState(self):
        super().copyToState()
        counts = np.asarray(self._get(self.complex, "counts"))
        gi = self._idx(self.complexGlobalIndexs)
        ci = self._idx(self.complexCompartmentIndexs)
        val = np.asarray(self.complexs)
        if counts.ndim < 3 or counts.shape[2] == 1:
            flat = counts.reshape(-1, order="F")
            flat[gi + ci * counts.shape[0]] = val.reshape(-1, order="F")
            counts = flat.reshape(counts.shape, order="F")
        else:
            counts[gi, ci, :] = val.reshape((len(gi), counts.shape[2]))
        self._set(self.complex, "counts", counts)

    def calcResourceRequirements_LifeCycle(self, *args):
        return (
            np.zeros_like(
                np.asarray(self.substrateWholeCellModelIDs, dtype=float)
            ),
            np.zeros_like(
                np.asarray(self.substrateWholeCellModelIDs, dtype=float)
            ),
            np.zeros_like(
                np.asarray(self.enzymeWholeCellModelIDs, dtype=float)
            ),
            np.full(np.asarray(self.enzymeWholeCellModelIDs).shape, np.inf),
        )

    def initializeState(self):
        self.evolveState()

    def calcResourceRequirements_Current(self):
        return np.zeros_like(np.asarray(self.substrates), dtype=float)

    def evolveState(self):
        new = np.zeros_like(np.asarray(self.complexs), dtype=float)
        sm = np.asarray(self.substrates2complexNetworks).ravel()
        cm = np.asarray(self.complexs2complexNetworks).ravel()
        network = lambda i: np.asarray(
            (
                self.complexNetworks[i].item()
                if np.asarray(self.complexNetworks[i]).dtype == object
                and np.asarray(self.complexNetworks[i]).size == 1
                else self.complexNetworks[i]
            ),
            dtype=float,
        )
        if np.any(cm == 1):
            new[np.flatnonzero(cm == 1), :] = buildProteinComplexs_bounds(
                np.asarray(self.substrates).reshape(-1)[sm == 1], network(0)
            )
        for i in range(2, len(self.complexNetworks) + 1):
            if np.any(cm == i):
                new[np.flatnonzero(cm == i), :] = (
                    buildProteinComplexs_montecarlokinetic(
                        np.asarray(self.substrates).reshape(-1)[sm == i],
                        network(i - 1),
                        self.randStream,
                    )
                )
        if not np.any(new):
            return
        self.complexs = np.asarray(self.complexs) + new.reshape(
            np.asarray(self.complexs).shape
        )
        # MATLAB keeps this product as a column vector. NumPy's matrix
        # multiplication with a 1-D complex vector returns shape (n,), which
        # broadcasts against the (n, 1) substrate state to produce an n×n
        # array. Preserve the column shape before subtracting.
        substrate_consumption = np.asarray(
            self.complexComposition
        ) @ new.reshape(-1, 1)
        self.substrates = np.asarray(self.substrates) - substrate_consumption

    def getDryWeight(self):
        base = np.asarray(super().getDryWeight())
        mw = np.asarray(self._get(self.complex, "molecularWeights")).ravel(
            order="F"
        )[self._idx(self.complexGlobalIndexs)]
        c = np.asarray(self.complexs)
        return base + np.tensordot(mw, c, axes=(0, 0)) / float(
            getattr(self, "nAvogadro", 6.022141793e23)
        )


def buildProteinComplexs_bounds(totalProteinMonomers, proteinComplexMatrix):
    total = np.asarray(totalProteinMonomers, dtype=float).reshape(-1)
    mat = np.asarray(proteinComplexMatrix, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.divide(
            total[:, None],
            mat,
            out=np.full((len(total), mat.shape[1]), np.inf),
            where=mat != 0,
        )
    return np.floor(np.min(ratios, axis=0)).astype(int).reshape(-1, 1)


def buildProteinComplexs_rates_collisionTheory(
    totalProteinMonomers, proteinComplexMatrix, normalization=None
):
    total = np.asarray(totalProteinMonomers, dtype=float).reshape(-1)
    mat = np.asarray(proteinComplexMatrix, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        rates = np.prod((total[:, None] / np.mean(total)) ** mat, axis=0)
    rates[np.asarray(buildProteinComplexs_bounds(total, mat)).ravel() == 0] = 0
    if normalization:
        if str(normalization)[0].lower() == "c":
            rates = np.cumsum(rates)
            with np.errstate(divide="ignore", invalid="ignore"):
                rates = rates / rates[-1]
        elif str(normalization)[0].lower() == "p":
            with np.errstate(divide="ignore", invalid="ignore"):
                rates = rates / np.sum(rates)
        else:
            raise ValueError("Invalid normalization")
    return rates.reshape(-1, 1)


def buildProteinComplexs_montecarlokinetic(
    totalProteinMonomers, proteinComplexMatrix, randStream
):
    total = np.asarray(totalProteinMonomers, dtype=float).reshape(-1).copy()
    mat = np.asarray(proteinComplexMatrix, dtype=float)
    formed = np.zeros((mat.shape[1], 1), dtype=int)
    while True:
        cum = buildProteinComplexs_rates_collisionTheory(
            total, mat, "cumulative probability"
        ).ravel()
        if not len(cum) or np.isnan(cum[0]):
            break
        selected = np.flatnonzero(
            float(np.asarray(randStream.rand()).ravel()[0]) < cum
        )
        if not len(selected):
            selected = np.flatnonzero(cum == 1)
        if not len(selected):
            break
        j = int(selected[0])
        formed[j, 0] += 1
        total -= mat[:, j]
    return formed
