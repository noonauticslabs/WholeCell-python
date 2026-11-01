"""Runtime port of MATLAB ``RibosomeAssembly``."""

from __future__ import annotations

import numpy as np

from ..process_base import Process, _get, _set


def _idx(value):
    return np.asarray(value, dtype=int).reshape(-1) - 1


def _dense(value):
    return value.full() if hasattr(value, "full") else np.asarray(value)


class RibosomeAssembly(Process):
    stimuliWholeCellModelIDs = ()
    substrateWholeCellModelIDs = ("GTP", "GDP", "PI", "H2O", "H")
    enzymeWholeCellModelIDs = (
        "MG_329_MONOMER",
        "MG_335_MONOMER",
        "MG_387_MONOMER",
        "MG_384_MONOMER",
        "MG_143_MONOMER",
        "MG_442_MONOMER",
    )
    complexWholeCellModelIDs = ("RIBOSOME_30S", "RIBOSOME_50S")
    enzymeIndexs_engA, enzymeIndexs_engB, enzymeIndexs_era = 1, 2, 3
    enzymeIndexs_obg, enzymeIndexs_rbfA, enzymeIndexs_rbgA = 4, 5, 6
    enzymeIndexs_30S_assembly_gtpase = np.array([3, 5])
    enzymeIndexs_50S_assembly_gtpase = np.array([1, 2, 4, 6])
    complexIndexs_30S_ribosome, complexIndexs_50S_ribosome = 1, 2
    optionNames__ = []
    fixedConstantNames__ = [
        "complexationCatalysisMatrix",
        "proteinComplexRNAComposition",
        "proteinComplexMonomerComposition",
    ]
    fittedConstantNames__ = []
    localStateNames__ = ["RNAs", "monomers", "complexs"]

    def __init__(
        self,
        wholeCellModelID="Process_RibosomeAssembly",
        name="Ribosomal Assembly",
        **properties,
    ):
        super().__init__(wholeCellModelID, name, **properties)

    def initializeConstants(
        self, knowledgeBase=None, simulation=None, options=None, **kwargs
    ):
        for key, value in kwargs.items():
            setattr(self, key, value)
        if knowledgeBase is None or simulation is None:
            raise NotImplementedError(
                "RibosomeAssembly.initializeConstants requires knowledge-base and simulation"
            )
        super().initializeConstants(knowledgeBase, simulation, options or {})
        self.substrateIndexs_gtp = 1
        self.substrateIndexs_gdp = 2
        self.substrateIndexs_phosphate = 3
        self.substrateIndexs_water = 4
        self.substrateIndexs_hydrogen = 5
        self.complexGlobalIndexs = self.initializeConstantsHelper(
            self.complexWholeCellModelIDs
        )[24].ravel(order="F")
        cyt = (
            int(np.asarray(_get(self.compartment, "cytosolIndexs")).item()) - 1
        )
        rna_composition = _dense(
            _get(knowledgeBase, "proteinComplexRNAComposition")
        )[:, :, cyt]
        gene_rna = (
            np.asarray(_get(self.gene, "rRNAIndexs"), dtype=int).ravel(
                order="F"
            )
            - 1
        )
        selected_rna = rna_composition[
            np.ix_(gene_rna, self.complexGlobalIndexs - 1)
        ]
        self.rnaGlobalIndexs = (
            np.flatnonzero(np.any(selected_rna != 0, axis=1)) + 1
        )
        mature_rna = np.asarray(
            _get(self.rna, "matureIndexs"), dtype=int
        ).ravel(order="F")
        rrna = np.asarray(_get(self.rna, "matureRRNAIndexs"), dtype=int).ravel(
            order="F"
        )
        self.rnaWholeCellModelIDs = (
            np.asarray(_get(self.rna, "wholeCellModelIDs"), dtype=object)
            .ravel(order="F")[
                mature_rna[rrna[self.rnaGlobalIndexs - 1] - 1] - 1
            ]
            .reshape(-1, 1)
        )
        self.proteinComplexRNAComposition = selected_rna[
            self.rnaGlobalIndexs - 1, :
        ]
        unique = np.unique(self.proteinComplexRNAComposition)
        if not np.all(np.isin(unique, [0, 1])):
            raise ValueError("Ribosome subunit RNA stoichiometry must be 0/1")
        self.proteinComplexRNAComposition = (
            self.proteinComplexRNAComposition == 1
        )

        monomer_composition = _dense(
            _get(knowledgeBase, "proteinComplexMonomerComposition")
        )[:, :, cyt]
        gene_mrna = (
            np.asarray(_get(self.gene, "mRNAIndexs"), dtype=int).ravel(
                order="F"
            )
            - 1
        )
        selected_monomer = monomer_composition[
            np.ix_(gene_mrna, self.complexGlobalIndexs - 1)
        ]
        self.monomerGlobalIndexs = (
            np.flatnonzero(np.any(selected_monomer != 0, axis=1)) + 1
        )
        mature_monomer = np.asarray(
            _get(self.monomer, "matureIndexs"), dtype=int
        ).ravel(order="F")
        self.monomerWholeCellModelIDs = (
            np.asarray(_get(self.monomer, "wholeCellModelIDs"), dtype=object)
            .ravel(order="F")[mature_monomer[self.monomerGlobalIndexs - 1] - 1]
            .reshape(-1, 1)
        )
        self.proteinComplexMonomerComposition = selected_monomer[
            self.monomerGlobalIndexs - 1, :
        ]
        unique = np.unique(self.proteinComplexMonomerComposition)
        if not np.all(np.isin(unique, [0, 1])):
            raise ValueError(
                "Ribosome subunit monomer stoichiometry must be 0/1"
            )
        self.proteinComplexMonomerComposition = (
            self.proteinComplexMonomerComposition == 1
        )
        self.complexationCatalysisMatrix = np.zeros(
            (
                len(self.enzymeWholeCellModelIDs),
                len(self.complexWholeCellModelIDs),
            ),
            dtype=bool,
        )
        self.complexationCatalysisMatrix[
            np.asarray(self.enzymeIndexs_30S_assembly_gtpase, dtype=int) - 1,
            int(self.complexIndexs_30S_ribosome) - 1,
        ] = True
        self.complexationCatalysisMatrix[
            np.asarray(self.enzymeIndexs_50S_assembly_gtpase, dtype=int) - 1,
            int(self.complexIndexs_50S_ribosome) - 1,
        ] = True

    def copyFromState(self):
        super().copyFromState()
        rna_gi = (
            np.asarray(self.rna.matureIndexs)
            .reshape(-1, order="F")[
                np.asarray(self.rna.matureRRNAIndexs)
                .reshape(-1, order="F")
                .astype(int)[_idx(self.rnaGlobalIndexs)]
                - 1
            ]
            .reshape(-1, 1)
        )
        ci = int(np.asarray(self.compartment.cytosolIndexs).reshape(-1)[0]) - 1

        def gather(state, gi):
            counts = np.asarray(state.counts)
            if counts.ndim == 2:
                counts = counts[:, :, None]
            return counts[
                np.asarray(gi, dtype=int).reshape(-1) - 1, ci : ci + 1, :
            ]

        self.RNAs = gather(self.rna, rna_gi)
        self.monomers = gather(
            self.monomer,
            np.asarray(self.monomer.matureIndexs).reshape(-1)[
                _idx(self.monomerGlobalIndexs)
            ],
        )
        self.complexs = gather(
            self.complex,
            np.asarray(self.complex.nascentIndexs).reshape(-1)[
                _idx(self.complexGlobalIndexs)
            ],
        )

    def copyToState(self):
        super().copyToState()
        mappings = (
            (
                self.rna,
                np.asarray(self.rna.matureIndexs).reshape(-1)[
                    np.asarray(self.rna.matureRRNAIndexs)
                    .reshape(-1)
                    .astype(int)[_idx(self.rnaGlobalIndexs)]
                    - 1
                ],
                self.RNAs,
            ),
            (
                self.monomer,
                np.asarray(self.monomer.matureIndexs).reshape(-1)[
                    _idx(self.monomerGlobalIndexs)
                ],
                self.monomers,
            ),
            (
                self.complex,
                np.asarray(self.complex.nascentIndexs).reshape(-1)[
                    _idx(self.complexGlobalIndexs)
                ],
                self.complexs,
            ),
        )
        ci = _idx(self.compartment.cytosolIndexs)
        for state, gi, value in mappings:
            counts = np.asarray(state.counts)
            if counts.ndim == 2:
                counts = counts.reshape((*counts.shape, 1))
            val = np.asarray(value)
            if val.ndim == 2:
                val = val.reshape((*val.shape, 1))
            for t in range(counts.shape[2]):
                counts[
                    np.asarray(gi, dtype=int).reshape(-1) - 1, int(ci[0]), t
                ] = val.reshape((len(gi), -1, counts.shape[2]))[:, 0, t]
            _set(state, "counts", counts)

    def allocateMemoryForState(self, numTimePoints):
        super().allocateMemoryForState(numTimePoints)
        self.RNAs = np.zeros((len(self.rnaGlobalIndexs), 1, numTimePoints))
        self.monomers = np.zeros(
            (len(self.monomerGlobalIndexs), 1, numTimePoints)
        )
        self.complexs = np.zeros(
            (len(self.complexGlobalIndexs), 1, numTimePoints)
        )

    def calcResourceRequirements_LifeCycle(self, *args):
        states = args[-1]
        bm = np.zeros((len(self.substrateWholeCellModelIDs), 1))
        by = np.zeros_like(bm)
        minimum = np.zeros((len(self.enzymeWholeCellModelIDs), 1))
        maximum = np.full_like(minimum, np.inf)
        rprod = np.asarray(_get(states, "rnaProductions")).reshape(
            -1, order="F"
        )
        mprod = np.asarray(_get(states, "monomerProductions")).reshape(
            -1, order="F"
        )
        ri = (
            np.asarray(self.rna.matureRRNAIndexs)[
                _idx(self.rnaGlobalIndexs)
            ].astype(int)
            - 1
        )
        mi = _idx(self.monomerGlobalIndexs)
        gtp_cost = 0
        for i in range(len(self.complexWholeCellModelIDs)):
            rmask = np.asarray(self.proteinComplexRNAComposition)[:, i].astype(
                bool
            )
            mmask = np.asarray(self.proteinComplexMonomerComposition)[
                :, i
            ].astype(bool)
            r = rprod[ri][rmask].reshape(-1)
            m = mprod[mi][mmask].reshape(-1)
            gtp_cost += self.getGtpPerComplex(i + 1) * np.min(
                np.concatenate((r, m))
            )
        bm[_idx(self.substrateIndexs_gtp)] = gtp_cost
        bm[_idx(self.substrateIndexs_water)] = gtp_cost
        for index in (
            self.substrateIndexs_gdp,
            self.substrateIndexs_phosphate,
            self.substrateIndexs_hydrogen,
        ):
            by[_idx(index)] = gtp_cost
        enz = np.r_[
            _idx(self.enzymeIndexs_30S_assembly_gtpase),
            _idx(self.enzymeIndexs_50S_assembly_gtpase),
        ]
        minimum[enz] = 2
        return bm, by, minimum, maximum

    def initializeState(self):
        rna = np.asarray(self.RNAs).reshape(-1, 1)
        mon = np.asarray(self.monomers).reshape(-1, 1)
        formed = np.zeros(len(self.complexWholeCellModelIDs), dtype=int)
        for i in range(len(formed)):
            rmask = np.asarray(self.proteinComplexRNAComposition)[:, i].astype(
                bool
            )
            mmask = np.asarray(self.proteinComplexMonomerComposition)[
                :, i
            ].astype(bool)
            formed[i] = int(
                np.floor(
                    min(
                        np.r_[
                            rna[rmask, 0]
                            / np.asarray(self.proteinComplexRNAComposition)[
                                rmask, i
                            ],
                            mon[mmask, 0]
                            / np.asarray(
                                self.proteinComplexMonomerComposition
                            )[mmask, i],
                        ]
                    )
                )
            )
        self.RNAs = (
            rna
            - np.asarray(self.proteinComplexRNAComposition) @ formed[:, None]
        ).reshape(np.shape(self.RNAs))
        self.monomers = (
            mon
            - np.asarray(self.proteinComplexMonomerComposition)
            @ formed[:, None]
        ).reshape(np.shape(self.monomers))
        self.complexs = np.asarray(self.complexs) + formed.reshape(
            np.shape(self.complexs)
        )

    def calcResourceRequirements_Current(self):
        result = np.zeros_like(np.asarray(self.substrates), dtype=float)
        for i in range(2):
            required = np.asarray(self.complexationCatalysisMatrix)[:, i]
            if np.all(np.asarray(self.enzymes).reshape(-1)[required]):
                avail = np.r_[
                    np.asarray(self.RNAs).reshape(-1)[
                        np.asarray(self.proteinComplexRNAComposition)[
                            :, i
                        ].astype(bool)
                    ],
                    np.asarray(self.monomers).reshape(-1)[
                        np.asarray(self.proteinComplexMonomerComposition)[
                            :, i
                        ].astype(bool)
                    ],
                ]
                result[
                    _idx(self.substrateIndexs_gtp)
                ] += self.getGtpPerComplex(2) * (1 + np.min(avail))
        result[_idx(self.substrateIndexs_water)] = result[
            _idx(self.substrateIndexs_gtp)
        ]
        return result

    def evolveState(self):
        if not np.asarray(self.substrates).reshape(-1)[
            _idx(self.substrateIndexs_gtp)[0]
        ]:
            return
        if self.randStream is None or not hasattr(self.randStream, "randperm"):
            raise NotImplementedError(
                "RibosomeAssembly.evolveState requires runtime RandStream.randperm"
            )
        order = np.asarray(
            self.randStream.randperm(len(self.complexWholeCellModelIDs))
        ).reshape(-1)
        for one_i in order:
            i = int(one_i) - 1
            gtp = self.getGtpPerComplex(i + 1)
            rmask = np.asarray(self.proteinComplexRNAComposition)[:, i].astype(
                bool
            )
            mmask = np.asarray(self.proteinComplexMonomerComposition)[
                :, i
            ].astype(bool)
            n = int(
                np.floor(
                    min(
                        np.r_[
                            np.asarray(self.substrates).reshape(-1)[
                                _idx(self.substrateIndexs_gtp)[0]
                            ]
                            / gtp,
                            np.asarray(self.substrates).reshape(-1)[
                                _idx(self.substrateIndexs_water)[0]
                            ]
                            / gtp,
                            np.asarray(self.RNAs).reshape(-1)[rmask],
                            np.asarray(self.monomers).reshape(-1)[mmask],
                        ]
                    )
                )
            )
            if n <= 0 or not np.all(
                np.asarray(self.enzymes).reshape(-1)[
                    np.asarray(self.complexationCatalysisMatrix)[:, i]
                ]
            ):
                continue
            self.complexs[i, 0] += n
            self.RNAs[:, 0, :] -= (
                np.asarray(self.proteinComplexRNAComposition)[:, i, None] * n
            )
            self.monomers[:, 0, :] -= (
                np.asarray(self.proteinComplexMonomerComposition)[:, i, None]
                * n
            )
            self.substrates = (
                np.asarray(self.substrates)
                + np.array([-1, 1, 1, -1, 1]).reshape(
                    np.asarray(self.substrates).shape, order="F"
                )
                * n
                * gtp
            )

    def getGtpPerComplex(self, i):
        return int(
            np.asarray(self.complexationCatalysisMatrix)[:, int(i) - 1].sum()
        )

    def getDryWeight(self):
        base = np.asarray(super().getDryWeight())
        weights = (
            (
                self.rna,
                self.rna.matureIndexs,
                self.rna.matureRRNAIndexs,
                self.rnaGlobalIndexs,
                self.RNAs,
            ),
            (
                self.monomer,
                self.monomer.matureIndexs,
                None,
                self.monomerGlobalIndexs,
                self.monomers,
            ),
            (
                self.complex,
                self.complex.nascentIndexs,
                None,
                self.complexGlobalIndexs,
                self.complexs,
            ),
        )
        extra = 0
        for state, kind, subkind, globals_, counts in weights:
            ix = _idx(globals_)
            if subkind is not None:
                ix = _idx(subkind)[ix]
            ix = np.asarray(kind)[ix] - 1
            molecular = np.asarray(state.molecularWeights).reshape(
                -1, order="F"
            )[ix]
            extra = extra + np.tensordot(
                molecular, np.asarray(counts), axes=(0, 0)
            )
        return base + extra / 6.022141793e23
