"""Runtime translation process, preserving MATLAB's one based indices."""

from __future__ import annotations

import numpy as np
from ..process_base import Process, _get, _set


def _matlab_round(value):
    """MATLAB round semantics (half away from zero)."""
    value = np.asarray(value, dtype=float)
    return np.sign(value) * np.floor(np.abs(value) + 0.5)


class Translation(Process):
    optionNames__ = []
    fixedConstantNames__ = [
        "ribosomeElongationRate",
        "tmRNABindingProbability",
    ]
    fittedConstantNames__ = []
    localStateNames__ = [
        "mRNAs",
        "freeTRNAs",
        "aminoacylatedTRNAs",
        "aminoacylatedTMRNA",
        "monomers",
    ]
    substrateWholeCellModelIDs = [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "FMET",
        "GTP",
        "GDP",
        "PI",
        "H2O",
        "H",
    ]
    enzymeWholeCellModelIDs = [
        "MG_173_MONOMER",
        "MG_142_MONOMER",
        "MG_196_MONOMER",
        "MG_089_DIMER",
        "MG_026_MONOMER",
        "MG_451_DIMER",
        "MG_433_DIMER",
        "MG_258_MONOMER",
        "MG_435_MONOMER",
        "RIBOSOME_30S",
        "RIBOSOME_30S_IF3",
        "RIBOSOME_50S",
        "RIBOSOME_70S",
        "MG_0004",
        "MG_059_MONOMER",
        "MG_083_MONOMER",
    ]
    substrateIndexs_aminoAcids = np.arange(1, 21)
    substrateIndexs_fMet, substrateIndexs_gtp, substrateIndexs_gdp = 21, 22, 23
    (
        substrateIndexs_phosphate,
        substrateIndexs_water,
        substrateIndexs_hydrogen,
    ) = (
        24,
        25,
        26,
    )
    enzymeIndexs_translationFactors = np.arange(1, 10)
    enzymeIndexs_initiationFactors = np.arange(1, 4)
    (
        enzymeIndexs_initiationFactor1,
        enzymeIndexs_initiationFactor2,
        enzymeIndexs_initiationFactor3,
    ) = (1, 2, 3)
    enzymeIndexs_elongationFactors = np.arange(4, 8)
    (
        enzymeIndexs_elongationGFactor,
        enzymeIndexs_terminationFactor,
        enzymeIndexs_recyclingFactor,
    ) = (4, 8, 9)
    (
        enzymeIndexs_ribosome30S,
        enzymeIndexs_ribosome30SIF3,
        enzymeIndexs_ribosome50S,
        enzymeIndexs_ribosome70S,
    ) = (10, 11, 12, 13)
    (
        enzymeIndexs_tmRNA,
        enzymeIndexs_tmRNABindingProtein,
        enzymeIndexs_peptidylTRNAHydrolase,
    ) = (14, 15, 16)

    def __init__(
        self,
        wholeCellModelID="Process_Translation",
        name="Translation",
        **props,
    ):
        super().__init__(wholeCellModelID, name, **props)
        self.polypeptide = self.ribosome = None
        for name, default in (
            ("ribosomeElongationRate", None),
            ("tmRNABindingProbability", None),
        ):
            if not hasattr(self, name):
                setattr(self, name, default)

    def storeObjectReferences(self, simulation):
        super().storeObjectReferences(simulation)
        fn = getattr(simulation, "state", None)
        states = _get(simulation, "states") if not callable(fn) else None
        self.polypeptide = (
            fn("Polypeptide") if callable(fn) else states["Polypeptide"]
        )
        self.ribosome = fn("Ribosome") if callable(fn) else states["Ribosome"]
        self.states += [self.polypeptide, self.ribosome]

    def initializeConstants(
        self, knowledgeBase=None, simulation=None, options=None, **kwargs
    ):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if knowledgeBase is not None or simulation is not None:
            if knowledgeBase is None or simulation is None:
                raise ValueError(
                    "knowledgeBase and simulation must be supplied together"
                )
            super().initializeConstants(
                knowledgeBase, simulation, options or {}
            )
            rna = self.rna
            enzyme_rna_local = np.asarray(
                self.enzymeRNALocalIndexs, dtype=int
            ).ravel()
            enzyme_rna_global = np.asarray(
                self.enzymeRNAGlobalIndexs, dtype=int
            ).ravel()
            tmrna = np.flatnonzero(
                enzyme_rna_local == int(self.enzymeIndexs_tmRNA)
            )
            if not tmrna.size:
                raise ValueError(
                    "Translation: tmRNA enzyme is absent from the RNA enzyme map"
                )
            self.tmRNAGlobalIndex = int(enzyme_rna_global[tmrna[0]])
            ids = np.asarray(
                _get(rna, "wholeCellModelIDs"), dtype=object
            ).ravel(order="F")
            mature = np.asarray(_get(rna, "matureIndexs"), dtype=int).ravel(
                order="F"
            )
            tRNA_roster = np.asarray(
                _get(rna, "matureTRNAIndexs"), dtype=int
            ).ravel(order="F")
            aminoacylated = np.asarray(
                _get(rna, "aminoacylatedIndexs"), dtype=int
            ).ravel(order="F")
            self.freeTRNAWholeCellModelIDs = ids[
                mature[tRNA_roster - 1] - 1
            ].reshape(-1, 1)
            self.aminoacylatedTRNAWholeCellModelIDs = ids[
                aminoacylated[tRNA_roster - 1] - 1
            ].reshape(-1, 1)
            self.aminoacylatedTMRNAWholeCellModelID = ids[
                aminoacylated[self.tmRNAGlobalIndex - 1] - 1
            ]
            absent = [
                x
                for x in ("ribosomeElongationRate", "tmRNABindingProbability")
                if not hasattr(self, x) or getattr(self, x) is None
            ]
        else:
            required = (
                "tmRNAGlobalIndex",
                "ribosomeElongationRate",
                "tmRNABindingProbability",
            )
            absent = [
                x
                for x in required
                if not hasattr(self, x) or getattr(self, x) is None
            ]
        if absent:
            raise NotImplementedError(
                "Translation KB initialization requires KnowledgeBase.computeTRNASequences and exported RNA/enzyme mappings"
                + (": " + ", ".join(absent) if absent else "")
            )

    def copyFromState(self):
        super().copyFromState()
        r, m = self.rna, self.monomer
        cyt = (
            int(np.asarray(self.compartment.cytosolIndexs).reshape(-1)[0]) - 1
        )
        counts = np.asarray(r.counts)
        counts = counts[:, :, None] if counts.ndim == 2 else counts
        mature = np.asarray(r.matureIndexs, int).reshape(-1) - 1
        gene_rows = np.asarray(self.gene.mRNAIndexs, int).reshape(-1) - 1
        mature_mrna = np.asarray(r.matureMRNAIndexs, int).reshape(-1) - 1
        mrna_comp = np.asarray(r.matureRNAGeneComposition)[
            np.ix_(gene_rows, mature_mrna)
        ]
        self.mRNAs = np.asarray(
            mrna_comp @ counts[mature[mature_mrna], cyt, :]
        ).reshape((len(self.gene.mRNAIndexs), 1, -1), order="F")
        mc = np.asarray(m.counts)
        mc = mc[:, :, None] if mc.ndim == 2 else mc
        self.monomers = mc[
            np.asarray(m.nascentIndexs, int).reshape(-1) - 1, cyt, :
        ]
        trna = np.asarray(r.matureTRNAIndexs, int).reshape(-1) - 1
        tm = np.asarray(self.tmRNAGlobalIndex, int).reshape(-1) - 1
        self.freeTRNAs = counts[mature[trna], cyt, :]
        self.freeTMRNA = counts[mature[tm], cyt, :]
        amino = np.asarray(r.aminoacylatedIndexs, int).reshape(-1) - 1
        bound = np.asarray(r.boundIndexs, int).reshape(-1) - 1
        self.aminoacylatedTRNAs = counts[amino[trna], cyt, :]
        self.aminoacylatedTMRNA = counts[amino[tm], cyt, :]
        self.boundTMRNA = counts[bound[tm], cyt, :]

    def copyToState(self):
        super().copyToState()
        r, m = self.rna, self.monomer
        cyt = (
            int(np.asarray(self.compartment.cytosolIndexs).reshape(-1)[0]) - 1
        )
        rc = np.asarray(r.counts)
        rc = rc[:, :, None] if rc.ndim == 2 else rc
        mc = np.asarray(m.counts)
        mc = mc[:, :, None] if mc.ndim == 2 else mc
        trna = np.asarray(r.matureTRNAIndexs, int).reshape(-1) - 1
        tm = np.asarray(self.tmRNAGlobalIndex, int).reshape(-1) - 1
        mat = np.asarray(r.matureIndexs, int).reshape(-1) - 1
        aa = np.asarray(r.aminoacylatedIndexs, int).reshape(-1) - 1
        bi = np.asarray(r.boundIndexs, int).reshape(-1) - 1
        mc[np.asarray(m.nascentIndexs, int).reshape(-1) - 1, cyt, :] = (
            self.monomers
        )
        rc[mat[trna], cyt, :] = self.freeTRNAs
        rc[mat[tm], cyt, :] = self.freeTMRNA
        rc[aa[trna], cyt, :] = self.aminoacylatedTRNAs
        rc[aa[tm], cyt, :] = self.aminoacylatedTMRNA
        rc[bi[tm], cyt, :] = self.boundTMRNA
        _set(r, "counts", rc)
        _set(m, "counts", mc)

    def allocateMemoryForState(self, numTimePoints=1):
        super().allocateMemoryForState(numTimePoints)
        n = int(numTimePoints)
        self.monomers = np.zeros((len(self.polypeptide.monomerLengths), 1, n))
        self.mRNAs = np.zeros((len(self.gene.mRNAIndexs), 1, n))
        self.freeTRNAs = np.zeros((len(self.rna.matureTRNAIndexs), 1, n))
        self.freeTMRNA = np.zeros((1, 1, n))
        self.aminoacylatedTRNAs = np.zeros_like(self.freeTRNAs)
        self.aminoacylatedTMRNA = np.zeros((1, 1, n))
        self.boundTMRNA = np.zeros((1, 1, n))

    def initializeState(self):
        """Initialize 70S ribosomes and distribute them along translated mRNAs."""
        rib, pol = self.ribosome, self.polypeptide
        for rib_name, pol_name in (
            ("boundMRNAs", "boundMRNAs"),
            ("mRNAPositions", "nascentMonomerLengths"),
            ("tmRNAPositions", "proteolysisTagLengths"),
        ):
            if not np.array_equal(_get(rib, rib_name), _get(pol, pol_name)):
                raise RuntimeError(
                    "Ribosomes and polypeptides must be synchronized"
                )

        enzymes = np.asarray(self.enzymes).reshape(-1).copy()
        bound_enzymes = np.asarray(self.boundEnzymes).reshape(-1).copy()
        enzymes += bound_enzymes
        bound_enzymes[:] = 0
        r30, r30if3, r50, r70 = (
            self.enzymeIndexs_ribosome30S - 1,
            self.enzymeIndexs_ribosome30SIF3 - 1,
            self.enzymeIndexs_ribosome50S - 1,
            self.enzymeIndexs_ribosome70S - 1,
        )
        i3 = self.enzymeIndexs_initiationFactor3 - 1
        enzymes[r30] += enzymes[r70]
        enzymes[r50] += enzymes[r70]
        enzymes[r70] = 0
        enzymes[r30] += enzymes[r30if3]
        enzymes[i3] += enzymes[r30if3]
        enzymes[r30if3] = 0

        free_factors = enzymes[:9].copy()
        bound_factors = bound_enzymes[:9].copy()
        r30_count, r30if3_count, r50_count, r70_count = enzymes[
            [r30, r30if3, r50, r70]
        ]
        n_mrnas = int(np.asarray(self.mRNAs).sum())
        n_bound = int(
            min(r30_count, r50_count, free_factors[3:7].min(), n_mrnas)
        )
        r30_count -= n_bound
        r50_count -= n_bound
        free_factors[3:7] -= n_bound
        bound_factors[3:7] += n_bound

        mrna_counts = np.asarray(self.mRNAs).reshape(-1, order="F")
        lengths = np.asarray(pol.monomerLengths, dtype=float).reshape(-1)
        if len(mrna_counts) != len(lengths):
            raise ValueError(
                "mRNA counts and monomer lengths must have equal length"
            )
        states = np.full(2 * n_bound, rib.notExistValue, dtype=int)
        bound_mrnas = np.zeros(2 * n_bound, dtype=int)
        nascent = np.zeros(2 * n_bound, dtype=int)
        tags = np.zeros(2 * n_bound, dtype=int)
        weights = mrna_counts * lengths
        elongation_rate = float(np.asarray(self.ribosomeElongationRate).item())
        for i in range(n_bound):
            choice = (
                int(
                    np.asarray(
                        self.randStream.randsample(
                            len(weights), 1, True, weights
                        )
                    ).reshape(-1)[0]
                )
                - 1
            )
            bound_mrnas[i] = choice + 1
        for i, length in enumerate(lengths, start=1):
            rows = np.flatnonzero(bound_mrnas == i)
            if not len(rows):
                continue
            partitions = _matlab_round(
                np.arange(len(rows) + 1)
                * (length + elongation_rate - 1)
                / len(rows)
                - elongation_rate / 2
            ).astype(int)
            for j, row in enumerate(rows):
                low = partitions[j] + elongation_rate / 2
                high = partitions[j + 1] - elongation_rate / 2
                states[row] = rib.activeValue
                if high < low:
                    nascent[row] = int(
                        _matlab_round(
                            max(1, min(length, np.mean(partitions[j : j + 2])))
                        )
                    )
                else:
                    nascent[row] = int(
                        _matlab_round(
                            low + float(self.randStream.rand()) * (high - low)
                        )
                    )

        _set(rib, "states", states[:, None])
        _set(pol, "boundMRNAs", bound_mrnas[:, None])
        _set(pol, "nascentMonomerLengths", nascent[:, None])
        _set(pol, "proteolysisTagLengths", tags[:, None])
        _set(rib, "boundMRNAs", bound_mrnas[:, None].copy())
        _set(rib, "mRNAPositions", nascent[:, None].copy())
        _set(rib, "tmRNAPositions", tags[:, None].copy())
        enzymes[:9] = free_factors
        enzymes[[r30, r30if3, r50, r70]] = [
            r30_count,
            r30if3_count,
            r50_count,
            r70_count,
        ]
        bound_enzymes[:9] = bound_factors
        bound_enzymes[r70] = n_bound
        self.enzymes = enzymes.reshape(np.shape(self.enzymes))
        self.boundEnzymes = bound_enzymes.reshape(np.shape(self.boundEnzymes))
        self.copyToState()

        monomer_counts = np.asarray(self.monomer.counts).copy()
        complex_counts = np.asarray(self.complex.counts).copy()
        # copyToState may retain a singleton time axis; external-state updates
        # consume the live molecule-by-compartment matrix.
        if monomer_counts.ndim == 3 and monomer_counts.shape[2] == 1:
            monomer_counts = monomer_counts[:, :, 0]
        if complex_counts.ndim == 3 and complex_counts.shape[2] == 1:
            complex_counts = complex_counts[:, :, 0]
        initial_monomer_counts = monomer_counts.copy()
        initial_complex_counts = complex_counts.copy()
        target_weight = max(
            0.0,
            float(np.asarray(self.monomer.dryWeight).sum())
            + float(np.asarray(self.complex.dryWeight).sum())
            - float(self.polypeptide.dryWeight),
        )
        mature_weight = float(
            np.asarray(self.monomer.dryWeight).sum()
        ) + float(np.asarray(self.complex.dryWeight).sum())
        while mature_weight > target_weight:
            flat_monomers = monomer_counts.ravel(order="F")
            flat_complexes = complex_counts.ravel(order="F")
            populations = np.r_[flat_monomers, flat_complexes]
            if not np.any(populations > 0):
                break
            selected = (
                int(
                    np.asarray(
                        self.randStream.randsample(
                            len(populations), 1, False, populations
                        )
                    ).reshape(-1)[0]
                )
                - 1
            )
            if selected < len(flat_monomers):
                index = np.unravel_index(
                    selected, monomer_counts.shape, order="F"
                )
                monomer_counts[index] -= 1
                weights_m = np.asarray(self.monomer.molecularWeights).reshape(
                    -1, order="F"
                )
                weight = float(weights_m[index[0]])
            else:
                ci = selected - len(flat_monomers)
                index = np.unravel_index(ci, complex_counts.shape, order="F")
                complex_counts[index] -= 1
                n_complexes = len(
                    np.asarray(self.complex.matureIndexs).reshape(-1)
                )
                species = index[0]
                weights_c = np.asarray(self.complex.molecularWeights).reshape(
                    -1, order="F"
                )
                weight = float(weights_c[species])
                composition = np.asarray(
                    self.complex.proteinComplexComposition
                )
                gene_mrna = (
                    np.asarray(self.gene.mRNAIndexs, dtype=int).reshape(-1) - 1
                )
                non_mrna_gene = np.setdiff1d(
                    np.arange(composition.shape[0]), gene_mrna
                )
                component_counts = composition[
                    non_mrna_gene, species % n_complexes, :
                ].sum(axis=-1)
                rna_non_mrna = np.setdiff1d(
                    np.arange(
                        len(np.asarray(self.rna.matureIndexs).reshape(-1))
                    ),
                    np.asarray(self.rna.matureMRNAIndexs, dtype=int).reshape(
                        -1
                    )
                    - 1,
                )
                if component_counts.size and np.any(component_counts):
                    if component_counts.size != rna_non_mrna.size:
                        raise ValueError(
                            "Complex non-mRNA subunit composition does not match mature non-mRNA species"
                        )
                    rna_counts = np.asarray(self.rna.counts)
                    rna_rows = (
                        np.asarray(self.rna.matureIndexs, dtype=int).reshape(
                            -1
                        )[rna_non_mrna]
                        - 1
                    )
                    cyt = (
                        int(
                            np.asarray(self.compartment.cytosolIndexs).reshape(
                                -1
                            )[0]
                        )
                        - 1
                    )
                    rna_counts[rna_rows, cyt] += component_counts
                    _set(self.rna, "counts", rna_counts)
            mature_weight -= weight / 6.0221417930e23
        _set(self.monomer, "counts", monomer_counts)
        _set(self.complex, "counts", complex_counts)
        delta_monomers = monomer_counts - initial_monomer_counts
        delta_complexes = complex_counts - initial_complex_counts
        if callable(getattr(self.monomer, "updateExternalState", None)):
            updated = self.monomer.updateExternalState(delta_monomers, True)
            _set(
                self.monomer,
                "counts",
                np.asarray(self.monomer.counts) + np.asarray(updated),
            )
        if callable(getattr(self.complex, "updateExternalState", None)):
            updated = self.complex.updateExternalState(delta_complexes, True)
            _set(
                self.complex,
                "counts",
                np.asarray(self.complex.counts) + np.asarray(updated),
            )
        self.copyFromState()

    def calcResourceRequirements_Current(self):
        e = np.asarray(self.enzymes).reshape(-1)
        b = np.asarray(self.boundEnzymes).reshape(-1)
        nR = float(e[12] + b[12] + min(e[10] + min(e[9], e[2]), e[11]))
        nEl = float(
            min(
                np.asarray(self.aminoacylatedTRNAs).sum()
                + np.asarray(self.aminoacylatedTMRNA).sum(),
                nR * self.ribosomeElongationRate,
            )
        )
        x = nR + 2 * nEl + 3 * nR
        out = np.zeros_like(np.asarray(self.substrates))
        out.reshape(-1)[21] = x
        out.reshape(-1)[24] = nEl
        return out

    def calcResourceRequirements_LifeCycle(self, *args):
        states = args[-1]
        lengths = np.asarray(self.polypeptide.monomerLengths).reshape(-1)
        prod = np.asarray(_get(states, "monomerProductions")).reshape(-1)
        prod0 = np.asarray(_get(states, "monomerProductions0")).reshape(-1)
        bm = np.zeros(26)
        by = np.zeros(26)
        water = float(lengths @ prod)
        energy = 2 * water + 3 * prod.sum()
        bm[24] = water + energy
        by[24] = np.maximum(0, lengths - 1) @ prod
        by[25] = water + energy
        by[22:24] = energy
        frac = (
            self.ribosomeElongationRate * (prod0 / lengths).sum() / prod0.sum()
        )
        aa = prod0 @ lengths
        enz = np.zeros(16)
        rib = 1.65 * aa / self.ribosomeElongationRate / (1 - frac)
        enz[9] = enz[11] = rib
        enz[:3] = 3 * rib * frac
        enz[3:7] = 3 * rib * (1 - frac)
        enz[7:9] = 3
        enz[3] += 3
        enz[13:16] = 3
        return bm, by, enz, np.full(16, np.inf)

    def evolveState(self):
        # Imported here so the process module remains importable while the
        # polymerization helper is supplied independently.
        from ..polymerize import polymerize

        rib, pol = self.ribosome, self.polypeptide

        def arr(obj, name):
            return np.asarray(_get(obj, name)).reshape(-1)

        def drawperm(n):
            rs = self.randStream
            if hasattr(rs, "randperm"):
                v = np.asarray(rs.randperm(n), int).reshape(-1)
                return v - 1 if len(v) and np.min(v) == 1 else v
            if hasattr(rs, "permutation"):
                return np.asarray(rs.permutation(n), int).reshape(-1)
            return np.random.permutation(n)

        def rand():
            rs = self.randStream
            return float(rs.rand() if hasattr(rs, "rand") else rs.random())

        states = arr(rib, "states").astype(int)
        bound = arr(pol, "boundMRNAs").astype(int)
        mpos = arr(pol, "nascentMonomerLengths").astype(int)
        tpos = arr(pol, "proteolysisTagLengths").astype(int)
        if not (
            np.array_equal(arr(rib, "boundMRNAs"), bound)
            and np.array_equal(arr(rib, "mRNAPositions"), mpos)
            and np.array_equal(arr(rib, "tmRNAPositions"), tpos)
        ):
            raise ValueError("Ribosomes and polypeptides must be synchronized")

        enzymes = np.asarray(self.enzymes).reshape(-1).copy()
        boundEnz = np.asarray(self.boundEnzymes).reshape(-1).copy()
        # MATLAB's one based enzyme and substrate indices, converted at access.
        ft = enzymes[:9].copy()
        bt = boundEnz[:9].copy()
        r30, r30if3, r50, r70 = (
            enzymes[9],
            enzymes[10],
            enzymes[11],
            enzymes[12],
        )
        br70 = boundEnz[12]
        freeTm = float(np.asarray(self.freeTMRNA).sum())
        bindProtein = enzymes[14]
        hydrolase = enzymes[15]
        boundTm = float(np.asarray(self.boundTMRNA).sum())
        aaTm = float(np.asarray(self.aminoacylatedTMRNA).sum())
        new = min(r30, ft[2])
        r30if3 += new
        r30 -= new
        ft[2] -= new
        if br70 + r30if3 > len(states):
            grow = int(br70 + r30if3)
            states = np.r_[
                states, np.full(grow, int(_get(rib, "notExistValue")))
            ]
            bound = np.r_[bound, np.zeros(grow, int)]
            mpos = np.r_[mpos, np.zeros(grow, int)]
            tpos = np.r_[tpos, np.zeros(grow, int)]
        active = int(_get(rib, "activeValue"))
        stalled = int(_get(rib, "stalledValue"))
        absent = int(_get(rib, "notExistValue"))
        act = np.flatnonzero((states == active) | (states == stalled))
        nonexist = np.flatnonzero(states == absent)
        rate = int(self.ribosomeElongationRate)
        lengths = arr(pol, "monomerLengths").astype(int)
        tagLength = int(_get(pol, "proteolysisTagLength"))
        term = (states == active) & (lengths[np.maximum(bound - 1, 0)] == mpos)
        term |= tpos == tagLength
        mrnas = np.asarray(self.mRNAs).reshape(-1).copy()
        ntrna = len(np.asarray(self.freeTRNAs).reshape(-1))
        totalTrna = np.asarray(self.freeTRNAs).reshape(-1) + np.asarray(
            self.aminoacylatedTRNAs
        ).reshape(-1)
        aa = np.asarray(self.aminoacylatedTRNAs).reshape(-1).copy()
        substrates = np.asarray(self.substrates).reshape(-1).copy()
        energy0 = max(0, substrates[21])
        energy = energy0
        water = substrates[24]
        ft[3:7] += bt[3:7]
        bt[3:7] = 0
        nEl = min(len(act), int(br70), int(np.floor(min(energy, water) / 2)))
        elng = np.zeros(len(states), bool)
        seq = np.zeros((len(states), rate), int)
        order = drawperm(int(br70))
        for j in range(nEl):
            i = act[order[j]]
            if not term[i]:
                if not np.all(ft[3:7]):
                    continue
                ft[3:7] -= 1
                bt[3:7] += 1
            if states[i] == active:
                mi = bound[i] - 1
                start = mpos[i]
                end = min(lengths[mi], start + rate)
                trseq = np.asarray(
                    _get(pol, "monomerTRNASequences"), dtype=object
                ).reshape(-1)[mi]
                seq[i, : end - start] = np.asarray(trseq).reshape(-1)[
                    start:end
                ]
            elif states[i] == stalled:
                trseq = np.asarray(
                    _get(pol, "proteolysisTagTRNASequence")
                ).reshape(-1)
                start = tpos[i]
                end = min(tagLength, start + rate)
                seq[i, : end - start] = trseq[start:end]
            elng[i] = True
        nel = int(elng.sum())
        ninit = max(
            0,
            int(
                min(
                    r30if3,
                    r50,
                    ft[0],
                    ft[1],
                    min(energy, water) - 2 * nel,
                    mrnas.sum(),
                )
            ),
        )
        if ninit:
            r30if3 -= ninit
            r50 -= ninit
            br70 += ninit
            ft[2] += ninit
            energy -= ninit
            water -= ninit
            bprobs = mrnas.copy()
            for i in nonexist[:ninit]:
                if hasattr(self.randStream, "randsample"):
                    choice = (
                        int(
                            self.randStream.randsample(
                                len(bprobs), 1, True, bprobs
                            )
                        )
                        - 1
                    )
                else:
                    choice = int(
                        np.random.choice(len(bprobs), p=bprobs / bprobs.sum())
                    )
                bound[i] = choice + 1
                states[i] = active
                bprobs[choice] = max(0, bprobs[choice] - 1)
        ninit0 = int(np.sum((mpos[elng] + tpos[elng]) == 0))
        aw = int(np.floor(min(energy, water - ninit0)))
        progress, aa, cost, aw2, _ = polymerize(
            seq[elng], aa, np.arange(1, ntrna + 1), 0, aw, 2, self.randStream
        )
        progress = np.asarray(progress).reshape(-1)
        cost = np.asarray(cost).reshape(-1)
        if np.any(progress):
            water -= (
                aw
                - aw2
                + int(
                    np.sum(((mpos[elng] + tpos[elng]) == 0) & (progress > 0))
                )
            )
            energy -= aw - aw2
            mpos[elng] += progress * (states[elng] == active)
            tpos[elng] += progress * (states[elng] == stalled)
        randomOrder = drawperm(len(elng))
        tmp = np.zeros(len(states), int)
        tmp[elng] = progress
        aborted = np.asarray(_get(pol, "abortedPolypeptides"))
        for i in randomOrder:
            if elng[i] and (tmp[i] or mpos[i] >= lengths[bound[i] - 1]):
                if (
                    (mpos[i] >= lengths[bound[i] - 1] or tpos[i] >= tagLength)
                    and ft[7]
                    and ft[8]
                    and (ft[3] or tmp[i])
                    and ft[2]
                    and energy >= 2
                    and water >= 2
                ):
                    energy -= 2
                    water -= 2
                    if states[i] == active:
                        self.monomers.reshape(-1)[bound[i] - 1] += 1
                    elif states[i] == stalled:
                        aborted = (
                            np.vstack(
                                (aborted, [[bound[i], mpos[i], tagLength]])
                            )
                            if aborted.size
                            else np.array([[bound[i], mpos[i], tagLength]])
                        )
                        boundTm -= 1
                        freeTm += 1
                    states[i] = absent
                    mpos[i] = tpos[i] = bound[i] = 0
                    br70 -= 1
                    r30 += 1
                    r50 += 1
            elif (
                states[i] == active
                and mpos[i] > 0
                and aaTm > 0
                and bindProtein > 0
                and hydrolase > 0
                and rand() < self.tmRNABindingProbability
            ):
                states[i] = stalled
                boundTm += 1
                aaTm -= 1
                tpos[i] = 1
        _set(pol, "abortedPolypeptides", aborted)
        _set(pol, "boundMRNAs", bound[:, None])
        _set(pol, "nascentMonomerLengths", mpos[:, None])
        _set(pol, "proteolysisTagLengths", tpos[:, None])
        _set(rib, "states", states[:, None])
        _set(rib, "boundMRNAs", bound[:, None].copy())
        _set(rib, "mRNAPositions", mpos[:, None].copy())
        _set(rib, "tmRNAPositions", tpos[:, None].copy())
        enzymes[:9] = ft
        boundEnz[:9] = bt
        enzymes[9] = r30
        enzymes[10] = r30if3
        enzymes[11] = r50
        enzymes[12] = r70
        boundEnz[12] = br70
        enzymes[14] = bindProtein
        enzymes[15] = hydrolase
        self.enzymes = enzymes.reshape(np.shape(self.enzymes))
        self.boundEnzymes = boundEnz.reshape(np.shape(self.boundEnzymes))
        self.freeTMRNA = np.asarray(freeTm).reshape(np.shape(self.freeTMRNA))
        self.boundTMRNA = np.asarray(boundTm).reshape(
            np.shape(self.boundTMRNA)
        )
        self.aminoacylatedTMRNA = np.asarray(aaTm).reshape(
            np.shape(self.aminoacylatedTMRNA)
        )
        self.aminoacylatedTRNAs = aa.reshape(np.shape(self.aminoacylatedTRNAs))
        self.freeTRNAs = (totalTrna - aa).reshape(np.shape(self.freeTRNAs))
        substrates[21] -= energy0 - energy
        substrates[24] = water
        substrates[22] += energy0 - energy
        substrates[23] += energy0 - energy
        substrates[25] += (energy0 - energy) + cost.sum()
        self.substrates = substrates.reshape(np.shape(self.substrates))

    def getDryWeight(self):
        result = super().getDryWeight()
        av = 6.0221417930e23
        add = (
            np.asarray(self.monomers)
            .reshape((len(self.monomer.nascentIndexs), -1), order="F")
            .T
            @ np.asarray(self.monomer.molecularWeights).reshape(-1)[
                np.asarray(self.monomer.nascentIndexs, int).reshape(-1) - 1
            ]
        )
        add += (
            np.asarray(self.freeTRNAs)
            .reshape((len(self.rna.matureTRNAIndexs), -1), order="F")
            .T
            @ np.asarray(self.rna.molecularWeights).reshape(-1)[
                np.asarray(self.rna.matureIndexs, int).reshape(-1)[
                    np.asarray(self.rna.matureTRNAIndexs, int).reshape(-1) - 1
                ]
                - 1
            ]
        )
        add += (
            np.asarray(self.aminoacylatedTRNAs)
            .reshape((len(self.rna.matureTRNAIndexs), -1), order="F")
            .T
            @ np.asarray(self.rna.molecularWeights).reshape(-1)[
                np.asarray(self.rna.aminoacylatedIndexs, int).reshape(-1)[
                    np.asarray(self.rna.matureTRNAIndexs, int).reshape(-1) - 1
                ]
                - 1
            ]
        )
        tmrna_aminoacylated_index = int(
            np.asarray(self.rna.aminoacylatedIndexs).reshape(-1)[
                int(np.asarray(self.tmRNAGlobalIndex).reshape(-1)[0]) - 1
            ]
        )
        add += (
            np.asarray(self.aminoacylatedTMRNA).reshape((1, -1), order="F").T
            * np.asarray(self.rna.molecularWeights).reshape(-1)[
                tmrna_aminoacylated_index - 1
            ]
        ).reshape(-1)
        return result + add.reshape((1, 1, -1)) / av
