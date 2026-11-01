"""Protein misfolding, refolding, complex and polypeptide decay.

Port of WholeCell/src/+edu/+stanford/+covert/+cell/+sim/+process/ProteinDecay.m.
Exported indices remain MATLAB one based; conversions are local to indexing.
"""

from __future__ import annotations
import numpy as np
from ..process_base import Process, _cell_text, _get, _set


class ProteinDecay(Process):
    stimuliWholeCellModelIDs = ()
    optionNames__ = ()
    fixedConstantNames__ = (
        "lonProteaseSpecificRate",
        "lonProteaseEnergyCost",
        "lonProteaseFragmentLength",
        "ftsHProteaseSpecificRate",
        "ftsHProteaseEnergyCost",
        "ftsHProteaseFragmentLength",
        "oligoendopeptidaseFSpecificRate",
        "proteinMisfoldingRate",
        "proteinComplexRNAComposition",
        "proteinComplexMonomerComposition",
        "monomerLonProteaseCleavages",
        "monomerDecayReactions",
        "complexDecayReactions",
    )
    fittedConstantNames__ = ()
    localStateNames__ = ("monomers", "complexs", "RNAs")
    enzymeWholeCellModelIDs = (
        "MG_355_HEXAMER",
        "MG_239_HEXAMER",
        "MG_457_HEXAMER",
        "MG_183_MONOMER",
        "MG_324_MONOMER",
        "MG_391_HEXAMER",
        "MG_208_DIMER",
        "MG_046_DIMER",
        "MG_020_MONOMER",
    )
    (
        enzymeIndexs_clpBProtease,
        enzymeIndexs_lonProtease,
        enzymeIndexs_ftsHProtease,
    ) = (
        1,
        2,
        3,
    )
    enzymeIndexs_peptidases = np.arange(4, 10)
    _ENERGY_VECTOR = np.array([1.0, -1.0, -1.0, 1.0, -1.0])

    def __init__(
        self,
        wholeCellModelID="Process_ProteinDecay",
        name="Protein Decay",
        **properties,
    ):
        super().__init__(wholeCellModelID, name, **properties)
        self.polypeptide = getattr(self, "polypeptide", None)

    def storeObjectReferences(self, simulation):
        super().storeObjectReferences(simulation)
        fn = getattr(simulation, "state", None)
        self.polypeptide = (
            fn("Polypeptide")
            if callable(fn)
            else _get(simulation, "states")["Polypeptide"]
        )
        self.states = self.states + [self.polypeptide]

    @staticmethod
    def _record_matrix(records, field):
        return np.column_stack(
            [
                np.asarray(_get(record, field), dtype=float).ravel(order="F")
                for record in records
            ]
        )

    def initializeConstants(
        self, knowledgeBase=None, simulation=None, **kwargs
    ):
        if knowledgeBase is not None or simulation is not None:
            if knowledgeBase is None or simulation is None:
                raise ValueError(
                    "knowledgeBase and simulation must be provided together"
                )
            monomers = np.asarray(
                _get(knowledgeBase, "proteinMonomers"), dtype=object
            ).ravel(order="F")
            complexes = np.asarray(
                _get(knowledgeBase, "proteinComplexs"), dtype=object
            ).ravel(order="F")
            metabolite_ids = np.asarray(
                _get(self.metabolite, "wholeCellModelIDs"), dtype=object
            ).ravel(order="F")
            monomer_reactions = {
                "nascent": self._record_matrix(monomers, "decayReaction"),
                "processedI": self._record_matrix(
                    monomers, "processedISequenceDecayReaction"
                ),
                "processedII": self._record_matrix(
                    monomers, "processedIISequenceDecayReaction"
                ),
                "signalSequence": self._record_matrix(
                    monomers, "signalSequenceDecayReaction"
                ),
                "folded": self._record_matrix(
                    monomers, "foldedSequenceDecayReaction"
                ),
                "mature": self._record_matrix(
                    monomers, "matureSequenceDecayReaction"
                ),
            }
            complex_nascent = self._record_matrix(complexes, "decayReaction")
            complex_mature = self._record_matrix(
                complexes, "matureDecayReaction"
            )
            met_lookup = {
                _cell_text(identifier): i
                for i, identifier in enumerate(metabolite_ids)
            }
            energy_rows = np.asarray(
                [met_lookup[name] for name in ("GTP", "H2O", "PI", "H")],
                dtype=int,
            )
            gdp_row = met_lookup["GDP"]
            for reactions in (complex_nascent, complex_mature):
                scaled = reactions[energy_rows] / np.array(
                    [[1.0], [1.0], [-1.0], [-1.0]]
                )
                matches = (scaled[0] != 0) & np.all(
                    scaled == scaled[0], axis=0
                )
                reactions[gdp_row, matches] = scaled[0, matches]
                reactions[energy_rows[0], matches] = 0

            active = np.zeros(len(metabolite_ids), dtype=bool)
            for reactions in (
                *monomer_reactions.values(),
                complex_nascent,
                complex_mature,
            ):
                if reactions.shape[0] != len(metabolite_ids):
                    raise ValueError(
                        "protein decay reaction rows must match Metabolite wholeCellModelIDs"
                    )
                active |= np.any(reactions != 0, axis=1)
            fixed = ("ATP", "ADP", "PI", "H", "H2O", "NH3", "FOR")
            self.substrateWholeCellModelIDs = np.unique(
                np.r_[metabolite_ids[active], np.asarray(fixed, dtype=object)]
            ).reshape(-1, 1)
            Process.initializeConstants(
                self,
                knowledgeBase,
                simulation,
                kwargs.pop("options", None) or {},
            )

            self.substrateIndexs_atp = self.substrateIndexs(["ATP"])
            self.substrateIndexs_adp = self.substrateIndexs(["ADP"])
            self.substrateIndexs_phosphate = self.substrateIndexs(["PI"])
            self.substrateIndexs_hydrogen = self.substrateIndexs(["H"])
            self.substrateIndexs_water = self.substrateIndexs(["H2O"])
            amino_acids = [
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
            ]
            self.substrateIndexs_aminoAcids = self.substrateIndexs(amino_acids)
            self.substrateIndexs_methionine = self.substrateIndexs(["MET"])
            self.substrateIndexs_fmethionine = self.substrateIndexs(["FMET"])
            self.substrateIndexs_glutamate = self.substrateIndexs(["GLU"])
            self.substrateIndexs_glutamine = self.substrateIndexs(["GLN"])
            self.substrateIndexs_ammonia = self.substrateIndexs(["NH3"])
            self.substrateIndexs_formate = self.substrateIndexs(["FOR"])

            rna_state = np.asarray(
                _get(self.rna, "wholeCellModelIDs"), dtype=object
            ).ravel(order="F")
            damaged = np.asarray(
                _get(self.rna, "damagedIndexs"), dtype=int
            ).ravel(order="F")
            mature_rna = np.r_[
                np.asarray(
                    _get(self.rna, "matureRRNAIndexs"), dtype=int
                ).ravel(order="F"),
                np.asarray(
                    _get(self.rna, "matureSRNAIndexs"), dtype=int
                ).ravel(order="F"),
            ]
            self.damagedRNAGlobalIndexs = damaged[mature_rna - 1].reshape(
                -1, 1
            )
            self.rnaWholeCellModelIDs = rna_state[
                self.damagedRNAGlobalIndexs.ravel() - 1
            ].reshape(-1, 1)

            cyt = (
                int(np.asarray(self.compartment.cytosolIndexs).ravel()[0]) - 1
            )
            all_rna = np.asarray(
                _get(knowledgeBase, "proteinComplexAllRNAComposition")
            )
            rna_gene_rows = (
                np.r_[
                    np.asarray(_get(self.gene, "rRNAIndexs"), dtype=int).ravel(
                        order="F"
                    ),
                    np.asarray(_get(self.gene, "sRNAIndexs"), dtype=int).ravel(
                        order="F"
                    ),
                ]
                - 1
            )
            complex_rna = all_rna[rna_gene_rows, :, cyt]
            all_monomer = np.asarray(
                _get(knowledgeBase, "proteinComplexAllMonomerComposition")
            )
            complex_monomer = all_monomer[:, :, cyt]
            n_state_complexes = len(
                np.asarray(
                    _get(self.complex, "wholeCellModelIDs"), dtype=object
                ).ravel(order="F")
            )
            self.proteinComplexRNAComposition = np.zeros(
                (len(rna_gene_rows), n_state_complexes)
            )
            self.proteinComplexMonomerComposition = np.zeros(
                (complex_monomer.shape[0], n_state_complexes)
            )
            for field in (
                "nascentIndexs",
                "matureIndexs",
                "inactivatedIndexs",
                "boundIndexs",
                "misfoldedIndexs",
                "damagedIndexs",
            ):
                state_indices = (
                    np.asarray(_get(self.complex, field), dtype=int).ravel(
                        order="F"
                    )
                    - 1
                )
                self.proteinComplexRNAComposition[:, state_indices] = (
                    complex_rna
                )
                self.proteinComplexMonomerComposition[:, state_indices] = (
                    complex_monomer
                )

            substrate_rows = (
                np.asarray(
                    self.substrateMetaboliteGlobalIndexs, dtype=int
                ).ravel(order="F")
                - 1
            )
            n_monomers = len(
                np.asarray(
                    _get(self.monomer, "wholeCellModelIDs"), dtype=object
                ).ravel(order="F")
            )
            n_complexes = len(
                np.asarray(
                    _get(self.complex, "wholeCellModelIDs"), dtype=object
                ).ravel(order="F")
            )
            self.monomerDecayReactions = np.zeros(
                (len(substrate_rows), n_monomers)
            )
            monomer_fields = (
                ("nascentIndexs", "nascent"),
                ("processedIIndexs", "processedI"),
                ("processedIIIndexs", "processedII"),
                ("signalSequenceIndexs", "signalSequence"),
                ("foldedIndexs", "folded"),
                ("matureIndexs", "mature"),
                ("inactivatedIndexs", "mature"),
                ("boundIndexs", "mature"),
                ("misfoldedIndexs", "mature"),
                ("damagedIndexs", "mature"),
            )
            for state_field, reaction_field in monomer_fields:
                columns = (
                    np.asarray(
                        _get(self.monomer, state_field), dtype=int
                    ).ravel(order="F")
                    - 1
                )
                source_columns = np.arange(
                    monomer_reactions[reaction_field].shape[1]
                )
                self.monomerDecayReactions[:, columns] = monomer_reactions[
                    reaction_field
                ][np.ix_(substrate_rows, source_columns)]
            self.complexDecayReactions = np.zeros(
                (len(substrate_rows), n_complexes)
            )
            for state_field, reactions in (
                ("nascentIndexs", complex_nascent),
                ("matureIndexs", complex_mature),
                ("inactivatedIndexs", complex_mature),
                ("boundIndexs", complex_mature),
                ("misfoldedIndexs", complex_mature),
                ("damagedIndexs", complex_mature),
            ):
                columns = (
                    np.asarray(
                        _get(self.complex, state_field), dtype=int
                    ).ravel(order="F")
                    - 1
                )
                self.complexDecayReactions[:, columns] = reactions[
                    np.ix_(substrate_rows, np.arange(len(complexes)))
                ]
            self.complexIndexs_modified = (
                np.flatnonzero(np.any(self.complexDecayReactions > 0, axis=0))
                + 1
            ).reshape(-1, 1)
            self.complexIndexs_unmodified = np.setdiff1d(
                np.arange(1, n_complexes + 1),
                self.complexIndexs_modified.ravel(),
            ).reshape(-1, 1)
            lengths = np.asarray(
                _get(self.monomer, "lengths"), dtype=float
            ).ravel(order="F")
            self.monomerLonProteaseCleavages = np.maximum(
                0,
                np.ceil(
                    lengths
                    / float(np.asarray(self.lonProteaseFragmentLength).item())
                )
                - 1,
            )
            return

        for k, v in kwargs.items():
            setattr(self, k, v)
        required = (
            "monomerDecayReactions",
            "complexDecayReactions",
            "monomerLonProteaseCleavages",
            "proteinComplexRNAComposition",
            "proteinComplexMonomerComposition",
            "substrateIndexs_atp",
            "substrateIndexs_adp",
            "substrateIndexs_phosphate",
            "substrateIndexs_water",
            "substrateIndexs_hydrogen",
            "substrateIndexs_aminoAcids",
        )
        missing = [x for x in required if not hasattr(self, x)]
        if missing:
            raise NotImplementedError(
                "ProteinDecay KB initialization requires exported reaction/index fields: "
                + ", ".join(missing)
            )

    def allocateMemoryForState(self, numTimePoints=1):
        nc = int(np.asarray(self.compartment.count).ravel()[0])
        ns = len(self.rnaWholeCellModelIDs)
        nm = len(_get(self.monomer, "wholeCellModelIDs"))
        nx = len(_get(self.complex, "wholeCellModelIDs"))
        shape = lambda n: (
            (n, nc) if numTimePoints == 1 else (n, nc, numTimePoints)
        )
        self.RNAs = np.zeros(shape(ns))
        self.monomers = np.zeros(shape(nm))
        self.complexs = np.zeros(shape(nx))

    def copyFromState(self):
        super().copyFromState()

        def matrix(value):
            value = np.asarray(value)
            if value.ndim == 3 and value.shape[2] == 1:
                value = value[:, :, 0]
            return value.copy()

        self.monomers = matrix(self.monomer.counts)
        self.complexs = matrix(self.complex.counts)
        ix = np.asarray(self.damagedRNAGlobalIndexs, dtype=int).ravel() - 1
        self.RNAs = matrix(self.rna.counts)[ix]

    def copyToState(self):
        super().copyToState()
        self.monomer.counts = np.asarray(self.monomers).reshape(
            np.asarray(self.monomer.counts).shape
        )
        self.complex.counts = np.asarray(self.complexs).reshape(
            np.asarray(self.complex.counts).shape
        )
        a = np.asarray(self.rna.counts)
        ix = np.asarray(self.damagedRNAGlobalIndexs, dtype=int).ravel() - 1
        a[ix] = self.RNAs.reshape(a[ix].shape)
        self.rna.counts = a

    def initializeState(self):
        return None

    def _idx(self, name):
        return np.asarray(getattr(self, name), dtype=int).ravel() - 1

    def _cytosols(self):
        return (
            np.asarray(
                [
                    self.compartment.cytosolIndexs,
                    self.compartment.terminalOrganelleCytosolIndexs,
                ],
                dtype=int,
            ).ravel()
            - 1
        )

    def calcResourceRequirements_Current(self):
        out = np.zeros_like(np.asarray(self.substrates))
        ci = self._cytosols()
        mn = np.asarray(self.monomers)[:, ci].sum(axis=1)
        cx = np.asarray(self.complexs)[:, ci].sum(axis=1)
        clp = (
            self._idx("enzymeIndexs_clpBProtease")[0]
            if hasattr(self, "enzymeIndexs_clpBProtease")
            else 0
        )
        if np.asarray(self.enzymes).ravel()[clp] > 0:
            n = (
                mn[
                    np.asarray(self.monomer.misfoldedIndexs, dtype=int).ravel()
                    - 1
                ].sum()
                + cx[
                    np.asarray(self.complex.misfoldedIndexs, dtype=int).ravel()
                    - 1
                ].sum()
            )
            if n > 0:
                out[self._idx("substrateIndexs_atp")[0]] += n
                out[self._idx("substrateIndexs_water")[0]] += n
        cr = np.asarray(self.complexDecayReactions)
        mr = np.asarray(self.monomerDecayReactions)
        cxdec = cx * np.minimum(
            1, np.asarray(_get(self.complex, "decayRates")).ravel()
        )
        mdec = mn * np.minimum(
            1, np.asarray(_get(self.monomer, "decayRates")).ravel()
        )
        out += (np.maximum(0, -cr) @ cxdec).reshape(out.shape)
        out += (np.maximum(0, -mr) @ mdec).reshape(out.shape)
        enz = np.asarray(self.enzymes).ravel()
        pep = (
            self._idx("enzymeIndexs_peptidases")
            if hasattr(self, "enzymeIndexs_peptidases")
            else np.arange(3, 9)
        )
        lon = int(np.asarray(self.enzymeIndexs_lonProtease).ravel()[0]) - 1
        energy = np.r_[
            self._idx("substrateIndexs_atp"),
            self._idx("substrateIndexs_water"),
        ]
        cleave = np.asarray(self.monomerLonProteaseCleavages).ravel() * mdec
        out[energy] += (
            np.all(enz[pep] > 0)
            * float(self.lonProteaseEnergyCost)
            * min(enz[lon] * float(self.lonProteaseSpecificRate), cleave.sum())
        )
        # MATLAB charges ATP and hydrolysis water for aborted tagged polypeptides, including nonzero sequences.
        pp = self.polypeptide
        aborted = (
            np.asarray(_get(pp, "abortedPolypeptides"))
            if pp is not None
            else np.empty((0, 0))
        )
        if aborted.size:
            lengths = np.asarray(_get(pp, "abortedSequenceLengths")).ravel()
            cleav = (
                np.ceil(lengths / float(self.ftsHProteaseFragmentLength)) - 1
            )
            fts = (
                int(np.asarray(self.enzymeIndexs_ftsHProtease).ravel()[0]) - 1
            )
            limit = np.ceil(
                float(self.ftsHProteaseEnergyCost)
                * min(
                    np.ceil(enz[fts] * float(self.ftsHProteaseSpecificRate)),
                    np.ceil(cleav).sum(),
                )
            )
            out[energy] += limit
            out[self._idx("substrateIndexs_water")[0]] += (lengths - 1).sum()
        return out

    def calcResourceRequirements_LifeCycle(self, *args):
        states = args[-1]
        n = len(np.asarray(self.substrates).ravel())
        e = len(self.enzymeWholeCellModelIDs)
        bm = np.zeros(n)
        by = np.zeros(n)
        lo = np.zeros(e)
        hi = np.full(e, np.inf)
        ref = (
            np.asarray(states.monomers).sum()
            + np.asarray(states.complexs).sum()
        ) * float(self.proteinMisfoldingRate)
        atp = self._idx("substrateIndexs_atp")[0]
        water = self._idx("substrateIndexs_water")[0]
        adp = self._idx("substrateIndexs_adp")[0]
        pi = self._idx("substrateIndexs_phosphate")[0]
        h = self._idx("substrateIndexs_hydrogen")[0]
        bm[[atp, water]] += ref
        by[[adp, pi, h]] += ref
        ci = np.asarray(self.complex.matureIndexs, dtype=int).ravel() - 1
        mi = np.asarray(self.monomer.matureIndexs, dtype=int).ravel() - 1
        cr = np.asarray(self.complexDecayReactions)[:, ci]
        mr = np.asarray(self.monomerDecayReactions)[:, mi]
        bm += np.maximum(0, -cr) @ np.asarray(states.complexDecays).ravel()
        by += np.maximum(0, cr) @ np.asarray(states.complexDecays).ravel()
        bm += np.maximum(0, -mr) @ np.asarray(states.monomerDecays).ravel()
        by += np.maximum(0, mr) @ np.asarray(states.monomerDecays).ravel()
        lc = (
            float(self.lonProteaseEnergyCost)
            * np.asarray(states.monomerDecays).ravel()
            @ np.asarray(self.monomerLonProteaseCleavages).ravel()[mi]
        )
        bm[[atp, water]] += lc
        by[[adp, pi, h]] += lc
        # Signal sequences are degraded during monomer production. Only species
        # with a nonzero signal-sequence decay reaction incur Lon cleavage cost.
        si = (
            np.asarray(self.monomer.signalSequenceIndexs, dtype=int).ravel()
            - 1
        )
        sr = np.asarray(self.monomerDecayReactions)[:, si]
        productions = np.asarray(states.monomerProductions).ravel()
        bm += np.maximum(0, -sr) @ productions
        by += np.maximum(0, sr) @ productions
        signal_cleavages = np.asarray(
            self.monomerLonProteaseCleavages
        ).ravel()[si] * np.any(sr != 0, axis=0)
        sl = float(self.lonProteaseEnergyCost) * productions @ signal_cleavages
        bm[[atp, water]] += sl
        by[[adp, pi, h]] += sl
        # MATLAB currently models lifecycle FtsH cleavage demand as zero.
        lo[self._idx("enzymeIndexs_lonProtease")] = (
            2
            * np.asarray(states.monomerDecays0).ravel()
            @ np.asarray(self.monomerLonProteaseCleavages).ravel()[mi]
            / float(self.lonProteaseSpecificRate)
        )
        lo[self._idx("enzymeIndexs_clpBProtease")] = 2
        lo[self._idx("enzymeIndexs_ftsHProtease")] = 70
        lo[self._idx("enzymeIndexs_peptidases")] = 2
        return bm, by, lo, hi

    def evolveState(self):
        # Preserve source phase order. The primitive source helpers below are
        # explicit methods so callers can invoke individual phases as MATLAB does.
        self.evolveState_MisfoldProteins()
        self.evolveState_RefoldProteins()
        self.evolveState_DegradeComplexes()
        self.evolveState_DegradeAbortedPolypeptides()
        self.evolveState_DegradeMonomers()

    def evolveState_MisfoldProteins(self):
        # Source-shaped misfold transition. External state hook handles locked rows.
        for objname, local, st, active in (
            (
                "monomer",
                "monomers",
                self.monomer,
                ("matureIndexs", "boundIndexs", "inactivatedIndexs"),
            ),
            (
                "complex",
                "complexs",
                self.complex,
                ("matureIndexs", "boundIndexs", "inactivatedIndexs"),
            ),
        ):
            arr = np.asarray(getattr(self, local))
            groups = [np.asarray(_get(st, k), int).ravel() - 1 for k in active]
            src = np.concatenate(groups)
            target = np.asarray(_get(st, "misfoldedIndexs"), int).ravel() - 1
            prob = min(
                1, float(self.proteinMisfoldingRate) * float(self.stepSizeSec)
            )
            tr = np.asarray(self.randStream.stochasticRound(arr[src] * prob))
            membranes = (
                np.r_[
                    np.asarray(self.compartment.membraneIndexs, int).ravel(),
                    np.asarray(
                        self.compartment.terminalOrganelleMembraneIndexs, int
                    ).ravel(),
                ]
                - 1
            )
            tr[:, membranes] = 0
            if not np.any(tr):
                continue
            tmp = np.zeros_like(arr)
            tmp[src] = -tr
            tmp[target] = tr.reshape((3, len(target), arr.shape[1])).sum(
                axis=0
            )
            reject = np.asarray(st.updateExternalState(tmp, True))
            accepted = tr - reject[src]
            arr[src] -= accepted
            arr[target] += accepted.reshape(
                (3, len(target), arr.shape[1])
            ).sum(axis=0)
            setattr(self, local, arr)

    def evolveState_RefoldProteins(self):
        ci = self._cytosols()
        mi = np.asarray(self.monomer.misfoldedIndexs, dtype=int).ravel() - 1
        xi = np.asarray(self.complex.misfoldedIndexs, dtype=int).ravel() - 1
        clp = int(np.asarray(self.enzymeIndexs_clpBProtease).ravel()[0]) - 1
        mc = np.asarray(self.monomers)[mi][:, ci]
        xc = np.asarray(self.complexs)[xi][:, ci]
        total = int(mc.sum() + xc.sum())
        n = (
            min(
                int(
                    np.asarray(self.substrates).ravel()[
                        self._idx("substrateIndexs_atp")[0]
                    ]
                ),
                int(
                    np.asarray(self.substrates).ravel()[
                        self._idx("substrateIndexs_water")[0]
                    ]
                ),
                total,
            )
            if np.asarray(self.enzymes).ravel()[clp] > 0
            else 0
        )
        if not n:
            return
        ids = np.asarray(self.randStream.randperm(total), int).ravel() - 1
        c = np.cumsum(np.r_[mc.ravel(order="F"), xc.ravel(order="F")])
        fm = np.zeros_like(mc)
        fx = np.zeros_like(xc)
        for q in ids[:n]:
            j = np.searchsorted(c, q, side="right")
            target = fm if j < mc.size else fx
            index = j if j < mc.size else j - mc.size
            target[np.unravel_index(index, target.shape, order="F")] += 1
        mm = np.asarray(self.monomer.matureIndexs, dtype=int).ravel() - 1
        xm = np.asarray(self.complex.matureIndexs, dtype=int).ravel() - 1
        self.monomers[np.ix_(mm, ci)] += fm
        self.complexs[np.ix_(xm, ci)] += fx
        np.asarray(self.monomers)[np.ix_(mi, ci)] -= fm
        np.asarray(self.complexs)[np.ix_(xi, ci)] -= fx
        self.substrates[self._idx("substrateIndexs_atp")[0]] -= n
        self.substrates[self._idx("substrateIndexs_water")[0]] -= n
        for product in (
            "substrateIndexs_adp",
            "substrateIndexs_phosphate",
            "substrateIndexs_hydrogen",
        ):
            self.substrates[self._idx(product)[0]] += n

    def evolveState_DegradeComplexes(self):
        # MATLAB operates on species x compartment matrices and uses column
        # major linear indexing in randsample. Flatten trailing singleton (or
        # other storage) dimensions into the compartment axis for that logic.
        original = np.asarray(self.complexs)
        n = len(_get(self.complex, "wholeCellModelIDs"))
        counts = original.reshape((n, -1))
        d = np.asarray(_get(self.complex, "decayRates")).ravel()
        iC = int(np.asarray(self.compartment.cytosolIndexs).ravel()[0]) - 1
        iTC = (
            int(
                np.asarray(
                    self.compartment.terminalOrganelleCytosolIndexs
                ).ravel()[0]
            )
            - 1
        )
        rates = counts * np.minimum(1e6, d[:, None]) * float(self.stepSizeSec)
        outside = [j for j in range(counts.shape[1]) if j not in (iC, iTC)]
        if outside:
            rates[np.ix_(np.flatnonzero(d < 1e6), outside)] = 0
        pending = np.minimum(
            counts, np.asarray(self.randStream.stochasticRound(rates))
        ).astype(int)
        rates[pending == 0] = 0
        if not np.any(pending):
            return

        dec = np.zeros_like(pending)
        simple = (
            np.asarray(self.complexIndexs_unmodified, dtype=int).ravel() - 1
        )
        modified = (
            np.asarray(self.complexIndexs_modified, dtype=int).ravel() - 1
        )
        dec[simple, :] = pending[simple, :]
        modified_pending = pending[modified, :].copy()
        modified_rates = rates[modified, :].copy()
        modified_decayed = np.zeros_like(modified_pending)
        substrates = np.asarray(self.substrates).reshape(-1).copy()
        reactions = np.asarray(self.complexDecayReactions)
        while np.any(modified_pending):
            # randsample returns a one-based MATLAB linear index. Convert it
            # back using MATLAB's column-major ordering of species x locations.
            idx = (
                int(
                    np.asarray(
                        self.randStream.randsample(
                            modified_rates.size,
                            1,
                            True,
                            modified_rates.ravel(order="F"),
                        )
                    ).ravel()[0]
                )
                - 1
            )
            mrow = idx % len(modified)
            col = idx // len(modified)
            row = modified[mrow]
            cost = np.maximum(0, -reactions[:, row])
            if np.any(substrates < cost):
                break
            modified_pending[mrow, col] -= 1
            modified_decayed[mrow, col] += 1
            if modified_pending[mrow, col] == 0:
                modified_rates[mrow, col] = 0
            else:
                modified_rates[mrow, col] = max(
                    0,
                    modified_rates[mrow, col]
                    - min(1e6, d[row]) * float(self.stepSizeSec),
                )
            substrates += reactions[:, row]
        dec[modified, :] = modified_decayed

        # The external-state hook may reject locked/bound complexes. Subtract
        # those rejections before updating counts or releasing components.
        shaped_dec = dec.reshape(original.shape)
        rejected = np.asarray(
            self.complex.updateExternalState(-shaped_dec, True)
        )
        dec = dec - rejected.reshape(dec.shape)
        self.complexs = original - dec.reshape(original.shape)
        decayed_by_species = dec.sum(axis=1)
        damaged = np.asarray(self.monomer.damagedIndexs, dtype=int).ravel() - 1
        self.monomers[damaged, iC] += (
            np.asarray(self.proteinComplexMonomerComposition)
            @ decayed_by_species
        )
        self.RNAs[:, iC] += (
            np.asarray(self.proteinComplexRNAComposition) @ decayed_by_species
        )
        self.substrates += (reactions @ decayed_by_species).reshape(
            self.substrates.shape
        )

    def evolveState_DegradeAbortedPolypeptides(self):
        pp = self.polypeptide
        aborted = (
            np.asarray(_get(pp, "abortedPolypeptides"))
            if pp is not None
            else np.empty((0, 0))
        )
        if not aborted.size:
            return
        seqs = list(
            np.asarray(_get(pp, "abortedSequences"), dtype=object).ravel(
                order="F"
            )
        )
        lengths = np.asarray(_get(pp, "abortedSequenceLengths")).ravel()
        if len(seqs) > 1:
            order = (
                np.asarray(self.randStream.randperm(len(seqs)), int).ravel()
                - 1
            )
            seqs = [seqs[i] for i in order]
            aborted = aborted[order]
            lengths = lengths[order]
        cleav = np.ceil(lengths / float(self.ftsHProteaseFragmentLength)) - 1
        pep = np.asarray(self.enzymes).ravel()[
            self._idx("enzymeIndexs_peptidases")
        ]
        fts = np.asarray(self.enzymes).ravel()[
            int(np.asarray(self.enzymeIndexs_ftsHProtease).ravel()[0]) - 1
        ]
        ftscost = np.asarray(
            self.randStream.stochasticRound(
                (fts * float(self.stepSizeSec))
                / np.cumsum(cleav / float(self.ftsHProteaseSpecificRate))
            )
            if np.any(cleav)
            else np.ones(len(cleav))
        ).ravel()
        if np.any(cleav):
            zero = np.flatnonzero(ftscost == 0)
            if zero.size:
                ftscost[zero[0] :] = 0
        energy = np.cumsum(
            np.asarray(
                self.randStream.stochasticRound(
                    cleav * float(self.ftsHProteaseEnergyCost)
                )
            ).ravel()
        )
        water = energy + np.cumsum(lengths - 1)
        sub = np.asarray(self.substrates).ravel()
        atp = self._idx("substrateIndexs_atp")[0]
        wi = self._idx("substrateIndexs_water")[0]
        feasible = np.r_[0, np.arange(1, len(lengths) + 1)] <= min(
            pep
        ) * float(self.stepSizeSec)
        feasible &= np.r_[1, ftscost] >= 1
        feasible &= np.r_[0, energy] <= sub[atp]
        feasible &= np.r_[0, water] <= sub[wi]
        nr = int(np.flatnonzero(feasible)[-1])
        if nr == 0:
            return
        # ProteinMonomer.computeBaseCount/computeDecayReaction equivalent: input
        # exports may provide reactions directly; otherwise translate sequence.
        aaids = (
            np.asarray(self.substrateIndexs_aminoAcids, dtype=int).ravel() - 1
        )
        counts = np.zeros(len(sub))

        def sequence_text(seq):
            # Preserve sequence contents if exported as a character array; str(array)
            # inserts spaces/brackets and corrupts both residue counts and length.
            if isinstance(seq, bytes):
                return seq.decode()
            if isinstance(seq, str):
                return seq
            a = np.asarray(seq)
            if a.ndim == 0:
                return str(a.item())
            parts = a.ravel(order="F").tolist()
            return "".join(
                x.decode() if isinstance(x, bytes) else str(x) for x in parts
            )

        for seq in seqs[:nr]:
            # ProteinMonomer.computeBaseCount(seq, 22, 1:21, true), followed by
            # computeDecayReaction(..., length(seq), 22).
            seq = sequence_text(seq)
            for j, aa in enumerate("ARNDCQEGHILKMFPSTWYV"):
                counts[aaids[j]] += seq.count(aa)
            counts[aaids[12]] -= 1
            counts[aaids[20]] += 1
            counts[self._idx("substrateIndexs_water")[0]] -= max(
                0, len(seq) - 1
            )
        # Peptide-bond hydrolysis consumes water only; hydrogen is added by the
        # separate ATP energy reaction below, as in the MATLAB source.
        self.substrates += counts.reshape(self.substrates.shape)
        self.substrates[self._idx("substrateIndexs_atp")[0]] -= energy[nr - 1]
        self.substrates[self._idx("substrateIndexs_adp")[0]] += energy[nr - 1]
        self.substrates[self._idx("substrateIndexs_phosphate")[0]] += energy[
            nr - 1
        ]
        self.substrates[self._idx("substrateIndexs_water")[0]] -= energy[
            nr - 1
        ]
        self.substrates[self._idx("substrateIndexs_hydrogen")[0]] += energy[
            nr - 1
        ]
        _set(pp, "abortedPolypeptides", aborted[nr:])

    def evolveState_DegradeMonomers(self):
        # MATLAB source: ProteinDecay.evolveState_DegradeMonomers.
        iC = int(np.asarray(self.compartment.cytosolIndexs).ravel()[0]) - 1
        iTC = (
            int(
                np.asarray(
                    self.compartment.terminalOrganelleCytosolIndexs
                ).ravel()[0]
            )
            - 1
        )
        counts = np.asarray(self.monomers)
        decay_rates = np.asarray(_get(self.monomer, "decayRates")).ravel()
        decaying_rates = (
            counts
            * np.minimum(1e6, decay_rates[:, None])
            * float(self.stepSizeSec)
        )
        outside = [j for j in range(counts.shape[1]) if j not in (iC, iTC)]
        decaying_rates[np.ix_(np.flatnonzero(decay_rates < 1e6), outside)] = 0
        decaying = np.minimum(
            counts, np.asarray(self.randStream.stochasticRound(decaying_rates))
        ).astype(int)
        decaying_rates[decaying == 0] = 0
        if not np.any(decaying):
            return
        decayed = np.zeros_like(decaying)
        energy = np.concatenate(
            [
                self._idx("substrateIndexs_atp"),
                self._idx("substrateIndexs_adp"),
                self._idx("substrateIndexs_phosphate"),
                self._idx("substrateIndexs_water"),
                self._idx("substrateIndexs_hydrogen"),
            ]
        )
        protease = (
            float(
                np.asarray(self.enzymes).ravel()[
                    int(np.asarray(self.enzymeIndexs_lonProtease).ravel()[0])
                    - 1
                ]
            )
            * float(self.stepSizeSec)
            * float(self.lonProteaseSpecificRate)
        )
        peptidase = float(
            np.asarray(
                self.randStream.stochasticRound(
                    min(
                        np.asarray(self.enzymes).ravel()[
                            self._idx("enzymeIndexs_peptidases")
                        ]
                    )
                    * float(self.stepSizeSec)
                )
            ).ravel()[0]
        )
        substrates = np.asarray(self.substrates).reshape(-1).copy()
        weights = decaying_rates.ravel(order="F").copy()
        cleavages = np.asarray(self.monomerLonProteaseCleavages).ravel()
        energy_vector = np.array([1.0, -1.0, -1.0, 1.0, -1.0])
        while np.any(decaying):
            # MATLAB randsample(decayingRates(:)) uses column-major linear indexing.
            idx = (
                int(
                    np.asarray(
                        self.randStream.randsample(
                            weights.size, 1, True, weights
                        )
                    ).ravel()[0]
                )
                - 1
            )
            row, col = idx % counts.shape[0], idx // counts.shape[0]
            protease_cost = cleavages[row]
            substrate_cost = -np.asarray(self.monomerDecayReactions)[
                :, row
            ].copy()
            substrate_cost[energy] += (
                energy_vector
                * float(self.lonProteaseEnergyCost)
                * protease_cost
            )
            if (
                np.any(
                    substrates[energy] < np.maximum(0, substrate_cost[energy])
                )
                or float(
                    np.asarray(
                        self.randStream.stochasticRound(
                            protease / protease_cost
                        )
                    ).ravel()[0]
                )
                < 1
                or peptidase < 1
            ):
                break
            decaying[row, col] -= 1
            decayed[row, col] += 1
            if decaying[row, col] <= 0:
                weights[idx] = 0
            else:
                weights[idx] = max(
                    0,
                    weights[idx]
                    - min(1e6, decay_rates[row]) * float(self.stepSizeSec),
                )
            substrates -= substrate_cost
            protease = max(0, protease - protease_cost)
            peptidase -= 1
        decayed -= np.asarray(self.monomer.updateExternalState(-decayed, True))
        self.monomers = counts - decayed
        self.substrates += (
            np.asarray(self.monomerDecayReactions) @ decayed.sum(axis=1)
        ).reshape(self.substrates.shape)
        self.substrates[energy] -= (
            energy_vector
            * float(self.lonProteaseEnergyCost)
            * (cleavages @ decayed.sum(axis=1))
        ).reshape(self.substrates[energy].shape)

    def getDryWeight(self):
        base = super().getDryWeight()
        ci = self._cytosols()
        rna = np.asarray(self.rna.molecularWeights)[
            self._idx("damagedRNAGlobalIndexs")
        ] @ np.asarray(self.RNAs).sum(axis=1)
        mon = np.asarray(self.monomer.molecularWeights) @ np.asarray(
            self.monomers
        ).sum(axis=1)
        cx = np.asarray(self.complex.molecularWeights) @ np.asarray(
            self.complexs
        ).sum(axis=1)
        return base + (rna + mon + cx) / 6.0221417930e23
