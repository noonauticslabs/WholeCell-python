"""Runtime port of WholeCell DNARepair.m.

MATLAB-origin indices on this object remain one-based.  Array access converts
at the boundary, keeping snapshot/runtime field names faithful to the source.
"""

from __future__ import annotations
import numpy as np
from ..process_base import Process, _get, _set
from ..chromosome_process_aspect import ChromosomeProcessAspect
from ..reaction_process import ReactionProcess


class DNARepair(ChromosomeProcessAspect, ReactionProcess):
    optionNames__ = ()
    fixedConstantNames__ = (
        "HR_PolA_ResectionLength",
        "HR_RecA_Spacing",
        "HR_RuvAB_JunctionMigrationHop",
        "HR_RecU_CleavageSequence",
        "HR_RecU_CleavagePosition",
        "NER_UvrABC_IncisionMargin3",
        "NER_UvrABC_IncisionMargin5",
        "NER_PcrA_StepSize",
        "hollidayJunctionResolutionSites",
        "RM_EcoD_RecognitionSequence",
        "RM_EcoD_RecognitionSites",
        "RM_EcoD_MethylationPosition",
        "RM_EcoD_RestrictionPosition",
        "RM_MunI_RecognitionSequence",
        "RM_MunI_RecognitionSites",
        "RM_MunI_MethylationPosition",
        "RM_MunI_RestrictionPosition",
        "RM_MunI_MethylatedPositions",
        "RM_MunI_RestrictionPositions",
        "reactionSmallMoleculeStoichiometryMatrix",
        "reactionDNAStoichiometryMatrix",
        "reactionDNABase",
    )
    fittedConstantNames__ = localStateNames__ = ()
    substrateWholeCellModelIDs = ()
    enzymeWholeCellModelIDs = ()
    _runtime_methods = (
        "evolveState_BER",
        "evolveState_NER",
        "evolveState_HR",
        "evolveState_Polymerize",
        "evolveState_Ligate",
        "evolveState_Modification",
        "evolveState_Restriction",
        "evolveState_DisA",
    )

    def __init__(
        self,
        wholeCellModelID="Process_DNARepair",
        name="DNA Repair",
        **properties,
    ):
        super().__init__(wholeCellModelID, name, **properties)
        self.chromosome = getattr(self, "chromosome", None)
        self.dnaDamage = getattr(self, "dnaDamage", None)

    def storeObjectReferences(self, simulation):
        Process.storeObjectReferences(self, simulation)
        ChromosomeProcessAspect.storeObjectReferences(self, simulation)
        state = getattr(simulation, "state", None)
        if callable(state):
            self.chromosome = state("Chromosome")
        else:
            st = _get(simulation, "states")
            self.chromosome = {str(k).lower(): v for k, v in st.items()}[
                "chromosome"
            ]
        pfn = getattr(simulation, "process", None)
        if callable(pfn):
            self.dnaDamage = pfn("DNADamage")
        self.states = [*self.states, self.chromosome]

    def initializeConstants(self, *args, **kwargs):
        """Initialize repair chemistry from KB or validate snapshot constants."""
        if len(args) >= 2 and args[0] is not None and args[1] is not None:
            knowledge_base, simulation = args[:2]
            options = kwargs.pop("options", args[2] if len(args) > 2 else None)
            ReactionProcess.initializeConstants(
                self, knowledge_base, simulation, options
            )
            ChromosomeProcessAspect.initializeConstants(
                self, knowledge_base, simulation
            )
            self._initialize_kb_constants(knowledge_base, simulation)
            return
        for name, value in kwargs.items():
            setattr(self, name, value)
        required = (
            "reactionStoichiometryMatrix",
            "reactionTypes",
            "reactionWholeCellModelIDs",
            "reactionCatalysisMatrix",
            "enzymeBounds",
            "reactionBounds",
        )
        missing = [n for n in required if not hasattr(self, n)]
        if missing:
            raise AttributeError(
                "DNARepair snapshot lacks source fields: " + ", ".join(missing)
            )
        # initializeConstants in MATLAB builds these partition matrices; snapshots that
        # already carry them preserve the exact original numeric values.
        if not hasattr(self, "reactionSmallMoleculeStoichiometryMatrix"):
            self.reactionSmallMoleculeStoichiometryMatrix = np.asarray(
                self.reactionStoichiometryMatrix
            ).copy()
        if not hasattr(self, "reactionDNAStoichiometryMatrix"):
            self.reactionDNAStoichiometryMatrix = np.zeros_like(
                self.reactionSmallMoleculeStoichiometryMatrix
            )
        if not hasattr(self, "reactionDNABase"):
            self.reactionDNABase = np.zeros(
                (np.asarray(self.reactionStoichiometryMatrix).shape[1],),
                dtype=int,
            )

    @staticmethod
    def _text(value):
        while (
            isinstance(value, np.ndarray)
            and value.dtype == object
            and value.size == 1
        ):
            value = value.reshape(-1)[0]
        return str(value)

    def _initialize_kb_constants(self, knowledge_base, simulation):
        metabolites = np.asarray(
            _get(knowledge_base, "metabolites"), dtype=object
        ).ravel(order="F")
        metabolite_state = simulation.state("Metabolite")
        all_metabolite_ids = np.asarray(
            _get(metabolite_state, "wholeCellModelIDs"), dtype=object
        ).ravel(order="F")
        categories = {
            str(_get(m, "wholeCellModelID")): _get(m, "category")
            for m in metabolites
        }
        extras = {"AD", "CSN", "GN", "THY", "URA", "URI"}
        extras.update(
            identifier
            for identifier, category in categories.items()
            if category
            in {
                "modified nucleobase",
                "modified deoxyribonucleoside",
                "modified deoxyribonucleotide monophosphate",
            }
        )
        wanted = extras | set(
            np.asarray(self.substrateWholeCellModelIDs, dtype=object).ravel()
        )
        missing = sorted(wanted.difference(all_metabolite_ids.tolist()))
        if missing:
            raise ValueError(
                f"DNARepair substrates missing from Metabolite state: {missing}"
            )
        self.initializeConstants_overrideSubstrates(
            np.asarray(
                [x for x in all_metabolite_ids if x in wanted], dtype=object
            )
        )
        enzyme_ids = set(
            np.asarray(self.enzymeWholeCellModelIDs, dtype=object).ravel()
        )
        enzyme_ids.add("MG_105_OCTAMER")
        available_enzymes = np.asarray(
            self.enzymeWholeCellModelIDs, dtype=object
        ).ravel()
        # The DisA complex is added even if no reaction catalysis row names it.
        if "MG_105_OCTAMER" not in available_enzymes:
            dis_a = np.asarray(["MG_105_OCTAMER"], dtype=object)
            self.initializeConstants_overrideEnzymes(
                np.r_[available_enzymes, dis_a]
            )
        for field in (
            "HR_RecU_CleavageSequence",
            "RM_EcoD_RecognitionSequence",
            "RM_MunI_RecognitionSequence",
        ):
            if hasattr(self, field):
                setattr(self, field, self._text(getattr(self, field)))

        for field, ids in {
            "substrateIndexs_water": ["H2O"],
            "substrateIndexs_hydrogen": ["H"],
            "substrateIndexs_DR5P": ["DR5P"],
            "substrateIndexs_dRibose5P_dRibose5P": ["dRibose5P_dRibose5P"],
            "substrateIndexs_NAD": ["NAD"],
            "substrateIndexs_AMP": ["AMP"],
            "substrateIndexs_NMN": ["NMN"],
            "substrateIndexs_PPI": ["PPI"],
            "substrateIndexs_m6AD": ["m6AD"],
            "substrateIndexs_undamagedNucleobases": ["AD", "CSN", "GN", "THY"],
            "substrateIndexs_dNMPs": ["DAMP", "DCMP", "DGMP", "DTMP"],
            "substrateIndexs_dNTPs": ["DATP", "DCTP", "DGTP", "DTTP"],
            "substrateIndexs_dNpdNps": [
                "dApdAp",
                "dCpdCp",
                "dGpdGp",
                "dTpdTp",
            ],
        }.items():
            setattr(self, field, self.substrateIndexs(ids))
        local_by_id = {
            str(identifier): i + 1
            for i, identifier in enumerate(
                np.asarray(
                    self.substrateWholeCellModelIDs, dtype=object
                ).ravel()
            )
        }
        for field, categories_for_field, special in (
            (
                "substrateIndexs_damagedNucleobases",
                {"modified nucleobase"},
                {"URA"},
            ),
            (
                "substrateIndexs_damagedNucleosides",
                {"modified deoxyribonucleoside"},
                {"URI"},
            ),
            (
                "substrateIndexs_damagedDNMPs",
                {"modified deoxyribonucleotide monophosphate"},
                {"DUMP"},
            ),
        ):
            identifiers = [
                identifier
                for identifier, category in categories.items()
                if category in categories_for_field
            ]
            identifiers.extend(special)
            setattr(
                self,
                field,
                np.asarray(
                    [local_by_id[x] for x in identifiers if x in local_by_id],
                    dtype=int,
                ).reshape(-1, 1),
            )

        for field, ids in {
            "enzymeIndexs_BER_baseExcision": [
                "MG_097_MONOMER",
                "MG_498_MONOMER",
            ],
            "enzymeIndexs_formamidopyrimidineGlycosylase": ["MG_498_MONOMER"],
            "enzymeIndexs_uracilGlycosylase": ["MG_097_MONOMER"],
            "enzymeIndexs_apurinicEndonuclease": ["MG_235_MONOMER"],
            "enzymeIndexs_incisionComplex": ["MG_073_206_421_TETRAMER"],
            "enzymeIndexs_helicase35": ["MG_244_DIMER"],
            "enzymeIndexs_recombinationStrandExchange": ["MG_339_MONOMER"],
            "enzymeIndexs_hollidayJunctionEndonuclease": ["MG_352_DIMER"],
            "enzymeIndexs_hollidayJunctionHelicase": ["MG_358_359_10MER"],
            "enzymeIndexs_exonuclease53": ["MG_262_MONOMER"],
            "enzymeIndexs_polymerase": ["MG_001_MONOMER"],
            "enzymeIndexs_ligase": ["MG_254_MONOMER"],
            "enzymeIndexs_phosphoesterase": ["MG_190_MONOMER"],
            "enzymeIndexs_DisA": ["MG_105_OCTAMER"],
            "enzymeIndexs_RM_typeI": ["MG_438_MONOMER"],
            "enzymeIndexs_RM_typeII": ["MG_184_DIMER"],
        }.items():
            setattr(self, field, self.enzymeIndexs(ids))
        self._initialize_repair_reactions()

    def _initialize_repair_reactions(self):
        """Partition repair reaction stoichiometry as in DNARepair.m:560-671."""
        types = np.asarray(self.reactionTypes, dtype=object).ravel()
        rxn_ids = np.asarray(
            self.reactionWholeCellModelIDs, dtype=object
        ).ravel()

        def by_type(*names):
            return np.flatnonzero(np.isin(types, names)) + 1

        self.reactionSmallMoleculeStoichiometryMatrix = np.zeros_like(
            self.reactionStoichiometryMatrix
        )
        self.reactionDNAStoichiometryMatrix = np.zeros_like(
            self.reactionStoichiometryMatrix
        )
        n_reactions = self.reactionStoichiometryMatrix.shape[1]
        self.reactionDNABase = np.zeros(n_reactions, dtype=int)
        self.reactionIndexs_BER = by_type(
            "base excision repair", "base excision repair, base excision"
        )
        self.reactionIndexs_BER_baseexcision = by_type(
            "base excision repair, base excision"
        )
        for field, ids in {
            "reactionIndexs_BER_aplyase": ["AP_lyase"],
            "reactionIndexs_BER_drpesterase": ["deoxyribosephosphodiesterase"],
            "reactionIndexs_BER_apendonuclease": ["AP_endonuclease"],
            "reactionIndexs_BER_dr5plyase": ["dRiboseP_lyase"],
            "reactionIndexs_NER_incision": ["NER_DNAIncision"],
            "reactionIndexs_NER_excision": ["NER_DNAExcision"],
            "reactionIndexs_HR_DNAResection": [
                "HR_DNAResection_dAMP",
                "HR_DNAResection_dCMP",
                "HR_DNAResection_dGMP",
                "HR_DNAResection_dTMP",
            ],
            "reactionIndexs_HR_strandExchange": ["HR_strandExchange"],
            "reactionIndexs_HR_junctionMigration": ["HR_junctionMigration"],
            "reactionIndexs_HR_junctionResolution": ["HR_junctionResolution"],
            "reactionIndexs_ligation": ["DNA_ligation_repair"],
            "reactionIndexs_RM_EcoD_Methylation": ["DNA_RM_EcoD_Methylation"],
            "reactionIndexs_RM_EcoD_Restriction": ["DNA_RM_EcoD_Restriction"],
            "reactionIndexs_RM_MunI_Methylation": ["DNA_RM_MunI_Methylation"],
            "reactionIndexs_RM_MunI_Restriction": ["DNA_RM_MunI_Restriction"],
        }.items():
            setattr(self, field, self.reactionIndexs(ids))
        self.reactionIndexs_NER = by_type("nucleotide excision repair")
        self.reactionIndexs_DNACleavage = by_type("DNA cleavage")
        self.reactionIndexs_HR_dsbr = by_type("homologous recombination")
        self.reactionIndexs_polymerization = by_type("DNA polymerization")
        self.reactionIndexs_RM = by_type("DNA restriction/modification")

        dna = self.reactionDNAStoichiometryMatrix
        stoich = np.asarray(self.reactionStoichiometryMatrix)

        def assign(rows, reactions, values=None):
            rows = np.asarray(rows, dtype=int).ravel() - 1
            reactions = np.asarray(reactions, dtype=int).ravel() - 1
            if rows.size and reactions.size:
                dna[np.ix_(rows, reactions)] = (
                    stoich[np.ix_(rows, reactions)]
                    if values is None
                    else values
                )

        ber_core = np.r_[
            self.substrateIndexs_damagedDNMPs.ravel(),
            self.substrateIndexs_dRibose5P_dRibose5P.ravel(),
            self.substrateIndexs_DR5P.ravel(),
        ]
        assign(
            ber_core,
            np.r_[
                self.reactionIndexs_BER_baseexcision,
                self.reactionIndexs_BER_aplyase,
                self.reactionIndexs_BER_apendonuclease,
            ],
        )
        ber_sugar = np.r_[
            self.substrateIndexs_damagedDNMPs.ravel(),
            self.substrateIndexs_dRibose5P_dRibose5P.ravel(),
        ]
        assign(
            ber_sugar,
            np.r_[
                self.reactionIndexs_BER_drpesterase,
                self.reactionIndexs_BER_dr5plyase,
            ],
        )
        dr5p = np.asarray(self.substrateIndexs_DR5P, dtype=int).ravel() - 1
        dr5_reactions = (
            np.r_[
                self.reactionIndexs_BER_drpesterase,
                self.reactionIndexs_BER_dr5plyase,
            ].astype(int)
            - 1
        )
        if dr5p.size and dr5_reactions.size:
            dna[np.ix_(dr5p, dr5_reactions)] = 1
        ner_dna = np.r_[ber_core]
        assign(ner_dna, self.reactionIndexs_NER)
        cleavage = (
            np.asarray(self.reactionIndexs_DNACleavage, dtype=int).ravel() - 1
        )
        assign(self.substrateIndexs_dNpdNps, self.reactionIndexs_DNACleavage)
        dnmps = np.asarray(self.substrateIndexs_dNMPs, dtype=int).ravel() - 1
        if dnmps.size and cleavage.size:
            dna[np.ix_(dnmps, cleavage)] = (
                0.5 * stoich[np.ix_(dnmps, cleavage)]
            )
        hr_dna = np.r_[
            self.substrateIndexs_dNMPs.ravel(),
            self.substrateIndexs_dNpdNps.ravel(),
            ber_sugar,
        ]
        hr_rows = np.asarray(hr_dna, dtype=int).ravel() - 1
        hr_cols = (
            np.asarray(self.reactionIndexs_HR_dsbr, dtype=int).ravel() - 1
        )
        if hr_rows.size and hr_cols.size:
            factor = np.r_[
                np.full(len(self.substrateIndexs_dNMPs), 0.5),
                np.ones(len(hr_rows) - len(self.substrateIndexs_dNMPs)),
            ]
            dna[np.ix_(hr_rows, hr_cols)] = (
                factor[:, None] * stoich[np.ix_(hr_rows, hr_cols)]
            )
        polymer_dna = np.r_[
            self.substrateIndexs_dNMPs.ravel(),
            self.substrateIndexs_dNpdNps.ravel(),
        ]
        assign(polymer_dna, self.reactionIndexs_polymerization)
        assign(
            np.r_[
                self.substrateIndexs_DR5P.ravel(),
                self.substrateIndexs_dRibose5P_dRibose5P.ravel(),
            ],
            self.reactionIndexs_ligation,
        )
        rm_dna = np.r_[
            self.substrateIndexs_undamagedNucleobases.ravel(),
            self.substrateIndexs_damagedNucleobases.ravel(),
            ber_sugar,
        ]
        assign(rm_dna, self.reactionIndexs_RM)

        sequence = self.chromosome.sequence
        self.hollidayJunctionResolutionSites = sequence.findSubsequence(
            self.HR_RecU_CleavageSequence, self.HR_RecU_CleavagePosition, True
        )
        self.RM_MunI_MethylatedPositions = np.array(
            [
                self.RM_MunI_MethylationPosition,
                len(self.RM_MunI_RecognitionSequence)
                - self.RM_MunI_MethylationPosition
                + 1,
            ],
            dtype=int,
        ).reshape(1, 2)
        self.RM_MunI_RestrictionPositions = np.array(
            [
                self.RM_MunI_RestrictionPosition,
                len(self.RM_MunI_RecognitionSequence)
                - self.RM_MunI_RestrictionPosition,
            ],
            dtype=int,
        ).reshape(1, 2)
        self.RM_EcoD_RecognitionSites = self._recognition_sites(
            sequence, self.RM_EcoD_RecognitionSequence
        )
        self.RM_MunI_RecognitionSites = self._recognition_sites(
            sequence, self.RM_MunI_RecognitionSequence
        )
        self.reactionSmallMoleculeStoichiometryMatrix = stoich - dna
        damaged = (
            np.asarray(self.substrateIndexs_damagedDNMPs, dtype=int).ravel()
            - 1
        )
        for reaction in range(n_reactions):
            consumed = (
                np.flatnonzero(dna[damaged, reaction] < 0)
                if damaged.size
                else np.empty(0, int)
            )
            if consumed.size:
                self.reactionDNABase[reaction] = damaged[consumed[0]] + 1

    def _recognition_sites(self, sequence, motif):
        starts = np.asarray(sequence.findSubsequence(motif, 1), dtype=int)
        length = len(motif)
        if starts.size == 0:
            return np.empty((0, length), dtype=int)
        starts = starts.reshape(-1, starts.shape[-1] if starts.ndim > 1 else 1)
        if starts.shape[1] > 1:
            starts = starts[:, :1]
        offsets = np.arange(length, dtype=int)
        return (starts[:, :1] + offsets[None, :] - 1) % len(sequence) + 1

    @staticmethod
    def _collapse(value):
        if hasattr(value, "collapse") and callable(value.collapse):
            return float(value.collapse(None))
        return float(np.asarray(value).sum())

    def _substrate_update(self, delta):
        """Apply a substrate flux in MATLAB's column-vector arithmetic."""
        current = np.asarray(self.substrates)
        self.substrates = (
            current.reshape(-1) + np.asarray(delta).reshape(-1)
        ).reshape(current.shape)

    def _substrate_limits(self, costs):
        costs = np.asarray(costs).reshape(-1)
        amounts = np.asarray(self.substrates).reshape(-1)
        return np.divide(
            amounts,
            costs,
            out=np.full(costs.shape, np.inf, dtype=float),
            where=costs > 0,
        )

    @staticmethod
    def _cell_string(value):
        """Decode a MATLAB cell string while leaving numeric motif cells intact."""
        while (
            isinstance(value, np.ndarray)
            and value.dtype == object
            and value.size == 1
        ):
            value = value.reshape(-1)[0]
        if isinstance(value, np.ndarray) and value.dtype.kind in "US":
            return "".join(value.reshape(-1, order="F").astype(str))
        return value

    @staticmethod
    def _shared_motif_site_count(motif_value, left, right):
        """Count coordinates whose values match a motif in both sparse maps."""

        def matching_coordinates(matrix):
            if hasattr(matrix, "find"):
                positions, values = matrix.find()
                positions = np.asarray(positions, dtype=int).reshape((-1, 2))
                values = np.asarray(values).reshape(-1)
            else:
                array = np.asarray(matrix)
                positions = np.argwhere(array != 0) + 1
                values = (
                    array[tuple((positions - 1).T)] if len(positions) else []
                )
            return {
                tuple(position)
                for position, value in zip(positions, values)
                if value == motif_value
            }

        return len(matching_coordinates(left) & matching_coordinates(right))

    @staticmethod
    def _matlab_rows(rows, indices):
        """Select rows using MATLAB's one-based candidate indices."""
        return np.asarray(rows)[np.asarray(indices, dtype=int).reshape(-1) - 1]

    @staticmethod
    def _matlab_find_coordinates(array):
        """Return nonzero 1-based coordinates in MATLAB column-major order."""
        array = np.asarray(array)
        linear = np.flatnonzero(array.reshape(-1, order="F"))
        row, col = np.unravel_index(linear, array.shape, order="F")
        return np.column_stack((row + 1, col + 1))

    @staticmethod
    def _chromosome_copy_coordinates(coordinates):
        """Map odd original strand columns to MATLAB's compressed copy index."""
        coordinates = (
            np.asarray(coordinates, dtype=int).reshape((-1, 2)).copy()
        )
        if len(coordinates):
            coordinates[:, 1] = (coordinates[:, 1] + 1) // 2
        return coordinates

    @staticmethod
    def _sort_matlab_subscripts(coordinates, first_dimension_size):
        """Sort [position, strand] coordinates by MATLAB's linear index."""
        coordinates = np.asarray(coordinates, dtype=int).reshape((-1, 2))
        if not len(coordinates):
            return coordinates
        key = (
            (coordinates[:, 1] - 1) * int(first_dimension_size)
            + coordinates[:, 0]
            - 1
        )
        return coordinates[np.argsort(key, kind="stable")]

    def calcResourceRequirements_LifeCycle(self, constants, states=None):
        rates = self.calcExpectedReactionRates(constants)
        sm = np.asarray(self.reactionSmallMoleculeStoichiometryMatrix)
        cat = np.asarray(self.reactionCatalysisMatrix)
        bounds = np.asarray(self.enzymeBounds)
        bm = np.maximum(0, -sm) @ rates
        by = np.maximum(0, sm) @ rates
        cell = constants.states.Time
        mn = np.maximum(
            cat.T @ (rates / bounds[:, 1]) / cell.cellCycleLength * np.log(2),
            cat[
                np.asarray(
                    self.reactionIndexs_RM_MunI_Methylation, dtype=int
                ).ravel()
                - 1,
                :,
            ].T
            @ (
                rates[
                    np.asarray(
                        self.reactionIndexs_RM_MunI_Methylation, dtype=int
                    ).ravel()
                    - 1
                ]
                / bounds[
                    np.asarray(
                        self.reactionIndexs_RM_MunI_Methylation, dtype=int
                    ).ravel()
                    - 1,
                    1,
                ]
            )
            / cell.cellCycleLength
            * np.log(2)
            / (
                np.exp(
                    np.log(2)
                    * (
                        cell.replicationInitiationDuration
                        + cell.replicationDuration
                    )
                    / cell.cellCycleLength
                )
                - np.exp(
                    np.log(2)
                    * cell.replicationInitiationDuration
                    / cell.cellCycleLength
                )
            ),
        )
        return (
            bm,
            by,
            mn,
            np.full(np.asarray(self.enzymeWholeCellModelIDs).size, np.inf),
        )

    def calcExpectedReactionRates(self, constants):
        """Expected repair fluxes over a cell cycle from DNADamage vulnerable motifs."""
        dmg = self.dnaDamage
        c = self.chromosome
        rates = np.zeros(np.asarray(self.reactionWholeCellModelIDs).size)
        comp = (
            np.array(
                [
                    1 - c.sequenceGCContent,
                    c.sequenceGCContent,
                    c.sequenceGCContent,
                    1 - c.sequenceGCContent,
                ]
            )
            / 2
        )
        bases = "ACGT"
        pol = self._collapse(c.polymerizedRegions)
        t = constants.states.Time
        motifs = np.asarray(
            dmg.reactionVulnerableMotifs, dtype=object
        ).reshape(-1)
        motif_types = np.asarray(
            dmg.reactionVulnerableMotifTypes, dtype=object
        ).reshape(-1)
        damage_types = np.asarray(
            dmg.reactionDamageTypes, dtype=object
        ).reshape(-1)
        for i in range(np.asarray(dmg.reactionStoichiometryMatrix).shape[1]):
            if np.asarray(dmg.reactionBounds)[i, 1] == 0:
                continue
            motif = self._cell_string(motifs[i])
            if isinstance(motif, str):
                nv = pol * np.prod(comp[[bases.index(x) for x in motif]])
            else:
                typ = self._cell_string(motif_types[i])
                motif_value = np.asarray(motif).reshape(-1)[0]
                nv = self._shared_motif_site_count(
                    motif_value, _get(c, typ), c.damagedSites_nonRedundant
                )
            if nv == 0:
                continue
            dr = (
                nv
                * (
                    t.replicationInitiationDuration
                    + 1.5 * t.replicationDuration
                    + 2 * t.cytokinesisDuration
                )
                * dmg.reactionBounds[i, 1]
            )
            rad = np.flatnonzero(
                np.asarray(dmg.reactionRadiationStoichiometryMatrix)[:, i]
            )
            if rad.size:
                glob = np.asarray(
                    dmg.substrateStimulusGlobalCompartmentIndexs
                ).ravel()[
                    np.asarray(dmg.substrateStimulusLocalIndexs).ravel()
                    == rad[0] + 1
                ]
                dr *= np.asarray(self.stimulus.values).ravel(order="F")[
                    glob.astype(int) - 1
                ][0]
            kind = self._cell_string(damage_types[i])
            mapping = {
                "abasicSites": (
                    ("reactionIndexs_BER_aplyase", 0.5),
                    ("reactionIndexs_BER_drpesterase", 0.5),
                    ("reactionIndexs_BER_apendonuclease", 0.5),
                    ("reactionIndexs_BER_dr5plyase", 0.5),
                    ("reactionIndexs_ligation", 1),
                ),
                "damagedBases": (
                    ("reactionIndexs_BER_baseexcision", 0.5),
                    ("reactionIndexs_BER_aplyase", 0.5),
                    ("reactionIndexs_BER_drpesterase", 0.5),
                    ("reactionIndexs_BER_apendonuclease", 0.5),
                    ("reactionIndexs_BER_dr5plyase", 0.5),
                    ("reactionIndexs_ligation", 1),
                ),
                "strandBreaks": (("reactionIndexs_ligation", 1),),
            }
            if kind in mapping:
                for field, f in mapping[kind]:
                    rates[
                        np.asarray(getattr(self, field), int).ravel() - 1
                    ] += (dr * f)
                if kind in ("abasicSites", "damagedBases"):
                    rates[
                        np.asarray(
                            self.reactionIndexs_polymerization, int
                        ).ravel()
                        - 1
                    ] += (dr * comp)
            elif kind == "intrastrandCrossLinks":
                n = (
                    self.NER_UvrABC_IncisionMargin5
                    + self.NER_UvrABC_IncisionMargin3
                    + 1
                )
                rates[
                    np.asarray(self.reactionIndexs_NER_incision, int).ravel()
                    - 1
                ] += dr
                rates[
                    np.asarray(self.reactionIndexs_NER_excision, int).ravel()
                    - 1
                ] += dr * np.ceil(n / self.NER_PcrA_StepSize)
                rates[
                    np.asarray(self.reactionIndexs_DNACleavage, int).ravel()
                    - 1
                ] += (
                    dr
                    * n
                    * comp[
                        : len(
                            np.asarray(self.reactionIndexs_DNACleavage).ravel()
                        )
                    ]
                )
                rates[
                    np.asarray(self.reactionIndexs_ligation, int).ravel() - 1
                ] += dr
                rates[
                    np.asarray(self.reactionIndexs_polymerization, int).ravel()
                    - 1
                ] += (2 * dr * comp)
            else:
                raise ValueError(f"undefined damage reaction type {kind}")
        idx = (
            np.asarray(
                self.reactionIndexs_RM_MunI_Methylation, dtype=int
            ).ravel()
            - 1
        )
        rates[idx] = 2 * np.asarray(self.RM_MunI_RecognitionSites).shape[0]
        return rates

    def initializeState(self):
        c = self.chromosome
        sites = np.asarray(self.RM_MunI_RecognitionSites, dtype=int)
        if sites.size:
            met = np.asarray(self.substrateGlobalIndexs, dtype=int).ravel()[
                int(np.asarray(self.substrateIndexs_m6AD).item()) - 1
            ]
            pos = (
                np.asarray(self.RM_MunI_MethylatedPositions, dtype=int).ravel()
                - 1
            )
            # CircularSparseMat deliberately retains MATLAB one-based indexing.
            c.damagedBases[sites[:, pos[0]], 1] = met
            c.damagedBases[sites[:, pos[1]], 2] = met

    def calcResourceRequirements_Current(self):
        c = self.chromosome
        nnz = lambda matrix: len(matrix.find()[1])
        ng, na = nnz(c.gapSites), nnz(c.abasicSites)
        nx, ns, nh = (
            nnz(c.intrastrandCrossLinks),
            nnz(c.strandBreaks),
            nnz(c.hollidayJunctions) / 2,
        )
        nd = 0
        if nnz(c.damagedSites):
            pos, vals = c.damagedBases.find()
            vals = np.asarray(vals).ravel()
            met = int(np.asarray(self.metabolite.m6ADIndexs).item())
            sites = np.asarray(self.RM_MunI_RecognitionSites, int)
            methyl = (
                np.asarray(self.RM_MunI_MethylatedPositions, int).ravel() - 1
            )
            protected = (vals == met) & (
                (
                    (pos[:, 1] % 2 == 1)
                    & np.isin(pos[:, 0], sites[:, methyl[0]])
                )
                | (
                    (pos[:, 1] % 2 == 0)
                    & np.isin(pos[:, 0], sites[:, methyl[1]])
                )
            )
            nd = len(vals) - np.count_nonzero(protected)
        dsbr = nnz(c.doubleStrandBreaks) if ns else 0
        rates = np.zeros(
            np.asarray(self.reactionSmallMoleculeStoichiometryMatrix).shape[1]
        )
        setr = lambda field, v: rates.__setitem__(
            np.asarray(getattr(self, field), int).ravel() - 1, v
        )
        setr("reactionIndexs_BER_baseexcision", nd)
        setr("reactionIndexs_BER_apendonuclease", 0.5 * na)
        setr("reactionIndexs_BER_aplyase", 0.5 * na)
        setr("reactionIndexs_BER_drpesterase", 0.5 * na)
        setr("reactionIndexs_BER_dr5plyase", 0.5 * na)
        n = nd + nx
        ex = (
            self.NER_UvrABC_IncisionMargin5
            + self.NER_UvrABC_IncisionMargin3
            + 1
        )
        setr("reactionIndexs_NER_incision", n)
        setr("reactionIndexs_NER_excision", n * ex)
        setr("reactionIndexs_DNACleavage", n * ex / 4)
        setr(
            "reactionIndexs_HR_DNAResection",
            dsbr * 2 * self.HR_PolA_ResectionLength,
        )
        setr("reactionIndexs_HR_strandExchange", dsbr * 2 + nh * 2)
        setr(
            "reactionIndexs_HR_junctionMigration",
            nh
            * (
                c.sequenceLen
                / (
                    np.asarray(self.hollidayJunctionResolutionSites).shape[1]
                    / 2
                )
                / 2
            )
            / self.HR_RuvAB_JunctionMigrationHop,
        )
        setr("reactionIndexs_HR_junctionResolution", nh)
        if ng and na:
            setr(
                "reactionIndexs_polymerization",
                np.count_nonzero(
                    c.abasicSites.get_coords(c.gapSites.find()[0])
                ),
            )
        setr("reactionIndexs_ligation", ns)
        setr(
            "reactionIndexs_RM_MunI_Methylation",
            nnz(c.hemiunmethylatedMunIRMSites),
        )
        setr(
            "reactionIndexs_RM_MunI_Restriction",
            nnz(c.restrictableMunIRMSites),
        )
        enz = np.asarray(self.enzymes).reshape(-1)
        cap = np.ceil(
            np.asarray(self.reactionCatalysisMatrix) @ enz * self.stepSizeSec
        )
        return (
            np.maximum(
                0, -np.asarray(self.reactionSmallMoleculeStoichiometryMatrix)
            )
            @ np.minimum(cap.ravel(), rates)
        ).reshape(np.asarray(self.substrates).shape)

    def evolveState(self):
        if self._collapse(self.chromosome.damagedSites) == 0:
            order = [
                self.evolveState_Modification,
                self.evolveState_Restriction,
            ]
            if self.randStream.rand() <= 0.5:
                order.reverse()
        else:
            order = [getattr(self, n) for n in self._runtime_methods]
            order = [
                order[i]
                for i in np.asarray(
                    self.randStream.randperm(len(order)), int
                ).ravel()
                - 1
            ]
        for fn in order:
            try:
                fn()
            except Exception as exc:
                values = np.asarray(self.substrates)
                minimum = (
                    float(np.min(values)) if values.size else float("nan")
                )
                detail = f"DNARepair.{fn.__name__} failed; min substrate={minimum}, substrate shape={values.shape}"
                try:
                    exc.add_note(detail)
                except AttributeError:
                    exc.args = (*exc.args, detail)
                raise

    # Pathway implementations follow DNARepair.m. Coordinates passed to chromosome
    # APIs remain one based.
    def _reaction_limit(self, positions, reaction, enzyme):
        positions = np.asarray(positions, dtype=int).reshape((-1, 2))
        if not len(positions):
            return 0
        reaction = int(np.asarray(reaction).ravel()[0])
        enzyme = int(np.asarray(enzyme).ravel()[0])
        catalytic = int(
            np.asarray(
                self.randStream.stochasticRound(
                    self.enzymes[enzyme - 1]
                    * self.stepSizeSec
                    * self.enzymeBounds[reaction - 1, 1]
                )
            ).item()
        )
        costs = np.maximum(
            0,
            -np.asarray(self.reactionSmallMoleculeStoichiometryMatrix)[
                :, reaction - 1
            ],
        )
        limits = self._substrate_limits(costs)
        return max(
            0, min(len(positions), catalytic, int(np.floor(np.min(limits))))
        )

    def evolveState_BER(self):
        c = self.chromosome

        # Keep this routine compatible with both dense state arrays and the
        # value-semantic SparseMat objects used by snapshots.
        def dense(field):
            value = getattr(c, field)
            return (
                value.full() if hasattr(value, "full") else np.asarray(value)
            )

        def dense_coordinates(array):
            return self._matlab_find_coordinates(array)

        def coordinates(field, mask=None):
            value = getattr(c, field)
            if hasattr(value, "find"):
                pos, vals = value.find()
                pos = np.asarray(pos, dtype=int).reshape((-1, 2))
                if mask is not None:
                    vals = np.asarray(vals).reshape(-1)
                    pos = pos[np.asarray(mask(vals), dtype=bool)]
                return pos
            arr = np.asarray(value)
            if mask is not None:
                arr = mask(arr)
            return dense_coordinates(arr)

        def assign(field, positions, values):
            positions = np.asarray(positions, dtype=int).reshape((-1, 2))
            if not len(positions):
                return
            value = getattr(c, field)
            if hasattr(value, "set_coords"):
                setattr(c, field, value.set_coords(positions, values))
            else:
                arr = np.asarray(value).copy()
                arr[tuple((positions - 1).T)] = values
                setattr(c, field, arr)

        def reaction_count(positions, reaction, enzyme):
            if not len(positions):
                return 0
            reaction, enzyme = int(reaction), int(enzyme)
            catalytic = self.randStream.stochasticRound(
                self.enzymes[enzyme - 1]
                * self.enzymeBounds[reaction - 1, 1]
                * self.stepSizeSec
            )
            costs = np.maximum(
                0,
                -np.asarray(self.reactionSmallMoleculeStoichiometryMatrix)[
                    :, reaction - 1
                ],
            )
            limits = self._substrate_limits(costs)
            return max(
                0,
                int(
                    np.floor(
                        min(
                            len(positions),
                            float(np.asarray(catalytic).item()),
                            np.min(limits),
                        )
                    )
                ),
            )

        def stoich(reactions):
            self._substrate_update(
                np.sum(
                    np.asarray(self.reactionSmallMoleculeStoichiometryMatrix)[
                        :, np.asarray(reactions, dtype=int) - 1
                    ],
                    axis=1,
                )
            )

        # 1. Base excision (Fpg and Ung).
        damaged = getattr(c, "damagedBases")
        if self._collapse(damaged):
            enzyme_order = (
                np.asarray(
                    self.randStream.randperm(
                        len(self.enzymeIndexs_BER_baseExcision)
                    ),
                    dtype=int,
                ).reshape(-1)
                - 1
            )
            enzyme_indexes = np.asarray(
                self.enzymeIndexs_BER_baseExcision, dtype=int
            ).reshape(-1)
            reactions_all = np.asarray(
                self.reactionIndexs_BER_baseexcision, dtype=int
            ).reshape(-1)
            substrates = np.asarray(
                self.substrateGlobalIndexs, dtype=int
            ).reshape(-1)
            rx_bases = np.asarray(self.reactionDNABase, dtype=int).reshape(-1)
            cat = np.asarray(self.reactionCatalysisMatrix)
            for order_idx in enzyme_order:
                damaged = getattr(c, "damagedBases")
                enzyme = int(enzyme_indexes[order_idx])
                if self.enzymes[enzyme - 1] == 0:
                    continue
                rx = reactions_all[cat[reactions_all - 1, enzyme - 1] != 0]
                if not len(rx):
                    continue
                positions, global_bases = (
                    damaged.find()
                    if hasattr(damaged, "find")
                    else (
                        np.argwhere(np.asarray(damaged)),
                        np.asarray(damaged)[
                            tuple(np.argwhere(np.asarray(damaged)).T)
                        ],
                    )
                )
                positions = np.asarray(positions, dtype=int).reshape((-1, 2))
                if not hasattr(damaged, "find"):
                    positions = positions + 1
                global_bases = np.asarray(global_bases, dtype=int).reshape(-1)
                # Translate global metabolite IDs to substrate-local IDs, as
                # ismembc2 does in the MATLAB implementation.
                local_bases = np.zeros(len(global_bases), dtype=int)
                for j, gid in enumerate(global_bases):
                    found = np.flatnonzero(substrates == gid)
                    if found.size:
                        local_bases[j] = found[0] + 1
                reaction_bases = rx_bases[rx - 1]
                chosen_rx = np.zeros(len(positions), dtype=int)
                for j, base in enumerate(local_bases):
                    matched = np.flatnonzero(reaction_bases == base)
                    if matched.size:
                        chosen_rx[j] = rx[matched[0]]
                keep = chosen_rx != 0
                positions, chosen_rx = positions[keep], chosen_rx[keep]
                if not len(positions):
                    continue
                n = reaction_count(positions, chosen_rx[0], enzyme)
                if n == 0:
                    continue
                _, idxs, bound_positions, n, _ = self.bindProteinToChromosome(
                    positions, enzyme, n, None, False, True, 1, False, 1
                )
                if not len(idxs):
                    continue
                selected_rx = chosen_rx[
                    np.asarray(idxs, dtype=int).reshape(-1) - 1
                ]
                stoich(selected_rx)
                assign("damagedBases", bound_positions, 0)
                assign("abasicSites", bound_positions, 1)

        # 2. Remove abasic sugar phosphates by AP lyase or AP endonuclease.
        ap_pos = coordinates("abasicSites")
        if len(ap_pos):
            rx_order = (
                np.asarray(self.randStream.randperm(2), dtype=int).reshape(-1)
                - 1
            )
            rx_options = np.r_[
                np.asarray(self.reactionIndexs_BER_aplyase, dtype=int).reshape(
                    -1
                ),
                np.asarray(
                    self.reactionIndexs_BER_apendonuclease, dtype=int
                ).reshape(-1),
            ]
            for order_idx in rx_order:
                rx = int(rx_options[order_idx])
                enzymes = (
                    np.flatnonzero(
                        np.asarray(self.reactionCatalysisMatrix)[rx - 1, :]
                    )
                    + 1
                )
                if not enzymes.size:
                    continue
                enzyme = int(enzymes[0])
                if self.enzymes[enzyme - 1] == 0:
                    continue
                # abasic site without a neighboring strand break
                ap = dense("abasicSites") != 0
                # Derived views may be sparse in a snapshot; form the mask in
                # dense coordinates to preserve MATLAB's logical expression.
                eligible = ap & ~(
                    (dense("strandBreaks3") != 0)
                    | (dense("strandBreaks5") != 0)
                )
                positions = dense_coordinates(eligible)
                if not len(positions):
                    continue
                n = reaction_count(positions, rx, enzyme)
                if not n:
                    continue
                check = positions.copy()
                if rx == int(
                    np.asarray(self.reactionIndexs_BER_aplyase).reshape(-1)[0]
                ):
                    check[check[:, 1] % 2 == 0, 0] -= 1
                    _, idxs, _, n, _ = self.bindProteinToChromosome(
                        check, enzyme, n, None, False, True, 2, False, 1
                    )
                    if not n:
                        continue
                    positions = positions[
                        np.asarray(idxs, dtype=int).reshape(-1) - 1
                    ]
                    assign(
                        "strandBreaks",
                        c.shiftPositionsStrandsBond3Prime(positions),
                        1,
                    )
                else:
                    check[check[:, 1] % 2 == 1, 0] -= 1
                    _, idxs, _, n, _ = self.bindProteinToChromosome(
                        check, enzyme, n, None, False, True, 2, False, [1, 2]
                    )
                    if not n:
                        continue
                    positions = positions[
                        np.asarray(idxs, dtype=int).reshape(-1) - 1
                    ]
                    assign(
                        "strandBreaks",
                        c.shiftPositionsStrandsBond5Prime(positions),
                        1,
                    )
                assign("damagedSugarPhosphates", positions, 0)
                self._substrate_update(
                    n
                    * np.asarray(
                        self.reactionSmallMoleculeStoichiometryMatrix
                    )[:, rx - 1]
                )

        # 3. Nfo 5'-deoxyribosephosphodiesterase.
        positions = dense_coordinates(
            (dense("abasicSites") != 0) & (dense("strandBreaks3") != 0)
        )
        if len(positions):
            rx = int(
                np.asarray(self.reactionIndexs_BER_drpesterase).reshape(-1)[0]
            )
            enzyme = int(
                np.asarray(self.enzymeIndexs_apurinicEndonuclease).reshape(-1)[
                    0
                ]
            )
            n = reaction_count(positions, rx, enzyme)
            check = positions.copy()
            check[:, 0] -= 1
            _, idxs, _, n, _ = self.bindProteinToChromosome(
                check, enzyme, n, None, False, True, 2, False, [1, 2, 3]
            )
            if n:
                positions = positions[
                    np.asarray(idxs, dtype=int).reshape(-1) - 1
                ]
                assign("damagedSugarPhosphates", positions, 0)
                assign(
                    "strandBreaks",
                    c.shiftPositionsStrandsBond5Prime(positions),
                    1,
                )
                assign("gapSites", positions, 1)
                self._substrate_update(
                    n
                    * np.asarray(
                        self.reactionSmallMoleculeStoichiometryMatrix
                    )[:, rx - 1]
                )

        # 4. dnaN 3'-(deoxyribose-5'-phosphate) lyase.
        positions = dense_coordinates(
            (dense("abasicSites") != 0) & (dense("strandBreaks5") != 0)
        )
        if len(positions):
            rx = int(
                np.asarray(self.reactionIndexs_BER_dr5plyase).reshape(-1)[0]
            )
            enzyme = int(
                np.asarray(self.enzymeIndexs_polymerase).reshape(-1)[0]
            )
            n = reaction_count(positions, rx, enzyme)
            check = positions.copy()
            check[:, 0] -= 1
            _, idxs, _, n, _ = self.bindProteinToChromosome(
                check, enzyme, n, None, False, True, 2, False, [1, 2, 3]
            )
            if n:
                positions = positions[
                    np.asarray(idxs, dtype=int).reshape(-1) - 1
                ]
                assign("damagedSugarPhosphates", positions, 0)
                assign(
                    "strandBreaks",
                    c.shiftPositionsStrandsBond3Prime(positions),
                    1,
                )
                assign("gapSites", positions, 1)
                self._substrate_update(
                    n
                    * np.asarray(
                        self.reactionSmallMoleculeStoichiometryMatrix
                    )[:, rx - 1]
                )

    def evolveState_NER(self):
        """Model UvrABC incision, PcrA excision, and excised DNA cleavage."""
        c = self.chromosome
        rx = np.concatenate(
            [
                np.asarray(
                    self.reactionIndexs_NER_incision, dtype=int
                ).reshape(-1),
                np.asarray(
                    self.reactionIndexs_NER_excision, dtype=int
                ).reshape(-1),
                np.asarray(self.reactionIndexs_DNACleavage, dtype=int).reshape(
                    -1
                ),
            ]
        )
        cat = np.asarray(self.reactionCatalysisMatrix)
        enzymes = np.asarray(self.enzymes).reshape(-1)
        if not np.all(cat[rx - 1, :] @ enzymes):
            return
        if c.damagedBases.nnz() == 0 and c.intrastrandCrossLinks.nnz() == 0:
            return

        # Find XOR(damagedBases, intrastrandCrossLinks), preserving sparse coordinates.
        dbpos, dbval = c.damagedBases.find()
        xlpos, xlval = c.intrastrandCrossLinks.find()
        db = {tuple(p): v for p, v in zip(dbpos, dbval.reshape(-1))}
        xl = {tuple(p): v for p, v in zip(xlpos, xlval.reshape(-1))}
        damaged = np.asarray(
            [
                p
                for p in sorted(set(db) | set(xl))
                if bool(db.get(p, 0)) != bool(xl.get(p, 0))
            ],
            dtype=int,
        ).reshape(-1, 2)
        substrate_globals = np.asarray(
            self.substrateGlobalIndexs, dtype=int
        ).reshape(-1)
        m6ad = int(
            substrate_globals[
                int(np.asarray(self.substrateIndexs_m6AD).reshape(-1)[0]) - 1
            ]
        )
        if len(damaged):
            damaged = damaged[
                np.asarray(c.damagedBases.get_coords(damaged)).reshape(-1)
                != m6ad
            ]
        if not len(damaged):
            return

        # Exclude methylated MunI modification sites, except m6AD damage.
        sites = np.asarray(self.RM_MunI_RecognitionSites, dtype=int)
        methylated = np.asarray(
            self.RM_MunI_MethylatedPositions, dtype=int
        ).reshape(-1)
        rmpos = sites[:, methylated - 1].reshape(-1, order="F")
        rm = (
            np.column_stack((rmpos, np.tile([1, 2], len(rmpos) // 2)))
            if len(rmpos)
            else np.empty((0, 2), int)
        )
        if len(rm):
            rm = rm[
                np.asarray(c.damagedBases.get_coords(rm)).reshape(-1) != m6ad
            ]
            rmset = {tuple(p) for p in rm}
            damaged = np.asarray(
                [p for p in damaged if tuple(p) not in rmset], dtype=int
            ).reshape(-1, 2)
        if not len(damaged):
            return

        lengths = int(
            self.NER_UvrABC_IncisionMargin5
            + self.NER_UvrABC_IncisionMargin3
            + 2
        )
        excised = np.arange(2, lengths + 1, dtype=int)
        incised = np.asarray([1, lengths], dtype=int)
        n_excised, n_incised = len(excised), len(incised)
        positions = damaged.copy()
        ignore = np.ones((len(positions), 2), dtype=int)
        xvals = np.asarray(
            c.intrastrandCrossLinks.get_coords(damaged)
        ).reshape(-1)
        for i in np.flatnonzero(xvals):
            ignore[i, 1] += 2 * (damaged[i, 1] % 2 == 1) - 1

        rxcounts = np.r_[
            1,
            int(np.ceil(n_excised / self.NER_PcrA_StepSize)),
            np.full(4, int(np.ceil(n_excised / 4)), dtype=int),
        ]
        sm = np.asarray(self.reactionSmallMoleculeStoichiometryMatrix)
        rates = (cat[rx - 1, :] @ enzymes) * float(self.stepSizeSec)
        enzyme_caps = np.asarray(self.enzymeBounds)[rx - 1, 1] * rxcounts
        substrate_caps = np.asarray(self.substrates).reshape(-1) / (
            np.maximum(0, -sm[:, rx - 1]) @ rxcounts
        )
        n = max(
            0,
            int(
                np.floor(
                    np.min(
                        np.r_[
                            len(positions),
                            self.randStream.stochasticRound(
                                rates * enzyme_caps
                            ),
                            substrate_caps,
                        ]
                    )
                )
            ),
        )
        if n == 0:
            return

        _, idxs, positions, _, _ = self.bindProteinToChromosome(
            positions,
            np.any(cat[rx - 1, :], axis=0),
            n,
            [],
            False,
            True,
            lengths,
            False,
            ignore,
        )
        idxs = np.asarray(idxs, dtype=int).reshape(-1)
        if not len(idxs):
            return
        damaged = damaged[idxs - 1]
        n = len(positions)
        reverse = positions[:, 1] % 2 == 1
        positions[reverse, 0] -= int(self.NER_UvrABC_IncisionMargin5) + 1
        positions[~reverse, 0] -= int(self.NER_UvrABC_IncisionMargin3) + 1

        rxcounts[0] = n
        rxcounts[1] = int(np.ceil(n_excised / self.NER_PcrA_StepSize)) * n
        excpos = (positions[:, 0, None] + excised[None, :] - 1).reshape(-1)
        excstr = np.repeat(positions[:, 1], n_excised)
        rxcounts[2:6] = np.asarray(
            c.sequence.subsequenceBaseCounts(excpos, excstr)
        ).reshape(-1)
        substrate_shape = np.asarray(self.substrates).shape
        self.substrates = (
            np.asarray(self.substrates).reshape(-1) + sm[:, rx - 1] @ rxcounts
        )

        # Correct metabolites for damaged bases.
        damage_values = np.asarray(c.damagedBases.get_coords(damaged)).reshape(
            -1
        )
        damaged_idx = np.flatnonzero(damage_values)
        if len(damaged_idx):
            undamaged = np.asarray(
                c.sequence.subsequenceBaseCounts(damaged[damaged_idx])
            ).reshape(-1)
            types, counts = np.unique(
                damage_values[damaged_idx], return_counts=True
            )
            rows = {int(g): i for i, g in enumerate(substrate_globals)}
            type_rows = np.asarray([rows[int(t)] for t in types], dtype=int)
            self.substrates[
                np.asarray(self.substrateIndexs_dNMPs, dtype=int).reshape(-1)
                - 1
            ] -= undamaged
            self.substrates[type_rows] += counts

        # Correct metabolites for intrastrand cross links.
        damage_values = np.asarray(
            c.intrastrandCrossLinks.get_coords(damaged)
        ).reshape(-1)
        cross_idx = np.flatnonzero(damage_values)
        if len(cross_idx):
            p = damaged[cross_idx]
            offsets = (2 * (p[:, 1] % 2 == 1).astype(int) - 1)[
                :, None
            ] * np.asarray([[0, 1]])
            bases = np.asarray(
                c.sequence.subsequenceBaseCounts(
                    p[:, 0, None] + offsets,
                    np.repeat(p[:, 1, None], 2, axis=1),
                )
            ).reshape(-1)
            types, counts = np.unique(
                damage_values[cross_idx], return_counts=True
            )
            rows = {int(g): i for i, g in enumerate(substrate_globals)}
            type_rows = np.asarray([rows[int(t)] for t in types], dtype=int)
            self.substrates[
                np.asarray(self.substrateIndexs_dNMPs, dtype=int).reshape(-1)
                - 1
            ] -= bases
            self.substrates[type_rows] += counts
            self.substrates[
                int(np.asarray(self.substrateIndexs_water).reshape(-1)[0]) - 1
            ] += counts.sum()
            self.substrates[
                int(np.asarray(self.substrateIndexs_hydrogen).reshape(-1)[0])
                - 1
            ] -= counts.sum()

        incised_coords = np.column_stack(
            (
                (positions[:, 0, None] + incised[None, :] - 1).reshape(-1),
                np.repeat(positions[:, 1], n_incised),
            )
        )
        excised_coords = np.column_stack((excpos, excstr))
        c.strandBreaks = c.strandBreaks.set_coords(incised_coords, 1)
        c.gapSites = c.gapSites.set_coords(excised_coords, 1)
        c.abasicSites = c.abasicSites.set_coords(excised_coords, 1)
        c.damagedBases = c.damagedBases.set_coords(excised_coords, 0)
        c.intrastrandCrossLinks = c.intrastrandCrossLinks.set_coords(
            excised_coords, 0
        )
        self.substrates = np.asarray(self.substrates).reshape(substrate_shape)

    def evolveState_HR(self):
        c = self.chromosome

        def find(mat):
            if hasattr(mat, "find"):
                p, v = mat.find()
                return np.asarray(p, int).reshape((-1, 2)), np.asarray(
                    v
                ).reshape(-1)
            a = np.asarray(mat)
            p = self._matlab_find_coordinates(a != 0)
            values = (
                a[tuple((p - 1).T)] if len(p) else np.zeros(0, dtype=a.dtype)
            )
            return p, values

        def get(mat, coords):
            if hasattr(mat, "get_coords"):
                return np.asarray(mat.get_coords(coords)).reshape(-1)
            a = np.asarray(mat)
            q = np.asarray(coords, int) - 1
            return a[tuple(q.T)] if len(q) else np.zeros(0)

        def put(field, coords, values):
            mat = getattr(c, field)
            coords = np.asarray(coords, int).reshape((-1, 2))
            if not len(coords):
                return
            if hasattr(mat, "set_coords"):
                setattr(c, field, mat.set_coords(coords, values))
            else:
                a = np.asarray(mat).copy()
                a[tuple((coords - 1).T)] = values
                setattr(c, field, a)

        # 5'-3' resection and first strand exchange.
        breaks, vals = find(c.doubleStrandBreaks)
        breaks = (
            breaks[(breaks[:, 1] % 2 == 1) & (vals != 0)]
            if len(breaks)
            else breaks
        )
        # MATLAB searches `doubleStrandBreaks(:, 1:2:end)`, so the returned
        # second coordinate is chromosome-copy numbered.
        breaks = self._chromosome_copy_coordinates(breaks)
        nres = 2 * int(self.HR_PolA_ResectionLength)
        if len(breaks):
            rx = np.r_[
                np.asarray(self.reactionIndexs_HR_strandExchange, int).ravel(),
                np.asarray(self.reactionIndexs_HR_DNAResection, int).ravel()[
                    :1
                ],
            ]
            counts = np.r_[2, nres]
            sm = np.asarray(self.reactionSmallMoleculeStoichiometryMatrix)
            costs = np.maximum(0, -sm[:, rx - 1]) @ counts
            available = np.min(
                np.divide(
                    np.asarray(self.substrates).reshape(-1),
                    costs,
                    out=np.full_like(costs, np.inf, dtype=float),
                    where=costs > 0,
                )
            )
            cat = np.asarray(
                self.randStream.stochasticRound(
                    (
                        np.asarray(self.reactionCatalysisMatrix)[rx - 1, :]
                        @ np.asarray(self.enzymes).reshape(-1)
                        * self.stepSizeSec
                    )
                    * (np.asarray(self.enzymeBounds)[rx - 1, 1] * counts)
                )
            )
            reca = int(
                np.asarray(self.enzymes).reshape(-1)[
                    int(
                        np.asarray(
                            self.enzymeIndexs_recombinationStrandExchange
                        ).ravel()[0]
                    )
                    - 1
                ]
            )
            n = max(
                0,
                int(
                    np.floor(
                        min(
                            len(breaks),
                            available,
                            np.min(cat),
                            reca
                            // (
                                2
                                * int(
                                    np.ceil(
                                        nres / 2 / int(self.HR_RecA_Spacing)
                                    )
                                )
                            ),
                        )
                    )
                ),
            )
            positions = np.column_stack(
                (
                    breaks[:, 0] - int(self.HR_PolA_ResectionLength),
                    2 * breaks[:, 1],
                )
            )
            _, idx, _, n, _ = self.bindProteinToChromosome(
                positions,
                self.enzymeIndexs_exonuclease53,
                n,
                None,
                False,
                True,
                nres,
                False,
                [
                    int(self.HR_PolA_ResectionLength) + 1,
                    int(self.HR_PolA_ResectionLength) + 2,
                ],
            )
            if n:
                selected = self._matlab_rows(breaks, idx)
                pos = selected[:, 0] - int(self.HR_PolA_ResectionLength)
                chrids = selected[:, 1]
                k = int(self.HR_PolA_ResectionLength)
                d = len(pos)
                p1 = pos[:, None] + np.arange(1, k + 1)
                p2 = pos[:, None] + np.arange(-k + 1, 1)
                put(
                    "gapSites",
                    np.column_stack((p1.ravel(), np.ones(p1.size, int))),
                    1,
                )
                put(
                    "gapSites",
                    np.column_stack((p2.ravel(), np.full(p2.size, 2, int))),
                    1,
                )
                put(
                    "abasicSites",
                    np.column_stack((p1.ravel(), np.ones(p1.size, int))),
                    1,
                )
                put(
                    "abasicSites",
                    np.column_stack((p2.ravel(), np.full(p2.size, 2, int))),
                    1,
                )
                put(
                    "hollidayJunctions",
                    np.column_stack((pos + k, np.ones(d, int))),
                    1,
                )
                put(
                    "hollidayJunctions",
                    np.column_stack((pos - k, np.full(d, 2, int))),
                    1,
                )
                rxs = np.r_[
                    np.asarray(
                        self.reactionIndexs_HR_strandExchange, int
                    ).ravel(),
                    np.asarray(
                        self.reactionIndexs_HR_DNAResection, int
                    ).ravel(),
                ]
                rc = np.r_[
                    2 * n,
                    np.asarray(
                        c.sequence.subsequenceBaseCounts(
                            p1.ravel(), np.ones(p1.size, int)
                        )
                    ).reshape(-1)
                    + np.asarray(
                        c.sequence.subsequenceBaseCounts(
                            p2.ravel(), np.full(p2.size, 2, int)
                        )
                    ).reshape(-1),
                ]
                self._substrate_update(sm[:, rxs - 1] @ rc)
                nr = int(np.ceil(nres / 2 / int(self.HR_RecA_Spacing)))
                r1 = pos[:, None] + np.asarray(
                    self.randStream.stochasticRound(
                        np.arange(1, nr + 1) * k / nr
                    )
                ).reshape(1, -1)
                r2 = pos[:, None] + np.asarray(
                    self.randStream.stochasticRound(
                        np.arange(-nr + 1, 1) * (k - 1) / (nr - 1)
                    )
                ).reshape(1, -1)
                recapos = np.vstack(
                    (
                        np.column_stack(
                            (r1.ravel(), np.repeat(2 * chrids - 1, nr))
                        ),
                        np.column_stack(
                            (r2.ravel(), np.repeat(2 * chrids, nr))
                        ),
                    )
                )
                self.bindProteinToChromosome(
                    recapos,
                    self.enzymeIndexs_recombinationStrandExchange,
                    None,
                    None,
                    True,
                    True,
                    1,
                    False,
                    np.arange(-k - 1, k + 1),
                )

        # Junction migration and resolution.
        hj, hv = find(c.hollidayJunctions)
        hj = hj[(hj[:, 1] % 2 == 1) & (hv != 0)] if len(hj) else hj
        # Match find(hollidayJunctions(:, 1:2:end))'s compressed columns.
        hj = self._chromosome_copy_coordinates(hj)
        if len(hj):
            k = nres
            nchr = (
                np.asarray(c.hollidayJunctions.siz)[1] // 2
                if hasattr(c.hollidayJunctions, "siz")
                else np.asarray(c.hollidayJunctions).shape[1] // 2
            )
            for row in (
                np.asarray(self.randStream.randperm(len(hj)), int).ravel() - 1
            ):
                pos, chridx = hj[row]
                otherpos = (pos - k - 1) % int(c.sequenceLen) + 1
                if not np.any(
                    get(
                        c.hollidayJunctions, np.array([[otherpos, 2 * chridx]])
                    )
                ):
                    continue
                formation = np.array(
                    [
                        (pos - 1) % int(c.sequenceLen) + 1,
                        (pos - k - 1) % int(c.sequenceLen) + 1,
                    ]
                )
                sites = np.asarray(
                    self.hollidayJunctionResolutionSites, int
                ).reshape((-1, 2))
                s1 = sites[sites[:, 1] == 1, 0]
                s2 = sites[sites[:, 1] == 2, 0]
                q1 = (
                    s1[0]
                    if formation[0] > s1[-1]
                    else s1[s1 > formation[0]][0]
                )
                q2 = (
                    s2[-1]
                    if formation[1] < s2[0]
                    else s2[s2 < formation[1]][-1]
                )
                resolve = np.array([q1, q2])
                hops = int(
                    np.sum(
                        self.randStream.stochasticRound(
                            np.abs(resolve - formation)
                            / float(self.HR_RuvAB_JunctionMigrationHop)
                        )
                    )
                )
                rx = np.r_[
                    np.asarray(
                        self.reactionIndexs_HR_strandExchange, int
                    ).ravel(),
                    np.asarray(
                        self.reactionIndexs_HR_junctionMigration, int
                    ).ravel(),
                    np.asarray(
                        self.reactionIndexs_HR_junctionResolution, int
                    ).ravel(),
                ]
                cnt = np.r_[2, hops, 1]
                sm = np.asarray(self.reactionSmallMoleculeStoichiometryMatrix)
                need = np.maximum(0, -sm[:, rx - 1]) @ cnt
                cap = self.randStream.stochasticRound(
                    np.asarray(self.reactionCatalysisMatrix)[rx - 1, :]
                    @ np.asarray(self.enzymes).reshape(-1)
                    * self.stepSizeSec
                    * np.asarray(self.enzymeBounds)[rx - 1, 1]
                )
                if np.any(
                    np.asarray(self.substrates).reshape(-1) < need
                ) or np.any(np.asarray(cap) < 1):
                    continue
                length = (q1 - q2) % int(c.sequenceLen) + 1
                ignore = (
                    (formation[:, None] + np.arange(-1, 2)[None, :] - q2 - 1)
                    % int(c.sequenceLen)
                    + 1
                ).ravel()
                if not self.bindProteinToChromosome(
                    [[q2, 2 * chridx]],
                    self.enzymeIndexs_hollidayJunctionHelicase,
                    1,
                    None,
                    False,
                    True,
                    length,
                    False,
                    ignore,
                )[3]:
                    continue
                self._substrate_update(sm[:, rx - 1] @ cnt)
                put(
                    "hollidayJunctions",
                    [
                        [formation[0], 2 * chridx - 1],
                        [formation[1], 2 * chridx],
                    ],
                    0,
                )
                put(
                    "hollidayJunctions",
                    [[q1, 2 * chridx - 1], [q2, 2 * chridx]],
                    0,
                )
                put(
                    "strandBreaks", [[q1, 2 * chridx - 1], [q2, 2 * chridx]], 1
                )

    def evolveState_Polymerize(self):
        c = self.chromosome
        gap_subs, _ = c.gapSites.find()
        abase_subs, _ = c.abasicSites.find()
        breaks_subs, _ = c.strandBreaks.find()
        if not len(gap_subs) or not len(abase_subs) or not len(breaks_subs):
            return

        # SparseMat follows MATLAB's one-based [position, strand] coordinates.
        def intersect(a, b):
            common = np.asarray(
                list(set(map(tuple, a)) & set(map(tuple, b))), dtype=int
            ).reshape((-1, 2))
            return self._sort_matlab_subscripts(common, c.sequenceLen)

        gap_starts = intersect(
            intersect(gap_subs, abase_subs), c.strandBreaks5.find()[0]
        )
        if not len(gap_starts):
            return
        gap_cont = intersect(
            intersect(gap_subs, abase_subs),
            intersect(c.gapSites5.find()[0], c.abasicSites5.find()[0]),
        )
        gap_cont = np.asarray(
            [
                p
                for p in gap_cont
                if not len(c.strandBreaks5.get_coords([p]))
                or c.strandBreaks5.get_coords([p])[0, 0] == 0
            ],
            dtype=int,
        ).reshape((-1, 2))
        gaps = np.unique(np.vstack((gap_starts, gap_cont)), axis=0)
        gaps = self._sort_matlab_subscripts(gaps, c.sequenceLen)
        if not len(gaps):
            return

        # Exclude damaged sites before grouping adjacent feature bases.
        excluded = [
            c.damagedSugarPhosphates3,
            c.damagedSugarPhosphates5,
            c.damagedBases,
            c.intrastrandCrossLinks3,
            c.intrastrandCrossLinks5,
        ]
        keep = np.ones(len(gaps), dtype=bool)
        for view in excluded:
            keep &= np.asarray(view.get_coords(gaps)).reshape(-1) == 0
        gaps = gaps[keep]
        if not len(gaps):
            return

        # ChromosomeSequence.locateFeatures groups adjacent sites independently
        # per strand, joining features that cross the circular origin.
        positions, lengths, strands, sequences = [], [], [], []
        seq_obj = c.sequence
        for strand in range(1, int(np.max(gaps[:, 1])) + 1):
            ps = np.sort(gaps[gaps[:, 1] == strand, 0])
            if not len(ps):
                continue
            runs = []
            for p in ps:
                if runs and p == runs[-1][-1] + 1:
                    runs[-1].append(int(p))
                else:
                    runs.append([int(p)])
            if (
                len(runs) > 1
                and runs[0][0] == 1
                and runs[-1][-1] == len(seq_obj)
            ):
                if strand % 2:
                    runs[-1].extend(runs[0])
                    runs.pop(0)
                else:
                    runs[0] = runs[-1] + runs[0]
                    runs.pop()
            for run in runs:
                start = run[0] if strand % 2 else run[-1]
                positions.append(start)
                lengths.append(len(run))
                strands.append(strand)
                direction = 1 if strand % 2 else -1
                readpos = start + direction * np.arange(len(run))
                sequences.append(seq_obj.subsequence(readpos, strand))
        if not positions:
            return
        positions, lengths, strands = (
            np.asarray(positions),
            np.asarray(lengths),
            np.asarray(strands),
        )
        dirs = np.where(strands % 2 == 1, 1, -1)

        from ..polymerize import polymerize

        enzyme = int(np.asarray(self.enzymeIndexs_polymerase).reshape(-1)[0])
        reactions = np.asarray(
            self.reactionIndexs_polymerization, dtype=int
        ).reshape(-1)
        reaction = int(reactions[0])
        dntp_indices = (
            np.asarray(self.substrateIndexs_dNTPs, dtype=int).reshape(-1) - 1
        )
        dntp_amounts = np.asarray(self.substrates).reshape(-1)[dntp_indices]
        polymerase = int(
            np.asarray(
                self.randStream.stochasticRound(
                    self.enzymes[enzyme - 1]
                    * self.enzymeBounds[reaction - 1, 1]
                    * self.stepSizeSec
                )
            ).item()
        )
        rand_order = (
            np.asarray(
                self.randStream.randperm(len(positions)), dtype=int
            ).reshape(-1)
            - 1
        )
        _, footprint3, footprint5, _, _ = c.getDNAFootprint(
            self.enzymeGlobalIndexs[enzyme - 1], []
        )
        for i in rand_order:
            bases = min(int(lengths[i]), polymerase)
            if bases <= 0:
                break
            seq = sequences[i][:bases]
            progress = int(
                polymerize(
                    np.asarray([list(seq)], dtype="U1"),
                    dntp_amounts,
                    "ACGT",
                    "\0",
                    0,
                    0,
                    self.randStream,
                )[0][0, 0]
            )
            if progress == 0:
                continue
            if strands[i] % 2:
                ignore_min, ignore_max = (
                    -1 - footprint5,
                    int(lengths[i]) + 1 + footprint3,
                )
            else:
                ignore_min, ignore_max = (
                    -1 - footprint3,
                    int(lengths[i]) + 1 + footprint5,
                )
            ignore = dirs[i] * np.arange(ignore_min, ignore_max + 1) + 1
            anchor = c.shiftPositionsStrandsBond5Prime(
                [positions[i], strands[i]]
            )
            _, _, _, _, bound_length = self.bindProteinToChromosome(
                anchor,
                enzyme,
                1,
                None,
                False,
                True,
                dirs[i] * (progress + 1),
                True,
                ignore,
            )
            progress = min(
                abs(int(np.asarray(bound_length).reshape(-1)[0])),
                int(lengths[i]),
            )
            if not progress:
                continue
            _, _, costs, _, _ = polymerize(
                np.asarray([list(sequences[i][:progress])], dtype="U1"),
                dntp_amounts,
                "ACGT",
                "\0",
                0,
                0,
                self.randStream,
            )
            polymerase -= progress
            stoich = np.asarray(self.reactionSmallMoleculeStoichiometryMatrix)[
                :, reactions - 1
            ]
            self._substrate_update(stoich @ costs.reshape(-1))
            dntp_amounts = dntp_amounts - costs.reshape(-1)
            pos = positions[i] + dirs[i] * np.arange(progress)
            c.gapSites = c.gapSites.set_coords(
                np.c_[pos, np.full(progress, strands[i])], 0
            )
            c.abasicSites = c.abasicSites.set_coords(
                np.c_[pos, np.full(progress, strands[i])], 0
            )
            five = c.shiftPositionsStrandsBond5Prime([pos[0], strands[i]])
            three = c.shiftPositionsStrandsBond3Prime([pos[-1], strands[i]])
            c.strandBreaks = c.strandBreaks.set_coords(five, 0).set_coords(
                three, 1
            )

    def evolveState_Ligate(self):
        c = self.chromosome
        if len(c.strandBreaks.find()[0]) == 0:
            return
        positions = np.asarray(
            c.singleStrandBreaks.find()[0], dtype=int
        ).reshape((-1, 2))
        if positions.size == 0:
            return
        reaction = int(np.asarray(self.reactionIndexs_ligation).ravel()[0])
        enzyme = int(np.asarray(self.enzymeIndexs_ligase).ravel()[0])
        substrate_shape = np.asarray(self.substrates).shape
        substrates = np.asarray(self.substrates).reshape(-1)
        stoich = np.asarray(self.reactionSmallMoleculeStoichiometryMatrix)[
            :, reaction - 1
        ].reshape(-1)
        catalytic = self.randStream.stochasticRound(
            self.enzymes[enzyme - 1]
            * self.stepSizeSec
            * self.enzymeBounds[reaction - 1, 1]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            limits = substrates / np.maximum(0, -stoich)
        limits = limits[~np.isnan(limits)]
        resource_limit = np.min(limits) if limits.size else np.inf
        n = int(
            max(
                0,
                np.floor(
                    min(
                        len(positions),
                        float(np.asarray(catalytic).item()),
                        resource_limit,
                    )
                ),
            )
        )
        if n == 0:
            return
        _, _, positions, n, _ = self.bindProteinToChromosome(
            positions, enzyme, n, None, False, True, 1, False, [0, 1, 2]
        )
        if n == 0:
            return
        c.strandBreaks = c.strandBreaks.set_coords(
            positions, np.zeros(n, dtype=int)
        )
        self.substrates = (substrates + n * stoich).reshape(substrate_shape)

    def evolveState_DisA(self):
        positions = np.asarray(
            self.chromosome.damagedSites_excm6AD.find()[0], dtype=int
        ).reshape((-1, 2))
        if positions.size == 0:
            return
        enzyme = int(np.asarray(self.enzymeIndexs_DisA).ravel()[0])
        self.bindProteinToChromosome(
            positions,
            enzyme,
            self.enzymes[enzyme - 1],
            None,
            True,
            True,
            1,
            False,
            1,
        )

    def evolveState_Modification(self):
        c = self.chromosome
        positions = np.asarray(
            c.hemiunmethylatedMunIRMSites.find()[0], dtype=int
        ).reshape((-1, 2))
        if positions.size == 0:
            return
        reaction = int(
            np.asarray(self.reactionIndexs_RM_MunI_Methylation).ravel()[0]
        )
        enzyme = int(np.asarray(self.enzymeIndexs_RM_typeII).ravel()[0])
        substrate_shape = np.asarray(self.substrates).shape
        substrates = np.asarray(self.substrates).reshape(-1)
        stoich = np.asarray(self.reactionSmallMoleculeStoichiometryMatrix)[
            :, reaction - 1
        ].reshape(-1)
        catalytic = self.randStream.stochasticRound(
            self.enzymes[enzyme - 1]
            * self.stepSizeSec
            * self.enzymeBounds[reaction - 1, 1]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            resource_limits = substrates / np.maximum(0, -stoich)
        # MATLAB min omits NaN by default (0/0 for unused metabolites).
        resource_limits = resource_limits[~np.isnan(resource_limits)]
        resource_limit = (
            np.min(resource_limits) if resource_limits.size else np.inf
        )
        n = int(
            max(
                0,
                np.floor(
                    min(
                        len(positions),
                        resource_limit,
                        float(np.asarray(catalytic).item()),
                    )
                ),
            )
        )
        if n == 0:
            return
        _, _, positions, n, _ = self.bindProteinToChromosome(
            positions, enzyme, n, None, False, True, 1, False, [0, 1, 2]
        )
        if n == 0:
            return
        met = int(
            np.asarray(self.substrateGlobalIndexs).ravel()[
                int(self.substrateIndexs_m6AD) - 1
            ]
        )
        c.damagedBases = c.damagedBases.set_coords(
            positions, np.full(n, met, dtype=int)
        )
        self.substrates = (substrates + n * stoich).reshape(substrate_shape)

    def evolveState_Restriction(self):
        c = self.chromosome
        positions = np.asarray(
            c.restrictableMunIRMSites.find()[0], dtype=int
        ).reshape((-1, 2))
        if positions.size == 0:
            return
        reaction = int(
            np.asarray(self.reactionIndexs_RM_MunI_Restriction).ravel()[0]
        )
        enzyme = int(np.asarray(self.enzymeIndexs_RM_typeII).ravel()[0])
        substrate_shape = np.asarray(self.substrates).shape
        substrates = np.asarray(self.substrates).reshape(-1)
        stoich = np.asarray(self.reactionSmallMoleculeStoichiometryMatrix)[
            :, reaction - 1
        ].reshape(-1)
        catalytic = self.randStream.stochasticRound(
            self.enzymes[enzyme - 1]
            * self.stepSizeSec
            * self.enzymeBounds[reaction - 1, 1]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            resource_limits = substrates / np.maximum(0, -stoich)
        # Match MATLAB min's default NaN omission for 0/0 entries.
        resource_limits = resource_limits[~np.isnan(resource_limits)]
        resource_limit = (
            np.min(resource_limits) if resource_limits.size else np.inf
        )
        n = int(
            max(
                0,
                np.floor(
                    min(
                        len(positions),
                        resource_limit,
                        float(np.asarray(catalytic).item()),
                    )
                ),
            )
        )
        if n == 0:
            return
        _, _, positions, n, _ = self.bindProteinToChromosome(
            positions, enzyme, n, None, False, True, 1, False, [-1, 0, 1]
        )
        if n == 0:
            return
        c.strandBreaks = c.strandBreaks.set_coords(
            positions, np.ones(n, dtype=int)
        )
        self.substrates = (substrates + n * stoich).reshape(substrate_shape)
