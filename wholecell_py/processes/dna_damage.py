"""DNA lesion generation, ported from WholeCell DNADamage.m."""

from __future__ import annotations

import numpy as np

from ..process_base import Process, _get
from ..reaction_process import ReactionProcess
from ..chromosome_process_aspect import ChromosomeProcessAspect


class DNADamage(ChromosomeProcessAspect, ReactionProcess):
    optionNames__ = ()
    fixedConstantNames__ = (
        "reactionVulnerableMotifs",
        "reactionVulnerableMotifTypes",
        "reactionDamageTypes",
        "reactionSmallMoleculeStoichiometryMatrix",
        "reactionDNAStoichiometryMatrix",
        "reactionRadiationStoichiometryMatrix",
        "reactionDNAProduct",
        "reactionRadiation",
    )
    fittedConstantNames__ = ()
    localStateNames__ = ()
    unmodifiedNucleobases = "ACGT"
    stimuliWholeCellModelIDs = ()
    substrateWholeCellModelIDs = ()
    # MATLAB uses one-based local positions for these reaction index vectors.
    _reaction_type_fields = {
        "spontaneous base loss": "reactionIndexs_spontaneousBaseLoss",
        "spontaneous base deamination": "reactionIndexs_spontaneousBaseDeamination",
        "radiation (gamma-ray) induced base oxidation": "reactionIndexs_gammaRayBaseOxidation",
        "UV-B photodimerization": "reactionIndexs_uvbDimerization",
        "base reduction": "reactionIndexs_baseReduction",
        "base amination": "reactionIndexs_baseAmination",
        "base glucosyl transfer": "reactionIndexs_baseGlucosylTransfer",
        "base alkylation": "reactionIndexs_baseAlkylation",
        "base ethylation": "reactionIndexs_baseEthylation",
        "base methylation": "reactionIndexs_baseMethylation",
        "photooxidation": "reactionIndexs_photooxidation",
        "strand break": "reactionIndexs_strandBreak",
    }

    def __init__(
        self,
        wholeCellModelID="Process_DNADamage",
        name="DNA Damage",
        **properties,
    ):
        super().__init__(wholeCellModelID, name, **properties)
        self.chromosome = getattr(self, "chromosome", None)

    def storeObjectReferences(self, simulation):
        Process.storeObjectReferences(self, simulation)
        fn = getattr(simulation, "state", None)
        if callable(fn):
            self.chromosome = fn("Chromosome")
        else:
            states = _get(simulation, "states")
            self.chromosome = {str(k).lower(): v for k, v in states.items()}[
                "chromosome"
            ]
        self.states = [*self.states, self.chromosome]

    def initializeConstants(
        self, knowledgeBase=None, simulation=None, **kwargs
    ):
        options = kwargs.pop("options", None)
        if knowledgeBase is not None or simulation is not None:
            if knowledgeBase is None or simulation is None:
                raise ValueError(
                    "knowledgeBase and simulation must be provided together"
                )
            ReactionProcess.initializeConstants(
                self, knowledgeBase, simulation, options
            )
            ChromosomeProcessAspect.initializeConstants(
                self, knowledgeBase, simulation
            )

            # Keep reactions with nonzero forward rate, then add all substrate
            # classes referenced by DNA lesions even when absent from a reaction
            # after compartment collapse.
            active = np.asarray(self.reactionBounds)[:, 1] != 0
            active_ids = np.asarray(
                self.reactionWholeCellModelIDs, dtype=object
            ).ravel()[active]
            self.initializeConstants_overrideReactions(simulation, active_ids)
            reaction_types = np.asarray(
                self.reactionTypes, dtype=object
            ).ravel(order="F")
            for label, field in self._reaction_type_fields.items():
                setattr(
                    self,
                    field,
                    np.flatnonzero(reaction_types == label) + 1,
                )
            metabolites = np.asarray(
                _get(knowledgeBase, "metabolites"), dtype=object
            ).ravel(order="F")
            modified_categories = {
                "modified nucleobase",
                "modified deoxyribonucleoside",
                "modified deoxyribonucleotide monophosphate",
            }
            extras = {
                "AD",
                "CSN",
                "GN",
                "THY",
                "DA",
                "DC",
                "DG",
                "DT",
                "DAMP",
                "DCMP",
                "DGMP",
                "DTMP",
                "DR5P",
                "dRibose5P_dRibose5P",
                "URA",
                "URI",
                "DUMP",
            }
            extras.update(
                _get(metabolite, "wholeCellModelID")
                for metabolite in metabolites
                if _get(metabolite, "category") in modified_categories
            )
            wanted = extras | set(
                np.asarray(
                    self.substrateWholeCellModelIDs, dtype=object
                ).ravel()
            )
            metabolite_ids = np.asarray(
                _get(simulation.state("Metabolite"), "wholeCellModelIDs"),
                dtype=object,
            ).ravel(order="F")
            missing_ids = sorted(wanted.difference(metabolite_ids.tolist()))
            if missing_ids:
                raise ValueError(
                    f"DNADamage substrates missing from Metabolite state: {missing_ids}"
                )
            self.initializeConstants_overrideSubstrates(
                np.asarray(
                    [x for x in metabolite_ids if x in wanted], dtype=object
                )
            )

            self.substrateIndexs_radiation = self.substrateIndexs(
                ["UVB_radiation", "gamma_radiation"]
            )
            self.substrateIndexs_hydrogen = self.substrateIndexs(["H"])
            self.substrateIndexs_hydroxylRadical = self.substrateIndexs(
                ["hydroxyl_radical"]
            )
            self.substrateIndexs_water = self.substrateIndexs(["H2O"])
            self.substrateIndexs_DR5P = self.substrateIndexs(["DR5P"])
            self.substrateIndexs_dRibose5P_dRibose5P = self.substrateIndexs(
                ["dRibose5P_dRibose5P"]
            )
            self.substrateIndexs_unmodifiedNucleobases = self.substrateIndexs(
                ["AD", "CSN", "GN", "THY"]
            )
            self.substrateIndexs_unmodifiedNucleosides = self.substrateIndexs(
                ["DA", "DC", "DG", "DT"]
            )
            self.substrateIndexs_unmodifiedDNMPs = self.substrateIndexs(
                ["DAMP", "DCMP", "DGMP", "DTMP"]
            )
            categories = np.asarray(
                [_get(m, "category") for m in metabolites], dtype=object
            )
            ids = np.asarray(
                [_get(m, "wholeCellModelID") for m in metabolites],
                dtype=object,
            )
            local_by_id = {
                str(identifier): i + 1
                for i, identifier in enumerate(
                    np.asarray(
                        self.substrateWholeCellModelIDs, dtype=object
                    ).ravel()
                )
            }
            self.substrateIndexs_modifiedNucleobases = np.asarray(
                [
                    local_by_id[str(identifier)]
                    for identifier, category in zip(ids, categories)
                    if category == "modified nucleobase"
                    and str(identifier) in local_by_id
                ],
                dtype=int,
            ).reshape(-1, 1)
            self.substrateIndexs_modifiedNucleosides = np.asarray(
                [
                    local_by_id[str(identifier)]
                    for identifier, category in zip(ids, categories)
                    if category == "modified deoxyribonucleoside"
                    and str(identifier) in local_by_id
                ],
                dtype=int,
            ).reshape(-1, 1)
            self.substrateIndexs_modifiedDNMPs = np.asarray(
                [
                    local_by_id[str(identifier)]
                    for identifier, category in zip(ids, categories)
                    if category == "modified deoxyribonucleotide monophosphate"
                    and str(identifier) in local_by_id
                ],
                dtype=int,
            ).reshape(-1, 1)
            # The source includes URA/URI/DUMP even if their KB category differs.
            for ids_field, special in (
                ("substrateIndexs_modifiedNucleobases", {"URA"}),
                ("substrateIndexs_modifiedNucleosides", {"URI"}),
                ("substrateIndexs_modifiedDNMPs", {"DUMP"}),
            ):
                existing = (
                    np.asarray(getattr(self, ids_field), dtype=int)
                    .ravel()
                    .tolist()
                )
                for identifier in special:
                    local = self.substrateIndexs([identifier]).item()
                    if local not in existing:
                        existing.append(local)
                setattr(
                    self,
                    ids_field,
                    np.asarray(existing, dtype=int).reshape(-1, 1),
                )

            self._initialize_reaction_partitions()
            return

        for k, v in kwargs.items():
            setattr(self, k, v)
        if hasattr(self, "reactionTypes"):
            types = [
                "".join(np.asarray(self._unwrap(x)).reshape(-1).astype(str))
                for x in self._vector(self.reactionTypes)
            ]
            for label, field in self._reaction_type_fields.items():
                setattr(
                    self,
                    field,
                    np.flatnonzero(np.asarray(types, dtype=object) == label)
                    + 1,
                )
        required = self.fixedConstantNames__
        missing = [k for k in required if not hasattr(self, k)]
        if missing:
            raise NotImplementedError(
                "DNADamage snapshot lacks initialized network fields: "
                + ", ".join(missing)
            )

    def _initialize_reaction_partitions(self):
        """Port DNADamage.m's DNA/radiation/small-molecule matrix split."""
        n_reactions = len(self.reactionWholeCellModelIDs)
        stoich = np.asarray(self.reactionStoichiometryMatrix).copy()
        dna_rows = np.concatenate(
            [
                np.asarray(getattr(self, name), dtype=int).ravel() - 1
                for name in (
                    "substrateIndexs_unmodifiedNucleobases",
                    "substrateIndexs_unmodifiedNucleosides",
                    "substrateIndexs_unmodifiedDNMPs",
                    "substrateIndexs_modifiedNucleobases",
                    "substrateIndexs_modifiedNucleosides",
                    "substrateIndexs_modifiedDNMPs",
                    "substrateIndexs_DR5P",
                    "substrateIndexs_dRibose5P_dRibose5P",
                )
            ]
        )
        self.reactionDNAStoichiometryMatrix = np.zeros_like(stoich)
        self.reactionDNAStoichiometryMatrix[dna_rows] = stoich[dna_rows]
        self.reactionRadiationStoichiometryMatrix = np.zeros_like(stoich)
        radiation_rows = (
            np.asarray(self.substrateIndexs_radiation, dtype=int).ravel() - 1
        )
        self.reactionRadiationStoichiometryMatrix[radiation_rows] = stoich[
            radiation_rows
        ]
        self.reactionSmallMoleculeStoichiometryMatrix = (
            stoich
            - self.reactionDNAStoichiometryMatrix
            - self.reactionRadiationStoichiometryMatrix
        )
        self.reactionDNAProduct = np.zeros(n_reactions, dtype=int)
        global_substrates = np.asarray(
            self.substrateGlobalIndexs, dtype=int
        ).ravel()
        for reaction in range(n_reactions):
            products = np.flatnonzero(
                self.reactionDNAStoichiometryMatrix[:, reaction] > 0
            )
            if products.size:
                self.reactionDNAProduct[reaction] = global_substrates[
                    products[-1]
                ]
        self.reactionRadiation = np.zeros(n_reactions, dtype=int)
        for reaction in range(n_reactions):
            radiation = np.flatnonzero(
                self.reactionRadiationStoichiometryMatrix[:, reaction] != 0
            )
            if radiation.size:
                self.reactionRadiation[reaction] = radiation[-1] + 1

        radical = self.reactionSmallMoleculeStoichiometryMatrix[
            int(np.asarray(self.substrateIndexs_hydroxylRadical).item()) - 1
        ].copy()
        water = int(np.asarray(self.substrateIndexs_water).item()) - 1
        hydrogen = int(np.asarray(self.substrateIndexs_hydrogen).item()) - 1
        self.reactionSmallMoleculeStoichiometryMatrix[water] += radical
        self.reactionSmallMoleculeStoichiometryMatrix[hydrogen] -= radical
        self.reactionSmallMoleculeStoichiometryMatrix[
            int(np.asarray(self.substrateIndexs_hydroxylRadical).item()) - 1
        ] = 0
        base_loss = (
            np.asarray(
                self.reactionIndexs_spontaneousBaseLoss, dtype=int
            ).ravel()
            - 1
        )
        nucleobases = (
            np.asarray(
                self.substrateIndexs_unmodifiedNucleobases, dtype=int
            ).ravel()
            - 1
        )
        self.reactionSmallMoleculeStoichiometryMatrix[
            np.ix_(nucleobases, base_loss)
        ] = stoich[np.ix_(nucleobases, base_loss)]
        self.reactionDNAStoichiometryMatrix[np.ix_(nucleobases, base_loss)] = 0

        damage_names = [
            "abasicSites",
            "damagedBases",
            "intrastrandCrossLinks",
            "strandBreaks",
        ]
        damage_types = {
            "spontaneous base loss": "abasicSites",
            "spontaneous base deamination": "damagedBases",
            "radiation (gamma-ray) induced base oxidation": "damagedBases",
            "UV-B photodimerization": "intrastrandCrossLinks",
            "strand break": "strandBreaks",
            "base reduction": "damagedBases",
            "base amination": "damagedBases",
            "base glucosyl transfer": "damagedBases",
            "base alkylation": "damagedBases",
            "base ethylation": "damagedBases",
            "base methylation": "damagedBases",
            "photooxidation": "damagedBases",
        }
        self.reactionDamageTypes = [
            damage_types.get(str(kind), "")
            for kind in np.asarray(self.reactionTypes, dtype=object).ravel()
        ]
        self.reactionVulnerableMotifs = [""] * n_reactions
        self.reactionVulnerableMotifTypes = [""] * n_reactions
        unmodified = np.concatenate(
            [
                np.asarray(getattr(self, name), dtype=int).ravel()
                for name in (
                    "substrateIndexs_unmodifiedNucleobases",
                    "substrateIndexs_unmodifiedNucleosides",
                    "substrateIndexs_unmodifiedDNMPs",
                )
            ]
        )
        base_alphabet = "ACGT"
        for reaction in range(n_reactions):
            reactants = np.flatnonzero(
                self.reactionDNAStoichiometryMatrix[:, reaction] < 0
            )
            if (
                reactants.size != 1
                or self.reactionDNAStoichiometryMatrix[reactants[0], reaction]
                != -1
            ):
                continue
            local = int(reactants[0] + 1)
            matched = np.flatnonzero(unmodified == local)
            if matched.size:
                self.reactionVulnerableMotifs[reaction] = base_alphabet[
                    matched[0] % 4
                ]
                self.reactionVulnerableMotifTypes[reaction] = "sequence"
            else:
                self.reactionVulnerableMotifs[reaction] = int(
                    global_substrates[reactants[0]]
                )
                self.reactionVulnerableMotifTypes[reaction] = (
                    self.reactionDamageTypes[reaction]
                )

        special_motifs = {
            "DNADamage_CSNCSN_cyclobutane_CSNCSN_UVB_radiation": "CC",
            "DNADamage_THYTHY_cyclobutane_THYTHY_UVB_radiation": "TT",
            "DNADamage_CSNTHY_cyclobutane_CSNTHY_UVB_radiation": "CT",
            "DNADamage_THYCSN_cyclobutane_THYCSN_UVB_radiation": "TC",
            "DNADamage_CSNTHY_CSN64THY_UVB_radiation": "CT",
            "DNADamage_THYCSN_THY64CSN_UVB_radiation": "TC",
            "DNADamage_THYTHY_THY64THY_UVB_radiation": "TT",
        }
        id_to_index = {
            str(identifier): i
            for i, identifier in enumerate(
                np.asarray(
                    self.reactionWholeCellModelIDs, dtype=object
                ).ravel()
            )
        }
        for identifier, motif in special_motifs.items():
            if identifier in id_to_index:
                i = id_to_index[identifier]
                self.reactionVulnerableMotifs[i] = motif
                self.reactionVulnerableMotifTypes[i] = "sequence"
        for i in (
            np.asarray(self.reactionIndexs_strandBreak, dtype=int).ravel() - 1
        ):
            self.reactionVulnerableMotifs[i] = True
            self.reactionVulnerableMotifTypes[i] = "abasicSites"
        self.reactionBounds[:, 0] = 0

    def calcResourceRequirements_LifeCycle(self, constants, states=None):
        rates = self.calcExpectedReactionRates() * (
            constants.states.Time.cellCycleLength
            + 0.5 * constants.states.Time.replicationDuration
            + constants.states.Time.cytokinesisDuration
        )
        return (
            np.maximum(0, -self.reactionSmallMoleculeStoichiometryMatrix)
            @ rates,
            np.maximum(0, self.reactionSmallMoleculeStoichiometryMatrix)
            @ rates,
            np.zeros(np.asarray(self.enzymeWholeCellModelIDs).size),
            np.full(np.asarray(self.enzymeWholeCellModelIDs).size, np.inf),
        )

    def initializeState(self):
        return None

    def calcResourceRequirements_Current(self):
        return np.maximum(
            0,
            -np.asarray(self.reactionSmallMoleculeStoichiometryMatrix)
            @ self.calcExpectedReactionRates(),
        )

    def calcExpectedReactionRates(self):
        rates = (
            self.calcNumberVulnerableSites()
            * np.asarray(self.reactionBounds)[:, 1]
        )
        rad = np.asarray(self.reactionRadiation).reshape(-1).astype(int)
        ix = np.flatnonzero(rad != 0)
        if ix.size:
            rates[ix] *= np.asarray(self.substrates).reshape(-1)[rad[ix] - 1]
        return rates

    @staticmethod
    def _find(matrix):
        if hasattr(matrix, "find"):
            p, v = matrix.find()
            return np.asarray(p), np.asarray(v).reshape(-1)
        a = np.asarray(matrix)
        if a.ndim == 1:
            a = a[:, None]
        p = np.argwhere(a != 0)
        return p, a[tuple(p.T)] if p.size else np.array([])

    @staticmethod
    def _unwrap(value):
        while (
            isinstance(value, np.ndarray)
            and value.dtype == object
            and value.size == 1
        ):
            value = value.reshape(-1)[0]
        return value

    @classmethod
    def _vector(cls, value):
        a = np.asarray(value)
        if a.dtype == object:
            return [cls._unwrap(x) for x in a.reshape(-1)]
        return list(a.reshape(-1))

    def calcNumberVulnerableSites(self):
        c = self.chromosome
        pr = _get(c, "polymerizedRegions")
        pol = float(
            pr.collapse(None)
            if hasattr(pr, "collapse")
            else np.asarray(pr).sum()
        )
        gc = float(np.asarray(_get(c, "sequenceGCContent")).reshape(-1)[0])
        comp = np.array([1 - gc, gc, gc, 1 - gc]) / 2
        positions, damage = self._find(_get(c, "damagedSites"))
        motifs = self._vector(self.reactionVulnerableMotifs)
        types = self._vector(self.reactionVulnerableMotifTypes)
        out = np.zeros(np.asarray(self.reactionStoichiometryMatrix).shape[1])
        for i, motif in enumerate(motifs):
            motif = self._unwrap(motif)
            if isinstance(motif, str) or (
                isinstance(motif, np.ndarray) and motif.dtype.kind in "US"
            ):
                motif = "".join(np.asarray(motif).reshape(-1).astype(str))
                out[i] = pol * np.prod(
                    comp[["ACGT".index(ch) for ch in motif]]
                )
                continue
            if not positions.size:
                continue
            typename = "".join(
                np.asarray(self._unwrap(types[i])).reshape(-1).astype(str)
            )
            sp, sv = self._find(_get(c, typename))
            if not sp.size:
                continue
            target = np.asarray(motif).reshape(-1)[0]
            a = positions[np.asarray(damage) == target]
            b = sp[np.asarray(sv) == target]
            if a.size and b.size:
                out[i] = sum(any(np.array_equal(x, y) for x in a) for y in b)
        return out

    def evolveState(self):
        order = (
            np.asarray(
                self.randStream.randperm(len(self.reactionWholeCellModelIDs))
            )
            .reshape(-1)
            .astype(int)
        )
        for j1 in order:
            j = int(j1) - 1
            sto = np.asarray(self.reactionSmallMoleculeStoichiometryMatrix)[
                :, j
            ]
            substrate = np.asarray(self.substrates).reshape(-1)
            needs = np.maximum(0, -sto)
            cap = np.min(
                np.divide(
                    substrate,
                    needs,
                    out=np.full_like(substrate, np.inf, dtype=float),
                    where=needs > 0,
                )
            )
            max_reactions = np.floor(cap)
            if max_reactions <= 0:
                continue
            rad = int(np.asarray(self.reactionRadiation).reshape(-1)[j])
            probability = (
                self.stepSizeSec
                * np.asarray(self.reactionBounds)[j, 1]
                * (substrate[rad - 1] if rad else 1)
            )
            if probability == 0:
                continue
            positions, side = self.chromosome.setSiteDamaged(
                self.reactionDamageTypes[j],
                self.reactionDNAProduct[j],
                probability,
                max_reactions,
                self.reactionVulnerableMotifs[j],
                self.reactionVulnerableMotifTypes[j],
            )
            if not len(positions):
                continue
            self.substrates = np.asarray(self.substrates) + len(
                positions
            ) * sto.reshape(np.asarray(self.substrates).shape)
            if len(side):
                self.simulationStateSideEffects = np.vstack(
                    (self.simulationStateSideEffects, side)
                )
