"""SMC binding and ATP hydrolysis during chromosome condensation."""

from __future__ import annotations

import numpy as np

from ..process_base import Process, _get
from ..chromosome_process_aspect import ChromosomeProcessAspect


class ChromosomeCondensation(Process, ChromosomeProcessAspect):
    """Port of MATLAB ``ChromosomeCondensation``.

    MATLAB indices exposed by the process remain one based. Stochastic binding
    is delegated to the chromosome process aspect, as in the MATLAB source.
    """

    optionNames__ = ()
    fixedConstantNames__ = ("smcSepNt", "smcSepProbCenter")
    fittedConstantNames__ = ()
    localStateNames__ = ()
    stimuliWholeCellModelIDs = ()
    substrateWholeCellModelIDs = ("ATP", "ADP", "PI", "H2O", "H")
    enzymeWholeCellModelIDs = (
        "MG_213_214_298_6MER",
        "MG_213_214_298_6MER_ADP",
    )
    substrateIndexs_atp = 1
    substrateIndexs_adp = 2
    substrateIndexs_phosphate = 3
    substrateIndexs_water = 4
    substrateIndexs_hydrogen = 5
    enzymeIndexs_SMC = 1
    enzymeIndexs_SMC_ADP = 2

    def __init__(
        self,
        wholeCellModelID="Process_ChromosomeCondensation",
        name="Chromosome Condensation",
        **properties,
    ):
        super().__init__(wholeCellModelID, name, **properties)
        self.chromosome = getattr(self, "chromosome", None)

    def storeObjectReferences(self, simulation):
        super().storeObjectReferences(simulation)
        ChromosomeProcessAspect.storeObjectReferences(self, simulation)

    def initializeConstants(
        self, knowledgeBase=None, simulation=None, **kwargs
    ):
        options = kwargs.pop("options", None)
        if knowledgeBase is not None or simulation is not None:
            if knowledgeBase is None or simulation is None:
                raise ValueError(
                    "knowledgeBase and simulation must be provided together"
                )
            Process.initializeConstants(
                self, knowledgeBase, simulation, options
            )
            ChromosomeProcessAspect.initializeConstants(
                self, knowledgeBase, simulation
            )
        for key, value in kwargs.items():
            setattr(self, key, value)
        required = ("smcSepNt", "smcSepProbCenter", "enzymeDNAFootprints")
        missing = [key for key in required if not hasattr(self, key)]
        if missing:
            raise NotImplementedError(
                "ChromosomeCondensation snapshot lacks initialized fields: "
                + ", ".join(missing)
            )

    def calcResourceRequirements_Current(self):
        result = np.zeros_like(np.asarray(self.substrates), dtype=float)
        n_smc = np.asarray(self.enzymes)[int(self.enzymeIndexs_SMC) - 1].item()
        result[int(self.substrateIndexs_atp) - 1] = n_smc
        result[int(self.substrateIndexs_water) - 1] = n_smc
        return result

    def calcResourceRequirements_LifeCycle(self, constants, states):
        """Return MATLAB biomass, byproduct and enzyme-expression vectors."""
        n_substrates = len(self.substrateWholeCellModelIDs)
        n_enzymes = len(self.enzymeWholeCellModelIDs)
        biomass = np.zeros(n_substrates)
        byproducts = np.zeros(n_substrates)
        min_expression = np.zeros(n_enzymes)
        max_expression = np.full(n_enzymes, np.inf)
        smc_adp_global = int(
            self.enzymeComplexGlobalIndexs[int(self.enzymeIndexs_SMC_ADP) - 1]
        )
        n_smcs = np.sum(
            np.asarray(_get(states, "complexProductions"))[smc_adp_global - 1]
        )
        n_atp = (
            0.2
            * n_smcs
            * self.enzymeDNAFootprints[int(self.enzymeIndexs_SMC_ADP) - 1]
            / self.chromosome.sequenceLen
            * np.sum(
                np.asarray(_get(states, "complexs0"))[
                    np.asarray(
                        self.complex.rnaPolymeraseIndexs, dtype=int
                    ).ravel()
                    - 1
                ]
            )
            * _get(_get(constants, "states"), "Time").cellCycleLength
            / np.log(2)
        )
        biomass[int(self.substrateIndexs_atp) - 1] = n_atp
        biomass[int(self.substrateIndexs_water) - 1] = n_atp
        byproducts[int(self.substrateIndexs_adp) - 1] = n_atp - n_smcs
        byproducts[int(self.substrateIndexs_phosphate) - 1] = n_atp
        byproducts[int(self.substrateIndexs_hydrogen) - 1] = n_atp
        min_expression[int(self.enzymeIndexs_SMC) - 1] = np.ceil(
            self.chromosome.sequenceLen / self.smcSepNt
        )
        return biomass, byproducts, min_expression, max_expression

    def initializeState(self):
        saved = np.asarray(self.substrates).copy()
        self.substrates[int(self.substrateIndexs_atp) - 1] = np.inf
        self.substrates[int(self.substrateIndexs_water) - 1] = np.inf
        try:
            for _ in range(20):
                self.evolveState()
        finally:
            self.substrates = saved

    def evolveState(self):
        # Dissociate free SMC-ADP complexes before choosing new binding events.
        n_free_adp = np.asarray(self.enzymes)[
            int(self.enzymeIndexs_SMC_ADP) - 1
        ].item()
        self.enzymes[int(self.enzymeIndexs_SMC) - 1] += n_free_adp
        self.substrates[int(self.substrateIndexs_adp) - 1] += n_free_adp
        self.enzymes[int(self.enzymeIndexs_SMC_ADP) - 1] = 0

        n_binding_max = min(
            np.asarray(self.substrates)[
                int(self.substrateIndexs_atp) - 1
            ].item(),
            np.asarray(self.substrates)[
                int(self.substrateIndexs_water) - 1
            ].item(),
            np.asarray(self.enzymes)[int(self.enzymeIndexs_SMC) - 1].item(),
        )
        if n_binding_max == 0:
            return

        c = self.chromosome
        sep = np.asarray(self.smcSepNt).item()
        center = np.asarray(self.smcSepProbCenter).item()
        sequence_len = np.asarray(c.sequenceLen).item()
        smc_footprint = np.asarray(self.enzymeDNAFootprints).reshape(-1)[
            int(self.enzymeIndexs_SMC_ADP) - 1
        ]
        pos_strands, lengths = c.polymerizedRegions.find()
        bound_sites, bound_values = c.complexBoundSites.find()
        smc_global = int(
            np.asarray(self.enzymeComplexGlobalIndexs).reshape(-1)[
                int(self.enzymeIndexs_SMC_ADP) - 1
            ]
        )
        smc_sites = bound_sites[
            np.asarray(bound_values).reshape(-1) == smc_global
        ]
        if smc_sites.size:
            centers = np.column_stack(
                (
                    (
                        smc_sites[:, 0]
                        - sep / 2
                        - center / 2
                        + smc_footprint / 2
                        - 1
                    )
                    % sequence_len
                    + 1,
                    2 * np.ceil(smc_sites[:, 1] / 2) - 1,
                )
            )
            both_strands = centers.copy()
            both_strands[:, 1] += 1
            exclusion_sites = np.vstack((centers, both_strands))
            # Snapshot scalars are 1x1 arrays; region algebra expects a scalar.
            original_sequence_len = c.sequenceLen
            try:
                c.sequenceLen = sequence_len
                pos_strands, lengths = c.excludeRegions(
                    pos_strands,
                    lengths,
                    exclusion_sites,
                    np.full(len(exclusion_sites), sep + center),
                )
            finally:
                c.sequenceLen = original_sequence_len
        if not len(pos_strands):
            return

        n_bound, _ = self.bindProteinToChromosomeStochastically(
            self.enzymeIndexs_SMC_ADP,
            n_binding_max,
            pos_strands,
            lengths,
            None,
            None,
            self.calcNewRegions,
        )
        if n_bound == 0:
            return
        self.enzymes[int(self.enzymeIndexs_SMC) - 1] -= n_bound
        self.enzymes[int(self.enzymeIndexs_SMC_ADP) - 1] += n_bound
        self.substrates[int(self.substrateIndexs_atp) - 1] -= n_bound
        self.substrates[int(self.substrateIndexs_water) - 1] -= n_bound
        self.substrates[int(self.substrateIndexs_hydrogen) - 1] += n_bound
        self.substrates[int(self.substrateIndexs_phosphate) - 1] += n_bound

    def calcNewRegions(self, rgnPosStrnds, rgnLens, rgnProbs, rgnIdx, offset):
        c = self.chromosome
        sep = np.asarray(self.smcSepNt).item()
        center = np.asarray(self.smcSepProbCenter).item()
        sequence_len = np.asarray(c.sequenceLen).item()
        footprint = np.asarray(self.enzymeDNAFootprints).reshape(-1)[
            int(self.enzymeIndexs_SMC_ADP) - 1
        ]
        pos, lengths = c.excludeRegions(
            rgnPosStrnds,
            rgnLens,
            [
                [
                    (
                        int(rgnPosStrnds[rgnIdx, 0])
                        + offset
                        - sep / 2
                        - center / 2
                        + footprint / 2
                        - 1
                    )
                    % sequence_len
                    + 1,
                    rgnPosStrnds[rgnIdx, 1],
                ]
            ],
            [sep + center],
        )
        probs = np.maximum(0, lengths - footprint + 1)
        return pos, lengths, probs
