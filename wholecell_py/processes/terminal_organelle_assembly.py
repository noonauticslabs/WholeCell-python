"""Runtime port of MATLAB ``TerminalOrganelleAssembly``."""

from __future__ import annotations

import numpy as np

from ..process_base import _get
from ..reaction_process import ReactionProcess


class TerminalOrganelleAssembly(ReactionProcess):
    """Localize terminal organelle proteins using the KB reaction dependency graph.

    MATLAB local/global indices remain one-based on the instance. Conversion to
    Python indices happens only when applying those indices to NumPy arrays.
    """

    optionNames__ = []
    fixedConstantNames__ = [
        "localizationReactions",
        "localizationSubstrates",
        "localizationThreshold",
    ]
    fittedConstantNames__ = []
    localStateNames__ = []
    substrateWholeCellModelIDs__ = (
        "MG_191_MONOMER",
        "MG_192_MONOMER",
        "MG_217_MONOMER",
        "MG_218_MONOMER",
        "MG_312_MONOMER",
        "MG_317_MONOMER",
        "MG_318_MONOMER",
        "MG_386_MONOMER",
    )

    def __init__(
        self,
        wholeCellModelID="Process_TerminalOrganelleAssembly",
        name="Terminal Organelle Assembly",
        **properties,
    ):
        super().__init__(wholeCellModelID, name, **properties)
        self.compartmentIndexs_incorporated = 1
        self.compartmentIndexs_unincorporated = 2
        for name, idx in zip(
            ("MgPa", "P110", "P65", "HMW2", "HMW1", "HMW3", "P32", "P200"),
            range(1, 9),
        ):
            setattr(self, f"substrateIndexs_{name}", idx)

    def initializeConstants(
        self, knowledgeBase=None, simulation=None, options=None, **kwargs
    ):
        if knowledgeBase is None or simulation is None:
            raise NotImplementedError(
                "TerminalOrganelleAssembly.initializeConstants requires knowledgeBase and simulation"
            )
        user_options = options or {}
        options = {
            "retainSubstrateCompartments": True,
            "retainEnzymeCompartments": True,
            "retainModificationCompartments": True,
        }
        options.update(user_options)
        super().initializeConstants(knowledgeBase, simulation, options)
        for name, identifier in {
            "HMW1": "MG_312_MONOMER",
            "HMW2": "MG_218_MONOMER",
            "HMW3": "MG_317_MONOMER",
            "P32": "MG_318_MONOMER",
        }.items():
            setattr(
                self, f"enzymeIndexs_{name}", self.enzymeIndexs([identifier])
            )
        ids = np.asarray(
            sorted(self.substrateWholeCellModelIDs__), dtype=object
        ).reshape(-1, 1)
        self.substrateWholeCellModelIDs = ids
        self._initialize_component_mapping("substrate", ids, False)
        n = len(ids)
        for category in (
            "Stimulus",
            "Metabolite",
            "RNA",
            "Monomer",
            "Complex",
        ):
            global_attr = f"substrate{category}GlobalIndexs"
            comp_attr = f"substrate{category}CompartmentIndexs"
            setattr(
                self,
                global_attr,
                np.repeat(np.asarray(getattr(self, global_attr)), 2, axis=1),
            )
            setattr(
                self,
                comp_attr,
                np.repeat(np.asarray(getattr(self, comp_attr)), 2, axis=1),
            )
        self.compartmentIndexs_incorporated = 1
        self.compartmentIndexs_unincorporated = 2
        monomer_comp = np.asarray(self.substrateMonomerCompartmentIndexs)
        complex_comp = np.asarray(self.substrateComplexCompartmentIndexs)
        toc = int(
            np.asarray(
                _get(self.compartment, "terminalOrganelleCytosolIndexs")
            ).item()
        )
        tom = int(
            np.asarray(
                _get(self.compartment, "terminalOrganelleMembraneIndexs")
            ).item()
        )
        cyt = int(np.asarray(_get(self.compartment, "cytosolIndexs")).item())
        mem = int(np.asarray(_get(self.compartment, "membraneIndexs")).item())
        monomer_comp[monomer_comp[:, 1] == toc, 1] = cyt
        monomer_comp[monomer_comp[:, 1] == tom, 1] = mem
        complex_comp[complex_comp[:, 1] == toc, 1] = cyt
        complex_comp[complex_comp[:, 1] == tom, 1] = mem

        gene_indices = np.asarray(
            _get(self.gene, "mRNAIndexs"), dtype=int
        ).ravel(order="F")
        substrate_genes = np.zeros(n, dtype=int)
        monomer_local = (
            np.asarray(self.substrateMonomerLocalIndexs, dtype=int).ravel() - 1
        )
        monomer_global = (
            np.asarray(self.substrateMonomerGlobalIndexs, dtype=int).ravel()
            - 1
        )
        complex_local = (
            np.asarray(self.substrateComplexLocalIndexs, dtype=int).ravel() - 1
        )
        complex_global = (
            np.asarray(self.substrateComplexGlobalIndexs, dtype=int).ravel()
            - 1
        )
        substrate_genes[monomer_local] = gene_indices[monomer_global]
        substrate_genes[complex_local] = (
            complex_global + int(_get(knowledgeBase, "numGenes")) + 1
        )
        modification = np.asarray(self.reactionModificationMatrix)
        self.localizationSubstrates = np.sum(
            np.isnan(modification[:, substrate_genes, :]), axis=2
        )

        catalysis = np.asarray(self.reactionCatalysisMatrix)
        self.localizationReactions = np.zeros(
            (catalysis.shape[0], n, 2), dtype=float
        )
        enz_mono_local = np.asarray(
            self.enzymeMonomerLocalIndexs, dtype=int
        ).ravel()
        enz_complex_local = np.asarray(
            self.enzymeComplexLocalIndexs, dtype=int
        ).ravel()
        enz_mono_global = np.asarray(
            self.enzymeMonomerGlobalIndexs, dtype=int
        ).ravel()
        enz_complex_global = np.asarray(
            self.enzymeComplexGlobalIndexs, dtype=int
        ).ravel()
        enz_global = np.asarray(self.enzymeGlobalIndexs, dtype=int).ravel()
        for reaction in range(catalysis.shape[0]):
            enzyme_rows, compartment_columns = np.where(
                catalysis[reaction] != 0
            )
            for enzyme_row, compartment_column in zip(
                enzyme_rows, compartment_columns
            ):
                enzyme_local = enzyme_row + 1
                if enzyme_local in enz_mono_local:
                    enzyme_global = enz_global[enzyme_row]
                    matching = monomer_local[
                        monomer_global == enzyme_global - 1
                    ]
                    local = int(matching[0]) if matching.size else None
                elif enzyme_local in enz_complex_local:
                    enzyme_global = enz_global[enzyme_row]
                    matching = complex_local[
                        complex_global == enzyme_global - 1
                    ]
                    local = int(matching[0]) if matching.size else None
                else:
                    continue
                if local is None:
                    continue
                compartments = (
                    monomer_comp[local]
                    if enzyme_local in enz_mono_local
                    else complex_comp[local]
                )
                compartment = compartment_column + 1
                positions = np.flatnonzero(compartments == compartment)
                if positions.size:
                    self.localizationReactions[
                        reaction, local, positions[0]
                    ] = 1
        self.localizationThreshold = np.sum(
            self.localizationReactions, axis=(1, 2)
        ).reshape(-1, 1)

    def initializeState(self, *args, **kwargs):
        """Protein state is initialized by the simulation's state initializer."""

    def calcResourceRequirements_LifeCycle(self, *args, **kwargs):
        n_sub = np.asarray(self.substrates).shape[0]
        n_enz = len(np.asarray(self.enzymeWholeCellModelIDs).reshape(-1))
        biomass = np.zeros((n_sub, 1))
        byproducts = np.zeros_like(biomass)
        minimum = np.zeros((n_enz, 1))
        maximum = np.full((n_enz, 1), np.inf)
        for attr in (
            "enzymeIndexs_HMW1",
            "enzymeIndexs_HMW2",
            "enzymeIndexs_HMW3",
            "enzymeIndexs_P32",
        ):
            minimum[
                np.asarray(getattr(self, attr), dtype=int).reshape(-1) - 1
            ] = 1
        return biomass, byproducts, minimum, maximum

    def calcResourceRequirements_Current(self):
        return np.zeros_like(np.asarray(self.substrates), dtype=float)

    def evolveState(self):
        """Iteratively move every currently permitted protein to the organelle."""
        substrates = np.asarray(self.substrates)
        incorporated = int(self.compartmentIndexs_incorporated) - 1
        unincorporated = int(self.compartmentIndexs_unincorporated) - 1
        reactions = np.asarray(self.localizationReactions)
        requirements = np.asarray(self.localizationSubstrates)
        thresholds = np.asarray(self.localizationThreshold).reshape(-1)
        if reactions.ndim != 3 or reactions.shape[2] != 2:
            raise ValueError(
                "localizationReactions must have shape (reactions, proteins, 2)"
            )
        while np.any(substrates[:, unincorporated]):
            active = (
                (reactions[:, :, 0] @ (substrates[:, 0] > 0))
                + (reactions[:, :, 1] @ (substrates[:, 1] > 0))
            ) >= thresholds
            localizing = (
                substrates[:, unincorporated] * (requirements.T @ active)
            ).astype(bool)
            if not np.any(localizing):
                break
            substrates[localizing, incorporated] += substrates[
                localizing, unincorporated
            ]
            substrates[localizing, unincorporated] = 0
        self.substrates = substrates

    def getDryWeight(self):
        weights = np.asarray(self.substrateMolecularWeights).reshape(
            -1, order="F"
        )
        counts = np.asarray(self.substrates)
        if counts.ndim == 2:
            total = weights @ counts.sum(axis=1)
        else:
            total = np.tensordot(weights, counts.sum(axis=1), axes=(0, 0))
        return np.asarray(total) / 6.022141793e23
