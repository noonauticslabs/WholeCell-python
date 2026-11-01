"""Transcription factor binding and activity from MATLAB TranscriptionalRegulation."""

from __future__ import annotations

import numpy as np

from ..process_base import Process, _get
from ..chromosome_process_aspect import ChromosomeProcessAspect


class TranscriptionalRegulation(Process, ChromosomeProcessAspect):
    """Regulate RNA polymerase promoter probabilities via bound TFs."""

    def __init__(
        self,
        wholeCellModelID="Process_TranscriptionalRegulation",
        name="Transcriptional Regulation",
        **properties,
    ):
        super().__init__(wholeCellModelID, name, **properties)
        self.chromosome = getattr(self, "chromosome", None)
        self.rnaPolymerase = getattr(self, "rnaPolymerase", None)

    def storeObjectReferences(self, simulation):
        Process.storeObjectReferences(self, simulation)
        ChromosomeProcessAspect.storeObjectReferences(self, simulation)
        statefn = getattr(simulation, "state", None)
        if callable(statefn):
            self.rnaPolymerase = statefn("RNAPolymerase")
        else:
            states = _get(simulation, "states")
            self.rnaPolymerase = {
                str(k).lower(): v for k, v in states.items()
            }["rnapolymerase"]
        self.states.append(self.rnaPolymerase)

    def initializeConstants(
        self, knowledgeBase=None, simulation=None, **kwargs
    ):
        for key, value in kwargs.items():
            setattr(self, key, value)
        if knowledgeBase is not None or simulation is not None:
            if knowledgeBase is None or simulation is None:
                raise ValueError(
                    "knowledgeBase and simulation must be supplied together"
                )
            tus = np.asarray(
                _get(knowledgeBase, "transcriptionUnits"), dtype=object
            ).ravel(order="F")
            monomers, complexes = [], []
            for tu in tus:
                monomers.extend(
                    np.asarray(
                        _get(tu, "transcriptionFactorProteinMonomers"),
                        dtype=object,
                    ).ravel(order="F")
                )
                complexes.extend(
                    np.asarray(
                        _get(tu, "transcriptionFactorProteinComplexs"),
                        dtype=object,
                    ).ravel(order="F")
                )
            self.enzymeWholeCellModelIDs = np.unique(
                [
                    *[_get(x, "wholeCellModelID") for x in monomers],
                    *[_get(x, "wholeCellModelID") for x in complexes],
                ]
            ).reshape(-1, 1)
            self.enzymeIndexs_fur = self.enzymeIndexs(["MG_236_MONOMER"])
            self.enzymeIndexs_gntR = self.enzymeIndexs(["MG_101_MONOMER"])
            self.enzymeIndexs_hrcA = self.enzymeIndexs(["MG_205_DIMER"])
            self.enzymeIndexs_luxR = self.enzymeIndexs(["MG_428_DIMER"])
            self.enzymeIndexs_spx = self.enzymeIndexs(["MG_127_MONOMER"])
            Process.initializeConstants(self, knowledgeBase, simulation)
            ChromosomeProcessAspect.initializeConstants(
                self, knowledgeBase, simulation
            )

            self.transcriptionUnitWholeCellModelIDs = np.asarray(
                [_get(tu, "wholeCellModelID") for tu in tus], dtype=object
            ).reshape(-1, 1)
            rows = []
            chromosome = self.chromosome
            sequence_len = int(
                np.asarray(_get(chromosome, "sequenceLen")).item()
            )
            monomer_global = np.asarray(
                self.enzymeMonomerGlobalIndexs, dtype=int
            ).reshape(-1)
            monomer_local = np.asarray(
                self.enzymeMonomerLocalIndexs, dtype=int
            ).reshape(-1)
            monomer_compartment = np.asarray(
                self.enzymeMonomerCompartmentIndexs, dtype=int
            ).reshape(-1)
            complex_global = np.asarray(
                self.enzymeComplexGlobalIndexs, dtype=int
            ).reshape(-1)
            complex_local = np.asarray(
                self.enzymeComplexLocalIndexs, dtype=int
            ).reshape(-1)
            complex_compartment = np.asarray(
                self.enzymeComplexCompartmentIndexs, dtype=int
            ).reshape(-1)

            def append_family(
                tu_index,
                proteins,
                compartments,
                activities,
                global_ids,
                local_ids,
                compartment_ids,
                footprints,
                start_name,
                length_name,
            ):
                proteins = np.asarray(proteins, dtype=object).ravel(order="F")
                compartments = np.asarray(compartments, dtype=object).ravel(
                    order="F"
                )
                activities = np.asarray(activities, dtype=float).ravel(
                    order="F"
                )
                if not len(proteins):
                    return
                protein_indices = np.asarray(
                    [int(_get(p, "idx")) for p in proteins], dtype=int
                )
                matches = {int(g): i for i, g in enumerate(global_ids)}
                selected = [
                    (i, matches[idx])
                    for i, idx in enumerate(protein_indices)
                    if idx in matches
                ]
                if not selected:
                    return
                start_coords = np.asarray(
                    _get(tus[tu_index], start_name), dtype=float
                ).ravel(order="F")
                site_lengths = np.asarray(
                    _get(tus[tu_index], length_name), dtype=float
                ).ravel(order="F")
                for original_index, mapped_index in selected:
                    footprint = footprints[protein_indices[original_index] - 1]
                    raw_coordinate = np.ceil(
                        start_coords[original_index]
                        + site_lengths[original_index] / 2
                        - footprint / 2
                    )
                    coordinate = (
                        (int(raw_coordinate) - 1) % sequence_len + 1
                        if np.isfinite(raw_coordinate)
                        else np.nan
                    )
                    rows.append(
                        (
                            tu_index + 1,
                            int(local_ids[mapped_index]),
                            int(compartment_ids[mapped_index])
                            == int(_get(compartments[original_index], "idx")),
                            activities[original_index],
                            coordinate,
                        )
                    )

            for i, tu in enumerate(tus):
                append_family(
                    i,
                    _get(tu, "transcriptionFactorProteinMonomers"),
                    _get(tu, "transcriptionFactorProteinMonomerCompartments"),
                    _get(tu, "transcriptionFactorProteinMonomerActivitys"),
                    monomer_global,
                    monomer_local,
                    monomer_compartment,
                    np.asarray(chromosome.monomerDNAFootprints).reshape(-1),
                    "transcriptionFactorProteinMonomerBindingSiteStartCoordinates",
                    "transcriptionFactorProteinMonomerBindingSiteLengths",
                )
                append_family(
                    i,
                    _get(tu, "transcriptionFactorProteinComplexs"),
                    _get(tu, "transcriptionFactorProteinComplexCompartments"),
                    _get(tu, "transcriptionFactorProteinComplexActivitys"),
                    complex_global,
                    complex_local,
                    complex_compartment,
                    np.asarray(chromosome.complexDNAFootprints).reshape(-1),
                    "transcriptionFactorProteinComplexBindingSiteStartCoordinates",
                    "transcriptionFactorProteinComplexBindingSiteLengths",
                )

            n = len(rows)
            tu_indices = np.asarray(
                [[row[0], row[0]] for row in rows], dtype=int
            ).reshape((-1, 2))
            tf_indices = np.asarray(
                [[row[1], row[1]] for row in rows], dtype=int
            ).reshape((-1, 2))
            affinities = np.asarray(
                [[row[3], row[3]] for row in rows], dtype=float
            ).reshape((-1, 2))
            activities = affinities.copy()
            activities[(activities < 1) & (affinities > 0)] **= -1
            positions = np.full((2 * n, 2), np.nan)
            for i, row in enumerate(rows):
                positions[i] = (row[4], 1)
                positions[n + i] = (row[4], 3)
            bound = np.isfinite(positions[:n, 0])
            self.otherActivities = np.ones(
                (len(tus), len(self.enzymeWholeCellModelIDs))
            )
            for i, row in enumerate(rows):
                if not bound[i]:
                    self.otherActivities[row[0] - 1, row[1] - 1] = activities[
                        i, 0
                    ]
            self.tuIndexs = tu_indices[bound]
            self.tfIndexs = tf_indices[bound]
            self.tfAffinities = affinities[bound]
            self.tfActivities = activities[bound]
            self.tfPositionStrands = positions[np.r_[bound, bound]]
        required = (
            "tfIndexs",
            "tuIndexs",
            "tfAffinities",
            "tfActivities",
            "tfPositionStrands",
            "otherActivities",
            "enzymeWholeCellModelIDs",
            "transcriptionUnitWholeCellModelIDs",
        )
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise ValueError(
                "TranscriptionalRegulation requires initialized source tables: "
                + ", ".join(missing)
            )
        self.tfIndexs = np.asarray(self.tfIndexs, dtype=int).reshape((-1, 2))
        self.tuIndexs = np.asarray(self.tuIndexs, dtype=int).reshape((-1, 2))
        self.tfAffinities = np.asarray(self.tfAffinities, dtype=float).reshape(
            (-1, 2)
        )
        self.tfActivities = np.asarray(self.tfActivities, dtype=float).reshape(
            (-1, 2)
        )
        self.tfPositionStrands = np.asarray(
            self.tfPositionStrands, dtype=int
        ).reshape((-1, 2))
        self.otherActivities = np.asarray(self.otherActivities, dtype=float)
        if np.any(self.tfActivities < 0):
            raise ValueError("tfActivities must be nonnegative")

    def calcResourceRequirements_LifeCycle(self, *args, **kwargs):
        n_substrates = np.asarray(self.substrateWholeCellModelIDs).size
        n_enzymes = np.asarray(self.enzymeWholeCellModelIDs).size
        return (
            np.zeros(n_substrates),
            np.zeros(n_substrates),
            np.zeros(n_enzymes),
            np.full(n_enzymes, np.inf),
        )

    def calcResourceRequirements_Current(self):
        return np.zeros_like(np.asarray(self.substrates), dtype=float)

    def initializeState(self):
        self.evolveState()

    @property
    def tfBoundPromoters(self):
        result = np.asarray(
            self.isDnaBound(
                self.tfPositionStrands, self.tfIndexs.reshape(-1, order="F")
            ),
            dtype=bool,
        )
        return result.reshape((-1, 2), order="F")

    @property
    def boundTFs(self):
        bound = self.tfBoundPromoters[:, 0]
        return np.bincount(
            self.tfIndexs[bound, 0],
            minlength=len(np.asarray(self.enzymes).reshape(-1)),
        )

    def evolveState(self):
        bound = self.bindTranscriptionFactors()
        self.rnaPolymerase.transcriptionFactorBindingProbFoldChange = (
            self.calcBindingProbabilityFoldChange(bound)
        )

    def bindTranscriptionFactors(self):
        bound = self.tfBoundPromoters.copy()
        polymerized = self.chromosome.isRegionPolymerized(
            self.tfPositionStrands, 1, False
        )
        # Chromosome's runtime query returns the four extent tuple from the
        # chromosome aspect; its first component is the source boolean mask.
        if isinstance(polymerized, tuple):
            polymerized = polymerized[0]
        accessible = (~bound) & np.asarray(polymerized, dtype=bool).reshape(
            (-1, 2), order="F"
        )
        for i, count in enumerate(
            np.asarray(self.enzymes).reshape(-1), start=1
        ):
            if count <= 0:
                continue
            sites = np.flatnonzero(
                ((self.tfIndexs == i) & accessible).reshape(-1, order="F")
            )
            if not len(sites):
                continue
            tf, *_ = self.bindProteinToChromosome(
                self.tfPositionStrands[sites],
                i,
                count,
                self.tfAffinities.reshape(-1, order="F")[sites],
                None,
                False,
                1,
                False,
                None,
            )
            flat_bound = bound.ravel(order="F")
            flat_bound[sites[np.asarray(tf, dtype=bool)]] = True
            bound = flat_bound.reshape(bound.shape, order="F")
        return bound

    def calcBindingProbabilityFoldChange(self, tfBoundPromoters):
        fold = np.ones(
            (
                np.asarray(
                    self.chromosome.transcriptionUnitStartCoordinates
                ).shape[0],
                2,
            )
        )
        bound = np.asarray(tfBoundPromoters, dtype=bool).reshape((-1, 2))
        for i, j in zip(*np.nonzero(bound)):
            tu = int(self.tuIndexs[i, j])
            fold[tu - 1, j] *= self.tfActivities[i, j]
        active = np.asarray(self.enzymes).reshape(-1) > 0
        other = np.prod(self.otherActivities ** active[np.newaxis, :], axis=1)
        return fold * other[:, np.newaxis]


TranscriptionalRegulationProcess = TranscriptionalRegulation
