"""Replication process coordinator and resource accounting from original model."""

import numpy as np
from ..process_base import Process
from ..chromosome_process_aspect import ChromosomeProcessAspect
from .replication_views import ReplicationViews
from .replication_updates import ReplicationUpdates


def scalar(value):
    return np.asarray(value).item()


class Replication(
    ReplicationViews, ReplicationUpdates, ChromosomeProcessAspect, Process
):
    leadingStrandIndexs = np.array([1, 4])
    laggingStrandIndexs = np.array([3, 2])
    substrateIndexs_dntp = np.arange(1, 5)
    substrateIndexs_ntp = np.arange(5, 9)
    substrateIndexs_diphosphate = 9
    substrateIndexs_water = 10
    substrateIndexs_hydrogen = 11
    substrateIndexs_nad = 12
    substrateIndexs_nmn = 13
    substrateIndexs_atp = 5
    substrateIndexs_adp = 14
    substrateIndexs_amp = 15
    substrateIndexs_phosphate = 16
    enzymeIndexs_replisome = 1
    enzymeIndexs_2coreBetaClampGammaComplexPrimase = 2
    enzymeIndexs_coreBetaClampGammaComplex = 3
    enzymeIndexs_coreBetaClampPrimase = 4
    enzymeIndexs_core = 5
    enzymeIndexs_gammaComplex = 6
    enzymeIndexs_betaClampMonomer = 7
    enzymeIndexs_betaClamp = 8
    enzymeIndexs_helicase = 9
    enzymeIndexs_ligase = 10
    enzymeIndexs_primase = 11
    enzymeIndexs_ssb4mer = 12
    enzymeIndexs_ssb8mer = 13

    def __init__(
        self,
        wholeCellModelID="Process_Replication",
        name="Replication",
        **fields,
    ):
        Process.__init__(self, wholeCellModelID, name, **fields)

    def restoreSnapshotRuntime(self):
        self.primaseBindingLocations = [
            np.asarray(v).ravel(order="F")
            for v in np.asarray(
                self.primaseBindingLocations, dtype=object
            ).ravel(order="F")
        ]
        for name in (
            "oriCPosition",
            "terCPosition",
            "primerLength",
            "dnaPolymeraseElongationRate",
            "ssbComplexSpacing",
            "okazakiFragmentMeanLength",
            "ligaseRate",
            "laggingBackupClampReloadingLength",
            "startingOkazakiLoopLength",
            "rnaPolymeraseCollisionMeanDwellTime",
            "ssbDissociationRate",
        ):
            setattr(self, name, np.asarray(getattr(self, name)).item())
        # MATLAB stores index constants as doubles; Python requires integers.
        # Restrict conversion to explicit index names, preserving all biology.
        for key, value in list(vars(self).items()):
            if "Indexs" not in key and not key.endswith("Index"):
                continue
            a = np.asarray(value)
            if (
                a.dtype.kind not in "fiu"
                or not np.all(np.isfinite(a))
                or not np.all(a == np.floor(a))
            ):
                continue
            local = key.startswith(
                (
                    "enzymeIndexs_",
                    "substrateIndexs_",
                    "dnaAFunctionalBoxIndexs_",
                    "complexIndexs_",
                    "leadingStrandIndexs",
                    "laggingStrandIndexs",
                )
            )
            setattr(
                self,
                key,
                (
                    (
                        int(a.item())
                        if a.size == 1
                        else a.astype(int).ravel(order="F")
                    )
                    if local
                    else a.astype(int)
                ),
            )

    def storeObjectReferences(self, simulation):
        Process.storeObjectReferences(self, simulation)
        ChromosomeProcessAspect.storeObjectReferences(self, simulation)

    def initializeState(self):
        # Original method is empty; initial replication is driven by DnaA.
        return None

    def calcResourceRequirements_Current(self):
        result = np.zeros_like(self.substrates)
        flat = result.reshape(-1)
        enzymes = np.asarray(self.enzymes).reshape(-1)

        def index(name):
            return int(scalar(getattr(self, name))) - 1

        atp, water = index("substrateIndexs_atp"), index(
            "substrateIndexs_water"
        )
        if (
            not self.isAnyPolymeraseBound
            and self.isDnaAORIComplexAssembled
            and np.all(
                enzymes
                >= 2
                * np.asarray(self.enzymeComposition)[
                    :, index("enzymeIndexs_2coreBetaClampGammaComplexPrimase")
                ]
            )
        ):
            unwind = (
                np.asarray(self.enzymeDNAFootprints3Prime).ravel()[
                    index("enzymeIndexs_helicase")
                ]
                + np.asarray(self.enzymeDNAFootprints5Prime).ravel()[
                    index("enzymeIndexs_core")
                ]
                + 1
            )
            flat[[atp, water]] += 2 * (1 + unwind)
        if self.isAnyHelicaseBound and np.all(self.leadingStrandElongating):
            lagging = np.count_nonzero(self.laggingStrandElongating)
            assembled = bool(self.isDnaAORIComplexAssembled)
            rate = scalar(self.dnaPolymeraseElongationRate)
            flat[[atp, water]] += min(2, 2 * assembled + 2) * rate
            flat[[atp, water]] += min(
                2, 2 * assembled + lagging
            ) - np.count_nonzero(self.laggingBackupBetaClampPosition)
            gc = scalar(self.chromosome.sequenceGCContent)
            composition = np.array([1 - gc, gc, gc, 1 - gc]) / 2
            flat[np.asarray(self.substrateIndexs_dntp, int).ravel() - 1] += (
                min(4, 2 * assembled + 2 + lagging) * rate * composition
            )
        flat[index("substrateIndexs_nad")] += min(
            len(self.chromosome.strandBreaks.find()[1]),
            np.ceil(
                enzymes[index("enzymeIndexs_ligase")]
                * scalar(self.stepSizeSec)
                * scalar(self.ligaseRate)
            ),
        )
        return result

    def evolveState(self):
        indices = [
            self.enzymeIndexs_2coreBetaClampGammaComplexPrimase,
            self.enzymeIndexs_coreBetaClampGammaComplex,
            self.enzymeIndexs_coreBetaClampPrimase,
            self.enzymeIndexs_helicase,
        ]
        counts = np.asarray(self.boundEnzymes).ravel()[
            np.asarray(indices, int).ravel() - 1
        ]
        total = np.dot([2, 1, 1], counts[:3])
        if not (
            (total == 0 or total == 4)
            and ((counts[3] == 0 and total == 0) or counts[3] == 2)
        ):
            raise RuntimeError(
                f"Replisome state out of sync: {counts.tolist()}"
            )
        actions = [
            self.initiateReplication,
            self.ligateDNA,
            self.dissociateFreeSSBComplexes,
            self.freeAndBindSSBs,
        ]
        if self.isAnyHelicaseBound and np.all(self.leadingStrandElongating):
            actions += [
                self.unwindAndPolymerizeDNA,
                self.initiateOkazakiFragment,
                self.terminateOkazakiFragment,
                self.terminateReplication,
            ]
        for index in (
            np.asarray(self.randStream.randperm(len(actions)), int).ravel() - 1
        ):
            actions[index]()
