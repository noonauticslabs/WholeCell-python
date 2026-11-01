"""Runtime port of the WholeCell Transcription process.

All exposed indices and chromosome coordinates follow MATLAB's one-based
convention. Conversion to Python indexing is local to array accesses.
Knowledge-base initialization is deliberately explicit: it needs KB features
and fitted constants that are not present in an initialized runtime snapshot.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from ..process_base import Process, _get, _set
from ..chromosome_process_aspect import ChromosomeProcessAspect


def _transcription_sequence_view(sequence):
    """Return a sequence-like view, preserving MATLAB column-major chars.

    Snapshot character matrices use one-character Unicode elements. Keep those
    arrays flat in Fortran order so callers can slice only the active segment
    before converting it to Python characters. Scalar string exports are
    already contiguous strings. Less common multi-character array elements
    retain the previous join behavior, where progress counts Python chars.
    """
    if isinstance(sequence, str):
        return sequence
    array = np.asarray(sequence)
    if array.ndim == 0:
        scalar = array.item()
        if isinstance(scalar, str):
            return scalar
    flat = array.ravel(order="F")
    if (
        array.dtype.kind == "U"
        and array.dtype.itemsize == np.dtype("U1").itemsize
    ):
        return flat
    return "".join(flat)


def _active_transcription_sequence_matrix(
    sequences, tu_indices, polymerase_states, active_indices, elongation_rate
):
    """Build only the active, rate-bounded sequence matrix for polymerases."""
    sequence_values = np.asarray(sequences, dtype=object).reshape(-1)
    sequence_views = [
        _transcription_sequence_view(sequence_values[tu - 1])
        for tu in tu_indices
    ]
    starts = []
    remaining = []
    for sequence, polymerase in zip(sequence_views, active_indices):
        start, stop, _ = slice(
            int(polymerase_states[polymerase]) - 1, None
        ).indices(len(sequence))
        starts.append(start)
        remaining.append(stop - start)
    width = min(int(elongation_rate), max(remaining, default=0))
    active_sequences = np.full((len(sequence_views), width), " ", dtype="<U1")
    for row, (sequence, start, available) in enumerate(
        zip(sequence_views, starts, remaining)
    ):
        count = min(width, available)
        if count:
            segment = sequence[start : start + count]
            active_sequences[row, :count] = (
                list(segment) if isinstance(segment, str) else segment
            )
    return active_sequences


def _specific_binding_coordinates(five_prime_coordinates, directions):
    """Return the RNA polymerase specific-binding coordinate for each TU.

    MATLAB directions are encoded as 0/1 and are converted to -1/+1 before
    subtracting from the TU 5-prime coordinate.
    """
    five_prime = np.asarray(five_prime_coordinates).reshape(-1)
    direction = np.asarray(directions).reshape(-1)
    return five_prime - (2 * direction - 1)


class Transcription(ChromosomeProcessAspect, Process):
    optionNames__ = ()
    fixedConstantNames__ = (
        "rnaPolymeraseElongationRate",
        "transcriptionUnitBaseCounts",
        "transcriptionUnitBindingProbabilities",
    )
    fittedConstantNames__ = ("transcriptionUnitBindingProbabilities",)
    localStateNames__ = ("RNAs",)
    stimuliWholeCellModelIDs = ()
    substrateWholeCellModelIDs = (
        "ATP",
        "CTP",
        "GTP",
        "UTP",
        "AMP",
        "CMP",
        "GMP",
        "UMP",
        "ADP",
        "PPI",
        "H2O",
        "H",
    )
    substrateIndexs_ntp = np.arange(1, 5)
    substrateIndexs_nmp = np.arange(5, 9)
    substrateIndexs_atp, substrateIndexs_adp = 1, 9
    (
        substrateIndexs_diphosphate,
        substrateIndexs_water,
        substrateIndexs_hydrogen,
    ) = (
        10,
        11,
        12,
    )
    enzymeWholeCellModelIDs = (
        "MG_249_MONOMER",
        "MG_282_MONOMER",
        "MG_141_MONOMER",
        "MG_027_MONOMER",
        "RNA_POLYMERASE",
        "RNA_POLYMERASE_HOLOENZYME",
    )
    enzymeIndexs_transcriptionFactors = np.arange(1, 5)
    enzymeIndexs_sigmaFactor, enzymeIndexs_elongationFactor = 1, 2
    enzymeIndexs_terminationFactor, enzymeIndexs_antiterminationFactor = 3, 4
    enzymeIndexs_rnaPolymerase, enzymeIndexs_rnaPolymeraseHoloenzyme = 5, 6

    def __init__(
        self,
        wholeCellModelID="Process_Transcription",
        name="Transcription",
        **properties,
    ):
        super().__init__(wholeCellModelID, name, **properties)
        self.rnaPolymerases = getattr(self, "rnaPolymerases", None)
        self.transcripts = getattr(self, "transcripts", None)

    def restoreSnapshotRuntime(self):
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
        state = getattr(simulation, "state", None)
        if callable(state):
            self.rnaPolymerases, self.transcripts = state(
                "RNAPolymerase"
            ), state("Transcript")
        else:
            states = _get(simulation, "states")
            lookup = {str(k).lower(): v for k, v in states.items()}
            self.rnaPolymerases, self.transcripts = (
                lookup["rnapolymerase"],
                lookup["transcript"],
            )
        self.states = [
            *(self.states or []),
            self.rnaPolymerases,
            self.transcripts,
        ]

    def initializeConstants(
        self, knowledgeBase=None, simulation=None, options=None, **kwargs
    ):
        for key, value in kwargs.items():
            setattr(self, key, value)
        if knowledgeBase is not None or simulation is not None:
            if knowledgeBase is None or simulation is None:
                raise ValueError(
                    "knowledgeBase and simulation must be supplied together"
                )
            Process.initializeConstants(
                self, knowledgeBase, simulation, options or {}
            )
            ChromosomeProcessAspect.initializeConstants(
                self, knowledgeBase, simulation
            )
            units = np.asarray(
                _get(knowledgeBase, "transcriptionUnits"), dtype=object
            ).ravel(order="F")
            base_counts = np.asarray(
                [_get(unit, "baseCount") for unit in units]
            )
            substrate_global = (
                np.asarray(
                    self.substrateMetaboliteGlobalIndexs, dtype=int
                ).ravel(order="F")
                - 1
            )
            self.transcriptionUnitBaseCounts = base_counts[:, substrate_global]
            self.transcriptionUnitBindingProbabilities = np.zeros(
                np.asarray(
                    _get(self.transcripts, "transcriptionUnitLengths")
                ).size
            )
            dnaA_ids = {
                "MG_469_1MER_ATP",
                "MG_469_2MER_1ATP_ADP",
                "MG_469_2MER_ATP",
                "MG_469_3MER_2ATP_ADP",
                "MG_469_3MER_ATP",
                "MG_469_4MER_3ATP_ADP",
                "MG_469_4MER_ATP",
                "MG_469_5MER_4ATP_ADP",
                "MG_469_5MER_ATP",
                "MG_469_6MER_5ATP_ADP",
                "MG_469_6MER_ATP",
                "MG_469_7MER_6ATP_ADP",
                "MG_469_7MER_ATP",
            }
            complexes = np.asarray(
                _get(knowledgeBase, "proteinComplexs"), dtype=object
            ).ravel(order="F")
            self.complexIndexs_DnaA_ATP = np.asarray(
                [
                    i + 1
                    for i, record in enumerate(complexes)
                    if _get(record, "wholeCellModelID") in dnaA_ids
                ],
                dtype=int,
            )
            feature_names = {
                "Functional box R1",
                "Functional box R2",
                "Functional box R3",
                "Functional box R4",
                "Functional box R5",
            }
            features = np.asarray(
                _get(knowledgeBase, "genomeFeatures"), dtype=object
            ).ravel(order="F")
            tu_indices = []
            for feature in features:
                if _get(feature, "name") not in feature_names:
                    continue
                tu_indices.extend(
                    int(_get(unit, "idx"))
                    for unit in np.asarray(
                        _get(feature, "transcriptionUnits"), dtype=object
                    ).ravel(order="F")
                )
            self.transcriptionUnitIndexs_DnaAR12345Boxes = np.unique(
                tu_indices
            ).astype(int)
        else:
            required = (
                "rnaPolymeraseElongationRate",
                "transcriptionUnitBaseCounts",
                "transcriptionUnitBindingProbabilities",
                "enzymeDNAFootprints",
                "enzymeDNAFootprints3Prime",
                "enzymeDNAFootprints5Prime",
                "complexIndexs_DnaA_ATP",
                "transcriptionUnitIndexs_DnaAR12345Boxes",
            )
            missing = [x for x in required if not hasattr(self, x)]
            if missing:
                raise NotImplementedError(
                    "Transcription constants missing from snapshot: "
                    + ", ".join(missing)
                )

    def copyFromState(self):
        counts = np.asarray(self.rna.counts)
        super().copyFromState()
        nascent = np.asarray(self.rna.nascentIndexs, dtype=int).reshape(-1) - 1
        cyt = (
            int(
                np.asarray(self.compartment.cytosolIndexs, dtype=int).reshape(
                    -1
                )[0]
            )
            - 1
        )
        if counts.ndim == 2:
            counts = counts[:, :, None]
        selected = counts[nascent, cyt, :]
        self.RNAs = selected.reshape((nascent.size, 1, counts.shape[2]))
        if counts.shape[2] == 1:
            self.RNAs = self.RNAs[:, :, 0]

    def copyToState(self):
        original_ndim = np.asarray(self.rna.counts).ndim
        counts = np.asarray(self.rna.counts)
        super().copyToState()
        rows = np.asarray(self.rna.nascentIndexs, int).ravel() - 1
        column = int(np.asarray(self.compartment.cytosolIndexs).item()) - 1
        if counts.ndim == 2:
            counts = counts[:, :, None]
        local = np.asarray(self.RNAs).reshape((rows.size, -1))
        for time in range(counts.shape[2]):
            counts[rows, column, time] = local[:, time]
        self.rna.counts = counts[:, :, 0] if original_ndim == 2 else counts

    def calcResourceRequirements_Current(self):
        result = np.zeros_like(np.asarray(self.substrates), dtype=float)
        nmps = np.asarray(self.metabolite.nmpComposition)
        active = float(np.asarray(self.rnaPolymerases.nActive).reshape(-1)[0])
        result[np.asarray(self.substrateIndexs_ntp, int).ravel() - 1, ...] = (
            2
            * np.sum(nmps, axis=1).reshape(-1, 1)
            * active
            * self.rnaPolymeraseElongationRate
        )
        result[int(np.asarray(self.substrateIndexs_water).item()) - 1] = active
        return result

    def calcStateTransitionProbabilities(
        self, nPols, ntpProdRate, tuBindingProbs, method="handTuned"
    ):
        """Source-equation transition matrix; row=destination, column=source."""
        r = self.rnaPolymerases
        e, s, n, f = (
            r.activelyTranscribingIndex - 1,
            r.specificallyBoundIndex - 1,
            r.nonSpecificallyBoundIndex - 1,
            r.freeIndex - 1,
        )
        expectations = np.asarray(r.stateExpectations, dtype=float).reshape(-1)
        lengths = np.asarray(self.rna.lengths)[
            np.asarray(self.rna.nascentIndexs, int).reshape(-1) - 1
        ]
        binding = np.asarray(tuBindingProbs, dtype=float).reshape(-1)
        p = np.zeros((4, 4), dtype=float)
        denom = float(binding @ np.asarray(lengths).reshape(-1))
        if nPols and expectations[e] and denom:
            p[n, e] = ntpProdRate / (nPols * expectations[e]) / denom
        if method == "handTuned":
            p[e, s], p[n, s], p[s, n] = 1, 0, 0.10
        else:
            p[e, s] = p[n, e] * expectations[e] / expectations[s]
            p[s, n] = p[n, e] * expectations[e] / expectations[n]
        p[e, e] = 1 - p[n, e]
        p[s, s] = 1 - p[e, s] - p[n, s]
        p[n, n] = 1 - p[s, n]
        p[:, f] = p[:, n]
        return p

    def computeRNAPolymeraseTUBindingProbabilities(
        self, protectDnaABoxes=True
    ):
        r = self.rnaPolymerases
        probs = (
            np.asarray(self.transcriptionUnitBindingProbabilities)
            * np.asarray(r.transcriptionFactorBindingProbFoldChange)
            * np.asarray(r.supercoilingBindingProbFoldChange)
        )
        if protectDnaABoxes:
            c = self.chromosome
            tus = np.asarray(
                self.transcriptionUnitIndexs_DnaAR12345Boxes, int
            ).reshape(-1)
            starts = np.asarray(c.transcriptionUnitStartCoordinates).reshape(
                -1
            )[tus - 1]
            lens = np.asarray(c.transcriptionUnitLengths).reshape(-1)[tus - 1]
            if len(tus):
                pos, complexes = c.complexBoundSites.find()
                pos, complexes = np.asarray(pos), np.asarray(
                    complexes
                ).reshape(-1).astype(int)
                mask = np.zeros(len(pos), bool)
                for st, le in zip(starts, lens):
                    mask |= (pos[:, 0] >= st) & (pos[:, 0] < st + le)
                bound = mask & np.isin(
                    complexes, np.asarray(self.complexIndexs_DnaA_ATP, int)
                )
                for j, tu in enumerate(tus):
                    inside = (
                        bound
                        & (pos[:, 0] >= starts[j])
                        & (pos[:, 0] < starts[j] + lens[j])
                    )
                    if np.any(inside & (pos[:, 1] <= 2)):
                        probs[tu - 1, 0] = 0
                    if np.any(inside & (pos[:, 1] > 2)):
                        probs[tu - 1, 1] = 0
        return probs

    def evolveState(self):
        """Advance RNA polymerases, transcribe sequences, terminate, and charge chemistry."""
        from ..polymerize import polymerize

        c, r, t = self.chromosome, self.rnaPolymerases, self.transcripts
        genome_length = int(np.asarray(c.sequenceLen).item())
        elongation_rate = float(
            np.asarray(self.rnaPolymeraseElongationRate).item()
        )
        states = np.asarray(r.states).reshape(-1)
        pos = np.asarray(r.positionStrands, dtype=int).reshape((-1, 2))
        bu = np.asarray(t.boundTranscriptionUnits).reshape(-1)
        bp = np.asarray(t.boundTranscriptProgress).reshape(-1)
        bc = np.asarray(t.boundTranscriptChromosome).reshape(-1)
        # Process enzyme vectors are local-indexed and MATLAB indices are kept at the boundary.
        tf = np.asarray(self.enzymes)[
            self.enzymeIndexs_transcriptionFactors - 1
        ].copy()
        btf = np.asarray(self.boundEnzymes)[
            self.enzymeIndexs_transcriptionFactors - 1
        ].copy()
        ep, bp_core = (
            self.enzymeIndexs_elongationFactor - 1,
            self.enzymeIndexs_rnaPolymerase - 1,
        )
        et, eh = (
            self.enzymeIndexs_terminationFactor - 1,
            self.enzymeIndexs_rnaPolymeraseHoloenzyme - 1,
        )
        frep, bndp = float(self.enzymes[bp_core]), float(
            self.boundEnzymes[bp_core]
        )
        freh, bndh = float(self.enzymes[eh]), float(self.boundEnzymes[eh])
        total = frep + bndp + freh + bndh
        if total > len(states):
            n = 2 * int(total) - len(states)
            states = np.r_[
                states, np.full(n, r.notExistValue, dtype=states.dtype)
            ]
            pos = np.vstack((pos, np.zeros((n, 2), int)))
            bu = np.r_[bu, np.full(n, t.nullTranscriptValue)]
            bp = np.r_[bp, np.full(n, t.nullTranscriptValue)]
            bc = np.r_[bc, np.full(n, t.nullTranscriptValue)]
        active = np.flatnonzero(states >= r.activelyTranscribingValue)
        sb = np.flatnonzero(states == r.specificallyBoundValue)
        free_nsb = np.flatnonzero(
            (states == r.nonSpecificallyBoundValue) | (states == r.freeValue)
        )
        sb = sb[
            np.asarray(self.randStream.randperm(len(sb)), int).reshape(-1) - 1
        ]
        free_nsb = free_nsb[
            np.asarray(self.randStream.randperm(len(free_nsb)), int).reshape(
                -1
            )
            - 1
        ]
        new_n = int(total) - len(active) - len(sb) - len(free_nsb)
        frep += freh
        tf[0] += freh
        freh = 0
        ptrans = np.asarray(self.stateTransitionProbabilities)
        ninit = int(
            np.asarray(
                self.randStream.stochasticRound(
                    len(sb)
                    * ptrans[
                        r.activelyTranscribingIndex - 1,
                        r.specificallyBoundIndex - 1,
                    ]
                )
            ).item()
        )
        if ninit:
            ids = sb[:ninit]
            released = self.releaseProteinFromSites(
                pos[ids],
                False,
                self.enzymeIndexs_rnaPolymeraseHoloenzyme,
                True,
                True,
            )
            if released[eh] != ninit:
                raise RuntimeError("Unable to release protein")
            tus = bu[ids].astype(int)
            starts = np.asarray(
                t.transcriptionUnitFivePrimeCoordinates
            ).reshape(-1)[tus - 1]
            ok, *_ = self.bindProteinToChromosome(
                np.c_[starts, pos[ids, 1]],
                self.enzymeIndexs_rnaPolymeraseHoloenzyme,
            )
            ok = np.asarray(ok, bool).reshape(-1)
            states[ids[ok]] = r.activelyTranscribingValue
            bp[ids[ok]] = t.newTranscriptValue
            pos[ids[ok], 0] = starts[ok]
            if not np.all(
                self.bindProteinToChromosome(
                    pos[ids[~ok]], self.enzymeIndexs_rnaPolymeraseHoloenzyme
                )[0]
            ):
                raise RuntimeError("Unable to rebind polymerase")
        nunbind = max(
            0,
            len(sb)
            - ninit
            - int(
                np.asarray(
                    self.randStream.stochasticRound(
                        len(sb)
                        * ptrans[
                            r.specificallyBoundIndex - 1,
                            r.specificallyBoundIndex - 1,
                        ]
                    )
                ).item()
            ),
        )
        if nunbind:
            ids = sb[-nunbind:]
            released = self.releaseProteinFromSites(
                pos[ids],
                False,
                self.enzymeIndexs_rnaPolymeraseHoloenzyme,
                True,
                True,
            )
            if released[eh] != nunbind:
                raise RuntimeError("Unable to unbind protein")
            states[ids] = r.freeValue
            pos[ids] = 0
            bu[ids] = bp[ids] = bc[ids] = t.nullTranscriptValue
            bndh -= nunbind
            frep += nunbind
            tf[0] += nunbind
        nbind = int(
            np.asarray(
                self.randStream.stochasticRound(
                    min(
                        len(free_nsb)
                        * ptrans[
                            r.specificallyBoundIndex - 1, r.freeIndex - 1
                        ],
                        tf[0],
                    )
                )
            ).item()
        )
        if nbind:
            ids = free_nsb[:nbind]
            ns = int(np.sum(states[ids] == r.nonSpecificallyBoundValue))
            released = self.releaseProteinFromSites(
                pos[ids], False, self.enzymeIndexs_rnaPolymerase, True, True
            )
            if released[bp_core] != ns:
                raise RuntimeError("Unable to release protein")
            states[ids] = r.freeValue
            pos[ids] = 0
            bu[ids] = bp[ids] = bc[ids] = t.nullTranscriptValue
            frep += ns
            bndp -= ns
            probs = self.computeRNAPolymeraseTUBindingProbabilities()
            tus = np.asarray(t.transcriptionUnitLengths).size
            coords = _specific_binding_coordinates(
                t.transcriptionUnitFivePrimeCoordinates,
                t.transcriptionUnitDirections,
            )
            poly = c.polymerizedRegions
            if poly.nnz() == 2:
                _, itu, pnew, nbind, _ = self.bindProteinToChromosome(
                    np.c_[
                        coords,
                        np.asarray(c.transcriptionUnitStrands).reshape(-1),
                    ],
                    self.enzymeIndexs_rnaPolymeraseHoloenzyme,
                    nbind,
                    probs[:, 0],
                    True,
                    True,
                    1,
                    False,
                    [],
                )
                ichr = np.ones(len(itu), int)
            else:
                strands = np.asarray(c.transcriptionUnitStrands).reshape(-1)
                pset = np.vstack(
                    (np.c_[coords, strands], np.c_[coords, strands + 2])
                )
                _, itu, pnew, nbind, _ = self.bindProteinToChromosome(
                    pset,
                    self.enzymeIndexs_rnaPolymeraseHoloenzyme,
                    nbind,
                    probs.ravel(order="F"),
                    True,
                    True,
                    1,
                    False,
                    [],
                )
                itu = (np.asarray(itu) - 1) % tus + 1
                ichr = np.ceil(np.asarray(pnew)[:, 1] / 2).astype(int)
            ids = free_nsb[:nbind]
            states[ids] = r.specificallyBoundValue
            pos[ids] = pnew
            bu[ids] = itu
            bp[ids] = t.nullTranscriptValue
            bc[ids] = ichr
            bndh += nbind
            frep -= nbind
            tf[0] -= nbind
        # Non-specific binding dissociates each tick before free polymerases rebind stochastically.
        ids = np.flatnonzero(states == r.nonSpecificallyBoundValue)
        if len(ids):
            rel = self.releaseProteinFromSites(
                pos[ids], False, self.enzymeIndexs_rnaPolymerase, True, True
            )
            if rel[bp_core] != len(ids):
                raise RuntimeError("Unable to release protein")
            states[ids] = r.freeValue
            pos[ids] = 0
            bu[ids] = bp[ids] = bc[ids] = t.nullTranscriptValue
            frep += len(ids)
            bndp -= len(ids)
        free = np.flatnonzero(states == r.freeValue)
        pn = ptrans[r.nonSpecificallyBoundIndex - 1, r.freeIndex - 1]
        pf = ptrans[r.freeIndex - 1, r.freeIndex - 1]
        nn = (
            int(
                np.asarray(
                    self.randStream.stochasticRound(
                        min(frep, len(free) * pn / (pn + pf))
                    )
                ).item()
            )
            if pn + pf
            else 0
        )
        if nn:
            nn, pnew = self.bindProteinToChromosomeStochastically(
                self.enzymeIndexs_rnaPolymerase, nn
            )
            fp5 = int(
                np.asarray(self.enzymeDNAFootprints5Prime).reshape(-1)[bp_core]
            )
            fp3 = int(
                np.asarray(self.enzymeDNAFootprints3Prime).reshape(-1)[bp_core]
            )
            pnew[np.asarray(pnew)[:, 1] % 2 == 1, 0] += fp5
            pnew[np.asarray(pnew)[:, 1] % 2 == 0, 0] += fp3
            ids = free[:nn]
            states[ids] = r.nonSpecificallyBoundValue
            pos[ids] = pnew
            bu[ids] = bp[ids] = bc[ids] = t.nullTranscriptValue
            frep -= nn
            bndp += nn
        if new_n > 0:
            ids = np.flatnonzero(states == r.notExistValue)[:new_n]
            states[ids] = r.freeValue
            pos[ids] = 0
            bu[ids] = bp[ids] = bc[ids] = t.nullTranscriptValue
        used = np.zeros(4, int)
        if len(active) and tf[ep] and elongation_rate > 0:
            old = pos[active].copy()
            tus = bu[active].astype(int)
            dirs = (
                2
                * np.asarray(t.transcriptionUnitDirections).reshape(-1)[
                    tus - 1
                ]
                - 1
            )
            rel = self.releaseProteinFromSites(
                old, False, self.enzymeIndexs_rnaPolymerase, True, True
            ) + self.releaseProteinFromSites(
                old,
                False,
                self.enzymeIndexs_rnaPolymeraseHoloenzyme,
                True,
                True,
            )
            if rel[bp_core] + rel[eh] != len(active):
                raise RuntimeError("Unable to unbind protein")
            actseq = _active_transcription_sequence_matrix(
                t.transcriptionUnitSequences,
                tus,
                states,
                active,
                elongation_rate,
            )
            maxel = np.minimum(
                elongation_rate,
                np.asarray(t.transcriptionUnitLengths).reshape(-1)[tus - 1]
                - states[active]
                + 1,
            ).astype(int)
            for strand in range(1, 5):
                ix = np.flatnonzero(old[:, 1] == strand)
                if len(ix) > 1:
                    order = ix[np.argsort(old[ix, 0])]
                    gap = (
                        np.diff(
                            np.r_[
                                old[order, 0], old[order[0], 0] + genome_length
                            ]
                        )
                        if strand % 2
                        else np.diff(
                            np.r_[
                                old[order[-1], 0] - genome_length,
                                old[order, 0],
                            ]
                        )
                    )
                    maxel[order] = np.minimum(
                        maxel[order], gap - self.enzymeDNAFootprints[bp_core]
                    )
            accessible = c.isRegionAccessible(
                old,
                (maxel + 1) * dirs,
                [],
                np.asarray(self.enzymeGlobalIndexs)[bp_core],
                True,
                [],
                True,
                False,
            )
            extent = np.asarray(accessible[3]).reshape(-1)
            if len(extent) and not np.all(extent):
                raise RuntimeError("Unable to bind proteins")
            if len(extent):
                maxel = np.minimum(np.abs(extent) - 1, maxel)
            for j, n in enumerate(maxel):
                actseq[j, int(n) :] = " "
            prog, self.substrates[self.substrateIndexs_ntp - 1], used, _, _ = (
                polymerize(
                    actseq,
                    self.substrates[self.substrateIndexs_ntp - 1],
                    "ACGU",
                    " ",
                    0,
                    0,
                    self.randStream,
                )
            )
            prog = np.asarray(prog).reshape(-1).astype(int)
            inits = int(
                np.sum(
                    (states[active] == r.activelyTranscribingValue)
                    & (prog > 0)
                )
            )
            bndh -= inits
            tf[0] += inits
            bndp += inits
            states[active] += prog
            pos[active, 0] = old[:, 0] + prog * dirs
            bp[active] = states[active]
            init = states[active] == r.activelyTranscribingValue
            if not np.all(
                self.bindProteinToChromosome(
                    pos[active[~init]], self.enzymeIndexs_rnaPolymerase
                )[0]
            ) or not np.all(
                self.bindProteinToChromosome(
                    pos[active[init]],
                    self.enzymeIndexs_rnaPolymeraseHoloenzyme,
                )[0]
            ):
                raise RuntimeError("Unable to bind proteins")
        # Complete transcripts consume water, release hydrogen, and increment TU RNA counts.
        if len(active) and tf[ep] and tf[et]:
            lens = np.asarray(t.transcriptionUnitLengths).reshape(-1)
            term = active[states[active] > lens[bu[active].astype(int) - 1]]
            nterm = len(term)
            water = self.substrates[self.substrateIndexs_water - 1]
            if nterm > water:
                term = term[
                    np.asarray(self.randStream.randperm(nterm), int).reshape(
                        -1
                    )[: int(water)]
                    - 1
                ]
                nterm = len(term)
            if nterm:
                np.add.at(self.RNAs, bu[term].astype(int) - 1, 1)
                rel = self.releaseProteinFromSites(
                    pos[term],
                    False,
                    self.enzymeIndexs_rnaPolymerase,
                    True,
                    True,
                )
                if rel[bp_core] != nterm:
                    raise RuntimeError("Unable to unbind protein")
                frep += nterm
                bndp -= nterm
                self.substrates[self.substrateIndexs_water - 1] -= nterm
                self.substrates[self.substrateIndexs_hydrogen - 1] += nterm
                states[term] = r.freeValue
                pos[term] = 0
                bu[term] = bp[term] = bc[term] = t.nullTranscriptValue
        self.substrates[self.substrateIndexs_diphosphate - 1] += np.sum(used)
        self.enzymes[self.enzymeIndexs_transcriptionFactors - 1] = tf
        self.boundEnzymes[self.enzymeIndexs_transcriptionFactors - 1] = btf
        self.enzymes[bp_core] = frep
        self.boundEnzymes[bp_core] = bndp
        self.enzymes[eh] = freh
        self.boundEnzymes[eh] = bndh
        r.states = states[:, None]
        r.positionStrands = pos
        t.boundTranscriptionUnits = bu[:, None]
        t.boundTranscriptProgress = bp[:, None]
        t.boundTranscriptChromosome = bc[:, None]

    def initializeState(self):
        r, t = self.rnaPolymerases, self.transcripts
        directions = np.asarray(t.transcriptionUnitDirections).reshape(-1)
        lengths = np.asarray(t.transcriptionUnitLengths).reshape(-1)
        five_prime = np.asarray(
            t.transcriptionUnitFivePrimeCoordinates
        ).reshape(-1)
        strands = 2 - directions
        dirs = 2 * directions - 1
        starts = (five_prime - (1 - directions) * (lengths - 1)).astype(float)

        def ix(name):
            return int(np.asarray(getattr(self, name)).reshape(-1)[0]) - 1

        ep, eh = ix("enzymeIndexs_rnaPolymerase"), ix(
            "enzymeIndexs_rnaPolymeraseHoloenzyme"
        )
        sigma = ix("enzymeIndexs_sigmaFactor")
        fp5 = np.asarray(self.enzymeDNAFootprints5Prime).reshape(-1)
        fp3 = np.asarray(self.enzymeDNAFootprints3Prime).reshape(-1)
        starts[dirs == 1] -= fp5[ep]
        starts[dirs == -1] += fp5[ep]

        free_tf = (
            np.asarray(self.enzymes)[
                np.asarray(self.enzymeIndexs_transcriptionFactors, int).ravel()
                - 1
            ]
            .reshape(-1)
            .copy()
        )
        bound_tf = (
            np.asarray(self.boundEnzymes)[
                np.asarray(self.enzymeIndexs_transcriptionFactors, int).ravel()
                - 1
            ]
            .reshape(-1)
            .copy()
        )
        free_poly, bound_poly = (
            float(np.asarray(x).reshape(-1)[0])
            for x in (
                np.asarray(self.enzymes)[ep],
                np.asarray(self.boundEnzymes)[ep],
            )
        )
        free_holo, bound_holo = (
            float(np.asarray(x).reshape(-1)[0])
            for x in (
                np.asarray(self.enzymes)[eh],
                np.asarray(self.boundEnzymes)[eh],
            )
        )
        n_poly = int(free_poly + bound_poly + free_holo + bound_holo)
        states = np.full(2 * n_poly, r.notExistValue, dtype=int)
        positions = np.zeros((2 * n_poly, 2), dtype=int)
        bound_tu = np.full(2 * n_poly, t.nullTranscriptValue, dtype=int)
        progress = np.full(2 * n_poly, t.nullTranscriptValue, dtype=int)
        bound_chr = np.full(2 * n_poly, t.nullTranscriptValue, dtype=int)
        chromosome = self.chromosome
        binding = np.asarray(
            self.computeRNAPolymeraseTUBindingProbabilities()
        )[:, 0]
        p_bind = lengths * binding
        p_specific = p_bind.copy()
        p_state = (
            np.asarray(r.stateExpectations, dtype=float).reshape(-1).copy()
        )

        for i in range(n_poly):
            if (
                free_poly == 0
                or not np.any(p_bind)
                or free_tf[self.enzymeIndexs_elongationFactor - 1] == 0
            ):
                p_state[r.activelyTranscribingIndex - 1] = 0
            if free_poly == 0 or not np.any(p_specific) or free_tf[sigma] == 0:
                p_state[r.specificallyBoundIndex - 1] = 0
            if free_poly == 0:
                p_state[r.nonSpecificallyBoundIndex - 1] = 0
            if not np.any(p_state):
                p_state[r.freeIndex - 1] = 1
            state = int(
                self.randStream.randsample(4, 1, True, p_state).reshape(-1)[0]
            )
            if state == r.activelyTranscribingIndex:
                trial = p_bind.copy()
                pos = np.empty((0, 2), dtype=int)
                while np.any(trial):
                    tu = int(
                        self.randStream.randsample(
                            len(trial), 1, True, trial
                        ).reshape(-1)[0]
                    )
                    trial[tu - 1] = 0
                    _, pos = self.bindProteinToChromosomeStochastically(
                        self.enzymeIndexs_rnaPolymerase,
                        1,
                        [starts[tu - 1] + 1, strands[tu - 1]],
                        lengths[tu - 1] - 2,
                    )
                    pos = np.asarray(pos, dtype=int).reshape((-1, 2))
                    if len(pos):
                        break
                if not len(pos):
                    raise RuntimeError(
                        "Transcription: unable to bind chromosome"
                    )
                pos[pos[:, 1] % 2 == 1, 0] += int(fp5[ep])
                pos[pos[:, 1] % 2 == 0, 0] += int(fp3[ep])
                value = (
                    pos[:, 0] - five_prime[tu - 1] + 1
                    if directions[tu - 1]
                    else five_prime[tu - 1] - pos[:, 0] + 1
                )
                states[i] = int(np.asarray(value).reshape(-1)[0])
                positions[i] = pos[0]
                progress[i] = states[i]
                bound_tu[i] = tu
                bound_chr[i] = 1
                free_poly -= 1
                bound_poly += 1
            elif state == r.specificallyBoundIndex:
                pos = np.empty((0, 2), dtype=int)
                while np.any(p_specific):
                    tu = int(
                        self.randStream.randsample(
                            len(p_specific), 1, True, p_specific
                        ).reshape(-1)[0]
                    )
                    p_specific[tu - 1] = 0
                    _, _, pos, _, _ = self.bindProteinToChromosome(
                        [[five_prime[tu - 1] - dirs[tu - 1], strands[tu - 1]]],
                        self.enzymeIndexs_rnaPolymeraseHoloenzyme,
                        1,
                    )
                    pos = np.asarray(pos, dtype=int).reshape((-1, 2))
                    if len(pos):
                        break
                if not len(pos):
                    raise RuntimeError(
                        "Transcription: unable to bind chromosome"
                    )
                states[i] = r.specificallyBoundValue
                positions[i] = pos[0]
                bound_tu[i] = tu
                bound_chr[i] = 1
                bound_holo += 1
                free_poly -= 1
                free_tf[sigma] -= 1
            elif state == r.nonSpecificallyBoundIndex:
                _, pos = self.bindProteinToChromosomeStochastically(
                    self.enzymeIndexs_rnaPolymerase, 1
                )
                pos = np.asarray(pos, dtype=int).reshape((-1, 2))
                if not len(pos):
                    raise RuntimeError(
                        "Transcription: unable to bind chromosome"
                    )
                pos[pos[:, 1] % 2 == 1, 0] += int(fp5[ep])
                pos[pos[:, 1] % 2 == 0, 0] += int(fp3[ep])
                states[i] = r.nonSpecificallyBoundValue
                positions[i] = pos[0]
                free_poly -= 1
                bound_poly += 1
            else:
                states[i] = r.freeValue

        _set(r, "states", states[:, None])
        _set(r, "positionStrands", positions)
        _set(t, "boundTranscriptionUnits", bound_tu[:, None])
        _set(t, "boundTranscriptProgress", progress[:, None])
        _set(t, "boundTranscriptChromosome", bound_chr[:, None])
        enzymes_out = np.asarray(self.enzymes).reshape(-1).copy()
        bound_out = np.asarray(self.boundEnzymes).reshape(-1).copy()
        enzymes_out[
            np.asarray(self.enzymeIndexs_transcriptionFactors, int).ravel() - 1
        ] = free_tf
        bound_out[
            np.asarray(self.enzymeIndexs_transcriptionFactors, int).ravel() - 1
        ] = bound_tf
        enzymes_out[ep], bound_out[ep] = free_poly, bound_poly
        enzymes_out[eh], bound_out[eh] = free_holo, bound_holo
        self.enzymes = enzymes_out.reshape(np.shape(self.enzymes))
        self.boundEnzymes = bound_out.reshape(np.shape(self.boundEnzymes))
        self.copyToState()
        rna = self.rna
        initial_counts = np.asarray(rna.counts).copy()
        target_weight = max(
            0.0, float(np.asarray(rna.dryWeight).sum()) - float(t.dryWeight)
        )
        cyt = (
            int(np.asarray(self.compartment.cytosolIndexs).reshape(-1)[0]) - 1
        )
        counts = np.asarray(rna.counts)
        while float(np.asarray(rna.dryWeight).sum()) > target_weight:
            weights = np.asarray(counts[:, cyt]).reshape(-1)
            if not np.any(weights > 0):
                break
            selected = (
                int(
                    self.randStream.randsample(
                        len(weights), 1, False, weights
                    ).reshape(-1)[0]
                )
                - 1
            )
            counts[selected, cyt] -= 1
        _set(rna, "counts", counts)
        if callable(getattr(rna, "updateExternalState", None)):
            delta = np.zeros_like(counts)
            delta[:, cyt] = counts[:, cyt] - initial_counts[:, cyt]
            correction = rna.updateExternalState(delta, True)
            counts += np.asarray(correction)
            _set(rna, "counts", counts)
        self.copyFromState()
