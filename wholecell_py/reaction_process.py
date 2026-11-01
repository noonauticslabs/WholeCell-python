"""Knowledge-base reaction-network initialization from MATLAB ReactionProcess."""

from __future__ import annotations

import numpy as np

from .process_base import Process, _get
from .sparsemat import SparseMat


def _ids(state, indices, mature=False):
    all_ids = np.asarray(_get(state, "wholeCellModelIDs"), dtype=object).ravel(
        order="F"
    )
    if mature:
        mature_indices = np.asarray(
            _get(state, "matureIndexs"), dtype=int
        ).ravel(order="F")
        indices = mature_indices[
            np.asarray(indices, dtype=int).ravel(order="F")
        ]
    else:
        indices = np.asarray(indices, dtype=int).ravel(order="F") + 1
    return all_ids[indices - 1].reshape(-1, 1)


def _reaction_rows(matrix, reaction_indices, axis):
    array = (
        matrix.full() if isinstance(matrix, SparseMat) else np.asarray(matrix)
    )
    if array.dtype.kind == "O":
        raise TypeError(
            "reaction tensor must be a dense NumPy array or wholecell_py SparseMat"
        )
    if array.ndim != 3:
        raise ValueError(
            "knowledge-base reaction matrices must be three-dimensional"
        )
    return np.take(
        array, np.asarray(reaction_indices, dtype=int).ravel() - 1, axis=axis
    )


def _active_rows(matrix, axes):
    array = np.asarray(matrix)
    if not array.size:
        remaining = tuple(
            size for axis, size in enumerate(array.shape) if axis not in axes
        )
        return np.zeros(remaining, dtype=bool)
    return np.sum(np.abs(array), axis=axes) != 0


def _select_compartment(matrix, compartments, axis, component_axis):
    """Select one-based compartment per substrate/enzyme/modified component."""
    array = np.asarray(matrix)
    compartments = np.asarray(compartments, dtype=int).ravel(order="F") - 1
    if not compartments.size:
        return np.take(array, np.empty(0, dtype=int), axis=axis)
    if (
        axis == 2
        and array.ndim == 3
        and array.shape[component_axis] == len(compartments)
    ):
        index_shape = [1, 1, 1]
        index_shape[component_axis] = len(compartments)
        ix = np.broadcast_to(
            compartments.reshape(index_shape), (*array.shape[:2], 1)
        )
        return np.take_along_axis(array, ix, axis=2)[:, :, 0]
    raise ValueError(
        "component compartments do not match reaction matrix dimensions"
    )


class ReactionProcess(Process):
    """Process whose state is represented by knowledge-base reactions.

    Indices on the process remain one-based. The initializer follows the
    source ReactionProcess.m ordering and applies compartment collapse only
    after Process has built global/local component maps.
    """

    stimuliWholeCellModelIDs = ()
    substrateWholeCellModelIDs = ()
    enzymeWholeCellModelIDs = ()
    reactionWholeCellModelIDs = ()

    def __init__(self, wholeCellModelID="", name="", **properties):
        super().__init__(wholeCellModelID, name, **properties)

    def reactionIndexs(self, wholeCellModelIDs):
        ids = np.asarray(wholeCellModelIDs, dtype=object).ravel(order="F")
        available = np.asarray(
            self.reactionWholeCellModelIDs, dtype=object
        ).ravel(order="F")
        return self.componentIndexs(ids, "reaction")

    def initializeConstants(self, knowledgeBase, simulation, options=None):
        options = options if isinstance(options, dict) else {}
        processes = np.asarray(
            _get(knowledgeBase, "processes"), dtype=object
        ).ravel(order="F")
        record = next(
            (
                p
                for p in processes
                if _get(p, "wholeCellModelID") == self.wholeCellModelID
            ),
            None,
        )
        if record is None:
            raise ValueError(
                f"ReactionProcess: {self.wholeCellModelID!r} missing from knowledge base"
            )
        reactions = np.asarray(_get(record, "reactions"), dtype=object).ravel(
            order="F"
        )
        reaction_indices = np.asarray(
            [_get(reaction, "idx") for reaction in reactions], dtype=int
        )
        self.reactionWholeCellModelIDs = np.asarray(
            [_get(reaction, "wholeCellModelID") for reaction in reactions],
            dtype=object,
        ).reshape(-1, 1)
        self.reactionNames = np.asarray(
            [_get(reaction, "name") for reaction in reactions], dtype=object
        ).reshape(-1, 1)
        self.reactionTypes = np.asarray(
            [_get(reaction, "type") for reaction in reactions], dtype=object
        ).reshape(-1, 1)

        kb = knowledgeBase
        rsm = _reaction_rows(
            _get(kb, "reactionStimuliStoichiometryMatrix"), reaction_indices, 1
        )
        rmm = _reaction_rows(
            _get(kb, "reactionMetaboliteStoichiometryMatrix"),
            reaction_indices,
            1,
        )
        rna = _reaction_rows(
            _get(kb, "reactionRNAStoichiometryMatrix"), reaction_indices, 1
        )
        mon = _reaction_rows(
            _get(kb, "reactionProteinMonomerStoichiometryMatrix"),
            reaction_indices,
            1,
        )
        cmp = _reaction_rows(
            _get(kb, "reactionProteinComplexStoichiometryMatrix"),
            reaction_indices,
            1,
        )
        cat = _reaction_rows(
            _get(kb, "reactionCatalysisMatrix"), reaction_indices, 0
        )
        mod = _reaction_rows(
            _get(kb, "reactionModificationMatrix"), reaction_indices, 0
        )
        coenzyme = _reaction_rows(
            _get(kb, "reactionCoenzymeMatrix"), reaction_indices, 0
        )
        self.enzymeBounds = _reaction_rows(
            np.asarray(_get(kb, "enzymeBounds"))[:, :, None],
            reaction_indices,
            0,
        )[:, :, 0]
        self.reactionBounds = _reaction_rows(
            np.asarray(_get(kb, "reactionBounds"))[:, :, None],
            reaction_indices,
            0,
        )[:, :, 0]

        stim_rows = np.flatnonzero(_active_rows(rsm, (1, 2)))
        met_rows = _active_rows(rmm, (1, 2)) | _active_rows(coenzyme, (0, 2))
        met_rows = np.flatnonzero(met_rows)
        rna_rows = np.flatnonzero(_active_rows(rna, (1, 2)))
        mon_rows = np.flatnonzero(_active_rows(mon, (1, 2)))
        cmp_rows = np.flatnonzero(_active_rows(cmp, (1, 2)))

        self.stimuliWholeCellModelIDs = _ids(self.stimulus, stim_rows)
        self.substrateWholeCellModelIDs = np.vstack(
            (
                self.stimuliWholeCellModelIDs,
                _ids(self.metabolite, met_rows),
                _ids(self.rna, rna_rows, mature=True),
                _ids(self.monomer, mon_rows, mature=True),
                _ids(self.complex, cmp_rows, mature=True),
            )
        )
        self.reactionStoichiometryMatrix = np.concatenate(
            (
                rsm[stim_rows],
                rmm[met_rows],
                rna[rna_rows],
                mon[mon_rows],
                cmp[cmp_rows],
            ),
            axis=0,
        )
        self.reactionCoenzymeMatrix = np.zeros(
            (
                len(reactions),
                len(self.substrateWholeCellModelIDs),
                coenzyme.shape[2],
            ),
            dtype=coenzyme.dtype,
        )
        start = len(stim_rows)
        self.reactionCoenzymeMatrix[:, start : start + len(met_rows)] = (
            coenzyme[:, met_rows]
        )

        enzyme_rows = np.flatnonzero(_active_rows(cat, (0, 2)))
        n_monomers = len(
            np.asarray(_get(self.monomer, "matureIndexs")).ravel()
        )
        monomer_rows = enzyme_rows[enzyme_rows < n_monomers]
        complex_rows = enzyme_rows[enzyme_rows >= n_monomers] - n_monomers
        self.enzymeWholeCellModelIDs = np.vstack(
            (
                _ids(self.monomer, monomer_rows, mature=True),
                _ids(self.complex, complex_rows, mature=True),
            )
        )
        self.reactionCatalysisMatrix = cat[:, enzyme_rows]
        self.reactionModificationMatrix = mod

        Process.initializeConstants(self, knowledgeBase, simulation, options)

        if not options.get("retainSubstrateCompartments", False):
            compartments = np.concatenate(
                (
                    np.asarray(
                        self.substrateStimulusCompartmentIndexs
                    ).ravel(),
                    np.asarray(
                        self.substrateMetaboliteCompartmentIndexs
                    ).ravel(),
                    np.asarray(self.substrateRNACompartmentIndexs).ravel(),
                    np.asarray(self.substrateMonomerCompartmentIndexs).ravel(),
                    np.asarray(self.substrateComplexCompartmentIndexs).ravel(),
                )
            )
            self.reactionStoichiometryMatrix = _select_compartment(
                self.reactionStoichiometryMatrix, compartments, 2, 0
            )
            self.reactionCoenzymeMatrix = _select_compartment(
                self.reactionCoenzymeMatrix, compartments, 2, 1
            )
        if not options.get("retainEnzymeCompartments", False):
            enzyme_compartments = np.concatenate(
                (
                    np.asarray(self.enzymeMonomerCompartmentIndexs).ravel(),
                    np.asarray(self.enzymeComplexCompartmentIndexs).ravel(),
                )
            )
            self.reactionCatalysisMatrix = _select_compartment(
                self.reactionCatalysisMatrix, enzyme_compartments, 2, 1
            )
        if not options.get("retainModificationCompartments", False):
            gene_compartments = np.full(
                np.asarray(self.reactionModificationMatrix).shape[1],
                int(np.asarray(self.compartment.cytosolIndexs).item()),
                dtype=int,
            )
            mrna = np.asarray(self.gene.mRNAIndexs, dtype=int).ravel() - 1
            gene_compartments[mrna] = np.asarray(
                self.monomer.compartments, dtype=int
            ).ravel()[
                np.asarray(self.monomer.matureIndexs, dtype=int).ravel() - 1
            ]
            self.reactionModificationMatrix = _select_compartment(
                self.reactionModificationMatrix, gene_compartments, 2, 1
            )

    def initializeConstants_overrideReactions(
        self, simulation, wholeCellModelIDs
    ):
        """Keep a requested reaction subset and discard unused components.

        This follows ReactionProcess.m:229-258. Tensor axes are preserved when
        compartment dimensions are retained; the same code handles already
        collapsed 2-D matrices.
        """
        del simulation  # MATLAB uses this argument only for the inherited overrides.
        old_ids = np.asarray(
            self.reactionWholeCellModelIDs, dtype=object
        ).ravel(order="F")
        requested = np.asarray(wholeCellModelIDs, dtype=object).ravel(
            order="F"
        )
        lookup = {}
        for i, identifier in enumerate(old_ids):
            lookup.setdefault(identifier, i)
        missing = [
            identifier for identifier in requested if identifier not in lookup
        ]
        if missing:
            raise ValueError(f"reactions not found: {missing!r}")
        indices = np.asarray([lookup[x] for x in requested], dtype=int)

        self.reactionWholeCellModelIDs = requested.reshape(-1, 1)
        for name in ("reactionNames", "reactionTypes"):
            values = np.asarray(getattr(self, name), dtype=object).ravel(
                order="F"
            )
            setattr(self, name, values[indices].reshape(-1, 1))
        self.reactionStoichiometryMatrix = np.take(
            np.asarray(self.reactionStoichiometryMatrix), indices, axis=1
        )
        for name in (
            "reactionCatalysisMatrix",
            "reactionModificationMatrix",
            "reactionCoenzymeMatrix",
            "reactionBounds",
            "enzymeBounds",
        ):
            setattr(
                self,
                name,
                np.take(np.asarray(getattr(self, name)), indices, axis=0),
            )

        stoich = np.asarray(self.reactionStoichiometryMatrix)
        coenzyme = np.asarray(self.reactionCoenzymeMatrix)
        substrate_used = np.any(stoich != 0, axis=tuple(range(1, stoich.ndim)))
        coenzyme_axes = (0,) + tuple(range(2, coenzyme.ndim))
        substrate_used |= np.any(coenzyme != 0, axis=coenzyme_axes)
        substrate_ids = np.asarray(
            self.substrateWholeCellModelIDs, dtype=object
        ).ravel(order="F")
        self.initializeConstants_overrideSubstrates(
            substrate_ids[substrate_used]
        )

        catalysis = np.asarray(self.reactionCatalysisMatrix)
        enzyme_axes = (0,) + tuple(range(2, catalysis.ndim))
        enzyme_used = np.any(catalysis != 0, axis=enzyme_axes)
        enzyme_ids = np.asarray(
            self.enzymeWholeCellModelIDs, dtype=object
        ).ravel(order="F")
        self.initializeConstants_overrideEnzymes(enzyme_ids[enzyme_used])

    def initializeConstants_overrideSubstrates(self, wholeCellModelIDs):
        wholeCellModelIDs = np.asarray(
            wholeCellModelIDs, dtype=object
        ).reshape(-1, 1)
        old = np.asarray(self.substrateWholeCellModelIDs, dtype=object).ravel(
            order="F"
        )
        new = wholeCellModelIDs.ravel(order="F")
        lookup = {identifier: i for i, identifier in enumerate(old)}
        stoich = np.zeros(
            (len(new),)
            + np.asarray(self.reactionStoichiometryMatrix).shape[1:]
        )
        coenzyme = np.zeros(
            (np.asarray(self.reactionCoenzymeMatrix).shape[0], len(new))
            + np.asarray(self.reactionCoenzymeMatrix).shape[2:]
        )
        for j, identifier in enumerate(new):
            if identifier in lookup:
                old_index = lookup[identifier]
                stoich[j] = self.reactionStoichiometryMatrix[old_index]
                coenzyme[:, j] = self.reactionCoenzymeMatrix[:, old_index]
        self.substrateWholeCellModelIDs = wholeCellModelIDs
        self.reactionStoichiometryMatrix = stoich
        self.reactionCoenzymeMatrix = coenzyme
        Process.initializeConstants_overrideSubstrates(self, wholeCellModelIDs)

    def initializeConstants_overrideEnzymes(self, wholeCellModelIDs):
        wholeCellModelIDs = np.asarray(
            wholeCellModelIDs, dtype=object
        ).reshape(-1, 1)
        wholeCellModelIDs = np.sort(wholeCellModelIDs, axis=0)
        old = np.asarray(self.enzymeWholeCellModelIDs, dtype=object).ravel(
            order="F"
        )
        new = wholeCellModelIDs.ravel(order="F")
        lookup = {identifier: i for i, identifier in enumerate(old)}
        catalysis = np.zeros(
            (np.asarray(self.reactionCatalysisMatrix).shape[0], len(new))
            + np.asarray(self.reactionCatalysisMatrix).shape[2:]
        )
        for j, identifier in enumerate(new):
            if identifier in lookup:
                catalysis[:, j] = self.reactionCatalysisMatrix[
                    :, lookup[identifier]
                ]
        self.enzymeWholeCellModelIDs = wholeCellModelIDs
        self.reactionCatalysisMatrix = catalysis
        Process.initializeConstants_overrideEnzymes(self, wholeCellModelIDs)
