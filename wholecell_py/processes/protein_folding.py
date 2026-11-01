"""Protein folding process port; exported MATLAB indices stay one based."""

from __future__ import annotations

import numpy as np

from ..process_base import Process


class ProteinFolding(Process):
    _LOCAL_STATES = (
        "unfoldedMonomers",
        "unfoldedComplexs",
        "foldedMonomers",
        "foldedComplexs",
    )
    optionNames__ = ()
    fixedConstantNames__ = (
        "proteinProstheticGroupMatrix",
        "proteinChaperoneMatrix",
        "monomerCompartments",
        "complexCompartments",
    )
    fittedConstantNames__ = ()
    localStateNames__ = (
        "unfoldedMonomers",
        "unfoldedComplexs",
        "foldedMonomers",
        "foldedComplexs",
    )
    substrateWholeCellModelIDs = ("ATP", "ADP", "PI", "H2O", "H")
    enzymeWholeCellModelIDs = (
        "MG_238_MONOMER",
        "MG_305_MONOMER",
        "MG_019_DIMER",
        "MG_201_DIMER",
        "MG_392_393_21MER",
    )
    _N_AVOGADRO = 6.0221417930e23

    def __init__(
        self,
        wholeCellModelID="Process_ProteinFolding",
        name="Protein Folding",
        **properties,
    ):
        super().__init__(wholeCellModelID, name, **properties)

    stimuliWholeCellModelIDs = ()

    def _initialize_from_kb(self, knowledgeBase, simulation, options=None):
        from ..process_base import _get, _cell_text

        def vector(obj, field, dtype=None):
            return np.asarray(_get(obj, field), dtype=dtype).ravel(order="F")

        def tensor(value):
            return np.asarray(
                value.full() if hasattr(value, "full") else value
            )

        cyt = int(np.asarray(self.compartment.cytosolIndexs).item())
        prosthetic = tensor(
            _get(knowledgeBase, "proteinProstheticGroupMatrix")
        )[:, :, cyt - 1].copy()
        prosthetic[prosthetic == -1] = 1
        metabolite_ids = vector(self.metabolite, "wholeCellModelIDs", object)
        self.substrateWholeCellModelIDs = np.asarray(
            sorted(
                set(map(_cell_text, self.substrateWholeCellModelIDs))
                | set(
                    map(_cell_text, metabolite_ids[prosthetic.sum(axis=0) > 0])
                )
            ),
            dtype=object,
        ).reshape(-1, 1)
        for suffix, identifier in (
            ("adp", "ADP"),
            ("atp", "ATP"),
            ("fe2", "FE2"),
            ("hydrogen", "H"),
            ("water", "H2O"),
            ("k", "K"),
            ("mg", "MG"),
            ("mn", "MN"),
            ("sodium", "NA"),
            ("phosphate", "PI"),
            ("zinc", "ZN"),
        ):
            setattr(
                self,
                "substrateIndexs_" + suffix,
                self.substrateIndexs([identifier]),
            )
        chaperones = tensor(_get(knowledgeBase, "proteinChaperoneMatrix"))
        nmon = len(vector(self.monomer, "matureIndexs"))
        active = chaperones.sum(axis=(1, 2)) != 0
        extra_ids = []
        for state, selected in (
            (self.monomer, active[:nmon]),
            (self.complex, active[nmon:]),
        ):
            mature = vector(state, "matureIndexs", int)
            extra_ids.extend(
                vector(state, "wholeCellModelIDs", object)[
                    mature[selected] - 1
                ]
            )
        self.enzymeWholeCellModelIDs = np.asarray(
            sorted(
                set(map(_cell_text, self.enzymeWholeCellModelIDs))
                | set(map(_cell_text, extra_ids))
            ),
            dtype=object,
        ).reshape(-1, 1)
        for suffix, identifier in (
            ("triggerFactor", "MG_238_MONOMER"),
            ("dnaK", "MG_305_MONOMER"),
            ("dnaJ", "MG_019_DIMER"),
            ("grpE", "MG_201_DIMER"),
            ("groELES", "MG_392_393_21MER"),
        ):
            setattr(
                self, "enzymeIndexs_" + suffix, self.enzymeIndexs([identifier])
            )
        Process.initializeConstants(self, knowledgeBase, simulation, options)
        self.proteinProstheticGroupMatrix = prosthetic[
            :, self.substrateMetaboliteGlobalIndexs.ravel() - 1
        ]
        compartments = np.zeros(len(self.enzymeWholeCellModelIDs), dtype=int)
        for category in ("Monomer", "Complex"):
            compartments[
                getattr(self, "enzyme" + category + "LocalIndexs").ravel() - 1
            ] = getattr(
                self, "enzyme" + category + "CompartmentIndexs"
            ).ravel()
        enzyme_indices = self.enzymeGlobalIndexs.ravel().copy()
        enzyme_indices[self.enzymeComplexLocalIndexs.ravel() - 1] += nmon
        self.proteinChaperoneMatrix = chaperones[
            enzyme_indices[None, :] - 1,
            np.arange(chaperones.shape[1])[:, None],
            compartments[None, :] - 1,
        ].copy()
        self.proteinChaperoneMatrix[
            :nmon, int(self.enzymeIndexs_triggerFactor.item()) - 1
        ] = 1
        for name in ("dnaJ", "grpE"):
            self.proteinChaperoneMatrix[
                :, int(getattr(self, "enzymeIndexs_" + name).item()) - 1
            ] = self.proteinChaperoneMatrix[
                :, int(self.enzymeIndexs_dnaK.item()) - 1
            ]
        for kind, state, unfolded, folded, kbname in (
            (
                "Monomer",
                self.monomer,
                "processedIIIndexs",
                "foldedIndexs",
                "proteinMonomerCompartments",
            ),
            (
                "Complex",
                self.complex,
                "nascentIndexs",
                "matureIndexs",
                "proteinComplexCompartments",
            ),
        ):
            compartments = vector(knowledgeBase, kbname, int).copy()
            if kind == "Monomer":
                compartments[
                    compartments
                    == int(
                        np.asarray(
                            self.compartment.terminalOrganelleCytosolIndexs
                        ).item()
                    )
                ] = cyt
                compartments[
                    compartments
                    == int(
                        np.asarray(
                            self.compartment.terminalOrganelleMembraneIndexs
                        ).item()
                    )
                ] = int(np.asarray(self.compartment.membraneIndexs).item())
            setattr(
                self,
                kind.lower() + "Compartments",
                compartments.reshape(-1, 1),
            )
            ids = vector(state, "wholeCellModelIDs", object)
            for prefix, index_name in (
                ("unfolded", unfolded),
                ("folded", folded),
            ):
                rows = vector(state, index_name, int)
                setattr(
                    self,
                    prefix + kind + "WholeCellModelIDs",
                    ids[rows - 1].reshape(-1, 1),
                )
                setattr(
                    self,
                    prefix + kind + "CompartmentIndexs",
                    (rows + (compartments - 1) * len(ids)).reshape(-1, 1),
                )
        self.initializeSpeciesNetwork()

    def initializeSpeciesNetwork(self):
        selected = np.flatnonzero(
            (self.proteinProstheticGroupMatrix.sum(axis=1) > 0)
            | (self.proteinChaperoneMatrix.sum(axis=1) > 0)
        )
        nmon = np.size(self.unfoldedMonomerWholeCellModelIDs)
        ncomplex = np.size(self.unfoldedComplexWholeCellModelIDs)
        self.monomerComplexIndexs_folded = (selected + 1).reshape(-1, 1)
        self.monomerIndexs_folding = (selected[selected < nmon] + 1).reshape(
            -1, 1
        )
        self.complexIndexs_folding = (
            selected[selected >= nmon] - nmon + 1
        ).reshape(-1, 1)
        self.speciesIndexs_monomers = (
            np.flatnonzero(selected < nmon) + 1
        ).reshape(-1, 1)
        self.speciesIndexs_complexs = (
            np.flatnonzero(selected >= nmon) + 1
        ).reshape(-1, 1)
        self.monomerIndexs_notFolding = np.setdiff1d(
            np.arange(1, nmon + 1), self.monomerIndexs_folding
        ).reshape(-1, 1)
        self.complexIndexs_notFolding = np.setdiff1d(
            np.arange(1, ncomplex + 1), self.complexIndexs_folding
        ).reshape(-1, 1)
        self.speciesReactantByproductMatrix = np.column_stack(
            (
                self.proteinProstheticGroupMatrix[selected],
                self.proteinChaperoneMatrix[selected],
                np.eye(len(selected)),
            )
        )
        self.speciesReactantMatrix = np.maximum(
            0, self.speciesReactantByproductMatrix
        )

    def initializeConstants(
        self, knowledgeBase=None, simulation=None, options=None, **kwargs
    ):
        if knowledgeBase is not None:
            self._initialize_from_kb(knowledgeBase, simulation, options)
        required = (
            "proteinProstheticGroupMatrix",
            "proteinChaperoneMatrix",
            "monomerCompartments",
            "complexCompartments",
            "unfoldedMonomerCompartmentIndexs",
            "unfoldedComplexCompartmentIndexs",
            "foldedMonomerCompartmentIndexs",
            "foldedComplexCompartmentIndexs",
            "monomerIndexs_folding",
            "complexIndexs_folding",
            "monomerIndexs_notFolding",
            "complexIndexs_notFolding",
            "monomerComplexIndexs_folded",
            "speciesIndexs_monomers",
            "speciesIndexs_complexs",
            "speciesReactantByproductMatrix",
            "speciesReactantMatrix",
        )
        missing = [x for x in required if not hasattr(self, x)]
        if missing:
            raise NotImplementedError(
                "initializeConstants requires initialized knowledge-base/simulation snapshot fields: "
                + ", ".join(missing)
            )

    @staticmethod
    def _idx(values):
        return np.asarray(values, dtype=int).ravel() - 1

    @staticmethod
    def _field(obj, key):
        return obj[key] if isinstance(obj, dict) else getattr(obj, key)

    @staticmethod
    def _assign(obj, key, value):
        if isinstance(obj, dict):
            obj[key] = value
        else:
            setattr(obj, key, value)

    def allocateMemoryForState(self, numTimePoints=1):
        nm, nc = len(self.unfoldedMonomerWholeCellModelIDs), len(
            self.unfoldedComplexWholeCellModelIDs
        )
        fm, fc = len(self.foldedMonomerWholeCellModelIDs), len(
            self.foldedComplexWholeCellModelIDs
        )
        shape = ((nm, 1), (nc, 1), (fm, 1), (fc, 1))
        vals = [
            (
                np.zeros(s)
                if numTimePoints == 1
                else np.zeros((*s, numTimePoints))
            )
            for s in shape
        ]
        (
            self.unfoldedMonomers,
            self.unfoldedComplexs,
            self.foldedMonomers,
            self.foldedComplexs,
        ) = vals

    def copyFromState(self):
        super().copyFromState()
        for typ, inds, comps, dest in (
            (
                "monomer",
                "processedIIIndexs",
                "monomerCompartments",
                "unfoldedMonomers",
            ),
            (
                "complex",
                "nascentIndexs",
                "complexCompartments",
                "unfoldedComplexs",
            ),
            (
                "monomer",
                "foldedIndexs",
                "monomerCompartments",
                "foldedMonomers",
            ),
            (
                "complex",
                "matureIndexs",
                "complexCompartments",
                "foldedComplexs",
            ),
        ):
            obj = getattr(self, typ)
            counts = np.asarray(self._field(obj, "counts"))
            gi = self._idx(self._field(obj, inds))
            ci = self._idx(getattr(self, comps))
            value = counts[gi, ci] if counts.ndim == 2 else counts[gi, ci, :]
            setattr(
                self,
                dest,
                (
                    np.asarray(value).reshape((-1, 1))
                    if counts.ndim == 2
                    else value
                ),
            )

    def copyToState(self):
        super().copyToState()
        for typ, inds, comps, src in (
            (
                "monomer",
                "processedIIIndexs",
                "monomerCompartments",
                "unfoldedMonomers",
            ),
            (
                "complex",
                "nascentIndexs",
                "complexCompartments",
                "unfoldedComplexs",
            ),
            (
                "monomer",
                "foldedIndexs",
                "monomerCompartments",
                "foldedMonomers",
            ),
            (
                "complex",
                "matureIndexs",
                "complexCompartments",
                "foldedComplexs",
            ),
        ):
            obj = getattr(self, typ)
            counts = np.asarray(self._field(obj, "counts"))
            gi = self._idx(self._field(obj, inds))
            ci = self._idx(getattr(self, comps))
            val = np.asarray(getattr(self, src))
            counts[gi, ci] = val.reshape(-1) if counts.ndim == 2 else val
            self._assign(obj, "counts", counts)

    def calcResourceRequirements_LifeCycle(self, *args):
        states = args[-1]
        prod = np.r_[
            np.asarray(self._field(states, "monomerProductions")).ravel(
                order="F"
            ),
            np.asarray(self._field(states, "complexProductions")).ravel(
                order="F"
            ),
        ]
        bm = np.asarray(self.proteinProstheticGroupMatrix).T @ prod
        min_enzyme_expression = np.zeros(
            len(self.enzymeWholeCellModelIDs), dtype=float
        )
        enzyme_ids = list(self.enzymeWholeCellModelIDs)
        for model_id in (
            "MG_238_MONOMER",  # trigger factor
            "MG_305_MONOMER",  # DnaK
            "MG_019_DIMER",  # DnaJ
            "MG_201_DIMER",  # GrpE
            "MG_392_393_21MER",  # GroEL/ES
        ):
            try:
                min_enzyme_expression[enzyme_ids.index(model_id)] = 2.0
            except ValueError as exc:
                raise ValueError(
                    f"ProteinFolding enzyme list is missing required enzyme {model_id}"
                ) from exc
        return (
            bm,
            np.zeros_like(
                np.asarray(self.substrateWholeCellModelIDs, dtype=object),
                dtype=float,
            ),
            min_enzyme_expression,
            np.full(len(self.enzymeWholeCellModelIDs), np.inf),
        )

    def initializeState(self):
        self.foldedComplexs = np.asarray(self.foldedComplexs) + np.asarray(
            self.unfoldedComplexs
        )
        self.unfoldedComplexs = np.zeros_like(self.unfoldedComplexs)

    def calcResourceRequirements_Current(self):
        proteins = np.r_[
            np.asarray(self.unfoldedMonomers).ravel(),
            np.asarray(self.unfoldedComplexs).ravel(),
        ]
        enz = np.asarray(self.enzymes)
        if np.all(enz):
            available = proteins
        else:
            ch = np.asarray(self.proteinChaperoneMatrix)
            limits = np.divide(
                enz.reshape(1, -1),
                ch,
                out=np.full(ch.shape, np.inf),
                where=ch != 0,
            )
            available = proteins * (np.min(limits, axis=1) > 0)
        return np.asarray(self.proteinProstheticGroupMatrix).T @ available

    def evolveState(self):
        um, uc, fm, fc = (
            np.asarray(getattr(self, x)) for x in self._LOCAL_STATES
        )
        mi, ci = self._idx(self.monomerIndexs_notFolding), self._idx(
            self.complexIndexs_notFolding
        )
        fc[ci] += uc[ci]
        uc[ci] = 0
        if not np.any(um) and not np.any(uc):
            return
        pg = np.asarray(self.proteinProstheticGroupMatrix)
        ch = np.asarray(self.proteinChaperoneMatrix)
        matrix = np.asarray(self.speciesReactantByproductMatrix)
        react = np.asarray(self.speciesReactantMatrix)
        folded = self._idx(self.monomerComplexIndexs_folded)
        n = len(folded)
        mi, cfi = self._idx(self.monomerIndexs_folding), self._idx(
            self.complexIndexs_folding
        )
        enzymes = np.asarray(self.enzymes).ravel()
        enzyme_capacity = np.where(enzymes > 0, np.inf, 0.0)
        species = np.maximum(
            0,
            np.r_[
                np.asarray(self.substrates).ravel(),
                enzyme_capacity,
                um[mi].ravel(),
                uc[cfi].ravel(),
            ],
        )
        water = int(np.asarray(self.substrateIndexs_water).ravel()[0]) - 1
        hydrogen = (
            int(np.asarray(self.substrateIndexs_hydrogen).ravel()[0]) - 1
        )
        flux = np.zeros(matrix.shape[0])
        while True:
            limits = np.divide(
                species[None, :],
                react,
                out=np.full(react.shape, np.inf, dtype=float),
                where=react != 0,
            )
            limits[:, [water, hydrogen]] = np.nan
            limits = np.nanmin(limits, axis=1)
            limits[~np.isfinite(limits) | (limits < 1)] = 0
            if not np.any(limits):
                break
            ix = (
                int(
                    np.asarray(
                        self.randStream.randsample(n, 1, True, limits)
                    ).ravel()[0]
                )
                - 1
            )
            flux[ix] += 1
            species -= matrix[ix]
        self.substrates = np.asarray(self.substrates) - (
            pg[folded, :].T @ flux
        ).reshape(np.asarray(self.substrates).shape)
        sm, sc = self._idx(self.speciesIndexs_monomers), self._idx(
            self.speciesIndexs_complexs
        )
        um[mi] -= flux[sm].reshape(um[mi].shape)
        fm[mi] += flux[sm].reshape(fm[mi].shape)
        foldc = self._idx(self.complexIndexs_folding)
        uc[foldc] -= flux[sc].reshape(uc[foldc].shape)
        fc[foldc] += flux[sc].reshape(fc[foldc].shape)

    def getDryWeight(self):
        base = np.asarray(super().getDryWeight())
        mon, comp = self.monomer, self.complex
        mwm, mwc = np.asarray(
            self._field(mon, "molecularWeights")
        ), np.asarray(self._field(comp, "molecularWeights"))
        pi, fi = self._idx(self._field(mon, "processedIIIndexs")), self._idx(
            self._field(mon, "foldedIndexs")
        )
        ni, mi = self._idx(self._field(comp, "nascentIndexs")), self._idx(
            self._field(comp, "matureIndexs")
        )

        def weighted(weights, indices, counts):
            # Exported vectors retain MATLAB's row/column orientation; contract
            # the species axis explicitly and preserve any time axis.
            w = np.asarray(weights).ravel(order="F")[indices]
            c = np.asarray(counts)
            return np.tensordot(w, c, axes=(0, 0))

        extra = (
            weighted(mwm, pi, self.unfoldedMonomers)
            + weighted(mwm, fi, self.foldedMonomers)
            + weighted(mwc, ni, self.unfoldedComplexs)
            + weighted(mwc, mi, self.foldedComplexs)
        )
        return base + extra / float(
            getattr(self, "nAvogadro", self._N_AVOGADRO)
        )
