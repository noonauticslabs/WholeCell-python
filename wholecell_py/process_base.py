"""Runtime state interface for WholeCell processes.

Public properties are loaded from cached MATLAB snapshots by the exporter. MATLAB
index arrays deliberately remain 1-based; conversion happens only at indexing.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from .core import copy_array


def _get(obj: Any, name: str) -> Any:
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def _set(obj: Any, name: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[name] = value
    else:
        setattr(obj, name, value)


def _arr(x: Any) -> np.ndarray:
    return np.asarray(x)


def _cell_text(value):
    if isinstance(value, str):
        return value
    return "".join(str(v) for v in np.asarray(value).ravel(order="F"))


class Process:
    """Base process object with MATLAB-compatible state copy contracts."""

    def allocateMemoryForState(self, numTimePoints=1):
        """Allocate local pools as in MATLAB Process.allocateMemoryForState."""
        if not isinstance(numTimePoints, int) or numTimePoints < 0:
            raise ValueError("numTimePoints must be a nonnegative integer")
        for prefix, field, ids in (
            ("stimulus", "stimuli", "stimuliWholeCellModelIDs"),
            ("substrate", "substrates", "substrateWholeCellModelIDs"),
            ("enzyme", "enzymes", "enzymeWholeCellModelIDs"),
        ):
            compartments = max(
                np.asarray(
                    getattr(self, prefix + kind + "GlobalIndexs")
                ).shape[1]
                for kind in (
                    "Stimulus",
                    "Metabolite",
                    "RNA",
                    "Monomer",
                    "Complex",
                )
            )
            shape = (np.size(getattr(self, ids)), compartments)
            if numTimePoints != 1:
                shape += (numTimePoints,)
            setattr(self, field, np.zeros(shape))
        self.boundEnzymes = np.zeros_like(self.enzymes)

    def __init__(
        self, wholeCellModelID: str = "", name: str = "", **properties: Any
    ):
        self.wholeCellModelID, self.name = wholeCellModelID, name
        self.initialized = False
        from .randstream import RandStream

        self.randStream = RandStream("mcg16807")
        self.stepSizeSec, self.verbosity, self.seed = 1, 1, None
        self.compartment = self.gene = self.states = None
        self.stimulus = self.metabolite = self.rna = None
        self.monomer = self.complex = self.geometry = None
        for key, value in properties.items():
            setattr(self, key, value)

    def storeObjectReferences(self, simulation: Any) -> None:
        self.compartment, self.gene = _get(simulation, "compartment"), _get(
            simulation, "gene"
        )
        statefn = getattr(simulation, "state", None)

        def state(name: str):
            if callable(statefn):
                return statefn(name)
            states = _get(simulation, "states")
            normalized = {str(k).lower(): v for k, v in states.items()}
            # State names are the MATLAB simulation.state() names. Do not
            # substitute short process attributes (e.g. monomer) for keys.
            return normalized[name.lower()]

        self.geometry, self.stimulus = state("Geometry"), state("Stimulus")
        self.metabolite, self.rna = state("Metabolite"), state("Rna")
        self.monomer, self.complex = state("ProteinMonomer"), state(
            "ProteinComplex"
        )
        self.states = [
            self.geometry,
            self.metabolite,
            self.rna,
            self.monomer,
            self.complex,
        ]

    @staticmethod
    def _copyFromStateHelper(
        gblState: Any, gblIdxs: Any, cmpIdxs: Any
    ) -> np.ndarray:
        from .sparsemat import SparseMat

        if isinstance(gblState, SparseMat):
            g, c = _arr(gblIdxs), _arr(cmpIdxs)
            if g.shape != c.shape:
                raise ValueError("global and compartment index shapes differ")
            size = np.asarray(gblState.siz, dtype=int).ravel()
            times = int(size[2]) if len(size) > 2 else 1
            out = np.empty((*g.shape, times), dtype=gblState.vals.dtype)
            for t in range(times):
                columns = [g.ravel(order="F"), c.ravel(order="F")]
                if len(size) > 2:
                    columns.append(np.full(g.size, t + 1))
                out[..., t] = gblState.get_coords(
                    np.column_stack(columns)
                ).reshape(g.shape, order="F")
            return out
        a, g, c = _arr(gblState), _arr(gblIdxs), _arr(cmpIdxs)
        if g.shape != c.shape:
            raise ValueError("global and compartment index shapes differ")
        if g.size == 0:
            return np.zeros(
                (*g.shape, a.shape[2] if a.ndim > 2 else 1), dtype=a.dtype
            )
        if a.ndim < 3:
            a = a.reshape((*a.shape, 1))
        out = np.empty((*g.shape, a.shape[2]), dtype=a.dtype)
        # Index maps are commonly exported as float64 MATLAB arrays. Convert
        # each map once per gather, rather than allocating the same integer map
        # for every time plane.
        gi, ci = (
            g.astype(np.intp, copy=False) - 1,
            c.astype(np.intp, copy=False) - 1,
        )
        for t in range(a.shape[2]):
            out[..., t] = a[gi, ci, t]
        return out

    def copyFromStateHelper(
        self, gblState: Any, gblIdxs: Any, cmpIdxs: Any
    ) -> np.ndarray:
        return self._copyFromStateHelper(gblState, gblIdxs, cmpIdxs)

    def _gather(
        self,
        state: Any,
        prop: str,
        globals_: str,
        comps: str,
        locals_: str,
        target: np.ndarray,
    ) -> None:
        gi, ci, li = (
            _arr(getattr(self, x)) for x in (globals_, comps, locals_)
        )
        if li.size == 0:
            return
        src = _arr(_get(state, prop))
        if src.ndim == 2:
            src = src.reshape((*src.shape, 1))
        if src.shape[2] == 1:
            # MATLAB linear indexing, column-major, for single-time fast path.
            target[li.astype(int).ravel() - 1, :] = src.ravel(order="F")[
                gi.astype(int).ravel() - 1
            ].reshape(gi.shape, order="F")
        else:
            target[li.astype(int).ravel() - 1, :, :] = (
                self.copyFromStateHelper(src, gi, ci)
            )

    def copyFromState(self) -> None:
        self.stimuli = self.copyStimuliFromState(
            _get(self.stimulus, "values"),
            _get(self.metabolite, "counts"),
            _get(self.rna, "counts"),
            _get(self.monomer, "counts"),
            _get(self.complex, "counts"),
        )
        self.substrates = self.copySubstratesFromState(
            _get(self.stimulus, "values"),
            _get(self.metabolite, "counts"),
            _get(self.rna, "counts"),
            _get(self.monomer, "counts"),
            _get(self.complex, "counts"),
        )
        self.enzymes, self.boundEnzymes = self.copyEnzymesFromState(
            _get(self.stimulus, "values"),
            _get(self.metabolite, "counts"),
            _get(self.rna, "counts"),
            _get(self.monomer, "counts"),
            _get(self.complex, "counts"),
        )

    def _copy_type(
        self, typ: str, arrays: tuple[Any, ...], bound: bool = False
    ):
        ids = getattr(
            self,
            (
                "stimuliWholeCellModelIDs"
                if typ == "stimulus"
                else typ + "WholeCellModelIDs"
            ),
        )
        time = (
            _arr(
                arrays[
                    0 if typ == "stimulus" else 1 if typ == "substrate" else 4
                ]
            ).shape[2]
            if _arr(
                arrays[
                    0 if typ == "stimulus" else 1 if typ == "substrate" else 4
                ]
            ).ndim
            > 2
            else 1
        )
        ncomp_field = {
            "stimulus": "stimulusStimulusGlobalCompartmentIndexs",
            "substrate": "substrateMetaboliteGlobalCompartmentIndexs",
            "enzyme": "enzymeBoundMonomerGlobalCompartmentIndexs",
        }[typ]
        ncomp = _arr(getattr(self, ncomp_field)).shape[1]
        out = np.zeros(
            (np.asarray(ids).size, ncomp)
            if time == 1
            else (np.asarray(ids).size, ncomp, time)
        )
        bout = np.zeros_like(out)
        kinds = (
            ("Stimulus", arrays[0], "values"),
            ("Metabolite", arrays[1], "counts"),
            ("RNA", arrays[2], "counts"),
            ("Monomer", arrays[3], "counts"),
            ("Complex", arrays[4], "counts"),
        )
        # The single-time MATLAB code uses linear global indices; multitime gathers component/compartment coordinates.
        for kind, src, prop in kinds:
            prefix = typ + kind
            li = _arr(getattr(self, prefix + "LocalIndexs"))
            gi = _arr(getattr(self, prefix + "GlobalIndexs"))
            ci = _arr(getattr(self, prefix + "CompartmentIndexs"))
            linear = _arr(getattr(self, prefix + "GlobalCompartmentIndexs"))
            if not li.size:
                continue
            # Reuse converted maps across the primary and bound gather paths.
            # State-derived mature/bound maps stay live: side effects update
            # them during a step, so these are deliberately local only.
            li0 = li.astype(np.intp, copy=False).ravel() - 1
            gi0 = gi.astype(np.intp, copy=False)
            linear0 = linear.astype(np.intp, copy=False).ravel(order="F") - 1
            if time == 1:
                out[li0, :] = (
                    _arr(src)
                    .ravel(order="F")[linear0]
                    .reshape(linear.shape, order="F")
                )
                if typ == "enzyme" and kind in ("RNA", "Monomer", "Complex"):
                    bound_map = _arr(
                        getattr(
                            self,
                            typ + "Bound" + kind + "GlobalCompartmentIndexs",
                        )
                    )
                    bound_linear = (
                        bound_map.astype(np.intp, copy=False).ravel(order="F")
                        - 1
                    )
                    bout[li0, :] = (
                        _arr(src)
                        .ravel(order="F")[bound_linear]
                        .reshape(bound_map.shape, order="F")
                    )
            else:
                g = gi
                if kind in ("RNA", "Monomer", "Complex"):
                    state = {
                        "RNA": self.rna,
                        "Monomer": self.monomer,
                        "Complex": self.complex,
                    }[kind]
                    g = _arr(_get(state, "matureIndexs")).ravel(order="F")[
                        gi0 - 1
                    ]
                out[li0, :, :] = self.copyFromStateHelper(src, g, ci)
                if typ == "enzyme" and kind in ("RNA", "Monomer", "Complex"):
                    bi = _arr(_get(state, "boundIndexs")).ravel(order="F")[
                        gi0 - 1
                    ]
                    bout[li0, :, :] = self.copyFromStateHelper(src, bi, ci)
        return (out, bout) if typ == "enzyme" else out

    def copyStimuliFromState(
        self,
        stimulusValues,
        metaboliteCounts,
        rnaCounts,
        monomerCounts,
        complexCounts,
    ):
        return self._copy_type(
            "stimulus",
            (
                stimulusValues,
                metaboliteCounts,
                rnaCounts,
                monomerCounts,
                complexCounts,
            ),
        )

    def copySubstratesFromState(
        self,
        stimulusValues,
        metaboliteCounts,
        rnaCounts,
        monomerCounts,
        complexCounts,
    ):
        return self._copy_type(
            "substrate",
            (
                stimulusValues,
                metaboliteCounts,
                rnaCounts,
                monomerCounts,
                complexCounts,
            ),
        )

    def copyEnzymesFromState(
        self,
        stimulusValues,
        metaboliteCounts,
        rnaCounts,
        monomerCounts,
        complexCounts,
    ):
        return self._copy_type(
            "enzyme",
            (
                stimulusValues,
                metaboliteCounts,
                rnaCounts,
                monomerCounts,
                complexCounts,
            ),
        )

    def copyToStateHelper(
        self,
        state: Any,
        statePropertyName: str,
        lclPropertyName: str,
        stateIdxs: Any,
        cmpIdxs: Any,
        lclIdxs: Any,
    ) -> None:
        si, ci, li = (
            _arr(stateIdxs).astype(int),
            _arr(cmpIdxs).astype(int),
            _arr(lclIdxs).astype(int),
        )
        if li.size == 0:
            return
        dest = _arr(_get(state, statePropertyName))
        if dest.ndim == 2:
            dest = dest.reshape((*dest.shape, 1))
        local = _arr(getattr(self, lclPropertyName))
        if local.ndim == 2:
            local = local[..., None]
        for t in range(dest.shape[2]):
            linear = (si - 1) + (ci - 1) * dest.shape[0]
            rr, cc = np.unravel_index(
                linear.ravel(order="F"), dest.shape[:2], order="F"
            )
            dest[rr, cc, t] = local[li.ravel().astype(int) - 1, :, t].ravel(
                order="F"
            )
        _set(state, statePropertyName, dest)

    def copyToState(self) -> None:
        # Explicit MATLAB mapping; mutates referenced state arrays after the full local values exist.
        for typ, local in (("enzyme", "enzymes"), ("substrate", "substrates")):
            value = getattr(self, local, None)
            if value is None or not _arr(value).size:
                continue
            for kind, state, prop in (
                ("Stimulus", self.stimulus, "values"),
                ("Metabolite", self.metabolite, "counts"),
                ("RNA", self.rna, "counts"),
                ("Monomer", self.monomer, "counts"),
                ("Complex", self.complex, "counts"),
            ):
                prefix = typ + kind
                li = _arr(getattr(self, prefix + "LocalIndexs"))
                gi = _arr(getattr(self, prefix + "GlobalIndexs"))
                ci = _arr(getattr(self, prefix + "CompartmentIndexs"))
                linear = _arr(
                    getattr(self, prefix + "GlobalCompartmentIndexs")
                )
                if not li.size:
                    continue
                li0 = li.astype(np.intp, copy=False).ravel() - 1
                gi0 = gi.astype(np.intp, copy=False)
                ci0 = ci.astype(np.intp, copy=False) - 1
                linear0 = (
                    linear.astype(np.intp, copy=False).ravel(order="F") - 1
                )
                dest = _arr(_get(state, prop))
                original_shape = dest.shape
                src = _arr(value)
                if dest.ndim == 2:
                    dest = dest.reshape((*dest.shape, 1))
                if dest.shape[2] == 1:
                    rr, cc = np.unravel_index(
                        linear0, dest.shape[:2], order="F"
                    )
                    dest[rr, cc, 0] = src[li0, :].ravel(order="F")
                else:
                    gi2 = gi0
                    if kind in ("RNA", "Monomer", "Complex"):
                        obj = {
                            "RNA": self.rna,
                            "Monomer": self.monomer,
                            "Complex": self.complex,
                        }[kind]
                        gi2 = _arr(_get(obj, "matureIndexs")).ravel(order="F")[
                            gi0 - 1
                        ]
                    gi2 = np.asarray(gi2).astype(np.intp, copy=False) - 1
                    for t in range(dest.shape[2]):
                        dest[gi2, ci0, t] = src[li0, :, t]
                _set(state, prop, dest.reshape(original_shape))
                if typ == "enzyme" and kind in ("RNA", "Monomer", "Complex"):
                    bound = _arr(self.boundEnzymes)
                    if dest.shape[2] == 1:
                        bound_linear = _arr(
                            getattr(
                                self,
                                typ
                                + "Bound"
                                + kind
                                + "GlobalCompartmentIndexs",
                            )
                        )
                        bound_linear0 = (
                            bound_linear.astype(np.intp, copy=False).ravel(
                                order="F"
                            )
                            - 1
                        )
                        rr, cc = np.unravel_index(
                            bound_linear0, dest.shape[:2], order="F"
                        )
                        dest[rr, cc, 0] = bound[li0, :].ravel(order="F")
                    else:
                        obj = {
                            "RNA": self.rna,
                            "Monomer": self.monomer,
                            "Complex": self.complex,
                        }[kind]
                        bi = _arr(_get(obj, "boundIndexs")).ravel(order="F")[
                            gi0 - 1
                        ]
                        bi0 = np.asarray(bi).astype(np.intp, copy=False) - 1
                        for t in range(dest.shape[2]):
                            dest[bi0, ci0, t] = bound[li0, :, t]
                _set(state, prop, dest.reshape(original_shape))

    def substrateToMetabolite(self, i):
        return self._convert(
            i,
            "substrateMetaboliteLocalIndexs",
            "substrateMetaboliteGlobalIndexs",
        )

    def metaboliteToSubstrate(self, i):
        return self._convert(
            i,
            "substrateMetaboliteGlobalIndexs",
            "substrateMetaboliteLocalIndexs",
        )

    def enzymeToMonomer(self, i):
        return self._convert(
            i, "enzymeMonomerLocalIndexs", "enzymeMonomerGlobalIndexs"
        )

    def monomerToEnzyme(self, i):
        return self._convert(
            i, "enzymeMonomerGlobalIndexs", "enzymeMonomerLocalIndexs"
        )

    def enzymeToComplex(self, i):
        return self._convert(
            i, "enzymeComplexLocalIndexs", "enzymeComplexGlobalIndexs"
        )

    def complexToEnzyme(self, i):
        return self._convert(
            i, "enzymeComplexGlobalIndexs", "enzymeComplexLocalIndexs"
        )

    def _convert(self, values, source, target):
        s, t = (
            _arr(getattr(self, source)).ravel(),
            _arr(getattr(self, target)).ravel(),
        )
        lookup = {int(v): int(ti) for v, ti in zip(s, t)}
        v = np.asarray(values)
        result = [lookup[int(x)] for x in v.ravel() if int(x) in lookup]
        return np.asarray(result, dtype=int).reshape(-1)

    def getDryWeight(self):
        result = None
        for counts_name, weights_name in (
            ("substrates", "substrateMolecularWeights"),
            ("enzymes", "enzymeMolecularWeights"),
            ("boundEnzymes", "enzymeMolecularWeights"),
        ):
            counts = np.asarray(getattr(self, counts_name))
            weights = np.asarray(getattr(self, weights_name)).ravel(order="F")
            if counts.ndim == 1:
                counts = counts.reshape(-1, 1)
            value = np.tensordot(weights, counts.sum(axis=1), axes=(0, 0))
            result = value if result is None else result + value
        result = np.asarray(result) / 6.0221417930e23
        return result.reshape(
            (1, 1) if result.size == 1 else (1, 1, result.size)
        )

    @property
    def dryWeight(self):
        value = self.getDryWeight()
        for state in () if self.states is None else self.states:
            value = value + state.dryWeight
        return value

    def initializeConstantsHelper(
        self, wholeCellModelIDs, retainCompartments=False
    ):
        """Process.m mapping helper; all returned indices remain one based."""
        ids = np.asarray(wholeCellModelIDs, dtype=object).ravel(order="F")
        n = len(ids)
        names = np.empty((n, 1), dtype=object)
        global_indices = np.zeros((n, 1), dtype=int)
        weights = np.zeros((n, 1))
        kinds = np.full(n, -1, dtype=int)
        compartments = np.zeros(n, dtype=int)
        states = [
            self.stimulus,
            self.metabolite,
            self.rna,
            self.monomer,
            self.complex,
        ]
        catalogs = []
        for kind, state in enumerate(states):
            all_ids = np.asarray(
                _get(state, "wholeCellModelIDs"), dtype=object
            ).ravel(order="F")
            mature = (
                np.asarray(_get(state, "matureIndexs"), dtype=int).ravel(
                    order="F"
                )
                if kind >= 2
                else np.arange(1, len(all_ids) + 1)
            )
            lookup = {}
            for j, identifier in enumerate(all_ids[mature - 1]):
                lookup.setdefault(_cell_text(identifier), j)
            catalogs.append((all_ids, mature, lookup))
        for i, identifier in enumerate(ids):
            identifier = _cell_text(identifier)
            for kind, (state, (_, mature, lookup)) in enumerate(
                zip(states, catalogs)
            ):
                if identifier not in lookup:
                    continue
                j = lookup[identifier]
                absolute = mature[j] - 1
                kinds[i] = kind
                global_indices[i, 0] = j + 1
                names[i, 0] = np.asarray(
                    _get(state, "names"), dtype=object
                ).ravel(order="F")[absolute]
                if kind == 0:
                    compartment = _get(self.compartment, "extracellularIndexs")
                elif kind <= 2:
                    compartment = _get(self.compartment, "cytosolIndexs")
                    if kind == 1 and j + 1 in np.asarray(
                        _get(state, "hydrophobicIndexs")
                    ):
                        compartment = _get(self.compartment, "membraneIndexs")
                else:
                    compartment = np.asarray(
                        _get(state, "compartments")
                    ).ravel(order="F")[absolute]
                compartments[i] = int(np.asarray(compartment).item())
                if kind:
                    weights[i, 0] = np.asarray(
                        _get(state, "molecularWeights")
                    ).ravel(order="F")[absolute]
                break
            else:
                raise ValueError(
                    f"Process:invalidIDs: {identifier!r} not found"
                )
        outputs, bound = [names], []
        n_cmp = int(np.asarray(_get(self.compartment, "count")).item())
        for kind, (state, (all_ids, mature, _)) in enumerate(
            zip(states, catalogs)
        ):
            local = (np.flatnonzero(kinds == kind) + 1).reshape(-1, 1)
            indices = global_indices[local[:, 0] - 1].copy()
            if retainCompartments:
                indices = np.repeat(indices, n_cmp, axis=1)
                cmp = np.tile(np.arange(1, n_cmp + 1), (len(local), 1))
            else:
                cmp = compartments[local[:, 0] - 1].reshape(-1, 1)
            linear = mature[indices - 1] + (cmp - 1) * len(all_ids)
            outputs.extend((local, indices, cmp, linear))
            if kind >= 2:
                bound_rows = np.asarray(
                    _get(state, "boundIndexs"), dtype=int
                ).ravel(order="F")
                bound.append(
                    bound_rows[indices - 1] + (cmp - 1) * len(all_ids)
                )
        return tuple(outputs + bound + [global_indices, weights])

    def _initialize_component_mapping(self, typ, ids, retain=False):
        values = self.initializeConstantsHelper(ids, retain)
        setattr(
            self,
            "stimuliNames" if typ == "stimulus" else typ + "Names",
            values[0],
        )
        for k, category in enumerate(
            ("Stimulus", "Metabolite", "RNA", "Monomer", "Complex")
        ):
            for offset, suffix in enumerate(
                (
                    "LocalIndexs",
                    "GlobalIndexs",
                    "CompartmentIndexs",
                    "GlobalCompartmentIndexs",
                )
            ):
                setattr(
                    self, typ + category + suffix, values[1 + 4 * k + offset]
                )
        if typ == "enzyme":
            for k, category in enumerate(("RNA", "Monomer", "Complex")):
                setattr(
                    self,
                    "enzymeBound" + category + "GlobalCompartmentIndexs",
                    values[21 + k],
                )
        setattr(self, typ + "GlobalIndexs", values[24])
        if typ != "stimulus":
            setattr(self, typ + "MolecularWeights", values[25])

    def initializeConstants(self, knowledgeBase, simulation, options=None):
        """Initialize parameters and molecule maps from the source KB contract."""
        options = options if isinstance(options, dict) else {}
        records = np.asarray(
            _get(knowledgeBase, "processes"), dtype=object
        ).ravel(order="F")
        record = next(
            (
                p
                for p in records
                if _get(p, "wholeCellModelID") == self.wholeCellModelID
            ),
            None,
        )
        if record is None:
            raise ValueError(
                f"Process {self.wholeCellModelID!r} missing from knowledge base"
            )
        parameters = np.asarray(
            _get(record, "parameters"), dtype=object
        ).ravel(order="F")
        from .kb_parameters import initialize_parameters

        initialize_parameters(self, parameters)
        for name in ("stepSizeSec", "verbosity", "seed"):
            setattr(self, name, _get(simulation, name))
        for typ, attr, option in (
            (
                "stimulus",
                "stimuliWholeCellModelIDs",
                "retainStimuliCompartments",
            ),
            (
                "substrate",
                "substrateWholeCellModelIDs",
                "retainSubstrateCompartments",
            ),
            ("enzyme", "enzymeWholeCellModelIDs", "retainEnzymeCompartments"),
        ):
            self._initialize_component_mapping(
                typ, getattr(self, attr), options.get(option, False)
            )
        self.initialized = True

    def initializeConstants_overrideSubstrates(self, wholeCellModelIDs):
        self.substrateWholeCellModelIDs = wholeCellModelIDs
        self._initialize_component_mapping(
            "substrate",
            wholeCellModelIDs,
            np.asarray(getattr(self, "substrates", np.empty((0, 1)))).shape[1]
            > 1,
        )

    def initializeConstants_overrideEnzymes(self, wholeCellModelIDs):
        self.enzymeWholeCellModelIDs = wholeCellModelIDs
        self._initialize_component_mapping(
            "enzyme",
            wholeCellModelIDs,
            np.asarray(getattr(self, "enzymes", np.empty((0, 1)))).shape[1]
            > 1,
        )

    def componentIndexs(self, wholeCellModelIDs, type):
        attr = type + "WholeCellModelIDs"
        if type == "stimulus" and not hasattr(self, attr):
            attr = "stimuliWholeCellModelIDs"
        lookup = {}
        for i, key in enumerate(
            np.asarray(getattr(self, attr), dtype=object).ravel(order="F")
        ):
            lookup.setdefault(_cell_text(key), i + 1)
        original_query = np.asarray(wholeCellModelIDs, dtype=object)
        query = np.asarray(
            [_cell_text(x) for x in original_query.ravel(order="F")],
            dtype=object,
        )
        missing = [v for v in query.ravel(order="F") if v not in lookup]
        if missing:
            raise ValueError(f"Process:error: {missing!r} not found")
        result = np.asarray(
            [lookup[v] for v in query.ravel(order="F")], dtype=int
        )
        if original_query.ndim == 2 and original_query.shape[0] == 1:
            return result.reshape(1, -1)
        return result.reshape(-1, 1)

    def stimulusIndexs(self, wholeCellModelIDs):
        return self.componentIndexs(wholeCellModelIDs, "stimulus")

    def substrateIndexs(self, wholeCellModelIDs):
        return self.componentIndexs(wholeCellModelIDs, "substrate")

    def enzymeIndexs(self, wholeCellModelIDs):
        return self.componentIndexs(wholeCellModelIDs, "enzyme")

    def initializeState(self, *args, **kwargs):
        raise NotImplementedError("subclasses must implement initializeState")

    def evolveState(self, *args, **kwargs):
        raise NotImplementedError("subclasses must implement evolveState")
