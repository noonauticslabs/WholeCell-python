"""ProteinActivation.evolveState and evaluateActivationRules port.

Source: WholeCell/src/+edu/+stanford/+covert/+cell/+sim/+process/ProteinActivation.m
Arrays use MATLAB's (species, compartment) ordering and are always 2-D.
"""

from __future__ import annotations

import ast
from typing import Any

import numpy as np

from ..process_base import Process, _get, _set


def _eval_rule(rule: str, variables: dict[str, float]) -> bool:
    """Evaluate the documented activation-rule subset without Python eval."""
    # Keep unary negation at MATLAB's high precedence using Python's
    # invert AST node, evaluated as logical NOT below. Python `not` binds
    # below comparisons and incorrectly changes ~A > 1 into ~(A > 1).
    translated = rule.replace("~=", "!=").replace("!", "~")
    translated = translated.replace("~=", "!=")
    translated = translated.replace("||", "|").replace("&&", "&")
    translated = translated.replace("|", " or ").replace("&", " and ").strip()
    tree = ast.parse(translated, mode="eval")

    def visit(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(
            node.value, (int, float, bool)
        ):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise KeyError(
                    f"activation rule references unknown regulator {node.id!r}"
                )
            return variables[node.id]
        if isinstance(node, ast.UnaryOp):
            value = visit(node.operand)
            if isinstance(node.op, (ast.Not, ast.Invert)):
                return not bool(value)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub)
        ):
            a, b = visit(node.left), visit(node.right)
            return a + b if isinstance(node.op, ast.Add) else a - b
        if isinstance(node, ast.BoolOp) and isinstance(
            node.op, (ast.And, ast.Or)
        ):
            vals = [bool(visit(v)) for v in node.values]
            return all(vals) if isinstance(node.op, ast.And) else any(vals)
        if isinstance(node, ast.Compare):
            left = visit(node.left)
            for op, comp in zip(node.ops, node.comparators):
                right = visit(comp)
                ok = (
                    left < right
                    if isinstance(op, ast.Lt)
                    else (
                        left <= right
                        if isinstance(op, ast.LtE)
                        else (
                            left > right
                            if isinstance(op, ast.Gt)
                            else (
                                left >= right
                                if isinstance(op, ast.GtE)
                                else (
                                    left == right
                                    if isinstance(op, ast.Eq)
                                    else (
                                        left != right
                                        if isinstance(op, ast.NotEq)
                                        else None
                                    )
                                )
                            )
                        )
                    )
                )
                if ok is None:
                    raise ValueError(
                        f"unsupported activation comparison: {type(op).__name__}"
                    )
                # MATLAB evaluates chained relational operators left-to-right.
                left = ok
            return left
        raise ValueError(
            f"unsupported activation rule expression: {ast.dump(node)}"
        )

    return bool(visit(tree))


def _matlab_strings(values) -> list[str]:
    """Normalize snapshot MATLAB cellstr values, including char arrays."""
    result = []
    for value in np.asarray(values, dtype=object).ravel(order="F"):
        if isinstance(value, str):
            result.append(value)
        else:
            result.append(
                "".join(
                    str(char) for char in np.asarray(value).ravel(order="F")
                )
            )
    return result


def run_case(inputs: dict, parameters: dict) -> dict:
    """Apply one source evolveState call.

    Inputs: substrates, inactivated_substrates, stimuli, extracellular_index
    (zero-based), stimulus_local_indices (rows whose values are raw stimuli),
    geometry_volume. Parameters: activation_rules (per species), stimuli_ids,
    n_avogadro. Rule arrays map one-to-one to the substrate rows.
    """
    active = np.asarray(inputs["substrates"], dtype=float).copy()
    inactive = np.asarray(inputs["inactivated_substrates"], dtype=float).copy()
    stimuli = np.asarray(inputs["stimuli"], dtype=float)
    if active.ndim != 2 or inactive.shape != active.shape:
        raise ValueError(
            "substrates and inactivated_substrates must be matching 2-D arrays"
        )
    if stimuli.ndim != 2:
        raise ValueError(
            "stimuli must be a 2-D (regulator, compartment) array"
        )
    rules = list(parameters["activation_rules"])
    ids = list(parameters["stimuli_ids"])
    if len(rules) != active.shape[0] or len(ids) != stimuli.shape[0]:
        raise ValueError(
            "rule/species and regulator/stimuli dimensions must match"
        )
    ext = int(inputs["extracellular_index"])
    vol = float(inputs["geometry_volume"])
    na = float(parameters["n_avogadro"])
    if not (0 <= ext < stimuli.shape[1]) or vol == 0 or na == 0:
        raise ValueError(
            "invalid extracellular index, volume, or Avogadro constant"
        )
    values = stimuli[:, ext] / (na / 1000.0) / vol
    for idx in inputs.get("stimulus_local_indices", []):
        values[int(idx)] = stimuli[int(idx), ext]
    # MATLAB evaluateActivationRules constructs each workspace assignment
    # with num2str(value), whose text precision is part of replay semantics.
    # The parent can provide those exact strings from MATLAB. Without them,
    # use the requested .5g approximation explicitly rather than pretending
    # Python's float text matches MATLAB num2str.
    matlab_text = inputs.get("stimulus_text_values")
    if matlab_text is not None:
        if len(matlab_text) != len(values):
            raise ValueError(
                "stimulus_text_values must have one MATLAB num2str string per regulator"
            )
        values = np.asarray([float(v) for v in matlab_text], dtype=float)
    else:
        values = np.asarray(
            [float(format(float(v), ".5g")) for v in values], dtype=float
        )
    activated = np.zeros(active.shape, dtype=bool)
    for compartment in range(active.shape[1]):
        if not np.any(active[:, compartment]) and not np.any(
            inactive[:, compartment]
        ):
            continue
        variables = dict(zip(ids, map(float, values)))
        for species, rule in enumerate(rules):
            if active[species, compartment] or inactive[species, compartment]:
                activated[species, compartment] = _eval_rule(
                    str(rule), variables
                )
    active[activated] += inactive[activated]
    inactive[activated] = 0
    deactivated = ~activated
    inactive[deactivated] += active[deactivated]
    active[deactivated] = 0
    return {
        "substrates": active,
        "inactivated_substrates": inactive,
        "activated_substrates": activated,
    }


class ProteinActivation(Process):
    """State-linked port of the MATLAB ProteinActivation process.

    Snapshot properties and all index arrays retain MATLAB's one-based indexing.
    """

    _N_AVOGADRO = 6.0221417930e23

    def _avogadro(self):
        return float(getattr(self, "nAvogadro", self._N_AVOGADRO))

    def __init__(
        self,
        wholeCellModelID="Process_ProteinActivation",
        name="Protein Activation",
        **properties,
    ):
        super().__init__(wholeCellModelID, name, **properties)

    enzymeWholeCellModelIDs = ()

    def initializeConstants(
        self, knowledgeBase=None, simulation=None, options=None, **kwargs
    ):
        if knowledgeBase is None:
            if not hasattr(self, "activationRules"):
                raise NotImplementedError(
                    "activationRules require knowledge-base regulatory links"
                )
            return

        def field(obj, key):
            return obj[key] if isinstance(obj, dict) else getattr(obj, key)

        regulated = [
            p
            for category in ("proteinMonomers", "proteinComplexs")
            for p in field(knowledgeBase, category)
            if field(p, "activationRule") != ""
        ]
        self.substrateWholeCellModelIDs = np.asarray(
            [field(p, "wholeCellModelID") for p in regulated], dtype=object
        ).reshape(-1, 1)
        self.activationRules = np.asarray(
            [field(p, "activationRule").replace("!", "~") for p in regulated],
            dtype=object,
        ).reshape(-1, 1)
        regulators = set()
        for p in regulated:
            for category in (
                "stimuliRegulators",
                "metaboliteRegulators",
                "proteinMonomerRegulators",
                "proteinComplexRegulators",
            ):
                regulators.update(
                    field(r, "wholeCellModelID") for r in field(p, category)
                )
        self.stimuliWholeCellModelIDs = np.asarray(
            sorted(regulators), dtype=object
        ).reshape(-1, 1)
        Process.initializeConstants(
            self,
            knowledgeBase,
            simulation,
            {
                "retainStimuliCompartments": True,
                "retainSubstrateCompartments": True,
                "retainEnzymeCompartments": True,
            },
        )
        for category, state in (
            ("Monomer", self.monomer),
            ("Complex", self.complex),
        ):
            rows = np.asarray(
                field(state, "inactivatedIndexs"), dtype=int
            ).ravel(order="F")[
                getattr(self, "substrate" + category + "GlobalIndexs") - 1
            ]
            compartments = getattr(
                self, "substrate" + category + "CompartmentIndexs"
            )
            linear = rows + (compartments - 1) * np.size(
                field(state, "wholeCellModelIDs")
            )
            setattr(
                self,
                "inactivatedSubstrate" + category + "GlobalCompartmentIndexs",
                linear,
            )

    def storeObjectReferences(self, simulation):
        super().storeObjectReferences(simulation)

    def allocateMemoryForState(self, numTimePoints=1):
        shape = np.asarray(self.substrates).shape
        self.inactivatedSubstrates = (
            np.zeros((*shape, numTimePoints))
            if numTimePoints != 1
            else np.zeros(shape)
        )

    def copyFromState(self):
        super().copyFromState()
        local = np.asarray(self.inactivatedSubstrates)
        ntime = local.shape[2] if local.ndim > 2 else 1
        if ntime == 1:
            # MATLAB's one-time-point path indexes counts with the flattened
            # sub2ind map. NumPy must use column-major flattening for that map.
            self._copy_inactivated_single(
                self.monomer,
                self.inactivatedSubstrateMonomerGlobalCompartmentIndexs,
                self.substrateMonomerLocalIndexs,
            )
            self._copy_inactivated_single(
                self.complex,
                self.inactivatedSubstrateComplexGlobalCompartmentIndexs,
                self.substrateComplexLocalIndexs,
            )
            return
        self._copy_inactivated_multitime(
            self.monomer,
            self.inactivatedSubstrateMonomerGlobalCompartmentIndexs,
            self.substrateMonomerCompartmentIndexs,
            self.substrateMonomerLocalIndexs,
        )
        self._copy_inactivated_multitime(
            self.complex,
            self.inactivatedSubstrateComplexGlobalCompartmentIndexs,
            self.substrateComplexCompartmentIndexs,
            self.substrateComplexLocalIndexs,
        )

    def _copy_inactivated_multitime(self, state, globals_, comps, locals_):
        linear = np.asarray(globals_, int)
        li = np.asarray(locals_, int).ravel() - 1
        counts = np.asarray(_get(state, "counts"))
        rows, columns = np.unravel_index(
            linear.ravel(order="F") - 1, counts.shape[:2], order="F"
        )
        self.inactivatedSubstrates[li, :, :] = counts[
            rows, columns, :
        ].reshape((*linear.shape, counts.shape[2]), order="F")

    def _copy_inactivated_single(self, state, linear_indices, local_indices):
        li = np.asarray(local_indices, dtype=int).ravel() - 1
        if not li.size:
            return
        linear = np.asarray(linear_indices, dtype=int)
        counts = np.asarray(_get(state, "counts"))
        values = counts.ravel(order="F")[linear.ravel(order="F") - 1].reshape(
            linear.shape, order="F"
        )
        self.inactivatedSubstrates[li, :] = values

    def copyToState(self):
        super().copyToState()
        self._write_inactivated(
            self.monomer,
            self.inactivatedSubstrateMonomerGlobalCompartmentIndexs,
            self.substrateMonomerCompartmentIndexs,
            self.substrateMonomerLocalIndexs,
        )
        self._write_inactivated(
            self.complex,
            self.inactivatedSubstrateComplexGlobalCompartmentIndexs,
            self.substrateComplexCompartmentIndexs,
            self.substrateComplexLocalIndexs,
        )

    def _write_inactivated(self, state, globals_, comps, locals_):
        gi = np.asarray(globals_, int).ravel() - 1
        ci = np.asarray(comps, int).ravel() - 1
        li = np.asarray(locals_, int).ravel() - 1
        if not li.size:
            return
        counts = np.asarray(_get(state, "counts"))
        local = np.asarray(self.inactivatedSubstrates)
        if counts.ndim == 2 or counts.shape[2] == 1:
            values = local[li, :, :] if local.ndim > 2 else local[li, :]
            flat = counts.ravel(order="F")
            flat[np.asarray(globals_, dtype=int).ravel(order="F") - 1] = (
                values.ravel(order="F")
            )
            counts = np.reshape(flat, counts.shape, order="F")
        else:
            rows, columns = np.unravel_index(
                np.asarray(globals_, dtype=int).ravel(order="F") - 1,
                counts.shape[:2],
                order="F",
            )
            counts[rows, columns, :] = local[li, :, :].reshape(
                (rows.size, local.shape[2]), order="F"
            )
        _set(state, "counts", counts)

    def initializeState(self):
        self.evolveState()

    def calcResourceRequirements_Current(self):
        return np.zeros_like(np.asarray(self.substrates))

    def calcResourceRequirements_LifeCycle(self, *args, **kwargs):
        n = len(self.substrateWholeCellModelIDs)
        e = len(self.enzymeWholeCellModelIDs)
        return np.zeros(n), np.zeros(n), np.zeros(e), np.full(e, np.inf)

    def evaluateActivationRules(self):
        # Match evaluateActivationRules' extracellular concentration conversion.
        stimulus = np.asarray(self.stimuli)
        ext = (
            np.asarray(self.compartment.extracellularIndexs, dtype=int).ravel()
            - 1
        )
        values = (
            stimulus[:, ext]
            / (self._avogadro() / 1000)
            / float(self.geometry.volume)
        )
        raw = (
            np.asarray(self.stimulusStimulusLocalIndexs, dtype=int).ravel() - 1
        )
        if raw.size:
            values[raw, :] = stimulus[raw, :][:, ext]
        activated = np.zeros(np.asarray(self.substrates).shape, dtype=bool)
        rules = _matlab_strings(self.activationRules)
        ids = _matlab_strings(self.stimuliWholeCellModelIDs)
        for compartment in range(activated.shape[1]):
            if not np.any(self.substrates[:, compartment]) and not np.any(
                self.inactivatedSubstrates[:, compartment]
            ):
                continue
            variables = {
                name: float(format(float(values[i, 0]), ".5g"))
                for i, name in enumerate(ids)
            }
            for species, rule in enumerate(rules):
                if (
                    self.substrates[species, compartment]
                    or self.inactivatedSubstrates[species, compartment]
                ):
                    activated[species, compartment] = _eval_rule(
                        str(rule), variables
                    )
        return activated

    def evolveState(self):
        i = self.evaluateActivationRules()
        self.substrates[i] += self.inactivatedSubstrates[i]
        self.inactivatedSubstrates[i] = 0
        j = ~i
        self.inactivatedSubstrates[j] += self.substrates[j]
        self.substrates[j] = 0

    def getDryWeight(self):
        base = (
            float(
                np.asarray(self.substrateMolecularWeights)
                @ np.sum(self.substrates, axis=1)
            )
            / self._avogadro()
        )
        extra = (
            float(
                np.asarray(self.substrateMolecularWeights)
                @ np.sum(self.inactivatedSubstrates, axis=1)
            )
            / self._avogadro()
        )
        return base + extra
