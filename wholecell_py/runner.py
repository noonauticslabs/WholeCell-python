"""Whole-cell entry point with explicit completeness and RNG policies.

Missing processes cause a hard failure before any evolution. This entry point
never runs a smaller model under the label of a whole-cell simulation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .randstream import RandStream
from .runtime_registry import implemented_registry
from .simulation import Simulation
from .snapshot import load_snapshot
from .checkpoint import load_checkpoint, save_checkpoint
from .states.chromosome import Chromosome


def _text(value):
    return (
        value
        if isinstance(value, str)
        else "".join(np.asarray(value).ravel(order="F"))
    )


class SimulationContext:
    """Snapshot object identity plus the independently executing scheduler."""

    def state(self, name):
        return self._lookup(self.states, name, "State_")

    def process(self, name):
        return self._lookup(self.processes, name, "Process_")

    @staticmethod
    def _lookup(objects, name, prefix):
        items = np.asarray(objects, dtype=object).ravel(order="F")
        name = _text(name)
        for candidate in (name, prefix + name):
            for item in items:
                if _text(item.wholeCellModelID) == candidate:
                    return item
        return np.empty((0, 0))

    def evolveState(self):
        state_items = np.asarray(self.states, dtype=object).ravel(order="F")
        process_items = np.asarray(self.processes, dtype=object).ravel(
            order="F"
        )
        step_size = float(np.asarray(self.stepSizeSec).item())
        trna_index = int(
            np.asarray(self.processIndex_tRNAAminoacylation).item()
        )
        translation_index = int(
            np.asarray(self.processIndex_translation).item()
        )

        # Snapshot object identity is stable while biological arrays mutate in
        # place. Key on object identities and model IDs so replacement,
        # reordering, or renaming rebuilds the scheduler instead of retaining
        # stale process/state references. Stream state is intentionally not in
        # the key: the coordinator holds the same mutable RNG object.
        state_signature = tuple(
            (id(item), _text(item.wholeCellModelID)) for item in state_items
        )
        process_signature = tuple(
            (id(item), _text(item.wholeCellModelID)) for item in process_items
        )
        signature = (
            state_signature,
            process_signature,
            id(self.randStream),
            step_size,
            trna_index,
            translation_index,
        )
        coordinator = getattr(self, "_evolution_coordinator", None)
        if (
            coordinator is None
            or signature != self._evolution_coordinator_signature
        ):
            states = {
                _text(item.wholeCellModelID).removeprefix("State_"): item
                for item in state_items
            }
            processes = list(process_items)
            coordinator = Simulation(
                states,
                processes,
                self.randStream,
                step_size,
                trna_index,
                translation_index,
            )
            self._evolution_coordinator = coordinator
            self._evolution_coordinator_signature = signature
        coordinator.after_process = getattr(self, "after_process", None)
        return coordinator.evolve_state()


def load_model(path, *, seed=1):
    path = Path(path)
    registry = implemented_registry()
    registry["edu.stanford.covert.cell.sim.state.Chromosome"] = Chromosome
    registry["edu.stanford.covert.cell.sim.Simulation"] = SimulationContext
    raw = json.loads(path.read_text())
    missing = [
        obj["id"]
        for obj in raw["objects"]
        if obj["id"].startswith(("state:", "process:"))
        and obj["class"] not in registry
    ]
    if missing:
        raise RuntimeError(
            "Whole-cell execution unavailable; missing runtime classes: "
            + ", ".join(missing)
        )
    state_count = sum(obj["id"].startswith("state:") for obj in raw["objects"])
    process_count = sum(
        obj["id"].startswith("process:") for obj in raw["objects"]
    )
    if (state_count, process_count) != (16, 28):
        raise ValueError(
            f"Expected complete 16-state, 28-process model; got {state_count}, {process_count}"
        )
    streams = []
    sequence = np.random.SeedSequence(seed)

    def rng_factory(record):
        # MATLAB RNG state cannot be passed to NumPy's PCG64. Independent child
        # streams are explicitly reseeded; all biological snapshot state is kept.
        child = sequence.spawn(1)[0]
        stream = RandStream(seed=child)
        streams.append(
            {
                "source_type": _text(record.type),
                "source_seed": np.asarray(record.seed).tolist(),
                "numpy_spawn_key": list(child.spawn_key),
            }
        )
        return stream

    graph = load_snapshot(
        path,
        class_registry=registry,
        runtime_values=True,
        rng_factory=rng_factory,
    )
    # Simulation's private scheduler stream is absent from the public snapshot.
    # Allocate an explicit independent stream rather than inheriting global RNG.
    child = sequence.spawn(1)[0]
    graph.simulation.randStream = RandStream(seed=child)
    streams.append(
        {
            "source_type": "private MATLAB scheduler stream, unexported",
            "numpy_spawn_key": list(child.spawn_key),
        }
    )
    from .checkpoint import source_digest

    graph.runtime_source_sha256 = source_digest()
    graph.rng_provenance = {
        "policy": "independent NumPy PCG64 child streams; MATLAB RNG state not replayed",
        "root_seed": seed,
        "streams": streams,
    }
    return graph


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot", type=Path)
    source.add_argument("--resume", type=Path)
    source.add_argument(
        "--initialize-from",
        type=Path,
        help="Sample a fresh cell using fitted constants from this snapshot",
    )
    parser.add_argument("--initialization-attempts", type=int, default=100)
    parser.add_argument(
        "--initialization-mode", choices=("expected", "multinomial")
    )
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=60)
    parser.add_argument("--stop-at-division", action="store_true")
    parser.add_argument("--allow-code-change", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    if args.allow_code_change and args.resume is None:
        parser.error("--allow-code-change is only valid with --resume")
    if args.initialize_from is not None:
        from .initialization import initialize_model

        graph = initialize_model(
            args.initialize_from,
            seed=args.seed,
            max_attempts=args.initialization_attempts,
            mode=args.initialization_mode,
        )
        print(
            json.dumps(
                {"initialization_provenance": graph.initialization_provenance}
            ),
            flush=True,
        )
    else:
        graph = (
            load_checkpoint(
                args.resume, allow_code_change=args.allow_code_change
            )
            if args.resume is not None
            else load_model(args.snapshot, seed=args.seed)
        )
    print(json.dumps(graph.rng_provenance), flush=True)
    if args.resume is not None:
        print(
            json.dumps({"checkpoint_provenance": graph.checkpoint_provenance}),
            flush=True,
        )
    for step in range(1, args.steps + 1):
        graph.simulation.evolveState()
        # A process may have completed before a later process fails. Check and
        # checkpoint only after the scheduler returns successfully for the step.
        for name, state in graph.states.items():
            counts = getattr(state, "counts", None)
            if counts is None:
                continue
            values = np.asarray(counts, dtype=float)
            if not np.all(np.isfinite(values)) or np.any(values < -1e-9):
                raise ValueError(
                    f"Invalid counts in state {name!r}: counts must be finite and nonnegative"
                )

        def count_sum(name):
            return float(
                np.asarray(graph.states[name].counts, dtype=float).sum()
            )

        chromosome = graph.states["Chromosome"]
        geometry = graph.states["Geometry"]
        mass = graph.states["Mass"]
        _, polymerized_lengths = chromosome.polymerizedRegions.find()
        pinched = bool(np.asarray(geometry.pinched).item())
        result = {
            "time_s": float(np.asarray(graph.states["Time"].values).item()),
            "growth": float(
                np.asarray(graph.states["MetabolicReaction"].growth).sum()
            ),
            "cell_dry_mass": float(
                np.asarray(graph.states["Mass"].cellDry).sum()
            ),
            "rna_counts": count_sum("Rna"),
            "protein_monomer_counts": count_sum("ProteinMonomer"),
            "protein_complex_counts": count_sum("ProteinComplex"),
            "polymerized_nucleotides": float(
                np.asarray(polymerized_lengths).sum()
            ),
            "pinched": pinched,
            "pinched_diameter_m": float(
                np.asarray(geometry.pinchedDiameter).item()
            ),
            "ploidy": float(np.asarray(chromosome.ploidy).item()),
            "segregated": bool(np.asarray(chromosome.segregated).item()),
            "cell_total_mass_g": float(np.asarray(mass.cell).sum()),
        }
        print(json.dumps(result), flush=True)
        stop_at_division = args.stop_at_division and pinched
        if args.checkpoint is not None and (
            step % args.checkpoint_every == 0
            or step == args.steps
            or stop_at_division
        ):
            save_checkpoint(graph, args.checkpoint)
        if stop_at_division:
            break


if __name__ == "__main__":
    main()
