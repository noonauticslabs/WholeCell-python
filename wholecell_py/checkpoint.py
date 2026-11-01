"""Native checkpoints for trusted, locally generated simulation graphs.

Pickle retains shared object identity and NumPy RNG states. These files are
Python runtime artifacts, not interchangeable with MATLAB JSON snapshots.
"""

from pathlib import Path
import hashlib
import json
import os
import pickle
import platform
import numpy as np


def source_digest():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def save_checkpoint(graph, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "wholecell-python-checkpoint-v1",
        "source_sha256": getattr(
            graph, "runtime_source_sha256", source_digest()
        ),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "time_s": float(np.asarray(graph.states["Time"].values).item()),
    }
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write((json.dumps(metadata) + "\n").encode())
            pickle.dump(graph, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return metadata


def load_checkpoint(path, *, allow_code_change=False):
    """Load only a trusted checkpoint produced by this project."""
    with Path(path).open("rb") as stream:
        metadata = json.loads(stream.readline())
        if metadata.get("format") != "wholecell-python-checkpoint-v1":
            raise ValueError("Unsupported checkpoint format")
        changed = metadata["source_sha256"] != source_digest()
        if changed and not allow_code_change:
            raise ValueError(
                "Runtime source differs from checkpoint; use the original source or explicitly allow a code change"
            )
        graph = pickle.load(stream)
    metadata["code_changed_on_resume"] = changed
    graph.checkpoint_provenance = metadata
    graph.runtime_source_sha256 = source_digest()
    return graph
