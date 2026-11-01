"""Check fixture/source identity, native LP behavior and restored initial biology."""

from pathlib import Path
import hashlib, json, platform, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_fixtures import main as prepare
import numpy as np
from wholecell_py.checkpoint import source_digest
from wholecell_py.solvers import solve_lp
from wholecell_py.runner import load_model

prepare()
source = json.loads((ROOT / "validation/source-manifest.json").read_text())
assert (
    source_digest() == source["runtime_source_digest"]
), "Runtime source changed: record a new validation version"
cases = json.loads((ROOT / "reference/matlab_seed1.json").read_text())["cases"]
checked = 0
for case in cases:
    if case["process"] != "linear_programming":
        continue
    result = solve_lp({**case["inputs"], "backend": "glpk444"})
    assert (
        result["status"] == "optimal"
        and result["solver"] == "GLPK 4.44 (standalone)"
    )
    np.testing.assert_allclose(
        result["x"], case["expected"]["x"], rtol=1e-12, atol=1e-9
    )
    np.testing.assert_allclose(
        result["fopt"], case["expected"]["fopt"], rtol=1e-12, atol=1e-15
    )
    checked += 1
assert checked
graph = load_model(ROOT / "reference/snapshot_t60.json", seed=101)
time_s = float(np.asarray(graph.states["Time"].values).item())
growth = float(np.asarray(graph.states["MetabolicReaction"].growth).item())
mass = float(np.asarray(graph.states["Mass"].cellDry).sum())
assert time_s == 60 and growth > 0 and mass > 0
record = dict(
    python=platform.python_version(),
    numpy=np.__version__,
    runtime_source_digest=source_digest(),
    solver_binary_sha256=hashlib.sha256(
        (ROOT / "vendor/diagnostic/glpk444_text").read_bytes()
    ).hexdigest(),
    native_lp_cases=checked,
    restored_time_s=time_s,
    growth=growth,
    dry_mass_g=mass,
    scope="LP oracle and initial-state restoration; no rollout performed",
)
print(json.dumps(record, indent=2))
