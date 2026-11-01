"""Mean-focused checks, family-wise correction across six averages + division."""

from pathlib import Path
import json
import numpy as np
from scipy.stats import ttest_ind

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "validation/generated"
(HERE / "derived").mkdir(parents=True, exist_ok=True)
with np.load(
    ROOT / "validation/fixed-cohort/trajectories.npz", allow_pickle=False
) as z:
    times = z["times"]
    keys = ["mass", "growth", "rna", "monomers", "complexes", "ploidy"]
    values = {
        k: {
            m: np.trapezoid(z["mats_" + k + "_" + m], times, axis=1)
            / (times[-1] - times[0])
            for m in ("Python", "MATLAB")
        }
        for k in keys
    }
    values["division"] = {
        m: z["ends_" + m] / 3600 for m in ("Python", "MATLAB")
    }
rows = []
for key, d in values.items():
    a, b = d["Python"], d["MATLAB"]
    test = ttest_ind(a, b, equal_var=False)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    ci = test.confidence_interval(0.95)
    rows.append(
        dict(
            metric=key,
            python_mean=float(a.mean()),
            matlab_mean=float(b.mean()),
            python_sd=float(a.std(ddof=1)),
            matlab_sd=float(b.std(ddof=1)),
            standardized_mean_bias=float((a.mean() - b.mean()) / pooled),
            welch_p=float(test.pvalue),
            welch_mean_difference_ci95=[float(ci.low), float(ci.high)],
        )
    )
order = np.argsort([r["welch_p"] for r in rows])
adjusted = 0
for rank, i in enumerate(order):
    adjusted = max(adjusted, min(1.0, (len(rows) - rank) * rows[i]["welch_p"]))
    rows[i]["holm_p"] = adjusted
out = dict(
    scope="Mean differences in six common-window averages and division time. Independent cells, no seed pairing. Welch tests with Holm adjustment over seven summaries; additional diagnostic on the fixed cohort, not a prospective equivalence test.",
    rows=rows,
)
(HERE / "derived/bias-tests.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
