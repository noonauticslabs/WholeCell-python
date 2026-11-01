"""Exploratory whole-cell discrimination; no independent time-point replication."""

from pathlib import Path
import json, csv
import numpy as np
from scipy.stats import t
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "validation/generated"
(HERE / "derived").mkdir(parents=True, exist_ok=True)
with np.load(
    ROOT / "validation/fixed-cohort/trajectories.npz", allow_pickle=False
) as data:
    times = data["times"].copy()
    ends = {m: data["ends_" + m].copy() for m in ["MATLAB", "Python"]}
    mats = {
        k: {
            m: data["mats_" + k + "_" + m].copy() for m in ["MATLAB", "Python"]
        }
        for k in [
            "mass",
            "growth",
            "rna",
            "monomers",
            "complexes",
            "ploidy",
            "diameter",
        ]
    }
    endpoints = {
        k: {
            m: data["endpoints_" + k + "_" + m].copy()
            for m in ["MATLAB", "Python"]
        }
        for k in mats
    }
keys = ["mass", "growth", "rna", "monomers", "complexes", "ploidy"]
models = ["MATLAB", "Python"]
rng = np.random.default_rng(20261102)
B = 9999
labels = np.r_[np.zeros(12), np.ones(12)]
Y = np.vstack([labels, *[rng.permutation(labels) for _ in range(B)]])
auc = {
    k: {
        m: np.trapezoid(mats[k][m], times, axis=1) / (times[-1] - times[0])
        for m in models
    }
    for k in keys
}
features = {
    "Cycle averages + division": np.vstack(
        [
            np.column_stack([auc[k][m] for k in keys] + [ends[m] / 3600])
            for m in models
        ]
    ),
    "Division endpoints": np.vstack(
        [
            np.column_stack(
                [endpoints[k][m] for k in keys[:-1]] + [ends[m] / 3600]
            )
            for m in models
        ]
    ),
    "Common-time trajectories": np.vstack(
        [
            np.column_stack(
                [
                    mats[k][m][
                        :, np.linspace(0, len(times) - 1, 12).astype(int)
                    ]
                    for k in keys
                ]
            )
            for m in models
        ]
    ),
}
results = []
allacc = []
for name, X in features.items():
    predictions = [np.empty_like(Y), np.empty_like(Y)]
    for i in range(24):
        keep = np.arange(24) != i
        train = X[keep]
        scale = train.std(0, ddof=1)
        scale[scale < 1e-12] = 1
        z = (train - train.mean(0)) / scale
        test = (X[i] - train.mean(0)) / scale
        yy = Y[:, keep]
        n1 = yy.sum(1)[:, None]
        n0 = 23 - n1
        mu1 = (yy @ z) / n1
        mu0 = ((1 - yy) @ z) / n0
        score = ((test - mu0) ** 2).sum(1) - ((test - mu1) ** 2).sum(1)
        predictions[0][:, i] = (score > 0) + 0.5 * (score == 0)
        d = ((z[:, None, :] - z[None, :, :]) ** 2).sum(2)
        bw = np.median(d[np.triu_indices(23, 1)])
        bw = max(bw, 1e-12)
        kernel = np.exp(-((z - test) ** 2).sum(1) / (2 * bw))
        score = (yy @ kernel) / n1[:, 0] - ((1 - yy) @ kernel) / n0[:, 0]
        predictions[1][:, i] = (score > 0) + 0.5 * (score == 0)
    for method, pred in zip(
        ["Nearest centroid", "RBF similarity"], predictions
    ):
        acc = (1 - abs(pred - Y)).mean(1)
        allacc.append(acc)
        results.append(
            dict(
                features=name,
                method=method,
                accuracy=float(acc[0]),
                correct=float(acc[0] * 24),
                p_permutation=float(
                    (1 + (acc[1:] >= acc[0] - 1e-12).sum()) / (B + 1)
                ),
                null_accuracy_95=np.quantile(acc[1:], [0.025, 0.975]).tolist(),
                predicted_python=pred[0].tolist(),
            )
        )
nullmax = np.max(np.array(allacc)[:, 1:], axis=0)
for r in results:
    r["p_family_max_accuracy"] = float(
        (1 + (nullmax >= r["accuracy"] - 1e-12).sum()) / (B + 1)
    )
# Relative arithmetic means: independent whole-cell bootstrap, include uncertainty in denominator.
bi = rng.integers(0, 12, (20000, 12))
bj = rng.integers(0, 12, (20000, 12))
rows = []
for scope, data in [
    ("Cycle average", auc),
    ("At division", {k: endpoints[k] for k in keys[:-1]}),
    ("Division time", {"division": {m: ends[m] / 3600 for m in models}}),
]:
    for k, d in data.items():
        a, b = d["Python"], d["MATLAB"]
        aa, bb = a[bi], b[bj]
        ratio = 100 * (aa.mean(1) / bb.mean(1) - 1)
        vr = aa.var(1, ddof=1) / bb.var(1, ddof=1)
        rows.append(
            dict(
                scope=scope,
                metric=k,
                bias_percent=float(100 * (a.mean() / b.mean() - 1)),
                mean_difference_percent_ci90=np.quantile(
                    ratio, [0.05, 0.95]
                ).tolist(),
                mean_difference_percent_ci95=np.quantile(
                    ratio, [0.025, 0.975]
                ).tolist(),
                variance_ratio=float(a.var(ddof=1) / b.var(ddof=1)),
                variance_ratio_ci95=np.quantile(vr, [0.025, 0.975]).tolist(),
            )
        )
out = {
    "scope": "12+12 fixed-cohort cells. Python uses one corrected source throughout from shared MATLAB t60; no mixed-version continuations. Same aggregate features and classifier rules as historical analysis, fixed before evaluating this cohort.",
    "permutations": B,
    "seed": 20261102,
    "cv": "Leave one whole cell out; scaling and RBF bandwidth fit on 23 training cells; fixed rules, no hyperparameter search. All folds refit for every balanced label permutation. Null is identical joint distributions/exchangeability. Accuracy has dependent folds; no naive binomial confidence interval.",
    "classifiers": results,
    "mean_and_spread": rows,
    "limits": "Bootstrap intervals exploratory/pointwise, not simultaneous. 90% bounds are equivalence-margin sensitivity, not a prespecified formal equivalence claim. No failure to reject establishes indistinguishability. Inference is conditional on one starting biological state and these aggregate features, not every internal process.",
}
(HERE / "derived/discrimination.json").write_text(
    json.dumps(out, indent=2) + "\n"
)
public = HERE
fig, axs = plt.subplots(1, 2, figsize=(15, 7))
a = axs[0]
for j, r in enumerate(results):
    lo, hi = r["null_accuracy_95"]
    a.plot([100 * lo, 100 * hi], [j, j], lw=8, color="#dce2eb")
    a.scatter(100 * r["accuracy"], j, s=70, color="#2676c9", zorder=3)
    a.text(
        99,
        j,
        f"p(adj)={r['p_family_max_accuracy']:.3f}",
        va="center",
        ha="right",
        fontsize=9,
    )
a.axvline(50, ls="--", color="gray")
a.set(
    yticks=range(6),
    yticklabels=[r["features"] + "\n" + r["method"] for r in results],
    xlabel="Held-out classification accuracy (%)",
    xlim=(0, 100),
    title="Can we identify the simulator?",
)
a.invert_yaxis()
a = axs[1]
selected = [r for r in rows if r["scope"] == "Cycle average"] + [rows[-1]]
for j, r in enumerate(selected):
    lo, hi = r["mean_difference_percent_ci95"]
    a.plot([lo, hi], [j, j], color="#2676c9", lw=2)
    a.scatter(r["bias_percent"], j, color="#2676c9")
a.axvline(0, ls="--", color="gray")
a.axvspan(-5, 5, color="#28a076", alpha=0.08)
a.set(
    yticks=range(len(selected)),
    yticklabels=[r["metric"] for r in selected],
    xlabel="Python mean relative to MATLAB (%)",
    title="Mean differences | 95% bootstrap intervals",
)
a.invert_yaxis()
fig.suptitle(
    "Simulator identity versus seed variability",
    fontsize=20,
    x=0.04,
    ha="left",
)
fig.text(
    0.04,
    0.90,
    "12 MATLAB + 12 Python cells. Python uses one corrected source version throughout.",
    fontsize=11,
)
fig.text(
    0.04,
    0.04,
    "Left gray bars: 95% permutation-null ranges; adjusted p accounts for six classifiers. Right green region: illustrative ±5%, not a preset acceptance rule.",
    fontsize=9,
)
fig.subplots_adjust(left=0.20, right=0.97, top=0.81, bottom=0.14, wspace=0.60)
fig.savefig(
    public / "11_simulator_discrimination.png", dpi=160, bbox_inches="tight"
)
plt.close(fig)
print(json.dumps(out, indent=2))
