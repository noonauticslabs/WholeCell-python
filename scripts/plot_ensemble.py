"""Completed fixed-cohort comparison. Read-only inputs; reproducible static figures."""

from pathlib import Path
import csv, json, hashlib, math
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "validation/generated"
DERIVED = HERE / "derived"
DERIVED.mkdir(parents=True, exist_ok=True)
PUBLIC = HERE / "figures"
PUBLIC.mkdir(exist_ok=True)
SEEDS = list(range(101, 113))
N = 12
B = 4000
rng = np.random.default_rng(20261101)
COL = {"MATLAB": "#D78A25", "Python": "#2676C9"}
METRICS = [
    ("mass", "Dry mass", "fg", "cell_dry_mass", "dry_mass_g", 1e15),
    ("growth", "Growth rate", "×10⁻⁵ s⁻¹", "growth", "growth", 1e5),
    ("rna", "RNA", "×10³ molecules", "rna_counts", "rna_count", 1e-3),
    (
        "monomers",
        "Protein monomers",
        "×10³ molecules",
        "protein_monomer_counts",
        "protein_monomer_count",
        1e-3,
    ),
    (
        "complexes",
        "Protein complexes",
        "×10³ molecules",
        "protein_complex_counts",
        "protein_complex_count",
        1e-3,
    ),
    (
        "ploidy",
        "DNA replication",
        "chromosome equivalents",
        "ploidy",
        "ploidy",
        1,
    ),
    (
        "diameter",
        "Constriction diameter",
        "nm",
        "pinched_diameter_m",
        "pinched_diameter_m",
        1e9,
    ),
]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "#FBFCFE",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#B7C0CC",
        "grid.color": "#E0E5EB",
        "grid.alpha": 0.7,
        "axes.grid": True,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    }
)


with np.load(
    ROOT / "validation/fixed-cohort/trajectories.npz", allow_pickle=False
) as z:
    times = z["times"].copy()
    phase = z["phase"].copy()
    ends = {m: z["ends_" + m].copy() for m in COL}
    mats = {
        k: {m: z["mats_" + k + "_" + m].copy() for m in COL}
        for k, *_ in METRICS
    }
    phases = {
        k: {m: z["phases_" + k + "_" + m].copy() for m in COL} for k in mats
    }
    endpoints = {
        k: {m: z["endpoints_" + k + "_" + m].copy() for m in COL} for k in mats
    }
    individuals = {
        (m, s): z[f"individual_{m}_{s}"].copy() for m in COL for s in SEEDS
    }
pi = rng.integers(0, N, (B, N))
mi = rng.integers(0, N, (B, N))


def boot_stats(py, ml):
    py = np.atleast_2d(py) if np.ndim(py) > 1 else np.asarray(py)[:, None]
    ml = np.atleast_2d(ml) if np.ndim(ml) > 1 else np.asarray(ml)[:, None]
    pm, mm = py.mean(0), ml.mean(0)
    pv, mv = py.var(0, ddof=1), ml.var(0, ddof=1)
    pb, mb = py[pi], ml[mi]
    bd = pb.mean(1) - mb.mean(1)
    bpv, bmv = pb.var(1, ddof=1), mb.var(1, ddof=1)
    pool = np.sqrt((pv + mv) / 2)
    bp = np.sqrt((bpv + bmv) / 2)
    tol = np.maximum(np.maximum(abs(pm), abs(mm)) * 1e-12, 1e-14)
    smd = np.divide(
        pm - mm, pool, out=np.full_like(pool, np.nan), where=pool > tol
    )
    vr = np.divide(pv, mv, out=np.full_like(pv, np.nan), where=mv > tol**2)
    bsmd = np.divide(bd, bp, out=np.full_like(bd, np.nan), where=bp > tol)
    bvr = np.divide(
        bpv, bmv, out=np.full_like(bpv, np.nan), where=bmv > tol**2
    )

    def q(a):
        z = np.full((2, a.shape[1]), np.nan)
        for j in range(a.shape[1]):
            v = a[:, j]
            v = v[np.isfinite(v)]
            if len(v):
                z[:, j] = np.quantile(v, [0.025, 0.975])
        return z

    return dict(
        python_mean=pm,
        matlab_mean=mm,
        python_sd=np.sqrt(pv),
        matlab_sd=np.sqrt(mv),
        difference=pm - mm,
        difference_ci=q(bd),
        smd=smd,
        smd_ci=q(bsmd),
        variance_ratio=vr,
        variance_ratio_ci=q(bvr),
        python_sd_ci=q(np.sqrt(bpv)),
        matlab_sd_ci=q(np.sqrt(bmv)),
    )


stats = {k: boot_stats(v["Python"], v["MATLAB"]) for k, v in mats.items()}
endpointstats = {
    k: boot_stats(v["Python"], v["MATLAB"]) for k, v in endpoints.items()
}
division = boot_stats(ends["Python"] / 3600, ends["MATLAB"] / 3600)


def scalar_summary(st):
    return {
        k: (v[:, 0].tolist() if v.ndim == 2 else float(v[0]))
        for k, v in st.items()
    }


summary = {
    "cohort": SEEDS,
    "n_per_model": N,
    "division_hours": scalar_summary(division),
    "common_grid": {
        "first_s": float(times[0]),
        "last_s": float(times[-1]),
        "samples": len(times),
    },
    "at_last_common_time": {
        k: scalar_summary(
            {a: (b[:, -1:] if b.ndim == 2 else b[-1:]) for a, b in st.items()}
        )
        for k, st in stats.items()
    },
    "at_division": {k: scalar_summary(st) for k, st in endpointstats.items()},
    "scope": "Fixed-source Python trajectories from original MATLAB t60 state, no Python continuations. Independent RNG ensembles; no paired-seed inference. Fixed original 12-cell cohort. Statistical analysis of aggregate observables, conditional on one initial state; no prespecified equivalence margins.",
}
(DERIVED / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
with (DERIVED / "division_times.csv").open("w") as f:
    w = csv.writer(f)
    w.writerow(["model", "seed", "division_time_s", "division_time_h"])
    for model in COL:
        for s, t in zip(SEEDS, ends[model]):
            w.writerow([model, s, t, t / 3600])
with (DERIVED / "time_matched_statistics.csv").open("w") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "metric",
            "time_s",
            "python_mean",
            "matlab_mean",
            "python_sd",
            "matlab_sd",
            "difference",
            "difference_ci_low",
            "difference_ci_high",
            "smd",
            "smd_ci_low",
            "smd_ci_high",
            "variance_ratio",
            "variance_ratio_ci_low",
            "variance_ratio_ci_high",
        ]
    )
    for k, st in stats.items():
        for j, t in enumerate(times):
            w.writerow(
                [
                    k,
                    t,
                    *[
                        st[a][j]
                        for a in [
                            "python_mean",
                            "matlab_mean",
                            "python_sd",
                            "matlab_sd",
                            "difference",
                        ]
                    ],
                    *st["difference_ci"][:, j],
                    st["smd"][j],
                    *st["smd_ci"][:, j],
                    st["variance_ratio"][j],
                    *st["variance_ratio_ci"][:, j],
                ]
            )

figures = []
pdf = PdfPages(PUBLIC / "WholeCell_full_cycle_comparison.pdf")


def decorate(fig, title, subtitle, foot):
    fig.suptitle(
        title,
        x=0.065,
        y=0.985,
        ha="left",
        fontsize=22,
        fontweight="bold",
        color="#172B45",
    )
    fig.text(0.065, 0.937, subtitle, ha="left", fontsize=11, color="#526379")
    fig.text(0.065, 0.017, foot, fontsize=8.5, color="#526379")
    fig.text(
        0.065,
        0.045,
        "Python: one corrected source version throughout; original shared MATLAB t=60 starting state.",
        fontsize=9,
        color="#526379",
    )
    fig.subplots_adjust(
        top=0.86,
        bottom=0.10,
        left=0.075,
        right=0.975,
        hspace=0.40,
        wspace=0.28,
    )


def save(fig, name):
    fig.savefig(PUBLIC / (name + ".png"), dpi=170)
    pdf.savefig(fig)
    figures.append(name)
    plt.close(fig)


def bands(ax, x, model, data, sd=True, ci=True):
    mean = data.mean(0)
    spread = data.std(0, ddof=1)
    c = COL[model]
    if sd:
        ax.fill_between(
            x, mean - spread, mean + spread, color=c, alpha=0.13, lw=0
        )
    if ci:
        h = student_t.ppf(0.975, N - 1) * spread / np.sqrt(N)
        ax.fill_between(x, mean - h, mean + h, color=c, alpha=0.25, lw=0)
    ax.plot(x, mean, color=c, lw=2, label=model)


def panel(ax, key, phase_plot=False, thin=False):
    met = next(m for m in METRICS if m[0] == key)
    data = phases[key] if phase_plot else mats[key]
    x = phase * 100 if phase_plot else times / 3600
    for model in COL:
        if thin:
            for y in data[model]:
                ax.plot(x, y, color=COL[model], lw=0.65, alpha=0.13)
        bands(ax, x, model, data[model])
    ax.set(
        title=met[1],
        ylabel=met[2],
        xlabel=(
            "Recorded cycle completed (%)"
            if phase_plot
            else "Simulation time (hours)"
        ),
    )
    ax.legend(frameon=False, fontsize=9, loc="best")
    ax.margins(x=0.015)


fig, axs = plt.subplots(2, 3, figsize=(16, 10))
for ax, k in zip(axs.flat, ["mass", "growth", "rna", "monomers", "complexes"]):
    panel(ax, k)
ax = axs.flat[-1]
for i, model in enumerate(COL):
    vals = ends[model] / 3600
    ax.scatter(
        np.full(N, i) + np.linspace(-0.12, 0.12, N),
        np.sort(vals),
        color=COL[model],
        s=32,
        alpha=0.9,
    )
    h = student_t.ppf(0.975, N - 1) * vals.std(ddof=1) / np.sqrt(N)
    ax.errorbar(
        i, vals.mean(), yerr=h, fmt="D", color="#172B45", capsize=7, ms=6
    )
ax.set(
    xticks=[0, 1],
    xticklabels=list(COL),
    ylabel="Division time (hours)",
    title="All 24 cells reached division",
)
ax.set_xlim(-0.5, 1.5)
d = scalar_summary(division)
subtitle = f"12 MATLAB + 12 Python cells  |  Mean division: {d['matlab_mean']:.2f} h vs {d['python_mean']:.2f} h  |  Python shift: {60*d['difference']:+.1f} min"
decorate(
    fig,
    "Whole-cell comparison | complete first cohort",
    subtitle,
    f"Trajectory panels: fixed 12+12 cohort through {times[-1]/3600:.2f} h; light bands ±1 SD, dark bands pointwise 95% mean CI. Python uses one corrected source throughout.",
)
save(fig, "01_full_cycle_overview")

fig, axs = plt.subplots(2, 2, figsize=(15, 10))
for ax, k in zip(axs[0], ["mass", "growth"]):
    panel(ax, k, thin=True)
for ax, k in zip(axs[1], ["mass", "growth"]):
    st = stats[k]
    for model, prefix in [("MATLAB", "matlab"), ("Python", "python")]:
        lo, hi = st[prefix + "_sd_ci"]
        ax.fill_between(times / 3600, lo, hi, color=COL[model], alpha=0.14)
        ax.plot(
            times / 3600,
            st[prefix + "_sd"],
            color=COL[model],
            lw=2,
            label=model,
        )
    ax.set(
        title=next(m[1] for m in METRICS if m[0] == k) + " | seed-to-seed SD",
        xlabel="Simulation time (hours)",
        ylabel=next(m[2] for m in METRICS if m[0] == k),
    )
    ax.legend(frameon=False)
decorate(
    fig,
    "Growth and mass | means versus variability",
    "Upper: means, individual cells and confidence bands. Lower: SD with independent whole-trajectory bootstrap uncertainty.",
    "12 cells per model at every timepoint. CI bands are pointwise, not simultaneous across time. Bootstrap: 4,000 independent resamples within each model.",
)
save(fig, "02_growth_mass_and_spread")

fig, axs = plt.subplots(2, 3, figsize=(16, 10))
for ax, k in zip(axs[0], ["rna", "monomers", "complexes"]):
    panel(ax, k)
for ax, k in zip(axs[1], ["rna", "monomers", "complexes"]):
    st = stats[k]
    for model, prefix in [("MATLAB", "matlab"), ("Python", "python")]:
        ax.plot(
            times / 3600,
            st[prefix + "_sd"],
            color=COL[model],
            lw=2,
            label=model,
        )
        ax.fill_between(
            times / 3600, *st[prefix + "_sd_ci"], color=COL[model], alpha=0.13
        )
    ax.set(
        title=next(m[1] for m in METRICS if m[0] == k) + " | SD",
        xlabel="Simulation time (hours)",
        ylabel=next(m[2] for m in METRICS if m[0] == k),
    )
    ax.legend(frameon=False)
decorate(
    fig,
    "Gene expression | abundance and seed-to-seed spread",
    "RNA, protein monomers and protein complexes across the same complete cohort.",
    "Upper: light ±1 SD / dark 95% mean CI. Lower: SD with pointwise bootstrap 95% CI. Fixed-cohort absolute-time window; no post-division survivor averaging.",
)
save(fig, "03_expression_and_spread")

fig, axs = plt.subplots(2, 2, figsize=(15, 10))
for ax, k in zip(axs[0], ["ploidy", "diameter"]):
    panel(ax, k)
for ax, k in zip(axs[1], ["ploidy", "diameter"]):
    met = next(m for m in METRICS if m[0] == k)
    for model, field in [("MATLAB", met[4]), ("Python", met[3])]:
        for j, s in enumerate(SEEDS):
            curve = individuals[(model, s)]
            ax.plot(
                curve[:, 0] / 3600,
                curve[:, 1 if k == "ploidy" else 2],
                color=COL[model],
                lw=1,
                alpha=0.5,
                label=model if j == 0 else None,
            )
    ax.set(
        title=met[1] + " | every complete trajectory",
        xlabel="Simulation time (hours)",
        ylabel=met[2],
    )
    ax.legend(frameon=False)
decorate(
    fig,
    "Replication and division | the complete transitions",
    "Upper panels compare the full cohort before its first division. Lower panels follow every cell to its own endpoint.",
    "No surviving-cell mean is drawn after the first division. Individual curves retain asynchronous replication and constriction timing.",
)
save(fig, "04_replication_and_constriction")

fig, axs = plt.subplots(2, 2, figsize=(15, 10))
ax = axs[0, 0]
for i, model in enumerate(COL):
    vals = ends[model] / 3600
    ax.scatter(
        vals,
        np.full(N, i) + np.linspace(-0.14, 0.14, N),
        s=45,
        color=COL[model],
    )
    ax.errorbar(
        vals.mean(),
        i,
        xerr=student_t.ppf(0.975, N - 1) * vals.std(ddof=1) / np.sqrt(N),
        fmt="D",
        color="#172B45",
        capsize=8,
    )
ax.set(
    yticks=[0, 1],
    yticklabels=list(COL),
    xlabel="Division time (hours)",
    title="Individual cells + mean 95% CI",
)
ax = axs[0, 1]
for model in COL:
    vals = np.sort(ends[model] / 3600)
    ax.step(
        np.r_[vals[0] - 0.1, vals],
        np.r_[0, np.arange(1, N + 1) / N],
        where="post",
        lw=2.2,
        color=COL[model],
        label=model,
    )
ax.set(
    xlabel="Division time (hours)",
    ylabel="Fraction divided",
    title="Empirical division-time distributions",
)
ax.legend(frameon=False)
ax = axs[1, 0]
diff = division["difference"][0] * 60
lo, hi = division["difference_ci"][:, 0] * 60
ax.axvline(0, color="#8792A1", ls="--")
ax.errorbar(
    diff,
    0,
    xerr=[[diff - lo], [hi - diff]],
    fmt="o",
    color=COL["Python"],
    capsize=9,
    ms=8,
)
ax.set(
    yticks=[],
    xlabel="Python − MATLAB mean division time (minutes)",
    title=f"Mean shift {diff:+.1f} min; independent bootstrap 95% CI",
)
ax.set_ylim(-1, 1)
ax = axs[1, 1]
for i, model in enumerate(COL):
    vals = ends[model] / 3600
    st = division
    prefix = model.lower()
    lo, hi = st[prefix + "_sd_ci"][:, 0] * 60
    sd = vals.std(ddof=1) * 60
    ax.bar(i, sd, color=COL[model], alpha=0.75, width=0.5)
    ax.errorbar(i, sd, yerr=[[sd - lo], [hi - sd]], color="#172B45", capsize=7)
ax.set(
    xticks=[0, 1],
    xticklabels=list(COL),
    ylabel="Division-time SD (minutes)",
    title="Dispersion across independent cells",
)
decorate(
    fig,
    "Division times | both cohorts completed",
    "All 12 cells in each simulator divided; no censoring in this selected comparison cohort.",
    "Identical seed labels are not paired random streams. Intervals resample cells independently within each model. Completion does not itself establish biological equivalence.",
)
save(fig, "05_division_time_distributions")

fig, axs = plt.subplots(2, 3, figsize=(16, 10))
for ax, k in zip(axs[0], ["mass", "growth", "monomers"]):
    st = stats[k]
    ax.axhline(0, color="#7C8795", ls="--")
    ax.fill_between(
        times / 3600, *st["smd_ci"], color=COL["Python"], alpha=0.16
    )
    ax.plot(times / 3600, st["smd"], color=COL["Python"], lw=2)
    ax.set(
        title=next(m[1] for m in METRICS if m[0] == k) + " | mean separation",
        xlabel="Simulation time (hours)",
        ylabel="Mean difference / pooled within-model SD",
    )
for ax, k in zip(axs[1], ["mass", "growth", "monomers"]):
    st = stats[k]
    ax.axhline(1, color="#7C8795", ls="--")
    ax.fill_between(
        times / 3600, *st["variance_ratio_ci"], color="#7A5AA6", alpha=0.16
    )
    ax.plot(times / 3600, st["variance_ratio"], color="#7A5AA6", lw=2)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylim(bottom=0)
    ax.set(
        title=next(m[1] for m in METRICS if m[0] == k) + " | variance ratio",
        xlabel="Simulation time (hours)",
        ylabel="Python variance / MATLAB variance",
    )
decorate(
    fig,
    "Model differences | compared with seed variability",
    "Mean separation and variance ratio answer different questions; visually overlapping means can coexist with different spread.",
    "Pointwise bootstrap 95% intervals. Variance axes: linear below 0.1, logarithmic above. Undefined ratios omitted; no formal equivalence margins prescribed.",
)
save(fig, "06_bias_and_variance_ratios")

fig, axs = plt.subplots(2, 3, figsize=(16, 10))
for ax, k in zip(
    axs.flat, ["mass", "growth", "rna", "monomers", "complexes", "diameter"]
):
    panel(ax, k, phase_plot=True)
decorate(
    fig,
    "Cycle-aligned behavior | from the first shared sample to division",
    "All 12 cells contribute at every phase. This alignment separates trajectory shape from differences in division time.",
    "Linear interpolation from t=120s to each cell’s actual division; light ±1 SD, dark 95% mean CI. Time warping removes timing differences—use alongside the division-time chart.",
)
save(fig, "07_cycle_aligned_trajectories")

fig, axs = plt.subplots(2, 3, figsize=(16, 10))
for ax, k in zip(
    axs.flat, ["mass", "growth", "rna", "monomers", "complexes", "ploidy"]
):
    met = next(m for m in METRICS if m[0] == k)
    for i, model in enumerate(COL):
        y = endpoints[k][model]
        ax.scatter(
            i + np.linspace(-0.13, 0.13, N),
            np.sort(y),
            color=COL[model],
            s=29,
            alpha=0.85,
        )
        h = student_t.ppf(0.975, N - 1) * y.std(ddof=1) / np.sqrt(N)
        ax.errorbar(i, y.mean(), yerr=h, fmt="D", color="#172B45", capsize=6)
    st = endpointstats[k]
    ax.set(
        xticks=[0, 1],
        xticklabels=list(COL),
        ylabel=met[2],
        title=met[1] + " | at division",
    )
    ax.set_xlim(-0.5, 1.5)
decorate(
    fig,
    "Cell composition | each cell at its own division",
    "Terminal biological records, rather than an arbitrary common clock time.",
    "Dots: individual cells; diamonds: model mean and 95% t CI. Endpoint differences can reflect both distinct trajectories and distinct division timing.",
)
save(fig, "08_composition_at_division")

# Each cell contributes one time-integrated value; no pseudoreplication over time.
aucstats = {
    k: boot_stats(
        np.trapezoid(v["Python"], times, axis=1) / (times[-1] - times[0]),
        np.trapezoid(v["MATLAB"], times, axis=1) / (times[-1] - times[0]),
    )
    for k, v in mats.items()
}
summary["common_window_time_average"] = {
    k: scalar_summary(st) for k, st in aucstats.items()
}
(DERIVED / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
fig, axs = plt.subplots(1, 3, figsize=(17, 8))
keys = [m[0] for m in METRICS]
labels = [m[1] for m in METRICS]
for ax, collection, title, last in [
    (axs[0], aucstats, "Time-averaged over shared window", False),
    (axs[1], stats, "At last common clock time", True),
    (axs[2], endpointstats, "At each cell’s division", False),
]:
    ax.axvline(0, color="#8792A1", ls="--")
    for j, k in enumerate(keys):
        st = collection[k]
        i = -1 if last else 0
        point = st["smd"][i]
        lo, hi = st["smd_ci"][:, i]
        if np.isfinite(point):
            ax.plot([lo, hi], [j, j], color=COL["Python"], lw=2)
            ax.scatter([point], [j], color=COL["Python"], s=45, zorder=3)
        else:
            ax.text(
                0.05,
                j,
                "Constant / undefined",
                fontsize=8,
                color="#68778A",
                va="center",
            )
    ax.set(
        yticks=range(len(keys)),
        yticklabels=labels,
        title=title,
        xlabel="Python − MATLAB, divided by pooled SD",
    )
    ax.invert_yaxis()
decorate(
    fig,
    "Mean differences | all observables against seed variability",
    "Each interval resamples whole cells, independently between simulators. Zero means equal ensemble means.",
    "Time averages use the fixed absolute-time window, not normalized lifetimes. Intervals are descriptive and not adjusted for multiple comparisons.",
)
fig.subplots_adjust(left=0.13, wspace=0.65)
save(fig, "09_all_observable_mean_differences")
fig, axs = plt.subplots(1, 2, figsize=(15, 8))
for ax, collection, title, last in [
    (axs[0], stats, "At last common clock time", True),
    (axs[1], endpointstats, "At each cell’s division", False),
]:
    ax.axvline(1, color="#8792A1", ls="--")
    for j, k in enumerate(keys):
        st = collection[k]
        i = -1 if last else 0
        point = st["variance_ratio"][i]
        lo, hi = st["variance_ratio_ci"][:, i]
        if np.isfinite(point) and point > 0:
            ax.plot([lo, hi], [j, j], color="#7A5AA6", lw=2)
            ax.scatter([point], [j], color="#7A5AA6", s=45, zorder=3)
        else:
            ax.text(
                1.1,
                j,
                "Both constant",
                fontsize=8,
                color="#68778A",
                va="center",
            )
    ax.set_xscale("log")
    ax.set(
        yticks=range(len(keys)),
        yticklabels=labels,
        title=title,
        xlabel="Python variance / MATLAB variance",
    )
    ax.invert_yaxis()
decorate(
    fig,
    "Variance differences | the remaining uncertainty",
    "Ratios above 1 indicate greater Python dispersion. Wide intervals reflect the limited 12-cell samples.",
    "Independent whole-cell bootstrap 95% intervals. Overlap with 1 does not establish equivalence; no acceptable variance-ratio margin was prescribed.",
)
fig.subplots_adjust(left=0.16, wspace=0.65)
save(fig, "10_all_observable_variance_ratios")

pdf.close()
