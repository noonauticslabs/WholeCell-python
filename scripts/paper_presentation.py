"""Paper-inspired cohort figures and a schematic animation of recorded data.

No new simulation. No invented molecular positions or post-division dynamics.
"""

from pathlib import Path
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Arc
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "validation/generated/paper_figures"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "validation/fixed-cohort"
with np.load(DATA / "presentation-trajectories.npz", allow_pickle=False) as z:
    curves = {
        m: [z[f"{m}_{s}"].copy() for s in range(101, 113)]
        for m in ("MATLAB", "Python")
    }
events = json.loads((DATA / "cycle-events.json").read_text())["events"]
colors = {"MATLAB": "#D78A25", "Python": "#2676C9"}
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.17,
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFCFF",
        "pdf.fonttype": 42,
    }
)
pdf = PdfPages(OUT / "Paper_inspired_comparisons.pdf")


def finish(fig, name, title, note):
    fig.suptitle(
        title,
        x=0.07,
        ha="left",
        fontsize=21,
        fontweight="bold",
        color="#172B45",
    )
    fig.text(0.07, 0.025, note, fontsize=9, color="#526379")
    fig.tight_layout(rect=(0.02, 0.075, 0.99, 0.93))
    fig.savefig(OUT / (name + ".png"), dpi=160)
    pdf.savefig(fig)
    plt.close(fig)


fig, axs = plt.subplots(
    2, 2, figsize=(13, 10), gridspec_kw={"height_ratios": [3, 1]}
)
bins = np.linspace(7.2, 9, 10)
for j, (m, rows) in enumerate(curves.items()):
    ends = np.array([a[-1, 0] for a in rows]) / 3600
    representative = np.argsort(ends)[
        5
    ]  # Lower median division-time cell, rule declared.
    for i, a in enumerate(rows):
        axs[0, j].plot(
            a[:, 0] / 3600, a[:, 1] * 1e15, color=colors[m], alpha=0.35, lw=1
        )
    a = rows[representative]
    axs[0, j].plot(
        a[:, 0] / 3600,
        a[:, 1] * 1e15,
        color="#172B45",
        lw=2,
        label=f"Lower-median division: seed {101+representative}",
    )
    axs[0, j].set(
        title=m + " | all 12 complete growth curves",
        xlabel="Simulation time (h)",
        ylabel="Dry mass (fg)",
        xlim=(0, 9),
        ylim=(3.8, 8.8),
    )
    axs[0, j].legend(frameon=False, fontsize=9)
    axs[1, j].hist(
        ends, bins=bins, color=colors[m], alpha=0.8, edgecolor="white"
    )
    axs[1, j].plot(ends, np.full(12, -0.15), "|", color="#172B45", ms=12)
    axs[1, j].axvline(
        ends.mean(),
        color="#172B45",
        ls="--",
        label=f"Mean {ends.mean():.3f} h",
    )
    axs[1, j].set(
        xlabel="Division time (h)",
        ylabel="Cells",
        xlim=(7.2, 9),
        ylim=(-0.4, 6),
    )
    axs[1, j].legend(frameon=False)
finish(
    fig,
    "12_growth_and_division_paper_style",
    "Growth to division | paper-inspired population view",
    "Inspired by Figure 2B. Fixed t=60 starting state; n=12 each, not the original paper population. Every curve ends at its own division.",
)

fig, axs = plt.subplots(2, 2, figsize=(14, 11))
for j, m in enumerate(curves):
    es = [e for e in events if e["model"] == m]
    for i, e in enumerate(es):
        end = e["division_s"] / 3600
        axs[0, j].plot(
            [60 / 3600, end], [i, i], color=colors[m], alpha=0.35, lw=5
        )
        for field, marker, c in [
            ("dna_completion", "s", "#7E57A3"),
            ("constriction_onset", "^", "#26A184"),
        ]:
            lo, hi = np.array(e[field + "_bracket_s"]) / 3600
            axs[0, j].plot([lo, hi], [i, i], color=c, lw=3)
            axs[0, j].scatter(
                hi,
                i,
                marker=marker,
                color=c,
                s=33,
                label=field.replace("_", " ") if i == 0 else None,
                zorder=4,
            )
        axs[0, j].scatter(
            end,
            i,
            marker="o",
            color=colors[m],
            s=30,
            label="Division" if i == 0 else None,
            zorder=4,
        )
    axs[0, j].set(
        title=m + " | recorded milestones",
        yticks=range(12),
        yticklabels=range(101, 113),
        xlabel="Simulation time (h)",
        ylabel="Seed",
        xlim=(0, 9),
    )
    axs[0, j].legend(frameon=False, fontsize=8, loc="upper left")
    axs[0, j].invert_yaxis()
    for k, (field, title) in enumerate(
        [
            ("dna_completion", "Remaining time to DNA completion"),
            ("constriction_onset", "Visible constriction to division"),
        ]
    ):
        vals = (
            np.array(
                [
                    (
                        (e[field + "_bracket_s"][1] - 60)
                        if k == 0
                        else (e["division_s"] - e[field + "_bracket_s"][1])
                    )
                    for e in es
                ]
            )
            / 3600
        )
        axs[1, k].scatter(
            j + np.linspace(-0.13, 0.13, 12), vals, color=colors[m], s=30
        )
        axs[1, k].hlines(vals.mean(), j - 0.2, j + 0.2, color="#172B45", lw=2)
        axs[1, k].set(
            xticks=[0, 1],
            xticklabels=list(curves),
            ylabel="Duration (h)",
            title=title,
        )
finish(
    fig,
    "13_cell_cycle_milestones",
    "Cell-cycle timing | replication, constriction, division",
    "DNA completion: first ploidy=2. Constriction: diameter <99% of starting value. Native sampling brackets retained.\nAll cells were already replicating at first observation; these are not the paper’s initiation/replication phase-duration distributions.",
)

fig, axs = plt.subplots(1, 2, figsize=(14, 7))
palette = ["#172B45", "#C34660", "#28A176", "#2876C9", "#9767AE"]
for ax, (m, rows) in zip(axs, curves.items()):
    ends = np.array([a[-1, 0] for a in rows])
    i = np.argsort(ends)[5]
    a = rows[i]
    for col, label, c in zip(
        [1, 6, 3, 4, 5],
        [
            "Dry mass",
            "DNA equivalents",
            "RNA count",
            "Monomer count",
            "Complex count",
        ],
        palette,
    ):
        ax.plot(
            a[:, 0] / 3600, a[:, col] / a[0, col], lw=2, color=c, label=label
        )
    ax.set(
        title=f"{m} | lower-median division cell, seed {101+i}",
        xlabel="Simulation time (h)",
        ylabel="Fold change from first saved t=120 s sample",
        xlim=(0, 9),
        ylim=(0.8, 2.6),
    )
    ax.legend(frameon=False, fontsize=9)
finish(
    fig,
    "14_normalized_cell_components",
    "Coordinated growth | normalized cellular components",
    "Inspired by Figure 2D. RNA/protein curves are molecule counts, not chemical mass fractions. Representative-cell selection uses division time only.",
)
pdf.close()

# Animated paired cohort: glyph geometry is explicitly schematic; values are data.
fig = plt.figure(figsize=(12, 8), dpi=90)
gs = fig.add_gridspec(2, 2, height_ratios=[1.12, 1], hspace=0.38, wspace=0.22)
cellaxes = [fig.add_subplot(gs[0, j]) for j in range(2)]
plots = [fig.add_subplot(gs[1, j]) for j in range(2)]
fig.subplots_adjust(top=0.86, bottom=0.18, left=0.07, right=0.97)
fig.text(
    0.07,
    0.965,
    "Two simulators. Independent cells. One cell cycle.",
    fontsize=19,
    weight="bold",
    color="#172B45",
)
clock = fig.text(0.07, 0.915, "", fontsize=12, color="#526379")
fig.text(
    0.07,
    0.055,
    "Schematic glyphs: size follows dry mass; neck follows recorded diameter; purple arc shows DNA replication.\nRecorded trajectories, interpolated for playback. Completed cells freeze at division; daughters are not simulated.",
    fontsize=9,
    color="#526379",
)
glyphs = {}
lines = {}
counts = {}
for ax, (m, rows) in zip(cellaxes, curves.items()):
    ax.set(xlim=(-0.6, 3.6), ylim=(-0.55, 2.6))
    ax.set_aspect("equal")
    ax.axis("off")
    counts[m] = ax.text(
        -0.5, 2.55, m, color=colors[m], fontsize=14, weight="bold"
    )
    for i, a in enumerate(rows):
        cx = i % 4
        cy = 2 - i // 4
        poly = Polygon(
            np.zeros((120, 2)),
            closed=True,
            facecolor=colors[m],
            alpha=0.25,
            edgecolor=colors[m],
            lw=1.5,
        )
        ax.add_patch(poly)
        arc = Arc(
            (cx, cy), 0.30, 0.30, theta1=90, theta2=90, color="#7E57A3", lw=2
        )
        ax.add_patch(arc)
        label = ax.text(
            cx,
            cy - 0.35,
            str(101 + i),
            ha="center",
            fontsize=8,
            color="#526379",
        )
        glyphs[(m, i)] = (poly, arc, label, cx, cy)
for ax, col, title, ylabel, scale in [
    (plots[0], 1, "Complete growth trajectories", "Dry mass (fg)", 1e15),
    (plots[1], 7, "Constriction and division", "Septum diameter (nm)", 1e9),
]:
    ax.set(
        title=title,
        xlabel="Simulation time (h)",
        ylabel=ylabel,
        xlim=(0, 9),
        ylim=(3.8, 8.8) if col == 1 else (-8, 310),
    )
    for m, rows in curves.items():
        for i, a in enumerate(rows):
            (line,) = ax.plot(
                [],
                [],
                color=colors[m],
                alpha=0.55,
                lw=1,
                label=m if i == 0 else None,
            )
            lines[(col, m, i)] = line
    ax.legend(frameon=False, fontsize=8)
cursors = [
    ax.axvline(0, color="#172B45", ls="--", lw=1, alpha=0.55) for ax in plots
]
frames = []
max_t = max(a[-1, 0] for rows in curves.values() for a in rows)
for frame, t in enumerate(np.linspace(120, max_t, 144)):
    clock.set_text(
        f"Simulation clock  {int(t//3600):02d}:{int(t%3600//60):02d} h   |   12 MATLAB + 12 Python cells"
    )
    for m, rows in curves.items():
        counts[m].set_text(f"{m}   {sum(t>=a[-1,0] for a in rows)}/12 divided")
        for i, a in enumerate(rows):
            val = np.array(
                [np.interp(t, a[:, 0], a[:, k]) for k in range(1, 8)]
            )
            done = t >= a[-1, 0]
            poly, arc, label, cx, cy = glyphs[(m, i)]
            size = np.sqrt(val[0] / a[0, 1])
            neck = np.clip(val[6] / a[0, 7], 0, 1)
            x = np.linspace(-1, 1, 60)
            height = np.sqrt(np.maximum(0, 1 - x * x)) * (
                neck + (1 - neck) * np.abs(x) ** 0.55
            )
            xx = x * 0.23 * size
            yy = height * 0.18 * size
            poly.set_xy(
                np.column_stack(
                    [np.r_[xx, xx[::-1]] + cx, np.r_[yy, -yy[::-1]] + cy]
                )
            )
            poly.set_alpha(0.12 if done else 0.30)
            arc.theta2 = 90 + 360 * np.clip(val[5] - 1, 0, 1)
            label.set_text(f"{101+i}" + ("  divided" if done else ""))
            for col, scale in [(1, 1e15), (7, 1e9)]:
                keep = a[:, 0] <= t
                ts = a[keep, 0]
                ys = a[keep, col]
                if not done:
                    ts = np.r_[ts, t]
                    ys = np.r_[ys, np.interp(t, a[:, 0], a[:, col])]
                lines[(col, m, i)].set_data(ts / 3600, ys * scale)
    for cursor in cursors:
        cursor.set_xdata([t / 3600, t / 3600])
    fig.canvas.draw()
    im = Image.fromarray(
        np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    ).convert("P", palette=Image.Palette.ADAPTIVE, colors=128)
    frames.append(im)
    if frame in (0, 72, 143):
        im.convert("RGB").save(OUT / f"animation_preview_{frame:03d}.png")
frames[0].save(
    OUT / "whole_cell_comparison.gif",
    save_all=True,
    append_images=frames[1:],
    duration=[85] * 143 + [1400],
    loop=0,
    optimize=False,
)
plt.close(fig)
print(
    json.dumps(
        {"figures": 3, "gif_frames": len(frames), "output": str(OUT)}, indent=2
    )
)
