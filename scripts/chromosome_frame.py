"""Frame renderer ported from WholeCell ChromosomeAnimation.m (MIT).

Original author: Jonathan Karr, Covert Lab, Stanford University, 2011.
Preserves chromosome-pair centers, strand radii, coordinate angles, and
protein-footprint arcs. The model supplies actual genomic coordinates.
"""

from pathlib import Path
import argparse
import json
import math
import numpy as np
from PIL import Image, ImageDraw


def frame_data(graph):
    chromosome = graph.states["Chromosome"]
    regions, lengths = chromosome.polymerizedRegions.find()
    regions = np.asarray(regions).reshape(-1, np.asarray(regions).shape[-1])
    arcs = []
    for row, length in zip(regions, np.asarray(lengths).ravel()):
        arcs.append((int(row[0]), int(row[1]), float(length), "black", 2))
    for field, footprints in [
        ("monomerBoundSites", "monomerDNAFootprints"),
        ("complexBoundSites", "complexDNAFootprints"),
    ]:
        coords, ids = getattr(chromosome, field).find()
        sizes = np.asarray(getattr(chromosome, footprints)).ravel(order="F")
        for row, idx in zip(np.asarray(coords), np.asarray(ids).ravel()):
            arcs.append(
                (
                    int(row[0]),
                    int(row[1]),
                    float(sizes[int(idx) - 1]),
                    "red",
                    1,
                )
            )
    return dict(
        time_s=float(np.asarray(graph.states["Time"].values).item()),
        sequence_length=int(np.asarray(chromosome.sequenceLen).item()),
        arcs=arcs,
    )


def arc_geometry(position, strand, length, sequence_length, protein=False):
    radii = (90, 110) if protein else (95, 105)
    return (
        150 if strand <= 2 else 450,
        150,
        radii[int(strand) % 2 == 0],
        position / sequence_length * 2 * math.pi - math.pi / 2,
        length / sequence_length * 2 * math.pi,
    )


def render(data, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="300" viewBox="0 0 600 300">',
        '<rect width="600" height="300" fill="white"/>',
    ]
    image = Image.new("RGB", (1200, 600), "white")
    draw = ImageDraw.Draw(image)
    for pos, strand, length, color, width in data["arcs"]:
        cx, cy, r, theta, delta = arc_geometry(
            pos, strand, length, data["sequence_length"], color == "red"
        )
        box = [2 * (cx - r), 2 * (cy - r), 2 * (cx + r), 2 * (cy + r)]
        if length == data["sequence_length"]:
            svg.append(
                f'<circle cx="{cx}" cy="{cy}" r="{r}" stroke="{color}" stroke-width="{width}" fill="none"/>'
            )
            draw.ellipse(box, outline=color, width=2 * width)
        else:
            x0, y0 = cx + r * math.cos(theta), cy + r * math.sin(theta)
            x1, y1 = cx + r * math.cos(theta + delta), cy + r * math.sin(
                theta + delta
            )
            svg.append(
                f'<path d="M{x0},{y0} A{r},{r} 0 {int(delta>=math.pi)},1 {x1},{y1}" stroke="{color}" stroke-width="{width}" fill="none"/>'
            )
            draw.arc(
                box,
                math.degrees(theta),
                math.degrees(theta + delta),
                fill=color,
                width=2 * width,
            )
    svg.append("</svg>")
    output.with_suffix(".svg").write_text("\n".join(svg) + "\n")
    image.save(output.with_suffix(".png"))
    output.with_suffix(".json").write_text(json.dumps(data, indent=2) + "\n")


def main():
    from wholecell_py.runner import load_model, SimulationContext

    # CLI-created checkpoints refer to the runner class in __main__.
    import __main__

    if not hasattr(__main__, "SimulationContext"):
        __main__.SimulationContext = SimulationContext
    from wholecell_py.checkpoint import load_checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot", type=Path)
    source.add_argument(
        "--trusted-checkpoint",
        type=Path,
        help="Read only a trusted locally generated checkpoint; permits source differences for visualization, never evolves it",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    graph = (
        load_model(args.snapshot)
        if args.snapshot
        else load_checkpoint(args.trusted_checkpoint, allow_code_change=True)
    )
    data = frame_data(graph)
    render(data, args.output)
    print(
        json.dumps(
            {
                "time_s": data["time_s"],
                "arcs": len(data["arcs"]),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
