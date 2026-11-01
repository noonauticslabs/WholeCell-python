# Paper figures, animation, and biological validation

Reference: [Karr et al., Cell (2012)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3413483/). The paper includes population growth and division distributions (Figure 2B), normalized mass components (2D), cell-cycle phase distributions (4A), and a supplementary cell-cycle movie. Figure 7 compares gene-disruption predictions with experimental growth measurements.

## Available now

- `scripts/paper_presentation.py` creates full growth curves with division-time histograms, replication/constriction event timelines, normalized cellular-component curves, and quantitative figure PDF. It uses the same twelve cells per simulator as the fixed-code analysis. Run it from the source checkout after installing validation dependencies.
- `scripts/paper_presentation.py` also creates the original 144-frame, white-background ensemble GIF with all 24 cells and their growth/constriction trajectories. Cell size follows dry mass, the neck follows recorded diameter, and the DNA arc encodes ploidy. Completed cells freeze at division; daughter dynamics are not simulated.
- `scripts/chromosome_frame.py` ports the drawing geometry of the original MIT-licensed `ChromosomeAnimation.m`: two chromosome-pair centers, alternating strand radii, genomic-coordinate arcs, full-circle handling, and red bound-protein footprints. It renders actual snapshot/checkpoint state to SVG and PNG. This is a source-derived frame renderer, not a completed port of the original disk-logger/movie pipeline.
- Rendered chromosome examples are the shared MATLAB t=60 state and fixed-code Python seed 101 at division, t=30,713. They are two actual states, not a continuous molecular movie.

```sh
python scripts/paper_presentation.py
python scripts/chromosome_frame.py --snapshot reference/snapshot_t60.json \
  --output validation/generated/chromosome_t60
```

The original project also contains `OverviewAnimation.m`, `CellGeometryAnimation.m`, `Animation.m`, and `runAnimation.m`. The original overview renderer combines many molecular panels and metabolic-map database data; it has not been fully translated. A full chromosome animation needs time histories of `polymerizedRegions`, `monomerBoundSites`, and `complexBoundSites`, plus molecular footprint constants. Aggregate trajectory logs and a terminal checkpoint cannot reconstruct those histories.

## Figure correspondence and limits

| Original panel | Current output | Important distinction |
|---|---|---|
| Figure 2B: growth and division distribution | `12_growth_and_division_paper_style.png` | Twelve cells per model, same t=60 state; lower-median division cell highlighted by a fixed rule. |
| Figure 2D: normalized mass components | `14_normalized_cell_components.png` | RNA/protein curves are molecule counts; they are not the paper's chemical mass fractions. |
| Figure 4A: cell-cycle timing | `13_cell_cycle_milestones.png` | We observe remaining time to ploidy=2 and constriction to division; all 24 cells were already replicating at their first record. Initiation durations cannot be recovered. |
| Chromosome movie | Source-derived initial/final frame examples | Intermediate molecular occupancy was not logged; no molecular motion is fabricated. |
| Metabolic fluxes, DnaA/dNTP dynamics, gene-specific bursts | Not reconstructed | The required species-level histories are absent from this cohort's aggregate logs. |
| Figure 7: disrupted-strain growth versus experiment | Not reproduced | Needs the specified perturbations, mutant ensembles, experimental measurements and matching analysis. |

`validation/fixed-cohort/cycle-events.json` stores event brackets on native recording cadence. DNA completion is the first recorded ploidy >= 2−1e−9. Visible constriction onset is an operational threshold: diameter below 99% of its first recorded value. The latter is not the original cytokinesis-stage variable. Python records every second; MATLAB records every 60 seconds, so event precision differs. No claims of equal phase-duration distributions follow from these charts.

## Gene-disruption validation is a separate task

The quoted experimental result is not supplied by a wild-type port comparison. The original source has `Simulation.applyPerturbationsToConstants`, `applyPerturbationsToState`, and `SingleGeneDeletions` analysis. No equivalent end-to-end knockout interface is currently wired into the Python runner. Reproducing the result requires porting and checking the exact perturbation semantics, identifying the published strains and measurements, running independent mutant cohorts with failures/nondivision retained, and comparing their growth under the same definitions. Later kinetic modifications in the paper must be distinguished from the baseline parameter set, rather than tuned until an expected plot appears.

The fresh initializer is already implemented separately: both `multinomial` and `expected` modes sampled new t=0 populations from fitted constants, passed the growth filter on their first attempt, saved/restored checkpoints, and advanced one timestep. That is a capability smoke test, not a mutant study or an initialization ensemble. The observed attempts took about 13 seconds each including the checkpoint and first step.
