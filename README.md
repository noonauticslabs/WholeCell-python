# WholeCell-python

An independent Python port of the Karr et al. (2012) *Mycoplasma genitalium* whole-cell model. WholeCell-python preserves the original model's biological logic, 28 processes and 16 state categories, and runs with supplied fitted model data without MATLAB. Noonautics Labs maintains the port; Karr and colleagues created the original model.

[Results](#fixed-cohort-comparison) · [Model](#model-and-implementation) · [Install and run](#install-and-run) · [Animation and figures](#growth-and-division) · [Reproduce](docs/reproducing.md) · [Citation](#citation-and-licenses)

## Fixed-cohort comparison

Twelve Python and twelve licensed MATLAB reference simulations, seeds 101–112, started from the same biological state at **t = 60 seconds** with independent random streams. All 24 cells divided. Each Python cell used the same corrected source version throughout, without checkpoint splices or code changes. Whole cells are the statistical replicates; matching seed labels do not pair trajectories.

![Growth, composition and replication across the completed MATLAB and Python ensembles](docs/figures/01_full_cycle_overview.png)

*Completed fixed-cohort ensembles. See [data, provenance and results](validation/fixed-cohort/README.md) for definitions and uncertainty.*

| Measure | MATLAB | Python | Difference |
| --- | ---: | ---: | --- |
| Mean division time | 8.032 h | 7.968 h | −3.80 min (95% whole-cell bootstrap interval −21.34 to +13.43 min) |
| Division endpoint mass | — | — | −0.12% (95% interval −0.65% to +0.39%) |

Over the common t = 120–26,700 s window, Python relative means differed by +0.53% for dry mass, +1.19% for growth, +0.10% for RNA, +0.57% for protein monomers, −0.09% for protein complexes and +0.45% for DNA replication.

The best of six previously selected classifier/feature combinations identified 14 of 24 cells, with multiplicity-adjusted permutation p ≈ 0.71; the common-window averages plus division time nearest-centroid rule identified 12 of 24. These tested rules detected no reliable separation. This does not establish universal indistinguishability or equivalence: the comparison is conditional on one starting state and the measured observables. Four additional completed Python cells were excluded by the original cohort designation. [Validation design](docs/validation.md)

<details>
<summary>Identification accuracy and uncertainty</summary>

![Simulator identification accuracy and uncertainty in mean differences](docs/figures/11_simulator_discrimination.png)

Whole cells are independent replicates in the identification and mean-difference analyses.

</details>

## Model and implementation

The [original WholeCell project](https://github.com/CovertLab/WholeCell) integrates molecular processes into a single cell model. This port traces its classes, methods, hooks, utilities and native boundaries, using original utility source and MathWorks documentation to reproduce the needed compatibility semantics. It preserves fitted data, algorithms, state and process responsibilities, and execution order.

The [scheduler](wholecell_py/simulation.py) coordinates [processes](wholecell_py/processes/) and [states](wholecell_py/states/). Compatibility components handle MATLAB indexing, dimensions and column-major order, [sparse](wholecell_py/sparsemat.py) and [circular sparse](wholecell_py/circular_sparsemat.py) arrays, [random sampling](wholecell_py/randstream.py), sequence and chromosome operations. The [solver boundary](wholecell_py/solvers.py) uses GLPK 4.44 as its principal backend; [loading](wholecell_py/loader.py) and [checkpointing](wholecell_py/checkpoint.py) restore model state. Deterministic compatibility tests and stochastic ensemble comparisons address different questions; independent random streams cannot provide an exact paired-trajectory oracle.

[MATLAB compatibility](docs/matlab-compatibility.md) · [Runtime contracts](docs/runtime-contracts.md) · [Reproduction instructions](docs/reproducing.md)

## Install and run

Use a source checkout on Linux or macOS with Python 3.11, a C compiler, `make` and `tar`. MATLAB is not required with the supplied fitted data; creating new MATLAB exports requires the original setup and a license. The supported installation is editable: a standalone wheel does not contain the complete fixtures and native driver.

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[validation]'
python scripts/prepare_fixtures.py
./scripts/build_glpk.sh
python scripts/verify_installation.py
mkdir -p runs
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m wholecell_py.runner \
  --snapshot reference/snapshot_t60.json --seed 101 \
  --steps 50000 --stop-at-division \
  --checkpoint runs/seed_101.pkl --checkpoint-every 600 \
  > runs/seed_101.jsonl 2> runs/seed_101.stderr.log
```

Use separate seeds and output files for independent cells. Inspect stdout and stderr as well as growth, mass, replication and division. Nondivision at the step limit is censored; an error is a failure. Load pickle checkpoints only from trusted sources. [Reproduction instructions](docs/reproducing.md) describe dependency versions, fixture checksums, reanalysis and provenance.

## Growth and division

![Individual growth trajectories and division-time distributions for MATLAB and Python](docs/figures/12_growth_and_division_paper_style.png)

*Individual growth curves and division distributions for the fixed cohort.*

![Animated comparison of 12 MATLAB and 12 Python cells through growth and division](docs/figures/whole_cell_comparison.gif)

*The 24 independent simulations start from the same biological state. Cell geometry and DNA placement are schematic; completed cells freeze at division.* [Watch the MP4](docs/figures/whole_cell_comparison.mp4)

<details>
<summary>Explore cell-cycle alignment, milestones and composition</summary>

![Cellular trajectories aligned to fractional progress through each completed cell cycle](docs/figures/07_cycle_aligned_trajectories.png)

*Alignment compares biological progression across cells with different division times.*

![Replication, constriction and division milestones](docs/figures/13_cell_cycle_milestones.png)

![Normalized cellular components over the cell cycle](docs/figures/14_normalized_cell_components.png)

</details>

These panels use recorded fixed-cohort observables. The paper-inspired panels are not exact reproductions of the original paper's analyses. [Figure methods and scope](docs/paper-reproduction.md)

## Scope and performance

Raw knowledge-base loading and full `fitConstants` are incomplete; fitted data are required. Fresh initialization has targeted tests and smoke coverage but no full-cycle ensemble validation. Rare compatibility paths, perturbations and internal states remain incompletely validated, and the paper's knockout studies and all figures have not been reproduced. See [limitations](docs/limitations.md) and [initialization](docs/initialization.md). Historical mixed-version evidence is [separate](validation/provisional/README.md).

On Ubuntu, 16 completed Python cells had a median wall time of 6.01 h; 10 uninterrupted MATLAB cells had a median of 2.40 h, with a median throughput ratio of about 2.48 in MATLAB's favor. Concurrency, logging and background load differed, so this is not a controlled same-resource benchmark. [Timing records](validation/ubuntu-full-cycle-timing.json). Earlier paired 20-step Python optimization measurements were approximately 1.26–1.30×; they are neither full-cycle nor MATLAB speedup measurements.

## Citation and licenses

Karr JR et al. “A Whole-Cell Computational Model Predicts Phenotype from Genotype.” *Cell* 150(2), 389–401 (2012). [doi:10.1016/j.cell.2012.05.044](https://doi.org/10.1016/j.cell.2012.05.044). Cite the original model and identify the version or source digest of this port; see [CITATION.cff](CITATION.cff).

The port is distributed under [MIT](LICENSE), with the [original WholeCell notice](LICENSES/WholeCell-MIT.txt) retained. GLPK and its native driver have separate GPL terms; see [third-party notices](THIRD_PARTY_NOTICES.md).
