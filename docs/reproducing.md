# Reproducing a run

## Supported source installation

Follow the README from the repository root. The release includes compressed fixtures, a checksummed GLPK source archive, and a small driver source. `prepare_fixtures.py` expands only known fixtures and verifies their hashes. It refuses altered files. `build_glpk.sh` compiles for the current machine; never transfer a macOS binary to Linux.

For the numerical dependencies used by the fixed-source Linux cohort:

```sh
python -m pip install -c constraints-ubuntu.txt -e '.[validation]'
```

That cohort uses Python 3.11.16, NumPy 2.0.2 and SciPy 1.17.1. Use one BLAS/OpenMP thread per cell when running several cells concurrently. Different operating systems or compiler builds produce different native binary hashes; record the actual binary, not an expected hash from a different platform.

The source package intentionally relies on checkout-relative fixtures and a native executable. Install with `pip install -e`; a wheel alone is not a complete runnable model. The application does not download data or automatically substitute a newer solver.

## Initial state and independent cells

`reference/snapshot_t60.json` is a stored state from the original MATLAB model at 60 simulated seconds. The Python loader restores state and uses independently seeded Python random streams. The same seed label in MATLAB is not a paired trajectory. Use a separate output prefix and seed for every independent cell.

`--steps 50000` from this snapshot stops by t=50,060 unless `--stop-at-division` terminates earlier. A run without division at that horizon is censored, not successful division. A nonzero exit or solver error is a failure, not a biological outcome. Preserve both stdout and stderr.

The original solver infrastructure failure produced a batch with zero metabolism. Those outputs are excluded from this release's biological evidence. Deployment validation must verify actual solver output and initial-state biology, not merely that Python imports successfully.

## Resume a trusted checkpoint

A checkpoint continues the same cell, including its random streams. For example, to advance at most 5,000 more steps and write a separate continuation:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m wholecell_py.runner --resume runs/seed_101.pkl \
  --steps 5000 --stop-at-division \
  --checkpoint runs/seed_101_continued.pkl --checkpoint-every 600 \
  > runs/seed_101_continued.jsonl 2> runs/seed_101_continued.stderr.log
```

Load only trusted pickle checkpoints. The loader checks source identity; `--allow-code-change` overrides that check and should only be used deliberately with the source boundary documented. Retain both output segments and their provenance; continuation is not a new replicate.

## Tests

The portable suite needs only files inside this repository after fixture preparation. It exercises both the default GLPK 4.44 driver and explicit optional swiglpk tests. The optional package is not a runtime fallback for a missing GLPK 4.44 driver.

The included MATLAB exporters accept explicit file arguments. They require an already configured original model and your own MATLAB license. The Python release tests do not execute MATLAB. Cached fixture hashes identify the reference values; fresh differential MATLAB export requires the upstream setup and is not automated by CI.

## Recompute the fixed-code statistics

```sh
python scripts/analyze_ensemble.py
python scripts/analyze_bias.py
python scripts/plot_ensemble.py
```

This consumes `validation/fixed-cohort/trajectories.npz`, an array-only, pickle-free export of all observables used in the stated comparisons. It contains exact shared-time samples, phase-normalized samples and terminal values for the fixed cohorts. The provenance manifest records input hashes, the source identity and MATLAB continuation boundary and the cohort definitions. Outputs go to `validation/generated/`; historical provisional results are not overwritten.

The exported analysis dataset is not a restart checkpoint and does not contain every internal molecular state. It is sufficient to recompute the supplied discrimination and mean/spread summaries, not to reconstruct every full internal rollout.

## Extend the evidence

Publish new results under a new cohort identifier, with source/solver/snapshot/environment hashes and all completed, failed and censored cells accounted for. Keep the mixed-version results available as historical evidence. Choose scientific mean-bias tolerances before confirmatory analysis. Do not select only fast-dividing cells or combine versions to improve agreement.
