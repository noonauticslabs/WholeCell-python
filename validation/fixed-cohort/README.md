# Completed fixed-code cohort

These are measured results from complete Python trajectories on one corrected source version.

- Twelve MATLAB and twelve Python cells, original cohort seeds 101–112. Every cell reached division; none was dropped for its trajectory or outcome.
- Python starts from the common MATLAB t=60 state, with independent NumPy random streams. No Python checkpoint resumptions, source changes, or mixed-version splices were used.
- Runtime source: `d4e29731194d75ed645f117b3ad312742feb0e27d9f3c954eca8537ba756f98a`.
- Snapshot: `ea7c06377a527c8c4cdae6a83abe8caf50af344a00acff9395771f9f9e7f2a96`.
- Linux GLPK 4.44 driver: `4ad091199e2e174aacc1e06a8744c51e30f3469cf08b28ab015f3f16bc9e8ce5`.
- Python 3.11.16, NumPy 2.0.2, SciPy 1.17.1; one BLAS/OpenMP thread per worker.
- MATLAB seed 101 was computed on the earlier host. Seed 109 joins its original branch through t=12,000 to its documented resumed branch afterward, without making a new independent cell.
- The shared grid is t=120–26,700 s every 60 seconds; all twelve cells contribute to every common-time statistic. Independent cells, not timesteps, are the replicates. Seed labels do not establish pairing across simulators.
- Four additional completed Python cells (113–116) and later queued cells are outside the original balanced primary cohort.

`trajectories.npz` contains numeric arrays without pickle objects: common-grid trajectories, division times, endpoints, phase-normalized curves, and downsampled individual replication/constriction curves. `provenance.json` records raw-file hashes and the MATLAB continuation boundary. `summary.json`, `discrimination.json`, `bias-tests.json`, and CSVs contain estimates and method details. Raw logs remain in the private experiment archive; these portable arrays reproduce the public estimates and figures.

Reproduce from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 python scripts/analyze_ensemble.py
python scripts/analyze_bias.py
OPENBLAS_NUM_THREADS=1 python scripts/plot_ensemble.py
```

Outputs go to `validation/generated/`. The historical mixed-version data remain under `validation/provisional/`, with a separate historical analysis script. They are never pooled with this cohort.

The current release also contains an experimental fresh-cell initializer added after this campaign was frozen; it has separate smoke coverage. This cohort validates neither fresh initialization nor every internal biological process. No prespecified equivalence margins or prospective power calculation were used. The finite-sample results quantify mean agreement and uncertainty; they do not prove universal simulator indistinguishability.
