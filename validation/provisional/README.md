# Provisional mixed-version cohort

**Not final fixed-source validation.** These are measured results from earlier Python trajectories with documented corrections during continuation. They are preserved historical evidence, superseded in the README by the separately analyzed fixed-source cohort.

- Python seeds 101–112 and MATLAB seeds 101–112; 12 cells per simulator, all reached division.
- One common biological starting state; independent RNG engines and streams. Seed labels do not establish pairing.
- Shared 60-second samples from t=120 through t=25,620; all 12 cells contribute at every included clock time.
- Python's first retained branch ends at t=2,460; its next branch ends at each cell's audit migration boundary; final rows come from the repaired continuation.
- MATLAB seed 109 uses its original rows through t=12,000 and its resumed branch afterward. It is not a new independent cell.
- The invalid missing-GLPK batch is excluded.

`trajectories.npz` stores the comparison arrays with no Python objects. `provenance.json` records the curated raw-file hashes and boundaries. `summary.json` and `discrimination.json` contain the original estimates and method scope. The CSV files provide division times and time-matched statistics. Bootstrap intervals are exploratory and per-observable.

Run `python scripts/analyze_historical_ensemble.py` from the repository root for the portable reanalysis. The runtime source manifest identifies the final code used at the end of these continuations; it does not imply the entire cohort ran on that source.
