# Validation design

## Primary scientific question

Given the same biological initial state but an unknown random seed, does simulator identity leave a systematic, reproducible mean signature beyond within-simulator variation?

This is stochastic equivalence assessment, not an experiment that removes stochasticity. Removing noise would change the target algorithm.

## Evidence levels

1. **Source correspondence:** identify the original methods and library contracts; retain unresolved adapter differences.
2. **Primitive and process tests:** compare deterministic cases, shapes, ordering, stochastic distribution parameters, and selected MATLAB exports. Exercise active branches rather than count unchanged zeros as decisive agreement.
3. **Integrated smoke tests:** verify native dependencies and live growth, mass, counts, replication, warnings, and solver output.
4. **Fixed-version ensemble:** independent seeded cells from the common state. Keep output directories immutable with source, solver, environment and snapshot hashes.
5. **Biological generalization:** different initial conditions, perturbations, and genotypes. This preview does not establish that level.

## Statistical unit and comparisons

A cell is an independent replicate; its timesteps are correlated. Equal seed numbers do not pair MATLAB and Python outcomes. The fixed-code common-time window is t=120–26,700 s at 60 s intervals, retaining all 12 cells per simulator. Time averages integrate over that same window. Terminal comparisons use each cell's actual division. Phase-normalized plots remove timing differences and are descriptive supplements.

Primary estimates are signed relative mean differences and their independent whole-cell bootstrap intervals. Standardized differences use the pooled within-model SD. Report raw units as well: a small standardized effect can result from large seed variation.

Spread is secondary: report SDs and variance-ratio intervals rather than infer equality from overlapping shaded envelopes. Report event times and failures separately. No survivor-only mean after the earliest division is used for the fixed-cohort comparison.

## Simulator identification

The fixed-code analysis reuses three feature sets selected for the historical analysis, before evaluating the fixed cohort: common-window averages plus division time; division endpoint composition plus division time; and six observables at twelve shared clock times. Each has a nearest-centroid rule and an RBF similarity rule. Scaling and RBF bandwidth are calculated within each training fold. Seed identifiers, file paths, output cadence and log lengths are excluded.

Leave one whole cell out and refit using the other 23. Repeat the entire fit/predict calculation for 9,999 balanced label permutations. A maximum-accuracy permutation comparison accounts for inspecting six combinations. Cross-validation folds share training data, so a naive binomial confidence interval for the 24 predictions is not reported. The best tested rule does not bound the best possible discriminator.

## What would justify equivalence

Choose biological mean-bias tolerances before confirmatory analysis, specify primary observables/time summaries, and require confidence intervals to fit inside those margins. Address multiplicity for claims covering many observables or times. Specify spread tolerances separately if distributional equivalence is also claimed. A nonsignificant difference test or classification result alone cannot establish equivalence.

The fixed-code bootstrap intervals are per-observable estimates. Neither an all-time equivalence test nor a uniform upper bound on simulator-identification accuracy has been established. No prospective power calculation was used to choose the displayed n=12 cohorts.

## References

- [SciPy: independent-sample permutation tests](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html)
- [Statsmodels: two one-sided equivalence testing](https://www.statsmodels.org/v0.13.5/generated/statsmodels.stats.weightstats.ttost_ind.html)

The supplied script computes bootstrap intervals and label-permutation tests; it does not invoke the Statsmodels TOST implementation. That reference explains why equivalence requires a margin rather than a failure to reject equality.

## Completed fixed-code evidence

The original twelve Python seeds (101–112) all completed on source `d4e29731194d75ed645f117b3ad312742feb0e27d9f3c954eca8537ba756f98a`, with no Python continuation or source change. Every output was checked for contiguous steps from t=61 to division, finite nonnegative plotted observables, terminal division, zero constriction diameter, positive growth/mass, and absence of fatal stderr markers. Snapshot and solver hashes match across cells. The original twelve MATLAB trajectories remain the reference; seed 101 was computed on the earlier host, and seed 109 includes its documented same-model checkpoint continuation. Those are provenance differences, not additional independent cells.

Relative mean and variance-ratio intervals use 20,000 independent whole-cell bootstrap resamples; plotted timewise estimates use 4,000. The division-time difference interval quoted in the README uses the 4,000-resample plot analysis, while its relative difference and variance ratio use the 20,000-resample summary. Six common-window averages plus division time also have Welch mean-difference tests, with Holm correction across all seven. The latter is a mean-focused diagnostic added for this analysis, not a preregistered equivalence test. All seven adjusted p values are 1.0. The six classifier rules, features, preprocessing and permutation procedure were retained from the historical analysis before inspecting this cohort.

The cohort was originally designated for comparison before completion. All twelve cells are retained; no failures or censored cells were discarded from this cohort. Four additional completed Python cells and later queued cells are excluded from the balanced primary analysis by seed designation, not by their outcomes. More Python cells alone cannot eliminate uncertainty from the twelve-cell MATLAB sample.

Statistical validity does not imply high power: mean confidence intervals quantify residual uncertainty, and variance intervals remain broad. Mean bias, distributional equivalence and universal inability to identify the simulator are different claims.
