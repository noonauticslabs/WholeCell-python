# Scope and remaining work

This preview supports coupled simulation from a supplied, fitted MATLAB state with 28 process categories and 16 state categories. It does not certify every original method, perturbation, or initialization path.

- **Fixed-code analysis has finite scope.** The displayed 12+12 cohort now contains complete fixed-source Python trajectories. Close aggregate means and nonsignificant classification do not establish formal equivalence, all-path correctness, or biological generalization.
- **Fresh cell sampling is experimental.** The simulation-level initializer now samples new cells from supplied fitted constants and applies the original growth rejection filter. It has no initialization-ensemble or full-cycle validation. Raw knowledge-base model construction and the full `FitConstants` workflow remain incomplete; they are separate from fresh cell sampling.
- **Compatibility APIs are partial.** Sparse and circular sparse coordinate APIs work in the tested model paths; full MATLAB range-selection, assignment/value semantics, and character-array output shapes are not universal replacements.
- **Explicit solver API differences remain.** The general wrapper normalizes lower-case type codes and uses positive infinity for omitted upper bounds; the historical source default differs. The model supplies its own bounds. Alternative LP backends are opt-in and have separate numerical behavior.
- **Rare and extended state regimes need more evidence.** Chromosome damage configurations, multiple time slices, and some KB-derived methods have limited or synthetic test coverage. Explicit unsupported-operation errors remain visible.
- **Warnings are not proof of harmlessness.** Some numerical/deprecation warnings occur in existing tests. Preserve them and inspect their origin; do not suppress them to create a clean report.
- **Same snapshot is not varied initialization.** Ensemble inference is conditional on one biological starting state. It does not cover the original paper's full initialization variability or gene-deletion study.

## Source audit

The records in `validation/audit/` identify source comparisons and unresolved adapter differences. A matched entry applies only to its stated comparison and fixture scope; it does not establish behavior over every possible state.
