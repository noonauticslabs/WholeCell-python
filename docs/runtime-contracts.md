# Architecture and runtime contracts

The model is organized into 28 biological processes and 16 state categories. The [runtime registry](../wholecell_py/runtime_registry.py) constructs these components; the [simulation](../wholecell_py/simulation.py) coordinates their execution in source order. Processes exchange molecular quantities through shared state, with allocation and synchronization following the original model's ownership and process boundaries. The [process base](../wholecell_py/process_base.py) and [state implementations](../wholecell_py/states/) define these interfaces.

## Arrays, coordinates and ownership

Snapshots retain source dimensions, species, compartment and time axes. MATLAB-facing coordinates are one-based and linear indexing uses column-major order; Python operations convert indices at explicit boundaries. Shape matters: a column vector, scalar and state tensor are not interchangeable. Value-like operations preserve independent ownership where the original model requires it, avoiding mutable aliases between state and temporary results.

[SparseMat](../wholecell_py/sparsemat.py) and [CircularSparseMat](../wholecell_py/circular_sparsemat.py) implement the sparse and circular coordinate operations needed by model paths. The [chromosome state](../wholecell_py/states/chromosome.py) and associated modules handle chromosome geometry, binding and sequence queries. [Sequence operations](../wholecell_py/chromosome_sequence.py) retain source coordinate conventions. These are targeted compatibility implementations rather than general MATLAB replacements; see [limitations](limitations.md).

## Randomness and numerical boundaries

[RandStream](../wholecell_py/randstream.py) supplies the model's random sampling interfaces. Python and MATLAB use independent random streams, so equal seed labels do not prescribe equal random draws. Changes to draw order or count can change trajectories; deterministic source comparisons and independent ensemble validation answer different questions.

The [solver interface](../wholecell_py/solvers.py) invokes the locally built GLPK 4.44 driver for the principal metabolic linear-programming backend. Its options, numeric precision and matrix insertion order are part of numerical provenance. A solver infrastructure error is fatal, rather than treated as a biological outcome. Alternative backends have distinct numerical behavior and are not validated substitutes.

## Loading and checkpoints

The [loader](../wholecell_py/loader.py) restores the fitted object graph and state from exported fixtures; the [runner](../wholecell_py/runner.py) advances it and records observables. [Checkpointing](../wholecell_py/checkpoint.py) preserves state and random streams for continuation and checks source identity. Checkpoints are Python pickle files and must be trusted before loading. A resumed checkpoint is a continuation of the same cell, not an independent replicate.

The supplied t=60 snapshot supports the fixed-cohort comparison; it does not establish complete raw knowledge-base fitting or fresh-initialization validation. See [initialization](initialization.md), [reproduction](reproducing.md) and [validation](validation.md).
