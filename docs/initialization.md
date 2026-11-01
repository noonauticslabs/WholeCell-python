# Fresh cell initialization

`--initialize-from` samples a new cell using the supplied snapshot's fitted model constants and object graph. It calls the state/process allocators to discard dynamic populations, resets time to zero, and follows the original `Simulation/initializeState.m` sequence:

1. Draw the initial mass multiplier and distribute biomass.
2. Initialize the chromosome and account for DNA and methylated bases.
3. Sample RNA and protein populations using the source's `multinomial` or `expected` mode; subtract their constituents from the metabolite pool. Expected mode also allocates protein complexes over interacting subunit networks.
4. Set nucleotide pools, initialize geometry and other state, and apply medium/stimulus conditions.
5. Initialize all 28 processes in `processInitOrderIndexs`, copying state and applying side effects in order.
6. Clear aborted-transcript/peptide records, recalculate mass and volume, and synchronize processes.
7. Accept only if relative initial growth error is within `initialGrowthFilterWidth`; otherwise draw another seed and repeat.

The Python implementation bounds attempts (default 100) instead of unbounded recursive retries. It raises on exhaustion, nonfinite growth, invalid final molecule counts, or an implementation error. It does not silently treat a programming failure as a biological rejection. The expected-mode complex-allocation loop also has an explicit convergence guard. Successful attempts and rejected growth values are retained in initialization provenance; trusted native checkpoints preserve that provenance and RNG state.

RNG streams use the runner's independent NumPy child-stream policy. This does not reproduce MATLAB random draws or establish identical joint initialization distributions. The two modes are source options, not interchangeable stochastic models.

## Evidence and limitations

Both modes have completed a bounded Ubuntu initialization smoke test using seed 101 and the original GLPK 4.44 driver. Each passed the original growth filter on its first attempt and completed one subsequent timestep with finite, nonnegative molecule counts. Targeted tests cover allocator dimensions, rejection/exhaustion behavior, identifier decoding, chromosome methylation coordinates, ribosome assembly broadcasting, and translation sampling weights. This is implementation smoke coverage, not initialization-ensemble or full-cycle validation.

A divide-by-zero warning in ribosome release remains visible: the source also divides the removed mRNA count by the remaining count, which can be zero. The comparison then selects bound ribosomes for release. No warning suppression or biological parameter adjustment was introduced.

The newly exercised path exposed and repaired missing base allocation, scalar/array and identifier conversion errors, ribosome-assembly broadcasting, chromosome sparse-coordinate indexing, a transcription correction-shape mismatch, and an extra translation sampling-weight decrement absent from MATLAB. These changes concern initialization and its supporting interfaces. The active snapshot-based ensemble was not modified or restarted.

## What this does not implement

The original `FitConstants` workflow reconciles experimental inputs and fits expression, composition, biomass, and energetic constraints, including nonlinear constrained optimization. That is separate from sampling a cell from fitted constants. Raw knowledge-base construction and the complete fitting workflow remain incomplete. The included fitted template is still required; no MATLAB license is needed to use it in Python.

The existing comparison charts start from the same MATLAB t=60 biological state. They do not provide validation evidence for independently initialized cells. Source hashes separate the initializer from the frozen source used for that cohort.
