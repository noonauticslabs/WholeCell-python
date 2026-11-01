# MATLAB compatibility contract

The original WholeCell implementation is the biological specification. MathWorks documentation supplies semantics for replaced library operations; it is not a replacement for reading the model source. The compatibility layer implements the model-facing subset. It does not claim that every MATLAB expression is interchangeable with Python.

| Boundary | Python implementation | Contract and evidence |
|---|---|---|
| Linear indexing and shape | `process_base.py`, `molecule_count_state.py`, state/process modules | Explicit one-based to zero-based conversion; Fortran-order linear indexing; scalar extraction and species/compartment/time dimensions. Regression tests cover column vectors and state aliasing. |
| Sparse and circular coordinates | `sparsemat.py`, `circular_sparsemat.py` | Source utility semantics for coordinate queries, implicit zeros, reductions, mutation, and wrapping. General MATLAB range indexing remains incomplete. |
| Sequence operations | `chromosome_sequence.py`, `gene.py` | Original sequence orientation, complement and coordinate conventions. General character-array shape behavior is not fully reproduced. |
| Random sampling | `randstream.py` | Distribution families, parameters, dimensions, and one-based returned indices where required. NumPy's generator is independent; MATLAB RNG state is not replayed. Distribution tests do not certify every rare branch. |
| Linear programming | `solvers.py`, `vendor/diagnostic/glpk444_text.c` | Native GLPK 4.44 standalone executable, full-precision text exchange, source matrix insertion order, source-compatible simplex settings. Cached MATLAB LP fixtures compare objectives and vectors. |
| State serialization | `snapshot.py`, `checkpoint.py` | JSON export restores stored fields and reference identity. Python checkpoints additionally retain native RNG state and source provenance. MATLAB function handles are not executed as MATLAB code. |
| Scheduler and resource transfer | `simulation.py`, `process_base.py` | Original process responsibilities and state gather/scatter behavior. Deterministic ordering matters; no simulator output is supplied as the next state during a Python rollout. |

## Documentation used for semantic checks

- [MATLAB array indexing](https://www.mathworks.com/help/matlab/math/array-indexing.html)
- [Reshape](https://www.mathworks.com/help/matlab/ref/double.reshape.html)
- [Modular arithmetic](https://www.mathworks.com/help/matlab/ref/mod.html)
- [Sparse arrays](https://www.mathworks.com/help/matlab/ref/sparse.html)
- [Find](https://www.mathworks.com/help/matlab/ref/find.html)
- [Multinomial sampling](https://www.mathworks.com/help/stats/mnrnd.html)
- [Sequence translation](https://www.mathworks.com/help/bioinfo/ref/nt2aa.html)

Read these alongside the source-linked audit rows in `validation/audit/`. A reference to documentation records the intended semantics; it does not establish that all input regimes are implemented or tested.

## What a repair is allowed to change

Repairs restore original behavior when a translated index, shape, reduction, allocation, ownership rule, or process branch is wrong. They may change the resulting trajectory. Such a correction is different from modifying a biological parameter to match a graph.

An optimization has a stricter contract: on the same Python seed and state, it should preserve checked state fields and random-stream state. Algorithm repairs and performance-only changes must not share an unlabeled benchmark result.
