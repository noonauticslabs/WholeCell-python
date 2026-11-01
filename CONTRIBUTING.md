# Contributing

Start with the original MATLAB method and the existing compatibility contract. Keep changes scoped to a demonstrated defect or a measured performance issue.

For a numerical or biological change, include the source-method reference, the triggering state/input, before-and-after behavior, and a regression that exercises the relevant branch. Explain any change in random draw count or ordering. Do not adjust model constants to improve agreement with one trajectory.

For a performance-only change, compare immutable copies under the same seed, state, interpreter and native solver. Record state/RNG agreement as well as timing. Avoid silently changing array ownership, dimensions, sparse behavior, or reaction order.

Run `python -m pytest -q` after preparing fixtures and building GLPK. Preserve warnings and failures. Never revise a saved validation result in place without updating its provenance and cohort label. Do not commit credentials, native checkpoints, local environments or machine-specific deployment scripts.

Use readable Python with explicit array contracts. New documentation should describe observable behavior and evidence, without generated status chatter or unsupported completeness claims.

## Machine formatting

```sh
python -m pip install -e '.[style]'
python -m black wholecell_py tests scripts
python -m flake8 wholecell_py tests scripts
```

Black uses 79 columns. Flake8 checks style; it does not format code.
