# GLPK 4.44 comparison backend

`glpk444_text.c` is a standalone driver linked to the locally built GLPK 4.44
static library. It does not link to or load the `swiglpk`/GLPK 5 ABI. The
Python API backend `backend="glpk444"` invokes the prebuilt
`glpk444_text` executable; it does not compile it at runtime.

Source archive: <https://ftp.gnu.org/gnu/glpk/glpk-4.44.tar.gz>

SHA-256: `289a4366041ee7360b5523dbfdcb083487743357b96529e4af2477a65a3c7e6d`

Build from the repository root with `./scripts/build_glpk.sh`. The script verifies the source archive and records the local binary hash.

The fixture writer emits full precision (`%.17g`) objective/bound/RHS values
and sparse matrix entries in the source MEX's dense row-major insertion order.
The driver solves with primal simplex, presolve enabled, and `tol_bnd=1e-6`.
With presolve enabled and simplex selected, the original MEX skips its explicit
scaling call; this driver follows the same path. `python scripts/verify_installation.py` checks the driver against the included linear-programming fixtures.
