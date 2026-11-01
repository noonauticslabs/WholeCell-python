# Third-party notices

## Original WholeCell model and exported data

The model algorithms and model-derived fixtures originate from the Karr et al. WholeCell project. Original source revision recorded in the exports: `6cdee6b355aa0f5ff2953b1ab356eea049108e07`. Preserve `LICENSES/WholeCell-MIT.txt`. The upstream README states that the code, training data and study results were released under MIT. Model authorship is separate from authorship of this Python translation.

## GLPK

`vendor/glpk-4.44.tar.gz` is the original GNU GLPK 4.44 source archive, SHA-256 `289a4366041ee7360b5523dbfdcb083487743357b96529e4af2477a65a3c7e6d`, obtained from `https://ftp.gnu.org/gnu/glpk/glpk-4.44.tar.gz`. The archive retains its source notices. See `LICENSES/GLPK-GPL-3.0.txt` and the archive for GLPK licensing and bundled component notices.

The standalone driver in `vendor/diagnostic/glpk444_text.c` is distributed under GPL-3.0-or-later and linked locally against GLPK. No precompiled executable is included in the release. This notice does not relicense GLPK under the Python package's MIT license.

## Dependencies and documentation

NumPy, SciPy, Matplotlib, pytest and swiglpk are installed separately and retain their own licenses. `swiglpk` is used for optional backend tests; it does not supply the required GLPK 4.44 runtime driver.

MathWorks documentation is cited for behavior and is not bundled. No MATLAB runtime, proprietary toolbox code, authentication material, or licensed MATLAB executable is included.

## Figures and media

The figures and animations in `docs/figures/` were generated for this port from model-derived ensemble data; they are not copies of the paper's figures or movie. Their generation scripts and data are included where available. Model-derived data remain subject to the original WholeCell notice above. The schematic chromosome frame renderer identifies its upstream source in its script and documentation.
