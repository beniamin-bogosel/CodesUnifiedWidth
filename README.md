# CodesUnifiedWidth

Reproducible numerical experiments accompanying **Shape optimization under
minimal width and constant width constraints**, by Beniamin Bogosel.

The repository contains the simulation sources, three FreeFEM solvers,
geometry and symbolic checks, figure generators, and numerical reference data.
The manuscript, article sources/PDFs, research notes, review reports, build
records, solver logs, and superseded experiments are excluded.

## Setup

Use Python 3.10 or newer and run all commands from this repository's root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
make check
```

The Python dependencies are declared in `pyproject.toml`: NumPy, SciPy, SymPy,
Matplotlib, and mpmath. The PDE studies additionally require `FreeFem++` on
`PATH`, with `getARGV.idp`, a sparse linear solver, and `EigenValue` support.
Install FreeFEM separately; it is not a Python dependency. `make check` can run
without FreeFEM using the supplied numerical data. `make smoke` runs fresh PDE
benchmarks for disk torsion and disk/triangle spectra.

The Makefile sets single-thread BLAS/OpenMP execution and a local Matplotlib
cache. When running scripts directly, use `OPENBLAS_NUM_THREADS=1` and, on a
headless machine, `MPLBACKEND=Agg`.

## Reproduce the paper's computations

```bash
make reproduce HEAT_FLAGS=--force
```

This runs the checks, all geometric figure generators, kernel and coordinate
experiments, the two torsion refinements, first-eigenvalue refinements, all seven
heat/spectral studies, and the numerical summary checks. FreeFEM is required.
The fine meshes and 64-mode computations take substantially longer than the
Python checks. Regenerated values can differ slightly with FreeFEM, mesher,
eigensolver, and numerical-library versions.

| Computation in the paper | Sources | Command / reference data |
| --- | --- | --- |
| Circular-hexagon torsion refinement table, at `N=80,160` | `explore_caps.py`, `torsion_arcs.edp` | `make torsion`; `outputs/hexagon_results_*.json` |
| First eigenvalue at `N=50,100`, paired boundary fluxes, Hadamard derivative and centered finite difference at `r=0.45` | `check_eigenvalues.py`, `eigenvalue_arcs.edp` | `make eigenvalues`; `outputs/eigenvalue_checks.json` |
| Exact center-velocity identities, circle intersections, endpoint geometry, and containment samples at 61 inradii | `check_geometry.py`, `cwgeom.py` | `python check_geometry.py`; `outputs/geometry_checks.json` |
| Inner parallel areas, inradius derivatives, distance moments, and Minkowski-concavity examples | `check_width_comparisons.py` | `python check_width_comparisons.py`; `outputs/width_comparisons_checks.json` |
| Concentric clipping, Riesz boundary quadrature with Duffy diagonal panels, three quadrature resolutions, exact disk energy, and unequal-gap comparison | `kernel_width/computations.py` | `make kernels`; `kernel_width/checks.json`, `energy_scan.json` |
| Outward-rounded scalar margins for the decreasing-cutoff counterexample | `kernel_width/certify_cutoff.py` | `python kernel_width/certify_cutoff.py`; `kernel_width/cutoff_certificate.json` |
| Coordinate moments, orientation dependence, regular Reuleaux chord profiles, pentagon search, critical exponent, and quadratic counterexample certificates | `kernel_width/coordinate_experiments.py` | `python kernel_width/coordinate_experiments.py`; `kernel_width/coordinate_checks.json`, `coordinate_certificate.json` |
| Angular splitting of higher eigenvalues, including the third and fifth eigenvalues; `N=55,100,160`, 24 modes | `heat_spectrum/investigate.py`, `heat_spectrum/spectrum.edp` | studies `scan`, `refine`; `scan_N55_m24.json`, `refinements.json` |
| 48 unequal configurations for the second eigenvalue, at `N=45`, 8 modes, seed `20261023` | same heat/spectrum sources | study `survey`; `lambda2_survey.json`, `data/survey_*.json` |
| Heat content with 64 modes at `N=100,160`, including the heat refinement table | same heat/spectrum sources | studies `heat`, `heat-refine`; `heat_checks.json`, `heat_refinements.json` |
| Disk Bessel-zero and exact equilateral-triangle spectral benchmarks | same heat/spectrum sources | study `benchmark`; `benchmark.json` |
| Four straight/curved hexagon heat comparisons at three times | same heat/spectrum sources | study `hex-heat`; `hex_heat_checks.json` |
| Spectral tables, 144 survey heat comparisons, truncation estimates, and the spectral/heat figure | `heat_spectrum/analyze.py` | `python heat_spectrum/analyze.py`; `summary.json`, `heat_survey.json` |

Heat/spectrum JSON filenames in the last six rows are relative to
`heat_spectrum/`. Every study can be run separately:

```bash
python heat_spectrum/investigate.py --study scan --N 55 --nev 24 --force
python heat_spectrum/investigate.py --study refine --force
python heat_spectrum/investigate.py --study survey --force
python heat_spectrum/investigate.py --study heat --force
python heat_spectrum/investigate.py --study heat-refine --force
python heat_spectrum/investigate.py --study benchmark --force
python heat_spectrum/investigate.py --study hex-heat --force
python heat_spectrum/analyze.py
```

Without `--force`, these studies reuse matching solver JSON in
`heat_spectrum/data/`. Thus `make heat` rebuilds summaries and plots from the
bundled results without requiring new PDE solves. `make heat HEAT_FLAGS=--force`
recomputes them. Run the default `N=55`, 24-mode scan before `analyze.py`, whose
published figure specifically uses that scan.

The torsion driver also exposes the original additional probes (cap families,
regular Reuleaux polygons, perturbed pentagons, and irregular caps):

```bash
python explore_caps.py --N 80
```

These additional probes are available in the source; their working outputs are
not bundled. The script always solves them afresh.

## Figures and outputs

`make figures` regenerates the cap construction, comparison panels, proof
overview, reflection arguments, and inner parallel comparison. Sources are
`draw_caps.py`, `draw_comparison.py`, `draw_proof_overview.py`,
`draw_reflection_arguments.py`, and `check_width_comparisons.py --draw`.
`make kernels` generates the clipping/Riesz and coordinate figures;
`make heat` generates the heat/spectrum figure.

Figure PDFs/PNGs and geometry diagnostics are generated locally and ignored by
Git. The repository retains only the numerical JSON needed to inspect and
recreate the reported results. Solvers regenerate analytic boundary inputs
(`.dat`), flux samples, and logs locally; these files are also ignored. The
`flux` fields in the torsion reference JSON name those generated files.

## Conventions and scope

Width is normalized to one. Straight equilateral caps have inradius
`r ∈ [1/3, 1/2]`; curved equilateral caps have
`r ∈ [1 - 1/sqrt(3), 1/2]`. Torsion uses `-Δu = 1` and `T = ∫u`.
First eigenfunctions use `∫φ² = 1`. Higher eigenvalues are sorted with
multiplicity. Dirichlet heat content is approximated by the sum of
`exp(-λ_j t) (∫φ_j)² / ∫φ_j²`.

`N` denotes approximately the number of boundary segments per unit arclength,
not the number of vertices per arc. FreeFEM uses continuous P2 elements on
inscribed polygonal approximations to analytic line/arc boundaries. The
circular hexagons are intermediate comparison domains; they need not satisfy
the original constant-width constraint.

Floating-point simulations and sampled containments are implementation checks
and numerical evidence. Heat-tail estimates concern spectral truncation and
do not bound finite-element or curved-boundary approximation errors. The
mpmath interval computations certify only the explicitly stated scalar
margins. Analytic conclusions require the arguments in the preprint.

## Provenance and validation

The simulation sources and numerical data were extracted from the preprint
workspace on 26 October 2026. `source_manifest.json` records SHA-256 hashes of
the original selected files before portability edits. `cwgeom.py` is an
unchanged copy of the geometry helper previously loaded from the author's
local constant-width geometry toolkit; no installed skill or neighboring
research project is required.

Portability changes replace absolute helper paths with a local import, create
missing figure directories, and add `--force` to the spectral driver. The
numerical formulas and FreeFEM solvers are preserved. The Makefile,
`verify_results.py`, dependency metadata, and GitHub Actions workflow provide
the reproduction and validation entry points.

`make check` verifies the symbolic and geometry checks, coordinate and scalar
certificates, rounded values in the torsion/first-eigenvalue table, spectral
refinement signs, and the 48-case survey (116 positive heat comparisons beyond
the estimated tails, 28 unresolved). GitHub Actions runs these checks without
FreeFEM. `make smoke` additionally runs fresh solvers in a temporary directory
and compares them with exact disk torsion and disk/triangle eigenvalues.

Python validation environment: Python 3.12.9, NumPy 2.4.6, SciPy 1.17.1,
SymPy 1.14.0, Matplotlib 3.11.0, and mpmath 1.3.0. Dependency ranges in
`pyproject.toml` allow other compatible versions; they are not a frozen lockfile.

Validation on 26 October 2026 passed: editable installation, `make check`,
`make smoke`, and `make figures kernels heat`. The full fine-mesh PDE studies
were not rerun during packaging; their original reference JSON is retained
unchanged, and `make smoke` independently exercised fresh FreeFEM solves.
