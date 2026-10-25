PYTHON ?= python3
export OPENBLAS_NUM_THREADS = 1
export OMP_NUM_THREADS = 1
export MPLBACKEND = Agg
export MPLCONFIGDIR = $(CURDIR)/.matplotlib
HEAT_FLAGS ?=

.PHONY: help check smoke figures kernels torsion eigenvalues heat reproduce

help:
	@echo "make check       Geometry, coordinate certificates, and recorded-result checks (no FreeFEM)"
	@echo "make smoke       Fresh FreeFEM solves checked against exact disk/triangle values"
	@echo "make figures     Regenerate all geometric illustrations"
	@echo "make kernels     Riesz/clipping and coordinate experiments, certificates, and plots"
	@echo "make torsion     Hexagon torsion table at N=80 and N=160"
	@echo "make eigenvalues First eigenvalue table and boundary derivative checks"
	@echo "make heat        Seven heat/spectrum studies and summary (uses supplied caches)"
	@echo "make heat HEAT_FLAGS=--force  Recompute all heat/spectrum cases"
	@echo "make reproduce HEAT_FLAGS=--force  Recompute every paper study and figure"

check:
	$(PYTHON) check_geometry.py
	$(PYTHON) check_width_comparisons.py
	$(PYTHON) kernel_width/certify_cutoff.py
	$(PYTHON) kernel_width/coordinate_experiments.py
	$(PYTHON) verify_results.py

smoke:
	$(PYTHON) verify_results.py --solve

figures:
	$(PYTHON) draw_caps.py
	$(PYTHON) draw_comparison.py
	$(PYTHON) draw_proof_overview.py
	$(PYTHON) draw_reflection_arguments.py
	$(PYTHON) check_width_comparisons.py --draw

kernels:
	$(PYTHON) kernel_width/computations.py --draw
	$(PYTHON) kernel_width/certify_cutoff.py
	$(PYTHON) kernel_width/coordinate_experiments.py

torsion:
	$(PYTHON) explore_caps.py --N 80 --hexagon
	$(PYTHON) explore_caps.py --N 160 --hexagon

eigenvalues:
	$(PYTHON) check_eigenvalues.py

heat:
	$(PYTHON) heat_spectrum/investigate.py --study scan $(HEAT_FLAGS)
	$(PYTHON) heat_spectrum/investigate.py --study refine $(HEAT_FLAGS)
	$(PYTHON) heat_spectrum/investigate.py --study survey $(HEAT_FLAGS)
	$(PYTHON) heat_spectrum/investigate.py --study heat $(HEAT_FLAGS)
	$(PYTHON) heat_spectrum/investigate.py --study heat-refine $(HEAT_FLAGS)
	$(PYTHON) heat_spectrum/investigate.py --study benchmark $(HEAT_FLAGS)
	$(PYTHON) heat_spectrum/investigate.py --study hex-heat $(HEAT_FLAGS)
	$(PYTHON) heat_spectrum/analyze.py

reproduce:
	$(MAKE) check
	$(MAKE) figures
	$(MAKE) kernels
	$(MAKE) torsion
	$(MAKE) eigenvalues
	$(MAKE) heat HEAT_FLAGS="$(HEAT_FLAGS)"
	$(PYTHON) verify_results.py
