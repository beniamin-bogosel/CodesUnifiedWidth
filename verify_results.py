"""Check the manuscript's numerical tables and optional fresh PDE benchmarks."""
from pathlib import Path
import argparse
import json
import math
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT / name).read_text())


def recorded_results():
    # Compare the supplied reference data with the precision printed in the paper.
    for n, expected in [(80, [.01831822, .01929619, .02003322]),
                        (160, [.01831986, .01929777, .02003490])]:
        rows = load(f"outputs/hexagon_results_{n}.json")
        for key, value in zip(["0.422649731", "0.450000000", "0.500000000"], expected):
            assert abs(rows[f"hexagon_{key}"]["torsion"] - value) < 5e-9
    eigen = load("outputs/eigenvalue_checks.json")
    for n, expected in [(50, [27.277425, 26.312668, 25.641488]),
                        (100, [27.274495, 26.309797, 25.638839])]:
        rows = sorted((r for r in eigen["rows"] if r["N"] == n), key=lambda r: r["r"])
        assert len(rows) == 3
        assert max(abs(r["eigenvalue"] - v) for r, v in zip(rows, expected)) < 5e-7
    assert abs(eigen["finite_difference_at_045"] + 27.19097) < 5e-6
    # Analyze the raw study data, including every survey case, and assert its signs.
    from heat_spectrum.analyze import main as analyze
    analyze()
    summary = load("heat_spectrum/summary.json")
    assert summary["survey_configurations"] == 48
    assert summary["heat_survey_sign_counts"] == {"-1": 0, "0": 28, "1": 116}
    for row in load("heat_spectrum/hex_heat_checks.json"):
        assert min(row["differences_from_Q"].values()) > 0
    print("PASS: torsion/eigenvalue tables, spectral refinements, 48-case survey, heat comparisons.")


def fresh_solves():
    # Isolate all new data from the bundled reference results and bypass caches.
    import explore_caps
    from heat_spectrum import investigate
    from scipy.special import jn_zeros

    old_caps, old_heat = explore_caps.ROOT, investigate.ROOT
    with tempfile.TemporaryDirectory(prefix="unified-width-") as directory:
        root = Path(directory)
        for name in ["torsion_arcs.edp", "spectrum.edp"]:
            source = ROOT / ("heat_spectrum/spectrum.edp" if name == "spectrum.edp" else name)
            (root / name).write_bytes(source.read_bytes())
        try:
            explore_caps.ROOT = investigate.ROOT = root
            disk_arcs = np.array([[0, 0, .5, 0, 2 * np.pi, 0, 0, 0]])
            torsion = explore_caps.evaluate(disk_arcs, 50, "disk")
            assert abs(torsion["torsion"] / (np.pi / 128) - 1) < 1e-3
            disk = investigate.solve("disk", np.array([[0, 0, 0, .5, 0, 2*np.pi]]), 50, 8, force=True)
            assert abs(disk["eigenvalues"][0] / (4*jn_zeros(0, 1)[0]**2) - 1) < 1e-3
            triangle = investigate.boundary("straight", 1/3, np.arange(3)*2*np.pi/3)
            spectrum = investigate.solve("triangle", triangle, 50, 8, force=True)
            assert abs(spectrum["eigenvalues"][0] / (4*math.pi**2) - 1) < 1e-4
            assert max(disk["max_rayleigh_defect"], spectrum["max_rayleigh_defect"]) < 1e-6
            print("PASS: fresh FreeFEM disk torsion, disk spectrum, and equilateral triangle spectrum.")
        finally:
            explore_caps.ROOT, investigate.ROOT = old_caps, old_heat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solve", action="store_true", help="Also run fresh FreeFEM benchmarks")
    args = parser.parse_args()
    recorded_results()
    if args.solve:
        fresh_solves()
