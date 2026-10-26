"""Explore the rotation-optimized cap stability quotient in Lemma 6.8.

This is a sampled numerical exploration, not a bound or certificate for C_*.
Requires NumPy and SciPy. Run from the repository root, for example:
    python kernel_width/stability_constant.py --quick
    python kernel_width/stability_constant.py --grid 12 --order 48

The angular integral is split at all cap junctions, axes and detected crossings
of the two squared radial profiles. A periodic phase scan followed by bounded
local minimizations approximates the infimum over reference-body rotations.
"""
from pathlib import Path
from functools import lru_cache
import argparse
import json
import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq, minimize_scalar

TAU = 2 * math.pi
PERIOD = TAU / 3
BASE = {"straight": 1 / 3, "curved": 1 - 1 / math.sqrt(3)}


def opening(kind, r):
    x = r / (1-r) if kind == "straight" else 1 / (2*(1-r))
    return (1 if kind == "straight" else 2) * math.acos(min(1., max(-1., x)))


def area(kind, r):
    if r == .5:
        return math.pi / 4
    a = opening(kind, r)
    if kind == "straight":
        return r*r * (math.pi + 3*(math.tan(a)-a))
    return math.pi*r*r + 1.5*(1-2*r*r)*a - 1.5*math.sqrt(4*(1-r)**2-1)


def reference_area(kind):
    return 1/math.sqrt(3) if kind == "straight" else (math.pi-math.sqrt(3))/2


class Profile:
    """Squared radial profile of the union of the disk and its three caps."""
    def __init__(self, kind, r, axes):
        self.kind, self.r = kind, r
        self.axes = np.asarray(axes, dtype=float)
        self.a = opening(kind, r)

    def __call__(self, phi):
        phi = np.asarray(phi)
        delta = np.min(abs((phi[..., None]-self.axes+math.pi) % TAU-math.pi), axis=-1)
        z = np.maximum(0., self.a-delta)
        if self.kind == "straight":
            p = self.r / np.cos(z)
        else:
            d = 1-self.r
            p = -d*np.cos(z) + np.sqrt(np.maximum(0., 1-d*d*np.sin(z)**2))
        return p*p

    def junctions(self, phase=0.):
        return np.r_[self.axes, self.axes-self.a, self.axes+self.a] + phase


@lru_cache(maxsize=None)
def gauss(order):
    return leggauss(order)


def partition(points):
    points = np.sort(np.r_[0., np.asarray(points) % TAU, TAU])
    return points[np.r_[True, np.diff(points) > 1e-13]]


def integrate_intervals(function, cuts, order):
    nodes, weights = gauss(order)
    mid, half = (cuts[1:]+cuts[:-1])/2, np.diff(cuts)/2
    values = function(mid[:, None]+half[:, None]*nodes)
    return float(np.sum(half*(values @ weights)))


def symmetric_difference(profile, reference, theta, order=32):
    cuts = partition(np.r_[profile.junctions(), reference.junctions(theta)])
    difference = lambda phi: profile(phi)-reference(np.asarray(phi)-theta)
    crossings = []
    # Individual profiles are smooth between these cuts. Resolve the absolute
    # value's additional kinks by bracketing sign changes on each piece.
    for left, right in zip(cuts[:-1], cuts[1:]):
        nodes = np.linspace(left, right, 9)
        values = difference(nodes)
        for lo, hi, f, g in zip(nodes[:-1], nodes[1:], values[:-1], values[1:]):
            if f == 0 and left < lo < right:
                crossings.append(lo)
            if f*g < 0:
                crossings.append(brentq(difference, lo, hi, xtol=1e-13, rtol=1e-13))
    cuts = partition(np.r_[cuts, crossings])
    return .5*integrate_intervals(lambda x: abs(difference(x)), cuts, order)


def minimize_rotation(profile, reference, order=32, phases=24):
    objective = lambda theta: symmetric_difference(profile, reference, theta, order)
    grid = np.linspace(-PERIOD/2, PERIOD/2, phases, endpoint=False)
    values = np.array([objective(theta) for theta in grid])
    best = int(np.argmin(values))
    answer = (float(values[best]), float(grid[best]))
    step = PERIOD/phases
    for i in range(phases):
        if values[i] <= values[(i-1) % phases] and values[i] <= values[(i+1) % phases]:
            result = minimize_scalar(objective, bounds=(grid[i]-step, grid[i]+step),
                                     method="bounded", options={"xatol": 1e-13})
            if result.fun < answer[0]:
                answer = (float(result.fun), float(result.x))
    return answer


def gap_parameters(resolution, quick=False):
    if quick:
        return [np.array(x) for x in [(0., 0., 1.), (0., .5, .5), (1/3, 1/3, 1/3)]]
    # Permuting three gaps amounts to a rotation or a reflection of the body.
    # The reference body has reflection symmetry, so one ordered representative
    # of each lattice triple suffices for this numerical quotient.
    return [np.array([i, j, resolution-i-j], dtype=float)/resolution
            for i in range(resolution+1) for j in range(i, resolution-i+1)
            if resolution-i-j >= j]


def check_implementation(order):
    for kind, b in BASE.items():
        for r in [b, (b+.5)/2, .5]:
            p = Profile(kind, r, np.arange(3)*PERIOD)
            measured = .5*integrate_intervals(p, partition(p.junctions()), order)
            assert abs(measured-area(kind, r)) < 2e-11
        q = Profile(kind, b, np.arange(3)*PERIOD)
        assert symmetric_difference(q, q, 0., order) < 1e-12
        assert symmetric_difference(q, q, PERIOD, order) < 1e-12


def run_family(kind, args):
    b = BASE[kind]
    reference = Profile(kind, b, np.arange(3)*PERIOD)
    radii = sorted(set([b+h for h in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2] if b+h < .5]
                       + [b+t*(.5-b) for t in [.25, .5, .75, 1.]]))
    if args.quick:
        radii = [b+1e-4, b+1e-3, (b+.5)/2, .5]
    rows = []
    for r in radii:
        a = opening(kind, r)
        denominator = area(kind, r)-reference_area(kind)
        assert denominator > 0
        for v in gap_parameters(args.grid, args.quick):
            # Straight gaps: delta_i >= 2a. Curved gaps additionally satisfy
            # delta_i <= pi-a; writing w_i=(1-v_i)/2 enforces this exactly.
            weights = v if kind == "straight" else (1-v)/2
            gaps = 2*a+(TAU-6*a)*weights
            assert min(gaps) >= 2*a-1e-12 and abs(sum(gaps)-TAU) < 1e-12
            if kind == "curved":
                assert max(gaps) <= math.pi-a+1e-12
            axes = np.r_[0., np.cumsum(gaps[:2])]
            p = Profile(kind, r, axes)
            numerator, theta = minimize_rotation(p, reference, args.order, args.phases)
            aligned = [symmetric_difference(p, reference, angle, args.order) for angle in axes]
            fixed = aligned[0]
            assert numerator >= denominator-2e-10
            assert numerator <= fixed+2e-10
            rows.append(dict(r=r, r_minus_b=r-b, gaps=gaps.tolist(), axes=axes.tolist(),
                             theta=theta, symmetric_difference=numerator,
                             area_excess=denominator, ratio=numerator/denominator,
                             ratio_with_first_axes_aligned=fixed/denominator,
                             axis_alignment_ratios=[value/denominator for value in aligned],
                             largest_axis_alignment_ratio=max(aligned)/denominator))
            if r == .5:
                break  # The disk is independent of the cap positions.
        print(f"{kind}: r-b={r-b:.7g}, maximum sampled ratio at this radius="
              f"{max(row['ratio'] for row in rows if row['r']==r):.9g}", flush=True)
    best = max(rows, key=lambda row: row['ratio'])
    p = Profile(kind, best['r'], best['axes'])
    numerator, theta = minimize_rotation(p, reference, 2*args.order, 2*args.phases)
    refinement = dict(order=2*args.order, phases=2*args.phases, theta=theta,
                      ratio=numerator/best['area_excess'],
                      absolute_ratio_change=abs(numerator/best['area_excess']-best['ratio']))
    aligned_best = max(rows, key=lambda row: row['largest_axis_alignment_ratio'])
    return dict(sampled_maximum=best, refinement=refinement,
                sampled_axis_aligned_maximum=aligned_best['largest_axis_alignment_ratio'],
                cases=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["both", "straight", "curved"], default="both")
    parser.add_argument("--grid", type=int, default=6, help="Resolution of the admissible gap lattice")
    parser.add_argument("--order", type=int, default=32, help="Gauss-Legendre nodes on each smooth interval")
    parser.add_argument("--phases", type=int, default=24, help="Initial phase grid over one symmetry period")
    parser.add_argument("--quick", action="store_true", help="Use fewer radii and gap triples")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("stability_scan.json"))
    args = parser.parse_args()
    if args.grid < 1 or args.order < 8 or args.phases < 6:
        parser.error("Use grid >= 1, order >= 8 and phases >= 6")
    check_implementation(args.order)
    families = ["straight", "curved"] if args.family == "both" else [args.family]
    output = dict(status="Sampled numerical exploration; no certified bound on C_*.",
                  note="The inner rotation minimization and outer supremum are numerical. "
                       "Increase gap resolution, phase resolution and quadrature order to investigate convergence. "
                       "The r=b endpoint is excluded because the quotient is 0/0 there.",
                  parameters=dict(grid=args.grid, order=args.order, phases=args.phases, quick=args.quick),
                  families={kind: run_family(kind, args) for kind in families})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2)+"\n")
    for kind, result in output['families'].items():
        print(kind, "sampled maximum:", result['sampled_maximum']['ratio'],
              "refined:", result['refinement']['ratio'],
              "axis-aligned:", result['sampled_axis_aligned_maximum'])
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
