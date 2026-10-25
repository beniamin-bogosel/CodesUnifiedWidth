"""Small reproducible eigenvalue probe; approximate values, not certificates.

Run with python check_eigenvalues.py. Requires FreeFem++ and NumPy.
P2 finite elements use inscribed polygonal meshes of analytic arc inputs.
"""
from pathlib import Path
import argparse
import json
import subprocess
import numpy as np
from explore_caps import R0, hexagon

ROOT = Path(__file__).resolve().parent


def evaluate(r, subdivisions):
    out = ROOT / 'outputs' / 'eigenvalue_checks'
    out.mkdir(exist_ok=True, parents=True)
    name = f'hexagon_{r:.9f}_{subdivisions}'
    data = out / f'{name}.dat'
    flux = out / f'{name}_flux.dat'
    arcs = hexagon(r)
    with data.open('w') as stream:
        stream.write(f'{len(arcs)}\n')
        np.savetxt(stream, arcs, fmt='%.17g')
    run = subprocess.run(
        ['FreeFem++', '-nw', '-v', '0', str(ROOT / 'eigenvalue_arcs.edp'),
         '-input', str(data), '-flux', str(flux), '-N', str(subdivisions)],
        capture_output=True, text=True, check=False,
    )
    (out / f'{name}.log').write_text(run.stdout + run.stderr)
    if run.returncode:
        raise RuntimeError(run.stdout + run.stderr)
    line = next(s for s in run.stdout.splitlines() if s.startswith('RESULT '))
    eigenvalue, derivative, area, vertices, defect = map(float, line.split()[1:])
    rows = np.loadtxt(flux)
    first_arc = rows[rows[:, 0] == 0, 2]
    half = len(first_arc) // 2
    paired = first_arc[-half:][::-1] - first_arc[:half]
    result = dict(r=r, N=subdivisions, eigenvalue=eigenvalue,
                  hadamard_derivative=derivative, area=area,
                  vertices=int(vertices), rayleigh_defect=defect,
                  min_sampled_paired_flux_difference=float(paired.min()),
                  max_sampled_paired_flux_difference=float(paired.max()))
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolutions', nargs=2, type=int, default=[50, 100])
    args = parser.parse_args()
    assert min(args.resolutions) >= 50, 'Flux sampling offset requires N >= 50.'
    rows = [evaluate(r, n) for n in args.resolutions for r in [R0, .45, .5]]
    step = .001
    sides = [evaluate(.45 + sign * step, max(args.resolutions)) for sign in [-1, 1]]
    finite_difference = (sides[1]['eigenvalue'] - sides[0]['eigenvalue']) / (2 * step)
    result = dict(
        status='Approximate P2 finite-element evidence; no certification.',
        rows=rows, finite_difference_rows=sides,
        finite_difference_at_045=finite_difference,
        note='At r=.5 exact paired flux differences and derivative vanish; mesh asymmetry remains.',
    )
    output = ROOT / 'outputs' / 'eigenvalue_checks.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Central finite difference at r=.45: {finite_difference:.9f}')
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
