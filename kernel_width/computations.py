"""Independent numerical checks for cap_kernel_note.tex.

The quadratures are floating-point checks. The note proves its theorems
analytically; its final inradius-energy question remains open.
"""
from pathlib import Path
import json
import argparse
import numpy as np
from scipy.integrate import quad
from scipy.special import beta as beta_function
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parent
BASE = {'straight': 1/3, 'curved': 1-1/np.sqrt(3)}


def opening(kind, r):
    if kind == 'straight':
        return np.arccos(np.clip(r/(1-r), -1, 1))
    return 2*np.arccos(np.clip(1/(2*(1-r)), -1, 1))


def area(kind, r):
    if r == .5:
        return np.pi/4
    a = opening(kind, r)
    if kind == 'straight':
        return r*r*(np.pi+3*(np.tan(a)-a))
    return np.pi*r*r+1.5*(1-2*r*r)*a-1.5*np.sqrt(4*(1-r)**2-1)


def profile(kind, r, phi, axes=None):
    if axes is None:
        axes = np.arange(3)*2*np.pi/3
    phi = np.asarray(phi)
    diffs = (phi[..., None]-np.asarray(axes)+np.pi) % (2*np.pi)-np.pi
    rel = np.take_along_axis(diffs, np.argmin(abs(diffs), axis=-1)[..., None], axis=-1)[..., 0]
    a = opening(kind, r)
    z = np.maximum(a-abs(rel), 0.)
    if kind == 'straight':
        p = r/np.cos(z)
        pz = p*np.tan(z)
    else:
        d = 1-r
        root = np.sqrt(1-d*d*np.sin(z)**2)
        p = -d*np.cos(z)+root
        pz = d*np.sin(z)-d*d*np.sin(z)*np.cos(z)/root
    derivative = -np.sign(rel)*pz
    return p, derivative


def clip_area(kind, r, radius):
    if radius <= r:
        return np.pi*radius**2
    if radius >= 1-r:
        return area(kind, r)
    a = opening(kind, r)
    val = quad(lambda phi: min(radius, float(profile(kind, r, phi)[0]))**2,
               0, a, epsabs=2e-12, epsrel=2e-12, points=[a/2])[0]
    return 3*val+(np.pi-3*a)*r*r


def boundary_nodes(kind, r, order, axes=None):
    if axes is None:
        axes = np.arange(3)*2*np.pi/3
    a = opening(kind, r)
    breaks = np.unique(np.r_[0., 2*np.pi, np.asarray(axes) % (2*np.pi),
                               (np.asarray(axes)-a) % (2*np.pi),
                               (np.asarray(axes)+a) % (2*np.pi)])
    q, w = leggauss(order)
    phis, weights = [], []
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        if hi-lo < 1e-12:
            continue
        phis.append((lo+hi)/2+(hi-lo)*q/2)
        weights.append((hi-lo)*w/2)
    phi, weights = np.concatenate(phis), np.concatenate(weights)
    p, pp = profile(kind, r, phi, axes)
    e = np.c_[np.cos(phi), np.sin(phi)]
    je = np.c_[-np.sin(phi), np.cos(phi)]
    x = p[:, None]*e
    normal_ds = (p[:, None]*e-pp[:, None]*je)*weights[:, None]
    return x, normal_ds


def energies(kind, r, order, exponents=(.5, 1., 1.5), axes=None):
    x, n = boundary_nodes(kind, r, order, axes)
    distance = np.linalg.norm(x[:, None, :]-x[None, :, :], axis=-1)
    dot = n @ n.T
    integrals = {s: float(np.sum(distance**(2-s)*dot)) for s in exponents}
    # Resolve each diagonal panel by a Duffy map on its two triangles.
    # This integrates the zero-distance endpoint instead of dropping a
    # tensor-quadrature diagonal, which has slow convergence for s near 2.
    if axes is None:
        axes = np.arange(3)*2*np.pi/3
    a = opening(kind, r)
    breaks = np.unique(np.r_[0., 2*np.pi, np.asarray(axes) % (2*np.pi),
                               (np.asarray(axes)-a) % (2*np.pi),
                               (np.asarray(axes)+a) % (2*np.pi)])
    u, w = leggauss(order)
    u, w = (u+1)/2, w/2
    index = 0
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        if hi-lo < 1e-12:
            continue
        phi = lo+(hi-lo)*np.broadcast_to(u[:, None], (order, order))
        psi = lo+(hi-lo)*u[:, None]*u[None, :]
        def evaluate(angle):
            p, pp = profile(kind, r, angle, axes)
            e = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
            je = np.stack([-np.sin(angle), np.cos(angle)], axis=-1)
            return p[..., None]*e, p[..., None]*e-pp[..., None]*je
        xp, np_ = evaluate(phi)
        xq, nq = evaluate(psi)
        dd = np.linalg.norm(xp-xq, axis=-1)
        nd = np.sum(np_*nq, axis=-1)
        weight = 2*(hi-lo)**2*u[:, None]*w[:, None]*w[None, :]
        sl = slice(index, index+order)
        for s in exponents:
            integrals[s] -= np.sum(distance[sl, sl]**(2-s)*dot[sl, sl])
            integrals[s] += np.sum(dd**(2-s)*nd*weight)
        index += order
    return {str(s): float(-integrals[s]/(2-s)**2) for s in exponents}


def checks():
    result = {'status': 'floating-point checks, not certificates', 'clipping': {},
              'energy_refinements': {}, 'cutoff_balancing_counterexample': {}}
    for kind, base in BASE.items():
        minimum_derivative = float('inf')
        for r in np.linspace(base+.0005, .499, 19):
            for radius in np.linspace(base+.002, .68, 23):
                if radius < r+1e-6:
                    continue
                h = 1e-6
                deriv = (clip_area(kind, r+h, radius)-clip_area(kind, r-h, radius))/(2*h)
                minimum_derivative = min(minimum_derivative, deriv)
                assert deriv > -2e-7
                assert clip_area(kind, r, radius) >= clip_area(kind, base, radius)-2e-10
        result['clipping'][kind] = {'minimum_sampled_nonzero_derivative': minimum_derivative}
        cases = {}
        for r in [base, (base+.5)/2, .499]:
            cases[str(r)] = {str(n): energies(kind, r, n) for n in [32, 64, 128]}
        result['energy_refinements'][kind] = cases
        a = opening(kind, .499)
        r, d = .499, .501
        upper = np.sqrt((d-r)**2+4*d*d*np.sin(np.pi/3+a)**2)
        lower = 2*r*np.cos(1.5*a)
        cap_area = (area(kind, r)-np.pi*r*r)/3
        assert a < np.pi/24 and upper < .95 < lower
        result['cutoff_balancing_counterexample'][kind] = dict(
            r=r, cutoff=.95, opening=a, upper_equal_and_short_pair_distances=upper,
            lower_long_pair_distance=lower, cap_area=cap_area,
            exact_energy_difference_unequal_minus_equal=-2*cap_area**2)
    # Independent disk normalization for the boundary representation.
    disk = energies('straight', .5, 160)
    exact = {str(s): float(2*np.pi*.5*1.**(3-s)*beta_function(.5, (5-s)/2)
                          /((2-s)*(3-s))) for s in [.5, 1., 1.5]}
    result['disk_energy_comparison'] = {'boundary': disk, 'exact': exact}
    assert max(abs(disk[k]-exact[k]) for k in exact) < 2e-6
    # A non-equilateral triple: convex decreasing Riesz kernels give the proved sign.
    kind, r = 'curved', .45
    a = opening(kind, r)
    surplus = 2*np.pi-6*a
    gaps = 2*a+surplus*np.array([.25, .35, .40])
    axes = np.r_[0., np.cumsum(gaps[:2])]
    assert max(gaps) <= np.pi-a
    equal, unequal = energies(kind, r, 128), energies(kind, r, 128, axes=axes)
    result['angular_riesz_check'] = {k: unequal[k]-equal[k] for k in equal}
    assert min(result['angular_riesz_check'].values()) > 0
    (ROOT/'checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return result


def draw():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11, 'axes.titlesize':11,
                         'axes.labelsize':11, 'xtick.labelsize':10,
                         'ytick.labelsize':10})
    fig, ax = plt.subplots(2, 2, figsize=(8, 7.5), constrained_layout=True)
    output = {}
    for col, (kind, base) in enumerate(BASE.items()):
        radii = np.linspace(0, .68, 180)
        for r,ls in zip([base, (base+.5)/2, .499, .5],['-','--','-.',':']):
            ax[0, col].plot(radii, [clip_area(kind, r, s) for s in radii],
                            label=rf'$r={r:.4f}$',ls=ls)
        ax[0, col].set(title=f'{kind.capitalize()} caps: centered clipping',
                       xlabel=r'Clipping radius $R$', ylabel=r'$|S_r\cap B_R|$')
        rs = np.linspace(base, .5, 18)
        es = [energies(kind, r, 64) for r in rs]
        output[kind] = {'r': rs.tolist(), 'energies': es}
        for exponent,ls in zip([.5, 1., 1.5],['-','--','-.']):
            vals = np.array([v[str(exponent)] for v in es])
            ax[1, col].plot(rs, vals/vals[0], label=rf'$s={exponent}$',ls=ls)
        ax[1, col].set(title='Equilateral cap energy (numerical)',
                       xlabel=r'Inradius parameter $r$', ylabel=r'$\mathcal{E}_s(E_r)/\mathcal{E}_s(Q)$')
    for a in ax.flat:
        a.legend(fontsize=9.5)
        a.grid(alpha=.2)
    fig.savefig(ROOT/'cap_kernel_profiles.pdf')
    fig.savefig(ROOT/'cap_kernel_profiles.png', dpi=160)
    plt.close(fig)
    (ROOT/'energy_scan.json').write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--draw', action='store_true')
    args = parser.parse_args()
    checks()
    if args.draw:
        draw()
