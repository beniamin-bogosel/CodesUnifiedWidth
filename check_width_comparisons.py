"""Checks for the unified paper's new erosion and concavity formulas.

Analytic identities are proved in the manuscript. Quadrature, finite
differences, and boundary samples here independently check their implementation.
Run with --draw to regenerate the vector/bitmap erosion figure.
"""
from pathlib import Path
import argparse
import json

import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parent
(ROOT / "figures").mkdir(exist_ok=True)
R0 = 1 - 1 / np.sqrt(3)
import cwgeom as cw


def angle(kind, r):
    if kind == 'straight':
        return np.arccos(np.clip(r / (1-r), -1, 1))
    return np.arccos(np.clip(1 / (2*(1-r)**2) - 1, -1, 1))


def inner_area(kind, r, t):
    if t >= r:
        return 0.
    if r == .5:
        return np.pi*(r-t)**2
    a = angle(kind, r)
    rho = r-t
    if kind == 'straight':
        return rho**2*(np.pi+3*(np.tan(a)-a))
    d, radius = 1-r, 1-t
    v = a-np.arcsin(np.clip(d*np.sin(a)/radius, -1, 1))
    return np.pi*rho**2+3*(radius**2*v-d*radius*np.sin(v)-rho**2*a)


def area_derivative(kind, r, t):
    a, rho, d = angle(kind, r), r-t, 1-r
    if kind == 'straight':
        c = np.pi+3*(np.tan(a)-a)
        cp = -3*np.tan(a)**2/(d*d*np.sin(a))
        return rho*(2*c+rho*cp)
    radius = 1-t
    v = a-np.arcsin(d*np.sin(a)/radius)
    return rho*(2*np.pi-6*a)+6*radius*(np.sin(v)-2/np.tan(a/2)*(1-np.cos(v)))


def radial(kind, r, t, phi, axes=None):
    if axes is None:
        axes = np.arange(3)*2*np.pi/3
    phi = np.asarray(phi)
    delta = np.min(np.abs((phi[..., None]-np.asarray(axes)+np.pi) % (2*np.pi)-np.pi), axis=-1)
    a = angle(kind, r)
    z = np.maximum(a-delta, 0.)
    if kind == 'straight':
        return (r-t)/np.cos(z)
    d, radius = 1-r, 1-t
    return -d*np.cos(z)+np.sqrt(np.maximum(0., radius**2-d*d*np.sin(z)**2))


def checks():
    derivative_error, quadrature_error, reuleaux_error = 0., 0., 0.
    for kind, lower in [('straight', 1/3), ('curved', R0)]:
        for r in np.linspace(lower+.0001, .4999, 31):
            a = angle(kind, r)
            for t in np.linspace(0, lower*.97, 11):
                area = inner_area(kind, r, t)
                rho = r-t
                sector = quad(lambda phi: float(radial(kind, r, t, phi))**2,
                              0, a, epsabs=2e-13)[0]
                independent = 3*sector+(np.pi-3*a)*rho*rho
                quadrature_error = max(quadrature_error, abs(area-independent))
                # Resolve the square-root dependence of the cap opening near
                # the disk endpoint; a fixed step there has a larger O(h^2) error.
                step = min(2e-6, (.5-r)/1000)
                numerical = (inner_area(kind, r+step, t)-inner_area(kind, r-step, t))/(2*step)
                derivative_error = max(derivative_error, abs(numerical-area_derivative(kind, r, t)))
                assert area_derivative(kind, r, t) > 0
                assert area > inner_area(kind, lower, t)
        # Unequal disjoint cap sectors: the full radial integral must agree.
        r = .40 if kind == 'straight' else .45
        a = angle(kind, r)
        extra = 2*np.pi-6*a
        gaps = 2*a+extra*np.array([.25, .35, .40])
        axes = np.r_[0, np.cumsum(gaps[:2])]
        breaks = sorted(set([0., 2*np.pi]+list((axes-a) % (2*np.pi))+
                            list((axes+a) % (2*np.pi))+list(axes)))
        for t in [0., .13, .29]:
            area = sum(quad(lambda phi: float(radial(kind, r, t, phi, axes))**2/2,
                            left, right, epsabs=2e-12)[0]
                       for left, right in zip(breaks[:-1], breaks[1:]))
            assert abs(area-inner_area(kind, r, t)) < 1e-10
    # Independently implemented exact disk intersections, not radial quadrature.
    centers = cw.regular_reuleaux(3)
    for t in np.linspace(0, R0*.97, 25):
        arcs = cw.disk_arcs(centers, radius=1-t)
        independent = cw.arc_area(arcs)
        reuleaux_error = max(reuleaux_error, abs(inner_area('curved', R0, t)-independent))
    assert reuleaux_error < 1e-10
    assert quadrature_error < 1e-10
    assert derivative_error < 2e-6
    for p in [.25, 1, 2, 3.5, 6]:
        val = p*quad(lambda t: t**(p-1)*inner_area('straight', 1/3, t), 0, 1/3)[0]
        expected = 2/(np.sqrt(3)*3**p*(p+1)*(p+2))
        assert abs(val-expected) < 1e-9
    eps = 1/60
    _, convexity = cw.support_curve(1., [[5, eps, 0.]])
    n = 32760  # Divisible by 3 and 5, resolves both symmetry classes.
    theta = np.arange(n)*2*np.pi/n
    h = .5+eps*np.cos(5*theta)
    f5 = (2*np.pi/n)*(np.sum(h)-abs(np.sum(h*np.exp(5j*theta))))
    assert abs(f5-np.pi*(1-eps)) < 1e-12
    reuleaux_arcs = cw.disk_arcs(centers)
    hr = np.array([cw.arc_support(reuleaux_arcs, th) for th in theta])
    assert abs(np.mean(hr*np.exp(5j*theta))) < 1e-12
    perimeter_gap = np.pi-(2*np.pi*.45+3*.1*angle('curved', .45))
    assert perimeter_gap > 0
    result = dict(derivative_max_abs_error=derivative_error,
                  radial_area_max_abs_error=quadrature_error,
                  independent_reuleaux_area_max_abs_error=reuleaux_error,
                  concave_counterexample_epsilon=eps,
                  concave_counterexample_value=f5,
                  reuleaux_counterexample_value=float(np.pi),
                  convexity_diagnostics=convexity,
                  curved_cap_perimeter_deficit_r045=float(perimeter_gap),
                  note='Floating-point implementation checks; analytic proofs are in the manuscript.')
    (ROOT/'outputs').mkdir(exist_ok=True)
    (ROOT/'outputs/width_comparisons_checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


def draw():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox
    colors = ['#235d88', '#aa6515', '#6d4680', '#278372']
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.0), constrained_layout=True)
    theta = np.linspace(0, 2*np.pi, 3601)
    for col, (kind, r, lower) in enumerate([('straight', .40, 1/3), ('curved', .45, R0)]):
        ax = axes[0, col]
        for t, color in zip([0., .12, .26], colors):
            p = radial(kind, r, t, theta)[:, None]*cw.unit(theta)
            ax.fill(p[:, 0], p[:, 1], color=color, alpha=.10)
            ax.plot(p[:, 0], p[:, 1], color=color, lw=1.7, label=rf'$t={t:g}$')
        circle = r*cw.unit(theta)
        ax.plot(circle[:, 0], circle[:, 1], ':', color='0.55', lw=.8)
        vertices = (1-r)*cw.unit(np.arange(3)*2*np.pi/3)
        ax.plot(vertices[:, 0], vertices[:, 1], 'o', color=colors[0], ms=3.5)
        ax.set_title(f'{kind.capitalize()} caps, $r={r:.2f}$', fontsize=12)
        ax.set_aspect('equal')
        ax.set_xlim(-.71, .71); ax.set_ylim(-.65, .65)
        ax.axis('off'); ax.legend(loc='lower center', ncol=3, frameon=False, fontsize=9)
        ax = axes[1, col]
        values = [lower, .40, .45, .50] if kind == 'straight' else [lower, .45, .48, .50]
        for radius, color in zip(values, colors):
            ts = np.linspace(0, radius, 400)
            label = r'$r=1/3$ (triangle)' if kind == 'straight' and radius == lower else (
                    r'$r=r_0$ (Reuleaux)' if radius == lower else rf'$r={radius:.2f}$')
            ax.plot(ts, [inner_area(kind, radius, t) for t in ts], color=color,
                    lw=2.1 if radius == lower else 1.5, label=label)
        ax.set_xlim(0, .50); ax.set_ylim(0, .82)
        ax.set_xlabel('Erosion depth $t$'); ax.set_ylabel(r'$|(S_r)_{-t}|$')
        ax.grid(alpha=.20); ax.legend(fontsize=9, frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
    out = ROOT/'figures/inner_parallel_comparison'
    fig.savefig(out.with_suffix('.pdf'))
    fig.savefig(out.with_suffix('.png'), dpi=180)
    # Freeze the composite layout before exporting each axis by itself.
    fig.canvas.draw()
    fig.set_layout_engine(None)
    renderer = fig.canvas.get_renderer()
    panel_axes = axes.ravel()
    panel_names = ['inner_parallel_straight_caps', 'inner_parallel_curved_caps',
                   'inner_parallel_straight_areas', 'inner_parallel_curved_areas']
    bounds = [ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
              for ax in panel_axes]
    width = max(box.width for box in bounds) + .08
    height = max(box.height for box in bounds) + .08
    for ax, box, name in zip(panel_axes, bounds, panel_names):
        for other in panel_axes:
            other.set_visible(other is ax)
        crop = Bbox.from_bounds((box.x0 + box.x1 - width)/2,
                                box.y1 + .04 - height, width, height)
        fig.savefig(ROOT/'figures'/f'{name}.pdf', bbox_inches=crop)
    plt.close(fig)
    out.with_suffix('.geometry.json').write_text(json.dumps(dict(
        width=1, straight_inradius=.40, curved_inradius=.45,
        erosion_depths=[0, .12, .26], cap_axes=[0, 2*np.pi/3, 4*np.pi/3],
        panel_files=[f'{name}.pdf' for name in panel_names],
        boundary_samples=3601, formulas='unified/inner_parallel.tex',
        note='Analytic radial coordinates sampled for rendering; equal axis scales.'), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draw', action='store_true')
    args = parser.parse_args()
    checks()
    if args.draw:
        draw()
