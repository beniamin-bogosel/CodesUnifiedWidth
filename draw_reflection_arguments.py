"""Reproduce the three reflection/source illustrations from analytic boundaries.

The geometry diagnostics are floating-point rendering checks, not PDE proofs.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kernel_width.computations import opening, profile
from check_geometry import cw

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'figures'
BLUE, GOLD, PURPLE = '#235d88', '#a05b0b', '#79508b'
PALE, LILAC = '#dcebf6', '#eee4f2'
B, C = 1.85, 2.22
AXES = np.array([0., B, 2*np.pi-C])
plt.rcParams.update({'font.size': 11, 'mathtext.fontset': 'cm'})
SCENE = {'width_normalization': 1, 'axes': AXES.tolist(), 'rows': [],
         'note': 'Analytic radial profiles sampled for rendering; diagnostics are not proofs.'}


def curve(kind, r, angles, axes=AXES):
    return profile(kind, r, angles, axes)[0][:, None]*cw.unit(angles)


def line(ax, p, **kw):
    ax.plot(p[:, 0], p[:, 1], **kw)


def fill(ax, p, **kw):
    ax.fill(p[:, 0], p[:, 1], **kw)


def mark(ax, p, label, offset=(5, 5), color='0.2'):
    ax.plot(*p, 'o', ms=3, color=color, zorder=7)
    ax.annotate(label, p, xytext=offset, textcoords='offset points',
                color=color, fontsize=11, zorder=8)


def normal(kind, r, angle, axes=AXES):
    p, dp = profile(kind, r, np.array([angle]), axes)
    e = cw.unit(angle)
    n = p[0]*e-dp[0]*np.array([-e[1], e[0]])
    return n/np.linalg.norm(n)


def arrow(ax, p, q, color='0.3', **kw):
    ax.annotate('', q, xytext=p, arrowprops=dict(arrowstyle='->', color=color,
                 lw=1.15, **kw), zorder=6)


def arc(ax, radius, lo, hi, **kw):
    line(ax, radius*cw.unit(np.linspace(lo, hi, 250)), **kw)


def setup(ax, title, right=.83):
    ax.set(aspect='equal', xlim=(-.78, right), ylim=(-.73, .73))
    ax.axis('off')
    ax.set_title(title, fontsize=12, pad=5)


def save(fig, stem, alt):
    fig.savefig(OUT/f'{stem}.pdf', bbox_inches='tight')
    fig.savefig(OUT/f'{stem}.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    SCENE.setdefault('figures', {})[stem] = {'alt_text': alt}


def remote_reflection():
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.8))
    ell = (B-C)/2
    for ax, (kind, r) in zip(axs, [('straight', .4), ('curved', .45)]):
        a = opening(kind, r)
        gaps = np.diff(np.r_[AXES, 2*np.pi])
        assert min(gaps) >= 2*a
        if kind == 'curved':
            assert max(gaps) <= np.pi-a
        full = curve(kind, r, np.linspace(0, 2*np.pi, 2501))
        fill(ax, full, color='#f5f3ed')
        negative_angles = np.linspace(ell-np.pi, ell, 1501)
        negative = curve(kind, r, negative_angles)
        fill(ax, np.vstack([[0, 0], negative]), facecolor='#f3dbb4', alpha=.75)
        reflected_angles = 2*ell-negative_angles
        reflected = np.linalg.norm(negative, axis=1)[:, None]*cw.unit(reflected_angles)
        residual = np.max(np.linalg.norm(reflected, axis=1)-profile(kind, r, reflected_angles, AXES)[0])
        assert residual < 1e-12
        fill(ax, np.vstack([[0, 0], reflected]), facecolor=PALE)
        line(ax, reflected, color=BLUE, lw=1.25)
        line(ax, full, color='0.3', lw=1.2)
        line(ax, np.array([-.72, .74])[:, None]*cw.unit(ell), color=PURPLE, ls='--', lw=1.2)
        ax.text(.69, -.22, r'$\ell$', color=PURPLE)
        arc(ax, r, 0, 2*np.pi, color='0.6', ls=':', lw=.8)
        arc(ax, r, B-a, B+a, color=BLUE, lw=2.7)
        arc(ax, r, -C-a, -C+a, color=GOLD, lw=2.7, ls='--')
        for angle, name, offset in [(0, 'A', (5, 3)), (B, 'B', (0, 8)), (-C, 'C', (-10, -16))]:
            mark(ax, (1-r)*cw.unit(angle), f'${name}$', offset)
        mark(ax, np.zeros(2), '$O$', (3, -14))
        for angle, name, color, offset in [(B+.4*a, 'X_B', BLUE, (-33, 7)),
                                         (-C-.4*a, 'X_C', GOLD, (-33, -13))]:
            p = curve(kind, r, np.array([angle]))[0]
            mark(ax, p, f'${name}$', offset, color)
            arrow(ax, p, p+.12*normal(kind, r, angle), color)
        ax.text(.03, .22, r'$\sigma_\ell(S_-)$', color=BLUE, ha='center')
        ax.text(.02, -.24, r'$S_-$', color=GOLD)
        ax.text(-.17, .31, r'$\gamma_B$', color=BLUE, ha='center')
        ax.text(-.10, -.32, r'$\gamma_C$', color=GOLD, ha='center')
        ax.text(0, -.69, r'$\sigma_\ell(S_-)\subset S_+,\quad q(X_B)\geq q(X_C)$',
                fontsize=11, ha='center')
        setup(ax, f'{kind.capitalize()} caps: $r={r:.2f}$')
        SCENE['rows'].append({'kind': kind, 'r': r, 'opening': float(a),
                              'reflection_angle': ell, 'containment_residual': float(residual)})
    fig.subplots_adjust(left=.03, right=.99, top=.90, bottom=.04, wspace=.08)
    save(fig, 'remote_cap_reflection', 'Straight and curved three-cap bodies: the ochre negative half reflects across a purple dashed line into the blue positive half. Paired boundary points on caps B and C carry outward normal arrows.')


def same_cap_reflection():
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.9))
    for ax, (kind, r) in zip(axs, [('straight', .4), ('curved', .45)]):
        a = opening(kind, r)
        full = curve(kind, r, np.linspace(0, 2*np.pi, 2501))
        fill(ax, full, color='#f5f5f5')
        line(ax, full, color='0.55', lw=1)
        upper_angles = np.linspace(0, np.pi, 1301)
        upper = curve(kind, r, upper_angles, [0.])
        fill(ax, np.vstack([[0, 0], upper]), color=PALE)
        line(ax, upper, color=BLUE, lw=1.3)
        cap_angles = np.linspace(0, a, 301)
        cap = curve(kind, r, cap_angles, [0.])
        fill(ax, np.vstack([cap, r*cw.unit(cap_angles[::-1])]), color=LILAC)
        line(ax, cap, color=PURPLE, lw=2)
        # Only the remote cap C is reflected for the boundary-data comparison.
        cprime = curve(kind, r, np.linspace(C-a, C+a, 501), [C])
        line(ax, cprime, color=GOLD, lw=1.5, ls='--')
        arc(ax, r, 0, 2*np.pi, color='0.65', ls=':', lw=.8)
        arc(ax, r, B-a, B+a, color=BLUE, lw=3)
        arc(ax, r, C-a, C+a, color=GOLD, lw=2, ls='--')
        line(ax, np.array([[-.72, 0], [.78, 0]]), color=PURPLE, ls='--', lw=1.1)
        ax.text(-.70, -.07, r'$\sigma$', color=PURPLE)
        for angle, name, offset, color in [(0, 'A', (5, 1), '0.2'), (B, 'B', (5, 7), BLUE),
                                          (-C, 'C', (-10, -16), '0.4'), (C, "C'", (-24, 5), GOLD)]:
            mark(ax, (1-r)*cw.unit(angle), f'${name}$', offset, color)
        mark(ax, np.zeros(2), '$O$', (3, -14))
        point = curve(kind, r, np.array([.33*a]), [0.])[0]
        for sign, label in [(1, '+'), (-1, '-')]:
            p = point*np.array([1, sign])
            mark(ax, p, f'$X_{label}$', (12, 7 if sign == 1 else -16), PURPLE)
            arrow(ax, p, p+.11*normal(kind, r, sign*.33*a, [0.]), PURPLE)
        line(ax, np.array([point, point*np.array([1, -1])]), color=PURPLE, ls=':', lw=1)
        ax.text(-.08, .18, '$G$', color=BLUE, fontsize=13)
        ax.text(.10, -.20, r'$q_+>q_-$', color=PURPLE)
        ax.text(0, -.69, r'$w=u-u\circ\sigma>0$ in the upper half-cap', ha='center', fontsize=11)
        setup(ax, f'{kind.capitalize()} caps: comparison at $A$')
    fig.subplots_adjust(left=.03, right=.99, top=.90, bottom=.04, wspace=.08)
    save(fig, 'same_cap_reflection', 'The upper half-disk with the upper cap at A is shaded blue. The reflected cap C-prime is dashed ochre. Paired points X-plus and X-minus on the same cap are reflected across the horizontal axis; purple arrows indicate their outward normals.')


def sliding_sources():
    fig, axs = plt.subplots(1, 3, figsize=(8.5, 2.7))
    start, stop, shift = 1.18, 1.63, .92
    for ax, (kind, r, title) in zip(axs, [('disk', .45, 'Half-disk'),
                                         ('straight', .40, 'Straight half-cap'),
                                         ('curved', .45, 'Curved half-cap')]):
        a = .86 if kind == 'disk' else opening(kind, r)
        theta = np.linspace(0, np.pi, 1501)
        outer = r*cw.unit(theta) if kind == 'disk' else curve(kind, r, theta, [0.])
        fill(ax, np.vstack([[0, 0], outer]), color=PALE)
        line(ax, outer, color='0.3', lw=1.3)
        line(ax, np.array([[-r, 0], [outer[0, 0], 0]]), color='0.3', lw=1.3)
        if kind != 'disk':
            ta = np.linspace(0, a, 301)
            cap = curve(kind, r, ta, [0.])
            fill(ax, np.vstack([cap, r*cw.unit(ta[::-1])]), color=LILAC)
            line(ax, cap, color=PURPLE, lw=2)
            mark(ax, np.array([1-r, 0]), '$A$', (3, -14))
        arc(ax, r, 0, a, color=PURPLE, ls=':', lw=1.7)
        ax.text(*(r*cw.unit(a*.45)+np.array([-.08, 0])), '$I$', color=PURPLE)
        for offset, color, style, label in [(0, BLUE, '-', '$f$'), (shift, GOLD, '--', r'$f_\tau$')]:
            s = np.linspace(start+offset, stop+offset, 151)
            arc(ax, r, s[0], s[-1], color=color, ls=style, lw=2.8)
            # Identical bumps encode boundary data; they are not geometric boundaries.
            bump = .065*np.sin(np.pi*(s-s[0])/(s[-1]-s[0]))**2
            line(ax, (r+bump)[:, None]*cw.unit(s), color=color, ls=style, lw=1.1)
            ax.text(*((r+.13)*cw.unit((s[0]+s[-1])/2)), label, color=color, ha='center')
        arrow(ax, (r+.08)*cw.unit(stop+.04), (r+.08)*cw.unit(start+shift-.04),
              '0.3', connectionstyle='arc3,rad=.12')
        mark(ax, np.zeros(2), '$O$', (-4, -15))
        ax.text(-.08, .10, 'Zero data elsewhere', fontsize=9, ha='center')
        ax.set(aspect='equal', xlim=(-.52, .64), ylim=(-.12, .65))
        ax.axis('off')
        ax.set_title(title, fontsize=12)
    fig.subplots_adjust(left=.01, right=.99, top=.85, bottom=.02, wspace=.035)
    save(fig, 'sliding_boundary_sources', 'Three panels show a half-disk, a half-disk with a straight half-cap, and one with a curved half-cap. Identical blue solid and ochre dashed boundary profiles occupy successively larger angles. The disk-cap interface is purple and dotted.')
    SCENE['figures']['sliding_boundary_sources']['layout'] = dict(
        figsize=[8.5, 2.7], xlim=[-.52, .64], ylim=[-.12, .65], wspace=.035)


if __name__ == '__main__':
    remote_reflection()
    same_cap_reflection()
    sliding_sources()
    (OUT/'reflection_arguments.geometry.json').write_text(json.dumps(SCENE, indent=2)+'\n')
    print('Generated three diagrams; admissibility and reflected-half containment checks passed.')
