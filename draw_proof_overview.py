"""Six-panel overview from analytic cap profiles and disk-polygon arcs."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kernel_width.computations import opening, profile
from explore_caps import hexagon
from check_geometry import cw

ROOT = Path(__file__).resolve().parent
(ROOT / "figures").mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 12, 'mathtext.fontset': 'cm'})
fig, axs = plt.subplots(2, 3, figsize=(7.2, 3.6))
theta = np.linspace(0, 2*np.pi, 2401)
directions = cw.unit(theta)
equal = np.arange(3)*2*np.pi/3
scene = {'description': 'Straight and curved S_r, E_r, H_r; width normalized to one',
         'rows': [], 'rendering': 'Analytic boundaries sampled for vector rendering'}
for row, (kind, r) in enumerate([('straight', .40), ('curved', .45)]):
    a = opening(kind, r)
    # Distribute the free angular gap unequally while keeping all caps disjoint.
    gaps = 2*a+(2*np.pi-6*a)*np.array([.20, .35, .45])
    angles = np.r_[0., np.cumsum(gaps[:2])]
    assert min(gaps) >= 2*a
    if kind == 'curved':
        assert max(gaps) <= np.pi-a
    vertices = np.array([1-r, r]*3)[:, None]*cw.unit(np.arange(6)*np.pi/3)
    if kind == 'straight':
        h = vertices
    else:
        centers = hexagon(r)[:, :2]
        h = cw.boundary(cw.disk_arcs(centers, 1.), 900)
    e = profile(kind, r, theta)[0][:, None]*directions
    s = profile(kind, r, theta, angles)[0][:, None]*directions
    # Verify the illustrated hexagon lies in its equilateral cap body.
    htheta = np.arctan2(h[:, 1], h[:, 0])
    assert np.max(np.linalg.norm(h, axis=1)-profile(kind, r, htheta)[0]) < 1e-10
    for col, (points, symbol) in enumerate([(s, 'S'), (e, 'E'), (h, 'H')]):
        ax = axs[row, col]
        ax.fill(e[:, 0], e[:, 1], color='#e9f1f7', zorder=0) if col == 2 else None
        if col == 2:
            ax.plot(e[:, 0], e[:, 1], color='#235d88', lw=1, ls='--')
        color = '#a05b0b' if col == 2 else '#235d88'
        fill = '#f5d39d' if col == 2 else '#d6eafa'
        ax.fill(points[:, 0], points[:, 1], facecolor=fill, edgecolor=color, lw=1.4)
        ax.plot(r*directions[:, 0], r*directions[:, 1], color='0.4', ls=':', lw=.8)
        ax.plot(0, 0, '.', color='0.4', ms=3)
        marks = vertices if col == 2 else (1-r)*cw.unit(angles if col == 0 else equal)
        ax.scatter(marks[:, 0], marks[:, 1], s=9, color=color, zorder=4)
        ax.set_title(rf'${symbol}_r^{{\mathrm{{{kind[0]}}}}}$', pad=1)
        ax.set(xlim=(-.66, .66), ylim=(-.60, .60), aspect='equal')
        ax.axis('off')
    axs[row, 0].text(-.77, 0, f'{kind.capitalize()}\n$r={r:.2f}$',
                     ha='right', va='center', fontsize=10)
    scene['rows'].append({'kind': kind, 'r': r, 'unequal_axes': angles.tolist(),
                          'equal_axes': equal.tolist(), 'hexagon_vertices': vertices.tolist()})
fig.subplots_adjust(left=.16, right=.995, top=.94, bottom=.02, wspace=.06, hspace=.22)
for extension in ['pdf', 'png']:
    fig.savefig(ROOT/'figures'/f'proof_overview.{extension}', dpi=200, bbox_inches='tight')
(ROOT/'figures'/'proof_overview.geometry.json').write_text(json.dumps(scene, indent=2)+'\n')
print('Generated six-panel overview; admissible gaps and H_r containment checked.')
