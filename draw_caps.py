"""Analytic cap construction and angular-balancing illustration.

Run ``python draw_caps.py`` to regenerate PDF/PNG and scene/diagnostic JSON.
The exact arcs are those used by explore_caps.py. Sampling is solely for
rendering; mathematical containment is proved in the manuscript.
"""
from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from explore_caps import caps

ROOT = Path(__file__).resolve().parent
import cwgeom as cw

BLUE = "#235d88"
PALE_BLUE = "#e2eef7"
OCHRE = "#9b5a0b"
PALE_OCHRE = "#f5cb8b"
PURPLE = "#79508b"


def e(t):
    return np.array([np.cos(t), np.sin(t)]).T


def boundary(arcs, count=220):
    points = []
    for cx, cy, radius, start, length, *_ in arcs:
        t = np.linspace(start, start + length, count)
        points.append(np.array([cx, cy]) + radius * e(t))
    return np.concatenate(points)


def cap_points(r, beta, angle=0):
    d = 1-r
    up = -d*e(angle+beta)+e(angle+np.linspace(beta/2, beta, 180))
    attachment = r*e(angle+np.linspace(beta, -beta, 180))
    low = -d*e(angle-beta)+e(angle+np.linspace(-beta, -beta/2, 180))
    return np.concatenate([up, attachment, low]), up, low


def mark(ax, p, text, offset=(5, 5), color="0.15"):
    ax.plot(*p, "o", ms=3.3, color=color, zorder=8)
    ax.annotate(text, p, xytext=offset, textcoords="offset points",
                fontsize=11, color=color, zorder=9)


def angle_mark(ax, start, stop, radius, label, label_radius=None):
    t = np.linspace(start, stop, 120)
    p = radius*e(t)
    ax.plot(p[:, 0], p[:, 1], lw=.85, color="0.45")
    mid = (start+stop)/2
    xy = (label_radius or radius+.045)*e(mid)
    ax.text(*xy, label, ha="center", va="center", fontsize=10, color="0.3")


def style(ax):
    ax.set_aspect("equal")
    ax.set_xlim(-.52, .67)
    ax.set_ylim(-.59, .62)
    ax.axis("off")


def draw_three(ax, r, beta, angles, labels):
    d = 1-r
    arcs, _ = caps(r, angles)
    p = boundary(arcs)
    ax.fill(p[:, 0], p[:, 1], color=PALE_BLUE, zorder=0)
    for a in angles:
        cap, _, _ = cap_points(r, beta, a)
        ax.fill(cap[:, 0], cap[:, 1], color=PALE_OCHRE, zorder=1)
        A = d*e(a)
        ax.plot([0, A[0]], [0, A[1]], color="0.6", lw=.8, ls=":", zorder=2)
    ax.plot(p[:, 0], p[:, 1], color=BLUE, lw=1.7, zorder=3)
    t = np.linspace(0, 2*np.pi, 600)
    circle = r*e(t)
    ax.plot(circle[:, 0], circle[:, 1], color=BLUE, lw=.8, ls=":", zorder=3)
    for a, label in zip(angles, labels):
        off = (6, -5) if np.cos(a) > .8 else (-16, 5 if np.sin(a) > 0 else -12)
        mark(ax, d*e(a), label, off)
    mark(ax, np.zeros(2), "$O$", (-13, -14))
    return arcs


def main():
    r = .45
    d = 1-r
    _, beta = caps(r)
    angles = np.array([0., 1.8, 4.05])
    b = angles[1]
    c = 2*np.pi-angles[2]
    ell = (b-c)/2
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2))
    fig.subplots_adjust(left=.02, right=.98, bottom=.13, top=.92,
                        wspace=.025)
    for ax in axes:
        style(ax)

    ax = axes[0]
    Cplus, Cminus = -d*e(beta), -d*e(-beta)
    A, Tplus, Tminus = d*e(0), r*e(beta), r*e(-beta)
    # Independent geometry backend verifies the selected unit-circle center.
    candidates = cw.circle_intersections(A, Tplus, 1.)
    residual = min(np.linalg.norm(Cplus-z) for z in candidates)
    assert residual < 2e-14
    single = np.array([
        [*Cplus, 1., beta/2, beta/2],
        [0., 0., r, beta, 2*np.pi-2*beta],
        [*Cminus, 1., -beta, beta/2],
    ])
    p = boundary(single)
    ax.fill(p[:, 0], p[:, 1], color=PALE_BLUE, zorder=0)
    cap, up, low = cap_points(r, beta)
    ax.fill(cap[:, 0], cap[:, 1], color=PALE_OCHRE, zorder=1)
    ax.plot(p[:, 0], p[:, 1], color=BLUE, lw=1.7, zorder=3)
    for piece in [up, low]:
        ax.plot(piece[:, 0], piece[:, 1], color=OCHRE, lw=2.3, zorder=4)
    t = np.linspace(0, 2*np.pi, 600)
    circle = r*e(t)
    ax.plot(circle[:, 0], circle[:, 1], color=BLUE, lw=.85, ls=":")
    for T in [Tplus, Tminus]:
        ax.plot([0, T[0]], [0, T[1]], color="0.55", lw=.8, ls=":")
    ax.plot([0, A[0]], [0, 0], color="0.55", lw=.8, ls=":")
    for P in [A, Tplus]:
        ax.plot([Cplus[0], P[0]], [Cplus[1], P[1]], color="0.52", lw=.9, ls="--")
    ax.text(-.16, -.13, "$1$", color="0.4", fontsize=11)
    ax.text(-.19, .19, "$B_r$", color=BLUE, fontsize=12)
    ax.text(.27, .035, "$1-r$", color="0.3", fontsize=11)
    angle_mark(ax, 0, beta, .12, "$\u03b2$", .16)
    mark(ax, A, "$A$", (6, -5))
    mark(ax, Tplus, "$T_+$", (5, 7))
    mark(ax, Tminus, "$T_-$", (5, -16))
    mark(ax, Cplus, "$C_+$", (-18, -12), PURPLE)
    mark(ax, Cminus, "$C_-$", (-18, 4), PURPLE)
    mark(ax, np.zeros(2), "$O$", (-13, -5))
    ax.set_title(r"Single curved cap $L_A$", fontsize=12, pad=12)
    ax.text(0, -.62, r"$C_\pm=-(1-r)e_{\pm\beta}$",
            ha="center", fontsize=11, color="0.3")

    ax = axes[1]
    draw_three(ax, r, beta, angles, ["$A$", "$B$", "$C$"])
    line = .64*np.array([-1., 1.])[:, None]*e(ell)
    ax.plot(line[:, 0], line[:, 1], color=PURPLE, ls="--", lw=1.0, zorder=4)
    angle_mark(ax, 0, b, .15, "$b$", .20)
    angle_mark(ax, -c, 0, .255, "$c$", .295)
    start, stop = .63*e(.18), .63*e(-.18)
    ax.annotate("", stop, xytext=start,
                arrowprops=dict(arrowstyle="->", color=OCHRE, lw=1.6,
                                connectionstyle="arc3,rad=-.12"))
    ax.text(.565, .18, "move $A$", color=OCHRE, fontsize=9, ha="center")
    ax.set_title(r"Unequal caps $S$: $b<c$", fontsize=12, pad=12)
    ax.text(0, -.62, r"Reflection $B\leftrightarrow C$", color=PURPLE,
            ha="center", fontsize=11)

    ax = axes[2]
    regular = np.arange(3)*2*np.pi/3
    draw_three(ax, r, beta, regular, ["$A_1$", "$A_2$", "$A_3$"])
    for j in range(3):
        angle_mark(ax, regular[j], regular[j]+2*np.pi/3, .18,
                   r"$2\pi/3$", .26)
    ax.set_title(r"Equilateral caps $E_r$", fontsize=12, pad=12)
    ax.text(0, -.62, r"$r=0.45$", color="0.3",
            ha="center", fontsize=11)

    fig.text(.5, .055,
             r"Angular balancing: $T(S)\geq T(E_r)$, $\quad\lambda_1(S)\leq\lambda_1(E_r)$",
             ha="center", fontsize=12, color="0.2")
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    for suffix in ["pdf", "png"]:
        fig.savefig(out / f"caps_geometry.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    scene = dict(width=1., r=r, d=d, beta=beta,
                 figsize=[10.5, 4.2],
                 layout=dict(rows=1, columns=3, wspace=.025,
                             xlim=[-.52, .67], ylim=[-.59, .62]),
                 unequal_contact_angles=angles.tolist(),
                 equilateral_contact_angles=regular.tolist(),
                 single_cap_centers=[Cplus.tolist(), Cminus.tolist()],
                 caption=("Curved-cap construction and angular balancing at r=0.45. "
                          "Blue shading is the incircle; ochre shading is the added cap. "
                          "The left panel identifies the two unit-circle centers. "
                          "The center panel shows the reflection interchanging the remote "
                          "caps and the admissible clockwise motion of A. The right panel "
                          "shows equal gaps. Boundaries use analytic circles sampled only "
                          "for rendering; the comparison domains need not have constant width."),
                 alt_text=("Three panels in one horizontal row show a single curved cap attached to a disk, "
                           "three unequally spaced caps with a reflection line, and three "
                           "equally spaced caps. Circular arc centers and contact points "
                           "are labeled in the first panel."))
    (out / "caps_geometry.scene.json").write_text(json.dumps(scene, indent=2)+"\n")
    diagnostics = dict(center_circle_intersection_residual=float(residual),
                       unequal_gaps=np.diff(np.r_[angles, 2*np.pi]).tolist(),
                       gap_lower_bound=2*beta, gap_upper_bound=np.pi-beta,
                       note="Floating-point rendering diagnostics, not geometric proofs.")
    (out / "caps_geometry.geometry.json").write_text(json.dumps(diagnostics, indent=2)+"\n")
    print(out / "caps_geometry.pdf")


if __name__ == "__main__":
    main()
