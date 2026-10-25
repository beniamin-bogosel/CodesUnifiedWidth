"""Reproducible mathematical illustration; see check_geometry.py for checks."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from explore_caps import R0, caps, hexagon
from check_geometry import samples, cw

root=Path(__file__).parent
fig, axes=plt.subplots(2,2,figsize=(8.5,7.4))
axes=axes.ravel()
for ax,r in zip(axes[:3],[R0,.45,.5]):
    H=hexagon(r)
    # Independent geometry backend reconstructs the disk polygon.
    centers=np.unique(np.round(H[:,:2],14),axis=0)
    arcs=cw.disk_arcs(centers,1.)
    p=cw.boundary(arcs,100)
    t=np.linspace(0,2*np.pi,700)
    if r<.5:
        S,_=caps(r)
        s=samples(S,120)
    else:
        s=.5*np.column_stack([np.cos(t),np.sin(t)])
    ax.fill(s[:,0],s[:,1],color='#d6eafa',label=r'$E_r$')
    ax.plot(s[:,0],s[:,1],color='#235d88',lw=1.7)
    ax.fill(p[:,0],p[:,1],color='#f5cb8b',alpha=.95,label=r'$H_r$')
    ax.plot(p[:,0],p[:,1],color='#9b5a0b',lw=1.5)
    ax.plot(r*np.cos(t),r*np.sin(t),color='#235d88',lw=.8,ls=':')
    ax.plot((1-r)*np.cos(t),(1-r)*np.sin(t),color='0.45',lw=.8,ls='--')
    theta=np.arange(6)*np.pi/3
    radius=np.array([1-r,r]*3)
    ax.scatter(radius*np.cos(theta),radius*np.sin(theta),s=12,color='#774306',zorder=4)
    ax.set_title(r'$r=r_0$: $H_r=E_r=R$' if r==R0 else rf'$r={r:.2f}$')
    ax.set_xlim(-.64,.64);ax.set_ylim(-.6,.6)
    ax.set_aspect('equal');ax.axis('off')
axes[1].legend(loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.1))
ax=axes[3];r=.45;d=1-r
A=np.array([d,0]); B=r*np.array([.5,np.sqrt(3)/2]); Q=np.array([(1-2*r)/(2-3*r),0])
arc=hexagon(r)[0];C=arc[:2];thm=arc[3]+arc[4]/2;M=C+np.array([np.cos(thm),np.sin(thm)])
u=np.linspace(0,1,201);th=arc[3]+u*arc[4];pts=C+np.array([np.cos(th),np.sin(th)]).T
domain=np.vstack([[0,0],pts,[0,0]])
ax.fill(domain[:,0],domain[:,1],color='#f5cb8b',alpha=.7)
half=np.vstack([Q,pts[100:],Q]);ax.fill(half[:,0],half[:,1],color='#8bc4e5',alpha=.8)
ax.plot(pts[:,0],pts[:,1],color='#9b5a0b',lw=2)
ax.plot([0,A[0]],[0,0],color='0.25');ax.plot([0,B[0]],[0,B[1]],color='0.25')
ax.plot([Q[0],B[0]],[Q[1],B[1]],color='#235d88',lw=1.3)
ax.plot([Q[0],M[0]],[Q[1],M[1]],color='0.35',ls='--')
for point,label,shift in [(np.zeros(2),'O',(-.025,-.035)),(A,'A',(.01,-.02)),(B,'B',(-.025,.02)),(Q,'Q',(-.01,-.035)),(M,'M',(.013,.01))]:
    ax.plot(*point,'o',ms=3,color='0.2');ax.text(*(point+shift),f'${label}$',fontsize=11)
ax.text(.38,.055,r'$1-r$',fontsize=11)
ax.text(.05,.21,r'$r$',fontsize=11)
ax.text(.23,.15,r'$D_B$',fontsize=11,color='#235d88')
ax.set_title('Mixed boundary reflection')
ax.set_xlim(-.05,.61);ax.set_ylim(-.08,.48);ax.set_aspect('equal');ax.axis('off')
fig.tight_layout(w_pad=2,h_pad=2.5)
(root/'figures').mkdir(exist_ok=True)
fig.savefig(root/'figures'/'comparison.pdf',bbox_inches='tight')
fig.savefig(root/'figures'/'comparison.png',dpi=180,bbox_inches='tight')
# Export each panel on a common canvas, with other axes hidden so each PDF
# contains only that panel's vector artwork. Keep the titles aligned at the top.
panel_names = ['comparison_r0', 'comparison_r045', 'comparison_r050',
               'comparison_reflection']
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
bounds = [ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
          for ax in axes]
width = max(box.width for box in bounds) + .08
height = max(box.height for box in bounds) + .08
for ax, box, name in zip(axes, bounds, panel_names):
    for other in axes:
        other.set_visible(other is ax)
    crop = Bbox.from_bounds((box.x0 + box.x1 - width)/2,
                            box.y1 + .04 - height, width, height)
    fig.savefig(root/'figures'/f'{name}.pdf', bbox_inches=crop)
plt.close(fig)
(root/'figures'/'comparison.scene.json').write_text(json.dumps(dict(
    width=1, inradii=[float(R0), .45, .5], reflection_inradius=.45,
    panel_files=[f'{name}.pdf' for name in panel_names],
    panel_order=['Reuleaux endpoint', 'Intermediate cap body', 'Disk endpoint',
                 'Mixed boundary reflection'],
    note='Analytic circular arcs sampled for rendering; each PDF is a separate vector panel.'),
    indent=2)+'\n')
print('Generated comparison.pdf and four separate panel PDFs.')
