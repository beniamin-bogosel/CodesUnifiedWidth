"""Exact symbolic identities and independent floating geometry checks.

Symbolic assertions verify algebra used in the proof. Sampling is an
implementation check, not the proof of geometric containment.
"""
from pathlib import Path
import json
import numpy as np
import sympy as sp
from explore_caps import R0, caps, hexagon

import cwgeom as cw


def samples(arcs, N=120):
    pts=[]
    for cx,cy,R,a,s,*_ in arcs:
        t=a+s*np.linspace(0,1,N)
        pts.append(np.array([cx+R*np.cos(t),cy+R*np.sin(t)]).T)
    return np.concatenate(pts)


def main():
    r=sp.symbols('r',real=True)
    L2=1-3*r+3*r*r
    identity=(4-L2)*(3*r-1)**2-3*(1-r)**2*L2
    assert sp.expand(identity+12*r*r*(3*r*r-6*r+2))==0
    assert sp.simplify(L2.subs(r,1-sp.sqrt(3)/3)-(2-sp.sqrt(3)))==0
    # Two expressions for the normal component of the center velocity.
    L=sp.sqrt(L2); h=sp.sqrt(1-L2/4)
    nm=sp.Matrix([sp.sqrt(3)*r/2,1-sp.Rational(3,2)*r])/L
    tm=sp.Matrix([-nm[1],nm[0]])
    M=sp.Matrix([(2-r)/4,sp.sqrt(3)*r/4])
    C=M-h*nm
    Cp=C.diff(r)
    a=(1-2*r)*(sp.sqrt(3)/(4*L)-3/(8*h))
    b=1/(4*L)+sp.sqrt(3)*h/(2*L2)
    assert sp.simplify(Cp.dot(nm)-a)==0
    assert sp.simplify(Cp.dot(tm)-b)==0
    checks=[]
    for rr in np.linspace(R0,.5,61):
        H=hexagon(float(rr)); p=samples(H)
        theta=np.arange(6)*np.pi/3
        radii=np.array([1-rr,rr]*3)
        v=radii[:,None]*np.array([np.cos(theta),np.sin(theta)]).T
        # Every generating disk contains all vertices and the explicit arcs.
        vertex_res=np.max(np.linalg.norm(v[:,None,:]-H[None,:,:2],axis=2)-1)
        arc_res=np.max(np.linalg.norm(p[:,None,:]-H[None,:,:2],axis=2)-1)
        assert vertex_res<2e-13 and arc_res<2e-13
        for j in range(6):
            possible=cw.circle_intersections(v[j],v[(j+1)%6],1.)
            assert min(np.linalg.norm(H[j,:2]-z) for z in possible)<2e-13
        # H lies in the explicitly constructed curved caps. A cap disk
        # contains its incircle plus apex; checking its whole family reduces
        # to the radial formula in the proof. Compare sampled radial curves.
        if rr<.5-1e-12:
            S,beta=caps(float(rr))
            ang=np.arctan2(p[:,1],p[:,0])
            delta=np.min(abs((ang[:,None]-np.arange(3)[None,:]*2*np.pi/3+np.pi)%(2*np.pi)-np.pi),axis=1)
            cap_c=-np.array([np.cos(beta),np.sin(beta)])*(1-rr)
            # On a ray inside a cap: |t e_delta - C|=1.
            projection=-(1-rr)*np.cos(beta-delta)
            bound=projection+np.sqrt(projection**2+1-(1-rr)**2)
            bound=np.where(delta<beta,bound,rr)
            assert np.max(np.linalg.norm(p,axis=1)-bound)<3e-13
        checks.append(dict(r=float(rr),vertex_residual=float(vertex_res),arc_residual=float(arc_res)))
    # Three distinct centers, each repeated twice, at the Reuleaux endpoint.
    C0=hexagon(R0)[:,:2]
    assert np.linalg.norm(C0[0]-C0[1])<1e-13
    assert np.linalg.norm(C0[2]-C0[3])<1e-13
    assert np.linalg.norm(C0[4]-C0[5])<1e-13
    out=Path(__file__).parent/'outputs'
    out.mkdir(exist_ok=True)
    (out/'geometry_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print('PASS: exact center-velocity identities, endpoint, and 61 independent geometry checks.')


if __name__=='__main__':
    main()
