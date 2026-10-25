"""Reproducible numerical probes for the curved three-cap comparison.

The cap geometry is analytic. FreeFEM P2 results use polygonal meshes and
are evidence only. No interval/PDE certification is claimed.
"""
from pathlib import Path
import argparse
import json
import math
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parent
R0 = 1 - 1 / math.sqrt(3)


def caps(r, angles=None):
    """CCW circular boundary and its inradius derivative at fixed angles."""
    angles = np.arange(3)*2*np.pi/3 if angles is None else np.array(angles)
    d = 1-r
    beta = math.acos(np.clip(1/(2*d*d)-1, -1, 1))
    assert np.all(np.diff(np.r_[angles, angles[0]+2*np.pi]) >= 2*beta-1e-12)
    arcs = []
    for j,a in enumerate(angles):
        # Apex -> upper tangent point.
        for upper in [True, False]:
            if not upper:
                nexta = angles[(j+1)%3] + (2*np.pi if j==2 else 0)
                gap = nexta-a-2*beta
                if gap > 1e-12:
                    arcs.append([0,0,r,a+beta,gap,0,0,1])
                ac = nexta-beta
                start = ac
                length = beta/2
            else:
                ac = a+beta
                start = a+beta/2
                length = beta/2
            e = np.array([np.cos(ac),np.sin(ac)])
            ep = np.array([-e[1],e[0]])
            sign = 1 if upper else -1
            center = -d*e
            centerprime = e + sign*ep/(d*d*np.sin(beta))
            arcs.append([*center,1,start,length,*centerprime,0])
    return np.array(arcs), beta


def regular(n):
    d=1/(2*np.cos(np.pi/(2*n)))
    a=np.arange(n)*2*np.pi/n
    vertices=d*np.column_stack([np.cos(a),np.sin(a)])
    return vertex_arcs(vertices)


def hexagon(r):
    """Six unit arcs through vertices at alternating radii 1-r and r."""
    d=1-r
    angle=np.arange(6)*np.pi/3
    radius=np.array([d,r,d,r,d,r])
    v=radius[:,None]*np.column_stack([np.cos(angle),np.sin(angle)])
    vp=np.column_stack([np.cos(angle),np.sin(angle)])*np.array([-1,1,-1,1,-1,1])[:,None]
    out=[]
    for j in range(6):
        A,B=v[j],v[(j+1)%6]
        Ap,Bp=vp[j],vp[(j+1)%6]
        edge=B-A; L=np.linalg.norm(edge); Lp=np.dot(edge,Bp-Ap)/L
        nm=np.array([edge[1],-edge[0]])/L
        nmp=np.array([(Bp-Ap)[1],-(Bp-Ap)[0]])/L-nm*Lp/L
        h=np.sqrt(1-L*L/4); hp=-L*Lp/(4*h)
        C=(A+B)/2-h*nm
        Cp=(Ap+Bp)/2-hp*nm-h*nmp
        start=np.arctan2((A-C)[1],(A-C)[0])
        out.append([*C,1,start,2*np.arcsin(L/2),*Cp,0])
    return np.array(out)


def pentagon(t):
    d=1/(2*np.cos(np.pi/10))
    aa=np.arange(5)*2*np.pi/5
    v=d*np.column_stack([np.cos(aa),np.sin(aa)])
    v[0,0]+=t
    for j,k,sgn in [(2,4,-1),(3,1,1)]:
        z=v[k]-v[0]
        length=np.linalg.norm(z)
        v[j]=(v[k]+v[0])/2+sgn*np.sqrt(1-length**2/4)/length*np.array([-z[1],z[0]])
    return vertex_arcs(v)


def vertex_arcs(v):
    n=len(v)
    centers=v[(np.arange(n)+(n+1)//2)%n]
    p=v-centers
    q=np.roll(v,-1,axis=0)-centers
    start=np.arctan2(p[:,1],p[:,0])
    lengths=np.arctan2(p[:,0]*q[:,1]-p[:,1]*q[:,0],(p*q).sum(axis=1))
    assert np.all(lengths>0)
    return np.column_stack([centers,np.ones(n),start,lengths,np.zeros((n,3))])


def evaluate(arcs, N, name):
    out=ROOT/'outputs'
    out.mkdir(exist_ok=True)
    data=out/f'{name}_{N}.dat'
    flux=out/f'{name}_{N}_flux.dat'
    with data.open('w') as f:
        f.write(f'{len(arcs)}\n')
        np.savetxt(f,arcs,fmt='%.17g')
    # Verify that every adjoining pair meets before asking the mesher.
    p=arcs[:,:2]+arcs[:,2,None]*np.column_stack([np.cos(arcs[:,3]),np.sin(arcs[:,3])])
    end=arcs[:,3]+arcs[:,4]
    q=arcs[:,:2]+arcs[:,2,None]*np.column_stack([np.cos(end),np.sin(end)])
    assert np.max(np.linalg.norm(q-np.roll(p,-1,axis=0),axis=1))<1e-10
    proc=subprocess.run(['FreeFem++','-nw','-v','0',str(ROOT/'torsion_arcs.edp'),
                         '-input',str(data),'-flux',str(flux),'-N',str(N)],
                        capture_output=True,text=True,check=False)
    (out/f'{name}_{N}.log').write_text(proc.stdout+proc.stderr)
    if proc.returncode:
        raise RuntimeError(proc.stdout+proc.stderr)
    line=next(s for s in proc.stdout.splitlines() if s.startswith('RESULT '))
    A,T,deriv,nv=map(float,line.split()[1:])
    return dict(area=A,torsion=T,derivative=deriv,vertices=int(nv),N=N,flux=str(flux.relative_to(ROOT)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--N',type=int,default=50)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--hexagon',action='store_true')
    args=parser.parse_args()
    res={}
    if args.hexagon:
        for r in [R0,R0+1e-4,.425,.43,.44,.45,.46,.48,.495,.5]:
            name=f'hexagon_{r:.9f}'
            res[name]=evaluate(hexagon(r),args.N,name)
            res[name].update(r=r)
            print(name,res[name],flush=True)
        (ROOT/'outputs'/f'hexagon_results_{args.N}.json').write_text(json.dumps(res,indent=2)+'\n')
        return
    for r in ([R0,.43,.45,.48,.495] if args.quick else [R0,R0+1e-4,.425,.43,.44,.45,.46,.48,.495]):
        name=f'caps_{r:.9f}'
        arcs,beta=caps(r)
        res[name]=evaluate(arcs,args.N,name)
        res[name].update(r=r,beta=beta)
        print(name,res[name],flush=True)
    for n in [3,5,7,9]:
        res[f'regular_{n}']=evaluate(regular(n),args.N,f'regular_{n}')
        print(f'regular_{n}',res[f'regular_{n}'],flush=True)
    for t in [-.02,-.01,0,.01,.02]:
        name=f'pentagon_{t:+.4f}'
        res[name]=evaluate(pentagon(t),args.N,name)
        print(name,res[name],flush=True)
    for r,angles in [(.43,[0,2.02,4.13]),(.45,[0,1.8,4.05]),(.48,[0,1.2,3.8])]:
        name=f'irregular_{r}'
        res[name]=evaluate(caps(r,angles)[0],args.N,name)
        print(name,res[name],flush=True)
    (ROOT/'outputs'/f'caps_results_{args.N}.json').write_text(json.dumps(res,indent=2)+'\n')


if __name__=='__main__':
    main()
