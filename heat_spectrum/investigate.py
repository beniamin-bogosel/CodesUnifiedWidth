"""Separate numerical investigation: angular spectra and Dirichlet heat content.

Analytic lines/arcs, chordal boundary meshes, FreeFEM P2 elements.
Floating-point evidence, not interval eigenvalue certification.
"""
from pathlib import Path
import sys,json,subprocess,argparse
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from explore_caps import caps,hexagon

USE_CACHE = True

def boundary(kind,r,angles):
    angles=np.asarray(angles)
    if kind=='curved':
        arcs,beta=caps(r,angles)
        gaps=np.diff(np.r_[angles,angles[0]+2*np.pi])
        assert np.max(gaps)<=np.pi-beta+1e-10
        return np.c_[np.zeros(len(arcs)),arcs[:,:5]]
    a=np.arccos(r/(1-r))
    assert np.min(np.diff(np.r_[angles,angles[0]+2*np.pi]))>=2*a-1e-10
    def point(rad,theta):return rad*np.array([np.cos(theta),np.sin(theta)])
    rows=[]
    for j,theta in enumerate(angles):
        nexttheta=angles[(j+1)%3]+(2*np.pi if j==2 else 0)
        rows.append([1,*point(1-r,theta),*point(r,theta+a),0])
        gap=nexttheta-theta-2*a
        if gap>1e-12:rows.append([0,0,0,r,theta+a,gap])
        rows.append([1,*point(r,nexttheta-a),*point(1-r,nexttheta),0])
    return np.array(rows)

def solve(name,rows,N=60,nev=24,force=False):
    folder=ROOT/'data';folder.mkdir(exist_ok=True)
    stem=f'{name}_N{N}_m{nev}'
    target=folder/(stem+'.json')
    if USE_CACHE and not force and target.exists():return json.loads(target.read_text())
    points=[];ends=[]
    for k,a,b,c,d,e in rows:
        if k==0:
            points.append(np.array([a,b])+c*np.array([np.cos(d),np.sin(d)]))
            ends.append(np.array([a,b])+c*np.array([np.cos(d+e),np.sin(d+e)]))
        else:points.append(np.array([a,b]));ends.append(np.array([c,d]))
    assert np.max(np.linalg.norm(np.array(ends)-np.roll(points,-1,axis=0),axis=1))<2e-10
    path=folder/(stem+'.dat')
    with path.open('w') as out:
        out.write(f'{len(rows)}\n');np.savetxt(out,rows,fmt='%.17g')
    proc=subprocess.run(['FreeFem++','-nw','-v','0',str(ROOT/'spectrum.edp'),
                         '-input',str(path),'-N',str(N),'-nev',str(nev)],capture_output=True,text=True)
    (folder/(stem+'.log')).write_text(proc.stdout+proc.stderr)
    if proc.returncode:raise RuntimeError(proc.stdout+proc.stderr)
    meta=[s for s in proc.stdout.splitlines() if s.startswith('DOMAIN ')][0].split()
    modes=np.array([list(map(float,s.split()[1:])) for s in proc.stdout.splitlines() if s.startswith('MODE ')])
    modes=modes[np.argsort(modes[:,0])]
    result={'name':name,'N':N,'nev':nev,'area':float(meta[1]),'vertices':int(meta[2]),'dofs':int(meta[3]),'eigenvalues':modes[:,0].tolist(),'heat_coefficients':modes[:,1].tolist(),'max_rayleigh_defect':float(modes[:,2].max())}
    times=np.array([.003,.005,.01,.02,.05,.1,.2])
    heat=np.exp(-times[:,None]*modes[:,0])@modes[:,1]
    tail=np.exp(-times*modes[-1,0])*max(0,result['area']-modes[:,1].sum())
    result['heat']={str(t):{'partial_sum':float(h),'spectral_tail_estimate':float(b)} for t,h,b in zip(times,heat,tail)}
    target.write_text(json.dumps(result,indent=2)+'\n')
    return result

def run(N,nev):
    equal=np.arange(3)*2*np.pi/3
    results=[]
    for kind,r in [('straight',.4),('curved',.45)]:
        for eps in ([0.,.05,.15,.3] if kind=='straight' else [0.,.05,.15]):
            angles=equal.copy();angles[1]+=eps
            name=f'{kind}_r{r}_eps{eps}'
            res=solve(name,boundary(kind,r,angles),N,nev)
            res.update(kind=kind,r=r,epsilon=eps,angles=angles.tolist())
            results.append(res)
            print(name,np.round(res['eigenvalues'][:9],6).tolist(),flush=True)
    (ROOT/f'scan_N{N}_m{nev}.json').write_text(json.dumps(results,indent=2)+'\n')
    return results

def refinements():
    rows=[]
    for N in [100,160]:
        for kind,r in [('straight',.4),('curved',.45)]:
            for eps in [0.,.15]:
                axes=np.arange(3)*2*np.pi/3;axes[1]+=eps
                row=solve(f'{kind}_r{r}_eps{eps}',boundary(kind,r,axes),N,24)
                row.update(kind=kind,r=r,epsilon=eps);rows.append(row)
    (ROOT/'refinements.json').write_text(json.dumps(rows,indent=2)+'\n')

def survey():
    rng=np.random.default_rng(20261023);equal=np.arange(3)*2*np.pi/3;rows=[]
    for kind,rs in [('straight',[.35,.4,.47]),('curved',[.43,.45,.49])]:
        for r in rs:
            a=np.arccos(r/(1-r)) if kind=='straight' else 2*np.arccos(1/(2*(1-r)))
            eq=solve(f'survey_{kind}_{r}_equal',boundary(kind,r,equal),45,8)
            trials=[]
            while len(trials)<8:
                gaps=2*a+(2*np.pi-6*a)*rng.dirichlet(np.ones(3))
                if kind=='curved' and max(gaps)>np.pi-a:continue
                axes=np.r_[0,np.cumsum(gaps[:2])]
                un=solve(f'survey_{kind}_{r}_{len(trials)}',boundary(kind,r,axes),45,8)
                trials.append({'gaps':gaps.tolist(),
                               'delta_lambda2':un['eigenvalues'][1]-eq['eigenvalues'][1],
                               'delta_lambda3':un['eigenvalues'][2]-eq['eigenvalues'][2]})
            rows.append(dict(kind=kind,r=r,baseline=eq['eigenvalues'][:3],trials=trials))
    (ROOT/'lambda2_survey.json').write_text(json.dumps(rows,indent=2)+'\n')

def heat_study(refine=False):
    rows=[]
    cases=[('straight',[.4]),('curved',[.45])] if refine else [
        ('straight',[1/3,.4,.46]),('curved',[1-1/np.sqrt(3),.45,.48])]
    for kind,rs in cases:
        for r in rs:
            for eps in ([0.,.15] if r in [.4,.45] else [0.]):
                axes=np.arange(3)*2*np.pi/3;axes[1]+=eps
                row=solve(f'heat_{kind}_{r}_eps{eps}',boundary(kind,r,axes),160 if refine else 100,64)
                row.update(kind=kind,r=r,epsilon=eps);rows.append(row)
    name='heat_refinements.json' if refine else 'heat_checks.json'
    (ROOT/name).write_text(json.dumps(rows,indent=2)+'\n')

def benchmark():
    from scipy.special import jn_zeros
    disk=solve('disk',np.array([[0,0,0,.5,0,2*np.pi]]),100,24)
    exact=[]
    for m in range(9):
        for z in jn_zeros(m,8):exact.extend([4*z*z]*(1 if m==0 else 2))
    exact=np.sort(exact)[:24]
    rel=float(np.max(abs(np.array(disk['eigenvalues'])[:12]/exact[:12]-1)))
    heat={};zeros=jn_zeros(0,100)
    for t in [.01,.02,.05,.1]:
        value=float(np.sum(np.pi/zeros**2*np.exp(-4*zeros**2*t)))
        observed=disk['heat'][str(t)]['partial_sum']
        heat[str(t)]=dict(exact_bessel_sum=value,FEM_partial_sum=observed,absolute_error=abs(value-observed))
    tri=solve('heat_straight_0.3333333333333333_eps0.0',boundary('straight',1/3,np.arange(3)*2*np.pi/3),100,64)
    target=np.array([4*np.pi**2,28*np.pi**2/3,28*np.pi**2/3,16*np.pi**2])
    trirel=float(np.max(abs(np.array(tri['eigenvalues'])[:4]/target-1)))
    assert rel<2e-4 and trirel<2e-6
    out=dict(disk_max_relative_error_first_12=rel,disk_heat_check=heat,
             triangle_first_four_relative_error=trirel,triangle_eigenvalues=tri['eigenvalues'][:4])
    (ROOT/'benchmark.json').write_text(json.dumps(out,indent=2)+'\n')

def hex_heat():
    rows=[]
    for kind,rs in [('straight',[.4,.46]),('curved',[.45,.48])]:
        base=1/3 if kind=='straight' else 1-1/np.sqrt(3)
        ref=solve(f'heat_{kind}_{base}_eps0.0',boundary(kind,base,np.arange(3)*2*np.pi/3),100,64)
        for r in rs:
            if kind=='curved':
                arcs=hexagon(r);shape=np.c_[np.zeros(len(arcs)),arcs[:,:5]]
            else:
                aa=np.arange(6)*np.pi/3
                vv=np.where(np.arange(6)%2==0,1-r,r)[:,None]*np.c_[np.cos(aa),np.sin(aa)]
                shape=np.array([[1,*a,*b,0] for a,b in zip(vv,np.roll(vv,-1,axis=0))])
            row=solve(f'hex_heat_{kind}_{r}',shape,75,24)
            differences={t:row['heat'][t]['partial_sum']-ref['heat'][t]['partial_sum'] for t in ['0.05','0.1','0.2']}
            assert min(differences.values())>0
            rows.append(dict(kind=kind,r=r,differences_from_Q=differences))
    (ROOT/'hex_heat_checks.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--N',type=int,default=55);parser.add_argument('--nev',type=int,default=24)
    parser.add_argument('--study',choices=['scan','refine','survey','heat','heat-refine','benchmark','hex-heat'],default='scan')
    parser.add_argument("--force", action="store_true", help="Recompute solver cases even if cached JSON exists")
    args=parser.parse_args()
    USE_CACHE = not args.force
    if args.study=='scan':run(args.N,args.nev)
    elif args.study=='refine':refinements()
    elif args.study=='survey':survey()
    elif args.study=='heat':heat_study()
    elif args.study=='heat-refine':heat_study(True)
    elif args.study=='hex-heat':hex_heat()
    else:benchmark()
