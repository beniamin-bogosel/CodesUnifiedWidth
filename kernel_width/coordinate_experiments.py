"""Coordinate weights, a width-normalized pentagon and scalar certificates.

The accompanying note supplies the geometry and deductions. mpmath.iv
certifies the scalar counterexample; plots and searches are numerical.
"""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from scipy.special import beta
from numpy.polynomial.legendre import leggauss
from mpmath import iv

from computations import BASE, profile, opening

ROOT = Path(__file__).resolve().parent


def chord_reuleaux(t, n=3):
    t = np.asarray(t)
    radius = 1/(2*np.cos(np.pi/(2*n)))
    angle = np.arange(n)*2*np.pi/n
    cx, cy = radius*np.cos(angle), radius*np.sin(angle)
    radicand = 1-(t[..., None]-cy)**2
    root = np.sqrt(np.maximum(0., radicand))
    left = np.max(cx-root, axis=-1)
    right = np.min(cx+root, axis=-1)
    return np.where((abs(t)<.5)&(np.min(radicand, axis=-1)>=0),
                    np.maximum(0.,right-left), 0.)


def moment_cap(kind, r, p, axes=None, order=160):
    if axes is None:
        axes = np.arange(3)*2*np.pi/3
    a = opening(kind,r)
    breaks = np.unique(np.r_[0., np.pi, 2*np.pi, np.asarray(axes)%(2*np.pi),
                               (np.asarray(axes)-a)%(2*np.pi),
                               (np.asarray(axes)+a)%(2*np.pi)])
    q,w=leggauss(order)
    value=0.
    for lo,hi in zip(breaks[:-1],breaks[1:]):
        theta=(lo+hi)/2+(hi-lo)*q/2
        rho=profile(kind,r,theta,axes)[0]
        value+=(hi-lo)/2*np.sum(w*abs(np.sin(theta))**p*rho**(p+2))/(p+2)
    return float(value)


def pentagon_moment(rho, p):
    h=np.sqrt(3)/2-rho
    w=rho/np.sqrt(.25+(2*rho-np.sqrt(3)/2)**2)
    raw=((rho**(p+2)-h**(p+2))/(rho-h)+h**(p+1))/((p+1)*(p+2))
    return raw/w**(p+2)


def polygon_centroid_moments(vertices):
    v=np.asarray(vertices)
    z=np.roll(v,-1,axis=0)
    cross=v[:,0]*z[:,1]-v[:,1]*z[:,0]
    area=cross.sum()/2
    first=((v+z)*cross[:,None]).sum(axis=0)/6
    raw=((v*v+v*z+z*z)*cross[:,None]).sum(axis=0)/12
    return area,first/area,raw-first*first/area


def interval_certificate():
    iv.dps=50
    rho=iv.mpf(9)/16
    c=iv.sqrt(3)/2
    h=c-rho
    width=rho/iv.sqrt(iv.mpf(1)/4+(2*rho-c)**2)
    # Green's polynomial formulas, evaluated wholly with intervals.
    v=[(iv.mpf(0),-rho),(iv.mpf(1)/2,-h),(iv.mpf(1),iv.mpf(0)),
       (iv.mpf(1)/2,h),(iv.mpf(0),rho)]
    v=[(x/width,y/width) for x,y in v]
    area=iv.mpf(0);mx=iv.mpf(0);my=iv.mpf(0)
    xx=iv.mpf(0);yy=iv.mpf(0)
    for a,b in zip(v,v[1:]+v[:1]):
        cr=a[0]*b[1]-a[1]*b[0]
        area+=cr/2
        mx+=(a[0]+b[0])*cr/6
        my+=(a[1]+b[1])*cr/6
        xx+=(a[0]**2+a[0]*b[0]+b[0]**2)*cr/12
        yy+=(a[1]**2+a[1]*b[1]+b[1]**2)*cr/12
    xx-=mx**2/area
    yy-=my**2/area
    triangle=1/(18*iv.sqrt(3))
    eps=iv.mpf(1)/10
    gaps={'quadratic_coordinate_gap':triangle-yy,
          'elliptic_quadratic_gap':(1+eps)*triangle-(eps*xx+yy),
          'triangle_minus_disk_tenth_moment':
              1/(16038*iv.sqrt(3))-21*iv.pi/(2**21)}
    assert all(val.a>0 for val in gaps.values())
    integer_margins={
        'quadratic':246024531**2-3*142041832**2,
        'elliptic_quadratic':848475652113**2-3*489867051512**2}
    assert integer_margins=={'quadratic':423740021289,
                            'elliptic_quadratic':1747757399216906337}
    result={'method':'50-digit outward-rounded mpmath.iv arithmetic',
            'rho':'9/16','epsilon':'1/10','width_before_rescaling':str(width),
            'centroid_x':str(mx/area),'central_x2':str(xx),'central_y2':str(yy),
            'exact_squared_integer_margins':integer_margins,
            'positive_margins':{k:str(val) for k,val in gaps.items()}}
    (ROOT/'coordinate_certificate.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def run():
    t=np.linspace(0,.5,1001)
    ref=np.sqrt(1-t*t)+np.sqrt(1-(t+.5)**2)-np.sqrt(3)/2
    assert np.max(abs(ref-chord_reuleaux(t,3)))<2e-8
    for n in [5,7,9]:
        assert np.min(chord_reuleaux(t,n)-ref)>-2e-8
    reference=2*quad(lambda y:y*float(chord_reuleaux(y,3)),0,.5,
                     epsabs=2e-13)[0]
    assert abs(reference-(4-np.pi)/6)<2e-12
    pc=brentq(lambda p:p-2+3*2**(-p-1),1,2)
    rows=[]
    for p in [0.,1.,1.4,1.5,2.,3.,5.,10.]:
        f=lambda rho:pentagon_moment(rho,p)
        opt=minimize_scalar(f,bounds=(.5,1/np.sqrt(3)),method='bounded')
        tv=2*3**(-(p+1)/2)/((p+1)*(p+2))
        rows.append(dict(p=p,rho=opt.x,pentagon=opt.fun,triangle=tv,
                         ratio=opt.fun/tv))
    orientations={str(p):{str(phi):moment_cap('curved',BASE['curved'],p,
                        np.arange(3)*2*np.pi/3+phi)
                        for phi in [0.,np.pi/6]} for p in [1,2,3,4,5]}
    # The intermediate symmetric caps can lose the transverse width.
    r=.499;a=opening('curved',r)
    axes=np.array([-a,a,np.pi])
    assert 2*a<np.pi/2 and (1-r)*np.sin(2*a)<r
    cap_test={str(p):{'cap':moment_cap('curved',r,p,axes),
                      'reference':moment_cap('curved',BASE['curved'],p)}
              for p in [10.,100.,1000.]}
    assert cap_test['1000.0']['cap']<cap_test['1000.0']['reference']
    cert=interval_certificate()
    result={'status':'numerical experiments, except the separate scalar interval certificate',
            'critical_exponent':pc,'pentagon_search':rows,
            'reference_orientation_moments':orientations,
            'cap_width_loss':{'r':r,'angle':a,'moments':cap_test},
            'reference_absolute_moment':reference}
    (ROOT/'coordinate_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


def draw():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11, 'axes.titlesize':11,
                         'axes.labelsize':11, 'xtick.labelsize':10,
                         'ytick.labelsize':10})
    fig,ax=plt.subplots(1,2,figsize=(8,4.1),constrained_layout=True)
    t=np.linspace(0,.5,500)
    for n,ls in [(3,'-'),(5,'--'),(7,'-.')]:
        ax[0].plot(t,chord_reuleaux(t,n),label=f'Reuleaux {n}-gon',ls=ls)
    ax[0].plot(t,2*np.sqrt(np.maximum(0,.25-t*t)),label='Disk',ls=':')
    ax[0].set(xlabel=r'$|y|$',ylabel='Horizontal chord length',
              title='Constant width\nReflection in the x-axis')
    ps=np.linspace(0,10,201)
    triangle=2*3**(-(ps+1)/2)/((ps+1)*(ps+2))
    disk=2*.5**(ps+2)*beta((ps+1)/2,1.5)
    ax[1].plot(ps,disk/triangle,label='Disk')
    ax[1].plot(ps,[pentagon_moment(9/16,p)/v for p,v in zip(ps,triangle)],
               label=r'Pentagon ($\rho=9/16$)',ls='--')
    pc=brentq(lambda p:p-2+3*2**(-p-1),1,2)
    ax[1].axvline(pc,color='.5',ls=':',label=r'$p_c\simeq1.452$')
    ax[1].axhline(1,color='.4',lw=.8)
    ax[1].set(xlabel='Power p',ylabel=r'$\int |y|^p$ relative to triangle',
              title='Minimal width\nReflection in the x-axis')
    for a in ax:
        a.legend(fontsize=9)
        a.grid(alpha=.2)
    ax[1].legend(fontsize=9,loc='upper right')
    fig.savefig(ROOT/'coordinate_profiles.pdf')
    fig.savefig(ROOT/'coordinate_profiles.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    run()
    draw()
