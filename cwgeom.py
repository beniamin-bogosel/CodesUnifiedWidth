"""Euclidean constant-width geometry. Angles in radians; coordinates in width units.

The disk-polygon evaluator keeps analytic circle arcs. Boundary samples are for
rendering; support values are evaluated on the exact arcs (up to floating point).
"""
from dataclasses import dataclass
import math
import numpy as np

TAU = 2 * math.pi

def unit(theta):
    theta = np.asarray(theta)
    return np.stack((np.cos(theta), np.sin(theta)), axis=-1)

def regular_reuleaux(n=3, width=1.):
    if not isinstance(n, int) or n < 3 or n % 2 != 1 or not np.isfinite(width) or width <= 0:
        raise ValueError('n must be odd and >=3; width must be positive')
    return width / (2 * np.cos(np.pi / (2 * n))) * unit(np.pi / 2 + np.arange(n) * TAU / n)

def circle_intersections(a, b, radius):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.linalg.norm(b-a)
    if d < 1e-13 or d > 2*radius + 1e-12:
        return []
    mid = (a+b)/2
    perpendicular = np.array([-(b-a)[1], (b-a)[0]])/d
    offset = math.sqrt(max(0., radius*radius-d*d/4))
    return [mid+offset*perpendicular, mid-offset*perpendicular]

def irregular_reuleaux(n=5, perturb=None, width=1.):
    """Close an odd equilateral star; reject perturbations outside its valid chart.

    There are n-3 free direction perturbations. This is a local construction
    near a regular Reuleaux polygon, not a parameterization of the whole class.
    """
    if not isinstance(n, int) or n < 5 or n % 2 != 1:
        raise ValueError('irregular construction requires odd n>=5')
    if perturb is None:
        perturb=[.04,-.02]+[0.]*(n-5)
    if len(perturb) != n-3 or not np.all(np.isfinite(perturb)) or not np.isfinite(width) or width <= 0:
        raise ValueError('n-3 finite perturbations and positive width required')
    angles = np.arange(n)*np.pi*(n-1)/n
    q = [np.zeros(2), unit(0.)]
    for i, delta in enumerate(perturb, start=1):
        q.append(q[-1]+unit(angles[i]+delta))
    choices = circle_intersections(q[-1], q[0], 1.)
    if len(choices) != 2:
        raise ValueError('perturbation cannot close a nondegenerate star')
    reference = np.sum(unit(angles[:n-1]), axis=0)
    q.append(min(choices, key=lambda x:np.linalg.norm(x-reference)))
    pts = np.array(q); pts -= pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:,1],pts[:,0]))]*width
    validate_reuleaux(pts, width)
    return pts

def irregular_reuleaux_pentagon(perturb=(.08, -.04), width=1.):
    return irregular_reuleaux(5, perturb, width)

@dataclass(frozen=True)
class Arc:
    center: np.ndarray
    radius: float
    start: float
    sweep: float
    center_index: int
    def points(self, count=80):
        return self.center+self.radius*unit(np.linspace(self.start,self.start+self.sweep,count))

def disk_arcs(centers, radius=1., tolerance=2e-10):
    """Boundary of an intersection of equal disks, including redundant disks.

    Returns CCW circular arcs; raises for empty/point-only intersections.
    """
    centers=np.asarray(centers,float)
    if centers.ndim != 2 or centers.shape[1] != 2 or not len(centers) or not np.all(np.isfinite(centers)) or not np.isfinite(radius) or radius <= 0:
        raise ValueError('finite planar centers and positive radius required')
    # Identical disks do not create duplicate full-circle arcs.
    centers=np.unique(centers,axis=0)
    arcs=[]
    for i,c in enumerate(centers):
        angles=[]
        for j,d in enumerate(centers):
            if i==j: continue
            for p in circle_intersections(c,d,radius):
                if np.all(np.linalg.norm(centers-p,axis=1)<=radius+tolerance):
                    angles.append(float(np.arctan2(*(p-c)[::-1]) % TAU))
        angles=sorted(angles)
        unique=[]
        for a in angles:
            if not unique or a-unique[-1]>1e-9: unique.append(a)
        if len(unique)>1 and unique[0]+TAU-unique[-1]<1e-9: unique.pop()
        if not unique:
            if len(centers)==1: arcs.append(Arc(c,radius,0.,TAU,i))
            continue
        for j,a in enumerate(unique):
            b=unique[(j+1)%len(unique)]
            sweep=(b-a)%TAU
            if sweep<1e-10: continue
            midpoint=c+radius*unit(a+sweep/2)
            if np.all(np.linalg.norm(centers-midpoint,axis=1)<=radius+tolerance):
                arcs.append(Arc(c,radius,a,sweep,i))
    if not arcs: raise ValueError('disk intersection is empty or has no positive-area interior')
    # Any mean of boundary points lies in the convex region and orders its arcs.
    interior=np.mean([a.points(3)[1] for a in arcs],axis=0)
    arcs.sort(key=lambda a:float(np.arctan2(*(a.points(3)[1]-interior)[::-1])))
    return arcs

def boundary(arcs, count=100):
    return np.vstack([a.points(max(3,int(count*a.sweep/TAU)+2))[:-1] for a in arcs])

def arc_support(arcs, theta):
    theta=np.asarray(theta,float)
    normals=unit(theta); values=[]
    for a in arcs:
        endpoints=a.points(2)
        endpoint_max=np.maximum(normals@endpoints[0],normals@endpoints[1])
        inside=((theta-a.start)%TAU <= a.sweep+1e-12)
        values.append(np.where(inside,normals@a.center+a.radius,endpoint_max))
    return np.max(values,axis=0)

def arc_area(arcs):
    """Green integral on exact circular arcs, not a sampled polygon area."""
    total=0.
    for a in arcs:
        ends=a.points(2); diff=ends[1]-ends[0]
        total += a.center[0]*diff[1]-a.center[1]*diff[0]+a.radius*a.radius*a.sweep
    return total/2

def validate_reuleaux(vertices, width=1., tolerance=1e-8):
    """Verify cyclic diametric edges, convex order, active arcs, and sampled widths.

    Returns a floating-point diagnostic, not an interval certificate.
    """
    v=np.asarray(vertices,float); n=len(v)
    if n<3 or n%2!=1 or width<=0: raise ValueError('odd vertex count >=3 and positive width required')
    distances=np.linalg.norm(v[:,None,:]-v[None,:,:],axis=2)
    if np.max(distances)>width+tolerance: raise ValueError('vertex diameter exceeds width')
    k=(n-1)//2
    if np.max(np.abs(distances[np.arange(n),(np.arange(n)+k)%n]-width))>tolerance:
        raise ValueError('opposite cyclic vertex distances must equal width')
    edges=np.roll(v,-1,axis=0)-v
    cross=edges[:,0]*np.roll(edges,-1,axis=0)[:,1]-edges[:,1]*np.roll(edges,-1,axis=0)[:,0]
    if np.any(cross<=tolerance): raise ValueError('vertices must be in strictly convex CCW order')
    arcs=disk_arcs(v,width)
    if len(arcs)!=n: raise ValueError('some centers do not contribute a nondegenerate arc')
    # Each circle must join exactly the two cyclic opposite vertices.
    for a in arcs:
        i=int(np.argmin(np.linalg.norm(v-a.center,axis=1)))
        expected=v[[(i+k)%n,(i+k+1)%n]]
        ends=a.points(2)
        if max(min(np.linalg.norm(e-x) for x in expected) for e in ends)>tolerance:
            raise ValueError('active arc has incorrect opposite endpoints')
    theta=np.linspace(0,TAU,1441,endpoint=False)
    error=float(np.max(np.abs(arc_support(arcs,theta)+arc_support(arcs,theta+np.pi)-width)))
    if error>tolerance: raise ValueError('directional width check failed')
    return dict(vertex_count=n,width=width,max_width_error=error,area=arc_area(arcs),perimeter=sum(a.radius*a.sweep for a in arcs),check='floating-point analytic arcs; sampled directions')

def support_curve(width, harmonics, samples=1200):
    """harmonics: [frequency, cosine coefficient, sine coefficient]."""
    if width<=0: raise ValueError('width must be positive')
    theta=np.linspace(0,TAU,samples,endpoint=False)
    h=np.full(samples,width/2); dh=np.zeros(samples); rho=h.copy()
    L=0.
    for k,a,b in harmonics:
        if int(k)!=k or k<1 or k%2!=1: raise ValueError('constant width permits positive odd harmonics only')
        z=a*np.cos(k*theta)+b*np.sin(k*theta)
        h+=z; dh+=k*(-a*np.sin(k*theta)+b*np.cos(k*theta));rho+=(1-k*k)*z
        L+=k*abs(1-k*k)*math.hypot(a,b)
    if rho.min()<-1e-10: raise ValueError('support candidate is nonconvex')
    # First try a global triangle bound, then a Lipschitz grid margin.
    analytic_margin=width/2-sum(abs(1-k*k)*math.hypot(a,b) for k,a,b in harmonics)
    grid_margin=float(rho.min()-L*np.pi/samples)
    if max(analytic_margin,grid_margin)<-1e-12:
        raise ValueError('convexity not certified between samples; increase samples or reduce coefficients')
    points=h[:,None]*unit(theta)+dh[:,None]*unit(theta+np.pi/2)
    return points,dict(min_sample_curvature=float(rho.min()),convexity_margin=max(analytic_margin,grid_margin),width=width)

def regular_inner_area(n,t,width=1.):
    delta=np.pi/(2*n); R=width/(2*np.cos(delta)); q=width-t
    if n<3 or n%2!=1 or t<0 or t>width-R+1e-12: raise ValueError('outside erosion range')
    alpha=np.arcsin(np.clip(width*np.sin(delta)/q,-1,1))-delta
    beta=2*delta-2*alpha
    rt=np.sqrt(max(0.,q*q-width*width*np.sin(delta)**2))-R*np.cos(np.pi/n)
    return float(n/2*rt*rt*np.sin(2*np.pi/n)+n*q*q/2*(beta-np.sin(beta)))

def spherical_triangle_area(a,b,c):
    a,b,c=(np.asarray(v,float) for v in (a,b,c))
    if any(abs(np.linalg.norm(v)-1)>1e-9 for v in (a,b,c)): raise ValueError('unit vectors required')
    return float(2*np.arctan2(np.linalg.det(np.stack((a,b,c))),1+a@b+b@c+c@a))
