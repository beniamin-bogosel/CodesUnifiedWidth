"""Summaries and plots for the separate heat/spectrum investigation."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parent

def load(name):return json.loads((ROOT/name).read_text())

def main():
    refs=load('refinements.json');scan=load('scan_N55_m24.json')
    heat=load('heat_checks.json');href=load('heat_refinements.json')
    survey=load('lambda2_survey.json')
    table=[]
    for kind in ['straight','curved']:
        for n,rows in [(55,scan),(100,refs),(160,refs)]:
            pair=[r for r in rows if r['kind']==kind and r['N']==n and r['epsilon'] in [0.,.15]]
            eq=next(r for r in pair if r['epsilon']==0.)
            un=next(r for r in pair if r['epsilon']==.15)
            table.append({'kind':kind,'N':n,'lambda2_equal':eq['eigenvalues'][1],
                          'lambda3_equal':eq['eigenvalues'][2],
                          'lambda2_unequal':un['eigenvalues'][1],
                          'lambda3_unequal':un['eigenvalues'][2],
                          'delta_lambda3':un['eigenvalues'][2]-eq['eigenvalues'][2]})
    htable=[]
    for kind in ['straight','curved']:
        for N,rows in [(100,heat),(160,href)]:
            r=.4 if kind=='straight' else .45
            eq=next(x for x in rows if x['kind']==kind and x['r']==r and x['epsilon']==0.)
            un=next(x for x in rows if x['kind']==kind and x['r']==r and x['epsilon']==.15)
            for t in ['0.01','0.02','0.05','0.1']:
                diff=un['heat'][t]['partial_sum']-eq['heat'][t]['partial_sum']
                tails=un['heat'][t]['spectral_tail_estimate']+eq['heat'][t]['spectral_tail_estimate']
                htable.append(dict(kind=kind,N=N,t=float(t),difference=diff,sum_tail_estimates=tails))
    result={'status':'Floating-point P2 evidence; tail estimates assume exact spectral data and do not certify FEM errors',
            'spectrum_refinement':table,'heat_refinement':htable,
            'survey_configurations':sum(len(r['trials']) for r in survey),
            'survey_max_delta_lambda2':max(t['delta_lambda2'] for r in survey for t in r['trials']),
            'disk_cutoffs':{'straight':float(np.sqrt(1/(np.pi*np.sqrt(3)))),
                            'curved':float(np.sqrt((np.pi-np.sqrt(3))/(2*np.pi)))}}
    assert min(x['delta_lambda3'] for x in table)>.49
    assert result['survey_max_delta_lambda2']<0
    assert min(x['difference']-x['sum_tail_estimates'] for x in htable)>0
    hsurvey=[]
    for entry in survey:
        kind,r=entry['kind'],entry['r']
        eq=load(f'data/survey_{kind}_{r}_equal_N45_m8.json')
        for j in range(len(entry['trials'])):
            un=load(f'data/survey_{kind}_{r}_{j}_N45_m8.json')
            for t in ['0.05','0.1','0.2']:
                diff=un['heat'][t]['partial_sum']-eq['heat'][t]['partial_sum']
                tail=un['heat'][t]['spectral_tail_estimate']+eq['heat'][t]['spectral_tail_estimate']
                hsurvey.append(dict(kind=kind,r=r,j=j,t=t,difference=diff,truncation_estimate=tail,
                                    resolved_sign=1 if diff>tail else (-1 if diff< -tail else 0)))
    assert not any(x['resolved_sign']<0 for x in hsurvey)
    (ROOT/'heat_survey.json').write_text(json.dumps(hsurvey,indent=2)+'\n')
    result['heat_survey_sign_counts']={str(k):sum(x['resolved_sign']==k for x in hsurvey) for k in [-1,0,1]}
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11, 'axes.titlesize':11,
                         'axes.labelsize':11, 'xtick.labelsize':10,
                         'ytick.labelsize':10})
    fig,axs=plt.subplots(1,2,figsize=(8,3.8),constrained_layout=True)
    for kind,marker in [('straight','o'),('curved','s')]:
        color='#235d88' if kind=='straight' else '#a55c0b'
        rows=[x for x in scan if x['kind']==kind]
        rows.sort(key=lambda x:x['epsilon'])
        b=rows[0]['eigenvalues'][1]
        for k,ls in [(1,'--'),(2,'-')]:
            axs[0].plot([x['epsilon'] for x in rows],
                        [x['eigenvalues'][k]-b for x in rows],ls=ls,marker=marker,color=color,
                        label=rf'{kind}, $\lambda_{k+1}$')
        eq=next(x for x in href if x['kind']==kind and x['epsilon']==0.)
        un=next(x for x in href if x['kind']==kind and x['epsilon']==.15)
        times=np.geomspace(.01,.2,100)
        hv=lambda x:np.exp(-times[:,None]*np.array(x['eigenvalues']))@np.array(x['heat_coefficients'])
        axs[1].semilogx(times,(hv(un)/hv(eq)-1),label=kind,color=color,
                       ls='-' if kind=='straight' else '--')
    axs[0].axhline(0,color='.5',lw=.8)
    axs[0].set(xlabel=r'Angular perturbation $\varepsilon$',ylabel='Eigenvalue change',
               title='First excited eigenvalue pair')
    axs[1].set(xlabel=r'Time $t$',ylabel=r'$\mathcal{H}_{S_{r,\varepsilon}}(t)/\mathcal{H}_{E_r}(t)-1$',
               title=r'Dirichlet heat content, $\varepsilon=0.15$')
    axs[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
    for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=9.5)
    fig.savefig(ROOT/'heat_spectrum_profiles.pdf')
    fig.savefig(ROOT/'heat_spectrum_profiles.png',dpi=170)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
