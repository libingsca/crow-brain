"""Pre-specified seed-level paired intervals and independent implementation checks."""
import numpy as np
from scipy import stats

def interval(x,alpha):
    x=np.asarray(x,float); n=len(x); mean=float(x.mean()); sd=float(x.std(ddof=1))
    radius=float(stats.t.ppf(1-alpha/2,n-1)*sd/np.sqrt(n))
    return {'n':n,'mean':mean,'sd':sd,'alpha':alpha,'lower':mean-radius,'upper':mean+radius,'approximation':'paired seed-level t coverage'}

def power_n(sd,alpha,delta=.03,power=.8,min_n=20):
    # Noncentral t: target true effect 2delta, null delta; use two-sided interval boundary.
    for n in range(min_n,10001):
        crit=stats.t.ppf(1-alpha/2,n-1)
        p=stats.nct.sf(crit,n-1,delta*np.sqrt(n)/max(sd,1e-12))
        if p>=power: return n,float(p)
    return None,None

def check(alpha=.05/12/2/4):
    rng=np.random.default_rng(900991);x=rng.normal(size=(100000,20));se=x.std(1,ddof=1)/np.sqrt(20)
    t=x.mean(1)/se;crit=stats.t.ppf(1-alpha/2,19)
    fails=int(np.sum(np.abs(t)>crit));ci=stats.binomtest(fails,len(t)).proportion_ci(.99,method='exact')
    v=rng.normal(.02,.1,30);ours=interval(v,alpha)
    ref=stats.t.interval(1-alpha,29,loc=v.mean(),scale=stats.sem(v))
    assert np.allclose([ours['lower'],ours['upper']],ref)
    return {'alpha':alpha,'normal_null_simulations':len(t),'misses':fails,'miss_rate':fails/len(t),'miss_rate_99ci':[float(ci.low),float(ci.high)],'nominal_within_simulation_ci':bool(ci.low<=alpha<=ci.high),'independent_scipy_crosscheck':True,'limitation':'validates implementation under normal iid seed effects, not actual seed normality'}
if __name__=='__main__':
    import h01
    result=check();h01.dump(h01.BASE/'statistics_implementation_audit.json',result);print(result)
