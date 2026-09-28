"""Registered conservative two-layer inference, bank-level measurement."""
import numpy as np
from scipy.stats import beta,t

def budgets(r):
 assert isinstance(r,int) and r>=1
 return .05/2**r/2,.05/2**r/2

def measurement(values,k,r=1):
 v=np.asarray(values);assert v.ndim==1 and len(v)>1
 a,_=budgets(r);m=float(v.mean());h=t.isf(a/(k*110*2),len(v)-1)*v.std(ddof=1)/np.sqrt(len(v));return float(m-h),float(m+h)

def classify(intervals):
 assert len(intervals)==110
 return 'S' if all(lo>.85 for lo,hi in intervals) else 'F' if any(hi<.85 for lo,hi in intervals) else 'U'

def rate(states,r=1):
 assert states and set(states)<=set('SFU');k=len(states);s=states.count('S');u=states.count('U');_,a=budgets(r)
 lo=0. if not s else float(beta.ppf(a/2,s,k-s+1));hi=1. if s+u==k else float(beta.ppf(1-a/2,s+u+1,k-s-u))
 return dict(K=k,S=s,F=states.count('F'),U=u,lower=lo,upper=hi,decision='supported' if lo>.8 else 'excluded' if hi<.8 else 'inconclusive',certified_fraction=s/k,possible_fraction=(s+u)/k)
