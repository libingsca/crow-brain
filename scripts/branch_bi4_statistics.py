"""Fixed sequence gatekeeping over fixed looks, no Bonferroni across the chain."""
import numpy as np
from scipy.stats import t

def prefix_update(previous,uppers):
 assert 0<=previous<=len(uppers);k=previous
 while k<len(uppers) and uppers[k]<.85:k+=1
 return k

def upper_ci(v,alpha):
 v=np.asarray(v);assert v.ndim==1 and len(v)>1 and 0<alpha<1;m=float(v.mean());sd=float(v.std(ddof=1));return dict(mean=m,sd=sd,upper=float(m+t.isf(alpha,len(v)-1)*sd/len(v)**.5),alpha=alpha,n_banks=len(v),kind='one-sided; certification only through fixed-sequence gate')
