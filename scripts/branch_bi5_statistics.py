"""Outer bounds of the same latent CP event, with simultaneous measurement."""
import numpy as np
from scipy.stats import beta,t

def reused_rate(states,alpha=.003125):
 assert len(states)==96 and set(states)<=set('SFU');s=states.count('S');f=states.count('F');u=states.count('U');lo=0. if s==0 else float(beta.ppf(alpha/2,s,97-s));hi=1. if f==0 else float(beta.ppf(1-alpha/2,97-f,f));return dict(K=96,S=s,F=f,U=u,lower=lo,upper=hi,rate_event_source_round=3,rate_alpha_reused=alpha,decision='excluded' if hi<.8 else 'inconclusive')
def family_upper(v,alpha):
 v=np.asarray(v);m=float(v.mean());sd=float(v.std(ddof=1));return dict(mean=m,sd=sd,upper=float(m+t.isf(alpha,len(v)-1)*sd/len(v)**.5),alpha=alpha,n_banks=len(v))
def union_states(old,failed_heads):
 out=dict(old)
 for head in set(failed_heads):
  b=head[1:];assert out[b] in ['F','U'];out[b]='F'
 assert len(out)==96;return out
