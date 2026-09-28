import json
from pathlib import Path
import numpy as np
from scipy.stats import t,beta
from branch_bi2_round import ROOT,CFG,HEADS,DELAYS,REFERENCE_DELAYS,N_BANKS
from branch_bi_statistics import classify,rate
from branch_b_round1 import save

def ci(v,alpha):
 v=np.asarray(v);assert v.shape==(N_BANKS,);m=float(v.mean());h=float(t.isf(alpha/2,N_BANKS-1)*v.std(ddof=1)/np.sqrt(N_BANKS));return dict(mean=m,lower=m-h,upper=m+h,n_banks=N_BANKS,alpha=alpha)
def main():
 assert json.loads((ROOT/'independent_audit.json').read_text())['passed'];rows=json.loads((ROOT/'seed301/results.json').read_text())['cells'];assert len(rows)==N_BANKS*len(DELAYS)
 vals=np.full((6,N_BANKS,len(DELAYS),2),np.nan);neg=np.full((6,N_BANKS,len(DELAYS),2,3),np.nan);ref=np.full((N_BANKS,2,2),np.nan);rn=np.full((N_BANKS,2,2,3),np.nan);orig=np.full((N_BANKS,2),np.nan)
 for x in rows:
  bi=x['bank'];di=DELAYS.index(x['delay']);a=np.array(x['scores']);vals[:,bi,di,0]=a[:6,0];vals[:,bi,di,1]=a[:6,1:].mean(1);neg[:,bi,di]=x['negative_means'][:6]
  if x['delay'] in REFERENCE_DELAYS:
   j=REFERENCE_DELAYS.index(x['delay']);ref[bi,j]=[a[6,0],a[6,1:].mean()];rn[bi,j]=x['negative_means'][6]
  if x['delay']==5:orig[bi]=[x['original_scores'][0],np.mean(x['original_scores'][1:])]
 assert all(np.isfinite(a).all() for a in [vals,neg,ref,rn,orig]);q=sum(len(x) for x in CFG['targets'].values());assert q==21;new={};combined={};states=[];perbatch={}
 for b in range(1,7):
  prior=json.loads(Path(f'artifacts/branch_bi/round1/batch{b}/summary.json').read_text());primary=dict(prior['primary']);new[str(b)]={}
  for target in CFG['targets'][str(b)]:
   _,ds,c=target.split('_');v=vals[b-1,:,DELAYS.index(int(ds[1:])),['clean','noise'].index(c)];new[str(b)][target]=ci(v,.00625/q);primary[target]=new[str(b)][target]
   # Independent scalar-loop moment and quantile implementation.
   m=sum(float(x) for x in v)/N_BANKS;sd=(sum((float(x)-m)**2 for x in v)/(N_BANKS-1))**.5;h=t.ppf(1-.00625/q/2,N_BANKS-1)*sd/N_BANKS**.5;np.testing.assert_allclose([m-h,m+h],[primary[target]['lower'],primary[target]['upper']],atol=1e-12)
  combined[str(b)]=primary;st=classify([(x['lower'],x['upper']) for x in primary.values()]);states.append(st);perbatch[str(b)]=dict(state=st,supported=[k for k,v in primary.items() if v['lower']>.85],excluded=[k for k,v in primary.items() if v['upper']<.85],unresolved=[k for k,v in primary.items() if v['lower']<=.85<=v['upper']])
 pos={f'reference{d}_{c}':ci(ref[:,j,k],.01) for j,d in enumerate(REFERENCE_DELAYS) for k,c in enumerate(['clean','noise'])};pos.update({f'original5_{c}':ci(orig[:,k],.01) for k,c in enumerate(['clean','noise'])});ng=np.r_[neg.mean(1).ravel(),rn.mean(0).ravel()];quality=all(x['lower']>.9 for x in pos.values()) and bool(((ng>=.47)&(ng<=.53)).all());assert quality
 result=rate(states,r=2);s=result['S'];u=result['U'];np.testing.assert_allclose(result['lower'],0 if s==0 else beta.isf(1-.003125,s,7-s),atol=1e-12);np.testing.assert_allclose(result['upper'],1 if s+u==6 else beta.isf(.003125,s+u+1,6-s-u),atol=1e-12)
 save(ROOT/'round_summary.json',dict(**result,quality_passed=quality,new_intervals=new,combined_intervals=combined,batches=perbatch,positive_controls=pos,negative_range=[float(ng.min()),float(ng.max())],aggregation_crosschecked=True,cumulative_round_budget=.0375,new_fits=0));np.savez_compressed(ROOT/'analysis_arrays.npz',scores=vals,negative=neg,references=ref,reference_negative=rn,original=orig)
 print(json.dumps({**result,'states':states,'unresolved':{b:x['unresolved'] for b,x in perbatch.items()}},indent=2),flush=True)
if __name__=='__main__':main()
