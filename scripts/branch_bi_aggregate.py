import json
from pathlib import Path
import numpy as np
from scipy.stats import t,beta
from branch_bi_statistics import measurement,classify,rate,budgets
from branch_b_round1 import save
R=Path('artifacts/branch_bi/round1')
def main():
 states=[];out={}
 for b in range(1,7):
  root=R/f'batch{b}';summary=json.loads((root/'summary.json').read_text());assert summary['quality_passed'];bounds=[]
  values=np.empty((5,40,11,2))
  for i,seed in enumerate(range(301,306)):
   rows=json.loads((root/f'seed{seed}/results.json').read_text())['cells']
   for x in rows:values[i,x['bank'],x['delay']-5]=[x['scores'][0][0],sum(x['scores'][0][1:])/10]
  for i,seed in enumerate(range(301,306)):
   for d in range(5,16):
    for c,condition in enumerate(['clean','noise']):
     v=values[i,:,d-5,c];m=sum(v)/40;sd=(sum((x-m)**2 for x in v)/39)**.5;h=t.ppf(1-.0125/1320,39)*sd/40**.5;lo,hi=measurement(v,6);np.testing.assert_allclose([lo,hi],[m-h,m+h],atol=1e-12)
     old=summary['primary'][f'{seed}_d{d}_{condition}'];np.testing.assert_allclose([lo,hi],[old['lower'],old['upper']],atol=1e-12);bounds.append((lo,hi))
  state=classify(bounds);states.append(state);out[str(b)]=dict(state=state,excluded=summary['excluded'],unresolved=summary['unresolved'],min_mean=float(values.mean(1).min()),supported_conditions=len(summary['supported']))
 result=rate(states);s=result['S'];u=result['U'];np.testing.assert_allclose(result['lower'],0 if s==0 else beta.isf(1-.00625,s,7-s),atol=1e-12);np.testing.assert_allclose(result['upper'],1 if s+u==6 else beta.isf(.00625,s+u+1,6-s-u),atol=1e-12)
 save(R/'round_summary.json',dict(**result,batches=out,measurement_alpha=.0125,rate_alpha=.0125,aggregation_crosschecked=True,quality_passed=True));print(json.dumps(dict(**result,batches=out),indent=2),flush=True)
if __name__=='__main__':main()
