"""Independently crosscheck B-C intervals without reusing analysis arrays."""
import json,math,time
from pathlib import Path
from scipy.stats import t,sem
from branch_bc_round1 import ROOT
from branch_b_round1 import save
r=json.loads((ROOT/'summary.json').read_text())
raw={seed:json.loads((ROOT/f'seed{seed}/results.json').read_text())['cells'] for seed in range(301,306)}
checked=[]
for name,v in r['primary'].items():
 kind,delay,condition=name.split('_');d=int(delay[1:]);banks=[]
 for b in range(20):
  models=[]
  for seed,cells in raw.items():
   c=next(x for x in cells if x['bank']==b and x['delay']==d)
   def score(head):
    a=c['scores'][c['head_order'].index(head)]
    return a[0] if condition=='clean' else math.fsum(a[1:])/10
   models.append(score('expanded_aug')-(score('base_aug') if kind=='gain' else 0))
  banks.append(math.fsum(models)/5)
 mean=math.fsum(banks)/20;lo,hi=t.interval(1-.05/28,19,loc=mean,scale=sem(banks))
 for key,value in [('mean',mean),('lower',lo),('upper',hi)]:assert math.isclose(v[key],value,abs_tol=1e-14),(name,key,v[key],value)
 threshold=.02 if kind=='gain' else .85;assert v['threshold']==threshold
 state='supported' if lo>threshold else 'excluded' if hi<threshold else 'unresolved'
 assert name in r['primary_classification'][state]
 checked.append({'name':name,'state':state,'mean':mean,'lower':float(lo),'upper':float(hi)})
assert len(checked)==28
save(ROOT/'summary_crosscheck.json',{'passed':True,'checked_at':time.time(),'endpoints':checked,'method':'Loop over raw result cells; head names independently located; scipy t.interval/sem; no saved analysis arrays'})
print('All 28 primary endpoints independently crosschecked.')
