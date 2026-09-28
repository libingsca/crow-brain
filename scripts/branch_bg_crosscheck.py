import json
import numpy as np
from scipy.stats import t
from branch_bg_round1 import ROOT
from branch_b_round1 import save
r=json.loads((ROOT/'summary.json').read_text());checked=0;primary=0
for seed in range(301,306):
 cells=json.loads((ROOT/f'seed{seed}/results.json').read_text())['cells']
 for d in range(5,16):
  rows=sorted([x for x in cells if x['delay']==d],key=lambda x:x['bank']);assert [x['bank'] for x in rows]==list(range(20));a=np.array([x['scores'] for x in rows])
  for ci,c in enumerate(['clean','noise']):
   v=a[:,3,0] if ci==0 else a[:,3,1:].sum(1)/10
   comparisons=[(v,r['secondary_full'][f'{seed}_d{d}_{c}'],110)]
   if seed in [301,302] and d==15:
    base=a[:,0,0] if ci==0 else a[:,0,1:].sum(1)/10
    comparisons.extend([(v,r['primary'][f'{seed}_d15_{c}_absolute'],8),(v-base,r['primary'][f'{seed}_d15_{c}_gain'],8)]);primary+=2
   for x,expected,family in comparisons:
    m=sum(x)/20;half=t.ppf(1-.05/(2*family),19)*np.sqrt(sum((x-m)**2)/19)/np.sqrt(20)
    np.testing.assert_allclose([m,m-half,m+half],[expected[k] for k in ['mean','lower','upper']],atol=1e-12,rtol=0);checked+=1
assert checked==118 and primary==8
for state in ['supported','excluded','unresolved']:
 keys=[k for k,v in r['primary'].items() if (v['lower']>v['threshold'] if state=='supported' else v['upper']<v['threshold'] if state=='excluded' else v['lower']<=v['threshold']<=v['upper'])];assert keys==r[state]
save(ROOT/'summary_crosscheck.json',{'passed':True,'independently_reaggregated_intervals':checked,'primary':primary,'from':'audited per-bank records'})
print('118 intervals and primary classifications independently reaggregated.')
