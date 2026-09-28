"""Reaggregate every primary result and factorial target from audited bank records."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
root=Path('artifacts/branch_bf/round1')
r=json.loads((root/'summary.json').read_text()); heads=['base_mix','base_split','cover_mix','cover_split']; checked=0
for seed in range(301,306):
 rows=json.loads((root/f'seed{seed}/results.json').read_text())['cells']
 for delay in range(5,16):
  cells=sorted([x for x in rows if x['delay']==delay],key=lambda x:x['bank'])
  assert [x['bank'] for x in cells]==list(range(20))
  a=np.asarray([x['scores'] for x in cells]);assert all(x['head_order'][:4]==heads for x in cells)
  for condition in ['clean','noise']:
   v=a[:,3,0] if condition=='clean' else a[:,3,1:].sum(axis=1)/10
   mu=sum(v)/20;sd=np.sqrt(sum((v-mu)**2)/19);half=t.isf(.05/220,19)*sd/np.sqrt(20)
   expected=r['primary'][f'{seed}_d{delay}_{condition}']
   np.testing.assert_allclose([mu,mu-half,mu+half],[expected[k] for k in ['mean','lower','upper']],atol=1e-12,rtol=0);checked+=1
  key=f'{seed}_d{delay}_clean'
  if key in r['targets']:
   means=a[:,:4,0].mean(axis=0);target=r['targets'][key]
   np.testing.assert_allclose(means,[target['means'][h] for h in heads],atol=1e-12)
   effects=[means[2]-means[0],means[3]-means[1],means[1]-means[0],means[3]-means[2],means[3]-means[1]-means[2]+means[0]]
   np.testing.assert_allclose(effects,[target[k] for k in ['coverage_mix','coverage_split','split_base','split_cover','interaction']],atol=1e-12)
assert checked==110
for state in ['supported','excluded','unresolved']:
 expected=[k for k,v in r['primary'].items() if (v['lower']>.85 if state=='supported' else v['upper']<.85 if state=='excluded' else v['lower']<=.85<=v['upper'])]
 assert expected==r[state]
(root/'summary_crosscheck.json').write_text(json.dumps({'passed':True,'primary_intervals':checked,'factorial_targets':4,'reaggregated_from':'audited results.json bank records'},indent=2)+'\n')
print('Primary 110 intervals, classifications and four factorial targets independently reaggregated.')
