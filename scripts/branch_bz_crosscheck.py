"""Independent scalar aggregation from saved cells, without analysis arrays."""
from pathlib import Path
import json,math
import numpy as np
from scipy.stats import t,sem
root=Path('artifacts/branch_bz');summary=json.loads((root/'summary.json').read_text());raw={}
for r in range(1,6):
 for seed in range(301,306):
  rows=json.loads((root/f'repeat{r}/seed{seed}/results.json').read_text())['cells']
  for d in range(5,16):
   for c in range(2):
    v=[]
    for row in rows:
     if row['delay']==d:
      h=row['head_order'].index('worst_aug');values=row['scores'][h]
      v.append(values[0] if c==0 else math.fsum(values[1:])/10)
    assert len(v)==4;raw[r,seed,d,c]=math.fsum(v)/4
checked=0
for family in ['panel','individual']:
 for name,expected in summary[family].items():
  if family=='panel':
   ds,cs=name.split('_');d=int(ds[1:]);c=['clean','noise'].index(cs)
   values=[math.fsum(raw[r,s,d,c] for s in range(301,306))/5 for r in range(1,6)]
  else:
   values=[min(raw[r,int(name),d,c] for d in range(5,16) for c in range(2)) for r in range(1,6)]
  mean=math.fsum(values)/5;se=sem(values);lo,hi=t.interval(1-.05/27,4,loc=mean,scale=se) if se else (mean,mean)
  np.testing.assert_allclose(values,expected['values'],atol=1e-14)
  np.testing.assert_allclose([mean,lo,hi],[expected[x] for x in ['mean','lower','upper']],atol=1e-14);checked+=1
assert checked==27
(root/'crosscheck.json').write_text(json.dumps(dict(passed=True,endpoints=checked,method='Raw cell scalar loops; independent scipy interval'),indent=2)+'\n')
print('27 endpoints independently verified')
