"""Data, chronology, frozen code and prior science integrity for five refits."""
from pathlib import Path
import json,time
import numpy as np
from scipy.ndimage import label
from branch_b_round1 import sha,digest
ROOT=Path('artifacts/branch_bz')
def historical():
 counts={}
 for b in ['b','bn','bt','br','bs','bl','bc','bw']:
  p=Path(f'artifacts/branch_{b}/reproducibility_manifest.json');m=json.loads(p.read_text())
  for name,v in m['files'].items():
   if Path(name).name=='NEXT_EXPERIMENT_PLAN.md':continue
   assert sha(name)==(v['sha256'] if isinstance(v,dict) else v),name
  counts[b]={'manifest_sha256':sha(p),'entries':len(m['files'])}
 return counts

def main():
 start=json.loads((ROOT/'execution_manifest.json').read_text())
 assert all(sha(p)==v for p,v in start['code'].items())
 hist=historical();assert hist==start['history']
 prior=json.loads((ROOT/'repeat1/data_manifest.json').read_text())['prior_files'];seen=set()
 for p,v in prior.items():
  assert sha(p)==v
  with np.load(p) as d:
   for key in ['images'] if 'images' in d else ['solid','ring']:
    seen.update(digest(x) for x in d[key].reshape(-1,1,32,32))
 assert len(seen)==224000
 locks=[json.loads((ROOT/f'repeat{r}/training_lock.json').read_text()) for r in range(1,6)]
 latest=max(x['locked_at'] for x in locks);new=0
 for r in range(1,6):
  folder=ROOT/f'repeat{r}';dm=json.loads((folder/'data_manifest.json').read_text());assert len(dm['banks'])==6
  lock=locks[r-1];assert lock['new_heads']==10 and lock['candidates']==50
  for p,v in lock['files'].items():assert sha(p)==v
  for name,m in dm['banks'].items():
   p=folder/f'{name}.npz';assert sha(p)==m['sha256'];d=dict(np.load(p));images=d['images'];k=images.shape[1];n=len(d['y'])
   hs=[digest(x) for x in images.reshape(-1,1,32,32)];assert len(set(hs))==len(hs) and not set(hs)&seen;seen.update(hs);new+=len(hs)
   np.testing.assert_array_equal([label(x[0]>.499,structure=np.ones((3,3)))[1] for x in images.reshape(-1,1,32,32)],np.repeat(np.arange(2,7),k))
   np.testing.assert_array_equal(d['y'],d['a']==d['b']);assert np.bincount(d['y']).tolist()==[n//2]*2
   for a in range(5):
    for b in range(5):assert ((d['a']==a)&(d['b']==b)).sum()==n//40*(4 if a==b else 1)
   assert not np.any((d['a']==d['b'])&(d['ia']==d['ib']))
   for v in ['ia','ib']:assert ((d[v]>=0)&(d[v]<k)).all()
   if name.startswith('test'):
    assert int(d['seed'])==980000+r*1000+100+int(name[4:]) and k==200 and n==2000 and p.stat().st_mtime>=latest
   else:
    assert int(d['seed'])==980000+r*1000+{'fit':1,'validation':2}[name] and k==1000 and n=={'fit':8000,'validation':2000}[name]
    assert p.stat().st_mtime<lock['locked_at']
  for seed in range(301,306):
   fit=json.loads((folder/f'seed{seed}/fits.json').read_text());ev=json.loads((folder/f'seed{seed}/evaluation_manifest.json').read_text())
   assert fit['finished_at']<lock['locked_at']<=latest<ev['started_at']
 assert new==70000
 result=dict(passed=True,checked_at=time.time(),new_images=new,prior_images=224000,all_fits_locked_before_any_test=True,original_code_unchanged=True,historical_science_unchanged=hist,planning_pointer_sha256=sha('NEXT_EXPERIMENT_PLAN.md'))
 (ROOT/'integrity.json').write_text(json.dumps(result,indent=2)+'\n');print('Integrity passed: 70000 new images, all 50 heads locked before tests')
if __name__=='__main__':main()
