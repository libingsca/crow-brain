"""Independent data, chronology and historical artifact checks for B-W."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.ndimage import label
from branch_bw_round1 import ROOT
from branch_b_round1 import sha,digest,save

def main():
 dm=json.loads((ROOT/'data_manifest.json').read_text());seen=set();historical=0
 for name,expected in dm['prior_files'].items():
  assert sha(name)==expected
  with np.load(name) as a:
   keys=['images'] if 'images' in a else ['solid','ring']
   for k in keys:
    for im in a[k].reshape(-1,1,32,32):seen.add(digest(im));historical+=1
 assert historical==194000
 lock=json.loads((ROOT/'training_lock.json').read_text());assert lock['new_heads']==20 and lock['candidates']==100
 bankinfo=[];newcount=0
 expected_names={'fit','validation'}|{f'test{i:02d}' for i in range(20)};assert set(dm['banks'])==expected_names
 for name,meta in dm['banks'].items():
  p=ROOT/f'{name}.npz';assert sha(p)==meta['sha256'];a=dict(np.load(p));ims=a['images'];seed=int(a['seed']);n=len(a['y']);k=ims.shape[1]
  hs=[digest(im) for im in ims.reshape(-1,1,32,32)];assert len(set(hs))==len(hs) and not set(hs)&seen;seen.update(hs);newcount+=len(hs)
  counts=[label(im[0]>.499,structure=np.ones((3,3)))[1] for im in ims.reshape(-1,1,32,32)]
  np.testing.assert_array_equal(counts,np.repeat(np.arange(2,7),k))
  np.testing.assert_array_equal(a['y'],(a['a']==a['b']).astype(np.int64));assert np.bincount(a['y']).tolist()==[n//2]*2
  for c in ['a','b']:assert np.bincount(a[c]).tolist()==[n//5]*5
  for x in range(5):
   for y in range(5):assert ((a['a']==x)&(a['b']==y)).sum()==n//40*(4 if x==y else 1)
  assert not np.any((a['a']==a['b'])&(a['ia']==a['ib']))
  assert all(np.all((a[c]>=0)&(a[c]<k)) for c in ['ia','ib'])
  if name.startswith('test'):
   assert seed==970100+int(name[4:]) and n==2000 and k==200
   assert p.stat().st_mtime>=lock['locked_at']
  else:assert seed=={'fit':970001,'validation':970002}[name] and p.stat().st_mtime<lock['locked_at']
  bankinfo.append({'name':name,'sha256':meta['sha256'],'images':len(hs),'trials':n,'mtime':p.stat().st_mtime})
 assert newcount==30000
 for seed in range(301,306):
  f=json.loads((ROOT/f'seed{seed}/fits.json').read_text());e=json.loads((ROOT/f'seed{seed}/evaluation_manifest.json').read_text())
  assert f['finished_at']<lock['locked_at']<e['started_at']
 before=json.loads((ROOT/'historical_integrity_before.json').read_text());verified={};pointer_changes=[]
 mismatch_paths={m['path']:m['actual'] for m in before['preexisting_planning_pointer_mismatches']}
 for b,row in before['branches'].items():
  assert sha(row['manifest'])==row['sha256'];m=json.loads(Path(row['manifest']).read_text());n=0
  for name,v in m['files'].items():
   expected=v['sha256'] if isinstance(v,dict) else v
   actual=sha(name)
   if actual!=expected:
    assert Path(name).name=='NEXT_EXPERIMENT_PLAN.md',name
    assert sha(ROOT/'planning_entry_before.md')==before['planning_entry_start_hash']
    pointer_changes.append({'path':name,'historical_hash':expected,'execution_start_hash':before['planning_entry_start_hash'],'current_hash':actual,'scope':'mutable planning entry only; original content archived'})
   n+=1
  verified[b]=n
 save(ROOT/'data_chronology_integrity.json',{'passed':True,'checked_at':time.time(),'new_images':newcount,'prior_images_checked':historical,'banks':bankinfo,'all_quantities_pairs_labels_verified':True,'all_new_image_hashes_disjoint':True,'training_before_lock_before_test_verified':True,'historical_manifest_entries_checked':verified,'documented_planning_entry_changes':pointer_changes})
 print('Data, chronology and historical integrity passed:',verified)
if __name__=='__main__':main()
