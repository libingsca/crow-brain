"""Global disjoint image generation; no test generation before all 30 fits lock."""
import sys,json,time
from pathlib import Path
import numpy as np
from scipy.ndimage import label
from crowbrain.task import COUNTS,draw_dots,balanced_pairs
from branch_b_round1 import sha,digest,save
from branch_bt_round1 import permutation_indices
R=Path('artifacts/branch_bi/round1')
def prior_files():
 m=json.loads(Path('artifacts/branch_bh/round3/data_manifest.json').read_text());p=dict(m['prior_files']);p.update({x['path']:x['sha256'] for x in m['banks']});return p

def prepare(fitting):
 prior=prior_files();seen=set();records=[];reject=[];oldseeds=set()
 if not fitting:
  lock=json.loads((R/'global_training_lock.json').read_text());assert all(sha(p)==h for p,h in lock['files'].items())
  for p in sorted(R.glob('batch*/*ation.npz'))+sorted(R.glob('batch*/fit.npz')):prior[str(p)]=sha(p)
 for p,h in prior.items():
  assert sha(p)==h
  with np.load(p) as d:
   if 'seed' in d:oldseeds.add(int(d['seed']))
   for key in ['images'] if 'images' in d else ['solid','ring']:seen.update(digest(x) for x in d[key].reshape(-1,1,32,32))
 assert len(seen)==(614000 if fitting else 674000)
 configs=[json.loads((R/f'batch{b}/config.json').read_text()) for b in range(1,7)]
 newseeds=[c['fit_seed']+j for c in configs for j in range(2)] if fitting else [c['test_seed']+j for c in configs for j in range(40)]
 assert len(set(newseeds))==len(newseeds) and not set(newseeds)&oldseeds
 assert not ((set(newseeds)|{s+10000 for s in newseeds})&(oldseeds|{s+10000 for s in oldseeds}))
 save(R/('fit_seed_audit.json' if fitting else 'test_seed_audit.json'),dict(passed=True,new_seeds=newseeds,historical_seed_count=len(oldseeds),structured_streams=['bank_seed,20000,rep','bank_seed,30000,rep','bank_seed,40000,rep']))
 for b in range(1,7):
  root=R/f'batch{b}';cfg=json.loads((root/'config.json').read_text());batchrecords=[];batchprior=dict(prior);priorcount=len(seen)
  configs=[('fit',cfg['fit_seed'],1000,8000),('validation',cfg['fit_seed']+1,1000,2000)] if fitting else [(f'test{i:02d}',cfg['test_seed']+i,200,2000) for i in range(40)]
  for name,seed,k,n in configs:
   p=root/f'{name}.npz';rng=np.random.default_rng(seed);imgs=np.stack([draw_dots(int(c),rng)[0] for c in COUNTS for _ in range(k)]).reshape(5,k,1,32,32)
   for ci,c in enumerate(COUNTS):
    for j in range(k):
     h=digest(imgs[ci,j]);old=h;tries=0
     while h in seen:imgs[ci,j,0]=draw_dots(int(c),rng)[0];h=digest(imgs[ci,j]);tries+=1
     if tries:reject.append(dict(path=str(p),count=int(c),index=j,draws=tries,old=old,replacement=h))
     seen.add(h)
   assert np.array_equal([label(x[0]>.499,structure=np.ones((3,3)))[1] for x in imgs.reshape(-1,1,32,32)],np.repeat(COUNTS,k))
   prng=np.random.default_rng(seed+10000);pairs=balanced_pairs(n//40);prng.shuffle(pairs);a,bb=pairs.T;ia,ib=prng.integers(0,k,(2,n));same=(a==bb)&(ia==ib);ib[same]=(ib[same]+1)%k;y=(a==bb).astype(np.int64)
   assert np.bincount(y).tolist()==[n//2]*2 and not np.any((a==bb)&(ia==ib))
   payload=dict(images=imgs,a=a,b=bb,ia=ia,ib=ib,y=y,seed=seed)
   if p.exists():
    assert fitting
    with np.load(p) as existing:
     for key,value in payload.items():np.testing.assert_array_equal(existing[key],value)
   else:np.savez_compressed(p,**payload)
   rec=dict(path=str(p),sha256=sha(p),seed=seed);records.append(rec);batchrecords.append(rec);prior[str(p)]=rec['sha256']
   if not fitting:
    assert p.stat().st_mtime>=lock['locked_at'];yi,ri=permutation_indices({'seed':seed,'y':y,'b':bb});np.savez_compressed(root/f'permutations{name[4:]}.npz',label_indices=yi.astype(np.int16),row_indices=ri.astype(np.int16))
  save(root/('fit_data_manifest.json' if fitting else 'data_manifest.json'),dict(banks=batchrecords,prior_files=batchprior,prior_images=priorcount,new_images=len(configs)*5*k,duplicate_rejections=reject,locked_before_generation=not fitting))
  print('prepared',b,'fit' if fitting else 'test',len(seen),flush=True)
 save(R/('all_fit_data.json' if fitting else 'all_test_data.json'),dict(banks=records,unique_total=len(seen),duplicate_rejections=reject))
if __name__=='__main__':prepare(sys.argv[1]=='fit')
