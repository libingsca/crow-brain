"""Bounded, lossless generation with original-generator pixel parity."""
import os,sys,json,time,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.ndimage import label
from crowbrain.task import COUNTS,draw_dots,balanced_pairs
from branch_b_round1 import sha,digest,save
from branch_bt_round1 import permutation_indices
from branch_bi_codec import draw_recorded,decode_geometry,load_bank,save_lzma_npz
R=Path('artifacts/branch_bi/round3');CFG=json.loads((R/'config.json').read_text())
def worker(args):
 seed,k,n=args;rng=np.random.default_rng(seed);orig=np.random.default_rng(seed);images=[];geo=[];hs=[]
 for c in COUNTS:
  for _ in range(k):
   im,g=draw_recorded(int(c),rng);reference,_=draw_dots(int(c),orig);np.testing.assert_array_equal(im,reference);images.append(im);geo.append(g);hs.append(digest(im))
 assert rng.bit_generator.state==orig.bit_generator.state
 images=np.array(images).reshape(5,k,1,32,32);geo=np.array(geo).reshape(5,k,13);np.testing.assert_array_equal(decode_geometry(geo),images)
 assert np.array_equal([label(im[0]>.499,structure=np.ones((3,3)))[1] for im in images.reshape(-1,1,32,32)],np.repeat(COUNTS,k))
 return dict(geometry=geo,hashes=hs,pixel_sha256=digest(images),rng_state=rng.bit_generator.state)

def previous():
 m=json.loads(Path('artifacts/branch_bi/round2/data_manifest.json').read_text());p=dict(m['prior_files']);p.update({x['path']:x['sha256'] for x in m['banks']});return p

def prepare(fitting):
 prior=previous();seen=set();oldseeds=set();records=[];rejections=[]
 if not fitting:
  lock=json.loads((R/'training_lock.json').read_text());assert all(sha(p)==h for p,h in lock['files'].items())
  m=json.loads((R/'fit_data_manifest.json').read_text());prior.update({x['path']:x['sha256'] for x in m['banks']})
 for p,h in prior.items():
  assert sha(p)==h;d=load_bank(p)
  if 'seed' in d:oldseeds.add(int(d['seed']))
  for key in ['images'] if 'images' in d else ['solid','ring']:seen.update(digest(im) for im in d[key].reshape(-1,1,32,32))
 expected=CFG['prior_images']+(0 if fitting else CFG['K']*10000);assert len(seen)==expected;configs=[]
 if fitting:
  for b in range(1,CFG['K']+1):
   c=json.loads((R/f'batch{b}/config.json').read_text());configs.extend([(R/f'batch{b}/fit.npz',c['fit_seed'],1000,8000),(R/f'batch{b}/validation.npz',c['fit_seed']+1,1000,2000)])
 else:configs=[(R/f'test{i:04d}.npz',CFG['test_seed']+i,200,2000) for i in range(CFG['banks'])]
 new={s for p,s,k,n in configs};assert len(new)==len(configs);streams=new|{s+10000 for s in new};assert len(streams)==2*len(new) and not streams&(oldseeds|{s+10000 for s in oldseeds})
 save(R/('fit_seed_audit.json' if fitting else 'test_seed_audit.json'),dict(passed=True,seeds=sorted(new),historical_seeds=len(oldseeds)))
 with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as ex:
  pending={i:ex.submit(worker,configs[i][1:]) for i in range(min(2,len(configs)))}
  for i,(p,seed,k,n) in enumerate(configs):
   result=pending.pop(i).result()
   if i+2<len(configs):pending[i+2]=ex.submit(worker,configs[i+2][1:])
   assert not p.exists();geo=result['geometry'];hs=result['hashes'];rng=np.random.default_rng();rng.bit_generator.state=result['rng_state'];orig=np.random.default_rng();orig.bit_generator.state=result['rng_state'];changed=False
   for j,h in enumerate(hs):
    old=h;ci=j//k;idx=j%k;tries=0
    while h in seen:
     im,g=draw_recorded(int(COUNTS[ci]),rng);reference,_=draw_dots(int(COUNTS[ci]),orig);np.testing.assert_array_equal(im,reference);geo[ci,idx]=g;h=digest(im);tries+=1
    if tries:changed=True;rejections.append(dict(path=str(p),count=int(COUNTS[ci]),index=idx,draws=tries,old=old,replacement=h))
    seen.add(h)
   if changed:result['pixel_sha256']=digest(decode_geometry(geo))
   prng=np.random.default_rng(seed+10000);pairs=balanced_pairs(n//40);prng.shuffle(pairs);a,b=pairs.T;ia,ib=prng.integers(0,k,(2,n));same=(a==b)&(ia==ib);ib[same]=(ib[same]+1)%k;y=(a==b).astype(np.int64)
   assert np.bincount(y).tolist()==[n//2]*2 and not np.any((a==b)&(ia==ib))
   for aa in range(5):
    for bb in range(5):assert np.sum((a==aa)&(b==bb))==n//40*(4 if aa==bb else 1)
   np.savez_compressed(p,geometry=geo,a=a,b=b,ia=ia,ib=ib,y=y,seed=seed)
   rec=dict(path=str(p),sha256=sha(p),seed=seed,pixel_sha256=result['pixel_sha256'],images=5*k,trials=n,original_generator_bit_parity=True,geometry_roundtrip_exact=True);records.append(rec)
   if not fitting:
    assert p.stat().st_mtime>=lock['locked_at'];yi,ri=permutation_indices(dict(seed=seed,y=y,b=b));save_lzma_npz(R/f'permutations{i:04d}.npz',label_indices=yi.astype(np.int16),row_indices=ri.astype(np.int16))
   if (i+1)%8==0:print('prepared',i+1,'/',len(configs),flush=True)
 save(R/('fit_data_manifest.json' if fitting else 'data_manifest.json'),dict(banks=records,prior_files=prior,prior_images=expected,new_images=len(seen)-expected,unique_total=len(seen),duplicate_rejections=rejections,locked_before_generation=not fitting))

def audit_fit_data():
 m=json.loads((R/'fit_data_manifest.json').read_text());seen=set()
 for p,h in m['prior_files'].items():
  assert sha(p)==h;d=load_bank(p)
  for key in ['images'] if 'images' in d else ['solid','ring']:seen.update(digest(im) for im in d[key].reshape(-1,1,32,32))
 assert len(seen)==m['prior_images']
 for record in m['banks']:
  p=record['path'];assert sha(p)==record['sha256'];d=load_bank(p);assert digest(d['images'])==record['pixel_sha256'];hs=[digest(im) for im in d['images'].reshape(-1,1,32,32)];assert len(set(hs))==5000 and not seen.intersection(hs);seen.update(hs)
  assert np.array_equal([label(im[0]>.499,structure=np.ones((3,3)))[1] for im in d['images'].reshape(-1,1,32,32)],np.repeat(COUNTS,1000));np.testing.assert_array_equal(d['y'],d['a']==d['b']);n=record['trials'];assert np.bincount(d['y']).tolist()==[n//2]*2 and not np.any((d['a']==d['b'])&(d['ia']==d['ib']))
  for a in range(5):
   for b in range(5):assert np.sum((d['a']==a)&(d['b']==b))==n//40*(4 if a==b else 1)
 assert len(seen)==m['unique_total'];save(R/'fit_data_audit.json',dict(passed=True,banks=len(m['banks']),unique_total=len(seen),decoded_pixel_hashes_counts_pairs_verified=True))
if __name__=='__main__':
 if sys.argv[1]=='audit_fit':audit_fit_data()
 else:prepare(sys.argv[1]=='fit')
