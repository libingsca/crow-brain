"""New common banks; original generator parity, full lossless permutations."""
import sys,json,multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.ndimage import label
from crowbrain.task import COUNTS,draw_dots,balanced_pairs
from branch_b_round1 import sha,digest,save
from branch_bi_codec import draw_recorded,decode_geometry,load_bank,save_lzma_npz
from branch_bi3_data import worker
from branch_bt_round1 import permutation_indices
from branch_bi4_common import *

def load_previous(j):
 prior=previous(j);seen=set();seeds=set()
 for p,h in prior.items():
  assert sha(p)==h;d=load_bank(p)
  if 'seed' in d:seeds.add(int(d['seed']))
  for key in ['images'] if 'images' in d else ['solid','ring']:seen.update(digest(x) for x in d[key].reshape(-1,1,32,32))
 assert len(seen)==CFG['prior_images']+bounds(j)[0]*1000
 return prior,seen,seeds

def prepare(j):
 lock=verify_lock();root=stage(j);assert (root/'decision.json').exists();prior,seen,oldseeds=load_previous(j);start,end=bounds(j);new=set(range(CFG['test_seed']+start,CFG['test_seed']+end));streams=new|{s+10000 for s in new};assert len(streams)==2*len(new) and not streams&(oldseeds|{s+10000 for s in oldseeds});save(root/'seed_audit.json',dict(passed=True,seeds=sorted(new),old_seed_count=len(oldseeds)));records=[];rejections=[]
 with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as ex:
  pending={i:ex.submit(worker,(CFG['test_seed']+i,200,2000)) for i in range(start,min(start+2,end))}
  for i in range(start,end):
   out=pending.pop(i).result()
   if i+2<end:pending[i+2]=ex.submit(worker,(CFG['test_seed']+i+2,200,2000))
   p=root/f'test{i:04d}.npz';assert not p.exists();geo=out['geometry'];rng=np.random.default_rng();rng.bit_generator.state=out['rng_state'];orig=np.random.default_rng();orig.bit_generator.state=out['rng_state'];changed=False
   for ix,h in enumerate(out['hashes']):
    old=h;ci=ix//200;ii=ix%200;tries=0
    while h in seen:
     im,g=draw_recorded(int(COUNTS[ci]),rng);ref,_=draw_dots(int(COUNTS[ci]),orig);np.testing.assert_array_equal(im,ref);geo[ci,ii]=g;h=digest(im);tries+=1
    if tries:changed=True;rejections.append(dict(path=str(p),count=int(COUNTS[ci]),index=ii,draws=tries,old=old,replacement=h))
    seen.add(h)
   if changed:out['pixel_sha256']=digest(decode_geometry(geo))
   seed=CFG['test_seed']+i;prng=np.random.default_rng(seed+10000);pairs=balanced_pairs(50);prng.shuffle(pairs);a,b=pairs.T;ia,ib=prng.integers(0,200,(2,2000));same=(a==b)&(ia==ib);ib[same]=(ib[same]+1)%200;y=(a==b).astype(np.int64)
   np.savez_compressed(p,geometry=geo,a=a,b=b,ia=ia,ib=ib,y=y,seed=seed);assert p.stat().st_mtime>=lock['locked_at'];yi,ri=permutation_indices(dict(seed=seed,y=y,b=b));save_lzma_npz(root/f'permutations{i:04d}.npz',label_indices=yi.astype(np.int16),row_indices=ri.astype(np.int16));records.append(dict(path=str(p),sha256=sha(p),seed=seed,pixel_sha256=out['pixel_sha256'],images=1000,trials=2000,original_generator_bit_parity=True,geometry_roundtrip_exact=True))
   if (i+1)%16==0:print('prepared',i+1,'/',end,flush=True)
 save(root/'data_manifest.json',dict(banks=records,prior_files=prior,prior_images=CFG['prior_images']+start*1000,new_images=(end-start)*1000,unique_total=len(seen),duplicate_rejections=rejections,locked_before_generation=True))

def audit(j):
 lock=verify_lock();root=stage(j);m=json.loads((root/'data_manifest.json').read_text());prior,seen,_=load_previous(j);assert prior==m['prior_files'];start,end=bounds(j)
 for i,rec in zip(range(start,end),m['banks'],strict=True):
  p=root/f'test{i:04d}.npz';assert sha(p)==rec['sha256'] and p.stat().st_mtime>=lock['locked_at'];d=load_bank(p);assert digest(d['images'])==rec['pixel_sha256'] and int(d['seed'])==CFG['test_seed']+i;hs=[digest(x) for x in d['images'].reshape(-1,1,32,32)];assert len(set(hs))==1000 and not seen.intersection(hs);seen.update(hs)
  assert np.array_equal([label(x[0]>.499,structure=np.ones((3,3)))[1] for x in d['images'].reshape(-1,1,32,32)],np.repeat(COUNTS,200));np.testing.assert_array_equal(d['y'],d['a']==d['b']);assert np.bincount(d['y']).tolist()==[1000,1000] and not np.any((d['a']==d['b'])&(d['ia']==d['ib']))
  for a in range(5):
   for b in range(5):assert ((d['a']==a)&(d['b']==b)).sum()==50*(4 if a==b else 1)
  yi,ri=permutation_indices(d)
  with np.load(root/f'permutations{i:04d}.npz') as f:np.testing.assert_array_equal(yi,f['label_indices']);np.testing.assert_array_equal(ri,f['row_indices'])
 assert len(seen)==m['unique_total'];save(root/'independent_data_audit.json',dict(passed=True,unique_total=len(seen),new_images=(end-start)*1000,prior_images=m['prior_images']))
if __name__=='__main__':
 (prepare if sys.argv[1]=='prepare' else audit)(int(sys.argv[2]))
