"""B-H adaptive rounds; each round config and protocol frozen before execution."""
import os,argparse,json,pickle,time,resource
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from scipy.special import expit,logsumexp
from scipy.ndimage import label
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from crowbrain.task import COUNTS,draw_dots,balanced_pairs
from branch_b_round1 import Probe,sha,digest,save,continue_test
from branch_bt_round1 import encoded_pairs,extract,perturb,check_forward,permutation_indices,frozen_snapshot
from branch_bf_round1 import load_core,pooled_pack,training_groups,condition_scores,BASE_DELAYS,DELAYS,REFERENCE_DELAYS
from branch_bg_solver import fit_candidate
ROOT=Path('artifacts/branch_bh/round3');CFG=json.loads((ROOT/'config.json').read_text());N_BANKS=CFG['banks'];HEADS=['ten','eleven'];DELAYS=[5,9,15];REFERENCE_DELAYS=[5,15]

def verify_lock():
 m=json.loads((ROOT/'training_lock.json').read_text());assert all(sha(p)==h for p,h in m['files'].items());return m

def lock_heads():
 import shutil
 assert not list(ROOT.glob('test*.npz'));folder=ROOT/'seed301';folder.mkdir(exist_ok=False)
 old=Path('artifacts/branch_bh/round2');m=json.loads((old/'training_lock.json').read_text());files=dict(m['files']);assert all(sha(p)==h for p,h in files.items())
 shutil.copyfile(old/'seed301/heads.pkl',folder/'heads.pkl')
 for p in [old/'seed301/heads.pkl',old/'analysis_arrays.npz',old/'summary.json',folder/'heads.pkl',ROOT/'config.json',ROOT/'protocol.md',ROOT/'bridge.json']+list(Path('scripts').glob('branch_bh3*.py')):files[str(p)]=sha(p)
 save(ROOT/'training_lock.json',{'files':files,'locked_at':time.time(),'no_new_training':True,'source_heads':str(old/'seed301/heads.pkl'),'head_copy_equal':sha(folder/'heads.pkl')==sha(old/'seed301/heads.pkl')})
 print('Frozen exact round-2 head copies before any new images.',flush=True)

def prepare(fitting=False):
 if not fitting:lock=verify_lock()
 prior=json.loads(Path('artifacts/branch_bh/round2/data_manifest.json').read_text())['prior_files'];prior.update({str(p):sha(p) for p in Path('artifacts/branch_bh/round2').glob('test*.npz')})
 seen=set()
 for p,h in prior.items():
  assert sha(p)==h
  with np.load(p) as a:
   for key in ['images'] if 'images' in a else ['solid','ring']:seen.update(digest(x) for x in a[key].reshape(-1,1,32,32))
 assert len(seen)==CFG['prior_images']
 prior_count=len(seen);records=[];rejections=[]
 configs=[('fit',CFG['fit_seed'],1000,8000),('validation',CFG['fit_seed']+1,1000,2000)] if fitting else [(f'test{b:02d}',CFG['test_seed']+b,200,2000) for b in range(N_BANKS)]
 for name,seed,k,n in configs:
  p=ROOT/f'{name}.npz';assert not p.exists();rng=np.random.default_rng(seed);images=np.stack([draw_dots(int(c),rng)[0] for c in COUNTS for _ in range(k)]).reshape(5,k,1,32,32)
  for ci,c in enumerate(COUNTS):
   for j in range(k):
    old=h=digest(images[ci,j]);attempts=0
    while h in seen:images[ci,j,0]=draw_dots(int(c),rng)[0];h=digest(images[ci,j]);attempts+=1
    if attempts:rejections.append({'name':name,'count':int(c),'index':j,'draws':attempts,'old':old,'replacement':h})
    seen.add(h)
  assert np.array_equal([label(x[0]>.499,structure=np.ones((3,3)))[1] for x in images.reshape(-1,1,32,32)],np.repeat(COUNTS,k))
  rng=np.random.default_rng(seed+10000);pairs=balanced_pairs(n//40);rng.shuffle(pairs);a,b=pairs.T;ia,ib=rng.integers(0,k,(2,n));same=(a==b)&(ia==ib);ib[same]=(ib[same]+1)%k;y=(a==b).astype(np.int64)
  assert np.bincount(y).tolist()==[n//2]*2 and not np.any((a==b)&(ia==ib))
  for aa in range(5):
   for bb in range(5):assert ((a==aa)&(b==bb)).sum()==n//40*(4 if aa==bb else 1)
  np.savez_compressed(p,images=images,a=a,b=b,ia=ia,ib=ib,y=y,seed=seed);records.append({'path':str(p),'sha256':sha(p),'seed':seed})
  if not fitting:
   assert p.stat().st_mtime>=lock['locked_at'];yi,ri=permutation_indices({'seed':seed,'y':y,'b':b});np.savez_compressed(ROOT/f'permutations{name[4:]}.npz',label_indices=yi.astype(np.int16),row_indices=ri.astype(np.int16))
  print('prepared',name,flush=True)
 save(ROOT/('fit_data_manifest.json' if fitting else 'data_manifest.json'),{'banks':records,'prior_files':prior,'prior_images':prior_count,'new_images':len(configs)*5*k,'duplicate_rejections':rejections,'locked_before_generation':not fitting})

def active_heads(heads,refs,delay):
 a=dict(heads)
 if delay in REFERENCE_DELAYS:a['reference']=refs[f'ref{delay}']
 return a

@torch.no_grad()
def evaluate(seed):
 torch.set_num_threads(1);verify_lock();folder=ROOT/f'seed{seed}';start=time.monotonic();assert not (folder/'results.json').exists();model,refs,sigma,inputs=load_core(seed)
 with (folder/'heads.pkl').open('rb') as f:heads=pickle.load(f)
 before=frozen_snapshot(model,{**refs,**heads});inputs[str(folder/'heads.pkl')]=sha(folder/'heads.pkl');inputs[str(ROOT/'training_lock.json')]=sha(ROOT/'training_lock.json');cells=[];checks=[]
 save(folder/'evaluation_manifest.json',{'inputs':inputs,'before':before,'started_at':time.time()})
 with patch.object(LogisticRegression,'fit',side_effect=RuntimeError('test forbids fit')),patch.object(StandardScaler,'fit',side_effect=RuntimeError('test forbids fit')):
  for bi in range(N_BANKS):
   d=dict(np.load(ROOT/f'test{bi:02d}.npz'));es,et=encoded_pairs(model,d);y=d['y'];yi,ri=permutation_indices(d)
   for delay in DELAYS:
    active=active_heads(heads,refs,delay);pre,end=extract(model,es,et,delay);checks.append(check_forward(model,d,es,et,delay,pre,end));states=[end]+[perturb(model,pre,et,int(d['seed']),rep,sigma)[1] for rep in range(10)]
    pred=np.asarray([[h.predict(s.numpy()) for s in states] for h in active.values()],dtype=np.int8);original=np.asarray([model.readout(s).argmax(1).numpy() for s in states],dtype=np.int8)
    resetstates=[continue_test(model,torch.zeros_like(pre),et),perturb(model,torch.zeros_like(pre),et,int(d['seed']),0,sigma)[1]];reset=np.asarray([[h.predict(s.numpy()) for s in resetstates] for h in active.values()],dtype=np.int8)
    neg=np.stack([(reset==y).mean(-1),(pred[:,[0,1],None,:]==y[yi]).mean((-1,-2)),(pred[:,[0,1],:][:,:,ri]==y).mean((-1,-2))],axis=-1)
    cells.append({'bank':bi,'delay':delay,'head_order':list(active),'scores':(pred==y).mean(-1).tolist(),'negative_means':neg.tolist(),'original_scores':(original==y).mean(-1).tolist()})
    np.savez_compressed(folder/f'bank{bi:02d}_d{delay:02d}.npz',predictions=pred,original=original,reset=reset)
   if bi%5==4:print(seed,'evaluated banks',bi+1,flush=True)
 assert before==frozen_snapshot(model,{**refs,**heads});assert all(sha(p)==h for p,h in inputs.items());verify_lock()
 save(folder/'results.json',{'cells':cells,'forward_checks':checks,'snapshot':before,'frozen_verified':True,'fit_guard':True,'seconds':time.monotonic()-start})



if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['lock','prepare','evaluate']);a=p.parse_args()
 try:
  if a.phase=='lock':lock_heads()
  elif a.phase=='prepare':prepare()
  else:evaluate(301)
 except Exception as e:save(ROOT/f'failure_{a.phase}_{int(time.time())}.json',{'error':repr(e)});raise
