"""Read-only full test reconstruction from pixels and fixed seeds."""
import json,pickle
import numpy as np
import torch
from scipy.ndimage import label
from branch_bh_round import ROOT,HEADS,DELAYS,REFERENCE_DELAYS,load_core,active_heads,verify_lock,CFG
from branch_b_round1 import save,sha,digest
from branch_bt_round1 import encoded_pairs,permutation_indices,frozen_snapshot

@torch.no_grad()
def reconstruct_clean(model,es,et,delay):
 pre=[];end=[];W=model.weights().T
 for sample,test in zip(es.split(256),et.split(256)):
  h=torch.zeros(len(sample),64)
  for _ in range(4):h=(1-model.alpha)*h+model.alpha*torch.relu(model.input(sample)+h@W)
  blank=model.input(torch.zeros_like(sample))
  for _ in range(delay):h=(1-model.alpha)*h+model.alpha*torch.relu(blank+h@W)
  pre.append(h)
  for _ in range(4):h=(1-model.alpha)*h+model.alpha*torch.relu(model.input(test)+h@W)
  end.append(h)
 return torch.cat(pre),torch.cat(end)

@torch.no_grad()
def main():
 torch.set_num_threads(1);lock=verify_lock();manifest=json.loads((ROOT/'data_manifest.json').read_text());seen=set();checked=0;vectors=0
 for p,h in manifest['prior_files'].items():
  assert sha(p)==h
  with np.load(p) as a:
   for key in ['images'] if 'images' in a else ['solid','ring']:seen.update(digest(x) for x in a[key].reshape(-1,1,32,32))
 assert len(seen)==manifest['prior_images']
 for bi,record in enumerate(manifest['banks']):
  p=ROOT/f'test{bi:02d}.npz';assert sha(p)==record['sha256'] and p.stat().st_mtime>=lock['locked_at'];d=dict(np.load(p));assert int(d['seed'])==CFG['test_seed']+bi
  hs=[digest(x) for x in d['images'].reshape(-1,1,32,32)];assert len(set(hs))==1000 and not seen.intersection(hs);seen.update(hs)
  assert np.array_equal([label(x[0]>.499,structure=np.ones((3,3)))[1] for x in d['images'].reshape(-1,1,32,32)],np.repeat(np.arange(2,7),200))
  np.testing.assert_array_equal(d['y'],d['a']==d['b']);assert np.bincount(d['y']).tolist()==[1000,1000]
  assert not np.any((d['a']==d['b'])&(d['ia']==d['ib']))
  for aa in range(5):
   for bb in range(5):assert ((d['a']==aa)&(d['b']==bb)).sum()==50*(4 if aa==bb else 1)
  yi,ri=permutation_indices(d);perm=np.load(ROOT/f'permutations{bi:02d}.npz');np.testing.assert_array_equal(yi,perm['label_indices']);np.testing.assert_array_equal(ri,perm['row_indices'])
 for seed in range(301,306):
  folder=ROOT/f'seed{seed}';model,refs,sigma,_=load_core(seed)
  with (folder/'heads.pkl').open('rb') as f:heads=pickle.load(f)
  result=json.loads((folder/'results.json').read_text());em=json.loads((folder/'evaluation_manifest.json').read_text());before=frozen_snapshot(model,{**refs,**heads});assert before==em['before']==result['snapshot'];assert result['fit_guard'] and result['frozen_verified'];assert all(sha(p)==h for p,h in em['inputs'].items());assert len(result['forward_checks'])==1100//5
  cells={(x['bank'],x['delay']):x for x in result['cells']};assert len(cells)==len(result['cells'])==220
  for bi in range(20):
   d=dict(np.load(ROOT/f'test{bi:02d}.npz'));es,et=encoded_pairs(model,d);y=d['y'];yi,ri=permutation_indices(d)
   for delay in DELAYS:
    # Independently spell out recurrence in the sample, delay and test periods.
    h,clean_end=reconstruct_clean(model,es,et,delay);W=model.weights().T
    active=active_heads(heads,refs,delay);raw=dict(np.load(folder/f'bank{bi:02d}_d{delay:02d}.npz'));cell=cells[bi,delay];assert cell['head_order']==list(active)
    assert raw['predictions'].shape==(len(active),11,2000)
    for rep in range(-1,10):
     state=h.clone()
     if rep>=0:
      z=np.random.default_rng(np.random.SeedSequence([int(d['seed']),20000,rep])).normal(size=tuple(h.shape)).astype(np.float32);state=torch.clamp(h+torch.from_numpy(z)*sigma,min=0)
     for _ in range(4):state=(1-model.alpha)*state+model.alpha*torch.relu(model.input(et)+state@W)
     if rep==-1:state=clean_end
     for j,head in enumerate(active.values()):
      np.testing.assert_array_equal(head.predict(state.numpy()),raw['predictions'][j,rep+1]);vectors+=1
     np.testing.assert_array_equal(model.readout(state).argmax(1).numpy(),raw['original'][rep+1])
    for cond in range(2):
     state=torch.zeros_like(h)
     if cond:
      z=np.random.default_rng(np.random.SeedSequence([int(d['seed']),20000,0])).normal(size=tuple(h.shape)).astype(np.float32);state=torch.clamp(torch.from_numpy(z)*sigma,min=0)
     for _ in range(4):state=(1-model.alpha)*state+model.alpha*torch.relu(model.input(et)+state@W)
     for j,head in enumerate(active.values()):np.testing.assert_array_equal(head.predict(state.numpy()),raw['reset'][j,cond])
    np.testing.assert_array_equal((raw['predictions']==y).mean(-1),cell['scores']);np.testing.assert_array_equal((raw['original']==y).mean(-1),cell['original_scores'])
    # Independently loop over permutation replicates instead of broadcasted rows.
    neg=np.empty((len(active),2,3))
    for j in range(len(active)):
     for c in range(2):
      p=raw['predictions'][j,c];neg[j,c]=[(raw['reset'][j,c]==y).mean(),np.mean([(p==y[ix]).mean() for ix in yi]),np.mean([(p[ix]==y).mean() for ix in ri])]
    np.testing.assert_allclose(neg,cell['negative_means'],rtol=0,atol=1e-15);checked+=1
  assert before==frozen_snapshot(model,{**refs,**heads});print('independently reconstructed',seed,flush=True)
 assert checked==1100 and vectors==20*5*(11*len(HEADS)+8)*11 and len(seen)==manifest['prior_images']+20000;verify_lock()
 save(ROOT/'independent_audit.json',{'passed':True,'cells':checked,'prediction_vectors':vectors,'prior_images':manifest['prior_images'],'new_images':20000,'all_noise_reconstructed':True,'recurrence_rebuilt_from_pixels':True,'chronology_verified':True,'frozen_parameters_verified':True})
if __name__=='__main__':main()
