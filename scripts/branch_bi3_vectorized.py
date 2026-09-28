"""Equivalent batched linear prediction; frozen model/data/protocol unchanged."""
import os,sys,json,pickle,time,subprocess,shutil
from pathlib import Path
import numpy as np
import torch
from unittest.mock import patch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from branch_bi3_round import ROOT,CFG,HEADS,DELAYS,REFERENCE_DELAYS,N_BANKS,load_core,active_heads,verify_lock,bank_indices
from branch_bi_codec import load_bank,load_predictions,save_predictions,save_gzip_json,load_gzip_json
from branch_bt_round1 import encoded_pairs,extract,perturb,check_forward,permutation_indices,frozen_snapshot
from branch_b_round1 import save,sha,continue_test

class FastPredictor:
 def __init__(self,heads):
  self.heads=list(heads.values());self.fallback_heads=0
  assert all(np.array_equal(h.clf.classes_,[0,1]) for h in self.heads)
  self.mean=np.stack([h.scaler.mean_ if h.scaler.with_mean else np.zeros_like(h.clf.coef_[0]) for h in self.heads]);scale=np.stack([h.scaler.scale_ if h.scaler.with_std else np.ones_like(h.clf.coef_[0]) for h in self.heads]);coef=np.stack([h.clf.coef_[0] for h in self.heads]);intercept=np.array([h.clf.intercept_[0] for h in self.heads]);assert np.all(scale>0) and np.isfinite(scale).all()
  self.weight=coef/scale;self.offset=intercept-(self.mean*self.weight).sum(1);self.error_offset=np.abs(self.mean*self.weight).sum(1)+np.abs(intercept)+1.
 def predict(self,state):
  original=np.asarray(state);x=np.asarray(original,dtype=np.float64);assert np.isfinite(x).all()
  z=x@self.weight.T+self.offset
  # Generous forward-error envelope for 64-dimensional subtraction, division,
  # products, sums and cancellation; close decisions use the original predictor.
  magnitude=np.abs(x)@np.abs(self.weight).T+self.error_offset
  # StandardScaler preserves float32 inputs, including both in-place roundings.
  dtype_eps=np.finfo(original.dtype).eps
  bound=np.maximum(1e-8,(512*np.finfo(np.float64).eps+4*dtype_eps)*magnitude)
  uncertain=np.any((np.abs(z)<=bound)|~np.isfinite(z)|~np.isfinite(bound),axis=0);p=(z>0).T.astype(np.int8)
  for i in np.flatnonzero(uncertain):p[i]=self.heads[i].predict(original);self.fallback_heads+=1
  return p

def verify_extension():
 verify_lock();m=json.loads((ROOT/'vectorized_execution_lock.json').read_text());assert all(sha(p)==h for p,h in m['files'].items());return m

@torch.no_grad()
def evaluate_part(part):
 torch.set_num_threads(1);verify_extension();folder=ROOT/'seed301';model,refs,sigma,inputs=load_core(301);start=time.monotonic()
 with (folder/'heads.pkl').open('rb') as f:heads=pickle.load(f)
 before=frozen_snapshot(model,{**refs,**heads});inputs.update({str(p):sha(p) for p in [folder/'heads.pkl',ROOT/'training_lock.json',ROOT/'vectorized_execution_lock.json']});save(folder/f'evaluation_manifest_part{part}.json',dict(inputs=inputs,before=before,started_at=time.time(),implementation='batched linear with original-predictor fallback'))
 caches={d:FastPredictor(active_heads(heads,refs,d)) for d in DELAYS};checks=[];compared=0;old=ROOT/'implementation_revisions/slow_evaluation/seed301';complete=set(json.loads((old.parent/'snapshot.json').read_text())['complete_banks'])
 with patch.object(LogisticRegression,'fit',side_effect=RuntimeError('test forbids fit')),patch.object(StandardScaler,'fit',side_effect=RuntimeError('test forbids fit')):
  for bi in bank_indices(part,N_BANKS):
   d=load_bank(ROOT/f'test{bi:04d}.npz');es,et=encoded_pairs(model,d);y=d['y'];yi,ri=permutation_indices(d);cells=[]
   for delay in DELAYS:
    active=active_heads(heads,refs,delay);pre,end=extract(model,es,et,delay);checks.append(check_forward(model,d,es,et,delay,pre,end));states=[end]+[perturb(model,pre,et,int(d['seed']),rep,sigma)[1] for rep in range(10)]
    pred=np.stack([caches[delay].predict(st.numpy()) for st in states],axis=1);original=np.asarray([model.readout(st).argmax(1).numpy() for st in states],dtype=np.int8)
    rs=[continue_test(model,torch.zeros_like(pre),et),perturb(model,torch.zeros_like(pre),et,int(d['seed']),0,sigma)[1]];reset=np.stack([caches[delay].predict(st.numpy()) for st in rs],axis=1)
    if bi in complete:
     previous=load_predictions(old/f'bank{bi:04d}_d{delay:02d}.npz')
     for key,value in dict(predictions=pred,original=original,reset=reset).items():np.testing.assert_array_equal(value,previous[key])
     compared+=len(active)*11
    neg=np.stack([(reset==y).mean(-1),(pred[:,[0,1],None,:]==y[yi]).mean((-1,-2)),(pred[:,[0,1],:][:,:,ri]==y).mean((-1,-2))],axis=-1)
    cells.append(dict(bank=bi,delay=delay,head_order=list(active),scores=(pred==y).mean(-1).tolist(),negative_means=neg.tolist(),original_scores=(original==y).mean(-1).tolist()))
    save_predictions(folder/f'bank{bi:04d}_d{delay:02d}.npz',predictions=pred,original=original,reset=reset)
   save_gzip_json(folder/f'bank{bi:04d}_scores.json.gz',cells)
   if bi%10==9:print('vectorized part',part,'bank',bi,flush=True)
 assert before==frozen_snapshot(model,{**refs,**heads});assert all(sha(p)==h for p,h in inputs.items());verify_extension()
 save_gzip_json(folder/f'results_part{part}.json.gz',dict(cells_count=len(bank_indices(part,N_BANKS))*len(DELAYS),forward_checks=checks,snapshot=before,frozen_verified=True,fit_guard=True,seconds=time.monotonic()-start,original_partial_prediction_vectors_compared=compared,fallback_head_calls=sum(c.fallback_heads for c in caches.values())))

def evaluate():
 verify_extension();children=[subprocess.Popen([sys.executable,__file__,'part',str(p)]) for p in [0,1]];codes=[p.wait() for p in children];assert codes==[0,0],codes;folder=ROOT/'seed301';rs=[load_gzip_json(folder/f'results_part{p}.json.gz') for p in [0,1]];ms=[json.loads((folder/f'evaluation_manifest_part{p}.json').read_text()) for p in [0,1]];assert rs[0]['snapshot']==rs[1]['snapshot'] and ms[0]['inputs']==ms[1]['inputs']
 save(folder/'evaluation_manifest.json',dict(inputs=ms[0]['inputs'],before=ms[0]['before'],started_at=min(x['started_at'] for x in ms)))
 save_gzip_json(folder/'results.json.gz',dict(cells_count=sum(x['cells_count'] for x in rs),forward_checks=sum([x['forward_checks'] for x in rs],[]),snapshot=rs[0]['snapshot'],frozen_verified=True,fit_guard=True,bank_files={str(folder/f'bank{i:04d}_scores.json.gz'):sha(folder/f'bank{i:04d}_scores.json.gz') for i in range(N_BANKS)},part_seconds=[r['seconds'] for r in rs],original_partial_prediction_vectors_compared=sum(r['original_partial_prediction_vectors_compared'] for r in rs),fallback_head_calls=sum(r['fallback_head_calls'] for r in rs)))
if __name__=='__main__':
 if sys.argv[1]=='part':evaluate_part(int(sys.argv[2]))
 else:evaluate()
