"""Independent standardized-space linear audit of every batched prediction."""
import sys,json,pickle,subprocess
import numpy as np
import torch
from branch_bi3_round import ROOT,CFG,HEADS,DELAYS,REFERENCE_DELAYS,N_BANKS,active_heads,load_core,bank_indices
from branch_bi3_audit import reconstruct_clean,audit_data
from branch_bi3_vectorized import verify_extension as verify_original_extension
from branch_bi_codec import load_bank,load_gzip_json,load_predictions
from branch_b_round1 import sha,save
from branch_bt_round1 import encoded_pairs,permutation_indices,frozen_snapshot

def explicit_predict(state,heads):
 x=np.asarray(state);out=[]
 for h in heads.values():
  assert np.array_equal(h.clf.classes_,[0,1]);z=x.copy()
  if h.scaler.with_mean:z-=np.asarray(h.scaler.mean_,dtype=z.dtype)
  if h.scaler.with_std:z/=np.asarray(h.scaler.scale_,dtype=z.dtype)
  scores=z@h.clf.coef_.T+h.clf.intercept_;out.append((scores.ravel()>0).astype(np.int8))
 return np.array(out)


def verify_extension():
 verify_original_extension();m=json.loads((ROOT/'audit_revision_04_lock.json').read_text());assert all(sha(p)==h for p,h in m['files'].items());return m

def permutation_means(raw,y,yi,ri):
 # Every stored permutation contributes exactly, via integer label counts.
 assert yi.shape==ri.shape and yi.shape[1]==len(y)
 p=raw['predictions'][:,[0,1]].astype(np.int64);reps=len(yi)
 label_counts=y[yi].sum(0);row_counts=y[np.argsort(ri,axis=1)].sum(0)
 base=reps*(len(y)-p.sum(-1));denom=reps*len(y)
 return np.stack([(raw['reset']==y).mean(-1),((2*p-1)@label_counts+base)/denom,((2*p-1)@row_counts+base)/denom],-1)

@torch.no_grad()
def audit_part(part):
 torch.set_num_threads(1);verify_extension();assert json.loads((ROOT/'independent_data_audit.json').read_text())['passed'];dm=json.loads((ROOT/'data_manifest.json').read_text());folder=ROOT/'seed301';model,refs,sigma,_=load_core(301)
 with (folder/'heads.pkl').open('rb') as f:heads=pickle.load(f)
 result=load_gzip_json(folder/'results.json.gz');em=json.loads((folder/'evaluation_manifest.json').read_text());before=frozen_snapshot(model,{**refs,**heads});assert before==result['snapshot']==em['before'];assert result['fit_guard'] and result['frozen_verified'] and result['cells_count']==N_BANKS*len(DELAYS);assert all(sha(p)==h for p,h in em['inputs'].items());checked=vectors=probe_checks=0
 for bi in bank_indices(part,N_BANKS):
  p=ROOT/f'test{bi:04d}.npz';assert sha(p)==dm['banks'][bi]['sha256'];d=load_bank(p);es,et=encoded_pairs(model,d);y=d['y'];yi,ri=permutation_indices(d);sp=folder/f'bank{bi:04d}_scores.json.gz';assert sha(sp)==result['bank_files'][str(sp)];rows=load_gzip_json(sp);cells={(x['bank'],x['delay']):x for x in rows};assert len(cells)==len(rows)==len(DELAYS)
  for delay in DELAYS:
   h,clean_end=reconstruct_clean(model,es,et,delay);W=model.weights().T;active=active_heads(heads,refs,delay);raw=load_predictions(folder/f'bank{bi:04d}_d{delay:02d}.npz');cell=cells[bi,delay];assert cell['head_order']==list(active);assert raw['predictions'].shape==(len(active),11,2000)
   for rep in range(-1,10):
    state=h.clone()
    if rep>=0:
     noise=np.random.default_rng(np.random.SeedSequence([int(d['seed']),20000,rep])).normal(size=tuple(h.shape)).astype(np.float32);state=torch.clamp(h+torch.from_numpy(noise)*sigma,min=0)
    for _ in range(4):state=(1-model.alpha)*state+model.alpha*torch.relu(model.input(et)+state@W)
    if rep==-1:state=clean_end
    pred=explicit_predict(state.numpy(),active);np.testing.assert_array_equal(pred,raw['predictions'][:,rep+1]);vectors+=len(active)
    # Additional direct production-Probe checks, fixed first 32 trials, clean/rep0.
    if rep in [-1,0]:
     for j,head in enumerate(active.values()):np.testing.assert_array_equal(head.predict(state[:32].numpy()),pred[j,:32]);probe_checks+=1
    np.testing.assert_array_equal(model.readout(state).argmax(1).numpy(),raw['original'][rep+1])
   for c in range(2):
    state=torch.zeros_like(h)
    if c:
     noise=np.random.default_rng(np.random.SeedSequence([int(d['seed']),20000,0])).normal(size=tuple(h.shape)).astype(np.float32);state=torch.clamp(torch.from_numpy(noise)*sigma,min=0)
    for _ in range(4):state=(1-model.alpha)*state+model.alpha*torch.relu(model.input(et)+state@W)
    np.testing.assert_array_equal(explicit_predict(state.numpy(),active),raw['reset'][:,c])
   np.testing.assert_array_equal((raw['predictions']==y).mean(-1),cell['scores']);np.testing.assert_array_equal((raw['original']==y).mean(-1),cell['original_scores'])
   neg=permutation_means(raw,y,yi,ri)
   np.testing.assert_allclose(neg,cell['negative_means'],rtol=0,atol=1e-15);checked+=1
  if bi%10==9:print('explicit audit part',part,'bank',bi,flush=True)
 assert before==frozen_snapshot(model,{**refs,**heads});verify_extension();assert checked==len(bank_indices(part,N_BANKS))*len(DELAYS);assert vectors==len(bank_indices(part,N_BANKS))*(len(DELAYS)*len(HEADS)+len(REFERENCE_DELAYS))*11
 save(ROOT/f'independent_audit_part{part}.json',dict(passed=True,cells=checked,prediction_vectors=vectors,original_probe_slice_checks=probe_checks,all_full_predictions_checked_in_standardized_space=True))

def main():
 verify_extension();assert json.loads((ROOT/'independent_data_audit.json').read_text())['passed']
 children=[subprocess.Popen([sys.executable,__file__,'part',str(p)]) for p in [0,1]];codes=[c.wait() for c in children];assert codes==[0,0],codes;rs=[json.loads((ROOT/f'independent_audit_part{p}.json').read_text()) for p in [0,1]];assert all(x['passed'] for x in rs)
 assert sum(x['cells'] for x in rs)==N_BANKS*len(DELAYS) and sum(x['prediction_vectors'] for x in rs)==N_BANKS*(len(DELAYS)*len(HEADS)+len(REFERENCE_DELAYS))*11
 save(ROOT/'independent_audit.json',dict(passed=True,cells=sum(x['cells'] for x in rs),prediction_vectors=sum(x['prediction_vectors'] for x in rs),original_probe_slice_checks=sum(x['original_probe_slice_checks'] for x in rs),all_noise_reconstructed=True,recurrence_rebuilt_from_pixels=True,all_full_predictions_checked_in_standardized_space=True,chronology_verified=True,frozen_parameters_verified=True,implementation_lock_sha256=sha(ROOT/'audit_revision_04_lock.json'),vectorized_lock_sha256=sha(ROOT/'vectorized_execution_lock.json'),data_audit_sha256=sha(ROOT/'independent_data_audit.json'),scaler_parameters_cast_to_input_dtype=True,all_permutations_checked_by_integer_counts=True))
if __name__=='__main__':
 if len(sys.argv)>1:audit_part(int(sys.argv[2]))
 else:main()
