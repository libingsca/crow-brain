"""Independent full recurrent and standardized-space reconstruction."""
import sys,json,subprocess
import numpy as np
import torch
from branch_bi4_common import *
from branch_bi_codec import load_bank,load_predictions,load_gzip_json
from branch_bi3_audit import reconstruct_clean
from branch_bi3_audit_revision04 import explicit_predict,permutation_means
from branch_bt_round1 import encoded_pairs,permutation_indices,frozen_snapshot

@torch.no_grad()
def part(j,part_id):
 torch.set_num_threads(1);verify_lock();root=stage(j);assert json.loads((root/'independent_data_audit.json').read_text())['passed'];folder=root/'seed301';model,refs,sigma,_=load_core(301);h=heads();meta=load_gzip_json(folder/'results.json.gz');em=json.loads((folder/'evaluation_manifest.json').read_text());before=frozen_snapshot(model,{**refs,**h});assert before==meta['snapshot']==em['before'] and meta['frozen_verified'] and meta['fit_guard'];assert all(sha(p)==v for p,v in em['inputs'].items());dm=json.loads((root/'data_manifest.json').read_text());records={x['path']:x for x in dm['banks']};checked=vectors=probe_checks=0
 for bi in indices(j,part_id):
  dp=root/f'test{bi:04d}.npz';assert sha(dp)==records[str(dp)]['sha256'];d=load_bank(dp);es,et=encoded_pairs(model,d);y=d['y'];yi,ri=permutation_indices(d);sp=folder/f'bank{bi:04d}_scores.json.gz';assert sha(sp)==meta['bank_files'][str(sp)];rows=load_gzip_json(sp);cells={x['delay']:x for x in rows};assert len(rows)==len(cells)==len(DELAYS) and all(x['bank']==bi for x in rows)
  for delay in DELAYS:
   pre,clean=reconstruct_clean(model,es,et,delay);w=model.weights().T;hh=active(h,refs,delay);raw=load_predictions(folder/f'bank{bi:04d}_d{delay:02d}.npz');cell=cells[delay];assert cell['head_order']==list(hh) and raw['predictions'].shape==(len(hh),11,2000)
   for rep in range(-1,10):
    st=pre.clone()
    if rep>=0:
     noise=np.random.default_rng(np.random.SeedSequence([int(d['seed']),20000,rep])).normal(size=tuple(pre.shape)).astype(np.float32);st=torch.clamp(pre+torch.from_numpy(noise)*sigma,min=0)
    for _ in range(4):st=(1-model.alpha)*st+model.alpha*torch.relu(model.input(et)+st@w)
    if rep==-1:st=clean
    pred=explicit_predict(st.numpy(),hh);np.testing.assert_array_equal(pred,raw['predictions'][:,rep+1]);vectors+=len(hh);np.testing.assert_array_equal(model.readout(st).argmax(1).numpy(),raw['original'][rep+1])
    if rep in [-1,0]:
     for i,head in enumerate(hh.values()):np.testing.assert_array_equal(head.predict(st[:32].numpy()),pred[i,:32]);probe_checks+=1
   for c in range(2):
    st=torch.zeros_like(pre)
    if c:
     noise=np.random.default_rng(np.random.SeedSequence([int(d['seed']),20000,0])).normal(size=tuple(pre.shape)).astype(np.float32);st=torch.clamp(torch.from_numpy(noise)*sigma,min=0)
    for _ in range(4):st=(1-model.alpha)*st+model.alpha*torch.relu(model.input(et)+st@w)
    np.testing.assert_array_equal(explicit_predict(st.numpy(),hh),raw['reset'][:,c])
   np.testing.assert_array_equal((raw['predictions']==y).mean(-1),cell['scores']);np.testing.assert_array_equal((raw['original']==y).mean(-1),cell['original_scores']);np.testing.assert_allclose(permutation_means(raw,y,yi,ri),cell['negative_means'],rtol=0,atol=1e-15);checked+=1
  if bi%20==19:print('audit stage',j,'part',part_id,'bank',bi,flush=True)
 assert before==frozen_snapshot(model,{**refs,**h});verify_lock();assert checked==len(indices(j,part_id))*len(DELAYS) and vectors==len(indices(j,part_id))*(len(HEADS)*len(DELAYS)+len(REFS))*11;save(root/f'independent_audit_part{part_id}.json',dict(passed=True,cells=checked,prediction_vectors=vectors,original_probe_slice_checks=probe_checks))

def main(j):
 ps=[subprocess.Popen([sys.executable,__file__,str(j),str(p)]) for p in [0,1]];codes=[p.wait() for p in ps];assert codes==[0,0],codes;root=stage(j);rs=[json.loads((root/f'independent_audit_part{p}.json').read_text()) for p in [0,1]];assert all(x['passed'] for x in rs);a,b=bounds(j);assert sum(x['cells'] for x in rs)==(b-a)*len(DELAYS);save(root/'independent_audit.json',dict(passed=True,cells=sum(x['cells'] for x in rs),prediction_vectors=sum(x['prediction_vectors'] for x in rs),original_probe_slice_checks=sum(x['original_probe_slice_checks'] for x in rs),all_noise_reconstructed=True,recurrence_rebuilt_from_pixels=True,frozen_parameters_verified=True,data_audit_sha256=sha(root/'independent_data_audit.json')))
if __name__=='__main__':
 if len(sys.argv)==3:part(int(sys.argv[1]),int(sys.argv[2]))
 else:main(int(sys.argv[1]))
