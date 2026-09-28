"""Canonical Probe evaluation of all registered stage cells."""
import sys,json,time,subprocess
import numpy as np
import torch
from unittest.mock import patch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from branch_bi4_common import *
from branch_bi_codec import load_bank,save_predictions,save_gzip_json,load_gzip_json
from branch_b_round1 import continue_test
from branch_bt_round1 import encoded_pairs,extract,perturb,check_forward,permutation_indices,frozen_snapshot

@torch.no_grad()
def part(j,part_id):
 torch.set_num_threads(1);verify_lock();root=stage(j);assert json.loads((root/'independent_data_audit.json').read_text())['passed'];folder=root/'seed301';folder.mkdir(exist_ok=True);model,refs,sigma,inputs=load_core(301);h=heads();before=frozen_snapshot(model,{**refs,**h});inputs.update({str(p):sha(p) for p in [R/'training_lock.json',R/'seed301/heads.pkl',root/'data_manifest.json',root/'decision.json']});checks=[]
 save(folder/f'evaluation_manifest_part{part_id}.json',dict(inputs=inputs,before=before,started_at=time.time()))
 with patch.object(StandardScaler,'fit',side_effect=RuntimeError('test forbids fit')),patch.object(LogisticRegression,'fit',side_effect=RuntimeError('test forbids fit')):
  for bi in indices(j,part_id):
   d=load_bank(root/f'test{bi:04d}.npz');es,et=encoded_pairs(model,d);y=d['y'];yi,ri=permutation_indices(d);rows=[]
   for delay in DELAYS:
    hh=active(h,refs,delay);pre,end=extract(model,es,et,delay);checks.append(check_forward(model,d,es,et,delay,pre,end));states=[end]+[perturb(model,pre,et,int(d['seed']),rep,sigma)[1] for rep in range(10)];pred=np.array([[head.predict(st.numpy()) for st in states] for head in hh.values()],dtype=np.int8);original=np.array([model.readout(st).argmax(1).numpy() for st in states],dtype=np.int8);rs=[continue_test(model,torch.zeros_like(pre),et),perturb(model,torch.zeros_like(pre),et,int(d['seed']),0,sigma)[1]];reset=np.array([[head.predict(st.numpy()) for st in rs] for head in hh.values()],dtype=np.int8);neg=np.stack([(reset==y).mean(-1),(pred[:,[0,1],None,:]==y[yi]).mean((-1,-2)),(pred[:,[0,1],:][:,:,ri]==y).mean((-1,-2))],-1)
    rows.append(dict(bank=bi,delay=delay,head_order=list(hh),scores=(pred==y).mean(-1).tolist(),negative_means=neg.tolist(),original_scores=(original==y).mean(-1).tolist()));save_predictions(folder/f'bank{bi:04d}_d{delay:02d}.npz',predictions=pred,original=original,reset=reset)
   save_gzip_json(folder/f'bank{bi:04d}_scores.json.gz',rows)
   if bi%20==19:print('evaluate stage',j,'part',part_id,'bank',bi,flush=True)
 assert before==frozen_snapshot(model,{**refs,**h});assert all(sha(p)==v for p,v in inputs.items());verify_lock();save_gzip_json(folder/f'results_part{part_id}.json.gz',dict(cells_count=len(indices(j,part_id))*len(DELAYS),forward_checks=checks,snapshot=before,frozen_verified=True,fit_guard=True))

def main(j):
 ps=[subprocess.Popen([sys.executable,__file__,str(j),str(p)]) for p in [0,1]];codes=[p.wait() for p in ps];assert codes==[0,0],codes;folder=stage(j)/'seed301';rs=[load_gzip_json(folder/f'results_part{p}.json.gz') for p in [0,1]];ms=[json.loads((folder/f'evaluation_manifest_part{p}.json').read_text()) for p in [0,1]];assert rs[0]['snapshot']==rs[1]['snapshot'] and ms[0]['inputs']==ms[1]['inputs'];a,b=bounds(j);assert sum(x['cells_count'] for x in rs)==(b-a)*len(DELAYS)
 save(folder/'evaluation_manifest.json',dict(inputs=ms[0]['inputs'],before=ms[0]['before'],started_at=min(x['started_at'] for x in ms)));save_gzip_json(folder/'results.json.gz',dict(cells_count=(b-a)*len(DELAYS),forward_checks=sum([x['forward_checks'] for x in rs],[]),snapshot=rs[0]['snapshot'],frozen_verified=True,fit_guard=True,bank_files={str(folder/f'bank{i:04d}_scores.json.gz'):sha(folder/f'bank{i:04d}_scores.json.gz') for i in range(a,b)}))
if __name__=='__main__':
 if len(sys.argv)==3:part(int(sys.argv[1]),int(sys.argv[2]))
 else:main(int(sys.argv[1]))
