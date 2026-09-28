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
from branch_bi_codec import load_bank,save_predictions,save_gzip_json,load_gzip_json
ROOT=Path('artifacts/branch_bi/round3');CFG=json.loads((ROOT/'config.json').read_text());N_BANKS=CFG['banks'];HEADS=[f'b{i}' for i in range(1,CFG['K']+1)];DELAYS=list(range(5,16));REFERENCE_DELAYS=[5,7,8,10,11,12,13,15]

def verify_lock():
 m=json.loads((ROOT/'training_lock.json').read_text());assert all(sha(p)==h for p,h in m['files'].items());return m

def lock_heads():
 assert not list(ROOT.glob('test*.npz'));folder=ROOT/'seed301';folder.mkdir(exist_ok=False);files={};heads={}
 for b in range(1,CFG['K']+1):
  p=ROOT/f'batch{b}/training_lock.json';m=json.loads(p.read_text());assert all(sha(p)==h for p,h in m['files'].items());files.update(m['files']);files[str(p)]=sha(p)
  with (ROOT/f'batch{b}/seed301/heads.pkl').open('rb') as f:heads[f'b{b}']=pickle.load(f)['eleven']
 with (folder/'heads.pkl').open('wb') as f:pickle.dump(heads,f)
 for p in [ROOT/'config.json',ROOT/'protocol.md',folder/'heads.pkl',ROOT/'fit_data_audit.json',ROOT/'execution_code_manifest.json',Path('artifacts/branch_bi/round2/evidence_manifest.json')]+list(Path('scripts').glob('branch_bi3*.py'))+[Path('scripts/branch_bi_codec.py')]:files[str(p)]=sha(p)
 save(ROOT/'training_lock.json',dict(files=files,locked_at=time.time(),new_heads=len(heads),core=301,test_absent=True))
 print('Locked all 96 fresh heads before testing',flush=True)

def active_heads(heads,refs,delay):
 a=dict(heads)
 if delay in REFERENCE_DELAYS:a['reference']=refs[f'ref{delay}']
 return a

@torch.no_grad()
def evaluate_part(part):
 seed=301
 torch.set_num_threads(1);verify_lock();folder=ROOT/f'seed{seed}';start=time.monotonic();assert not (folder/f'results_part{part}.json.gz').exists();model,refs,sigma,inputs=load_core(seed)
 with (folder/'heads.pkl').open('rb') as f:heads=pickle.load(f)
 before=frozen_snapshot(model,{**refs,**heads});inputs[str(folder/'heads.pkl')]=sha(folder/'heads.pkl');inputs[str(ROOT/'training_lock.json')]=sha(ROOT/'training_lock.json');cells=[];checks=[]
 save(folder/f'evaluation_manifest_part{part}.json',{'inputs':inputs,'before':before,'started_at':time.time()})
 with patch.object(LogisticRegression,'fit',side_effect=RuntimeError('test forbids fit')),patch.object(StandardScaler,'fit',side_effect=RuntimeError('test forbids fit')):
  for bi in bank_indices(part,N_BANKS):
   cells=[]
   d=load_bank(ROOT/f'test{bi:04d}.npz');es,et=encoded_pairs(model,d);y=d['y'];yi,ri=permutation_indices(d)
   for delay in DELAYS:
    active=active_heads(heads,refs,delay);pre,end=extract(model,es,et,delay);checks.append(check_forward(model,d,es,et,delay,pre,end));states=[end]+[perturb(model,pre,et,int(d['seed']),rep,sigma)[1] for rep in range(10)]
    pred=np.asarray([[h.predict(s.numpy()) for s in states] for h in active.values()],dtype=np.int8);original=np.asarray([model.readout(s).argmax(1).numpy() for s in states],dtype=np.int8)
    resetstates=[continue_test(model,torch.zeros_like(pre),et),perturb(model,torch.zeros_like(pre),et,int(d['seed']),0,sigma)[1]];reset=np.asarray([[h.predict(s.numpy()) for s in resetstates] for h in active.values()],dtype=np.int8)
    neg=np.stack([(reset==y).mean(-1),(pred[:,[0,1],None,:]==y[yi]).mean((-1,-2)),(pred[:,[0,1],:][:,:,ri]==y).mean((-1,-2))],axis=-1)
    cells.append({'bank':bi,'delay':delay,'head_order':list(active),'scores':(pred==y).mean(-1).tolist(),'negative_means':neg.tolist(),'original_scores':(original==y).mean(-1).tolist()})
    save_predictions(folder/f'bank{bi:04d}_d{delay:02d}.npz',predictions=pred,original=original,reset=reset)
   save_gzip_json(folder/f'bank{bi:04d}_scores.json.gz',cells)
   if bi%10==9:print('part',part,'evaluated bank',bi,flush=True)
 assert before==frozen_snapshot(model,{**refs,**heads});assert all(sha(p)==h for p,h in inputs.items());verify_lock()
 save_gzip_json(folder/f'results_part{part}.json.gz',{'cells_count':len(bank_indices(part,N_BANKS))*len(DELAYS),'forward_checks':checks,'snapshot':before,'frozen_verified':True,'fit_guard':True,'seconds':time.monotonic()-start})



def bank_indices(part,n):
 assert part in [0,1]
 return range(part*n//2,(part+1)*n//2)

def evaluate():
 import subprocess,sys
 children=[subprocess.Popen([sys.executable,__file__,'evaluate_part','--part',str(p)]) for p in [0,1]]
 codes=[c.wait() for c in children];assert codes==[0,0],codes
 folder=ROOT/'seed301';rs=[load_gzip_json(folder/f'results_part{p}.json.gz') for p in [0,1]];ms=[json.loads((folder/f'evaluation_manifest_part{p}.json').read_text()) for p in [0,1]]
 assert rs[0]['snapshot']==rs[1]['snapshot'] and ms[0]['inputs']==ms[1]['inputs'];assert sum(r['cells_count'] for r in rs)==N_BANKS*len(DELAYS)
 save(folder/'evaluation_manifest.json',dict(inputs=ms[0]['inputs'],before=ms[0]['before'],started_at=min(x['started_at'] for x in ms)))
 save_gzip_json(folder/'results.json.gz',dict(cells_count=N_BANKS*len(DELAYS),forward_checks=sum([x['forward_checks'] for x in rs],[]),snapshot=rs[0]['snapshot'],frozen_verified=True,fit_guard=True,bank_files={str(folder/f'bank{i:04d}_scores.json.gz'):sha(folder/f'bank{i:04d}_scores.json.gz') for i in range(N_BANKS)},part_seconds=[r['seconds'] for r in rs]))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['lock','evaluate','evaluate_part']);p.add_argument('--part',type=int);a=p.parse_args()
 try:
  if a.phase=='lock':lock_heads()
  elif a.phase=='evaluate_part':evaluate_part(a.part)
  else:evaluate()
 except Exception as e:save(ROOT/f'failure_{a.phase}_{int(time.time())}.json',{'error':repr(e)});raise
