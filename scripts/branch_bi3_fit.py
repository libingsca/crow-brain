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
from branch_bi_codec import load_bank
ROUND=Path('artifacts/branch_bi/round3');ROOT=ROUND/('batch'+os.environ.get('BI_BATCH','1'));CFG=json.loads((ROOT/'config.json').read_text());HEADS=list(CFG['heads']);N_BANKS=CFG['banks']

def verify_lock():
 g=json.loads((ROUND/'global_training_lock.json').read_text());assert all(sha(p)==h for p,h in g['files'].items());m=json.loads((ROOT/'training_lock.json').read_text());assert all(sha(p)==h for p,h in m['files'].items());return m

@torch.no_grad()
def raw_states(model,data,delays,sigma):
 es,et=encoded_pairs(model,data);raw={};checks=[]
 for d in delays:
  pre,end=extract(model,es,et,d);checks.append(check_forward(model,data,es,et,d,pre,end));raw[d]={'pre':pre.numpy(),'end':end.numpy(),'noise_end':np.stack([perturb(model,pre,et,int(data['seed']),r,sigma)[1].numpy() for r in range(4)])}
 return raw,checks

@torch.no_grad()
def train(seed):
 torch.set_num_threads(1);assert not list(ROOT.glob('test*.npz'));assert json.loads(Path('artifacts/branch_bi/round1/history_audit.json').read_text())['passed'];start=time.monotonic();folder=ROOT/f'seed{seed}';folder.mkdir(exist_ok=False);model,refs,sigma,inputs=load_core(seed)
 inputs.update({CFG[k]:sha(CFG[k]) for k in ['fit_path','validation_path']});inputs[str(ROOT/'config.json')]=sha(ROOT/'config.json');inputs[str(ROOT/'protocol.md')]=sha(ROOT/'protocol.md');before=frozen_snapshot(model,refs)
 data=load_bank(CFG['fit_path']);raw,checks=raw_states(model,data,DELAYS,sigma)
 if CFG['reuse_baseline']:
  p=Path(f'artifacts/branch_bg/round1/seed{seed}/heads.pkl');inputs[str(p)]=sha(p)
  with p.open('rb') as f:base=pickle.load(f)['joint']
  scaler=base.scaler;heads={'ten':base};r=json.loads(Path(f'artifacts/branch_bg/round1/seed{seed}/fits.json').read_text())['heads']['joint'];records={'ten':dict(history=r['history'],reused=True)}
 else:
  scaler=StandardScaler().fit(np.concatenate([raw[d]['end'] for d in BASE_DELAYS]).astype(np.float64));heads={};records={}
 save(folder/'training_manifest.json',{'inputs':inputs,'started_at':time.time(),'sigma':sigma,'before':before})
 for name,delays in CFG['heads'].items():
  if name in heads:continue
  x,y,w=pooled_pack(raw,data['y'],'cover_split',delays);xx=scaler.transform(x.astype(np.float64));del x
  clf,hist=fit_candidate(xx,y,w,10000.,'worst',groups=training_groups(len(data['y']),delays,'cover_split'),tau=.005);heads[name]=Probe(scaler,clf);records[name]={'history':hist,'reused':False};del xx,y,w
  print(seed,name,'converged',hist[-1]['gradient_inf'],flush=True)
 del raw
 vd=load_bank(CFG['validation_path']);raw,vchecks=raw_states(model,vd,DELAYS,sigma);x,y,w=pooled_pack(raw,vd['y'],'cover_split',DELAYS);xx=scaler.transform(x.astype(np.float64));del raw,x
 for h in HEADS:records[h]['validation']=condition_scores(heads[h].clf,xx,y,w,DELAYS)
 assert frozen_snapshot(model,refs)==before and all(sha(p)==h for p,h in inputs.items())
 with (folder/'heads.pkl').open('wb') as f:pickle.dump(heads,f)
 save(folder/'fits.json',{'heads':records,'finished_at':time.time(),'seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'snapshot':frozen_snapshot(model,{**refs,**heads}),'forward_checks':checks+vchecks})

@torch.no_grad()
def audit_train():
 torch.set_num_threads(1);assert not list(ROOT.glob('test*.npz'));files={};checked=0;maxgrad=0
 for seed in [301]:
  model,refs,sigma,_=load_core(seed);folder=ROOT/f'seed{seed}';r=json.loads((folder/'fits.json').read_text());m=json.loads((folder/'training_manifest.json').read_text());assert all(sha(p)==v for p,v in m['inputs'].items());files.update(m['inputs'])
  with (folder/'heads.pkl').open('rb') as f:heads=pickle.load(f)
  assert list(heads)==HEADS and all(h.scaler is heads[HEADS[0]].scaler for h in heads.values());assert frozen_snapshot(model,{**refs,**heads})==r['snapshot'];scaler=heads[HEADS[0]].scaler;acc={h:{'loss':[],'grad':[]} for h in HEADS};pool=[]
  if CFG['reuse_baseline']:
   with open(f'artifacts/branch_bg/round1/seed{seed}/heads.pkl','rb') as f:base=pickle.load(f)['joint']
   assert frozen_snapshot(model,{'ten':base})==frozen_snapshot(model,{'ten':heads['ten']})
  for part in ['fit','validation']:
   data=load_bank(CFG[part+'_path']);es,et=encoded_pairs(model,data);y=data['y']
   for delay in DELAYS:
    pre,end=extract(model,es,et,delay);check_forward(model,data,es,et,delay,pre,end)
    if part=='fit' and delay in BASE_DELAYS:pool.append(end.numpy())
    noisy=np.concatenate([perturb(model,pre,et,int(data['seed']),rep,sigma)[1].numpy() for rep in range(4)])
    for condition,state in [('clean',end.numpy()),('noise',noisy)]:
     x=scaler.transform(state.astype(np.float64));yy=np.tile(y,1 if condition=='clean' else 4)
     for h,head in heads.items():
      z=x@head.clf.coef_[0]+head.clf.intercept_[0];loss=float(np.mean(np.logaddexp(0,z)-yy*z));score=float(np.mean((z>0)==yy))
      if part=='fit' and delay in CFG['heads'][h]:e=expit(z)-yy;acc[h]['loss'].append(loss);acc[h]['grad'].append(np.r_[x.T@e/len(yy),e.mean()])
      elif part=='validation':np.testing.assert_allclose([loss,score],[r['heads'][h]['validation'][f'{delay}_{condition}'][k] for k in ['loss','ba']],atol=1e-12)
  p=np.concatenate(pool).astype(np.float64);np.testing.assert_allclose(scaler.mean_,p.mean(0),atol=1e-12);np.testing.assert_allclose(scaler.var_,p.var(0),atol=1e-10);assert scaler.n_samples_seen_==40000
  for h in HEADS:
   ls=np.array(acc[h]['loss']);q=np.exp(ls/.005-logsumexp(ls/.005));beta=heads[h].clf.coef_[0];g=q@np.array(acc[h]['grad']);g[:-1]+=beta/(10000*8000);gn=float(np.abs(g).max());obj=.005*(logsumexp(ls/.005)-np.log(len(ls)))+beta@beta/(2*10000*8000);row=r['heads'][h]['history'][-1];assert gn<=1e-6 and row['passed'];np.testing.assert_allclose([obj,gn],[row['objective'],row['gradient_inf']],atol=1e-10);np.testing.assert_allclose(q,row['group_weights'],atol=1e-10);checked+=1;maxgrad=max(maxgrad,gn)
  for p in folder.glob('*'):files[str(p)]=sha(p)
  print('audited training',seed,flush=True)
 save(ROOT/'training_audit.json',{'passed':True,'heads':checked,'max_gradient':maxgrad,'scaler_validation_forward_verified':True})
 for p in list(Path('scripts').glob('branch_bi*.py'))+[ROOT/'config.json',ROOT/'protocol.md',ROOT/'training_audit.json']:files[str(p)]=sha(p)
 save(ROOT/'training_lock.json',{'files':files,'locked_at':time.time(),'test_absent':True})


def configure(b):
 global ROOT,CFG,HEADS,N_BANKS
 ROOT=ROUND/f'batch{b}';CFG=json.loads((ROOT/'config.json').read_text());HEADS=list(CFG['heads']);N_BANKS=CFG['banks']

if __name__=='__main__':
 import sys
 assert json.loads((ROUND/'fit_data_audit.json').read_text())['passed']
 assert not list(ROUND.glob('test*.npz'))
 for b in range(1,97):
  configure(b)
  if sys.argv[1]=='train':
   if not (ROOT/'seed301/fits.json').exists():train(301)
  elif sys.argv[1]=='audit':
   if not (ROOT/'training_lock.json').exists():audit_train()
  else:raise ValueError(sys.argv[1])
  print('completed',sys.argv[1],b,'/96',flush=True)
