"""B-G fixed factorial intervention; immutable B-F development inputs, fresh tests."""
import argparse,json,pickle,time,resource
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
from branch_bf_round1 import load_core,pooled_pack,training_groups,condition_scores,TRAIN_DELAYS,BASE_DELAYS,DELAYS,REFERENCE_DELAYS
from branch_bg_solver import fit_candidate
ROOT=Path('artifacts/branch_bg/round1');OLD=Path('artifacts/branch_bf/round1')
ARMS={'baseline':(100.,.05),'weak':(10000.,.05),'sharp':(100.,.005),'joint':(10000.,.005)};HEADS=list(ARMS)

def verify_lock():
 m=json.loads((ROOT/'training_lock.json').read_text())
 assert all(sha(p)==h for p,h in m['files'].items())
 return m

@torch.no_grad()
def train(seed):
 torch.set_num_threads(1);assert json.loads((ROOT/'history_audit.json').read_text())['passed'];assert not list(ROOT.glob('test*.npz'));start=time.monotonic()
 folder=ROOT/f'seed{seed}';folder.mkdir(exist_ok=False);src=OLD/f'seed{seed}';model,refs,sigma,inputs=load_core(seed)
 paths=[src/'heads.pkl',src/'fits.json',OLD/'fit.npz',OLD/'validation.npz']+[src/f'{part}_d{d}.npz' for part in ['fit','validation'] for d in TRAIN_DELAYS]
 inputs.update({str(p):sha(p) for p in paths});save(folder/'training_manifest.json',{'inputs':inputs,'started_at':time.time(),'sigma':sigma})
 with (src/'heads.pkl').open('rb') as f:base=pickle.load(f)['cover_split']
 assert json.loads((src/'fits.json').read_text())['heads']['cover_split']['C']==100
 before=frozen_snapshot(model,{'baseline':base,**refs});raw={d:dict(np.load(src/f'fit_d{d}.npz')) for d in TRAIN_DELAYS};y=np.load(OLD/'fit.npz')['y']
 x,yy,w=pooled_pack(raw,y,'cover_split',TRAIN_DELAYS);del raw
 xx=base.scaler.transform(x.astype(np.float64));del x
 heads={'baseline':base};records={}
 oldrow=next(v for v in json.loads((src/'fits.json').read_text())['heads']['cover_split']['grid'] if v['C']==100)
 records['baseline']={'C':100.,'tau':.05,'history':oldrow['history'],'reused':True}
 for name,(c,tau) in list(ARMS.items())[1:]:
  t=time.monotonic();clf,hist=fit_candidate(xx,yy,w,c,'worst',groups=training_groups(len(y),TRAIN_DELAYS,'cover_split'),tau=tau)
  heads[name]=Probe(base.scaler,clf);records[name]={'C':c,'tau':tau,'history':hist,'reused':False,'seconds':time.monotonic()-t}
  save(folder/'progress.json',records);print(seed,name,'gradient',hist[-1]['gradient_inf'],'seconds',time.monotonic()-t,flush=True)
 del xx,yy,w
 raw={d:dict(np.load(src/f'validation_d{d}.npz')) for d in TRAIN_DELAYS};vy=np.load(OLD/'validation.npz')['y'];vx,vy,vw=pooled_pack(raw,vy,'cover_split',TRAIN_DELAYS);vv=base.scaler.transform(vx.astype(np.float64));del raw,vx
 for name,head in heads.items():records[name]['validation']=condition_scores(head.clf,vv,vy,vw,TRAIN_DELAYS)
 assert before==frozen_snapshot(model,{'baseline':base,**refs});assert all(sha(p)==v for p,v in inputs.items())
 with (folder/'heads.pkl').open('wb') as f:pickle.dump(heads,f)
 save(folder/'fits.json',{'heads':records,'finished_at':time.time(),'seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'snapshot':frozen_snapshot(model,{**refs,**heads})})

@torch.no_grad()
def training_audit():
 torch.set_num_threads(1);assert not list(ROOT.glob('test*.npz'));checked=0;maxgrad=0;files={};forward=[]
 for seed in range(301,306):
  folder=ROOT/f'seed{seed}';src=OLD/f'seed{seed}';fits=json.loads((folder/'fits.json').read_text());m=json.loads((folder/'training_manifest.json').read_text());assert all(sha(p)==h for p,h in m['inputs'].items());files.update(m['inputs'])
  model,refs,sigma,inputs=load_core(seed)
  with (folder/'heads.pkl').open('rb') as f:heads=pickle.load(f)
  with (src/'heads.pkl').open('rb') as f:base=pickle.load(f)['cover_split']
  assert frozen_snapshot(model,{'baseline':base})==frozen_snapshot(model,{'baseline':heads['baseline']})
  assert frozen_snapshot(model,{**refs,**heads})==fits['snapshot']
  assert all(h.scaler is heads['baseline'].scaler for h in heads.values())
  pool=np.concatenate([np.load(src/f'fit_d{d}.npz')['end'] for d in BASE_DELAYS]).astype(np.float64);scaler=heads['baseline'].scaler
  np.testing.assert_allclose(scaler.mean_,pool.mean(0),atol=1e-12);np.testing.assert_allclose(scaler.var_,pool.var(0),atol=1e-10);assert scaler.n_samples_seen_==40000;del pool
  accum={h:{'loss':[],'grad':[]} for h in HEADS}
  for part in ['fit','validation']:
   data=dict(np.load(OLD/f'{part}.npz'));es,et=encoded_pairs(model,data);y=data['y']
   for delay in TRAIN_DELAYS:
    raw=dict(np.load(src/f'{part}_d{delay}.npz'));pre,end=extract(model,es,et,delay)
    np.testing.assert_allclose(raw['pre'],pre.numpy(),rtol=1e-6,atol=1e-6);np.testing.assert_allclose(raw['end'],end.numpy(),rtol=1e-6,atol=1e-6)
    forward.append(check_forward(model,data,es,et,delay,pre,end))
    for rep in range(4):np.testing.assert_allclose(raw['noise_end'][rep],perturb(model,pre,et,int(data['seed']),rep,sigma)[1].numpy(),rtol=1e-6,atol=1e-6)
    for ci,(condition,states) in enumerate([('clean',raw['end']),('noise',raw['noise_end'].reshape(-1,64))]):
     x=scaler.transform(states.astype(np.float64));yy=np.tile(y,1 if ci==0 else 4)
     for h,head in heads.items():
      z=x@head.clf.coef_[0]+head.clf.intercept_[0];loss=float(np.mean(np.logaddexp(0,z)-yy*z));score=float(np.mean((z>0)==yy))
      if part=='fit':
       residual=expit(z)-yy;accum[h]['loss'].append(loss);accum[h]['grad'].append(np.r_[x.T@residual/len(yy),residual.mean()])
      else:np.testing.assert_allclose([loss,score],[fits['heads'][h]['validation'][f'{delay}_{condition}'][k] for k in ['loss','ba']],atol=1e-12)
  for h,(c,tau) in ARMS.items():
   ls=np.array(accum[h]['loss']);q=np.exp(ls/tau-logsumexp(ls/tau));beta=heads[h].clf.coef_[0];g=q@np.array(accum[h]['grad']);g[:-1]+=beta/(c*8000);grad=float(np.abs(g).max());obj=tau*(logsumexp(ls/tau)-np.log(20))+beta@beta/(2*c*8000)
   row=fits['heads'][h]['history'][-1];assert grad<=1e-6 and row['passed'];np.testing.assert_allclose([obj,grad],[row['objective'],row['gradient_inf']],atol=1e-10);np.testing.assert_allclose(q,row['group_weights'],atol=1e-10);maxgrad=max(maxgrad,grad);checked+=1
  for p in folder.glob('*'):files[str(p)]=sha(p)
  print('training audited',seed,flush=True)
 assert checked==20
 save(ROOT/'training_audit.json',{'passed':True,'heads':20,'new_heads':15,'max_gradient':maxgrad,'cache_noise_scaler_validation_reconstructed':True,'forward_checks':len(forward)})
 for p in list(Path('scripts').glob('branch_bg*.py'))+[ROOT/'protocol.md',ROOT/'training_audit.json']:files[str(p)]=sha(p)
 save(ROOT/'training_lock.json',{'files':files,'locked_at':time.time(),'heads':20,'new_heads':15,'test_absent':True})

def prepare():
 lock=verify_lock();assert not list(ROOT.glob('test*.npz'));prior=json.loads((OLD/'data_manifest.json').read_text())['prior_files'];prior.update({str(p):sha(p) for p in [OLD/'fit.npz',OLD/'validation.npz']+list(OLD.glob('test*.npz'))})
 seen=set()
 for p,h in prior.items():
  assert sha(p)==h
  with np.load(p) as a:
   for key in ['images'] if 'images' in a else ['solid','ring']:seen.update(digest(x) for x in a[key].reshape(-1,1,32,32))
 assert len(seen)==324000;records=[];rejected=[]
 for bi in range(20):
  seed=1000100+bi;rng=np.random.default_rng(seed);images=np.stack([draw_dots(int(c),rng)[0] for c in COUNTS for _ in range(200)]).reshape(5,200,1,32,32)
  for ci,c in enumerate(COUNTS):
   for j in range(200):
    original=h=digest(images[ci,j]);n=0
    while h in seen:images[ci,j,0]=draw_dots(int(c),rng)[0];h=digest(images[ci,j]);n+=1
    if n:rejected.append({'bank':bi,'count':int(c),'index':j,'draws':n,'original':original,'replacement':h})
    seen.add(h)
  assert np.array_equal([label(x[0]>.499,structure=np.ones((3,3)))[1] for x in images.reshape(-1,1,32,32)],np.repeat(COUNTS,200))
  rng=np.random.default_rng(seed+10000);pairs=balanced_pairs(50);rng.shuffle(pairs);a,b=pairs.T;ia,ib=rng.integers(0,200,(2,2000));same=(a==b)&(ia==ib);ib[same]=(ib[same]+1)%200;y=(a==b).astype(np.int64)
  assert np.bincount(y).tolist()==[1000,1000];assert all(np.bincount(v).tolist()==[400]*5 for v in [a,b]);assert not np.any((a==b)&(ia==ib))
  for aa in range(5):
   for bb in range(5):assert ((a==aa)&(b==bb)).sum()==50*(4 if aa==bb else 1)
  p=ROOT/f'test{bi:02d}.npz';np.savez_compressed(p,images=images,a=a,b=b,ia=ia,ib=ib,y=y,seed=seed);assert p.stat().st_mtime>=lock['locked_at'];records.append({'path':str(p),'sha256':sha(p),'seed':seed})
  yi,ri=permutation_indices({'y':y,'b':b,'seed':seed});np.savez_compressed(ROOT/f'permutations{bi:02d}.npz',label_indices=yi.astype(np.int16),row_indices=ri.astype(np.int16))
  print('prepared',bi,flush=True)
 save(ROOT/'data_manifest.json',{'banks':records,'prior_files':prior,'prior_images':324000,'new_images':20000,'duplicate_rejections':rejected,'locked_before_generation':True})

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
  for bi in range(20):
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
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','audit_train','prepare','evaluate']);p.add_argument('--seed',type=int);a=p.parse_args()
 try:
  if a.phase=='train':train(a.seed)
  elif a.phase=='audit_train':training_audit()
  elif a.phase=='prepare':prepare()
  else:evaluate(a.seed)
 except Exception as e:save(ROOT/f'failure_{a.phase}_{a.seed}_{int(time.time())}.json',{'error':repr(e)});raise
