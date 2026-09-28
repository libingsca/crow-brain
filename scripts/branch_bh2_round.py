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
ROOT=Path('artifacts/branch_bh')/('round'+os.environ.get('BH_ROUND','2'));CFG=json.loads((ROOT/'config.json').read_text());HEADS=list(CFG['heads']);N_BANKS=CFG['banks']

def verify_lock():
 m=json.loads((ROOT/'training_lock.json').read_text());assert all(sha(p)==h for p,h in m['files'].items());return m

@torch.no_grad()
def raw_states(model,data,delays,sigma):
 es,et=encoded_pairs(model,data);raw={};checks=[]
 for d in delays:
  pre,end=extract(model,es,et,d);checks.append(check_forward(model,data,es,et,d,pre,end));raw[d]={'pre':pre.numpy(),'end':end.numpy(),'noise_end':np.stack([perturb(model,pre,et,int(data['seed']),r,sigma)[1].numpy() for r in range(4)])}
 return raw,checks

@torch.no_grad()
def train(seed):
 torch.set_num_threads(1);assert not list(ROOT.glob('test*.npz'));assert json.loads(Path('artifacts/branch_bh/round1/history_audit.json').read_text())['passed'];start=time.monotonic();folder=ROOT/f'seed{seed}';folder.mkdir(exist_ok=False);model,refs,sigma,inputs=load_core(seed)
 inputs.update({CFG[k]:sha(CFG[k]) for k in ['fit_path','validation_path']});inputs[str(ROOT/'config.json')]=sha(ROOT/'config.json');inputs[str(ROOT/'protocol.md')]=sha(ROOT/'protocol.md');before=frozen_snapshot(model,refs)
 data=dict(np.load(CFG['fit_path']));raw,checks=raw_states(model,data,DELAYS,sigma)
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
 vd=dict(np.load(CFG['validation_path']));raw,vchecks=raw_states(model,vd,DELAYS,sigma);x,y,w=pooled_pack(raw,vd['y'],'cover_split',DELAYS);xx=scaler.transform(x.astype(np.float64));del raw,x
 for h in HEADS:records[h]['validation']=condition_scores(heads[h].clf,xx,y,w,DELAYS)
 assert frozen_snapshot(model,refs)==before and all(sha(p)==h for p,h in inputs.items())
 with (folder/'heads.pkl').open('wb') as f:pickle.dump(heads,f)
 save(folder/'fits.json',{'heads':records,'finished_at':time.time(),'seconds':time.monotonic()-start,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'snapshot':frozen_snapshot(model,{**refs,**heads}),'forward_checks':checks+vchecks})

@torch.no_grad()
def audit_train():
 torch.set_num_threads(1);assert not list(ROOT.glob('test*.npz'));files={};checked=0;maxgrad=0
 for seed in range(301,306):
  model,refs,sigma,_=load_core(seed);folder=ROOT/f'seed{seed}';r=json.loads((folder/'fits.json').read_text());m=json.loads((folder/'training_manifest.json').read_text());assert all(sha(p)==v for p,v in m['inputs'].items());files.update(m['inputs'])
  with (folder/'heads.pkl').open('rb') as f:heads=pickle.load(f)
  assert list(heads)==HEADS and all(h.scaler is heads[HEADS[0]].scaler for h in heads.values());assert frozen_snapshot(model,{**refs,**heads})==r['snapshot'];scaler=heads[HEADS[0]].scaler;acc={h:{'loss':[],'grad':[]} for h in HEADS};pool=[]
  if CFG['reuse_baseline']:
   with open(f'artifacts/branch_bg/round1/seed{seed}/heads.pkl','rb') as f:base=pickle.load(f)['joint']
   assert frozen_snapshot(model,{'ten':base})==frozen_snapshot(model,{'ten':heads['ten']})
  for part in ['fit','validation']:
   data=dict(np.load(CFG[part+'_path']));es,et=encoded_pairs(model,data);y=data['y']
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
 for p in list(Path('scripts').glob('branch_bh*.py'))+[ROOT/'config.json',ROOT/'protocol.md',ROOT/'training_audit.json']:files[str(p)]=sha(p)
 save(ROOT/'training_lock.json',{'files':files,'locked_at':time.time(),'test_absent':True})

def prepare(fitting=False):
 if not fitting:lock=verify_lock()
 prior=json.loads(Path('artifacts/branch_bg/round1/data_manifest.json').read_text())['prior_files'];prior.update({str(p):sha(p) for p in Path('artifacts/branch_bg/round1').glob('test*.npz')})
 for ri in range(1,CFG['round']):
  prior.update({str(p):sha(p) for p in (ROOT.parent/f'round{ri}').glob('test*.npz')})
 seen=set()
 for p,h in prior.items():
  assert sha(p)==h
  with np.load(p) as a:
   for key in ['images'] if 'images' in a else ['solid','ring']:seen.update(digest(x) for x in a[key].reshape(-1,1,32,32))
 assert len(seen)==CFG['prior_images']
 if not fitting and not CFG['reuse_baseline']:
  for p in [Path(CFG[k]) for k in ['fit_path','validation_path']]:
   prior[str(p)]=sha(p)
   with np.load(p) as a:
    hs={digest(x) for x in a['images'].reshape(-1,1,32,32)};assert len(hs)==5000 and not seen.intersection(hs);seen.update(hs)
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
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','audit_train','prepare','prepare_fit','evaluate']);p.add_argument('--seed',type=int);a=p.parse_args()
 try:
  if a.phase=='train':train(a.seed)
  elif a.phase=='audit_train':audit_train()
  elif a.phase in ['prepare','prepare_fit']:prepare(a.phase=='prepare_fit')
  else:evaluate(a.seed)
 except Exception as e:save(ROOT/f'failure_{a.phase}_{a.seed}_{int(time.time())}.json',{'error':repr(e)});raise
