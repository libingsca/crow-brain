"""Preregistered B-R: frozen cores, delay-specific readouts, locked evaluation."""
import argparse,json,pickle,time,warnings
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from scipy.ndimage import label
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss
from crowbrain.model import VisualRNN
from crowbrain.task import Bank,balanced_pairs
from branch_b_round1 import Probe,sha,digest,save,continue_test
from branch_bt_round1 import encoded_pairs,extract,perturb,check_forward,permutation_indices,frozen_snapshot,previous_data
from branch_bn_round1 import mixture
from diagnose_bn_solver import objective
ROOT=Path('artifacts/branch_br/round1');DELAYS=[5,7,10,13];CS=[.01,.1,1.,10.,100.]

def load_core(seed):
    cp=Path(f'artifacts/round6/final/rnn_lr0.001_seed{seed}/final.pt');hp=Path(f'artifacts/branch_bn/round1/seed{seed}/heads.pkl');mp=hp.with_name('manifest.json')
    model=VisualRNN(normalize=True);model.load_state_dict(torch.load(cp,weights_only=True));model.eval();model.requires_grad_(False)
    with hp.open('rb') as f:old=pickle.load(f)['aug']
    return model,old,json.loads(mp.read_text())['sigma_primary'],{str(p):sha(p) for p in [cp,hp,mp,Path(__file__),ROOT/'protocol.md']}

def verify_lock():
    m=json.loads((ROOT/'training_lock.json').read_text())
    for p,v in m['files'].items():assert sha(p)==v,p
    assert m['new_heads']==40 and m['all_candidates_passed']
    return m

def prepare(test=False):
    if test: verify_lock()
    old=previous_data()+[Path(f'artifacts/branch_bt/round1/test{i:02d}.npz') for i in range(20)]
    seen=set();refs={}
    for p in old:
        with np.load(p) as d:seen.update(digest(im) for im in d['images'].reshape(-1,1,32,32))
        refs[str(p)]=sha(p)
    assert len(seen)==70000
    records={}
    if test:
        records=json.loads((ROOT/'data_manifest.json').read_text())['banks']
        for name in ['fit','validation']:
            p=ROOT/f'{name}.npz';assert sha(p)==records[name]['sha256']
            with np.load(p) as d:seen.update(digest(im) for im in d['images'].reshape(-1,1,32,32))
    configs=[(f'test{i:02d}',930100+i,200,2000) for i in range(20)] if test else [('fit',930001,1000,8000),('validation',930002,1000,2000)]
    for name,seed,k,n in configs:
        p=ROOT/f'{name}.npz';assert not p.exists()
        images=Bank(seed,k).images.numpy();hs={digest(im) for im in images.reshape(-1,1,32,32)}
        assert len(hs)==5*k and not hs.intersection(seen);seen.update(hs)
        counts=[label(im[0]>.499,structure=np.ones((3,3)))[1] for im in images.reshape(-1,1,32,32)]
        assert np.array_equal(counts,np.repeat(np.arange(2,7),k))
        rng=np.random.default_rng(seed+10000);pairs=balanced_pairs(n//40);rng.shuffle(pairs);a,b=pairs.T
        ia,ib=rng.integers(0,k,(2,n));same=(a==b)&(ia==ib);ib[same]=(ib[same]+1)%k;y=(a==b).astype(np.int64)
        assert np.bincount(y).tolist()==[n//2]*2
        assert all(np.bincount(v).tolist()==[n//5]*5 for v in [a,b])
        assert not np.any((a==b)&(ia==ib))
        for aa in range(5):
            for bb in range(5):assert ((a==aa)&(b==bb)).sum()==n//40*(4 if aa==bb else 1)
        np.savez_compressed(p,images=images,a=a,b=b,ia=ia,ib=ib,y=y,seed=seed)
        records[name]={'sha256':sha(p),'seed':seed,'trials':n,'images':5*k,'objects_and_balance_verified':True}
        print('prepared',name,flush=True)
    save(ROOT/'data_manifest.json',{'banks':records,'prior_files':refs,'image_sets_disjoint':True,'prior_unique_images':70000,'test_created_after_all_heads_locked':test})


def fit_candidate(xx,y,w,c):
    history=[]
    for limit in [200,1000]:
        clf=LogisticRegression(C=c,solver='newton-cholesky',tol=1e-7,max_iter=limit)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');clf.fit(xx,y,sample_weight=w)
        obj,grad=objective(xx,y,w,clf,c)
        ok=not any(issubclass(z.category,ConvergenceWarning) for z in caught) and grad<=1e-6
        history.append({'limit':limit,'iterations':clf.n_iter_.tolist(),'warnings':[str(z.message) for z in caught],'objective':obj,'gradient_inf':grad,'passed':ok})
        if ok:return clf,history
    raise RuntimeError(f'Unresolved candidate C={c}: {history}')

@torch.no_grad()
def train(seed):
    assert not list(ROOT.glob('test*.npz'))
    torch.set_num_threads(1);start=time.monotonic();folder=ROOT/f'seed{seed}';folder.mkdir(exist_ok=False)
    model,old,sigma,inputs=load_core(seed);before=frozen_snapshot(model,{'old':old})
    inputs.update({str(ROOT/f'{p}.npz'):sha(ROOT/f'{p}.npz') for p in ['fit','validation']})
    save(folder/'training_manifest.json',{'seed':seed,'sigma_primary':sigma,'input_sha256':inputs,'before':before,'started_at':time.time()})
    data={p:dict(np.load(ROOT/f'{p}.npz')) for p in ['fit','validation']};encoded={p:encoded_pairs(model,d) for p,d in data.items()}
    heads={};candidates={};records={};audits=[]
    for delay in DELAYS:
        packs={}
        for part,d in data.items():
            es,et=encoded[part];pre,end=extract(model,es,et,delay)
            audits.append({'part':part,'delay':delay,'errors':check_forward(model,d,es,et,delay,pre,end)})
            clean=end.numpy();noisy=[perturb(model,pre,et,int(d['seed']),r,sigma)[1].numpy() for r in range(4)]
            xx,w=mixture(clean,noisy);packs[part]={'clean':(clean,d['y'],np.ones(len(clean))),'aug':(xx,np.tile(d['y'],5),w)}
            np.savez_compressed(folder/f'{part}_d{delay}.npz',pre=pre.numpy(),end=clean,noise_end=np.stack(noisy))
        scaler=StandardScaler().fit(packs['fit']['clean'][0])
        for name in ['clean','aug']:
            key=f'd{delay}_{name}';x,y,w=packs['fit'][name];vx,vy,vw=packs['validation'][name]
            xx=scaler.transform(x).astype(np.float64);vv=scaler.transform(vx).astype(np.float64)
            rows=[];bestloss=float('inf');best=None
            for c in CS:
                t0=time.monotonic();clf,hist=fit_candidate(xx,y,w,c)
                tl=float(log_loss(y,clf.predict_proba(xx),sample_weight=w));vl=float(log_loss(vy,clf.predict_proba(vv),sample_weight=vw))
                rows.append({'C':c,'training_loss':tl,'validation_loss':vl,'history':hist,'seconds':time.monotonic()-t0})
                candidates[f'{key}_C{c}']=Probe(scaler,clf)
                if vl<bestloss:bestloss=vl;best=Probe(scaler,clf);selected=c
                save(folder/'candidate_progress.json',{'completed_heads':records,'current_head':key,'current_candidates':rows})
                if len(records)==0 and c==CS[0]:print(json.dumps({'first_candidate_seconds':time.monotonic()-t0,'seed':seed}),flush=True)
            heads[key]=best;records[key]={'C':selected,'validation_loss':bestloss,'grid':rows,'weight_sum':float(w.sum()),'n_original':len(data['fit']['y']),'scaler_fit_on':'clean fit only'}
            print(json.dumps({'seed':seed,'fitted':key,'C':selected,'seconds':time.monotonic()-start}),flush=True)
    assert before==frozen_snapshot(model,{'old':old}) and all(sha(p)==v for p,v in inputs.items())
    with (folder/'heads.pkl').open('wb') as f:pickle.dump(heads,f)
    with (folder/'candidates.pkl').open('wb') as f:pickle.dump(candidates,f)
    save(folder/'fits.json',{'heads':records,'forward_audits':audits,'core_and_old_unchanged':True,'frozen_new_snapshot':frozen_snapshot(model,{'old':old,**heads}),'seconds':time.monotonic()-start,'finished_at':time.time()})


def lock_training():
    assert not list(ROOT.glob('test*.npz'));files={};maximum=0.
    for seed in range(301,306):
        folder=ROOT/f'seed{seed}';r=json.loads((folder/'fits.json').read_text());assert len(r['heads'])==8 and r['core_and_old_unchanged']
        for h in r['heads'].values():
            assert len(h['grid'])==5
            assert h['C']==min(h['grid'],key=lambda z:(z['validation_loss'],z['C']))['C']
            for c in h['grid']:
                last=c['history'][-1];assert last['passed'] and last['gradient_inf']<=1e-6;maximum=max(maximum,last['gradient_inf'])
        for p in folder.glob('*'):files[str(p)]=sha(p)
    for name in ['fit.npz','validation.npz','protocol.md']:files[str(ROOT/name)]=sha(ROOT/name)
    files[__file__]=sha(__file__)
    save(ROOT/'training_lock.json',{'files':files,'new_heads':40,'all_candidates_passed':True,'candidates':200,'max_gradient':maximum,'locked_at':time.time(),'test_files_absent':True})

@torch.no_grad()
def evaluate(seed):
    verify_lock();torch.set_num_threads(1);start=time.monotonic();folder=ROOT/f'seed{seed}'
    model,old,sigma,inputs=load_core(seed)
    with (folder/'heads.pkl').open('rb') as f:new=pickle.load(f)
    heads={'old':old,**new};before=frozen_snapshot(model,heads)
    assert before==json.loads((folder/'fits.json').read_text())['frozen_new_snapshot']
    inputs.update({str(folder/'heads.pkl'):sha(folder/'heads.pkl'),str(ROOT/'training_lock.json'):sha(ROOT/'training_lock.json')})
    save(folder/'evaluation_manifest.json',{'input_sha256':inputs,'before':before,'started_at':time.time(),'sigma_primary':sigma})
    results=[];audits=[]
    with patch.object(LogisticRegression,'fit',side_effect=RuntimeError('B-R forbids fitting during test')),patch.object(StandardScaler,'fit',side_effect=RuntimeError('B-R forbids fitting during test')):
        for bank in range(20):
            d=dict(np.load(ROOT/f'test{bank:02d}.npz'));es,et=encoded_pairs(model,d);y=d['y'];yi,ri=permutation_indices(d)
            for delay in DELAYS+[15]:
                active={'old':old} if delay==15 else {'old':old,'clean':new[f'd{delay}_clean'],'aug':new[f'd{delay}_aug']}
                pre,end=extract(model,es,et,delay);audits.append({'bank':bank,'delay':delay,'errors':check_forward(model,d,es,et,delay,pre,end)})
                states=[end]
                for rep in range(10):
                    np0,ne0=perturb(model,pre,et,int(d['seed']),rep,sigma)
                    if rep==0:noise_pre0=np0.numpy();noise_end0=ne0.numpy()
                    states.append(ne0)
                preds=np.asarray([[head.predict(state.numpy()) for state in states] for head in active.values()],dtype=np.int8)
                original=np.asarray([model.readout(state).argmax(1).numpy() for state in states],dtype=np.int8)
                reset=continue_test(model,torch.zeros_like(pre),et);_,resetnoise=perturb(model,torch.zeros_like(pre),et,int(d['seed']),0,sigma)
                rp=np.asarray([[head.predict(state.numpy()) for state in [reset,resetnoise]] for head in active.values()],dtype=np.int8)
                # Deterministic rowwise classifiers commute with row permutations.
                lp=np.broadcast_to(preds[:,[0,1],None,:],(len(active),2,64,len(y)))
                sp=preds[:,[0,1],:][:,:,ri]
                negative=np.stack([(rp==y).mean(-1),(lp==y[yi]).mean((-1,-2)),(sp==y).mean((-1,-2))],axis=-1)
                results.append({'bank':bank,'delay':delay,'head_order':list(active),'scores':(preds==y).mean(-1).tolist(),'original_scores':(original==y).mean(-1).tolist(),'negative_means':negative.tolist()})
                np.savez_compressed(folder/f'bank{bank:02d}_d{delay:02d}.npz',pre=pre.numpy(),end=end.numpy(),noise_pre0=noise_pre0,noise_end0=noise_end0,predictions=preds,original=original,reset=rp,label_indices=yi,row_indices=ri,label_predictions=lp,row_predictions=sp)
            if bank%5==4:print(json.dumps({'seed':seed,'banks':bank+1,'seconds':time.monotonic()-start}),flush=True)
    assert before==frozen_snapshot(model,heads);assert all(sha(p)==v for p,v in inputs.items());verify_lock()
    save(folder/'results.json',{'cells':results,'forward_audits':audits,'frozen_before_after_equal':True,'fit_guard_enabled':True,'disk_inputs_unchanged':True,'after':frozen_snapshot(model,heads),'seconds':time.monotonic()-start})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','train','lock','prepare_test','evaluate']);p.add_argument('--seed',type=int);a=p.parse_args()
    try:
        if a.action=='prepare':prepare()
        elif a.action=='prepare_test':prepare(True)
        elif a.action=='train':train(a.seed)
        elif a.action=='lock':lock_training()
        else:evaluate(a.seed)
    except Exception as e:
        save(ROOT/f'failure_{a.action}_{a.seed}_{int(time.time())}.json',{'error':repr(e),'time':time.time()});raise
