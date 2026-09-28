"""Preregistered B-Z: frozen cores, shared readouts, locked evaluation."""
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
import os
REPEAT=int(os.environ.get('BZ_REPEAT','1'));assert 1<=REPEAT<=5
ROOT=Path(f'artifacts/branch_bz/repeat{REPEAT}');BASE_DELAYS=[5,7,10,13,15];TRAIN_DELAYS=[5,7,8,10,11,12,13,15];DELAYS=list(range(5,16));HELDOUT=[6,9,14];HEADS=['mean_aug','worst_aug'];CS=[.01,.1,1.,10.,100.]

import resource
from branch_bw_solver import fit_candidate,TAU

def load_core(seed):
    cp=Path(f'artifacts/round6/final/rnn_lr0.001_seed{seed}/final.pt');hp=Path(f'artifacts/branch_bn/round1/seed{seed}/heads.pkl');mp=hp.with_name('manifest.json');rp=Path(f'artifacts/branch_br/round1/seed{seed}/heads.pkl')
    model=VisualRNN(normalize=True);model.load_state_dict(torch.load(cp,weights_only=True));model.eval();model.requires_grad_(False)
    with hp.open('rb') as f:ref15=pickle.load(f)['aug']
    with rp.open('rb') as f:previous=pickle.load(f)
    lp=Path(f'artifacts/branch_bl/round1/seed{seed}/heads.pkl')
    with lp.open('rb') as f:local=pickle.load(f)
    refs={f'ref{d}':previous[f'd{d}_aug'] for d in BASE_DELAYS if d!=15};refs['ref15']=ref15
    refs.update({f'ref{d}':local[f'd{d}_aug'] for d in [8,11,12]})
    return model,refs,json.loads(mp.read_text())['sigma_primary'],{str(p):sha(p) for p in [cp,hp,mp,rp,lp,Path(__file__),ROOT/'protocol.md']}

def active_heads(new,refs,delay):
    active={name:new[name] for name in HEADS}
    if delay in TRAIN_DELAYS:active['reference']=refs[f'ref{delay}']
    return active

def verify_lock():
    m=json.loads((ROOT/'training_lock.json').read_text())
    for p,v in m['files'].items():assert sha(p)==v,p
    assert m['new_heads']==10 and m['all_candidates_passed']
    return m

def prepare(test=False):
    if test: verify_lock()
    old=previous_data()+[Path(f'artifacts/branch_bt/round1/test{i:02d}.npz') for i in range(20)]+[Path('artifacts/branch_br/round1')/f'{n}.npz' for n in ['fit','validation']+[f'test{i:02d}' for i in range(20)]]
    old += [Path(f'artifacts/branch_{b}/round1/{n}.npz') for b in ['bs','bl','bc','bw'] for n in ['fit','validation']+[f'test{i:02d}' for i in range(20)]]
    old += [p for p in Path('artifacts/branch_bz').glob('repeat*/*.npz') if p.parent!=ROOT]
    seen=set();refs={}
    for p in old:
        with np.load(p) as d:seen.update(digest(im) for im in d['images'].reshape(-1,1,32,32))
        refs[str(p)]=sha(p)
    assert len(seen)>=220000
    prior_count=len(seen)
    with np.load('artifacts/round6/valid_shape_bank.npz') as a:
        for k in ['solid','ring']:seen.update(digest(im) for im in a[k].reshape(-1,1,32,32))
    refs['artifacts/round6/valid_shape_bank.npz']=sha('artifacts/round6/valid_shape_bank.npz')
    records={}
    if test:
        records=json.loads((ROOT/'data_manifest.json').read_text())['banks']
        for name in ['fit','validation']:
            p=ROOT/f'{name}.npz';assert sha(p)==records[name]['sha256']
            with np.load(p) as d:seen.update(digest(im) for im in d['images'].reshape(-1,1,32,32))
    configs=[(f'test{i:02d}',980000+REPEAT*1000+100+i,200,2000) for i in range(4)] if test else [('fit',980000+REPEAT*1000+1,1000,8000),('validation',980000+REPEAT*1000+2,1000,2000)]
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
    save(ROOT/'data_manifest.json',{'banks':records,'prior_files':refs,'image_sets_disjoint':True,'prior_unique_images':prior_count,'test_created_after_all_heads_locked':test})


def exact_loss(clf,x,y,w):
    z=clf.decision_function(x)
    return float(np.average(np.logaddexp(0,z)-y*z,weights=w))

def pooled_pack(raw,y,name,delays):
    x=[];weights=[];targets=[]
    m=len(delays)
    for delay in delays:
        clean=raw[delay]['end'];n=len(clean)
        if name.endswith('clean'):
            x.append(clean);weights.append(np.full(n,1/m));targets.append(y)
        else:
            x.extend([clean,*raw[delay]['noise_end']]);weights.extend([np.full(n,.5/m)]+[np.full(n,.125/m) for _ in range(4)]);targets.extend([y]*5)
    xx=np.concatenate(x);yy=np.concatenate(targets);w=np.concatenate(weights)
    np.testing.assert_allclose(w.reshape(-1,len(y)).sum(0),np.ones(len(y)),atol=1e-14)
    return xx,yy,w

@torch.no_grad()
def train(seed):
    assert not list(ROOT.glob('test*.npz'));torch.set_num_threads(1);start=time.monotonic();folder=ROOT/f'seed{seed}';folder.mkdir(exist_ok=False)
    model,refs,sigma,inputs=load_core(seed);before=frozen_snapshot(model,refs)
    inputs.update({str(ROOT/f'{p}.npz'):sha(ROOT/f'{p}.npz') for p in ['fit','validation']})
    save(folder/'training_manifest.json',{'seed':seed,'sigma_primary':sigma,'input_sha256':inputs,'before':before,'started_at':time.time(),'allowed_delays':TRAIN_DELAYS})
    data={p:dict(np.load(ROOT/f'{p}.npz')) for p in ['fit','validation']};encoded={p:encoded_pairs(model,d) for p,d in data.items()};raw={p:{} for p in data};audits=[]
    for delay in TRAIN_DELAYS:
        for part,d in data.items():
            es,et=encoded[part];pre,end=extract(model,es,et,delay)
            audits.append({'part':part,'delay':delay,'errors':check_forward(model,d,es,et,delay,pre,end)})
            raw[part][delay]={'pre':pre.numpy(),'end':end.numpy(),'noise_end':np.stack([perturb(model,pre,et,int(d['seed']),r,sigma)[1].numpy() for r in range(4)])}
            np.savez_compressed(folder/f'{part}_d{delay}.npz',**raw[part][delay])
    pooled=np.concatenate([raw['fit'][d]['end'] for d in BASE_DELAYS]).astype(np.float64)
    scaler=StandardScaler().fit(pooled);del pooled
    heads={};candidates={};records={}
    for name in HEADS:
        fitting=TRAIN_DELAYS;kind=name.split('_')[0]
        x,y,w=pooled_pack(raw['fit'],data['fit']['y'],name,fitting);vx,vy,vw=pooled_pack(raw['validation'],data['validation']['y'],name,TRAIN_DELAYS)
        xx=scaler.transform(x.astype(np.float64));vv=scaler.transform(vx.astype(np.float64));del x,vx
        rows=[];bestloss=float('inf');best=None
        for c in CS:
            t0=time.monotonic();clf,hist=fit_candidate(xx,y,w,c,kind)
            tl=exact_loss(clf,xx,y,w);vl=exact_loss(clf,vv,vy,vw)
            perdelay={}
            for part,z,yy,ww in [('fit',xx,y,w),('validation',vv,vy,vw)]:
                used=fitting if part=='fit' else TRAIN_DELAYS
                block=len(z)//len(used);perdelay[part]={}
                for di,d in enumerate(used):
                    sl=slice(di*block,(di+1)*block);prob=clf.predict_proba(z[sl]);pred=clf.predict(z[sl])
                    perdelay[part][str(d)]={'loss':exact_loss(clf,z[sl],yy[sl],ww[sl]),'ba':float(np.average(pred==yy[sl],weights=ww[sl]))}
            selection=max(v['loss'] for v in perdelay['validation'].values())
            rows.append({'C':c,'training_loss':tl,'validation_mean_loss':vl,'validation_loss':selection,'objective_kind':kind,'tau':TAU,'per_delay':perdelay,'history':hist,'seconds':time.monotonic()-t0})
            candidates[f'{name}_C{c}']=Probe(scaler,clf)
            if selection<bestloss:bestloss=selection;best=Probe(scaler,clf);selected=c
            save(folder/'candidate_progress.json',{'completed_heads':records,'current_head':name,'current_candidates':rows})
            print(json.dumps({'seed':seed,'head':name,'C':c,'candidate_seconds':time.monotonic()-t0,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}),flush=True)
        heads[name]=best;records[name]={'C':selected,'validation_loss':bestloss,'grid':rows,'weight_sum':float(w.sum()),'n_original':8000,'rows':len(xx),'scaler_fit_on':BASE_DELAYS,'fitting_delays':fitting,'validation_delays':TRAIN_DELAYS}
        del xx,vv,y,w,vy,vw
    assert all(heads[n].scaler is heads[HEADS[0]].scaler for n in HEADS)
    assert before==frozen_snapshot(model,refs) and all(sha(p)==v for p,v in inputs.items())
    with (folder/'heads.pkl').open('wb') as f:pickle.dump(heads,f)
    with (folder/'candidates.pkl').open('wb') as f:pickle.dump(candidates,f)
    save(folder/'fits.json',{'heads':records,'forward_audits':audits,'core_and_old_unchanged':True,'frozen_new_snapshot':frozen_snapshot(model,{**refs,**heads}),'seconds':time.monotonic()-start,'finished_at':time.time(),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})

def lock_training():
    assert not list(ROOT.glob('test*.npz'));assert json.loads((ROOT/'training_audit.json').read_text())['passed'];files={};maximum=0.
    for seed in range(301,306):
        folder=ROOT/f'seed{seed}';r=json.loads((folder/'fits.json').read_text());assert len(r['heads'])==2 and r['core_and_old_unchanged']
        assert set(a['delay'] for a in r['forward_audits'])==set(TRAIN_DELAYS)
        for h in r['heads'].values():
            assert len(h['grid'])==5 and h['C']==min(h['grid'],key=lambda z:(z['validation_loss'],z['C']))['C']
            for c in h['grid']:
                last=c['history'][-1];assert last['passed'] and last['gradient_inf']<=1e-6;maximum=max(maximum,last['gradient_inf'])
        for p in folder.glob('*'):files[str(p)]=sha(p)
    for name in ['fit.npz','validation.npz','protocol.md','training_audit.json']:files[str(ROOT/name)]=sha(ROOT/name)
    files[__file__]=sha(__file__)
    save(ROOT/'training_lock.json',{'files':files,'new_heads':10,'all_candidates_passed':True,'candidates':50,'max_gradient':maximum,'locked_at':time.time(),'test_files_absent':True,'training_delays':TRAIN_DELAYS,'heldout_delays':HELDOUT})

@torch.no_grad()
def evaluate(seed):
    verify_lock();torch.set_num_threads(1);start=time.monotonic();folder=ROOT/f'seed{seed}'
    model,refs,sigma,inputs=load_core(seed)
    with (folder/'heads.pkl').open('rb') as f:new=pickle.load(f)
    heads={**refs,**new};before=frozen_snapshot(model,heads)
    assert before==json.loads((folder/'fits.json').read_text())['frozen_new_snapshot']
    inputs.update({str(folder/'heads.pkl'):sha(folder/'heads.pkl'),str(ROOT/'training_lock.json'):sha(ROOT/'training_lock.json')})
    save(folder/'evaluation_manifest.json',{'input_sha256':inputs,'before':before,'started_at':time.time(),'sigma_primary':sigma})
    results=[];audits=[]
    with patch.object(LogisticRegression,'fit',side_effect=RuntimeError('B-Z forbids fitting during test')),patch.object(StandardScaler,'fit',side_effect=RuntimeError('B-Z forbids fitting during test')):
        for bank in range(4):
            d=dict(np.load(ROOT/f'test{bank:02d}.npz'));es,et=encoded_pairs(model,d);y=d['y'];yi,ri=permutation_indices(d)
            for delay in DELAYS:
                active=active_heads(new,refs,delay)
                assert all(active[n] is new[n] for n in HEADS)
                assert all(active[n].scaler is new[HEADS[0]].scaler for n in HEADS)
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
            if bank==3:print(json.dumps({'seed':seed,'banks':bank+1,'seconds':time.monotonic()-start}),flush=True)
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
