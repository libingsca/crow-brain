"""B-F read-only audits, before test generation and after full evaluation."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit,logsumexp
from sklearn.metrics import log_loss
from branch_bf_round1 import ROOT,DELAYS,TRAIN_DELAYS,BASE_DELAYS,HEADS,HELDOUT,EIGHT,REFERENCE_DELAYS,load_core,verify_lock,active_heads
from branch_bt_round1 import encoded_pairs,permutation_indices,frozen_snapshot
from branch_b_round1 import Probe,sha,save

@torch.no_grad()
def training_audit():
    torch.set_num_threads(1);checked=0;maxgrad=0.
    assert not list(ROOT.glob('test*.npz'))
    for seed in range(301,306):
        folder=ROOT/f'seed{seed}';model,refs,_,_=load_core(seed);m=json.loads((folder/'training_manifest.json').read_text());fits=json.loads((folder/'fits.json').read_text())
        assert all(sha(p)==v for p,v in m['input_sha256'].items())
        assert frozen_snapshot(model,refs)==m['before']
        assert set(x['delay'] for x in fits['forward_audits'])==set(TRAIN_DELAYS)
        assert not any((folder/f'{part}_d{d}.npz').exists() for part in ['fit','validation'] for d in HELDOUT)
        with (folder/'heads.pkl').open('rb') as f:heads=pickle.load(f)
        with (folder/'candidates.pkl').open('rb') as f:candidates=pickle.load(f)
        assert len(heads)==4 and len(candidates)==20 and all(heads[n].scaler is heads[HEADS[0]].scaler for n in HEADS)
        assert frozen_snapshot(model,{**refs,**heads})==fits['frozen_new_snapshot']
        data={p:dict(np.load(ROOT/f'{p}.npz')) for p in ['fit','validation']}
        raw={p:{d:dict(np.load(folder/f'{p}_d{d}.npz')) for d in TRAIN_DELAYS} for p in data}
        pool=np.concatenate([raw['fit'][d]['end'] for d in BASE_DELAYS]).astype(np.float64)
        scaler=heads['cover_split'].scaler
        np.testing.assert_allclose(scaler.mean_,pool.mean(0),atol=1e-12);np.testing.assert_allclose(scaler.var_,pool.var(0),atol=1e-10)
        assert int(scaler.n_samples_seen_)==40000;del pool
        for name in HEADS:
            packs={}
            for part in data:
                yy=data[part]['y'];n=len(yy);xx=[];weights=[]
                used=EIGHT if part=='fit' and name.startswith('base') else TRAIN_DELAYS
                mdel=len(used);clean=name.endswith('clean')
                for d in used:
                    xx.append(raw[part][d]['end']);weights.append(np.full(n,(1 if clean else .5)/mdel))
                    if not clean:
                        for state in raw[part][d]['noise_end']:xx.append(state);weights.append(np.full(n,.125/mdel))
                w=np.concatenate(weights);x=scaler.transform(np.concatenate(xx).astype(np.float64));y=np.tile(yy,len(xx))
                np.testing.assert_allclose(w.reshape(-1,n).sum(0),1,atol=1e-14)
                np.testing.assert_allclose(w.reshape(mdel,-1).sum(1),n/mdel,atol=1e-10)
                packs[part]=(x,y,w)
            vals=[]
            for row in fits['heads'][name]['grid']:
                c=row['C'];head=candidates[f'{name}_C{c}'];computed={}
                assert head.scaler is candidates[f'base_mix_C{c}'].scaler
                for part,(x,y,w) in packs.items():
                    zz=head.clf.decision_function(x)
                    computed[part]=float(np.average(np.logaddexp(0,zz)-y*zz,weights=w))
                    used=EIGHT if part=='fit' and name.startswith('base') else TRAIN_DELAYS
                    block=len(x)//len(used)
                    for di,d in enumerate(used):
                        sl=slice(di*block,(di+1)*block);loss=float(np.average(np.logaddexp(0,zz[sl])-y[sl]*zz[sl],weights=w[sl]));score=np.average(head.clf.predict(x[sl])==y[sl],weights=w[sl])
                        np.testing.assert_allclose([loss,score],[row['per_delay'][part][str(d)]['loss'],row['per_delay'][part][str(d)]['ba']],atol=1e-12)
                    if part=='fit':
                        beta=head.clf.coef_[0];z=x@beta+head.clf.intercept_[0]
                        pergroup=[];losses=[]
                        bounds=[]
                        for di in range(len(used)):
                            start=di*block
                            bounds.extend([(start,start+block//5),(start+block//5,start+block)] if name.endswith('split') else [(start,start+block)])
                        for aa,bb in bounds:
                            sl=slice(aa,bb);ww=w[sl]/w[sl].sum()
                            ee=ww*(expit(z[sl])-y[sl]);pergroup.append(np.r_[x[sl].T@ee,ee.sum()])
                            losses.append(float(np.sum(ww*(np.logaddexp(0,z[sl])-y[sl]*z[sl]))))
                        losses=np.array(losses);qq=np.exp(losses/.05-logsumexp(losses/.05))
                        g=qq@np.asarray(pergroup);g[:-1]+=beta/(c*w.sum());grad=float(np.abs(g).max());maxgrad=max(maxgrad,grad)
                        obj=float(.05*(logsumexp(losses/.05)-np.log(len(losses)))+beta@beta/(2*c*w.sum()))
                        assert grad<=1e-6 and row['history'][-1]['passed']
                        np.testing.assert_allclose([obj,grad],[row['history'][-1]['objective'],row['history'][-1]['gradient_inf']],atol=1e-10)
                        np.testing.assert_allclose(qq,row['history'][-1]['group_weights'],atol=1e-10)
                        assert np.all(qq>=0) and abs(qq.sum()-1)<1e-12
                    else:
                        n=len(x)//(len(TRAIN_DELAYS)*5)
                        for di,d in enumerate(TRAIN_DELAYS):
                            for cond,aa,bb in [('clean',di*5*n,di*5*n+n),('noise',di*5*n+n,(di+1)*5*n)]:
                                loss=float(np.average(np.logaddexp(0,zz[aa:bb])-y[aa:bb]*zz[aa:bb],weights=w[aa:bb]))
                                score=float(np.average(head.clf.predict(x[aa:bb])==y[aa:bb],weights=w[aa:bb]))
                                np.testing.assert_allclose([loss,score],[row['condition_validation'][f'{d}_{cond}']['loss'],row['condition_validation'][f'{d}_{cond}']['ba']],atol=1e-12)
                np.testing.assert_allclose([computed['fit'],computed['validation']],[row['training_loss'],row['validation_mean_loss']],atol=1e-12)
                worst_validation=max(v['loss'] for v in row['condition_validation'].values())
                np.testing.assert_allclose(worst_validation,row['validation_loss'],atol=1e-12)
                vals.append((worst_validation,c));checked+=1
            best=min(vals)[1];assert best==fits['heads'][name]['C'];chosen=candidates[f'{name}_C{best}']
            np.testing.assert_array_equal(chosen.clf.coef_,heads[name].clf.coef_);np.testing.assert_array_equal(chosen.clf.intercept_,heads[name].clf.intercept_)
            del packs
        print('training audited',seed,flush=True)
    assert checked==100
    save(ROOT/'training_audit.json',{'passed':True,'candidate_count':checked,'max_gradient_inf':maxgrad,'validation_selection_recomputed':True,'pooled_scaler_verified':True,'per_trial_and_delay_weights_verified':True,'test_files_absent':True,'heldout_states_absent':True})

@torch.no_grad()
def evaluation_audit():
    torch.set_num_threads(1);verify_lock();checks=0;predchecks=0
    dm=json.loads((ROOT/'data_manifest.json').read_text())
    assert len(dm['banks'])==22 and dm['test_created_after_all_heads_locked']
    for name,b in dm['banks'].items():assert sha(ROOT/f'{name}.npz')==b['sha256']
    for p,v in dm['prior_files'].items():assert sha(p)==v
    for seed in range(301,306):
        folder=ROOT/f'seed{seed}';model,refs,sigma,_=load_core(seed)
        with (folder/'heads.pkl').open('rb') as f:new=pickle.load(f)
        m=json.loads((folder/'evaluation_manifest.json').read_text());r=json.loads((folder/'results.json').read_text())
        assert all(sha(p)==v for p,v in m['input_sha256'].items())
        before=frozen_snapshot(model,{**refs,**new});assert before==m['before']==r['after']
        assert r['fit_guard_enabled'] and r['frozen_before_after_equal'] and r['disk_inputs_unchanged']
        indexed={(x['bank'],x['delay']):x for x in r['cells']};assert len(indexed)==220 and len(r['cells'])==220 and len(r['forward_audits'])==220
        for bank in range(20):
            d=dict(np.load(ROOT/f'test{bank:02d}.npz'));y=d['y'];_,et=encoded_pairs(model,d);yi,ri=permutation_indices(d)
            assert np.bincount(y).tolist()==[1000,1000]
            for delay in DELAYS:
                active=active_heads(new,refs,delay)
                assert all(active[n] is new[n] for n in HEADS)
                assert all(active[n].scaler is new[HEADS[0]].scaler for n in HEADS)
                raw=dict(np.load(folder/f'bank{bank:02d}_d{delay:02d}.npz'));cell=indexed[bank,delay];assert cell['head_order']==list(active)
                assert raw['predictions'].shape==(len(active),11,2000)
                np.testing.assert_array_equal((raw['predictions']==y).mean(-1),cell['scores']);np.testing.assert_array_equal((raw['original']==y).mean(-1),cell['original_scores'])
                np.testing.assert_array_equal(raw['label_indices'],yi);np.testing.assert_array_equal(raw['row_indices'],ri)
                expectedlabel=np.broadcast_to(raw['predictions'][:,[0,1],None,:],(len(active),2,64,2000))
                np.testing.assert_array_equal(raw['label_predictions'],expectedlabel)
                expectedrow=np.stack([np.stack([raw['predictions'][h,j][ri] for j in [0,1]]) for h in range(len(active))])
                np.testing.assert_array_equal(raw['row_predictions'],expectedrow)
                neg=np.stack([(raw['reset']==y).mean(-1),(raw['label_predictions']==y[yi]).mean((-1,-2)),(raw['row_predictions']==y).mean((-1,-2))],axis=-1)
                np.testing.assert_array_equal(neg,cell['negative_means'])
                for rep in range(-1,10):
                    if rep==-1:state=raw['end']
                    else:
                        z=np.random.default_rng(np.random.SeedSequence([int(d['seed']),20000,rep])).normal(size=raw['pre'].shape).astype(np.float32)
                        state=np.maximum(raw['pre']+z*sigma,0).astype(np.float32)
                        if rep==0:np.testing.assert_array_equal(state,raw['noise_pre0'])
                        state=torch.from_numpy(state)
                        for _ in range(4):state=(1-model.alpha)*state+model.alpha*torch.relu(model.input(et)+state@model.weights().T)
                        state=state.numpy()
                        if rep==0:np.testing.assert_allclose(state,raw['noise_end0'],rtol=1e-6,atol=1e-6)
                    for i,head in enumerate(active.values()):np.testing.assert_array_equal(head.predict(state),raw['predictions'][i,rep+1]);predchecks+=1
                    np.testing.assert_array_equal(model.readout(torch.from_numpy(state)).argmax(1).numpy(),raw['original'][rep+1])
                for j in range(2):
                    state=np.zeros_like(raw['pre'])
                    if j:
                        z=np.random.default_rng(np.random.SeedSequence([int(d['seed']),20000,0])).normal(size=state.shape).astype(np.float32);state=np.maximum(z*sigma,0)
                    state=torch.from_numpy(state)
                    for _ in range(4):state=(1-model.alpha)*state+model.alpha*torch.relu(model.input(et)+state@model.weights().T)
                    for i,head in enumerate(active.values()):np.testing.assert_array_equal(head.predict(state.numpy()),raw['reset'][i,j])
                checks+=1
        assert before==frozen_snapshot(model,{**refs,**new});print('evaluation audited',seed,flush=True)
    assert checks==1100 and predchecks==57200
    save(ROOT/'independent_audit.json',{'passed':True,'complete_cells':checks,'head_prediction_vectors_recomputed':predchecks,'all_noise_replicates_reconstructed':True,'negative_predictions_and_rng_verified':True,'reset_recomputed':True,'input_hashes_unchanged':True,'frozen_parameters_unchanged':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['training','evaluation']);a=p.parse_args()
    if a.phase=='training':training_audit()
    else:evaluation_audit()
