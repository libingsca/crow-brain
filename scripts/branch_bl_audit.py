"""Read-only audit of B-L fitted candidates, selection and complete raw predictions."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import log_loss
from branch_bl_round1 import ROOT,DELAYS,load_core,verify_lock,active_heads
from branch_bt_round1 import encoded_pairs,permutation_indices,frozen_snapshot
from branch_b_round1 import Probe,sha,save

@torch.no_grad()
def training_audit():
    torch.set_num_threads(1);checked=0;maxgrad=0.
    assert not list(ROOT.glob('test*.npz'))
    for seed in range(301,306):
        folder=ROOT/f'seed{seed}';model,old,_,_=load_core(seed)
        m=json.loads((folder/'training_manifest.json').read_text());fits=json.loads((folder/'fits.json').read_text())
        assert all(sha(p)==v for p,v in m['input_sha256'].items())
        assert frozen_snapshot(model,old)==m['before']
        with (folder/'heads.pkl').open('rb') as f:heads=pickle.load(f)
        with (folder/'candidates.pkl').open('rb') as f:candidates=pickle.load(f)
        assert frozen_snapshot(model,{**old,**heads})==fits['frozen_new_snapshot']
        assert len(heads)==6 and len(candidates)==30
        data={p:dict(np.load(ROOT/f'{p}.npz')) for p in ['fit','validation']}
        for delay in DELAYS:
            raw={p:dict(np.load(folder/f'{p}_d{delay}.npz')) for p in data}
            assert heads[f'd{delay}_clean'].scaler is heads[f'd{delay}_aug'].scaler
            for name in ['clean','aug']:
                key=f'd{delay}_{name}';meta=fits['heads'][key];vals=[]
                for row in meta['grid']:
                    c=row['C'];head=candidates[f'{key}_C{c}'];computed={}
                    np.testing.assert_allclose(head.scaler.mean_,raw['fit']['end'].mean(0,dtype=np.float64),atol=1e-12)
                    np.testing.assert_allclose(head.scaler.var_,raw['fit']['end'].astype(np.float64).var(0),atol=1e-10)
                    assert int(head.scaler.n_samples_seen_)==8000
                    for part in ['fit','validation']:
                        clean=raw[part]['end'];n=len(clean);y=data[part]['y']
                        if name=='clean':x=clean;w=np.ones(n)
                        else:
                            x=np.concatenate([clean,*raw[part]['noise_end']]);w=np.tile(np.array([.5,.125,.125,.125,.125])[:,None],(1,n)).ravel();y=np.tile(y,5)
                            np.testing.assert_array_equal(w.reshape(5,n).sum(0),np.ones(n))
                        xx=head.scaler.transform(x).astype(np.float64)
                        computed[part]=float(log_loss(y,head.clf.predict_proba(xx),sample_weight=w))
                        if part=='fit':
                            beta=head.clf.coef_[0];intercept=head.clf.intercept_[0];z=xx@beta+intercept
                            error=w*(expit(z)-y)/w.sum();g=np.r_[xx.T@error+beta/(c*w.sum()),error.sum()]
                            grad=float(np.abs(g).max());maxgrad=max(maxgrad,grad)
                            assert grad<=1e-6 and row['history'][-1]['passed']
                            np.testing.assert_allclose(grad,row['history'][-1]['gradient_inf'],atol=1e-12)
                    np.testing.assert_allclose(computed['fit'],row['training_loss'],atol=1e-12)
                    np.testing.assert_allclose(computed['validation'],row['validation_loss'],atol=1e-12)
                    vals.append((computed['validation'],c));checked+=1
                assert len(vals)==5
                best=min(vals)[1];assert best==meta['C']
                chosen=candidates[f'{key}_C{best}'];np.testing.assert_array_equal(chosen.clf.coef_,heads[key].clf.coef_);np.testing.assert_array_equal(chosen.clf.intercept_,heads[key].clf.intercept_)
        print('training audited',seed,flush=True)
    assert checked==150
    save(ROOT/'training_audit.json',{'passed':True,'candidate_count':checked,'max_gradient_inf':maxgrad,'validation_selection_recomputed':True,'all_scalers_fit_on_clean_fit_only':True,'test_files_absent':True,'weights_per_original_verified':True})

@torch.no_grad()
def evaluation_audit():
    torch.set_num_threads(1);verify_lock();checks=0;predchecks=0
    dm=json.loads((ROOT/'data_manifest.json').read_text())
    assert len(dm['banks'])==22 and dm['test_created_after_all_heads_locked']
    for name,b in dm['banks'].items():assert sha(ROOT/f'{name}.npz')==b['sha256']
    for p,v in dm['prior_files'].items():assert sha(p)==v
    for seed in range(301,306):
        folder=ROOT/f'seed{seed}';model,old,sigma,_=load_core(seed)
        with (folder/'heads.pkl').open('rb') as f:new=pickle.load(f)
        m=json.loads((folder/'evaluation_manifest.json').read_text());r=json.loads((folder/'results.json').read_text())
        assert all(sha(p)==v for p,v in m['input_sha256'].items())
        before=frozen_snapshot(model,{**old,**new});assert before==m['before']==r['after']
        assert r['fit_guard_enabled'] and r['frozen_before_after_equal'] and r['disk_inputs_unchanged']
        indexed={(x['bank'],x['delay']):x for x in r['cells']};assert set(indexed)=={(b,d) for b in range(20) for d in DELAYS+[5,15]} and len(r['cells'])==100 and len(r['forward_audits'])==100
        for bank in range(20):
            d=dict(np.load(ROOT/f'test{bank:02d}.npz'));y=d['y'];_,et=encoded_pairs(model,d);yi,ri=permutation_indices(d)
            assert np.bincount(y).tolist()==[1000,1000]
            for delay in DELAYS+[5,15]:
                active=active_heads(old,new,delay)
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
        assert before==frozen_snapshot(model,{**old,**new});print('evaluation audited',seed,flush=True)
    assert checks==500 and predchecks==15400
    save(ROOT/'independent_audit.json',{'passed':True,'complete_cells':checks,'head_prediction_vectors_recomputed':predchecks,'all_noise_replicates_reconstructed':True,'negative_predictions_and_rng_verified':True,'reset_recomputed':True,'input_hashes_unchanged':True,'frozen_parameters_unchanged':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['training','evaluation']);a=p.parse_args()
    if a.phase=='training':training_audit()
    else:evaluation_audit()
