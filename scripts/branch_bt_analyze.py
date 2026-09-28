"""Complete-matrix analysis and raw-prediction audit of preregistered B-T."""
import json,pickle
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t
from branch_bt_round1 import ROOT,DELAYS,HEADS,frozen_snapshot,encoded_pairs,permutation_indices
from branch_b_round1 import Probe,sha,save,digest
from crowbrain.model import VisualRNN

@torch.no_grad()
def audit():
    torch.set_num_threads(1);checks=0;prediction_checks=0
    dm=json.loads((ROOT/'data_manifest.json').read_text())
    assert all(sha(p)==v for p,v in dm['prior_files'].items())
    for bank in dm['banks']: assert sha(bank['path'])==bank['sha256']
    for seed in range(301,306):
        folder=ROOT/f'seed{seed}';m=json.loads((folder/'manifest.json').read_text());r=json.loads((folder/'results.json').read_text())
        assert all(sha(p)==v for p,v in m['input_sha256'].items())
        model=VisualRNN(normalize=True);model.load_state_dict(torch.load(f'artifacts/round6/final/rnn_lr0.001_seed{seed}/final.pt',weights_only=True));model.eval();model.requires_grad_(False)
        with open(f'artifacts/branch_bn/round1/seed{seed}/heads.pkl','rb') as f: h=pickle.load(f)
        heads={name:h[name] for name in HEADS};assert frozen_snapshot(model,heads)==m['before']==r['after']
        assert len(r['cells'])==100 and len(r['forward_audits'])==100
        indexed={(x['bank'],x['delay']):x for x in r['cells']};assert len(indexed)==100
        for bank in range(20):
            d=dict(np.load(ROOT/f'test{bank:02d}.npz'));y=d['y'];_,et=encoded_pairs(model,d);yi,ri=permutation_indices(d)
            assert np.bincount(y).tolist()==[1000,1000]
            for delay in DELAYS:
                raw=dict(np.load(folder/f'bank{bank:02d}_d{delay:02d}.npz'));cell=indexed[bank,delay]
                assert raw['predictions'].shape==(2,11,2000)
                np.testing.assert_array_equal((raw['predictions']==y).mean(-1),cell['scores'])
                np.testing.assert_array_equal((raw['original']==y).mean(-1),cell['original_scores'])
                np.testing.assert_array_equal(raw['label_indices'],yi);np.testing.assert_array_equal(raw['row_indices'],ri)
                expectedlabel=np.broadcast_to(raw['predictions'][:,[0,1],None,:],(2,2,64,2000))
                np.testing.assert_array_equal(raw['label_predictions'],expectedlabel)
                expectedrow=np.stack([np.stack([raw['predictions'][h,j][ri] for j in [0,1]]) for h in range(2)])
                np.testing.assert_array_equal(raw['row_predictions'],expectedrow)
                negatives=np.stack([(raw['reset']==y).mean(-1),(raw['label_predictions']==y[yi]).mean((-1,-2)),(raw['row_predictions']==y).mean((-1,-2))],axis=-1)
                np.testing.assert_array_equal(negatives,cell['negative_means'])
                # Independently reconstruct noise and the four test updates from saved pre-test state.
                for rep in range(-1,10):
                    if rep==-1: state=raw['end']
                    else:
                        z=np.random.default_rng(np.random.SeedSequence([int(d['seed']),20000,rep])).normal(size=raw['pre'].shape).astype(np.float32)
                        state=np.maximum(raw['pre']+z*m['sigma_primary'],0).astype(np.float32)
                        if rep==0: np.testing.assert_array_equal(state,raw['noise_pre0'])
                        state=torch.from_numpy(state)
                        for _ in range(4): state=(1-model.alpha)*state+model.alpha*torch.relu(model.input(et)+state@model.weights().T)
                        state=state.numpy()
                        if rep==0: np.testing.assert_allclose(state,raw['noise_end0'],rtol=1e-6,atol=1e-6)
                    for i,name in enumerate(HEADS):
                        np.testing.assert_array_equal(heads[name].predict(state),raw['predictions'][i,rep+1]);prediction_checks+=1
                    np.testing.assert_array_equal(model.readout(torch.from_numpy(state)).argmax(1).numpy(),raw['original'][rep+1])
                for j in range(2):
                    reset=np.zeros_like(raw['pre'])
                    if j:
                        z=np.random.default_rng(np.random.SeedSequence([int(d['seed']),20000,0])).normal(size=reset.shape).astype(np.float32)
                        reset=np.maximum(z*m['sigma_primary'],0)
                    state=torch.from_numpy(reset)
                    for _ in range(4): state=(1-model.alpha)*state+model.alpha*torch.relu(model.input(et)+state@model.weights().T)
                    for i,name in enumerate(HEADS): np.testing.assert_array_equal(heads[name].predict(state.numpy()),raw['reset'][i,j])
                checks+=1
        assert frozen_snapshot(model,heads)==m['before']
        print('audited',seed,flush=True)
    save(ROOT/'independent_audit.json',{'passed':True,'complete_cells':checks,'head_prediction_vectors_recomputed':prediction_checks,'negative_predictions_and_rng_verified':True,'all_noise_replicates_reconstructed':True,'reset_recomputed':True,'input_hashes_unchanged':True,'frozen_parameters_unchanged':True})

def interval(values):
    values=np.asarray(values);mean=float(values.mean());half=float(t.ppf(1-.05/8/2,19)*values.std(ddof=1)/np.sqrt(20))
    return {'mean':mean,'lower':mean-half,'upper':mean+half,'confidence':.99375,'n_banks':20}

def analyze():
    # axes model, bank, delay, head, condition(clean/noise).
    scores=np.empty((5,20,5,2,2));original=np.empty((5,20,2));negative=np.empty((5,20,5,2,2,3))
    for si,seed in enumerate(range(301,306)):
        r=json.loads((ROOT/f'seed{seed}/results.json').read_text())
        assert r['frozen_before_after_equal'] and r['disk_inputs_unchanged'] and r['fit_guard_enabled']
        for cell in r['cells']:
            bi=cell['bank'];di=DELAYS.index(cell['delay']);s=np.asarray(cell['scores'])
            scores[si,bi,di,:,0]=s[:,0];scores[si,bi,di,:,1]=s[:,1:].mean(-1)
            negative[si,bi,di]=cell['negative_means']
            if cell['delay']==5:
                o=np.asarray(cell['original_scores']);original[si,bi]=[o[0],o[1:].mean()]
    primary={f'd{d}_{c}':interval(scores[:,:,di,1,ci].mean(0)) for di,d in enumerate(DELAYS[:4]) for ci,c in enumerate(['clean','noise'])}
    positives={f'aug15_{c}':interval(scores[:,:,4,1,ci].mean(0)) for ci,c in enumerate(['clean','noise'])}
    positives.update({f'original5_{c}':interval(original[:,:,ci].mean(0)) for ci,c in enumerate(['clean','noise'])})
    negmean=negative.mean((0,1));negpass=bool(np.all((negmean[:4]>=.47)&(negmean[:4]<=.53)))
    quality=negpass and all(v['lower']>.90 for v in positives.values()) and json.loads((ROOT/'independent_audit.json').read_text())['passed']
    decision='quality_failed' if not quality else ('supported' if all(v['lower']>.85 for v in primary.values()) else ('excluded' if any(v['upper']<.85 for v in primary.values()) else 'inconclusive'))
    means=scores.mean(1)
    result={'decision':decision,'quality_passed':quality,'primary':primary,'positive_controls':positives,'negative_controls_passed':negpass,'negative_means_delay_head_condition_control':negmean.tolist(),'negative_control_names':['reset','label_shuffle','within_test_count_state_shuffle'],'per_model_means_delay_head_condition':{str(s):means[i].tolist() for i,s in enumerate(range(301,306))},'per_model_aug_minimum_new_delays':{str(s):float(means[i,:4,1].min()) for i,s in enumerate(range(301,306))},'paired_aug_minus_clean_delay_condition':(scores[:,:,:,1]-scores[:,:,:,0]).mean((0,1)).tolist(),'panel_all_cells':{f'{h}_d{d}_{c}':interval(scores[:,:,di,hi,ci].mean(0)) for hi,h in enumerate(HEADS) for di,d in enumerate(DELAYS) for ci,c in enumerate(['clean','noise'])}}
    np.savez_compressed(ROOT/'analysis_arrays.npz',scores=scores,original=original,negative=negative)
    save(ROOT/'summary.json',result);print(json.dumps({k:result[k] for k in ['decision','quality_passed','primary','positive_controls']},indent=2))

if __name__=='__main__': audit();analyze()
