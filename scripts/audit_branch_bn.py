"""Recompute BN scores and interventions from saved artifacts; no new model selection."""
import json,pickle
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from branch_bn_round1 import Probe,sha,score,noisy,encode_states
from crowbrain.model import VisualRNN
R=Path('artifacts/branch_bn/round1')
def main(seeds):
    torch.set_num_threads(1);checks=[]
    for name,m in json.loads((R/'data_manifest.json').read_text()).items():assert sha(R/f'{name}.npz')==m['sha256']
    for s in seeds:
        p=R/f'seed{s}';r=json.loads((p/'result.json').read_text());m=json.loads((p/'manifest.json').read_text());fits=json.loads((p/'fits.json').read_text())
        assert sha(m['checkpoint'])==m['checkpoint_sha256'];assert sha('scripts/branch_bn_round1.py')==m['script_sha256'];assert sha(R/'protocol.md')==m['protocol_sha256']
        assert sha(f'artifacts/branch_b/round1/seed{s}/probes.pkl')==m['old_head_sha256']
        assert sha(m['solver_override_path'])==m['solver_override_sha256']
        assert sha('docs/decisions/branch_bn/002_solver_repair.md')==m['solver_protocol_sha256']
        with (p/'heads.pkl').open('rb') as f:heads=pickle.load(f)
        for k,f in fits.items():
            assert all(g['history'][-1]['converged'] for g in f['grid'])
            assert f['C']==min(f['grid'],key=lambda z:(z['validation_loss'],z['C']))['C']
            assert f['weight_sum']==f['n_original']
        fs=np.load(p/'fit_states.npz');sc=StandardScaler().fit(fs['end'])
        for name in ['clean','aug']:
            assert np.array_equal(heads[name].scaler.mean_,sc.mean_);assert np.array_equal(heads[name].scaler.scale_,sc.scale_)
        gradients={}
        fit_y=np.load(R/'fit.npz')['y']
        for name in ['clean','aug']:
            xx=fs['end'] if name=='clean' else np.concatenate([fs['end']]+list(fs['noisy_end']))
            ww=np.ones(len(fit_y)) if name=='clean' else np.concatenate([np.full(len(fit_y),.5)]+[np.full(len(fit_y),.125)]*4)
            yy=fit_y if name=='clean' else np.tile(fit_y,5)
            xx=heads[name].scaler.transform(xx).astype(np.float64);cl=heads[name].clf
            beta=cl.coef_[0];err=ww*(expit(xx@beta+cl.intercept_[0])-yy)/ww.sum()
            grad=np.r_[xx.T@err+beta/(cl.C*ww.sum()),err.sum()]
            gradients[name]=float(abs(grad).max());assert gradients[name]<1e-6
        model=VisualRNN(normalize=True);model.load_state_dict(torch.load(m['checkpoint'],weights_only=True));model.eval();model.requires_grad_(False)
        hnames=['old','clean','aug','clean_0.25','clean_0.5','aug_0.25','aug_0.5']
        for b in range(20):
            d=dict(np.load(R/f'test{b:02}.npz'));a=dict(np.load(p/f'bank{b:02}.npz'));row=r['banks'][b];y=d['y']
            assert np.bincount(y).tolist()==[1000,1000]
            for i,k in enumerate(hnames):
                assert np.array_equal(heads[k].predict(a['end']),a['clean_predictions'][i]);assert score(y,a['clean_predictions'][i])==row['clean'][k]
                assert [score(y,z) for z in a['primary_predictions'][:,i]]==row['noise']['primary'][k]
            pre,end,et=encode_states(model,d)
            assert np.array_equal(pre,a['pre']) and np.array_equal(end,a['end'])
            npre,nend=noisy(model,pre,et,int(d['seed']),0,m['sigma_primary'])
            assert np.array_equal(npre,a['noisy_pre0']) and np.array_equal(nend,a['noisy_end0'])
            for i,k in enumerate(hnames):assert np.array_equal(heads[k].predict(nend),a['primary_predictions'][0,i])
            for pos,xx in enumerate([npre,nend]):assert np.array_equal(heads[f'quantity{pos}'].predict(xx),a['quantity_predictions'][0,pos])
        checks.append({'seed':s,'source_and_input_hashes':True,'selected_C_and_convergence':True,'selected_binary_objective_gradient_max':gradients,'objective_weight_sum':True,'shared_training_scaler':True,'all_20_bank_scores_recomputed':True,'all_20_bank_states_and_first_noise_recomputed':True})
    name='audit.json' if len(seeds)==5 else 'audit_'+ '_'.join(map(str,seeds))+'.json'
    (R/name).write_text(json.dumps({'data_hashes_verified':True,'models':checks},indent=2)+'\n');print(json.dumps({'models':len(checks),'all_checks_passed':True}))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=list(range(301,306)));main(p.parse_args().seeds)
