import json,pickle
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from branch_b_round1 import Probe,sha,ba
R=Path('artifacts/branch_b/round2');B=Path('artifacts/branch_b/round1')
def main():
    torch.set_num_threads(1)
    dm=json.loads((R/'data_manifest.json').read_text());assert sha(R/'test_data.npz')==dm['file_sha256']
    data=np.load(R/'test_data.npz');y=data['y'];checks=[]
    assert np.bincount(y).tolist()==[2000,2000]
    for s in range(301,306):
        p=R/f'seed{s}';r=json.loads((p/'result.json').read_text());a=dict(np.load(p/'raw.npz'))
        assert sha('scripts/branch_b_round2.py')==r['integrity']['script_sha256']
        assert sha(R/'protocol.md')==r['integrity']['protocol_sha256']
        cp=Path(f'artifacts/round6/final/rnn_lr0.001_seed{s}/final.pt');hp=B/f'seed{s}'/'probes.pkl'
        assert sha(cp)==r['integrity']['checkpoint_sha256'];assert sha(hp)==r['integrity']['head_file_sha256']
        with hp.open('rb') as f:head=pickle.load(f)['head_15']
        assert np.array_equal(head.predict(a['states15'][:,-1]),a['rescue'])
        assert ba(y,a['rescue'])==r['rescue15_ba']
        w=torch.load(cp,weights_only=True)
        original=(torch.from_numpy(a['states15'][:,-1])@w['readout.weight'].T+w['readout.bias']).argmax(1).numpy()
        assert np.array_equal(original,a['original']);assert ba(y,original)==r['original15_ba']
        tr=np.load(B/f'seed{s}'/'states_delay15.npz')['fit'][:,-1];scaler=StandardScaler().fit(tr)
        assert np.array_equal(scaler.mean_,a['scaler_mean']);assert np.array_equal(scaler.scale_,a['scaler_scale'])
        xx=scaler.transform(a['states15'][:,-1]).astype(np.float64)
        nullpred=(xx@a['null_coef'][:,0,:].T+a['null_intercept'][:,0])>0
        assert np.array_equal(nullpred.T,a['null_predictions'])
        assert all(ba(y,pred)==score for pred,score in zip(a['null_predictions'],r['null_train']))
        fs=json.loads((p/'fits.json').read_text())
        assert all(next(iter(f.values())) is not None for f in fs)
        for f in fs:
            history=f['history'] if 'history' in f else next(iter(f.values()))
            assert history[-1]['converged']
        checks.append({'seed':s,'all_hashes_verified':True,'frozen_head_predictions_recomputed':True,'original_head_recomputed':True,'all_256_null_predictions_recomputed':True,'training_scaler_recomputed':True,'all_fits_converged':True})
    (R/'audit.json').write_text(json.dumps({'data_hash_and_label_balance_verified':True,'models':checks},indent=2)+'\n')
    print(json.dumps({'models':len(checks),'all_checks_passed':True}))
if __name__=='__main__':main()
