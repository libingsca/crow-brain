"""Independent artifact checks; also export portable linear probe coefficients."""
import hashlib,json,pickle
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from branch_b_round1 import Probe  # trusted local pickle resolves __main__.Probe here
R=Path('artifacts/branch_b/round1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(seeds=range(301,306)):
    torch.set_num_threads(1)
    dm=json.loads((R/'data_manifest.json').read_text())
    for name,row in dm['partitions'].items():assert sha(R/f'{name}_data.npz')==row['file_sha256']
    data={k:dict(np.load(R/f'{k}_data.npz')) for k in ['fit','validation','test']}
    rows=[]
    for seed in seeds:
        p=R/f'seed{seed}';r=json.loads((p/'result.json').read_text());m=json.loads((p/'manifest.json').read_text())
        assert sha(m['checkpoint'])==m['checkpoint_sha256']
        assert sha('scripts/branch_b_round1.py')==m['script_sha256']
        assert sha(R/'protocol.md')==m['protocol_sha256']
        fits=json.loads((p/'fits.json').read_text())
        for name,f in fits.items():
            last={entry['C']:entry for entry in f['search']}
            assert all(entry['converged'] for entry in last.values())
            selected=max(last,key=lambda c:(last[c]['validation_ba'],-c))
            assert f['C']==selected
        predictions=dict(np.load(p/'predictions.npz'))
        with (p/'probes.pkl').open('rb') as f:probes=pickle.load(f)
        portable={}
        for name,probe in probes.items():
            for k,v in [('mean',probe.scaler.mean_),('scale',probe.scaler.scale_),('coef',probe.clf.coef_),('intercept',probe.clf.intercept_),('classes',probe.clf.classes_)]:portable[f'{name}_{k}']=v
        np.savez_compressed(p/'portable_probes.npz',**portable)
        for delay in ['5','15']:
            states=dict(np.load(p/f'states_delay{delay}.npz'))
            head=probes[f'head_{delay}'];x=states['test'][:,-1]
            assert np.array_equal(head.predict(x),predictions[f'rescue_{delay}'])
            assert abs(balanced_accuracy_score(data['test']['y'],head.predict(x))-r['delays'][delay]['rescue_ba'])<1e-12
            assert np.allclose(head.scaler.mean_,states['fit'][:,-1].mean(0,dtype=np.float64),atol=1e-12)
            w=torch.load(m['checkpoint'],weights_only=True)
            old=(torch.from_numpy(x)@w['readout.weight'].T+w['readout.bias']).argmax(1).numpy()
            assert np.array_equal(old,predictions[f'original_{delay}'])
            for repeat in r['noise'][f'delay{delay}_fraction0']:
                assert repeat['original_ba']==r['delays'][delay]['original_ba']
                assert repeat['rescue_ba']==r['delays'][delay]['rescue_ba']
        rows.append({'seed':seed,'checkpoint_and_code_hashes':True,'selected_fits_converged':True,'C_selection_verified':True,'raw_predictions_recomputed':True,'scaler_fit_partition_verified':True,'zero_noise_matches_clean':True,'portable_probes_exported':True,'n_probes':len(probes)})
    (R/('audit.json' if len(seeds)==5 else 'audit_'+ '_'.join(map(str,seeds))+'.json')).write_text(json.dumps({'data_hashes_verified':True,'models':rows},indent=2)+'\n')
    print(json.dumps({'models_checked':len(rows),'all_checks_passed':True}))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=list(range(301,306)))
    main(p.parse_args().seeds)
