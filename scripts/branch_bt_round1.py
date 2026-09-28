"""B-T frozen-head time transfer; no fitting. See archived round1 protocol."""
import argparse, hashlib, json, pickle, time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from scipy.ndimage import label
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from crowbrain.task import Bank, balanced_pairs
from crowbrain.model import VisualRNN
from branch_b_round1 import Probe, continue_test, sha, digest, save

ROOT=Path('artifacts/branch_bt/round1')
DELAYS=[5,7,10,13,15]
HEADS=['clean','aug']

def previous_data():
    return [Path('artifacts/branch_b/round1')/f'{p}_data.npz' for p in ['fit','validation','test']]+[Path('artifacts/branch_b/round2/test_data.npz')]+[Path('artifacts/branch_bn/round1')/f'{p}.npz' for p in ['fit','validation']+[f'test{i:02d}' for i in range(20)]]

def prepare():
    start=time.monotonic(); seen=set(); old={}
    for p in previous_data():
        with np.load(p) as d: hashes={digest(im) for im in d['images'].reshape(-1,1,32,32)}
        seen.update(hashes); old[str(p)]=sha(p)
    records=[]
    for k,seed in enumerate(range(920100,920120)):
        images=Bank(seed,200).images.numpy()
        hashes={digest(im) for im in images.reshape(-1,1,32,32)}
        assert len(hashes)==1000 and not seen.intersection(hashes)
        seen.update(hashes)
        counts=[label(im[0]>.499,structure=np.ones((3,3)))[1] for im in images.reshape(-1,1,32,32)]
        assert np.array_equal(counts,np.repeat(np.arange(2,7),200))
        rng=np.random.default_rng(seed+10000); pairs=balanced_pairs(50);rng.shuffle(pairs)
        a,b=pairs.T;ia,ib=rng.integers(0,200,(2,2000));same=(a==b)&(ia==ib);ib[same]=(ib[same]+1)%200
        y=(a==b).astype(np.int64)
        assert np.bincount(y).tolist()==[1000,1000]
        assert all(np.bincount(v).tolist()==[400]*5 for v in [a,b])
        assert not np.any((a==b)&(ia==ib))
        p=ROOT/f'test{k:02d}.npz';np.savez_compressed(p,images=images,a=a,b=b,ia=ia,ib=ib,y=y,seed=seed)
        records.append({'seed':seed,'path':str(p),'sha256':sha(p),'objects_verified':True})
        print(json.dumps({'prepared':k,'seconds':time.monotonic()-start}),flush=True)
    save(ROOT/'data_manifest.json',{'banks':records,'prior_files':old,'all_new_pixels_disjoint':True,'prior_unique_images':len(seen)-20000})

@torch.no_grad()
def encoded_pairs(model,d):
    enc=torch.cat([model.encode(x) for x in torch.from_numpy(d['images']).flatten(0,1).split(128)]).reshape(5,d['images'].shape[1],16)
    return enc[d['a'],d['ia']],enc[d['b'],d['ib']]

@torch.no_grad()
def extract(model,es,et,delay):
    pre=[];end=[]
    for s,t in zip(es.split(256),et.split(256)):
        _,h,_=model(s,t,delay=delay,encoded=True,return_states=True)
        pre.append(h[:,3+delay]);end.append(h[:,-1])
    return torch.cat(pre),torch.cat(end)

@torch.no_grad()
def perturb(model,pre,et,bank_seed,rep,sigma):
    z=np.random.default_rng(np.random.SeedSequence([int(bank_seed),20000,rep])).normal(size=tuple(pre.shape)).astype(np.float32)
    h=torch.clamp(pre+torch.from_numpy(z)*sigma,min=0)
    return h,continue_test(model,h,et)

def frozen_snapshot(model,heads):
    result={f'core/{k}':digest(v.detach().numpy()) for k,v in model.state_dict().items()}
    for name,head in heads.items():
        for part in ['scaler','clf']:
            for k,v in vars(getattr(head,part)).items():
                if isinstance(v,np.ndarray): result[f'{name}/{part}/{k}']=digest(v)
                elif isinstance(v,(str,int,float,bool,type(None),np.generic)): result[f'{name}/{part}/{k}']=str(v)
    return result

@torch.no_grad()
def check_forward(model,d,es,et,delay,pre,end):
    s=torch.from_numpy(d['images'][d['a'][:32],d['ia'][:32]])
    t=torch.from_numpy(d['images'][d['b'][:32],d['ib'][:32]])
    _,pixel,_=model(s,t,delay=delay,return_states=True)
    _,cached,_=model(es[:32],et[:32],delay=delay,encoded=True,return_states=True)
    _,reset,_=model(s,t,delay=delay,reset=True,return_states=True)
    pairs=[(pixel,cached),(pre[:32],pixel[:,3+delay]),(end[:32],pixel[:,-1]),
           (continue_test(model,pre[:32],et[:32]),end[:32]),
           (continue_test(model,torch.zeros_like(pre[:32]),et[:32]),reset[:,-1])]
    errors=[]
    for a,b in pairs:
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5);errors.append(float((a-b).abs().max()))
    return errors

def permutation_indices(d):
    n=len(d['y']);ys=[];rows=[]
    for rep in range(64):
        ys.append(np.random.default_rng(np.random.SeedSequence([int(d['seed']),30000,rep])).permutation(n))
        rng=np.random.default_rng(np.random.SeedSequence([int(d['seed']),40000,rep]));row=np.arange(n)
        for b in range(5):
            ids=np.flatnonzero(d['b']==b);row[ids]=rng.permutation(ids)
        rows.append(row)
    return np.asarray(ys),np.asarray(rows)

@torch.no_grad()
def run(seed):
    torch.set_num_threads(1);start=time.monotonic();out=ROOT/f'seed{seed}';out.mkdir(exist_ok=False)
    checkpoint=Path(f'artifacts/round6/final/rnn_lr0.001_seed{seed}/final.pt')
    hp=Path(f'artifacts/branch_bn/round1/seed{seed}/heads.pkl');mp=hp.with_name('manifest.json')
    model=VisualRNN(normalize=True);model.load_state_dict(torch.load(checkpoint,weights_only=True));model.eval();model.requires_grad_(False)
    with hp.open('rb') as f: loaded=pickle.load(f)
    heads={h:loaded[h] for h in HEADS};sigma=json.loads(mp.read_text())['sigma_primary']
    inputs={str(p):sha(p) for p in [checkpoint,hp,mp,Path(__file__),ROOT/'protocol.md',ROOT/'execution_details.md']}
    before=frozen_snapshot(model,heads)
    save(out/'manifest.json',{'seed':seed,'sigma_primary':sigma,'input_sha256':inputs,'before':before,'started_at':time.time()})
    results=[];audits=[]
    with patch.object(LogisticRegression,'fit',side_effect=RuntimeError('B-T forbids fitting')),patch.object(StandardScaler,'fit',side_effect=RuntimeError('B-T forbids fitting')):
        for bank in range(20):
            path=ROOT/f'test{bank:02d}.npz';d=dict(np.load(path));es,et=encoded_pairs(model,d);y=d['y'];yi,ri=permutation_indices(d)
            for delay in DELAYS:
                pre,end=extract(model,es,et,delay)
                audits.append({'bank':bank,'delay':delay,'max_errors':check_forward(model,d,es,et,delay,pre,end)})
                predictions=np.empty((2,11,len(y)),dtype=np.int8)
                original=np.empty((11,len(y)),dtype=np.int8)
                states=[end]
                for rep in range(10):
                    noisy_pre,noisy_end=perturb(model,pre,et,int(d['seed']),rep,sigma)
                    if rep==0: noise_pre0=noisy_pre.numpy();noise_end0=noisy_end.numpy()
                    states.append(noisy_end)
                for j,state in enumerate(states):
                    for i,h in enumerate(HEADS): predictions[i,j]=heads[h].predict(state.numpy())
                    original[j]=model.readout(state).argmax(1).numpy()
                reset=continue_test(model,torch.zeros_like(pre),et)
                _,reset_noise=perturb(model,torch.zeros_like(pre),et,int(d['seed']),0,sigma)
                resetpred=np.asarray([[heads[h].predict(s.numpy()) for s in [reset,reset_noise]] for h in HEADS],dtype=np.int8)
                # Predicting independently shuffled rows equals indexing their deterministic predictions.
                labelpred=predictions[:,[0,1],:][:,:,None,:].repeat(64,axis=2)
                rowpred=predictions[:,[0,1],:][:,:,ri]
                neg=np.stack([(resetpred==y).mean(-1),(labelpred==y[yi]).mean((-1,-2)),(rowpred==y).mean((-1,-2))],axis=-1)
                scores=(predictions==y).mean(-1);origscore=(original==y).mean(-1)
                result={'bank':bank,'delay':delay,'scores':scores.tolist(),'original_scores':origscore.tolist(),'negative_means':neg.tolist()}
                results.append(result)
                np.savez_compressed(out/f'bank{bank:02d}_d{delay:02d}.npz',pre=pre.numpy(),end=end.numpy(),noise_pre0=noise_pre0,noise_end0=noise_end0,predictions=predictions,original=original,reset=resetpred,label_indices=yi,row_indices=ri,label_predictions=labelpred,row_predictions=rowpred)
            print(json.dumps({'seed':seed,'bank':bank,'seconds':time.monotonic()-start}),flush=True)
    after=frozen_snapshot(model,heads);assert before==after
    assert all(sha(p)==v for p,v in inputs.items())
    save(out/'results.json',{'seed':seed,'cells':results,'frozen_before_after_equal':True,'disk_inputs_unchanged':True,'fit_guard_enabled':True,'forward_audits':audits,'after':after,'seconds':time.monotonic()-start})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--seed',type=int);args=parser.parse_args()
    if args.prepare: prepare()
    if args.seed is not None: run(args.seed)
