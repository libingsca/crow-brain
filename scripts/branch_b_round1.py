"""Frozen-core delayed-readout rescue; see preregistered B1 protocol."""
import argparse, hashlib, json, time, warnings, pickle
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import label
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.exceptions import ConvergenceWarning
from crowbrain.task import Bank, balanced_pairs
from crowbrain.model import VisualRNN

ROOT=Path('artifacts/branch_b/round1')
DELAYS=[0,5,10,15,25,40]
CS=[.01,.1,1.,10.,100.]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def save(p,x): Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def ba(y,p): return float(balanced_accuracy_score(y,p))

def prepare():
    start=time.monotonic(); sets=[]; manifest={}
    for name,seed,n in [('fit',810001,8000),('validation',810002,2000),('test',810003,4000)]:
        bank=Bank(seed,1000)
        images=bank.images.numpy()
        counts=[label(im[0]>.499,structure=np.ones((3,3)))[1] for im in images.reshape(-1,1,32,32)]
        assert np.array_equal(counts,np.repeat(np.arange(2,7),1000))
        hashes={digest(im) for im in images.reshape(-1,1,32,32)}
        assert len(hashes)==5000
        assert all(not hashes.intersection(s) for s in sets)
        sets.append(hashes)
        rng=np.random.default_rng(seed+100)
        pairs=balanced_pairs(n//40); rng.shuffle(pairs)
        a,b=pairs.T; ia,ib=rng.integers(0,1000,(2,n)); same=(a==b)&(ia==ib); ib[same]=(ib[same]+1)%1000
        y=(a==b).astype(np.int64)
        assert np.bincount(y).tolist()==[n//2,n//2]
        for v in [a,b]: assert np.bincount(v).tolist()==[n//5]*5
        np.savez_compressed(ROOT/f'{name}_data.npz',images=images,a=a,b=b,ia=ia,ib=ib,y=y)
        manifest[name]={'seed':seed,'trials':n,'images':5000,'images_sha256':digest(images),'file_sha256':sha(ROOT/f'{name}_data.npz'),'objects_verified':True}
        print(json.dumps({'prepared':name,'seconds':time.monotonic()-start}),flush=True)
    save(ROOT/'data_manifest.json',{'partitions':manifest,'image_sets_disjoint':True,'source_sha256':{str(p):sha(p) for p in [Path(__file__),Path('src/crowbrain/task.py'),Path('src/crowbrain/model.py')]}})

@torch.no_grad()
def continue_test(model,h,et):
    w=model.weights()
    for _ in range(4): h=.5*h+.5*F.relu(model.input(et)+F.linear(h,w))
    return h

@torch.no_grad()
def states(model,data,delay):
    enc=torch.cat([model.encode(im) for im in torch.from_numpy(data['images']).flatten(0,1).split(128)]).reshape(5,1000,16)
    es=enc[data['a'],data['ia']]; et=enc[data['b'],data['ib']]
    hs=[]; logits=[]
    for a,b in zip(es.split(256),et.split(256)):
        out,h,_=model(a,b,delay=delay,return_states=True,encoded=True)
        hs.append(h[:,3+delay:8+delay].numpy());logits.append(out.numpy())
    return np.concatenate(hs),np.concatenate(logits),et

class Probe:
    def __init__(self,scaler,clf): self.scaler=scaler;self.clf=clf
    def predict(self,x): return self.clf.predict(self.scaler.transform(x))

def fit(x,y,vx,vy):
    scaler=StandardScaler().fit(x); xx=scaler.transform(x).astype(np.float64); vv=scaler.transform(vx).astype(np.float64)
    records=[]; best=None; best_score=-1
    for c in CS:
        for limit in [3000,15000]:
            clf=LogisticRegression(C=c,solver='lbfgs',tol=1e-7,max_iter=limit)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always'); clf.fit(xx,y)
            converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
            records.append({'C':c,'max_iter':limit,'iterations':clf.n_iter_.tolist(),'converged':converged,'warnings':[str(w.message) for w in caught]})
            if converged: break
        if not converged: raise RuntimeError(f'Probe did not converge at C={c}')
        score=ba(vy,clf.predict(vv));records[-1]['validation_ba']=score
        if score>best_score: best=Probe(scaler,clf);best_score=score;selected=c
    return best,{'C':selected,'validation_ba':best_score,'train_ba':ba(y,best.predict(x)),'search':records}

def curve(x,y,vx,vy,tx,ty,c,a,b):
    out=[]
    for frac in [.25,.5]:
        ids=np.concatenate([np.flatnonzero((a==i)&(b==j))[:int(((a==i)&(b==j)).sum()*frac)] for i in range(5) for j in range(5)])
        scaler=StandardScaler().fit(x[ids]);clf=LogisticRegression(C=c,solver='lbfgs',tol=1e-7,max_iter=15000)
        with warnings.catch_warnings(record=True) as caught: warnings.simplefilter('always');clf.fit(scaler.transform(x[ids]).astype(np.float64),y[ids])
        assert not any(issubclass(w.category,ConvergenceWarning) for w in caught)
        p=Probe(scaler,clf)
        out.append({'fraction':frac,'n':len(ids),'train_ba':ba(y[ids],p.predict(x[ids])),'validation_ba':ba(vy,p.predict(vx)),'test_ba':ba(ty,p.predict(tx))})
    return out

def run(seed):
    torch.set_num_threads(1);start=time.monotonic()
    out=ROOT/f'seed{seed}';out.mkdir(exist_ok=False)
    source=Path(f'artifacts/round6/final/rnn_lr0.001_seed{seed}/final.pt')
    model=VisualRNN(normalize=True);model.load_state_dict(torch.load(source,weights_only=True));model.eval();model.requires_grad_(False)
    original={k:v.clone() for k,v in model.state_dict().items()}
    save(out/'manifest.json',{'seed':seed,'checkpoint':str(source),'checkpoint_sha256':sha(source),'script_sha256':sha(__file__),'protocol_sha256':sha(ROOT/'protocol.md'),'started_at':time.time()})
    data={k:dict(np.load(ROOT/f'{k}_data.npz')) for k in ['fit','validation','test']}
    # Verify encoded implementation and test continuation against original forward.
    d=data['test'];s=torch.from_numpy(d['images'][d['a'][:32],d['ia'][:32]]);t=torch.from_numpy(d['images'][d['b'][:32],d['ib'][:32]])
    with torch.no_grad():
        lg,h,_=model(s,t,return_states=True)
        lg2,h2,_=model(model.encode(s),model.encode(t),encoded=True,return_states=True)
        assert torch.equal(lg,lg2) and torch.equal(h,h2)
        assert torch.equal(continue_test(model,h[:,8],model.encode(t)),h[:,-1])
    result={'seed':seed,'integrity':{'forward_exact':True},'delays':{},'cross_time':{},'noise':{},'controls':{}}
    probes={};cache={};allfits={};predictions={}
    for delay in DELAYS:
        arr={};ets={}
        for part in data:
            arr[part],logits,ets[part]=states(model,data[part],delay)
            if part=='test': oldpred=logits.argmax(1)
        ys=[data[k]['y'] for k in data]; xs=[arr[k][:,-1] for k in data]
        head,meta=fit(xs[0],ys[0],xs[1],ys[1]);probes[f'head_{delay}']=head;allfits[f'head_{delay}']=meta
        pred=head.predict(xs[2]);row={'original_ba':ba(ys[2],oldpred),'rescue_ba':ba(ys[2],pred),'selected_C':meta['C'],'validation_ba':meta['validation_ba'],'train_ba':meta['train_ba'],'quantity':{}}
        predictions[f'original_{delay}']=oldpred;predictions[f'rescue_{delay}']=pred
        if delay in [5,15]:
            cache[delay]=(arr,ets)
            row['learning_curve']=curve(xs[0],ys[0],xs[1],ys[1],xs[2],ys[2],meta['C'],data['fit']['a'],data['fit']['b'])
        positions=range(5) if delay in [5,15] else [0]
        for pos in positions:
            qr={}
            for target in (['a','b'] if delay in [5,15] else ['a']):
                xx=[arr[k][:,pos] for k in data]; yy=[data[k][target] for k in data]
                probe,m=fit(xx[0],yy[0],xx[1],yy[1]); key=f'quantity_{delay}_{pos}_{target}';probes[key]=probe;allfits[key]=m
                pp=probe.predict(xx[2]);predictions[key]=pp
                q={'test_ba':ba(yy[2],pp),'train_ba':m['train_ba'],'validation_ba':m['validation_ba'],'C':m['C']}
                for group,mask in [('match',ys[2]==1),('nonmatch',ys[2]==0)]:q[group+'_ba']=ba(yy[2][mask],pp[mask])
                if target=='a':
                    rng=np.random.default_rng(810004); baselines=[]
                    for _ in range(20):
                        perm=yy[2].copy()
                        for b in range(5):
                            ids=np.flatnonzero(data['test']['b']==b);perm[ids]=rng.permutation(perm[ids])
                        baselines.append(ba(perm,pp))
                    q['conditional_permutation_ba']=baselines
                    if delay in [5,15]:q['learning_curve']=curve(xx[0],yy[0],xx[1],yy[1],xx[2],yy[2],m['C'],data['fit']['a'],data['fit']['b'])
                qr[target]=q
            row['quantity'][str(pos)]=qr
        result['delays'][str(delay)]=row
        save(out/'progress.json',result)
        print(json.dumps({'seed':seed,'delay':delay,'old':row['original_ba'],'rescue':row['rescue_ba'],'seconds':time.monotonic()-start}),flush=True)
    for trained in [5,15]:
        for tested in [5,15]:
            arr,_=cache[tested];result['cross_time'][f'head_{trained}_to_{tested}']=ba(data['test']['y'],probes[f'head_{trained}'].predict(arr['test'][:,-1]))
            result['cross_time'][f'sample_{trained}_to_{tested}']=ba(data['test']['a'],probes[f'quantity_{trained}_0_a'].predict(arr['test'][:,0]))
    arr,ets=cache[15]
    perm_y=np.random.default_rng(810004).permutation(data['fit']['y'])
    permhead,meta=fit(arr['fit'][:,-1],perm_y,arr['validation'][:,-1],data['validation']['y']);allfits['permuted_head']=meta;probes['permuted_head']=permhead
    result['controls']['permuted_head_ba']=ba(data['test']['y'],permhead.predict(arr['test'][:,-1]))
    with torch.no_grad():
        reset=continue_test(model,torch.zeros((4000,64)),ets['test']).numpy()
        result['controls']['reset_original_ba']=ba(data['test']['y'],model.readout(torch.from_numpy(reset)).argmax(1).numpy())
        result['controls']['reset_rescue_ba']=ba(data['test']['y'],probes['head_15'].predict(reset))
    scale=float(np.sqrt(np.mean(cache[5][0]['fit'][:,0]**2)));result['noise_scale_rms']=scale
    for delay in [5,15]:
        arr,ets=cache[delay]
        for frac in [0,.01,.05]:
            rng=np.random.default_rng(810004);scores=[]
            for rep in range(10):
                z=rng.normal(size=arr['test'][:,0].shape).astype(np.float32)
                noisy=np.maximum(arr['test'][:,0]+z*(scale*frac),0)
                with torch.no_grad():
                    final=continue_test(model,torch.from_numpy(noisy),ets['test']); old=model.readout(final).argmax(1).numpy()
                new=probes[f'head_{delay}'].predict(final.numpy())
                scores.append({'original_ba':ba(data['test']['y'],old),'rescue_ba':ba(data['test']['y'],new)})
            result['noise'][f'delay{delay}_fraction{frac}']=scores
    result['integrity']['state_dict_unchanged']=all(torch.equal(original[k],v) for k,v in model.state_dict().items())
    result['integrity']['checkpoint_unchanged']=sha(source)==json.loads((out/'manifest.json').read_text())['checkpoint_sha256']
    assert all(result['integrity'].values())
    result['seconds']=time.monotonic()-start
    np.savez_compressed(out/'predictions.npz',**predictions)
    for delay,(arr,ets) in cache.items():np.savez_compressed(out/f'states_delay{delay}.npz',**arr)
    with (out/'probes.pkl').open('wb') as f:pickle.dump(probes,f)
    save(out/'fits.json',allfits);save(out/'result.json',result)
    print(json.dumps({'completed':seed,'seconds':result['seconds']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--seed',type=int);a=p.parse_args()
    if a.prepare:prepare()
    else:
        try:run(a.seed)
        except Exception as e:
            save(ROOT/f'seed{a.seed}'/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
