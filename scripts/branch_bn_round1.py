"""BN1 fixed-core noisy-readout experiment. Protocol and execution details are archived."""
import argparse,hashlib,json,pickle,time,warnings
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import label
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.exceptions import ConvergenceWarning
from crowbrain.task import Bank,balanced_pairs
from crowbrain.model import VisualRNN
from branch_b_round1 import Probe,continue_test,sha,digest,save
R=Path('artifacts/branch_bn/round1');CS=[.01,.1,1.,10.,100.]
def score(y,p):return float(np.mean(y==p)) # all evaluated label sets explicitly balanced

def prepare():
    seen=set()
    for p in [Path('artifacts/branch_b/round1')/f'{n}_data.npz' for n in ['fit','validation','test']]+[Path('artifacts/branch_b/round2/test_data.npz')]:
        seen.update(digest(im) for im in np.load(p)['images'].reshape(-1,1,32,32))
    configs=[('fit',910001,1000,8000),('validation',910002,1000,2000)]+[(f'test{i:02}',910100+i,200,2000) for i in range(20)]
    manifest={};start=time.monotonic()
    for name,seed,k,n in configs:
        p=R/f'{name}.npz';assert not p.exists()
        bank=Bank(seed,k);images=bank.images.numpy()
        assert np.array_equal([label(im[0]>.499,structure=np.ones((3,3)))[1] for im in images.reshape(-1,1,32,32)],np.repeat(np.arange(2,7),k))
        hs={digest(im) for im in images.reshape(-1,1,32,32)};assert len(hs)==5*k and not seen.intersection(hs);seen.update(hs)
        rng=np.random.default_rng(seed+10000);pairs=balanced_pairs(n//40);rng.shuffle(pairs);a,b=pairs.T
        ia,ib=rng.integers(0,k,(2,n));same=(a==b)&(ia==ib);ib[same]=(ib[same]+1)%k;y=(a==b).astype(np.int64)
        assert np.bincount(y).tolist()==[n//2]*2
        for mask in [np.ones(n,bool),y==1,y==0]:assert len(set(np.bincount(a[mask])))==1
        np.savez_compressed(p,images=images,a=a,b=b,ia=ia,ib=ib,y=y,seed=seed)
        manifest[name]={'seed':seed,'images':5*k,'trials':n,'sha256':sha(p),'objects_verified':True,'image_sets_disjoint':True}
        print(json.dumps({'prepared':name,'seconds':time.monotonic()-start}),flush=True)
    save(R/'data_manifest.json',manifest)

@torch.no_grad()
def encode_states(model,d):
    images=torch.from_numpy(d['images']);k=images.shape[1]
    e=torch.cat([model.encode(im) for im in images.flatten(0,1).split(128)]).reshape(5,k,16)
    es=e[d['a'],d['ia']];et=e[d['b'],d['ib']];pre=[];end=[]
    for a,b in zip(es.split(256),et.split(256)):
        _,h,_=model(a,b,delay=15,encoded=True,return_states=True)
        pre.append(h[:,18].numpy());end.append(h[:,-1].numpy())
    return np.concatenate(pre),np.concatenate(end),et

def noisy(model,pre,et,seed,rep,sigma):
    z=np.random.default_rng(np.random.SeedSequence([int(seed),20000,rep])).normal(size=pre.shape).astype(np.float32)
    h=np.maximum(pre+z*sigma,0).astype(np.float32)
    return h,continue_test(model,torch.from_numpy(h),et).numpy()

def mixture(clean,noise):
    n=len(clean);x=np.concatenate([clean]+noise)
    w=np.concatenate([np.full(n,.5)]+[np.full(n,.125) for _ in noise])
    assert len(noise)==4 and abs(w.sum()-n)<1e-9
    return x,w

def optimize(xx,y,w,c):
    history=[]
    for limit in [3000,15000]:
        clf=LogisticRegression(C=c,solver='lbfgs',tol=1e-7,max_iter=limit)
        with warnings.catch_warnings(record=True) as caught:warnings.simplefilter('always');clf.fit(xx,y,sample_weight=w)
        ok=not any(issubclass(z.category,ConvergenceWarning) for z in caught)
        history.append({'max_iter':limit,'iterations':clf.n_iter_.tolist(),'converged':ok,'warnings':[str(z.message) for z in caught]})
        if ok:return clf,history
    raise RuntimeError(f'Unconverged weighted logistic regression C={c}')

def fit_head(clean,x,y,w,vx,vy,vw,cs=CS):
    sc=StandardScaler().fit(clean);xx=sc.transform(x).astype(np.float64);vv=sc.transform(vx).astype(np.float64)
    best=None;loss=float('inf');meta=[]
    for c in cs:
        clf,hist=optimize(xx,y,w,c)
        val=float(log_loss(vy,clf.predict_proba(vv),sample_weight=vw))
        train=float(log_loss(y,clf.predict_proba(xx),sample_weight=w))
        meta.append({'C':c,'validation_loss':val,'training_loss':train,'history':hist})
        if val<loss:loss=val;best=Probe(sc,clf);selected=c
    return best,{'C':selected,'validation_loss':loss,'grid':meta,'weight_sum':float(w.sum()),'n_original':len(clean)}

def run(seed):
    torch.set_num_threads(1);start=time.monotonic();out=R/f'seed{seed}';out.mkdir(exist_ok=False)
    cp=Path(f'artifacts/round6/final/rnn_lr0.001_seed{seed}/final.pt');hp=Path(f'artifacts/branch_b/round1/seed{seed}/probes.pkl')
    oldr=json.loads(Path(f'artifacts/branch_b/round1/seed{seed}/result.json').read_text());sigma=.01*oldr['noise_scale_rms']
    model=VisualRNN(normalize=True);model.load_state_dict(torch.load(cp,weights_only=True));model.eval();model.requires_grad_(False)
    before={k:v.clone() for k,v in model.state_dict().items()}
    with hp.open('rb') as f:old=pickle.load(f)['head_15']
    manifest={'seed':seed,'checkpoint':str(cp),'checkpoint_sha256':sha(cp),'old_head_sha256':sha(hp),'script_sha256':sha(__file__),'protocol_sha256':sha(R/'protocol.md'),'sigma_primary':sigma,'started_at':time.time()}
    save(out/'manifest.json',manifest)
    d={k:dict(np.load(R/f'{k}.npz')) for k in ['fit','validation']};cached={};noise_cache={}
    for part in d:
        pre,end,et=encode_states(model,d[part]);cached[part]=(pre,end,et)
        noise_cache[part]=[noisy(model,pre,et,int(d[part]['seed']),r,sigma) for r in range(4)]
    # Cached and pixel forward + intervention equivalence, using actual new inputs.
    with torch.no_grad():
        f=d['validation'];s=torch.from_numpy(f['images'][f['a'][:32],f['ia'][:32]]);t=torch.from_numpy(f['images'][f['b'][:32],f['ib'][:32]])
        l,h,_=model(s,t,delay=15,return_states=True);l2,h2,_=model(model.encode(s),model.encode(t),delay=15,encoded=True,return_states=True)
        assert torch.equal(l,l2) and torch.equal(h,h2)
        assert torch.equal(continue_test(model,h[:,18],model.encode(t)),h[:,-1])
    heads={'old':old};records={};training={}
    for name in ['clean','aug']:
        packs=[]
        for part in d:
            pre,end,et=cached[part];y=d[part]['y']
            if name=='clean':x,w=end,np.ones(len(end));yy=y
            else:x,w=mixture(end,[q[1] for q in noise_cache[part]]);yy=np.tile(y,5)
            packs.append((x,yy,w))
        a,b=packs;head,meta=fit_head(cached['fit'][1],*a,*b);heads[name]=head;records[name]=meta;training[name]=packs
        print(json.dumps({'seed':seed,'fitted':name,'C':meta['C'],'validation_loss':meta['validation_loss'],'seconds':time.monotonic()-start}),flush=True)
        save(out/'fit_progress.json',records)
    # Sample count probes on the same augmented distributions at two positions.
    for pos in [0,1]:
        packs=[]
        for part in d:
            x,w=mixture(cached[part][pos],[q[pos] for q in noise_cache[part]])
            packs.append((x,np.tile(d[part]['a'],5),w))
        head,meta=fit_head(cached['fit'][pos],*packs[0],*packs[1]);heads[f'quantity{pos}']=head;records[f'quantity{pos}']=meta
        print(json.dumps({'seed':seed,'fitted':f'quantity{pos}','C':meta['C'],'seconds':time.monotonic()-start}),flush=True);save(out/'fit_progress.json',records)
    # Predeclared smaller-data diagnostic, preserving count-pair proportions.
    for name in ['clean','aug']:
        x,y,w=training[name][0];v=training[name][1];n=len(d['fit']['y'])
        for frac in [.25,.5]:
            ids=np.concatenate([np.flatnonzero((d['fit']['a']==a)&(d['fit']['b']==b))[:int(((d['fit']['a']==a)&(d['fit']['b']==b)).sum()*frac)] for a in range(5) for b in range(5)])
            expanded=np.concatenate([ids+i*n for i in range(1 if name=='clean' else 5)])
            head,meta=fit_head(cached['fit'][1][ids],x[expanded],y[expanded],w[expanded],*v,cs=[records[name]['C']]);heads[f'{name}_{frac}']=head;records[f'{name}_{frac}']=meta
    pre=cached['fit'][0];scales={'primary':sigma,'legacy5pct':5*sigma,'rms15':.01*float(np.sqrt(np.mean(pre**2))),'centered15':.01*float(np.sqrt(np.mean((pre-pre.mean(0))**2)))}
    np.savez_compressed(out/'fit_states.npz',pre=pre,end=cached['fit'][1],noisy_pre=np.stack([q[0] for q in noise_cache['fit']]),noisy_end=np.stack([q[1] for q in noise_cache['fit']]))
    save(out/'fits.json',records)
    with (out/'heads.pkl').open('wb') as f:pickle.dump(heads,f)
    result={'seed':seed,'scales':scales,'banks':[],'fit_seconds':time.monotonic()-start}
    for bank in range(20):
        name=f'test{bank:02}';data=dict(np.load(R/f'{name}.npz'));y=data['y'];a=data['a'];b=data['b'];pre,end,et=encode_states(model,data)
        hnames=['old','clean','aug','clean_0.25','clean_0.5','aug_0.25','aug_0.5']
        cleanpred=np.stack([heads[k].predict(end) for k in hnames]);row={'bank':bank,'clean':{k:score(y,p) for k,p in zip(hnames,cleanpred)},'noise':{},'controls':{},'quantity':{}}
        primarypred=[];qp=[];raw={}
        for condition,sd in scales.items():
            scores={k:[] for k in hnames if condition=='primary' or k in ['old','clean','aug']}
            for rep in range(10):
                npre,nend=noisy(model,pre,et,int(data['seed']),rep,sd)
                pp=np.stack([heads[k].predict(nend) for k in scores])
                for k,p in zip(scores,pp):scores[k].append(score(y,p))
                if condition=='primary':
                    primarypred.append(pp);qp.append(np.stack([heads['quantity0'].predict(npre),heads['quantity1'].predict(nend)]))
                    if rep==0:raw={'noisy_pre0':npre,'noisy_end0':nend}
            row['noise'][condition]=scores
        primarypred=np.stack(primarypred);qp=np.stack(qp)
        reset=continue_test(model,torch.zeros_like(torch.from_numpy(pre)),et).numpy()
        _,noisyreset=noisy(model,np.zeros_like(pre),et,int(data['seed']),0,sigma)
        for k in ['clean','aug']:
            pred=primarypred[0,hnames.index(k)];c={'reset':score(y,heads[k].predict(reset)),'noisy_reset':score(y,heads[k].predict(noisyreset)),'label_shuffle':[],'state_shuffle':[]}
            for j in range(64):
                rng=np.random.default_rng(np.random.SeedSequence([int(data['seed']),30000,j]));c['label_shuffle'].append(score(rng.permutation(y),pred));ids=np.arange(len(y))
                for val in range(5):ii=np.flatnonzero(b==val);ids[ii]=rng.permutation(ii)
                c['state_shuffle'].append(score(y,pred[ids]))
            row['controls'][k]=c
        for pos in [0,1]:
            pred=qp[:,pos];q={'all':float(np.mean(pred==a)),'match':float(np.mean(pred[:,y==1]==a[y==1])),'nonmatch':float(np.mean(pred[:,y==0]==a[y==0])),'conditional_permutation':[]}
            for j in range(64):
                rng=np.random.default_rng(np.random.SeedSequence([int(data['seed']),40000,j]));pa=a.copy()
                for val in range(5):ii=np.flatnonzero(b==val);pa[ii]=rng.permutation(pa[ii])
                q['conditional_permutation'].append(float(np.mean(pred==pa)))
            row['quantity'][str(pos)]=q
        np.savez_compressed(out/f'bank{bank:02}.npz',pre=pre,end=end,clean_predictions=cleanpred,primary_predictions=primarypred,quantity_predictions=qp,**raw)
        result['banks'].append(row);save(out/'progress.json',result)
        if (bank+1)%5==0:print(json.dumps({'seed':seed,'banks_complete':bank+1,'seconds':time.monotonic()-start}),flush=True)
    result['integrity']={'state_dict_unchanged':all(torch.equal(before[k],v) for k,v in model.state_dict().items()),'checkpoint_unchanged':sha(cp)==manifest['checkpoint_sha256'],'old_head_unchanged':sha(hp)==manifest['old_head_sha256'],'forward_exact':True}
    assert all(result['integrity'].values());result['seconds']=time.monotonic()-start;save(out/'result.json',result)
    print(json.dumps({'completed':seed,'seconds':result['seconds']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--seed',type=int);args=p.parse_args()
    if args.prepare:prepare()
    else:
        try:run(args.seed)
        except Exception as e:
            save(R/f'seed{args.seed}'/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
