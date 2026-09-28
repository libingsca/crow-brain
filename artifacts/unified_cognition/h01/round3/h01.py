"""UC H01: structured delayed relation task; isolated from historical branches."""
import argparse, hashlib, json, math, time, platform
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy import stats

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/unified_cognition/h01'

def dump(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

# Pair ID is first_object*16 + second_object; object=color*4+shape.
PAIRS=np.array([(a//4,a%4,b//4,b%4) for a in range(16) for b in range(16)],dtype=np.int64)
REL=np.stack([PAIRS[:,0]==PAIRS[:,2],PAIRS[:,1]==PAIRS[:,3]],1).astype(np.int64)
HELD=(PAIRS[:,0]==PAIRS[:,1]) | (PAIRS[:,2]==PAIRS[:,3])
POOLS={(ood,c,s):np.flatnonzero((HELD==ood)&(REL[:,0]==c)&(REL[:,1]==s)) for ood in (False,True) for c in (0,1) for s in (0,1)}
assert all(len(v)>0 for v in POOLS.values())

def partition(rec):
    # Injective mixed-radix ID over all seven discrete factors; splitmix64 hash.
    a,b,u,q,d,p0,p1=rec.T.astype(np.uint64)
    key=((((((a*256+b)*2+u)*2+q)*10+d)*2+p0)*2+p1)
    with np.errstate(over='ignore'):
        z=key+np.uint64(0x9e3779b97f4a7c15)
        z=(z^(z>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
        z=(z^(z>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
        z=z^(z>>np.uint64(31))
    return (z%10).astype(int)

def sample(rng,n,split='train',ood=False,delay=4):
    # Exact balance update x query x both relevant relations (16 strata).
    assert n%16==0
    out=[]
    for u in (0,1):
      for q in (0,1):
       for c in (0,1):
        for s in (0,1):
         chunks=[]; left=n//16
         while left:
            m=max(32,left*12 if split!='train' else left*2)
            rel=rng.choice(POOLS[ood,c,s],m)
            irr_c=rng.integers(2,size=m); irr_s=rng.integers(2,size=m)
            irr=np.empty(m,dtype=int)
            for ic in (0,1):
             for iss in (0,1):
              ix=(irr_c==ic)&(irr_s==iss)
              irr[ix]=rng.choice(POOLS[False,ic,iss],ix.sum())
            a,b=(rel,irr) if u==0 else (irr,rel)
            rec=np.column_stack([a,b,np.full(m,u),np.full(m,q),np.full(m,delay),rng.integers(2,size=m),rng.integers(2,size=m)])
            bucket=partition(rec)
            valid=bucket<8 if split=='train' else bucket==(8 if split=='val' else 9)
            v=rec[valid][:left]; chunks.append(v); left-=len(v)
         out.append(np.concatenate(chunks))
    rec=np.concatenate(out).astype(np.int16); rng.shuffle(rec)
    return rec

def labels(rec):
    a,b,u,q=rec[:,:4].T
    return REL[np.where(u,b,a),q]

def encode(rec):
    delay=int(rec[0,4]); assert np.all(rec[:,4]==delay)
    n=len(rec); x=np.zeros((n,delay+4,24),np.float32)
    for t,col,flip in ((0,0,5),(2,1,6)):
        p=PAIRS[rec[:,col]]
        for j,off in ((0,0),(1,10)):
            x[np.arange(n),t,off+p[:,2*j]]=1
            x[np.arange(n),t,off+4+p[:,2*j+1]]=1
            x[np.arange(n),t,off+8+((rec[:,flip]+j)%2)]=1
    x[np.arange(n),1,20+rec[:,2]]=1
    x[np.arange(n),-1,22+rec[:,3]]=1
    return torch.from_numpy(x),torch.from_numpy(labels(rec).copy()).long()

class Net(nn.Module):
    def __init__(self,kind='fused',width=64):
        super().__init__(); self.kind=kind; self.width=width
        if kind=='split':
            self.rw=width//4; self.cw=width-self.rw
            self.content=nn.GRU(22,self.cw,batch_first=True)
            self.rule=nn.GRU(2,self.rw,batch_first=True)
        else: self.core=nn.GRU(24,width,batch_first=True)
        self.head=nn.Sequential(nn.Linear(width,32),nn.Tanh(),nn.Linear(32,2))
    def states(self,x):
        if self.kind=='split':
            c,_=self.content(x[:,:,:22]); r,_=self.rule(x[:,:,22:]); return torch.cat([c,r],-1)
        return self.core(x)[0]
    def forward(self,x): return self.head(self.states(x)[:,-1])
    def reset(self,x):
        # Clear ALL memory immediately before the question; run question from zero.
        return self.forward(x[:,-1:])

def nparams(m): return sum(p.numel() for p in m.parameters())

def model_specs():
    target=nparams(Net('split',64))
    w=min(range(16,65),key=lambda z:abs(nparams(Net('fused',z))-target))
    return {'F64':('fused',64),'S64':('split',64),'FP':('fused',w)}

@torch.no_grad()
def evaluate(model,rec,save=None):
    logits=[]; reset=[]
    for start in range(0,len(rec),512):
        x,y=encode(rec[start:start+512]); logits.append(model(x).numpy()); reset.append(model.reset(x).argmax(1).numpy())
    logits=np.concatenate(logits); pred=logits.argmax(1); y=labels(rec)
    loss=float(nn.functional.cross_entropy(torch.from_numpy(logits),torch.from_numpy(y)).item())
    m={'accuracy':float((pred==y).mean()),'loss':loss,'reset_accuracy':float((np.concatenate(reset)==y).mean())}
    for u in (0,1):
        ix=rec[:,2]==u;m['update' if u else 'hold']=float((pred[ix]==y[ix]).mean())
    if save: np.savez_compressed(save,records=rec,labels=y,logits=logits,reset=np.concatenate(reset))
    return m

def run(args):
    spec=model_specs(); kind,w=spec[args.model]; torch.manual_seed(args.seed)
    model=Net(kind,w); optim=torch.optim.Adam(model.parameters(),lr=args.lr)
    out=BASE/args.round/f'{args.model}_s{args.seed}_lr{args.lr:g}'
    if (out/'result.json').exists(): return
    out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(np.random.SeedSequence([args.seed,1])); records=[]; curve=[]
    start=time.monotonic()
    for step in range(args.steps):
        delay=int(rng.choice([2,4,6])); rec=sample(rng,64,delay=delay); records.append(rec)
        x,y=encode(rec); optim.zero_grad(); loss=nn.functional.cross_entropy(model(x),y); loss.backward()
        norm=nn.utils.clip_grad_norm_(model.parameters(),1.0); optim.step()
        if step%200==0 or step==args.steps-1: curve.append({'step':step+1,'loss':float(loss),'grad_norm':float(norm)})
    torch.save({'state':model.state_dict(),'kind':kind,'width':w},out/'model.pt')
    np.savez_compressed(out/'training.npz',records=np.stack(records))
    result={'model':args.model,'seed':args.seed,'lr':args.lr,'steps':args.steps,'nparams':nparams(model),'width':w,'curve':curve,'source_sha256':digest(__file__)}
    for name,split,ood,delay in [('val','val',False,4),('id','test',False,4),('ood','test',True,4),('delay8','test',False,8)]:
        rr=sample(np.random.default_rng(np.random.SeedSequence([args.seed,{'val':2,'id':3,'ood':4,'delay8':5}[name]])),args.eval_n,split,ood,delay)
        result[name]=evaluate(model,rr,out/f'{name}.npz')
    result['seconds']=time.monotonic()-start
    dump(out/'result.json',result)
    print(json.dumps({k:result[k] for k in ['model','seed','lr','seconds','id','ood']},ensure_ascii=False),flush=True)

def audit():
    specs=model_specs(); info={k:{'kind':v[0],'width':v[1],'params':nparams(Net(*v))} for k,v in specs.items()}
    rng=np.random.default_rng(270000); checks=[]
    for split in ['train','val','test']:
      for ood in [False,True]:
        rec=sample(rng,4096,split,ood); x,y=encode(rec)
        # Independent truth from input tensors, using observed update instruction.
        obs=x.numpy(); use=2*(obs[:,1,21]>0).astype(int)
        rows=np.arange(len(rec)); q=obs[:,-1,22:].argmax(1)
        c0=obs[rows,use,:4].argmax(1);c1=obs[rows,use,10:14].argmax(1)
        s0=obs[rows,use,4:8].argmax(1);s1=obs[rows,use,14:18].argmax(1)
        truth=np.where(q==0,c0==c1,s0==s1).astype(int)
        assert np.array_equal(truth,y.numpy())
        assert len(np.unique(partition(rec)))== (8 if split=='train' else 1)
        for u in (0,1):
         for qv in (0,1):
          ix=(rec[:,2]==u)&(rec[:,3]==qv);assert y[ix].float().mean()==.5
        p=BASE/'round1'/f'audit_{split}_{int(ood)}.npz';p.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(p,records=rec,truth=truth)
        checks.append({'split':split,'ood':ood,'oracle_accuracy':float((truth==y.numpy()).mean()),'n':len(rec)})
    # Split-state operation correctness: other query leaves content exactly invariant.
    rec=sample(rng,128,'test',True);x,y=encode(rec);xx=x.clone();xx[:,-1,22:]=xx[:,-1,22:].flip(1)
    model=Net('split',64);a=model.states(x);b=model.states(xx)
    assert torch.equal(a[:,:,:48],b[:,:,:48]);assert not torch.equal(a[:,-1,48:],b[:,-1,48:])
    result={'specs':info,'checks':checks,'split_content_invariance':True,'torch':torch.__version__,'python':platform.python_version(),'source_sha256':digest(__file__)}
    dump(BASE/'round1/audit.json',result); print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['audit','run']);p.add_argument('--round',default='round1');p.add_argument('--model',default='F64');p.add_argument('--seed',type=int,default=270101);p.add_argument('--lr',type=float,default=.001);p.add_argument('--steps',type=int,default=2000);p.add_argument('--eval-n',type=int,default=4096);args=p.parse_args()
    audit() if args.action=='audit' else run(args)
