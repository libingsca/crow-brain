"""Single reproducible run. Hyperparameter selection happens outside this command."""
import argparse, hashlib, json, os, platform, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import balanced_accuracy_score
from .task import Bank, balanced_pairs, COUNTS
from .model import VisualRNN
from .analysis import decode, tuning

def save_json(path, x):
    Path(path).write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')

@torch.no_grad()
def evaluate(model, bank, seed, repeats=40, delay=5, reset=False):
    was_training=model.training
    model.eval()
    batch = bank.batch(np.random.default_rng(seed), pairs=balanced_pairs(repeats))
    a,b,y,counts,_,_ = batch
    pred=[]
    for start in range(0,len(y),128):
        logits=model(a[start:start+128], b[start:start+128],delay=delay,reset=reset)
        pred.extend(logits.argmax(1).numpy())
    model.train(was_training)
    return float(balanced_accuracy_score(y.numpy(),pred))

@torch.no_grad()
def representations(model, bank, seed, delay=5):
    # Direct bank traversal gives independent images with exact count balance.
    was_training=model.training
    model.eval()
    a=bank.images.flatten(0,1)
    ys=np.repeat(np.arange(5),bank.per_count)
    xs, enc=[],[]
    for i in range(0,len(a),128):
        _,h,e=model(a[i:i+128],torch.zeros_like(a[i:i+128]),delay=delay,return_states=True)
        xs.append(h.numpy()); enc.append(e.numpy())
    model.train(was_training)
    return np.concatenate(xs),np.concatenate(enc),ys

def analyze(model, base_seed, out, prefix, delay=5):
    # Three disjoint image sets: ID selection, shifted probe fit, shifted evaluation.
    banks=[Bank(base_seed+i,120,p) for i,p in enumerate(['mixed','shift','shift'])]
    arr=[representations(model,b,base_seed+i,delay) for i,b in enumerate(banks)]
    hs,es,ys=arr[0]; ht,et,yt=arr[1]; hv,ev,yv=arr[2]
    last=4+delay-1
    result={'delay_decode':decode(ht[:,last],yt,hv[:,last],yv),
            'encoder_decode':decode(et,yt,ev,yv),
            'sample_tuning':tuning(hs[:,3],ys,hv[:,3],yv),
            'delay_tuning':tuning(hs[:,last],ys,hv[:,last],yv)}
    np.savez_compressed(out/f'{prefix}_representations.npz',selection=hs,probe_train=ht,
                        probe_test=hv,selection_labels=ys,probe_train_labels=yt,probe_test_labels=yv,
                        encoder_selection=es,encoder_train=et,encoder_test=ev)
    return result

def run(args):
    config=json.loads(Path(args.config).read_text())
    if config['counts'] != COUNTS.tolist() or config['image_size'] != 32:
        raise ValueError('This renderer implements counts 2–6 at 32×32; use archived source for Round 1.')
    if (config['sample_steps'],config['delay_steps'],config['test_steps']) != (4,5,4):
        raise ValueError('Registered runner supports timing 4/5/4; update evaluation and analysis together for a new protocol.')
    torch.set_num_threads(config['threads'])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    manifest={**vars(args),'config_values':config,'started_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              'python':platform.python_version(),'torch':torch.__version__,
              'code_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('src').rglob('*.py')}}
    save_json(out/'manifest.json',manifest)
    start=time.monotonic()
    model=VisualRNN(config['hidden'],config['alpha'],args.kind,config.get('normalize',False),config.get('matched_init',False))
    train=Bank(10000+args.seed,config['train_bank_per_count'])
    frozen=bool(config.get('frozen_encoder'))
    if frozen:
        source=Path(config['frozen_encoder'])
        weights=torch.load(source,weights_only=True)
        model.load_state_dict({k:v for k,v in weights.items() if k.startswith(('visual.','project.','encode_norm.'))},strict=False)
        for name,param in model.named_parameters():
            if name.startswith(('visual.','project.','encode_norm.')):param.requires_grad_(False)
        model.freeze_encoder=True
        with torch.no_grad():
            raw=torch.cat([torch.relu(model.project(model.visual(im).sum((2,3))/32)) for im in train.images.flatten(0,1).split(128)])
            model.encode_norm.running_mean.copy_(raw.mean(0));model.encode_norm.running_var.copy_(raw.var(0,unbiased=True))
            model.encode_norm.eval()
            cached=model.encode_norm(raw)
            train.images=cached.reshape(5,train.per_count,16)
        manifest['frozen_encoder_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
        save_json(out/'manifest.json',manifest)
    torch.save(model.state_dict(),out/'initial.pt')
    val=Bank(20000,200)
    optimizer=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=args.lr)
    rng=np.random.default_rng(30000+args.seed)
    loss_fn=nn.CrossEntropyLoss()
    trace=[]
    for step in range(1,args.steps+1):
        a,b,y,*_=train.batch(rng,config['batch_size'])
        optimizer.zero_grad(set_to_none=True)
        logits=model(a,b,delay=config['delay_steps'],encoded=frozen)
        loss=loss_fn(logits,y)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite training loss')
        loss.backward()
        grad=float(nn.utils.clip_grad_norm_(model.parameters(),1.))
        optimizer.step()
        if step%250==0 or step==args.steps:
            score=evaluate(model,val,40000,10,config['delay_steps'])
            row={'step':step,'loss':float(loss.detach()),'validation_ba':score,'gradient_norm':grad,
                 'elapsed_seconds':time.monotonic()-start}
            trace.append(row)
            save_json(out/'trace.json',trace)
            print(json.dumps({'run':out.name,**row}),flush=True)
        if time.monotonic()-start>config['run_timeout_seconds']:
            raise TimeoutError('registered per-run budget exceeded')
    torch.save(model.state_dict(),out/'final.pt')
    result={'validation_ba':trace[-1]['validation_ba'],'parameters':sum(p.numel() for p in model.parameters()),
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'train_seconds':time.monotonic()-start}
    if args.final:
        test_seed=args.test_base
        idbank=Bank(test_seed,300)
        shiftbank=Bank(test_seed+1,300,'shift')
        result.update(id_ba=evaluate(model,idbank,test_seed+2),shift_ba=evaluate(model,shiftbank,test_seed+3),
                      reset_ba=evaluate(model,shiftbank,test_seed+3,reset=True),
                      long_delay_ba=evaluate(model,shiftbank,test_seed+3,delay=15))
        result['final_analysis']=analyze(model,test_seed+10,out,'final')
        initial=VisualRNN(config['hidden'],config['alpha'],args.kind,config.get('normalize',False),config.get('matched_init',False))
        initial.load_state_dict(torch.load(out/'initial.pt',weights_only=True))
        initial.freeze_encoder=frozen
        result['initial_analysis']=analyze(initial,test_seed+10,out,'initial')
        if config.get('shape_test_base'):
            from .challenge import shape_evaluate
            result['shape']=shape_evaluate(model,config['shape_test_base'])
    result['total_seconds']=time.monotonic()-start
    save_json(out/'result.json',result)
    print(json.dumps({'completed':str(out),'metrics':{k:v for k,v in result.items() if not isinstance(v,dict)}}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True); p.add_argument('--kind',choices=['rnn','ei','reservoir'],default='rnn')
    p.add_argument('--seed',type=int,required=True); p.add_argument('--lr',type=float,required=True)
    p.add_argument('--steps',type=int,required=True); p.add_argument('--output',required=True)
    p.add_argument('--final',action='store_true'); p.add_argument('--test-base',type=int,default=50000)
    a=p.parse_args()
    try: run(a)
    except Exception as e:
        out=Path(a.output)
        if out.exists(): save_json(out/'failure.json',{'type':type(e).__name__,'message':str(e)})
        raise
