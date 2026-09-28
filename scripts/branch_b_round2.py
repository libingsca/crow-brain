"""B2: new-image confirmation, fixed supervised heads, repeated null calibration."""
import hashlib,json,pickle,warnings,time,argparse
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import label
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from branch_b_round1 import Probe,states,continue_test,sha,digest,ba,save
from crowbrain.task import Bank,balanced_pairs
from crowbrain.model import VisualRNN
R=Path('artifacts/branch_b/round2');B=Path('artifacts/branch_b/round1')

def prepare():
    bank=Bank(810013,1000);images=bank.images.numpy()
    assert np.array_equal([label(im[0]>.499,structure=np.ones((3,3)))[1] for im in images.reshape(-1,1,32,32)],np.repeat(np.arange(2,7),1000))
    hs={digest(im) for im in images.reshape(-1,1,32,32)};assert len(hs)==5000
    for name in ['fit','validation','test']:
        old=np.load(B/f'{name}_data.npz')['images']
        assert not hs.intersection(digest(im) for im in old.reshape(-1,1,32,32))
    rng=np.random.default_rng(810113);pairs=balanced_pairs(100);rng.shuffle(pairs);a,b=pairs.T
    ia,ib=rng.integers(0,1000,(2,len(a)));same=(a==b)&(ia==ib);ib[same]=(ib[same]+1)%1000
    np.savez_compressed(R/'test_data.npz',images=images,a=a,b=b,ia=ia,ib=ib,y=(a==b).astype(np.int64))
    save(R/'data_manifest.json',{'seed':810013,'pair_seed':810113,'file_sha256':sha(R/'test_data.npz'),'image_sha256':digest(images),'images':5000,'trials':4000,'counts_verified':True,'disjoint_from_all_B1_partitions':True})
    print('B2 image audit passed',flush=True)

def nullfit(x,y,c):
    history=[]
    for limit in [3000,15000]:
        clf=LogisticRegression(C=c,solver='lbfgs',tol=1e-7,max_iter=limit)
        with warnings.catch_warnings(record=True) as caught:warnings.simplefilter('always');clf.fit(x,y)
        ok=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
        history.append({'max_iter':limit,'iterations':clf.n_iter_.tolist(),'converged':ok})
        if ok:return clf,history
    raise RuntimeError('B2 control fitting did not converge')

def run(seed):
    torch.set_num_threads(1);start=time.monotonic();out=R/f'seed{seed}';out.mkdir(exist_ok=False)
    source=Path(f'artifacts/round6/final/rnn_lr0.001_seed{seed}/final.pt')
    model=VisualRNN(normalize=True);model.load_state_dict(torch.load(source,weights_only=True));model.eval();model.requires_grad_(False)
    before={k:v.clone() for k,v in model.state_dict().items()}
    oldprobe=B/f'seed{seed}'/'probes.pkl'
    checkpoint_before,head_before=sha(source),sha(oldprobe)
    with oldprobe.open('rb') as f:probes=pickle.load(f)
    head=probes['head_15'];data=dict(np.load(R/'test_data.npz'));ys=data['y']
    h5,log5,_=states(model,data,5);h15,log15,et=states(model,data,15);x=h15[:,-1]
    pred=head.predict(x);original=log15.argmax(1)
    row={'seed':seed,'original5_ba':ba(ys,log5.argmax(1)),'original15_ba':ba(ys,original),'rescue15_ba':ba(ys,pred),'null_train':[],'test_label_shuffle':[],'state_shuffle':[]}
    with torch.no_grad():
        reset=continue_test(model,torch.zeros((4000,64)),et)
        row['reset_original_ba']=ba(ys,model.readout(reset).argmax(1).numpy())
        row['reset_rescue_ba']=ba(ys,head.predict(reset.numpy()))
    tr=np.load(B/f'seed{seed}'/'states_delay15.npz')['fit'][:,-1]
    ty=np.load(B/'fit_data.npz')['y']
    scaler=StandardScaler().fit(tr);xx=scaler.transform(tr).astype(np.float64);vv=scaler.transform(x).astype(np.float64)
    fit_records=[];coef=[];inter=[];allpred=[]
    for i in range(256):
        py=np.random.default_rng(811000+i).permutation(ty)
        clf,history=nullfit(xx,py,.01);pp=clf.predict(vv)
        row['null_train'].append(ba(ys,pp));fit_records.append({'permutation':i,'history':history});coef.append(clf.coef_);inter.append(clf.intercept_);allpred.append(pp)
        if (i+1)%64==0:print(json.dumps({'seed':seed,'nulls_completed':i+1,'seconds':time.monotonic()-start}),flush=True)
    real,history=nullfit(xx,ty,.01);row['fixed_C_real_label_ba']=ba(ys,real.predict(vv));fit_records.append({'real_label_reference':history})
    c=json.loads((B/f'seed{seed}'/'fits.json').read_text())['permuted_head']['C'];yp=np.random.default_rng(810004).permutation(ty)
    complement=[]
    for y in [yp,1-yp]:
        clf,history=nullfit(xx,y,c);complement.append(ba(ys,clf.predict(vv)));fit_records.append({'original_permutation_pair':history})
    row['original_permutation_fixed_C_pair']={'C':c,'scores':complement,'sum':sum(complement)}
    for i in range(64):
        row['test_label_shuffle'].append(ba(np.random.default_rng(812000+i).permutation(ys),pred))
        rng=np.random.default_rng(813000+i);indices=np.arange(len(ys))
        for b in range(5):
            ids=np.flatnonzero(data['b']==b);indices[ids]=rng.permutation(ids)
        row['state_shuffle'].append(ba(ys,head.predict(x[indices])))
    row['seconds']=time.monotonic()-start
    row['integrity']={'state_dict_unchanged':all(torch.equal(before[k],v) for k,v in model.state_dict().items()),'checkpoint_sha256':sha(source),'head_file_sha256':sha(oldprobe),'script_sha256':sha(__file__),'protocol_sha256':sha(R/'protocol.md')}
    assert row['integrity']['state_dict_unchanged']
    assert sha(source)==checkpoint_before and sha(oldprobe)==head_before
    row['integrity']['checkpoint_and_head_files_unchanged']=True
    np.savez_compressed(out/'raw.npz',states5=h5,states15=h15,original=original,rescue=pred,labels=ys,null_predictions=np.array(allpred),null_coef=np.array(coef),null_intercept=np.array(inter),scaler_mean=scaler.mean_,scaler_scale=scaler.scale_)
    save(out/'fits.json',fit_records);save(out/'result.json',row)
    print(json.dumps({'completed':seed,'original':row['original15_ba'],'rescue':row['rescue15_ba'],'null_mean':float(np.mean(row['null_train'])),'seconds':row['seconds']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--seed',type=int);a=p.parse_args()
    if a.prepare:prepare()
    else:
        try:run(a.seed)
        except Exception as e:save(R/f'seed{a.seed}'/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
