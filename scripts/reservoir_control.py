import json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
import joblib
from crowbrain.model import VisualRNN
from crowbrain.task import Bank,balanced_pairs
from crowbrain.run import save_json
from crowbrain.analysis import interval

torch.set_num_threads(1)
root=Path('artifacts/round5/reservoir');root.mkdir(parents=True,exist_ok=True)
@torch.no_grad()
def features(m,b,seed,repeats,reset=False):
 a,z,y,*_=b.batch(np.random.default_rng(seed),pairs=balanced_pairs(repeats))
 x=[]
 for start in range(0,len(a),128):
  _,h,_=m(a[start:start+128],z[start:start+128],reset=reset,return_states=True)
  x.append(h[:,-1].numpy())
 return np.concatenate(x),y.numpy()
@torch.no_grad()
def setup(seed):
 torch.manual_seed(seed);m=VisualRNN(normalize=True);m.eval()
 bank=Bank(10000+seed,1000)
 x=torch.cat([F.relu(m.project(m.visual(im).sum((2,3))/32)) for im in bank.images.flatten(0,1).split(128)])
 m.encode_norm.running_mean.copy_(x.mean(0));m.encode_norm.running_var.copy_(x.var(0,unbiased=True))
 return m,bank

validation=Bank(190000,300)
dev=[]
for seed in [1,2,3]:
 m,train=setup(seed);x,y=features(m,train,191000+seed,200);vx,vy=features(m,validation,192000,40)
 for alpha in [.1,1.,10.]:
  clf=make_pipeline(StandardScaler(),RidgeClassifier(alpha=alpha));clf.fit(x,y)
  dev.append({'seed':seed,'alpha':alpha,'validation_ba':float(balanced_accuracy_score(vy,clf.predict(vx)))})
 print({'development_seed':seed,'results':dev[-3:]},flush=True)
means={alpha:np.mean([r['validation_ba'] for r in dev if r['alpha']==alpha]) for alpha in [.1,1.,10.]}
alpha=max(means,key=lambda a:(means[a],a));save_json(root/'selection.json',{'development':dev,'means':means,'selected_alpha':alpha})
tests=[Bank(193000,400),Bank(193001,400,'shift')]
rows=[]
for seed in range(201,206):
 out=root/f'seed{seed}';out.mkdir(exist_ok=True)
 m,train=setup(seed);x,y=features(m,train,194000+seed,200)
 clf=make_pipeline(StandardScaler(),RidgeClassifier(alpha=alpha));clf.fit(x,y)
 r={'seed':seed,'alpha':alpha,'train_pairs':len(y)}
 for name,b,reset in [('id_ba',tests[0],False),('shift_ba',tests[1],False),('reset_ba',tests[1],True)]:
  tx,ty=features(m,b,195000,100,reset)
  r[name]=float(balanced_accuracy_score(ty,clf.predict(tx)))
  np.savez_compressed(out/(name+'_features.npz'),features=tx,labels=ty)
 torch.save(m.state_dict(),out/'fixed_network.pt');joblib.dump(clf,out/'readout.joblib')
 np.savez_compressed(out/'train_features.npz',features=x,labels=y)
 save_json(out/'result.json',r);rows.append(r);print(r,flush=True)
summary={k:interval([r[k] for r in rows]) for k in ['id_ba','shift_ba','reset_ba']}
save_json(root/'summary.json',summary);print(json.dumps(summary),flush=True)
