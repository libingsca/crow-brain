"""Exploratory diagnosis, not a fair extra-supervision benchmark."""
import sys,json
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
import h01
root=h01.BASE/sys.argv[1];summary=json.loads((root/'summary_audited.json').read_text());outs=[]
for folder in sorted(root.glob('S64_*')):
 if not (folder/'result.json').exists():continue
 r=json.loads((folder/'result.json').read_text())
 if r['lr']!=summary['selected_lrs']['S64']:continue
 ck=torch.load(folder/'model.pt',weights_only=True);m=h01.Net(ck['kind'],ck['width']);m.load_state_dict(ck['state']);m.eval()
 reps={}
 for cond in ['val','id','ood']:
  rec=np.load(folder/f'{cond}.npz')['records'];hs=[];pred=[];counter=[]
  with torch.no_grad():
   for i in range(0,len(rec),512):
    x,_=h01.encode(rec[i:i+512]);hh=m.states(x)[:,-1];hs.append(hh.numpy());pred.append(m.head(hh).argmax(1).numpy())
    xx=x.clone();xx[:,-1,22:]=xx[:,-1,22:].flip(1);counter.append(m(xx).argmax(1).numpy())
  rel=h01.REL[np.where(rec[:,2],rec[:,1],rec[:,0])]
  reps[cond]=(np.concatenate(hs),rel,np.concatenate(pred),np.concatenate(counter),rec)
 z={'seed':r['seed'],'lr':r['lr'],'relation_decoding':{}}
 for q in (0,1):
  clf=LogisticRegression(C=1,max_iter=1000).fit(reps['val'][0][:,:48],reps['val'][1][:,q])
  z['relation_decoding'][str(q)]={k:float(clf.score(reps[k][0][:,:48],reps[k][1][:,q])) for k in ['id','ood']}
 for cond in ['id','ood']:
  h,rel,p,cp,rec=reps[cond];discord=rel[:,0]!=rel[:,1]
  truth_cf=rel[np.arange(len(rec)),1-rec[:,3]]
  z[cond]={'discordant_rule_switch_fraction':float((p[discord]!=cp[discord]).mean()),'counterfactual_accuracy':float((cp==truth_cf).mean()),'rule_state_gap':float(np.linalg.norm(h[rec[:,3]==0,48:].mean(0)-h[rec[:,3]==1,48:].mean(0)))}
 outs.append(z)
h01.dump(root/'exploratory_diagnosis.json',outs);print(json.dumps(outs,indent=2))
