"""Independent saved-record truth, partition and prediction audit, after runs finish."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
import h01
root=h01.BASE/sys.argv[1]
results=[]; failures=[]; blocks={}; unique={'train':set(),'val':set(),'id':set()}
for p in sorted(root.glob('*/result.json')):
 r=json.loads(p.read_text());folder=p.parent
 ck=torch.load(folder/'model.pt',weights_only=True);m=h01.Net(ck['kind'],ck['width']);m.load_state_dict(ck['state']);m.eval()
 tr=np.load(folder/'training.npz')['records'];trhash=hashlib.sha256(tr.tobytes()).hexdigest()
 key=r['seed'];blocks.setdefault(key,set()).add(trhash)
 assert np.all(h01.partition(tr.reshape(-1,7))<8)
 for cond in ['val','id','ood','delay8']:
  z=np.load(folder/f'{cond}.npz');a=z['records'];first=a[:,0];second=a[:,1]; chosen=np.where(a[:,2]==1,second,first)
  ob1=chosen//16;ob2=chosen%16
  truth=np.where(a[:,3]==0,ob1//4==ob2//4,ob1%4==ob2%4).astype(int)
  assert np.array_equal(truth,z['labels'])
  actual=float((z['logits'].argmax(1)==truth).mean());assert actual==r[cond]['accuracy']
  assert np.all(h01.partition(a)==(8 if cond=='val' else 9))
  withheld=(ob1//4==ob1%4)|(ob2//4==ob2%4)
  assert np.all(withheld==(cond=='ood'))
  for u in (0,1):
   for q in (0,1):
    ix=(a[:,2]==u)&(a[:,3]==q);assert truth[ix].mean()==.5
  # Reload every saved model and reconstruct logits for all trials.
  lp=[]
  with torch.no_grad():
   for i in range(0,len(a),512):lp.append(m(h01.encode(a[i:i+512])[0]).numpy())
  assert np.array_equal(np.concatenate(lp),z['logits']),str(folder)
 results.append(r)
assert all(len(v)==1 for v in blocks.values()),blocks
rows=[]; selected={}
for model in sorted({r['model'] for r in results}):
 rs=[r for r in results if r['model']==model]; lrs=sorted({r['lr'] for r in rs})
 best=min(lrs,key=lambda lr:(np.mean([r['val']['loss'] for r in rs if r['lr']==lr]),lr));selected[model]=best
 for lr in lrs:
  ss=[r for r in rs if r['lr']==lr]
  row={'model':model,'lr':lr,'n':len(ss),'selected':lr==best,'params':ss[0]['nparams']}
  for cond in ['val','id','ood','delay8']:
   vals=[r[cond]['accuracy'] for r in ss];row[cond]={'mean':float(np.mean(vals)),'values':vals,'sd':float(np.std(vals,ddof=1)) if len(vals)>1 else 0.}
  rows.append(row)
out={'n_runs':len(results),'all_input_truth_partitions_predictions_match':True,'identical_training_records_within_seed':True,'selected_lrs':selected,'rows':rows}
h01.dump(root/'summary_audited.json',out)
for row in rows:print(row['model'],row['lr'],'SELECTED' if row['selected'] else '',*[f'{c}={row[c]["mean"]:.4f}' for c in ['val','id','ood','delay8']])
