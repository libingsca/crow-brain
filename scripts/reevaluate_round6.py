"""Recover evaluation after renderer assertion, with unchanged saved weights."""
import json,hashlib,time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from crowbrain.model import VisualRNN
from crowbrain.task import Bank,balanced_pairs
from crowbrain.run import evaluate,save_json
from crowbrain.analysis import decode,tuning

torch.set_num_threads(1)
root=Path('artifacts/round6');bank_path=root/'valid_shape_bank.npz';data=np.load(bank_path)
solid=torch.from_numpy(data['solid']);ring=torch.from_numpy(data['ring'])
rng=np.random.default_rng(230002);pairs=balanced_pairs(100);a,b=pairs.T;y=a==b
ia,ib=rng.integers(0,400,(2,len(a)));same=(a==b)&(ia==ib);ib[same]=(ib[same]+1)%400
idbank=Bank(220000,300);shiftbank=Bank(220001,300,'shift')

def analysis_from_saved(path):
 z=np.load(path);hs,ht,hv=z['selection'],z['probe_train'],z['probe_test'];ys,yt,yv=z['selection_labels'],z['probe_train_labels'],z['probe_test_labels']
 return {'delay_decode':decode(ht[:,8],yt,hv[:,8],yv),
         'encoder_decode':decode(z['encoder_train'],yt,z['encoder_test'],yv),
         'sample_tuning':tuning(hs[:,3],ys,hv[:,3],yv),'delay_tuning':tuning(hs[:,8],ys,hv[:,8],yv)}

for seed in range(301,306):
 for kind in ['rnn','ei']:
  out=root/'final'/f'{kind}_lr0.001_seed{seed}';checkpoint=out/'final.pt'
  assert (out/'failure.json').exists() and checkpoint.exists()
  manifest=json.loads((out/'manifest.json').read_text());trace=json.loads((out/'trace.json').read_text())
  assert trace[-1]['step']==2000
  m=VisualRNN(kind=kind,normalize=True,matched_init=True);m.load_state_dict(torch.load(checkpoint,weights_only=True));m.freeze_encoder=True;m.eval()
  r={'validation_ba':trace[-1]['validation_ba'],'parameters':sum(p.numel() for p in m.parameters()),
     'trainable_parameters':5314,'train_seconds':trace[-1]['elapsed_seconds'],
     'timing_note':'training time from last update log; final evaluation recovered separately',
     'id_ba':evaluate(m,idbank,220002),'shift_ba':evaluate(m,shiftbank,220003),
     'reset_ba':evaluate(m,shiftbank,220003,reset=True),'long_delay_ba':evaluate(m,shiftbank,220003,delay=15),
     'final_analysis':analysis_from_saved(out/'final_representations.npz'),
     'initial_analysis':analysis_from_saved(out/'initial_representations.npz')}
  shape={};predictions={'labels':y,'pairs':pairs,'sample_indices':ia,'test_indices':ib}
  with torch.no_grad():
   for name,x,z in [('solid_solid',solid,solid),('ring_ring',ring,ring),('solid_ring',solid,ring),('ring_solid',ring,solid)]:
    aa,bb=x[a,ia],z[b,ib];logits=[]
    for i in range(0,len(a),128):logits.append(m(aa[i:i+128],bb[i:i+128]).numpy())
    logits=np.concatenate(logits);shape[name]=float(balanced_accuracy_score(y,logits.argmax(1)));predictions[name]=logits
  shape['cross_shape_ba']=(shape['solid_ring']+shape['ring_solid'])/2;r['shape']=shape
  np.savez_compressed(out/'shape_predictions.npz',**predictions)
  save_json(out/'evaluation_repair.json',{'decision':'docs/decisions/008_round6_render_repair.md','checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    'shape_bank_sha256':hashlib.sha256(bank_path.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'weights_changed':False,'original_failure_retained':True,'evaluated_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
  save_json(out/'result.json',r)
  print(json.dumps({'kind':kind,'seed':seed,'id_ba':r['id_ba'],'shift_ba':r['shift_ba'],'long_delay_ba':r['long_delay_ba'],'shape':shape}),flush=True)
