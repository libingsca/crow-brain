import json, hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from crowbrain.task import Bank
from crowbrain.model import VisualRNN
from crowbrain.run import evaluate,analyze,save_json
from crowbrain.analysis import interval

torch.set_num_threads(1)
root=Path('artifacts/round4');root.mkdir(exist_ok=True)
banks=[Bank(140000,400),Bank(140001,400,'shift')]
rows=[]
for seed in range(101,106):
 out=root/f'seed{seed}';out.mkdir(exist_ok=True)
 source=Path(f'artifacts/round3/final/rnn_lr0.001_seed{seed}/final.pt')
 m=VisualRNN(normalize=True);m.load_state_dict(torch.load(source,weights_only=True));m.eval()
 r={'seed':seed,'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
 r['before']={name:evaluate(m,b,140002,100) for name,b in zip(['id_ba','shift_ba'],banks)}
 trained={k:p.detach().clone() for k,p in m.named_parameters()}
 train=Bank(10000+seed,1000);features=[]
 with torch.no_grad():
  for image in train.images.flatten(0,1).split(128):
   features.append(F.relu(m.project(m.visual(image).sum((2,3))/32)))
  x=torch.cat(features)
  r['mean_change_l2']=float(torch.linalg.vector_norm(m.encode_norm.running_mean-x.mean(0)))
  m.encode_norm.running_mean.copy_(x.mean(0));m.encode_norm.running_var.copy_(x.var(0,unbiased=True))
 assert all(torch.equal(trained[k],p) for k,p in m.named_parameters())
 r['after']={name:evaluate(m,b,140002,100) for name,b in zip(['id_ba','shift_ba'],banks)}
 r['after']['reset_ba']=evaluate(m,banks[1],140002,100,reset=True)
 r['after']['long_delay_ba']=evaluate(m,banks[1],140002,100,delay=15)
 torch.save(m.state_dict(),out/'calibrated.pt')
 r['analysis']=analyze(m,140010,out,'calibrated')
 save_json(out/'result.json',r);rows.append(r)
 print(json.dumps({k:v for k,v in r.items() if k!='analysis'}),flush=True)
summary={key:interval([r['after'][key] for r in rows]) for key in rows[0]['after']}
summary['delay_decode']=interval([r['analysis']['delay_decode'] for r in rows])
for period in ['sample','delay']:
 summary[period+'_peaked_fraction']=interval([r['analysis'][period+'_tuning']['interior_peaked_fraction'] for r in rows])
save_json(root/'summary.json',summary);print(json.dumps(summary),flush=True)
