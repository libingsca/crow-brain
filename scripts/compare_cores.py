import json
from pathlib import Path
import numpy as np
import torch
from crowbrain.analysis import interval
from crowbrain.run import save_json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('artifacts/round6');seeds=list(range(301,306));rows=[];checks=[]
for seed in seeds:
 runs={}
 initials={}
 for kind in ['rnn','ei']:
  p=root/'final'/f'{kind}_lr0.001_seed{seed}'
  d=json.loads((p/'result.json').read_text())
  init=torch.load(p/'initial.pt',weights_only=True);final=torch.load(p/'final.pt',weights_only=True)
  initials[kind]=init
  w=final['recurrent'] if kind=='rnn' else final['recurrent'].abs()*final['signs']
  frozen=all(torch.equal(v,final[k]) for k,v in init.items() if k.startswith(('visual.','project.','encode_norm.')))
  assert frozen
  violations=float((w*final['signs']<0).float().mean())
  if kind=='ei':assert violations==0
  checks.append({'seed':seed,'kind':kind,'encoder_unchanged':frozen,'dale_violating_fraction':violations})
  runs[kind]={'id_ba':d['id_ba'],'shift_ba':d['shift_ba'],'reset_ba':d['reset_ba'],
              'long_delay_ba':d['long_delay_ba'],'delay_decode':d['final_analysis']['delay_decode'],
              'sample_peaked':d['final_analysis']['sample_tuning']['interior_peaked_fraction'],
              'delay_peaked':d['final_analysis']['delay_tuning']['interior_peaked_fraction'],**d['shape']}
 assert torch.equal(initials['rnn']['recurrent'],initials['ei']['recurrent'].abs()*initials['ei']['signs'])
 for k in initials['rnn']:
  if k!='recurrent':assert torch.equal(initials['rnn'][k],initials['ei'][k]), k
 rows.append({'seed':seed,**runs})
summary={kind:{k:interval([r[kind][k] for r in rows]) for k in rows[0][kind]} for kind in ['rnn','ei']}
delta=interval([r['ei']['cross_shape_ba']-r['rnn']['cross_shape_ba'] for r in rows])
control_gate=summary['rnn']['id_ba']['low']>.75
supported=summary['ei']['cross_shape_ba']['low']>.65 and delta['low']>.05
excluded=summary['ei']['cross_shape_ba']['high']<.65 or delta['high']<.05
status=('control_gate_failed' if not control_gate else 'supported' if supported else 'useful_repair_excluded' if excluded else 'inconclusive')
result={'runs':rows,'summary':summary,'paired_cross_shape_delta':delta,'integrity_checks':checks,
        'control_gate_passed':control_gate,'primary_status':status,
        'paired_practical_equivalence':delta['low']>-.05 and delta['high']<.05,
        'scope':'This frozen encoder, 64-state core, 80/20 constraint, specified training and shape test; not biological NCL fit.'}
save_json(root/'comparison.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ['runs','integrity_checks']},indent=2))
fig,axes=plt.subplots(1,2,figsize=(11,4))
metrics=['id_ba','shift_ba','reset_ba','long_delay_ba','cross_shape_ba']
labels=['Circle ID','Area/extent shift','Memory reset','Long delay','Cross shape']
for j,kind in enumerate(['rnn','ei']):
 for i,k in enumerate(metrics):
  x=i+(j-.5)*.23;y=[r[kind][k] for r in rows]
  axes[0].scatter(np.full(5,x),y,alpha=.7,color={'rnn':'tab:blue','ei':'tab:orange'}[kind],label=kind if i==0 else None)
  axes[0].plot([x-.07,x+.07],[np.mean(y)]*2,color='black',lw=2)
axes[0].axhline(.5,color='gray',ls='--',lw=1)
axes[0].set(xticks=range(len(metrics)),xticklabels=labels,ylim=(0,1.04),ylabel='Balanced accuracy',title='Five fresh paired seeds; frozen shared encoder')
axes[0].tick_params(axis='x',labelrotation=20);axes[0].legend()
for r in rows:axes[1].plot([0,1],[r['rnn']['cross_shape_ba'],r['ei']['cross_shape_ba']],'o-',alpha=.7,label=str(r['seed']))
axes[1].axhline(.65,color='red',ls='--',label='Registered useful-transfer threshold')
axes[1].set(xticks=[0,1],xticklabels=['Unconstrained','E/I'],ylim=(.35,.8),ylabel='Cross-shape BA',title='Primary endpoint (paired)')
axes[1].legend(fontsize=6)
fig.tight_layout();fig.savefig(root/'comparison.png',dpi=180)
