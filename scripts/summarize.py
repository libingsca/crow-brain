import argparse,json
from pathlib import Path
import numpy as np
from crowbrain.analysis import interval
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('round');a=p.parse_args()
root=Path('artifacts')/a.round
rows=[]
for path in sorted(root.glob('*/*/result.json')):
 d=json.loads(path.read_text());m=json.loads((path.parent/'manifest.json').read_text())
 row={'stage':path.parent.parent.name,'kind':m['kind'],'seed':m['seed'],'lr':m['lr'],
      **{k:v for k,v in d.items() if not isinstance(v,dict)}}
 if 'shape' in d: row.update(d['shape'])
 for prefix in ['initial','final']:
  if prefix+'_analysis' in d:
   x=d[prefix+'_analysis'];row[prefix+'_delay_decode']=x['delay_decode'];row[prefix+'_encoder_decode']=x['encoder_decode']
   for phase in ['sample','delay']:
    for metric in ['selective_fraction','interior_peaked_fraction']:
     row[f'{prefix}_{phase}_{metric}']=x[phase+'_tuning'][metric]
 rows.append(row)
summary={}
for kind in sorted(set(r['kind'] for r in rows)):
 dev=[r for r in rows if r['kind']==kind and r['stage']=='dev']
 lr_summary={str(lr):float(np.mean([r['validation_ba'] for r in dev if r['lr']==lr])) for lr in sorted(set(r['lr'] for r in dev))}
 final=[r for r in rows if r['kind']==kind and r['stage']=='final']
 metrics=['cross_shape_ba','solid_solid','ring_ring','solid_ring','ring_solid','id_ba','shift_ba','reset_ba','long_delay_ba','initial_delay_decode','final_delay_decode',
          'initial_encoder_decode','final_encoder_decode','final_sample_interior_peaked_fraction','final_delay_interior_peaked_fraction']
 summary[kind]={'development_lr_means':lr_summary,'final':{k:interval([r[k] for r in final]) for k in metrics if final and k in final[0]}}
result={'runs':rows,'summary':summary}
(root/'summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(summary,indent=2))
fig,axes=plt.subplots(1,2,figsize=(10,4))
for path in sorted(root.glob('dev/*/trace.json')):
 t=json.loads(path.read_text());axes[0].plot([x['step'] for x in t],[x['validation_ba'] for x in t],label=path.parent.name,alpha=.7)
axes[0].set(xlabel='Training updates',ylabel='Validation balanced accuracy',ylim=(.4,1.01),title=a.round+' development')
axes[0].legend(fontsize=5)
for j,(kind,s) in enumerate(summary.items()):
 finals=[r for r in rows if r['kind']==kind and r['stage']=='final']
 for i,k in enumerate(['id_ba','shift_ba','reset_ba','final_delay_decode']):
  if finals:axes[1].scatter(np.full(len(finals),i+j*.15),[r[k] for r in finals],label=kind if i==0 else None,alpha=.7)
axes[1].set(xticks=range(4),xticklabels=['ID','cue shift','reset','delay decode'],ylim=(0,1.02),title='Independent final seeds')
if any(s['final'] for s in summary.values()): axes[1].legend()
fig.tight_layout();fig.savefig(root/'summary.png',dpi=170)
