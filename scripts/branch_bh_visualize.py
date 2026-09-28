from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
fig,axes=plt.subplots(2,2,figsize=(13,6.5),sharex=True,sharey=True)
for ri in [1,2]:
 root=Path(f'artifacts/branch_bh/round{ri}');r=json.loads((root/'summary.json').read_text());s=np.load(root/'analysis_arrays.npz')['scores'].mean(1);cfg=json.loads((root/'config.json').read_text());h=list(cfg['heads']).index(cfg['candidate'])
 for ci,c in enumerate(['clean','noise']):
  ax=axes[ri-1,ci];data=s[:,:,h,ci]*100;im=ax.imshow(data,cmap='RdYlGn',norm=TwoSlopeNorm(vmin=75,vcenter=85,vmax=100),aspect='auto')
  for i,seed in enumerate(range(301,306)):
   for j,d in enumerate(range(5,16)):
    k=f'{seed}_d{d}_{c}';marker='*' if k in r['unresolved'] else '!' if k in r['excluded'] else '';ax.text(j,i,f'{data[i,j]:.1f}{marker}',ha='center',va='center',fontsize=8)
  ax.set_title(f'Round {ri}: '+('development' if ri==1 else 'fresh fit confirmation')+f' / {c}');ax.set_xticks(range(11),range(5,16));ax.set_yticks(range(5),range(301,306));ax.set_xlabel('Delay');ax.set_ylabel('Fixed model')
fig.suptitle('B-H eleven-delay readout: mean BA% (* interval crosses 85%; ! upper bound <85%)');fig.tight_layout(rect=[0,0,.94,.95]);cax=fig.add_axes([.95,.15,.012,.65]);fig.colorbar(im,cax=cax,label='BA (%)');fig.savefig('reports/branch_bh/coverage_confirmation.png',dpi=180);plt.close(fig)
