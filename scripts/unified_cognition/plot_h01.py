from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h01
rounds=sys.argv[1:];fig,axes=plt.subplots(1,len(rounds),figsize=(5*len(rounds),4),squeeze=False)
for ax,rnd in zip(axes[0],rounds):
 s=json.loads((h01.BASE/rnd/'summary_audited.json').read_text());rows={r['model']:r for r in s['rows'] if r['selected']}
 names=['S64','FP','F64'];x=np.arange(3)
 for off,metric,color in [(-.16,'id','#3876a8'),(.16,'ood','#df8539')]:
  means=[rows[m][metric]['mean']*100 for m in names]
  ax.bar(x+off,means,.3,label='New trials (ID)' if metric=='id' else 'Unseen attribute combinations',color=color,alpha=.75)
  for j,m in enumerate(names):
   vals=np.asarray(rows[m][metric]['values'])*100
   ax.scatter(j+off+np.linspace(-.08,.08,len(vals)),vals,s=10,color='black',alpha=.6)
 ax.set_xticks(x,names);ax.set_ylim(40,103);ax.set_ylabel('Balanced accuracy (%)');ax.set_title(rnd+' (development)');ax.axhline(50,color='gray',ls=':');ax.legend(fontsize=7,loc='lower right')
fig.tight_layout();out=h01.ROOT/'reports/unified_cognition'/('H01_'+'_'.join(rounds)+'.png');fig.savefig(out,dpi=150);print(out)
