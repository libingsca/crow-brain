from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d=loadmat('data/ditz2022/data_tuningcurve.mat')
fig,axes=plt.subplots(2,2,figsize=(9,6));summaries={}
for row,period in enumerate(['sample','delay']):
 for col,domain in enumerate(['small','large']):
  ax=axes[row,col]
  for group,color in [('BS','tab:blue'),('NS','tab:red')]:
   key=f'{group}_{period}_tuning_{domain}'
   x=d[key];center=x.shape[1]//2
   norm=x/x[:,center,None]
   mean=np.nanmean(norm,axis=0)
   distance=np.arange(x.shape[1])-center
   ax.plot(distance,mean,'o-',color=color,label=f'{group}, selected units n={len(x)}')
   summaries[key]={'n_units':len(x),'n_finite_each_distance':np.isfinite(norm).sum(0).tolist(),'normalized_mean':mean.tolist()}
  ax.set(title=f'{domain}: {period}',xlabel='Aligned distance (paper bins)',ylabel='Response / preferred response')
  ax.legend(fontsize=7)
fig.suptitle('Ditz et al. 2022 — processed-data replot; not independent animal replication',fontsize=10)
fig.tight_layout();fig.savefig('artifacts/biological_reference.png',dpi=160)
Path('artifacts/biological_reference.json').write_text(json.dumps(summaries,indent=2))
