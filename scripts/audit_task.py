import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from crowbrain.task import Bank, balanced_pairs

banks=[Bank(610+i,300,p) for i,p in enumerate(['mixed','mixed','shift'])]
sets=[b.batch(np.random.default_rng(700+i),pairs=balanced_pairs(100)) for i,b in enumerate(banks)]
result={}
for name,extract in [('test_pixels',lambda d:d[1].flatten(1).numpy()),('test_statistics',lambda d:d[5][:,4:]),('pair_statistics',lambda d:d[5])]:
    clf=(make_pipeline(StandardScaler(),RidgeClassifier(alpha=10)) if name=='test_pixels'
         else HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,random_state=1))
    clf.fit(extract(sets[0]),sets[0][2].numpy())
    result[name]={p:float(balanced_accuracy_score(s[2],clf.predict(extract(s)))) for p,s in zip(['id','shift'],sets[1:])}
for p,b in zip(['train','id','shift'],banks):
    count=np.repeat(np.arange(2,7),b.per_count)
    result[p+'_count_stats_correlation']=[float(np.corrcoef(count,b.stats[:,:,i].ravel())[0,1]) if b.stats[:,:,i].std()>0 else None for i in range(4)]
Path('artifacts/round2_task_audit.json').write_text(json.dumps(result,indent=2))
fig,axes=plt.subplots(3,5,figsize=(9,5))
for row,b in enumerate(banks):
 for n in range(5):
  axes[row,n].imshow(b.images[n,0,0],cmap='gray_r',vmin=0,vmax=1);axes[row,n].axis('off');axes[row,n].set_title(f'{b.protocol}, n={n+2}')
fig.tight_layout();fig.savefig('artifacts/round2_stimuli.png',dpi=160)
print(json.dumps(result,indent=2))
