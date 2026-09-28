"""Exploratory alternative explanation: image boundary statistics, not component count."""
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from crowbrain.task import Bank, balanced_pairs

def features(batch):
 xs=[]
 for image in batch[:2]:
  x=image.numpy()[:,0]
  perimeter=np.abs(np.diff(x,axis=1)).sum((1,2))+np.abs(np.diff(x,axis=2)).sum((1,2))
  mass=x.sum((1,2))
  xs.extend([mass,perimeter,perimeter**2/(mass+1e-6)])
 return np.stack(xs,1)
banks=[Bank(97000+i,250,p) for i,p in enumerate(['mixed','mixed','shift'])]
sets=[b.batch(np.random.default_rng(98000+i),pairs=balanced_pairs(100)) for i,b in enumerate(banks)]
clf=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,random_state=1)
clf.fit(features(sets[0]),sets[0][2])
r={name:float(balanced_accuracy_score(d[2],clf.predict(features(d)))) for name,d in zip(['id','shift'],sets[1:])}
Path('artifacts/extended_cue_audit.json').write_text(json.dumps(r,indent=2));print(r)
