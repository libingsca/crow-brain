import json
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import label
from sklearn.metrics import balanced_accuracy_score
from crowbrain.task import Bank,balanced_pairs,COUNTS
from crowbrain.model import VisualRNN
from crowbrain.analysis import interval
from crowbrain.run import save_json

def draw(n,rng,ring):
 target=rng.uniform(45,75);width=int(rng.integers(21,27));lo=(32-width)//2;hi=lo+width-1
 hole=.45 if ring else 0
 rad=np.sqrt(target/(np.pi*n*(1-hole**2)))
 anchors=np.array([[0.,0.],[1.,1.]])
 if rng.random()<.5:anchors[:,1]=1-anchors[:,1]
 points=list(anchors)
 for _ in range(n-2):
  for attempt in range(1000):
   p=rng.uniform(.05,.95,2)
   if all(np.linalg.norm(p-q)*(width-1-2*rad)>2*rad+1.5 for q in points):points.append(p);break
  else:return draw(n,rng,ring)
 points=np.array(points);yy,xx=np.mgrid[:32,:32]
 def render(r):
  centers=lo+r+points*(hi-lo-2*r)
  dist=np.sqrt((yy[None]-centers[:,0,None,None])**2+(xx[None]-centers[:,1,None,None])**2)
  x=np.clip(r+.5-dist,0,1)
  if ring:x-=np.clip(hole*r+.5-dist,0,1)
  return np.clip(x,0,1).max(0)
 left,right=rad*.7,rad*1.3
 for _ in range(24):
  mid=(left+right)/2
  if render(mid).sum()<target:left=mid
  else:right=mid
 im=render((left+right)/2).astype(np.float32)
 assert label(im>.499,structure=np.ones((3,3)))[1]==n
 return im

def bank(seed,ring,n=400):
 rng=np.random.default_rng(seed)
 return torch.tensor(np.stack([draw(int(k),rng,ring) for k in COUNTS for _ in range(n)]).reshape(5,n,1,32,32))

if __name__=='__main__':
 torch.set_num_threads(1)
 out=Path('artifacts/round5');out.mkdir(exist_ok=True)
 solid=bank(180000,False);ring=bank(180001,True)
 rng=np.random.default_rng(180002);pairs=balanced_pairs(100);a,b=pairs.T;y=a==b
 ia,ib=rng.integers(0,400,(2,len(a)));same=(a==b)&(ia==ib);ib[same]=(ib[same]+1)%400
 rows=[]
 for seed in range(101,106):
  m=VisualRNN(normalize=True);m.load_state_dict(torch.load(f'artifacts/round4/seed{seed}/calibrated.pt',weights_only=True));m.eval()
  row={'seed':seed}
  with torch.no_grad():
   for name,x,z in [('solid_solid',solid,solid),('ring_ring',ring,ring),('solid_ring',solid,ring),('ring_solid',ring,solid)]:
    aa,bb=x[a,ia],z[b,ib];pred=[]
    for i in range(0,len(a),128):pred.extend(m(aa[i:i+128],bb[i:i+128]).argmax(1).tolist())
    row[name]=float(balanced_accuracy_score(y,pred))
  row['cross_shape_ba']=(row['solid_ring']+row['ring_solid'])/2
  rows.append(row);print(row,flush=True)
 summary={k:interval([r[k] for r in rows]) for k in rows[0] if k!='seed'}
 save_json(out/'shape_results.json',{'runs':rows,'summary':summary,'component_checks':4000,'inner_radius_fraction':.45})
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,5,figsize=(9,4))
 for row,data in enumerate([solid,ring]):
  for j,n in enumerate(COUNTS):axes[row,j].imshow(data[j,0,0],cmap='gray_r');axes[row,j].set_title(f'n={n}');axes[row,j].axis('off')
 fig.tight_layout();fig.savefig(out/'shape_stimuli.png',dpi=170)
 print(json.dumps(summary),flush=True)
