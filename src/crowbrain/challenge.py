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
 images=[]
 for k in COUNTS:
  accepted=[];rejected=0
  while len(accepted)<n:
   try: im=draw(int(k),rng,ring)
   except AssertionError:
    rejected+=1
    if rejected>100:raise RuntimeError('Excessive invalid shape rendering')
    continue
   accepted.append(im)
  images.extend(accepted)
 return torch.tensor(np.stack(images).reshape(5,n,1,32,32))


@torch.no_grad()
def shape_evaluate(model,base):
 was_training=model.training;model.eval()
 solid=bank(base,False);ring=bank(base+1,True)
 pairs=balanced_pairs(100);a,b=pairs.T;y=a==b
 rng=np.random.default_rng(base+2);ia,ib=rng.integers(0,400,(2,len(a)))
 same=(a==b)&(ia==ib);ib[same]=(ib[same]+1)%400
 results={}
 for name,x,z in [('solid_solid',solid,solid),('ring_ring',ring,ring),('solid_ring',solid,ring),('ring_solid',ring,solid)]:
  aa,bb=x[a,ia],z[b,ib];pred=[]
  for i in range(0,len(a),128):pred.extend(model(aa[i:i+128],bb[i:i+128]).argmax(1).tolist())
  results[name]=float(balanced_accuracy_score(y,pred))
 results['cross_shape_ba']=(results['solid_ring']+results['ring_solid'])/2
 model.train(was_training)
 return results
