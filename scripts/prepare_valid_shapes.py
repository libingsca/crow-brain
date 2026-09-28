import numpy as np
from pathlib import Path
import json, hashlib
from scipy.ndimage import label
from crowbrain.challenge import draw
from crowbrain.task import COUNTS
out=Path('artifacts/round6');banks={};rejections={}
for domain,seed,ring in [('solid',230000,False),('ring',230001,True)]:
 rng=np.random.default_rng(seed);images=[];rejections[domain]={}
 for n in COUNTS:
  accepted=[];rejected=0
  while len(accepted)<400:
   try:im=draw(int(n),rng,ring)
   except AssertionError:
    rejected+=1
    if rejected>100:raise RuntimeError('excessive invalid shape generation')
    continue
   assert label(im>.499,structure=np.ones((3,3)))[1]==n
   accepted.append(im)
  images.append(accepted);rejections[domain][str(n)]=rejected
 banks[domain]=np.asarray(images,dtype=np.float32)[:,:,None]
p=out/'valid_shape_bank.npz';np.savez_compressed(p,**banks)
info={'seeds':{'solid':230000,'ring':230001},'accepted_images':4000,'rejections':rejections,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'filter':'pre-existing 8-connected component assertion; no model outputs consulted'}
(out/'valid_shape_bank.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))
