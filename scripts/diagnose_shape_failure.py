from pathlib import Path
import numpy as np
source=Path('src/crowbrain/challenge.py').read_text().split('@torch.no_grad()')[0]
source=source.replace("assert label(im>.499,structure=np.ones((3,3)))[1]==n", """actual=label(im>.499,structure=np.ones((3,3)))[1]
 if actual!=n:
  np.save('artifacts/round6/invalid_shape.npy',im)
  print({'expected':n,'actual':actual,'target_mass':target,'width':width,'initial_radius':rad,'final_radius':(left+right)/2},flush=True)
  raise RuntimeError('invalid ring example saved')""")
exec(source)
bank(230001,True)
