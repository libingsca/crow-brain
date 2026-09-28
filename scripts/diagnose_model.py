import json
from pathlib import Path
import numpy as np
import torch
from crowbrain.model import VisualRNN
from crowbrain.task import Bank
from crowbrain.run import representations
from crowbrain.analysis import decode

torch.set_num_threads(1)
a,b=Bank(96000,100),Bank(96001,100)
rows=[]
for p in sorted(Path('artifacts/round2/dev').glob('*/final.pt')):
 m=VisualRNN();m.load_state_dict(torch.load(p,weights_only=True))
 ha,ea,ya=representations(m,a,1);hb,eb,yb=representations(m,b,2)
 rows.append({'run':p.parent.name,'encoder_decode':decode(ea,ya,eb,yb),
              'sample_decode':decode(ha[:,3],ya,hb[:,3],yb),
              'delay_decode':decode(ha[:,8],ya,hb[:,8],yb),
              'encoder_std':float(ea.std(0).mean()),'delay_std':float(ha[:,8].std(0).mean()),
              'active_fraction':float((ha[:,8]>1e-5).mean())})
print(json.dumps(rows,indent=2));Path('artifacts/round2/diagnostics.json').write_text(json.dumps(rows,indent=2))
