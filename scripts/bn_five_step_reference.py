"""Complete the protocol's five-step reference; original heads only, no fitting."""
import json
from pathlib import Path
import numpy as np
import torch
from crowbrain.model import VisualRNN
from branch_bn_round1 import noisy,score,sha,save
R=Path('artifacts/branch_bn/round1')
def main():
    torch.set_num_threads(1);rows=[];predictions={}
    for s in range(301,306):
        model=VisualRNN(normalize=True);cp=Path(f'artifacts/round6/final/rnn_lr0.001_seed{s}/final.pt');model.load_state_dict(torch.load(cp,weights_only=True));model.eval();model.requires_grad_(False)
        sigma=json.loads((R/f'seed{s}'/'manifest.json').read_text())['sigma_primary'];bs=[]
        with torch.no_grad():
            for bank in range(20):
                d=dict(np.load(R/f'test{bank:02}.npz'));images=torch.from_numpy(d['images']);k=images.shape[1]
                e=torch.cat([model.encode(im) for im in images.flatten(0,1).split(128)]).reshape(5,k,16)
                es=e[d['a'],d['ia']];et=e[d['b'],d['ib']]
                out,h,_=model(es,et,encoded=True,delay=5,return_states=True);clean=out.argmax(1).numpy();pre=h[:,8].numpy();ps=[]
                for rep in range(10):
                    _,end=noisy(model,pre,et,int(d['seed']),rep,sigma);ps.append(model.readout(torch.from_numpy(end)).argmax(1).numpy())
                bs.append({'bank':bank,'clean_ba':score(d['y'],clean),'noisy_ba':[score(d['y'],p) for p in ps]});predictions[f'{s}_{bank}']=np.array([clean]+ps,dtype=np.int8)
        rows.append({'seed':s,'checkpoint_sha256':sha(cp),'banks':bs,'clean_mean':float(np.mean([b['clean_ba'] for b in bs])),'noisy_mean':float(np.mean([b['noisy_ba'] for b in bs]))})
    np.savez_compressed(R/'five_step_predictions.npz',**predictions);save(R/'five_step_reference.json',{'script_sha256':sha(__file__),'models':rows})
    print(json.dumps([{k:v for k,v in r.items() if k!='banks'} for r in rows]))
if __name__=='__main__':main()
