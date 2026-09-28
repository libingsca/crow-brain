"""B-S fixed-family analysis: 22 primary cells, complete quality gates."""
import json
import numpy as np
from scipy.stats import t
from branch_bs_round1 import ROOT,DELAYS,TRAIN_DELAYS,HELDOUT
from branch_b_round1 import save

def interval(values):
    v=np.asarray(values);assert v.shape==(20,)
    mean=float(v.mean());half=float(t.ppf(1-.05/22/2,19)*v.std(ddof=1)/np.sqrt(20))
    return {'mean':mean,'lower':mean-half,'upper':mean+half,'confidence':1-.05/22,'n_banks':20}

def main():
    assert json.loads((ROOT/'training_audit.json').read_text())['passed']
    assert json.loads((ROOT/'independent_audit.json').read_text())['passed']
    scores=np.full((5,20,11,2,2),np.nan);negative=np.full((5,20,11,2,2,3),np.nan)
    refs=np.full((5,20,5,2),np.nan);refnegative=np.full((5,20,5,2,3),np.nan);original=np.full((5,20,2),np.nan)
    for si,seed in enumerate(range(301,306)):
        r=json.loads((ROOT/f'seed{seed}/results.json').read_text());assert len(r['cells'])==220
        for cell in r['cells']:
            bi=cell['bank'];d=cell['delay'];di=DELAYS.index(d);s=np.asarray(cell['scores']);neg=np.asarray(cell['negative_means'])
            assert cell['head_order'][:2]==['clean','aug']
            scores[si,bi,di,:,0]=s[:2,0];scores[si,bi,di,:,1]=s[:2,1:].mean(-1);negative[si,bi,di]=neg[:2]
            if d in TRAIN_DELAYS:
                ti=TRAIN_DELAYS.index(d);refs[si,bi,ti]=[s[2,0],s[2,1:].mean()];refnegative[si,bi,ti]=neg[2]
            if d==5:
                o=np.asarray(cell['original_scores']);original[si,bi]=[o[0],o[1:].mean()]
    assert all(np.isfinite(x).all() for x in [scores,negative,refs,refnegative,original])
    primary={f'd{d}_{c}':{**interval(scores[:,:,di,1,ci].mean(0)),'role':'heldout' if d in HELDOUT else 'trained'} for di,d in enumerate(DELAYS) for ci,c in enumerate(['clean','noise'])}
    positive={f'reference_d{d}_{c}':interval(refs[:,:,ti,ci].mean(0)) for ti,d in enumerate(TRAIN_DELAYS) for ci,c in enumerate(['clean','noise'])}
    positive.update({f'original5_{c}':interval(original[:,:,ci].mean(0)) for ci,c in enumerate(['clean','noise'])})
    neg=negative.mean((0,1));refneg=refnegative.mean((0,1));ng=np.r_[neg.ravel(),refneg.ravel()];negpass=bool(((ng>=.47)&(ng<=.53)).all());quality=negpass and all(v['lower']>.9 for v in positive.values())
    supported=[k for k,v in primary.items() if v['lower']>.85];excluded=[k for k,v in primary.items() if v['upper']<.85]
    decision='quality_failed' if not quality else ('supported' if len(supported)==22 else ('excluded' if excluded else 'inconclusive'))
    result={'decision':decision,'quality_passed':quality,'primary':primary,'primary_supported_cells':supported,'primary_excluded_cells':excluded,'primary_unresolved_cells':[k for k in primary if k not in supported+excluded],'positive_controls':positive,'negative_controls_passed':negpass,'negative_range':[float(ng.min()),float(ng.max())],'negative_means_delay_head_condition_control':neg.tolist(),'reference_negative_means':refneg.tolist(),'head_order':['clean','aug'],'per_model_means_delay_head_condition':{str(seed):scores.mean(1)[i].tolist() for i,seed in enumerate(range(301,306))},'per_model_aug_minimum':{str(seed):float(scores.mean(1)[i,:,1].min()) for i,seed in enumerate(range(301,306))},'paired_aug_minus_clean':(scores[:,:,:,1]-scores[:,:,:,0]).mean((0,1)).tolist(),'paired_aug_minus_reference':(scores[:,:,[DELAYS.index(d) for d in TRAIN_DELAYS],1]-refs).mean((0,1)).tolist(),'panel_all_cells':{f'{h}_d{d}_{c}':interval(scores[:,:,di,hi,ci].mean(0)) for hi,h in enumerate(['clean','aug']) for di,d in enumerate(DELAYS) for ci,c in enumerate(['clean','noise'])}}
    np.savez_compressed(ROOT/'analysis_arrays.npz',scores=scores,negative=negative,references=refs,reference_negative=refnegative,original=original);save(ROOT/'summary.json',result)
    print(json.dumps({k:result[k] for k in ['decision','quality_passed','primary','primary_excluded_cells','primary_unresolved_cells','negative_range','per_model_aug_minimum']},indent=2))

if __name__=='__main__':main()
