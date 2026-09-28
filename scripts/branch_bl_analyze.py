"""Fixed six-endpoint B-L analysis, only after complete independent audits."""
import json
import numpy as np
from scipy.stats import t
from branch_bl_round1 import ROOT,DELAYS
from branch_b_round1 import save
HEADS=['shared_clean','shared_aug','clean','aug']
def interval(v):
    v=np.asarray(v,dtype=float);assert v.shape==(20,) and np.isfinite(v).all()
    mean=float(v.mean());half=float(t.ppf(1-.05/6/2,19)*v.std(ddof=1)/np.sqrt(20))
    return {'mean':mean,'lower':mean-half,'upper':mean+half,'confidence':1-.05/6,'n_banks':20}
def main():
    assert json.loads((ROOT/'training_audit.json').read_text())['passed']
    assert json.loads((ROOT/'independent_audit.json').read_text())['passed']
    assert json.loads((ROOT/'data_chronology_integrity.json').read_text())['passed']
    scores=np.full((5,20,3,4,2),np.nan);negative=np.full((5,20,3,4,2,3),np.nan)
    original=np.full((5,20,2),np.nan);anchor=np.full_like(original,np.nan)
    for si,seed in enumerate(range(301,306)):
        r=json.loads((ROOT/f'seed{seed}/results.json').read_text())
        assert len(r['cells'])==100 and len({(c['bank'],c['delay']) for c in r['cells']})==100
        for cell in r['cells']:
            bi=cell['bank'];s=np.asarray(cell['scores']);delay=cell['delay']
            if delay==15:anchor[si,bi]=[s[0,0],s[0,1:].mean()];continue
            if delay==5:
                o=np.asarray(cell['original_scores']);original[si,bi]=[o[0],o[1:].mean()];continue
            assert cell['head_order']==HEADS
            di=DELAYS.index(delay);scores[si,bi,di,:,0]=s[:,0];scores[si,bi,di,:,1]=s[:,1:].mean(-1);negative[si,bi,di]=cell['negative_means']
    assert all(np.isfinite(x).all() for x in [scores,negative,original,anchor])
    primary={f'd{d}_{c}':interval(scores[:,:,di,3,ci].mean(0)) for di,d in enumerate(DELAYS) for ci,c in enumerate(['clean','noise'])}
    positives={f'aug15_{c}':interval(anchor[:,:,ci].mean(0)) for ci,c in enumerate(['clean','noise'])}
    positives.update({f'original5_{c}':interval(original[:,:,ci].mean(0)) for ci,c in enumerate(['clean','noise'])})
    neg=negative.mean((0,1));negpass=bool(((neg>=.47)&(neg<=.53)).all());quality=negpass and all(v['lower']>.9 for v in positives.values())
    decision='quality_failed' if not quality else 'supported' if all(v['lower']>.85 for v in primary.values()) else 'excluded' if any(v['upper']<.85 for v in primary.values()) else 'inconclusive'
    means=scores.mean(1)
    r={'decision':decision,'quality_passed':quality,'primary':primary,'positive_controls':positives,'negative_controls_passed':negpass,'negative_range':[float(neg.min()),float(neg.max())],'negative_means_delay_head_condition_control':neg.tolist(),'head_order':HEADS,'negative_control_names':['reset','label_shuffle','within_test_count_state_shuffle'],'per_model_means_delay_head_condition':{str(seed):means[i].tolist() for i,seed in enumerate(range(301,306))},'per_model_aug_minimum':{str(seed):float(means[i,:,3].min()) for i,seed in enumerate(range(301,306))},'paired_aug_minus_shared_aug':(scores[:,:,:,3]-scores[:,:,:,1]).mean((0,1)).tolist(),'paired_aug_minus_clean':(scores[:,:,:,3]-scores[:,:,:,2]).mean((0,1)).tolist(),'panel_all_cells':{f'{h}_d{d}_{c}':interval(scores[:,:,di,hi,ci].mean(0)) for hi,h in enumerate(HEADS) for di,d in enumerate(DELAYS) for ci,c in enumerate(['clean','noise'])}}
    np.savez_compressed(ROOT/'analysis_arrays.npz',scores=scores,negative=negative,original=original,anchor=anchor);save(ROOT/'summary.json',r)
    print(json.dumps({k:r[k] for k in ['decision','quality_passed','primary','positive_controls','negative_range','per_model_aug_minimum']},indent=2))
if __name__=='__main__':main()
