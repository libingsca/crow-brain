"""Individual 110-cell primary family and preregistered factorial descriptions."""
import json
import numpy as np
from scipy.stats import t
from branch_bf_round1 import ROOT,DELAYS,REFERENCE_DELAYS,HEADS
from branch_b_round1 import save

def interval(v):
    v=np.asarray(v);assert v.shape==(20,)
    mean=float(v.mean());half=float(t.ppf(1-.05/110/2,19)*v.std(ddof=1)/np.sqrt(20))
    return dict(mean=mean,lower=mean-half,upper=mean+half,confidence=1-.05/110,n_banks=20)

def main():
    assert json.loads((ROOT/'training_audit.json').read_text())['passed'] and json.loads((ROOT/'independent_audit.json').read_text())['passed']
    scores=np.full((5,20,11,4,2),np.nan);neg=np.full((5,20,11,4,2,3),np.nan);refs=np.full((5,20,8,2),np.nan);rn=np.full((5,20,8,2,3),np.nan);orig=np.full((5,20,2),np.nan)
    for si,seed in enumerate(range(301,306)):
        r=json.loads((ROOT/f'seed{seed}/results.json').read_text());assert len(r['cells'])==220
        for cell in r['cells']:
            bi=cell['bank'];di=DELAYS.index(cell['delay']);s=np.array(cell['scores']);negative=np.array(cell['negative_means']);assert cell['head_order'][:4]==HEADS
            scores[si,bi,di,:,0]=s[:4,0];scores[si,bi,di,:,1]=s[:4,1:].mean(-1);neg[si,bi,di]=negative[:4]
            if cell['delay'] in REFERENCE_DELAYS:
                ri=REFERENCE_DELAYS.index(cell['delay']);refs[si,bi,ri]=[s[4,0],s[4,1:].mean()];rn[si,bi,ri]=negative[4]
            if cell['delay']==5:
                o=np.array(cell['original_scores']);orig[si,bi]=[o[0],o[1:].mean()]
    assert all(np.isfinite(x).all() for x in [scores,neg,refs,rn,orig])
    primary={f'{seed}_d{d}_{c}':interval(scores[si,:,di,3,ci]) for si,seed in enumerate(range(301,306)) for di,d in enumerate(DELAYS) for ci,c in enumerate(['clean','noise'])}
    positive={f'reference_d{d}_{c}':interval(refs[:,:,ri,ci].mean(0)) for ri,d in enumerate(REFERENCE_DELAYS) for ci,c in enumerate(['clean','noise'])};positive.update({f'original5_{c}':interval(orig[:,:,ci].mean(0)) for ci,c in enumerate(['clean','noise'])})
    ng=np.r_[neg.mean((0,1)).ravel(),rn.mean((0,1)).ravel()];quality=bool(((ng>=.47)&(ng<=.53)).all()) and all(v['lower']>.90 for v in positive.values())
    support=[k for k,v in primary.items() if v['lower']>.85];exclude=[k for k,v in primary.items() if v['upper']<.85];unresolved=[k for k in primary if k not in support+exclude]
    decision='quality_failed' if not quality else ('supported' if len(support)==110 else ('excluded' if exclude else 'inconclusive'))
    targets={}
    for seed,d in [(301,15),(302,15),(304,9),(305,14)]:
        a=scores[seed-301,:,DELAYS.index(d),:,0].mean(0)
        targets[f'{seed}_d{d}_clean']={'means':dict(zip(HEADS,a.tolist())),'coverage_mix':float(a[2]-a[0]),'coverage_split':float(a[3]-a[1]),'split_base':float(a[1]-a[0]),'split_cover':float(a[3]-a[2]),'interaction':float(a[3]-a[1]-a[2]+a[0])}
    summary=dict(decision=decision,quality_passed=quality,primary=primary,supported=support,excluded=exclude,unresolved=unresolved,positive_controls=positive,negative_range=[float(ng.min()),float(ng.max())],negative_means=neg.mean((0,1)).tolist(),reference_negative_means=rn.mean((0,1)).tolist(),targets=targets,per_model_means={str(s):scores.mean(1)[i].tolist() for i,s in enumerate(range(301,306))},per_model_minimum={str(s):float(scores.mean(1)[i,:,3].min()) for i,s in enumerate(range(301,306))},panel_means=scores.mean((0,1)).tolist())
    save(ROOT/'summary.json',summary);np.savez_compressed(ROOT/'analysis_arrays.npz',scores=scores,negative=neg,references=refs,reference_negative=rn,original=orig)
    print(json.dumps({**{k:summary[k] for k in ['decision','quality_passed','negative_range','targets','per_model_minimum']},'supported':len(support),'excluded':len(exclude),'unresolved':len(unresolved)},indent=2))
if __name__=='__main__':main()
