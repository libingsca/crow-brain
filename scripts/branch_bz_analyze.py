"""Five independent refit batches; never treat nested test banks as refits."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import t
ROOT=Path('artifacts/branch_bz');DELAYS=list(range(5,16));HEADS=['mean_aug','worst_aug']
def interval(v,confidence=1-.05/27):
 v=np.asarray(v);assert v.shape==(5,)
 m=float(v.mean());h=float(t.ppf((1+confidence)/2,4)*v.std(ddof=1)/np.sqrt(5))
 return dict(mean=m,lower=m-h,upper=m+h,values=v.tolist(),confidence=confidence,n_refits=5)
def classify(cells):
 yes=[n for n,v in cells.items() if v['lower']>.85];no=[n for n,v in cells.items() if v['upper']<.85];un=[n for n in cells if n not in yes+no]
 return dict(decision='supported' if len(yes)==len(cells) else 'excluded' if no else 'inconclusive',supported=yes,excluded=no,unresolved=un)
def endpoints(scores):
 # axes: refit, model, bank, delay, method, condition
 assert scores.shape==(5,5,4,11,2,2)
 per=scores.mean(2);panel=per[:,:,:,1,:].mean(1)
 a={f'd{d}_{c}':interval(panel[:,di,ci]) for di,d in enumerate(DELAYS) for ci,c in enumerate(['clean','noise'])}
 individual=per[:,:,:,1,:].min((2,3))
 b={str(s):interval(individual[:,i]) for i,s in enumerate(range(301,306))}
 return a,b,per

def main():
 assert json.loads((ROOT/'integrity.json').read_text())['passed']
 scores=np.full((5,5,4,11,2,2),np.nan);negative=[];refs=np.full((5,5,4,8,2),np.nan);orig=np.full((5,5,4,2),np.nan)
 train=[5,7,8,10,11,12,13,15]
 for ri in range(5):
  folder=ROOT/f'repeat{ri+1}'
  for n in ['training_audit','independent_audit']:assert json.loads((folder/f'{n}.json').read_text())['passed']
  nr=[]
  for si,s in enumerate(range(301,306)):
   rows=json.loads((folder/f'seed{s}/results.json').read_text())['cells'];assert len(rows)==44
   for row in rows:
    bi=row['bank'];d=row['delay'];di=DELAYS.index(d);v=np.array(row['scores']);assert row['head_order'][:2]==HEADS
    scores[ri,si,bi,di,:,0]=v[:2,0];scores[ri,si,bi,di,:,1]=v[:2,1:].mean(1)
    nr.append((si,bi,di,np.array(row['negative_means'])))
    if d in train:refs[ri,si,bi,train.index(d)]=[v[2,0],v[2,1:].mean()]
    if d==5:
     o=row['original_scores'];orig[ri,si,bi]=[o[0],np.mean(o[1:])]
  # each batch x delay x active head x condition x control, average models/banks
  negative.append([np.stack([v for si,bi,di,v in nr if di==k]).mean(0).tolist() for k in range(11)])
 assert np.isfinite(scores).all() and np.isfinite(refs).all() and np.isfinite(orig).all()
 a,b,per=endpoints(scores)
 positive={f'ref{d}_{c}':interval(refs.mean((1,2))[:,i,j]) for i,d in enumerate(train) for j,c in enumerate(['clean','noise'])}
 positive.update({f'original5_{c}':interval(orig.mean((1,2))[:,j]) for j,c in enumerate(['clean','noise'])})
 ng=np.concatenate([np.asarray(x).ravel() for rr in negative for x in rr]);posbatch=min(float(refs.mean((1,2)).min()),float(orig.mean((1,2)).min()))
 quality=bool(np.all((ng>=.47)&(ng<=.53)) and posbatch>.9 and all(v['lower']>.9 for v in positive.values()))
 gains={f'd{d}_{c}':interval((per[:,:,DELAYS.index(d),1,j]-per[:,:,DELAYS.index(d),0,j]).mean(1),.95) for d in [14,15] for j,c in enumerate(['clean','noise'])}
 previous=json.loads(Path('artifacts/branch_bw/round1/summary.json').read_text())['per_model_means_delay_head_condition']
 stability={}
 for si,s in enumerate(range(301,306)):
  old=np.array(previous[str(s)])[:,3,:];oi=np.unravel_index(old.argmin(),old.shape);current=per[:,si,:,1,:];mins=[]
  for ri in range(5):
   ix=np.unravel_index(current[ri].argmin(),current[ri].shape)
   mins.append(dict(repeat=ri+1,delay=DELAYS[ix[0]],condition=['clean','noise'][ix[1]],value=float(current[ri][ix])))
  stability[str(s)]={'bw_minimum':dict(delay=DELAYS[oi[0]],condition=['clean','noise'][oi[1]],value=float(old[oi])),'same_condition_values':current[:,oi[0],oi[1]].tolist(),'same_condition_ranks':[int(1+(x<x[oi]).sum()) for x in current],'new_minima':mins}
 result=dict(quality_passed=quality,panel=a,individual=b,panel_classification=classify(a),individual_classification=classify(b),gains_descriptive=gains,positive_controls=positive,positive_batch_minimum=posbatch,negative_range=[float(ng.min()),float(ng.max())],negative_by_batch_delay=negative,stability=stability)
 np.savez_compressed(ROOT/'analysis_arrays.npz',scores=scores,per_refit=per,references=refs,original=orig)
 (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ['quality_passed','panel_classification','individual_classification','negative_range']},indent=2))
if __name__=='__main__':main()
