"""B-C complete-matrix analysis: 22 absolute and six paired endpoints."""
import json
import numpy as np
from scipy.stats import t
from branch_bc_round1 import ROOT,DELAYS,TRAIN_DELAYS,BASE_DELAYS,HEADS
from branch_b_round1 import save
TARGETS=[8,11,12]
def interval(values):
 v=np.asarray(values,dtype=float);assert v.shape==(20,) and np.isfinite(v).all()
 mean=float(v.mean());half=float(t.ppf(1-.05/28/2,19)*v.std(ddof=1)/np.sqrt(20))
 return {'mean':mean,'lower':mean-half,'upper':mean+half,'confidence':1-.05/28,'n_banks':20}
def classify(cells):
 yes=[k for k,v in cells.items() if v['lower']>v['threshold']]
 no=[k for k,v in cells.items() if v['upper']<v['threshold']]
 undecided=[k for k in cells if k not in yes+no]
 return {'decision':'supported' if len(yes)==len(cells) else 'excluded' if no else 'inconclusive','supported':yes,'excluded':no,'unresolved':undecided}
def main():
 for name in ['training_audit','independent_audit','data_chronology_integrity']:assert json.loads((ROOT/f'{name}.json').read_text())['passed']
 scores=np.full((5,20,11,4,2),np.nan);negative=np.full((5,20,11,4,2,3),np.nan)
 refs=np.full((5,20,8,2),np.nan);refnegative=np.full((5,20,8,2,3),np.nan);original=np.full((5,20,2),np.nan)
 for si,seed in enumerate(range(301,306)):
  r=json.loads((ROOT/f'seed{seed}/results.json').read_text());assert len(r['cells'])==220
  assert {(c['bank'],c['delay']) for c in r['cells']}=={(b,d) for b in range(20) for d in DELAYS}
  for cell in r['cells']:
   bi=cell['bank'];d=cell['delay'];di=DELAYS.index(d);s=np.asarray(cell['scores']);neg=np.asarray(cell['negative_means']);assert cell['head_order'][:4]==HEADS
   scores[si,bi,di,:,0]=s[:4,0];scores[si,bi,di,:,1]=s[:4,1:].mean(-1);negative[si,bi,di]=neg[:4]
   if d in TRAIN_DELAYS:
    ti=TRAIN_DELAYS.index(d);refs[si,bi,ti]=[s[4,0],s[4,1:].mean()];refnegative[si,bi,ti]=neg[4]
   if d==5:
    o=np.asarray(cell['original_scores']);original[si,bi]=[o[0],o[1:].mean()]
 assert all(np.isfinite(x).all() for x in [scores,negative,refs,refnegative,original])
 absolute={f'absolute_d{d}_{c}':{**interval(scores[:,:,di,3,ci].mean(0)),'threshold':.85,'role':'base_trained' if d in BASE_DELAYS else 'added' if d in TARGETS else 'not_fitted'} for di,d in enumerate(DELAYS) for ci,c in enumerate(['clean','noise'])}
 gain={f'gain_d{d}_{c}':{**interval((scores[:,:,DELAYS.index(d),3,ci]-scores[:,:,DELAYS.index(d),1,ci]).mean(0)),'threshold':.02} for d in TARGETS for ci,c in enumerate(['clean','noise'])}
 primary={**absolute,**gain};assert len(primary)==28
 positive={f'reference_d{d}_{c}':interval(refs[:,:,ti,ci].mean(0)) for ti,d in enumerate(TRAIN_DELAYS) for ci,c in enumerate(['clean','noise'])}
 positive.update({f'original5_{c}':interval(original[:,:,ci].mean(0)) for ci,c in enumerate(['clean','noise'])})
 neg=negative.mean((0,1));refneg=refnegative.mean((0,1));ng=np.r_[neg.ravel(),refneg.ravel()];ngpass=bool(((ng>=.47)&(ng<=.53)).all());quality=ngpass and all(v['lower']>.9 for v in positive.values())
 stats=classify(primary);decision=stats['decision'] if quality else 'quality_failed'
 result={'decision':decision,'quality_passed':quality,'primary':primary,'absolute':absolute,'gains':gain,'primary_classification':stats,'absolute_classification':classify(absolute),'gain_classification':classify(gain),'positive_controls':positive,'negative_controls_passed':ngpass,'negative_range':[float(ng.min()),float(ng.max())],'negative_means_delay_head_condition_control':neg.tolist(),'reference_negative_means':refneg.tolist(),'head_order':HEADS,'per_model_means_delay_head_condition':{str(seed):scores.mean(1)[i].tolist() for i,seed in enumerate(range(301,306))},'per_model_expanded_aug_minimum':{str(seed):float(scores.mean(1)[i,:,3].min()) for i,seed in enumerate(range(301,306))},'paired_expanded_aug_minus_base_aug':(scores[:,:,:,3]-scores[:,:,:,1]).mean((0,1)).tolist(),'paired_expanded_aug_minus_expanded_clean':(scores[:,:,:,3]-scores[:,:,:,2]).mean((0,1)).tolist(),'panel_all_cells':{f'{h}_d{d}_{c}':interval(scores[:,:,di,hi,ci].mean(0)) for hi,h in enumerate(HEADS) for di,d in enumerate(DELAYS) for ci,c in enumerate(['clean','noise'])}}
 np.savez_compressed(ROOT/'analysis_arrays.npz',scores=scores,negative=negative,references=refs,reference_negative=refnegative,original=original);save(ROOT/'summary.json',result)
 print(json.dumps({k:result[k] for k in ['decision','quality_passed','primary_classification','absolute_classification','gain_classification','negative_range','per_model_expanded_aug_minimum']},indent=2))
if __name__=='__main__':main()
