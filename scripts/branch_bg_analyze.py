"""Preregistered eight-criterion primary family, separate descriptive full matrix."""
import json
import numpy as np
from scipy.stats import t
from branch_bg_round1 import ROOT,HEADS,DELAYS,REFERENCE_DELAYS
from branch_b_round1 import save

def interval(v,family):
 v=np.asarray(v);assert v.shape==(20,);mu=float(v.mean());half=float(t.isf(.05/family/2,19)*v.std(ddof=1)/np.sqrt(20))
 return {'mean':mu,'lower':mu-half,'upper':mu+half,'confidence':1-.05/family,'n_banks':20}

def main():
 assert all(json.loads((ROOT/f'{n}.json').read_text())['passed'] for n in ['training_audit','independent_audit'])
 scores=np.full((5,20,11,4,2),np.nan);neg=np.full((5,20,11,4,2,3),np.nan);refs=np.full((5,20,8,2),np.nan);rn=np.full((5,20,8,2,3),np.nan);orig=np.full((5,20,2),np.nan)
 for si,seed in enumerate(range(301,306)):
  result=json.loads((ROOT/f'seed{seed}/results.json').read_text());assert len(result['cells'])==220
  for row in result['cells']:
   b=row['bank'];d=row['delay'];di=DELAYS.index(d);a=np.array(row['scores']);assert row['head_order'][:4]==HEADS
   scores[si,b,di,:,0]=a[:4,0];scores[si,b,di,:,1]=a[:4,1:].mean(1);neg[si,b,di]=row['negative_means'][:4]
   if d in REFERENCE_DELAYS:
    j=REFERENCE_DELAYS.index(d);refs[si,b,j]=[a[4,0],a[4,1:].mean()];rn[si,b,j]=row['negative_means'][4]
   if d==5:orig[si,b]=[row['original_scores'][0],np.mean(row['original_scores'][1:])]
 assert all(np.isfinite(a).all() for a in [scores,neg,refs,rn,orig])
 primary={}
 for si,seed in enumerate([301,302]):
  for ci,c in enumerate(['clean','noise']):
   candidate=scores[si,:,10,3,ci];baseline=scores[si,:,10,0,ci]
   for metric,v,threshold in [('absolute',candidate,.85),('gain',candidate-baseline,.02)]:primary[f'{seed}_d15_{c}_{metric}']={**interval(v,8),'threshold':threshold}
 states={k:[name for name,v in primary.items() if (v['lower']>v['threshold'] if k=='supported' else v['upper']<v['threshold'] if k=='excluded' else v['lower']<=v['threshold']<=v['upper'])] for k in ['supported','excluded','unresolved']}
 full={f'{seed}_d{d}_{c}':interval(scores[si,:,di,3,ci],110) for si,seed in enumerate(range(301,306)) for di,d in enumerate(DELAYS) for ci,c in enumerate(['clean','noise'])}
 positive={f'reference_d{d}_{c}':interval(refs[:,:,j,ci].mean(0),8) for j,d in enumerate(REFERENCE_DELAYS) for ci,c in enumerate(['clean','noise'])};positive.update({f'original5_{c}':interval(orig[:,:,ci].mean(0),8) for ci,c in enumerate(['clean','noise'])})
 negative=np.r_[neg.mean((0,1)).ravel(),rn.mean((0,1)).ravel()];quality=bool(((negative>=.47)&(negative<=.53)).all()) and all(v['lower']>.9 for v in positive.values())
 decision='quality_failed' if not quality else ('supported' if len(states['supported'])==8 else 'excluded' if states['excluded'] else 'inconclusive')
 effects={}
 for seed,d in [(301,6),(301,15),(302,15),(304,7)]:
  v=scores[seed-301,:,DELAYS.index(d),:,0].mean(0);effects[f'{seed}_d{d}_clean']={'means':dict(zip(HEADS,v.tolist())),'regularization_soft':float(v[1]-v[0]),'regularization_sharp':float(v[3]-v[2]),'temperature_strong':float(v[2]-v[0]),'temperature_weak':float(v[3]-v[1]),'interaction':float(v[3]-v[2]-v[1]+v[0])}
 summary={'decision':decision,'quality_passed':quality,'primary':primary,**states,'secondary_full':full,'secondary_classification':{k:[n for n,v in full.items() if (v['lower']>.85 if k=='supported' else v['upper']<.85 if k=='excluded' else v['lower']<=.85<=v['upper'])] for k in states},'effects':effects,'positive_controls':positive,'negative_range':[float(negative.min()),float(negative.max())],'negative_means':neg.mean((0,1)).tolist(),'reference_negative_means':rn.mean((0,1)).tolist(),'per_model_minimum':{str(s):float(scores[i,:,:,3,:].mean(0).min()) for i,s in enumerate(range(301,306))}}
 save(ROOT/'summary.json',summary);np.savez_compressed(ROOT/'analysis_arrays.npz',scores=scores,negative=neg,references=refs,reference_negative=rn,original=orig)
 print(json.dumps({k:summary[k] for k in ['decision','quality_passed','primary','supported','excluded','unresolved','effects','per_model_minimum','negative_range']},indent=2))
if __name__=='__main__':main()
