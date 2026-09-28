import json
import numpy as np
from scipy.stats import t
from branch_bh2_round import ROOT,CFG,HEADS,DELAYS,REFERENCE_DELAYS,N_BANKS
from branch_b_round1 import save

def interval(v):
 v=np.asarray(v);assert v.shape==(N_BANKS,);m=float(v.mean());h=float(t.isf(.05/220,N_BANKS-1)*v.std(ddof=1)/np.sqrt(N_BANKS));return dict(mean=m,lower=m-h,upper=m+h,confidence=1-.05/110,n_banks=N_BANKS)

def main():
 assert all(json.loads((ROOT/f'{f}.json').read_text())['passed'] for f in ['training_audit','independent_audit']);nh=len(HEADS);s=np.full((5,N_BANKS,11,nh,2),np.nan);neg=np.full((5,N_BANKS,11,nh,2,3),np.nan);ref=np.full((5,N_BANKS,8,2),np.nan);rn=np.full((5,N_BANKS,8,2,3),np.nan);orig=np.full((5,N_BANKS,2),np.nan)
 for si,seed in enumerate(range(301,306)):
  rows=json.loads((ROOT/f'seed{seed}/results.json').read_text())['cells'];assert len(rows)==N_BANKS*11
  for row in rows:
   b=row['bank'];di=row['delay']-5;a=np.array(row['scores']);assert row['head_order'][:nh]==HEADS;s[si,b,di,:,0]=a[:nh,0];s[si,b,di,:,1]=a[:nh,1:].mean(1);neg[si,b,di]=row['negative_means'][:nh]
   if row['delay'] in REFERENCE_DELAYS:
    j=REFERENCE_DELAYS.index(row['delay']);ref[si,b,j]=[a[nh,0],a[nh,1:].mean()];rn[si,b,j]=row['negative_means'][nh]
   if row['delay']==5:orig[si,b]=[row['original_scores'][0],np.mean(row['original_scores'][1:])]
 assert all(np.isfinite(a).all() for a in [s,neg,ref,rn,orig]);hi=HEADS.index(CFG['candidate'])
 primary={f'{seed}_d{d}_{c}':interval(s[i,:,d-5,hi,ci]) for i,seed in enumerate(range(301,306)) for d in DELAYS for ci,c in enumerate(['clean','noise'])}
 states={k:[n for n,v in primary.items() if (v['lower']>.85 if k=='supported' else v['upper']<.85 if k=='excluded' else v['lower']<=.85<=v['upper'])] for k in ['supported','excluded','unresolved']}
 pos={f'reference_d{d}_{c}':interval(ref[:,:,j,ci].mean(0)) for j,d in enumerate(REFERENCE_DELAYS) for ci,c in enumerate(['clean','noise'])};pos.update({f'original5_{c}':interval(orig[:,:,ci].mean(0)) for ci,c in enumerate(['clean','noise'])})
 ng=np.r_[neg.mean((0,1)).ravel(),rn.mean((0,1)).ravel()];quality=bool(((ng>=.47)&(ng<=.53)).all()) and all(v['lower']>.9 for v in pos.values());decision='quality_failed' if not quality else ('supported' if len(states['supported'])==110 else 'excluded' if states['excluded'] else 'inconclusive')
 result={'stage':CFG['stage'],'decision':decision,'quality_passed':quality,'primary':primary,**states,'positive_controls':pos,'negative_range':[float(ng.min()),float(ng.max())],'negative_means':neg.mean((0,1)).tolist(),'reference_negative_means':rn.mean((0,1)).tolist(),'per_model_minimum':{str(seed):float(s[i,:,:,hi,:].mean(0).min()) for i,seed in enumerate(range(301,306))}}
 save(ROOT/'summary.json',result);np.savez_compressed(ROOT/'analysis_arrays.npz',scores=s,negative=neg,references=ref,reference_negative=rn,original=orig)
 print(json.dumps({k:result[k] for k in ['stage','decision','quality_passed','per_model_minimum','negative_range']},indent=2));print({k:len(v) for k,v in states.items()})
 for n in states['excluded']+states['unresolved']:print(n,primary[n])
if __name__=='__main__':main()
