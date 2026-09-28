import json
import numpy as np
from scipy.stats import t
from branch_bh3_round import ROOT,CFG,N_BANKS,HEADS,DELAYS,REFERENCE_DELAYS,verify_lock
from branch_b_round1 import save

def interval(v,alpha=.025):
 v=np.asarray(v);assert v.shape==(N_BANKS,);m=float(v.mean());h=float(t.isf(alpha/2,N_BANKS-1)*v.std(ddof=1)/np.sqrt(N_BANKS));return dict(mean=m,lower=m-h,upper=m+h,n_banks=N_BANKS,confidence=1-alpha)

def main():
 assert json.loads((ROOT/'independent_audit.json').read_text())['passed'];verify_lock();r=json.loads((ROOT/'seed301/results.json').read_text());assert len(r['cells'])==N_BANKS*3
 scores=np.full((N_BANKS,3,2,2),np.nan);neg=np.full((N_BANKS,3,2,2,3),np.nan);ref=np.full((N_BANKS,2,2),np.nan);rn=np.full((N_BANKS,2,2,3),np.nan);orig=np.full((N_BANKS,2),np.nan)
 for row in r['cells']:
  b=row['bank'];di=DELAYS.index(row['delay']);a=np.array(row['scores']);assert row['head_order'][:2]==HEADS;scores[b,di,:,0]=a[:2,0];scores[b,di,:,1]=a[:2,1:].mean(1);neg[b,di]=row['negative_means'][:2]
  if row['delay'] in REFERENCE_DELAYS:
   j=REFERENCE_DELAYS.index(row['delay']);ref[b,j]=[a[2,0],a[2,1:].mean()];rn[b,j]=row['negative_means'][2]
  if row['delay']==5:orig[b]=[row['original_scores'][0],np.mean(row['original_scores'][1:])]
 assert all(np.isfinite(x).all() for x in [scores,neg,ref,rn,orig]);target=interval(scores[:,1,1,1]);pos={f'ref{d}_{c}':interval(ref[:,j,ci]) for j,d in enumerate(REFERENCE_DELAYS) for ci,c in enumerate(['clean','noise'])};pos.update({f'original5_{c}':interval(orig[:,ci]) for ci,c in enumerate(['clean','noise'])})
 nv=np.r_[neg.mean(0).ravel(),rn.mean(0).ravel()];quality=bool(((nv>=.47)&(nv<=.53)).all()) and all(v['lower']>.9 for v in pos.values())
 # Recompute all 110 tightened old intervals from audited per-bank records.
 bridge=json.loads((ROOT/'bridge.json').read_text());count=0
 for seed in range(301,306):
  rows=json.loads(open(f'artifacts/branch_bh/round2/seed{seed}/results.json').read())['cells']
  for d in range(5,16):
   a=np.array([x['scores'] for x in sorted([x for x in rows if x['delay']==d],key=lambda x:x['bank'])]);assert len(a)==80
   for ci,c in enumerate(['clean','noise']):
    v=a[:,1,0] if ci==0 else a[:,1,1:].sum(1)/10;mu=sum(v)/80;h=t.ppf(1-.025/220,79)*np.sqrt(sum((v-mu)**2)/79)/np.sqrt(80);key=f'{seed}_d{d}_{c}';e=bridge['all_old_intervals'][key];np.testing.assert_allclose([mu,mu-h,mu+h],[e[k] for k in ['mean','lower','upper']],atol=1e-12,rtol=0);count+=1
 assert count==110 and len(bridge['passed'])==109 and all(bridge['all_old_intervals'][k]['lower']>.85 for k in bridge['passed']) and bridge['remaining']==['301_d9_noise']
 # Independent new interval aggregation directly from bank records.
 v=np.array([np.mean(x['scores'][1][1:]) for x in sorted([x for x in r['cells'] if x['delay']==9],key=lambda x:x['bank'])]);mu=sum(v)/N_BANKS;half=t.ppf(1-.025/2,N_BANKS-1)*np.sqrt(sum((v-mu)**2)/(N_BANKS-1))/np.sqrt(N_BANKS);np.testing.assert_allclose([mu,mu-half,mu+half],[target[k] for k in ['mean','lower','upper']],atol=1e-12,rtol=0)
 decision='quality_failed' if not quality else 'supported' if target['lower']>.85 else 'excluded' if target['upper']<.85 else 'inconclusive'
 final=dict(bridge['all_old_intervals']);final['301_d9_noise']=target
 save(ROOT/'summary.json',{'stage':'targeted_confirmation','decision':decision,'quality_passed':quality,'target':target,'final_intervals':final,'positive_controls':pos,'negative_range':[float(nv.min()),float(nv.max())],'negative_means':neg.mean(0).tolist(),'reference_negative_means':rn.mean(0).tolist(),'final_joint_error_bound':.05,'old_family_alpha':.025,'new_target_alpha':.025,'source_counts':{'old_conditions':109,'old_banks':80,'new_conditions':1,'new_banks':N_BANKS}})
 save(ROOT/'summary_crosscheck.json',{'passed':True,'old_intervals':110,'new_target_intervals':1,'old_109_passed':True});np.savez_compressed(ROOT/'analysis_arrays.npz',scores=scores,negative=neg,references=ref,reference_negative=rn,original=orig)
 print(json.dumps({'decision':decision,'quality':quality,'target':target,'negative_range':[float(nv.min()),float(nv.max())]},indent=2))
if __name__=='__main__':main()
