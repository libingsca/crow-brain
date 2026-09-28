"""Simultaneous inference for the entire pre-round4 uncertainty/look family."""
import sys,json,copy
import numpy as np
from scipy.stats import t,binom
from branch_bi5_common import *
from branch_bi5_statistics import family_upper,reused_rate,union_states
from branch_bi_codec import load_gzip_json

def quality_ci(v):
 v=np.asarray(v);m=float(v.mean());h=float(t.isf(.005,len(v)-1)*v.std(ddof=1)/len(v)**.5);return dict(mean=m,lower=m-h,upper=m+h,n_banks=len(v),alpha=.01)
def main(j):
 verify_lock();n=bounds(j)[1];targets=CFG['targets'];m=len(targets);slots={(x['head'],x['delay'],x['input']):i for i,x in enumerate(targets)};pairs=[(h,d) for d in DELAYS for h in BY_DELAY[d]];pair_ix={p:i for i,p in enumerate(pairs)};primary=np.full((m,n),np.nan);neg=np.full((n,len(pairs),2,3),np.nan);refs=np.full((n,len(REFS),2),np.nan);rn=np.full((n,len(REFS),2,3),np.nan);orig=np.full((n,2),np.nan);audits=[];records=[];overlap=0
 for k in range(2,j+1):
  root=stage(k);aud=json.loads((root/'independent_audit.json').read_text());assert aud['passed'];audits.append(aud);records+=json.loads((root/'data_manifest.json').read_text())['banks'];meta=load_gzip_json(root/'seed301/results.json.gz');a,b=bounds(k);assert meta['cells_count']==(b-a)*len(DELAYS);overlap+=meta['overlap_prediction_vectors']
  for bi in range(a,b):
   sp=root/f'seed301/bank{bi:04d}_scores.json.gz';assert sha(sp)==meta['bank_files'][str(sp)]
   for row in load_gzip_json(sp):
    assert row['bank']==bi;d=row['delay'];expected=BY_DELAY[d]+(['reference'] if d in REFS else []);assert row['head_order']==expected;v=np.asarray(row['scores'])
    for hi,head in enumerate(expected):
     means=[v[hi,0],v[hi,1:].mean()]
     if head=='reference':
      ri=REFS.index(d);refs[bi,ri]=means;rn[bi,ri]=row['negative_means'][hi]
     else:
      neg[bi,pair_ix[head,d]]=row['negative_means'][hi]
      for ci,c in enumerate(['clean','noise']):
       if (head,d,c) in slots:primary[slots[head,d,c],bi]=means[ci]
    if d==5:orig[bi]=[row['original_scores'][0],np.mean(row['original_scores'][1:])]
 assert all(np.isfinite(x).all() for x in [primary,neg,refs,rn,orig]);point_alpha=CFG['measurement_alpha']/(m*len(CFG['family_look_sizes']));available=[v for v in CFG['family_look_sizes'] if v<=n];evidence=[];failed=set()
 for i,target in enumerate(targets):
  bounds_by_n={}
  for size in available:
   v=primary[i,:size];x=family_upper(v,point_alpha);mean=sum(float(z) for z in v)/size;sd=(sum((float(z)-mean)**2 for z in v)/(size-1))**.5;upper=mean+t.ppf(1-point_alpha,size-1)*sd/size**.5;np.testing.assert_allclose([mean,upper],[x['mean'],x['upper']],rtol=0,atol=1e-12);bounds_by_n[str(size)]=x
  best=min(available,key=lambda size:bounds_by_n[str(size)]['upper']);upper=bounds_by_n[str(best)]['upper'];cert=upper<.85
  if cert:failed.add(target['head'])
  evidence.append({**target,'bounds_by_n':bounds_by_n,'best_n':best,'best_upper':upper,'certified_failure':cert})
 prior=json.loads((P4/'round_summary.json').read_text());old={b:x['state'] for b,x in prior['batches'].items()};combined=union_states(old,failed);states=[combined[str(i)] for i in range(1,97)];gate=reused_rate(states,CFG['reused_rate_alpha']);a=CFG['reused_rate_alpha']/2
 if gate['F']:np.testing.assert_allclose(binom.cdf(96-gate['F'],96,gate['upper']),a,rtol=0,atol=1e-12)
 if gate['S']:np.testing.assert_allclose(binom.sf(gate['S']-1,96,gate['lower']),a,rtol=0,atol=1e-12)
 pos={f'reference{d}_{c}':quality_ci(refs[:,di,ci]) for di,d in enumerate(REFS) for ci,c in enumerate(['clean','noise'])};pos.update({f'original5_{c}':quality_ci(orig[:,ci]) for ci,c in enumerate(['clean','noise'])});ng=np.r_[neg.mean(0).ravel(),rn.mean(0).ravel()];assert all(x['lower']>.9 for x in pos.values()) and ((ng>=.47)&(ng<=.53)).all();batches=copy.deepcopy(prior['batches'])
 for b,state in combined.items():
  batches[b]['state']=state;batches[b]['round5_certificates']=[e['condition'] for e in evidence if str(e['batch'])==b and e['certified_failure']]
 out=dict(decision=gate['decision'],gate=gate,main_p_lower=0.,main_p_upper=gate['upper'],bridge='Original five-core event E implies the same round3 core301 event A, hence p<=q.',look=j,n_banks=n,K=96,new_fits=0,measurement_family_conditions=m,measurement_family_looks=CFG['family_look_sizes'],computed_looks=available,measurement_alpha=CFG['measurement_alpha'],pointwise_alpha=point_alpha,new_rate_alpha_spent=0,reused_rate_event_round=3,reused_rate_alpha=CFG['reused_rate_alpha'],certified_heads_from_family=sorted(failed,key=lambda h:int(h[1:])),newly_certified_heads=sorted([h for h in failed if old[h[1:]]=='U'],key=lambda h:int(h[1:])),target_evidence=evidence,batches=batches,quality_passed=True,aggregation_crosschecked=True,beta_binomial_identity_crosschecked=True,positive_controls=pos,negative_range=[float(ng.min()),float(ng.max())],old_overlap_prediction_vectors=overlap,data_reuse_disclosed=True,reused_banks=CFG['reused_banks'],new_images_generated=max(0,n-CFG['reused_banks'])*1000,conservative_cumulative_budget=.0484375,next_action='stop_main' if gate['decision']=='excluded' else 'review_complete_result')
 root=stage(j);save(root/'round_summary.json',out);np.savez_compressed(root/'analysis_arrays.npz',primary=primary,negative=neg,references=refs,reference_negative=rn,original=orig);save(R/'round_summary.json',out);save(R/'independent_audit.json',dict(passed=True,completed_looks=list(range(2,j+1)),cells=sum(x['cells'] for x in audits),prediction_vectors=sum(x['prediction_vectors'] for x in audits),original_probe_slice_checks=sum(x['original_probe_slice_checks'] for x in audits),all_noise_reconstructed=True,recurrence_rebuilt_from_pixels=True,frozen_parameters_verified=True,source_audit_hashes={str(stage(k)/'independent_audit.json'):sha(stage(k)/'independent_audit.json') for k in range(2,j+1)}));save(R/'data_manifest.json',dict(banks=records,prior_files=base_prior(),prior_images=CFG['prior_images'],reused_banks=CFG['reused_banks'],new_images_generated=max(0,n-CFG['reused_banks'])*1000,unique_total=CFG['prior_images']+max(0,n-CFG['reused_banks'])*1000))
 print(json.dumps({k:out[k] for k in ['look','n_banks','gate','newly_certified_heads','next_action']},indent=2),flush=True)
if __name__=='__main__':main(int(sys.argv[1]))
