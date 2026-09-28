"""Prospectively spent fixed-look tests; fixed sequence controls family error."""
import sys,json,copy
import numpy as np
from scipy.stats import t,beta
from branch_bi4_common import *
from branch_bi_codec import load_gzip_json
from branch_bi4_statistics import upper_ci,prefix_update
from branch_bi_statistics import rate

def twosided(v,alpha=.01):
 v=np.asarray(v);m=float(v.mean());h=float(t.isf(alpha/2,len(v)-1)*v.std(ddof=1)/len(v)**.5);return dict(mean=m,lower=m-h,upper=m+h,n_banks=len(v),alpha=alpha)
def main(j):
 verify_lock();n=bounds(j)[1];nh=len(HEADS);scores=np.full((nh,n,len(DELAYS),2),np.nan);neg=np.full((nh,n,len(DELAYS),2,3),np.nan);refs=np.full((n,len(REFS),2),np.nan);rn=np.full((n,len(REFS),2,3),np.nan);orig=np.full((n,2),np.nan);audits=[];banks=[]
 for k in range(1,j+1):
  root=stage(k);aud=json.loads((root/'independent_audit.json').read_text());assert aud['passed'];audits.append(aud);dm=json.loads((root/'data_manifest.json').read_text());banks+=dm['banks'];meta=load_gzip_json(root/'seed301/results.json.gz');a,b=bounds(k);assert meta['cells_count']==(b-a)*len(DELAYS)
  for bi in range(a,b):
   sp=root/f'seed301/bank{bi:04d}_scores.json.gz';assert sha(sp)==meta['bank_files'][str(sp)]
   for row in load_gzip_json(sp):
    assert row['bank']==bi and row['head_order'][:nh]==HEADS;di=DELAYS.index(row['delay']);v=np.asarray(row['scores']);scores[:,bi,di,0]=v[:nh,0];scores[:,bi,di,1]=v[:nh,1:].mean(1);neg[:,bi,di]=row['negative_means'][:nh]
    if row['delay'] in REFS:
     ix=REFS.index(row['delay']);refs[bi,ix]=[v[nh,0],v[nh,1:].mean()];rn[bi,ix]=row['negative_means'][nh]
    if row['delay']==5:orig[bi]=[row['original_scores'][0],np.mean(row['original_scores'][1:])]
 assert all(np.isfinite(v).all() for v in [scores,neg,refs,rn,orig]);alpha=CFG['measurement_alpha']*CFG['look_alpha_fractions'][j-1];evidence=[];diag=[]
 for i,target in enumerate(CFG['targets']):
  _,ds,cond=target['condition'].split('_');v=scores[i,:,DELAYS.index(int(ds[1:])),['clean','noise'].index(cond)];x=upper_ci(v,alpha);mean=sum(float(z) for z in v)/n;sd=(sum((float(z)-mean)**2 for z in v)/(n-1))**.5;upper=mean+t.ppf(1-alpha,n-1)*sd/n**.5;np.testing.assert_allclose([mean,upper],[x['mean'],x['upper']],rtol=0,atol=1e-12);evidence.append({**target,**x});diag.append(twosided(v))
 previous_prefix=0 if j==1 else json.loads((stage(j-1)/'round_summary.json').read_text())['certified_prefix'];prefix=prefix_update(previous_prefix,[x['upper'] for x in evidence]);prior=json.loads((PRIOR/'round_summary.json').read_text());batches=copy.deepcopy(prior['batches'])
 for x in CFG['targets'][:prefix]:
  b=str(x['batch']);assert prior['batches'][b]['state']=='U';batches[b]['state']='F';batches[b]['round4_failure_certificate']=dict(condition=x['condition'],fixed_sequence=True,prefix=prefix,look=j)
 states=[batches[str(i)]['state'] for i in range(1,97)];gate=rate(states,r=4);assert gate['F']==CFG['initial_F']+prefix and gate['S']==CFG['initial_S'];np.testing.assert_allclose(gate['upper'],beta.isf(CFG['rate_alpha']/2,97-gate['F'],gate['F']),rtol=0,atol=1e-12)
 pos={f'reference{d}_{c}':twosided(refs[:,di,ci]) for di,d in enumerate(REFS) for ci,c in enumerate(['clean','noise'])};pos.update({f'original5_{c}':twosided(orig[:,ci]) for ci,c in enumerate(['clean','noise'])});ng=np.r_[neg.mean(1).ravel(),rn.mean(0).ravel()];assert all(x['lower']>.9 for x in pos.values()) and ((ng>=.47)&(ng<=.53)).all()
 futility=[CFG['targets'][i]['head'] for i,x in enumerate(diag) if i>=prefix and x['lower']>.85];decision='excluded' if gate['upper']<.8 else 'inconclusive';next_action='stop_main' if decision=='excluded' else ('replan' if futility or j==len(CFG['look_sizes']) else 'continue_precision')
 out=dict(decision=decision,gate=gate,main_p_lower=0.,main_p_upper=gate['upper'],bridge='Original E implies core301 event A; p<=q. No gate lower bound is a p lower bound.',look=j,n_banks=n,new_fits=0,certified_prefix=prefix,previous_prefix=previous_prefix,target_evidence=evidence,diagnostic_99_intervals=diag,diagnostic_futility_targets=futility,diagnostic_intervals_are_not_joint_confirmatory=True,batches=batches,quality_passed=True,aggregation_crosschecked=True,positive_controls=pos,negative_range=[float(ng.min()),float(ng.max())],measurement_alpha_this_look=alpha,measurement_alpha_spent=CFG['measurement_alpha']*sum(CFG['look_alpha_fractions'][:j]),rate_alpha=CFG['rate_alpha'],maximum_cumulative_round_budget=.046875,next_action=next_action)
 root=stage(j);save(root/'round_summary.json',out);np.savez_compressed(root/'analysis_arrays.npz',scores=scores,negative=neg,references=refs,reference_negative=rn,original=orig);save(R/'round_summary.json',out);save(R/'independent_audit.json',dict(passed=True,completed_stages=j,cells=sum(x['cells'] for x in audits),prediction_vectors=sum(x['prediction_vectors'] for x in audits),all_noise_reconstructed=True,recurrence_rebuilt_from_pixels=True,frozen_parameters_verified=True,stage_audit_hashes={str(stage(k)/'independent_audit.json'):sha(stage(k)/'independent_audit.json') for k in range(1,j+1)}));save(R/'data_manifest.json',dict(banks=banks,prior_files=base_prior(),prior_images=CFG['prior_images'],new_images=n*1000,unique_total=CFG['prior_images']+n*1000));print(json.dumps({k:out[k] for k in ['look','n_banks','gate','certified_prefix','diagnostic_futility_targets','next_action']},indent=2),flush=True)
if __name__=='__main__':main(int(sys.argv[1]))
