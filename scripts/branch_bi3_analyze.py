"""Necessary-condition reliability; a supported gate is not H-BI support."""
import json
import numpy as np
from scipy.stats import t,beta
from branch_bi3_round import ROOT,CFG,HEADS,DELAYS,REFERENCE_DELAYS,N_BANKS
from branch_bi_codec import load_gzip_json,save_gzip_json
from branch_bi_statistics import rate
from branch_b_round1 import save

def gate_classify(intervals):
 assert len(intervals)==22
 return 'S' if all(lo>.85 for lo,hi in intervals) else 'F' if any(hi<.85 for lo,hi in intervals) else 'U'
def main_decision(gate):return 'excluded' if gate['upper']<.8 else 'inconclusive'
def ci(v,alpha):
 v=np.asarray(v);assert v.shape==(N_BANKS,);m=float(v.mean());h=float(t.isf(alpha/2,N_BANKS-1)*v.std(ddof=1)/N_BANKS**.5);return dict(mean=m,lower=m-h,upper=m+h,alpha=alpha,n_banks=N_BANKS)
def main():
 assert json.loads((ROOT/'independent_audit.json').read_text())['passed'];nh=len(HEADS);scores=np.full((nh,N_BANKS,11,2),np.nan);neg=np.full((nh,N_BANKS,11,2,3),np.nan);ref=np.full((N_BANKS,8,2),np.nan);rn=np.full((N_BANKS,8,2,3),np.nan);orig=np.full((N_BANKS,2),np.nan)
 meta=load_gzip_json(ROOT/'seed301/results.json.gz');assert meta['cells_count']==N_BANKS*11
 rows=(row for b in range(N_BANKS) for row in load_gzip_json(ROOT/f'seed301/bank{b:04d}_scores.json.gz'))
 for row in rows:
  bi=row['bank'];di=row['delay']-5;a=np.array(row['scores']);assert row['head_order'][:nh]==HEADS;scores[:,bi,di,0]=a[:nh,0];scores[:,bi,di,1]=a[:nh,1:].mean(1);neg[:,bi,di]=row['negative_means'][:nh]
  if row['delay'] in REFERENCE_DELAYS:
   j=REFERENCE_DELAYS.index(row['delay']);ref[bi,j]=[a[nh,0],a[nh,1:].mean()];rn[bi,j]=row['negative_means'][nh]
  if row['delay']==5:orig[bi]=[row['original_scores'][0],np.mean(row['original_scores'][1:])]
 assert all(np.isfinite(x).all() for x in [scores,neg,ref,rn,orig]);alpha=CFG['measurement_alpha']/(nh*22);primary={};batches={};states=[]
 for i in range(nh):
  p={}
  for di,d in enumerate(DELAYS):
   for c,name in enumerate(['clean','noise']):
    v=scores[i,:,di,c];x=ci(v,alpha);m=sum(float(y) for y in v)/N_BANKS;sd=(sum((float(y)-m)**2 for y in v)/(N_BANKS-1))**.5;h=t.ppf(1-alpha/2,N_BANKS-1)*sd/N_BANKS**.5;np.testing.assert_allclose([x['mean'],x['lower'],x['upper']],[m,m-h,m+h],atol=1e-12);p[f'301_d{d}_{name}']=x
  state=gate_classify([(x['lower'],x['upper']) for x in p.values()]);states.append(state);primary[str(i+1)]=p;batches[str(i+1)]=dict(state=state,supported=[k for k,v in p.items() if v['lower']>.85],excluded=[k for k,v in p.items() if v['upper']<.85],unresolved=[k for k,v in p.items() if v['lower']<=.85<=v['upper']],minimum_point=min(v['mean'] for v in p.values()))
 pos={f'reference{d}_{name}':ci(ref[:,j,c],.01) for j,d in enumerate(REFERENCE_DELAYS) for c,name in enumerate(['clean','noise'])};pos.update({f'original5_{name}':ci(orig[:,c],.01) for c,name in enumerate(['clean','noise'])});ng=np.r_[neg.mean(1).ravel(),rn.mean(0).ravel()];quality=all(v['lower']>.9 for v in pos.values()) and bool(((ng>=.47)&(ng<=.53)).all());assert quality
 gate=rate(states,r=3);s=gate['S'];u=gate['U'];np.testing.assert_allclose(gate['lower'],0 if s==0 else beta.isf(1-.0015625,s,nh-s+1),atol=1e-12);np.testing.assert_allclose(gate['upper'],1 if s+u==nh else beta.isf(.0015625,s+u+1,nh-s-u),atol=1e-12)
 out=dict(decision=main_decision(gate),gate=gate,main_p_lower=0.,main_p_upper=gate['upper'],bridge='E implies A; p<=q; q support alone does not support p',batches=batches,primary=primary,quality_passed=True,aggregation_crosschecked=True,positive_controls=pos,negative_range=[float(ng.min()),float(ng.max())],cumulative_round_budget=.04375)
 save(ROOT/'round_summary.json',out);np.savez_compressed(ROOT/'analysis_arrays.npz',scores=scores,negative=neg,references=ref,reference_negative=rn,original=orig);print(json.dumps(dict(decision=out['decision'],gate=gate,negative_range=out['negative_range']),indent=2),flush=True)
if __name__=='__main__':main()
