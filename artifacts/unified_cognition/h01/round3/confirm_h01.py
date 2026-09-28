"""Fixed-look confirmation; refuses to evaluate incomplete registered batches."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
from scipy import stats
import h01
from statistics_h01 import interval
roundname=sys.argv[1] if len(sys.argv)>1 else 'round3'
p=h01.BASE/roundname
cfg=json.loads((p/'config.json').read_text());reg=json.loads((p/'registration.json').read_text())
assert h01.digest(p/'config.json')==reg['config_sha256']
for f,digest in reg['files'].items():assert h01.digest(h01.ROOT/f)==digest, f
assert not list(p.glob('failure_*.json'))
lookup={}
for j in cfg['jobs']:
 f=p/f'{j["model"]}_s{j["seed"]}_lr{j["lr"]:g}'/'result.json'
 assert f.exists(),f
 r=json.loads(f.read_text());assert r['steps']==cfg['steps'] and r['lr']==j['lr']
 assert r['source_sha256']==reg['files']['scripts/unified_cognition/h01.py']
 assert r['product_source_sha256']==reg['files']['scripts/unified_cognition/h01_product.py']
 lookup[j['model'],j['seed']]=r
assert len(lookup)==len(cfg['jobs'])
aud=json.loads((p/'summary_audited.json').read_text());assert aud['n_runs']==len(lookup)
assert aud['all_input_truth_partitions_predictions_match'] and aud['identical_training_records_within_seed']
# Check identical evaluation inputs within independent training seed blocks.
for seed in sorted({s for m,s in lookup}):
 for cond in ['val','id','ood','delay8']:
  hashes=[]
  for m in ['S64','FP','F64']:
   if (m,seed) not in lookup:continue
   rec=np.load(p/f'{m}_s{seed}_lr0.003'/f'{cond}.npz')['records']
   hashes.append(hashlib.sha256(rec.tobytes()).hexdigest())
  assert len(set(hashes))==1
alpha=reg['alpha_each'];out={'round':roundname,'alpha_total':reg['alpha_total'],'alpha_each':alpha,'paired_evaluation_records_match':True,'comparisons':{},'models':{}}
for m in ['S64','FP','F64']:
 rows=[r for (mm,s),r in lookup.items() if mm==m]
 out['models'][m]={'n':len(rows),'params':rows[0]['nparams'],'id_mean':float(np.mean([r['id']['accuracy'] for r in rows])),'ood_mean':float(np.mean([r['ood']['accuracy'] for r in rows])),'id_below_85_count':sum(r['id']['accuracy']<.85 for r in rows),'reset_min':min(r[c]['reset_accuracy'] for r in rows for c in ['id','ood','delay8']),'reset_max':max(r[c]['reset_accuracy'] for r in rows for c in ['id','ood','delay8'])}
for baseline in ['FP','F64']:
 seeds=sorted(s for (m,s) in lookup if m==baseline)
 measures={}
 for cond in ['ood','id']:
  d=np.array([lookup['S64',s][cond]['accuracy']-lookup[baseline,s][cond]['accuracy'] for s in seeds])
  ci=interval(d,alpha);ref=stats.t.interval(1-alpha,len(d)-1,loc=d.mean(),scale=stats.sem(d)) if d.std()>0 else (d.mean(),d.mean())
  assert np.allclose([ci['lower'],ci['upper']],ref)
  ci.update({'seed_ids':seeds,'differences':d.tolist(),'min':float(d.min()),'max':float(d.max()),'median':float(np.median(d)),'skew_descriptive':float(stats.skew(d)) if d.std()>0 else 0.})
  measures[cond]=ci
 measures['useful_ood_gain_excluded']=measures['ood']['upper']<.03
 measures['useful_ood_gain_supported_once']=measures['ood']['lower']>.03
 measures['id_noninferiority_supported']=measures['id']['lower']>-.02
 out['comparisons'][baseline]=measures
out['quality_gate']=all(v['id_mean']>=.85 and v['reset_min']==.5 and v['reset_max']==.5 for v in out['models'].values())
primary=out['comparisons']['FP']
out['decision']='exclude_useful_gain_scoped' if out['quality_gate'] and primary['useful_ood_gain_excluded'] else 'requires_next_result_based_round'
h01.dump(p/'confirmation.json',out)
print(json.dumps({k:v for k,v in out.items() if k!='comparisons'},indent=2))
for b,m in out['comparisons'].items():
 for c in ['ood','id']:print(b,c,{k:m[c][k] for k in ['n','mean','lower','upper']})
if out['decision']=='exclude_useful_gain_scoped':
 h01.dump(p/'STOP.json',{'hypothesis':'UC-H01','decision':out['decision'],'threshold':.03,'primary_baseline':'FP52','primary_interval':{k:primary['ood'][k] for k in ['n','mean','lower','upper','alpha']},'alpha_cumulative':reg['alpha_total'],'quality_gate':True,'reason':'upper confidence limit below registered +3pp useful-effect threshold','limits':['fixed structured task and attribute split','48/16 state partition with symmetric product readout','fixed 2000-step training algorithm','approximate seed-level t coverage','not a universal or biological falsification'],'confirmation_sha256':h01.digest(p/'confirmation.json')})
