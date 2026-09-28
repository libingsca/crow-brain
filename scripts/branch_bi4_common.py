"""Round4 fixed heads and prospectively registered precision looks."""
import json,pickle,time
from pathlib import Path
from branch_b_round1 import sha,save
from branch_bf_round1 import load_core
from branch_bt_round1 import frozen_snapshot
R=Path('artifacts/branch_bi/round4');PRIOR=Path('artifacts/branch_bi/round3')
CFG=json.loads((R/'config.json').read_text());HEADS=[x['head'] for x in CFG['targets']];DELAYS=CFG['delays'];REFS=CFG['reference_delays']
def stage(j):return R/f'stage{j}'
def bounds(j):return (0 if j==1 else CFG['look_sizes'][j-2]),CFG['look_sizes'][j-1]
def indices(j,part):
 a,b=bounds(j);return range(a+part*(b-a)//2,a+(part+1)*(b-a)//2)
def verify_lock():
 m=json.loads((R/'training_lock.json').read_text());assert all(sha(p)==h for p,h in m['files'].items());return m
def heads():
 with (R/'seed301/heads.pkl').open('rb') as f:return pickle.load(f)
def active(h,refs,d):
 out=dict(h)
 if d in REFS:out['reference']=refs[f'ref{d}']
 return out
def base_prior():
 m=json.loads((PRIOR/'data_manifest.json').read_text());return {**m['prior_files'],**{x['path']:x['sha256'] for x in m['banks']}}
def previous(j):
 out=base_prior()
 for k in range(1,j):out.update({x['path']:x['sha256'] for x in json.loads((stage(k)/'data_manifest.json').read_text())['banks']})
 return out
def lock():
 assert not list(R.glob('stage*/test*.npz'));m=json.loads((PRIOR/'training_lock.json').read_text());assert all(sha(p)==h for p,h in m['files'].items());h={k:v for k,v in heads_from_prior().items() if k in HEADS};h={k:h[k] for k in HEADS};folder=R/'seed301';folder.mkdir(exist_ok=False)
 with (folder/'heads.pkl').open('wb') as f:pickle.dump(h,f)
 copied=heads();orig=heads_from_prior();model,refs,sigma,inputs=load_core(301);assert frozen_snapshot(model,copied)==frozen_snapshot(model,{k:orig[k] for k in HEADS})
 save(R/'head_copy_audit.json',dict(passed=True,heads=HEADS,no_new_fits=True))
 files=dict(m['files']);files.update(inputs)
 for p in [PRIOR/'round_summary.json',PRIOR/'evidence_manifest.json',R/'config.json',R/'protocol.md',R/'planning_diagnostics.json',R/'execution_code_manifest.json',R/'head_copy_audit.json',folder/'heads.pkl']:files[str(p)]=sha(p)
 files.update(json.loads((R/'execution_code_manifest.json').read_text())['files']);save(R/'training_lock.json',dict(files=files,locked_at=time.time(),no_new_training=True,K=96,selected_heads=HEADS))
def heads_from_prior():
 with (PRIOR/'seed301/heads.pkl').open('rb') as f:return pickle.load(f)
if __name__=='__main__':lock()
