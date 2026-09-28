"""Fixed pre-round4 uncertainty family; immutable heads and bank-prefix mapping."""
import json,pickle,time
from pathlib import Path
from branch_b_round1 import sha,save
from branch_bf_round1 import load_core
from branch_bt_round1 import frozen_snapshot
R=Path('artifacts/branch_bi/round5');P3=Path('artifacts/branch_bi/round3');P4=Path('artifacts/branch_bi/round4');CFG=json.loads((R/'config.json').read_text());HEADS=sorted({x['head'] for x in CFG['targets']},key=lambda x:int(x[1:]));DELAYS=CFG['delays'];REFS=CFG['reference_delays'];BY_DELAY={d:[h for h in HEADS if any(x['head']==h and x['delay']==d for x in CFG['targets'])] for d in DELAYS};CANDIDATE_PAIRS=sum(len(v) for v in BY_DELAY.values())
def stage(j):return R/f'look{j}'
def bounds(j):
 assert j in [2,3,4];return (0 if j==2 else CFG['family_look_sizes'][j-2]),CFG['family_look_sizes'][j-1]
def indices(j,part):
 a,b=bounds(j);return range(a+part*(b-a)//2,a+(part+1)*(b-a)//2)
def bank_path(i):
 if i<1024:return P4/'stage1'/f'test{i:04d}.npz'
 if i<2048:return P4/'stage2'/f'test{i:04d}.npz'
 return stage(3 if i<4096 else 4)/f'test{i:04d}.npz'
def permutation_path(i):return bank_path(i).with_name(f'permutations{i:04d}.npz')
def verify_lock():
 m=json.loads((R/'training_lock.json').read_text());assert all(sha(p)==h for p,h in m['files'].items());return m
def heads():
 with (R/'seed301/heads.pkl').open('rb') as f:return pickle.load(f)
def active(h,refs,d):
 out={key:h[key] for key in BY_DELAY[d]}
 if d in REFS:out['reference']=refs[f'ref{d}']
 return out
def base_prior():
 m=json.loads((P4/'data_manifest.json').read_text());return {**m['prior_files'],**{x['path']:x['sha256'] for x in m['banks']}}
def previous(j):
 out=base_prior()
 for k in range(3,j):out.update({x['path']:x['sha256'] for x in json.loads((stage(k)/'data_manifest.json').read_text())['banks']})
 return out
def lock():
 for root in [P3,P4]:
  evidence=json.loads((root/'evidence_manifest.json').read_text());assert all(sha(p)==h for p,h in evidence['files'].items())
 m=json.loads((P4/'training_lock.json').read_text());assert all(sha(p)==h for p,h in m['files'].items());with_source=P3/'seed301/heads.pkl'
 with with_source.open('rb') as f:original=pickle.load(f)
 folder=R/'seed301';folder.mkdir(exist_ok=False)
 with (folder/'heads.pkl').open('wb') as f:pickle.dump({k:original[k] for k in HEADS},f)
 model,refs,sigma,inputs=load_core(301);assert frozen_snapshot(model,heads())==frozen_snapshot(model,{k:original[k] for k in HEADS});save(R/'head_copy_audit.json',dict(passed=True,heads=HEADS,new_fits=0,original_heads_frozen_before_round4_data=True))
 files=dict(m['files']);files.update(inputs);files.update(json.loads((R/'execution_code_manifest.json').read_text())['files'])
 for p in [P3/'round_summary.json',P3/'evidence_manifest.json',P4/'round_summary.json',P4/'evidence_manifest.json',P4/'ROUND_DISPOSITION.md',P4/'first_certification_ledger.json',P4/'data_manifest.json',P4/'stage1/independent_data_audit.json',P4/'stage2/independent_data_audit.json',R/'config.json',R/'protocol.md',R/'execution_code_manifest.json',R/'head_copy_audit.json',folder/'heads.pkl']:files[str(p)]=sha(p)
 save(R/'training_lock.json',dict(files=files,locked_at=time.time(),K=96,new_fits=0,data_already_generated_and_partly_viewed=True,simultaneous_family_size=68,simultaneous_look_sizes=CFG['family_look_sizes']))
if __name__=='__main__':lock()
