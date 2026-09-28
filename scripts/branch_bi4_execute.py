"""Each precision look follows the preceding complete, audited result."""
import sys,os,json,subprocess,time
from branch_bi4_common import *
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(Path('src').resolve())+':'+str(Path('scripts').resolve()))
def run(script,args,j):
 with (stage(j)/f"log_{script}_{'_'.join(args)}.txt").open('w') as f:subprocess.run([str(Path('.venv/bin/python').absolute()),'scripts/'+script,*args],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('completed',j,script,args,flush=True)
def main():
 verify_lock()
 for j in range(1,len(CFG['look_sizes'])+1):
  prior=PRIOR/'round_summary.json' if j==1 else stage(j-1)/'round_summary.json';s=json.loads(prior.read_text())
  if j>1 and s['next_action']!='continue_precision':break
  root=stage(j);root.mkdir(exist_ok=False);save(root/'decision.json',dict(previous_result=str(prior),sha256=sha(prior),reason='R3 unresolved measurement' if j==1 else 'Complete prior look inconclusive; no diagnostic contradiction of selected targets; continue registered precision schedule',previous_gate=s['gate'],cumulative_banks=CFG['look_sizes'][j-1],measurement_alpha=CFG['measurement_alpha']*CFG['look_alpha_fractions'][j-1],decided_at=time.time()))
  for script,args in [('branch_bi4_data.py',['prepare',str(j)]),('branch_bi4_data.py',['audit',str(j)]),('branch_bi4_evaluate.py',[str(j)]),('branch_bi4_audit.py',[str(j)]),('branch_bi4_analyze.py',[str(j)])]:run(script,args,j)
  result=json.loads((root/'round_summary.json').read_text());print(json.dumps(dict(look=j,gate=result['gate'],next_action=result['next_action'])),flush=True);save(root/'evidence_manifest.json',dict(files={str(p):sha(p) for p in sorted(root.rglob('*')) if p.is_file()},reason='Complete precision look before any further data'))
  if result['next_action']!='continue_precision':break
if __name__=='__main__':main()
