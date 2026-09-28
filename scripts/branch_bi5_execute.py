"""One complete audited look, then return for result-dependent planning."""
import sys,os,json,subprocess,time
from branch_bi5_common import *
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(Path('src').resolve())+':'+str(Path('scripts').resolve()))
def run(script,args,j):
 with (stage(j)/f"log_{script}_{'_'.join(args)}.txt").open('w') as f:subprocess.run([str(Path('.venv/bin/python').absolute()),'scripts/'+script,*args],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('completed',j,script,args,flush=True)
def main(j):
 verify_lock();prior=P4/'round_summary.json' if j==2 else stage(j-1)/'round_summary.json';s=json.loads(prior.read_text());assert s['quality_passed'] and s['aggregation_crosschecked'] and s['decision']=='inconclusive';root=stage(j);root.mkdir(exist_ok=False);save(root/'decision.json',dict(previous_result=str(prior),sha256=sha(prior),previous_gate=s['gate'],cumulative_banks=bounds(j)[1],reason='Expand original complete uncertainty family with simultaneous correction' if j==2 else 'Previous complete simultaneous synthesis unresolved; implement separately reviewed next precision point',decided_at=time.time(),new_fits=0))
 for script,args in [('branch_bi5_data.py',['prepare',str(j)]),('branch_bi5_data.py',['audit',str(j)]),('branch_bi5_evaluate.py',[str(j)]),('branch_bi5_audit.py',[str(j)]),('branch_bi5_analyze.py',[str(j)])]:run(script,args,j)
 save(root/'evidence_manifest.json',dict(files={str(p):sha(p) for p in sorted(root.rglob('*')) if p.is_file()},reason='Full look completed before further result-dependent planning'));print((root/'round_summary.json').read_text()[:500],flush=True)
if __name__=='__main__':main(int(sys.argv[1]))
