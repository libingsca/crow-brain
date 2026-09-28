"""Run remaining fixed stages after all fits exist, at most two worker processes."""
from pathlib import Path
import subprocess,os,concurrent.futures,time,json
ROOT=Path('artifacts/branch_bz');PYTHON=str(Path('.venv/bin/python').absolute())
def run(script,args,r,log):
 env=dict(os.environ,BZ_REPEAT=str(r),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 with (ROOT/f'repeat{r}'/log).open('w') as out:
  p=subprocess.run([PYTHON,f'scripts/{script}',*args],env=env,stdout=out,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError((script,args,r,log,p.returncode))
 print(json.dumps(dict(stage=script,args=args,repeat=r,finished=time.time())),flush=True)
def batch(tasks):
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(run,*x) for x in tasks]
  for f in concurrent.futures.as_completed(futures):f.result()
assert all((ROOT/f'repeat{r}/seed{s}/fits.json').exists() for r in range(1,6) for s in range(301,306))
batch([('branch_bz_audit.py',['training'],r,'training_audit.log') for r in range(1,6)])
for r in range(1,6):run('branch_bz_round1.py',['lock'],r,'lock.log')
assert all((ROOT/f'repeat{r}/training_lock.json').exists() for r in range(1,6))
# Sequential generation is required for cross-batch duplicate checking.
for r in range(1,6):run('branch_bz_round1.py',['prepare_test'],r,'prepare_test.log')
batch([('branch_bz_round1.py',['evaluate','--seed',str(s)],r,f'evaluate{s}.log') for r in range(1,6) for s in range(301,306)])
batch([('branch_bz_audit.py',['evaluation'],r,'evaluation_audit.log') for r in range(1,6)])
for script in ['branch_bz_integrity_final.py','branch_bz_analyze.py','branch_bz_crosscheck.py','branch_bz_report.py']:
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 with (ROOT/(script.removesuffix('.py')+'.log')).open('w') as out:
  subprocess.run([PYTHON,f'scripts/{script}'],env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
 print('completed',script,flush=True)
