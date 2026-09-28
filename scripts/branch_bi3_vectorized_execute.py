"""Continue the frozen third round after numerical-equivalence optimization."""
import subprocess,os
from pathlib import Path
R=Path('artifacts/branch_bi/round3');env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(Path('src').resolve()))
def run(script,args):
 with (R/f"log_{script}_{'_'.join(args)}.txt").open('w') as f:subprocess.run([str(Path('.venv/bin/python').absolute()),'scripts/'+script,*args],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('done',script,args,flush=True)
if __name__=='__main__':
 run('branch_bi3_vectorized.py',['evaluate']);run('branch_bi3_vectorized_audit.py',[]);run('branch_bi3_analyze.py',[])
