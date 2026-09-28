import subprocess,os,sys,json,time,hashlib,shutil
from pathlib import Path
R=Path('artifacts/branch_bi/round2');env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(Path('src').resolve()))
def run(script,args):
 p=R/f"log_{script}_{'_'.join(args)}.txt"
 with p.open('w') as f:subprocess.run([str(Path('.venv/bin/python').absolute()),'scripts/'+script,*args],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('done',script,args,flush=True)
def main():
 if not (R/'training_lock.json').exists():run('branch_bi2_round.py',['lock'])
 if not (R/'data_manifest.json').exists():run('branch_bi2_round.py',['prepare'])
 if not (R/'seed301/results.json').exists():run('branch_bi2_round.py',['evaluate'])
 if not (R/'independent_audit.json').exists():run('branch_bi2_audit.py',[])
 run('branch_bi2_analyze.py',[])
if __name__=='__main__':main()
