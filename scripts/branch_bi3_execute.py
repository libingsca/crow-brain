import os,subprocess,json
from pathlib import Path
from branch_b_round1 import sha
R=Path('artifacts/branch_bi/round3');env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(Path('src').resolve()))
def run(script,args):
 with (R/f"log_{script}_{'_'.join(args)}.txt").open('w') as f:subprocess.run([str(Path('.venv/bin/python').absolute()),'scripts/'+script,*args],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('done',script,args,flush=True)
def main():
 assert all(sha(p)==h for p,h in json.loads((R/'execution_code_manifest.json').read_text())['files'].items())
 if not (R/'fit_data_manifest.json').exists():run('branch_bi3_data.py',['fit'])
 if not (R/'fit_data_audit.json').exists():run('branch_bi3_data.py',['audit_fit'])
 run('branch_bi3_fit.py',['train']);run('branch_bi3_fit.py',['audit'])
 if not (R/'training_lock.json').exists():run('branch_bi3_round.py',['lock'])
 if not (R/'data_manifest.json').exists():run('branch_bi3_data.py',['test'])
 if not (R/'seed301/results.json.gz').exists():run('branch_bi3_round.py',['evaluate'])
 if not (R/'independent_audit.json').exists():run('branch_bi3_audit.py',[])
 run('branch_bi3_analyze.py',[])
if __name__=='__main__':main()
