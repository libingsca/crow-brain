"""Resume-safe stage scheduler; never starts tests before global freeze."""
import os,sys,json,time,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from branch_b_round1 import sha,save
R=Path('artifacts/branch_bi/round1');PY=str(Path('.venv/bin/python').absolute());env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(Path('src').resolve()))
def run(script,args,b=None):
 e=dict(env)
 if b:e['BI_BATCH']=str(b)
 log=R/f"log_{Path(script).stem}_{'_'.join(map(str,args))}_b{b}.txt"
 with log.open('w') as f:subprocess.run([PY,'scripts/'+script,*map(str,args)],env=e,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('done',script,args,b,flush=True)
def pool(tasks):
 with ThreadPoolExecutor(max_workers=1 if any(x[1] and x[1][0]=='train' for x in tasks) else 2) as ex:
  futures=[ex.submit(run,*x) for x in tasks]
  for f in futures:f.result()
def main():
 assert json.loads((R/'history_audit.json').read_text())['passed']
 assert all(sha(p)==h for p,h in json.loads((R/'execution_code_manifest.json').read_text())['files'].items())
 if not (R/'all_fit_data.json').exists():run('branch_bi_data.py',['fit'])
 if not (R/'batch1/seed301/fits.json').exists():run('branch_bi_round.py',['train','--seed',301],1)
 peak=json.loads((R/'batch1/seed301/fits.json').read_text())['peak_rss_kib'];assert peak<1400000;save(R/'resource_check.json',dict(first_fit_peak_rss_kib=peak,max_processes=2))
 tasks=[('branch_bi_round.py',['train','--seed',s],b) for b in range(1,7) for s in range(301,306) if not (R/f'batch{b}/seed{s}/fits.json').exists()];pool(tasks)
 pool([('branch_bi_round.py',['audit_train'],b) for b in range(1,7) if not (R/f'batch{b}/training_lock.json').exists()])
 if not (R/'global_training_lock.json').exists():
  assert not list(R.glob('batch*/test*.npz'));files={}
  for b in range(1,7):
   p=R/f'batch{b}/training_lock.json';m=json.loads(p.read_text());assert all(sha(p)==h for p,h in m['files'].items());files.update(m['files']);files[str(p)]=sha(p)
  files[str(R/'execution_code_manifest.json')]=sha(R/'execution_code_manifest.json');save(R/'global_training_lock.json',dict(files=files,locked_at=time.time(),heads=30,test_absent=True))
 if not (R/'all_test_data.json').exists():run('branch_bi_data.py',['test'])
 pool([('branch_bi_round.py',['evaluate','--seed',s],b) for b in range(1,7) for s in range(301,306) if not (R/f'batch{b}/seed{s}/results.json').exists()])
 pool([('branch_bi_audit.py',[],b) for b in range(1,7) if not (R/f'batch{b}/independent_audit.json').exists()])
 for b in range(1,7):run('branch_bi_analyze.py',[],b)
 run('branch_bi_aggregate.py',[])
if __name__=='__main__':main()
