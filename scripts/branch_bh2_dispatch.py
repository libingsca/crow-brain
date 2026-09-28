import argparse,subprocess,os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
round_id=os.environ.get('BH_ROUND','2')
p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','evaluate']);p.add_argument('--seeds',nargs='+',type=int,required=True);a=p.parse_args()
def run(seed):
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 with Path(f'artifacts/branch_bh/round{round_id}/{a.phase}{seed}.log').open('w') as f:subprocess.run(['.venv/bin/python','scripts/branch_bh2_round.py',a.phase,'--seed',str(seed)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print(a.phase,'completed',seed,flush=True)
with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,a.seeds))
