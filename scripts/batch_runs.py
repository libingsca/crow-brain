import argparse, concurrent.futures, json, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--kind',required=True);p.add_argument('--stage',choices=['dev','final'],required=True);p.add_argument('--lr',type=float);p.add_argument('--test-base',type=int,default=80000)
a=p.parse_args();c=json.loads(Path(a.config).read_text());root=Path('artifacts')/c['round']/a.stage;root.mkdir(parents=True,exist_ok=True)
lrs=c['learning_rates'] if a.stage=='dev' else [a.lr]
seeds=c['dev_seeds'] if a.stage=='dev' else c['final_seeds']
def run(item):
 lr,seed=item;name=f'{a.kind}_lr{lr}_seed{seed}';out=root/name
 if (out/'result.json').exists(): return {'skipped_complete':name}
 cmd=[sys.executable,'-m','crowbrain.run','--config',a.config,'--kind',a.kind,'--lr',str(lr),'--seed',str(seed),'--steps',str(c[a.stage+'_steps']),'--output',str(out),'--test-base',str(a.test_base)]
 if a.stage=='final':cmd+=['--final']
 with (root/f'{name}.log').open('w') as log:
  code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
 return {'name':name,'returncode':code}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for item in pool.map(run,[(lr,s) for lr in lrs for s in seeds]): print(json.dumps(item),flush=True)
