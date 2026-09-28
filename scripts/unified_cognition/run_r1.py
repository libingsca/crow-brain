import subprocess,sys,concurrent.futures
from pathlib import Path
root=Path(__file__).resolve().parents[2]
jobs=[(model,seed,lr) for seed in range(270101,270106) for lr in (.001,.003) for model in ('F64','S64','FP')]
def run(j):
    model,seed,lr=j
    cmd=[sys.executable,str(root/'scripts/unified_cognition/h01.py'),'run','--model',model,'--seed',str(seed),'--lr',str(lr)]
    r=subprocess.run(cmd,cwd=root,capture_output=True,text=True)
    print(r.stdout,flush=True)
    if r.returncode: print(r.stderr,flush=True);raise RuntimeError(j)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    list(ex.map(run,jobs))
