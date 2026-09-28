"""Execute a registered fixed job list; no result-dependent scheduling or stopping."""
import subprocess,sys,json,concurrent.futures
from pathlib import Path
root=Path(__file__).resolve().parents[2];cfg=json.loads(Path(sys.argv[1]).read_text())
def run(j):
 cmd=[sys.executable,str(root/cfg['implementation']),'--round',cfg['round'],'--model',j['model'],'--seed',str(j['seed']),'--lr',str(j['lr']),'--steps',str(cfg['steps']),'--eval-n',str(cfg['eval_n'])]
 r=subprocess.run(cmd,cwd=root,capture_output=True,text=True);print(r.stdout,flush=True)
 if r.returncode:print(r.stderr,flush=True);raise RuntimeError(j)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(run,cfg['jobs']))
