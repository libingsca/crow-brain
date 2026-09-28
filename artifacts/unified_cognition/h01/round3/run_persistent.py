"""Fixed jobs, isolated persistent processes. Seed reset remains inside h01.run."""
import sys,json,concurrent.futures,argparse,traceback,os
from pathlib import Path

def job(j,cfg):
 import h01_product,h01
 args=argparse.Namespace(round=cfg['round'],model=j['model'],seed=j['seed'],lr=j['lr'],steps=cfg['steps'],eval_n=cfg['eval_n'])
 try:
  h01.run(args)
  p=h01.BASE/args.round/f'{args.model}_s{args.seed}_lr{args.lr:g}'/'result.json'
  r=json.loads(p.read_text());r['head']='symmetric_product';r['product_source_sha256']=h01.digest(h01_product.__file__);h01.dump(p,r)
  return str(p)
 except Exception:
  failure={'job':j,'traceback':traceback.format_exc()}
  h01.dump(h01.BASE/cfg['round']/f'failure_{j["model"]}_{j["seed"]}.json',failure)
  raise
if __name__=='__main__':
 cfg=json.loads(Path(sys.argv[1]).read_text())
 with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
  futures=[ex.submit(job,j,cfg) for j in cfg['jobs']]
  for f in futures:f.result()
