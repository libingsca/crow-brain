from pathlib import Path
import hashlib,json,time
root=Path('artifacts/branch_bi');cache={};records={};exceptions=[]
def sha(p):
 p=str(Path(p).resolve())
 if p not in cache:
  h=hashlib.sha256()
  with open(p,'rb') as f:
   for b in iter(lambda:f.read(2**20),b''):h.update(b)
  cache[p]=h.hexdigest()
 return cache[p]
for branch in ['b','bn','bt','br','bs','bl','bc','bw','bz','bf','bg','bh']:
 p=Path(f'artifacts/branch_{branch}/reproducibility_manifest.json');m=json.loads(p.read_text())
 for name,v in m['files'].items():
  expected=v['sha256'] if isinstance(v,dict) else v
  if Path(name).name=='NEXT_EXPERIMENT_PLAN.md':exceptions.append({'branch':branch,'expected':expected,'current':sha(name)});continue
  assert sha(name)==expected,(branch,name)
 records[branch]={'entries':len(m['files']),'sha256':sha(p)}
 print('verified history',branch,flush=True)
(root/'history_audit.json').write_text(json.dumps({'passed':True,'records':records,'mutable_pointer_exceptions':exceptions,'checked_at':time.time()},indent=2)+'\n')
