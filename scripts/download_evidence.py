import json,hashlib,subprocess
from pathlib import Path
from scipy.io import whosmat
out=Path('data/ditz2022');out.mkdir(exist_ok=True)
manifest=[]
for f in json.loads(Path('data/evidence/mendeley_files.json').read_text()):
 name=f['filename']; size=f['size']
 if size>30_000_000: continue
 p=out/name
 subprocess.run(['curl','-fL','--retry','2','--max-time','60','-sS',f['content_details']['download_url'],'-o',str(p)],check=True)
 sha=hashlib.sha256(p.read_bytes()).hexdigest()
 assert sha==f['content_details']['sha256_hash']
 row={'name':name,'size':size,'sha256':sha,'license':'CC BY 4.0','dataset':'10.17632/7kmtz7kj4c.1'}
 if name.endswith('.mat'): row['variables']=whosmat(p)
 manifest.append(row)
 print(row,flush=True)
Path('data/evidence/download_manifest.json').write_text(json.dumps(manifest,indent=2))
