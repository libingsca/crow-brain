from pathlib import Path
import json,re
r=Path('artifacts/branch_bi/round5');stages=sorted(r.glob('look*'),key=lambda p:int(p.name[4:]));s=stages[-1]
logs=list(s.glob('log_*'));p=max(logs,key=lambda x:x.stat().st_mtime);lines=p.read_text().splitlines();print(s.name,p.name);print('\n'.join(lines[-3:]))
if (s/'round_summary.json').exists():
 d=json.loads((s/'round_summary.json').read_text());print({k:d[k] for k in ['look','gate','next_action']})

cfg=json.loads((r/'config.json').read_text());j=int(s.name[4:]);start=0 if j==2 else cfg['family_look_sizes'][j-2];end=cfg['family_look_sizes'][j-1]
seen={}
for line in lines:
 m=re.search(r'(evaluate|audit) stage (\d+) part (\d+) bank (\d+)',line)
 if m:seen[int(m[3])]=int(m[4])
if seen:print('completed_at_least',sum(v-(start+k*(end-start)//2)+1 for k,v in seen.items()),'of',end-start)
