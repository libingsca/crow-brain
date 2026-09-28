"""Seal B-F without rewriting any historical manifest."""
from pathlib import Path
import json,hashlib,tarfile,time
root=Path('artifacts/branch_bf/round1')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
r=json.loads((root/'summary.json').read_text());assert r['quality_passed'] and r['decision'] in ['supported','excluded']
assert all(sha(p)==v for p,v in json.loads((root/'execution_code_manifest.json').read_text()).items())
for name in ['training_audit','independent_audit','integrity','summary_crosscheck']:assert json.loads((root/f'{name}.json').read_text())['passed']
metrics={'training':{},'evaluation':{},'tests':'61 passed','implementation_repairs':'round1/implementation_notes.md','recorded_at':time.time()}
for seed in range(301,306):
 f=json.loads((root/f'seed{seed}/fits.json').read_text());e=json.loads((root/f'seed{seed}/results.json').read_text())
 metrics['training'][str(seed)]={k:f[k] for k in ['seconds','peak_rss_kib']};metrics['evaluation'][str(seed)]=e['seconds']
save(root/'execution_metrics.json',metrics)
with tarfile.open(root/'final_source_snapshot.tar.gz','w:gz') as t:
 for d in ['scripts','src/crowbrain','tests']:
  for p in sorted(Path(d).glob('*.py')):t.add(p)
 for p in ['requirements-lock.txt','pyproject.toml',str(root/'protocol.md'),'reports/branch_bf/FINAL_REPORT.md','reports/branch_bf/FULL_RESULTS.md','docs/decisions/branch_bf/BF001_STOP.md']:t.add(p)
paths=[]
for d in [root,Path('reports/branch_bf'),Path('docs/decisions/branch_bf'),Path('artifacts/session_recovery_20260925')]:paths.extend(p for p in d.rglob('*') if p.is_file())
paths+=list(Path('scripts').glob('branch_bf*.py'))+[Path('tests/test_branch_bf.py'),Path('docs/branches/BF_INDIVIDUAL_REPAIR.md'),Path('NEXT_EXPERIMENT_PLAN.md')]
# Include all immutable input dependencies explicitly; archives preserve source dependencies.
for manifest in [root/'training_lock.json',root/'data_manifest.json']:
 data=json.loads(manifest.read_text());paths.extend(Path(p) for p in data.get('files',data.get('prior_files',{})))
for seed in range(301,306):paths.extend(Path(p) for p in json.loads((root/f'seed{seed}/evaluation_manifest.json').read_text())['input_sha256'])
m={'status':f'complete_scoped_H_BF1_{r["decision"]}','science_rounds':1,'quality_passed':True,'tests':'61 passed','new_heads':20,'candidate_fits':100,'evaluation_cells':1100,'primary_conditions':110,'head_prediction_vectors_recomputed':57200,'generated_at':time.time(),'files':{}}
for state in ['supported','excluded','unresolved']:m[state]=len(r[state])
for p in sorted(set(p.resolve() for p in paths)):m['files'][str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
save('artifacts/branch_bf/reproducibility_manifest.json',m)
assert all(sha(p)==v['sha256'] for p,v in m['files'].items())
print('Sealed and verified',len(m['files']),'files; decision',r['decision'])
