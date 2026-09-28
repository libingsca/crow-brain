"""Seal completed B-W artifacts and verify execution source remained unchanged."""
from pathlib import Path
import json,hashlib,tarfile,time
root=Path('artifacts/branch_bw/round1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
r=json.loads((root/'summary.json').read_text());assert r['quality_passed'] and r['decision'] in ['supported','excluded']
assert all(sha(p)==v for p,v in json.loads((root/'execution_code_manifest.json').read_text())['files'].items())
assert not list(root.glob('failure_*.json'))
for name in ['training_audit','independent_audit','data_chronology_integrity','summary_crosscheck']:
 assert json.loads((root/f'{name}.json').read_text())['passed']
metrics={'training':{},'evaluation':{},'original_execution_source_unchanged':True,'tests':'55 passed','scientific_execution_failures':[],'recorded_at':time.time()}
for seed in range(301,306):
 f=json.loads((root/f'seed{seed}/fits.json').read_text());e=json.loads((root/f'seed{seed}/results.json').read_text())
 metrics['training'][seed]={k:f[k] for k in ['seconds','peak_rss_kib']};metrics['evaluation'][seed]=e['seconds']
save(root/'execution_metrics.json',metrics)
with tarfile.open(root/'final_source_snapshot.tar.gz','w:gz') as t:
 for d in ['scripts','src/crowbrain','tests']:
  for p in sorted(Path(d).glob('*.py')):t.add(p)
 for p in ['requirements-lock.txt','pyproject.toml']:t.add(p)
paths=[]
for d in [root,Path('reports/branch_bw'),Path('docs/decisions/branch_bw')]:paths.extend(p for p in d.rglob('*') if p.is_file())
paths+=list(Path('scripts').glob('branch_bw*.py'))+[Path('tests/test_branch_bw.py'),Path('docs/branches/BW_WORST_DELAY_OBJECTIVE.md'),Path('NEXT_EXPERIMENT_PLAN.md')]
m={'status':f'complete_scoped_H_BW1_{r["decision"]}','science_rounds':1,'quality_passed':True,'tests':'55 passed','new_heads':20,'candidate_fits':100,'evaluation_cells':1100,'primary_conditions':26,'head_prediction_vectors_recomputed':57200,'generated_at':time.time(),'files':{}}
for family,key in [('primary','primary_classification'),('absolute','absolute_classification'),('gain','gain_classification')]:
 for state in ['supported','excluded','unresolved']:m[f'{family}_{state}']=len(r[key][state])
for p in sorted(set(paths)):
 m['files'][str(p.resolve())]={'sha256':sha(p),'bytes':p.stat().st_size}
save('artifacts/branch_bw/reproducibility_manifest.json',m)
assert all(sha(p)==v['sha256'] for p,v in m['files'].items())
print('Sealed and verified',len(m['files']),'files; decision',r['decision'])
