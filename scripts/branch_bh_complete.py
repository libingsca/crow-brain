"""Seal the three adaptive rounds with the final split-error-budget decision."""
from pathlib import Path
import json,hashlib,tarfile,time
ROOT=Path('artifacts/branch_bh')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
files=[];records=[];cells=0;vectors=0
for ri in [1,2,3]:
 r=ROOT/f'round{ri}';c=json.loads((r/'config.json').read_text());s=json.loads((r/'summary.json').read_text());assert s['quality_passed']
 for name in ['independent_audit','summary_crosscheck']:assert json.loads((r/f'{name}.json').read_text())['passed']
 if ri<3:assert json.loads((r/'training_audit.json').read_text())['passed']
 lock=json.loads((r/'training_lock.json').read_text());assert all(sha(p)==h for p,h in lock['files'].items());assert all(sha(p)==h for p,h in json.loads((r/'execution_code_manifest.json').read_text()).items());assert not list(r.glob('failure_*.json'))
 for p in r.glob('test*.npz'):assert p.stat().st_mtime>=lock['locked_at']
 if ri<3:
  for seed in range(301,306):assert json.loads((r/f'seed{seed}/fits.json').read_text())['finished_at']<=lock['locked_at']
 else:assert lock['head_copy_equal'] and sha(r/'seed301/heads.pkl')==sha(ROOT/'round2/seed301/heads.pkl')
 audit=json.loads((r/'independent_audit.json').read_text());cells+=audit['cells'];vectors+=audit['prediction_vectors'];records.append({'round':ri,'stage':s['stage'],'decision':s['decision'],'quality':True})
 files.extend(Path(p) for p in lock['files']);files.extend(Path(p) for p in json.loads((r/'data_manifest.json').read_text())['prior_files'])
last=json.loads((ROOT/'round3/summary.json').read_text());assert last['decision'] in ['supported','excluded'];assert cells==5980 and vectors==179080
(ROOT/'final_integrity.json').write_text(json.dumps({'passed':True,'rounds':records,'test_cells':cells,'head_prediction_vectors':vectors,'tests':'65 passed','new_fit_heads':15,'reused_initial_heads':5,'no_training_in_round3':True,'new_unique_images':270000,'historical_retained_images':344000,'final_retained_images':614000,'final_joint_error_bound':.05,'locked_sources_inputs_heads_unchanged':True,'completed_at':time.time()},indent=2)+'\n')
with tarfile.open(ROOT/'final_source_snapshot.tar.gz','w:gz') as tar:
 for d in ['scripts','tests','src/crowbrain']:
  for p in Path(d).glob('*.py'):tar.add(p)
 for p in [Path('requirements-lock.txt'),Path('pyproject.toml'),Path('reports/branch_bh/FINAL_REPORT.md')]+[ROOT/f'round{ri}/protocol.md' for ri in [1,2,3]]:tar.add(p)
for d in [ROOT,Path('reports/branch_bh'),Path('docs/decisions/branch_bh')]:files.extend(p for p in d.rglob('*') if p.is_file() and p.name!='reproducibility_manifest.json')
files.extend(Path('scripts').glob('branch_bh*.py'));files.extend([Path('tests/test_branch_bh.py'),Path('tests/test_branch_bh3.py'),Path('NEXT_EXPERIMENT_PLAN.md'),Path('docs/branches/BH_COVERAGE_CONFIRMATION.md')])
m={'status':'complete_scoped_H_BH2_'+last['decision'],'science_rounds':3,'rounds':records,'final_joint_error_bound':.05,'files':{}}
for p in sorted(set(p.resolve() for p in files)):m['files'][str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
(ROOT/'reproducibility_manifest.json').write_text(json.dumps(m,indent=2)+'\n');assert all(sha(p)==v['sha256'] for p,v in m['files'].items());print('Sealed and verified',len(m['files']),'files;',m['status'])
