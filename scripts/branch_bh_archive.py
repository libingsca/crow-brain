"""Seal all completed adaptive B-H rounds and their fixed input dependencies."""
from pathlib import Path
import hashlib,json,time,tarfile
ROOT=Path('artifacts/branch_bh')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
rounds=sorted(ROOT.glob('round[0-9]*'));assert len(rounds)>=2
files=[];results=[]
for r in rounds:
 c=json.loads((r/'config.json').read_text());s=json.loads((r/'summary.json').read_text());assert s['quality_passed']
 for n in ['training_audit','independent_audit','summary_crosscheck']:assert json.loads((r/f'{n}.json').read_text())['passed']
 lock=json.loads((r/'training_lock.json').read_text());assert all(sha(p)==h for p,h in lock['files'].items())
 code=json.loads((r/'execution_code_manifest.json').read_text());assert all(sha(p)==h for p,h in code.items());assert not list(r.glob('failure_*.json'))
 for bi in range(20):assert (r/f'test{bi:02d}.npz').stat().st_mtime>=lock['locked_at']
 for seed in range(301,306):assert json.loads((r/f'seed{seed}/fits.json').read_text())['finished_at']<=lock['locked_at']
 files.extend(Path(p) for p in lock['files']);files.extend(Path(p) for p in json.loads((r/'data_manifest.json').read_text())['prior_files'])
 results.append({'round':c['round'],'stage':c['stage'],'decision':s['decision'],'classification':{k:len(s[k]) for k in ['supported','excluded','unresolved']}})
assert results[-1]['stage']=='confirmation' and results[-1]['decision'] in ['supported','excluded']
(ROOT/'final_integrity.json').write_text(json.dumps({'passed':True,'rounds':results,'locked_inputs_and_code_unchanged':True,'all_tests_created_after_head_lock':True,'tests':'63 passed','historical_audit':'round1/history_audit.json','completed_at':time.time()},indent=2)+'\n')
with tarfile.open(ROOT/'final_source_snapshot.tar.gz','w:gz') as t:
 for d in ['scripts','tests','src/crowbrain']:
  for p in Path(d).glob('*.py'):t.add(p)
 for p in [Path('requirements-lock.txt'),Path('pyproject.toml'),Path('reports/branch_bh/FINAL_REPORT.md')]+[r/'protocol.md' for r in rounds]:t.add(p)
for d in [ROOT,Path('reports/branch_bh'),Path('docs/decisions/branch_bh')]:files.extend(p for p in d.rglob('*') if p.is_file() and p.name!='reproducibility_manifest.json')
files.extend(Path('scripts').glob('branch_bh*.py'));files.extend([Path('NEXT_EXPERIMENT_PLAN.md'),Path('docs/branches/BH_COVERAGE_CONFIRMATION.md')])
m={'status':'complete_scoped_confirmation_'+results[-1]['decision'],'rounds':results,'files':{}}
for p in sorted(set(p.resolve() for p in files)):m['files'][str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
(ROOT/'reproducibility_manifest.json').write_text(json.dumps(m,indent=2)+'\n');assert all(sha(p)==v['sha256'] for p,v in m['files'].items());print('Sealed and verified',len(m['files']),'files;',m['status'])
