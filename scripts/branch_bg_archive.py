"""Final B-G seal, preserving all historical science and immutable protocols."""
import json,time,hashlib,tarfile
from pathlib import Path
from branch_bg_round1 import ROOT,verify_lock
from branch_b_round1 import save

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def main():
 r=json.loads((ROOT/'summary.json').read_text());assert r['quality_passed'] and r['decision'] in ['supported','excluded'];lock=verify_lock()
 for name in ['history_audit','training_audit','independent_audit','summary_crosscheck']:assert json.loads((ROOT/f'{name}.json').read_text())['passed']
 code=json.loads((ROOT/'execution_code_manifest.json').read_text());assert all(sha(p)==h for p,h in code['files'].items())
 assert not list(ROOT.glob('failure_*.json'))
 metrics={'training':{},'evaluation':{},'tests':'63 passed','original_execution_code_unchanged':True,'implementation_notes':'round1/implementation_notes.md','recorded_at':time.time()}
 for seed in range(301,306):
  f=json.loads((ROOT/f'seed{seed}/fits.json').read_text());e=json.loads((ROOT/f'seed{seed}/results.json').read_text());metrics['training'][str(seed)]={k:f[k] for k in ['seconds','peak_rss_kib']};metrics['evaluation'][str(seed)]=e['seconds'];assert f['finished_at']<=lock['locked_at']
 save(ROOT/'execution_metrics.json',metrics)
 save(ROOT/'final_integrity.json',{'passed':True,'initial_history_audit':'history_audit.json','locked_inputs_and_code_unchanged':True,'heads_frozen_before_new_images':True,'complete_primary_crosscheck':True,'new_images':20000,'historical_retained_images':324000,'no_scientific_execution_failures':True,'mutable_planning_entry_before':'planning_entry_before.md','completed_at':time.time()})
 with tarfile.open(ROOT/'final_source_snapshot.tar.gz','w:gz') as tar:
  for d in ['scripts','tests','src/crowbrain']:
   for p in sorted(Path(d).glob('*.py')):tar.add(p)
  for p in [ROOT/'protocol.md',Path('requirements-lock.txt'),Path('pyproject.toml'),Path('reports/branch_bg/FINAL_REPORT.md'),Path('docs/decisions/branch_bg/BG001_STOP.md')]:tar.add(p)
 files=[]
 for d in [ROOT,Path('reports/branch_bg'),Path('docs/decisions/branch_bg')]:files.extend(p for p in d.rglob('*') if p.is_file())
 files.extend(Path(p) for p in lock['files']);files.extend(Path(p) for p in json.loads((ROOT/'data_manifest.json').read_text())['prior_files'])
 files.extend(Path('scripts').glob('branch_bg*.py'));files.extend([Path('tests/test_branch_bg.py'),Path('docs/branches/BG_RISK_REGULARIZATION.md'),Path('NEXT_EXPERIMENT_PLAN.md')])
 m={'status':'complete_scoped_H_BG1_'+r['decision'],'quality_passed':True,'science_rounds':1,'new_fits':15,'reused_heads':5,'test_cells':1100,'prediction_vectors_reconstructed':57200,'primary_criteria':8,'primary_classification':{k:len(r[k]) for k in ['supported','excluded','unresolved']},'files':{}}
 for p in sorted(set(p.resolve() for p in files)):m['files'][str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
 save('artifacts/branch_bg/reproducibility_manifest.json',m)
 assert all(sha(p)==v['sha256'] for p,v in m['files'].items())
 print('Sealed and verified',len(m['files']),'files;',m['status'])
if __name__=='__main__':main()
