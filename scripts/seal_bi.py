"""Seal only a scientifically stopped B-I branch; preserve every prior round."""
import json,hashlib,tarfile,time
from pathlib import Path
R=Path('artifacts/branch_bi');cache={}
def sha(p,fresh=False):
 p=Path(p).resolve();key=str(p)
 if not fresh and key in cache:return cache[key]
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(2**20),b''):h.update(x)
 cache[key]=h.hexdigest();return cache[key]
def verify_files(m):
 for p,h in m['files'].items():assert sha(p)==(h['sha256'] if isinstance(h,dict) else h),p

def main():
 stop=json.loads(Path('docs/decisions/branch_bi/STOP.json').read_text());assert stop['criterion_met'];source=R/f"round{stop['source_round']}";s=json.loads((source/'round_summary.json').read_text());assert s['quality_passed'] and s['aggregation_crosschecked']
 if stop['criterion']=='hypothesis_excluded':assert s['decision']=='excluded' and s.get('main_p_upper',s.get('upper',1.))<.8
 elif stop['criterion']=='hypothesis_supported':assert s['decision']=='supported' and s.get('main_p_lower',s.get('lower',0.))>.8
 else:assert stop['criterion']=='low_information_value' and stop['residual_explanations'] and stop['interventions_and_limits']
 assert json.loads((R/'history_audit.json').read_text())['passed'];dependencies=set();rounds=sorted(p for p in R.glob('round*') if p.is_dir())
 for rr in rounds:
  result=json.loads((rr/'round_summary.json').read_text());assert result['quality_passed'] and result['aggregation_crosschecked']
  audits=list(rr.glob('batch*/independent_audit.json')) if rr.name=='round1' else [rr/'independent_audit.json'];assert audits and all(json.loads(p.read_text())['passed'] for p in audits)
  for stage in [*rr.glob('stage*'), *rr.glob('look*')]:
   if stage.is_dir():
    summary=json.loads((stage/'round_summary.json').read_text());assert summary['quality_passed'] and summary['aggregation_crosschecked'];assert json.loads((stage/'independent_audit.json').read_text())['passed']
  for p in list(rr.glob('batch*/training_audit.json'))+list(rr.glob('head_copy_audit.json')):assert json.loads(p.read_text())['passed']
  for p in list(rr.glob('*lock.json'))+list(rr.glob('batch*/*lock.json'))+list(rr.glob('execution_code_manifest.json'))+list(rr.glob('evidence_manifest.json'))+list(rr.glob('stage*/evidence_manifest.json'))+list(rr.glob('look*/evidence_manifest.json')):
   m=json.loads(p.read_text());verify_files(m);dependencies.update(Path(x) for x in m['files'])
  for p in list(rr.glob('data_manifest.json'))+list(rr.glob('batch*/data_manifest.json')):
   m=json.loads(p.read_text());dependencies.update(Path(x) for x in m['prior_files'])
 archive=R/'source_snapshot.tar.gz'
 with tarfile.open(archive,'w:gz') as tar:
  for base in ['scripts','tests','src']:
   for p in sorted(Path(base).rglob('*.py')):tar.add(p,arcname=str(p))
  for p in [Path('docs/branches/BI_REFIT_RELIABILITY.md'),*Path('.').glob('requirements*.txt'),*Path('.').glob('pyproject.toml')]:
   if p.exists():tar.add(p,arcname=str(p))
  for rr in rounds:
   for p in list(rr.glob('protocol*.md'))+list(rr.glob('engineering*.md'))+list(rr.glob('execution_environment.json')):tar.add(p,arcname=str(p))
 files=set(dependencies)
 for base in [R,Path('reports/branch_bi'),Path('docs/decisions/branch_bi')]:files.update(p for p in base.rglob('*') if p.is_file() and p.name not in ['reproducibility_manifest.json','final_integrity.json'])
 for pattern in ['branch_bi*.py','report_bi*.py','seal_bi.py','verify_bi_history.py']:files.update(Path('scripts').glob(pattern))
 files.update(Path('tests').glob('test_branch_bi*.py'));files.add(Path('tests/test_bi_lossless_codec.py'));files.update([Path('NEXT_EXPERIMENT_PLAN.md'),Path('docs/branches/BI_REFIT_RELIABILITY.md')])
 for b in ['b','bn','bt','br','bs','bl','bc','bw','bz','bf','bg','bh']:files.add(Path(f'artifacts/branch_{b}/reproducibility_manifest.json'))
 files={p.resolve() for p in files};m={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(files)};out=R/'reproducibility_manifest.json';out.write_text(json.dumps(dict(files=m,created_at=time.time(),scope='B-I outputs, source, inputs, and prior manifest chain; NEXT_EXPERIMENT_PLAN.md is a mutable pointer'),indent=2)+'\n')
 for p,x in m.items():assert sha(p,fresh=True)==x['sha256'],p
 (R/'final_integrity.json').write_text(json.dumps(dict(passed=True,files=len(m),manifest_sha256=sha(out,fresh=True),verified_at=time.time()),indent=2)+'\n');print('sealed',len(m),'files',flush=True)
if __name__=='__main__':main()
