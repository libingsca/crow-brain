"""Close the fixed-sample replication without turning unresolved results into success."""
from pathlib import Path
import json,time,hashlib,tarfile
ROOT=Path('artifacts/branch_bz');s=json.loads((ROOT/'summary.json').read_text());assert s['quality_passed']
for p in ['integrity.json','crosscheck.json']:assert json.loads((ROOT/p).read_text())['passed']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
initial=json.loads((ROOT/'execution_manifest.json').read_text());assert all(sha(p)==v for p,v in initial['code'].items())
supp=json.loads((ROOT/'supplementary_code_manifest.json').read_text());assert all(sha(p)==v for p,v in supp['files'].items())
fail=list(ROOT.glob('repeat*/failure_*.json'));assert not fail
state={'supported':'支持','excluded':'排除','inconclusive':'未决'}
panel=s['panel_classification'];individual=s['individual_classification']
msg=f'平均功能假设：{state[panel["decision"]]}；个体最差表现假设：{state[individual["decision"]]}。'
stop=Path('docs/decisions/branch_bz/BZ001_FIXED_SAMPLE_STOP.md')
stop.write_text('# BZ001：完成预定5批，按原样保留未决\n\n'+msg+'\n\n平均功能：'+json.dumps(panel,ensure_ascii=False)+'\n\n个体最差表现：'+json.dumps(individual,ensure_ascii=False)+'\n\n所有质量审计通过。停止依据为执行前固定5个独立批次；没有因中途结果增减样本。未决结果不是支持或排除，后续研究需另立协议。\n\n[完整报告](../../../reports/branch_bz/FINAL_REPORT.md)。\n')
Path('NEXT_EXPERIMENT_PLAN.md').write_text('# 当前实验 B-Z 已完成\n\n'+msg+'\n\n完成5个独立拟合／验证／测试批次、50个新读出头、250候选和1,100个测试单元。58项测试与全部质量审计通过。主要判据按27项联合校正；未决不强行裁决。\n\n[最终报告](reports/branch_bz/FINAL_REPORT.md) · [执行前协议](artifacts/branch_bz/protocol.md) · [停止决定](docs/decisions/branch_bz/BZ001_FIXED_SAMPLE_STOP.md)。\n\n后续应根据完整结果另立协议，不在本轮追加样本或改温度来改变结论。\n')
p=Path('docs/branches/BZ_REFIT_REPLICATION.md');text=p.read_text();text=text.replace('2026-09-25，执行前固定协议。尚未生成B-Z数据或观察结果。','2026-09-25，本轮已完成。'+msg+'\n\n[最终报告](../../reports/branch_bz/FINAL_REPORT.md)。以下保留原设计；执行前原文另存artifacts/branch_bz/protocol.md。');p.write_text(text)
metrics={'recorded_at':time.time(),'tests':'58 passed','scientific_failures':[],'resolved_implementation_issues':['Historical image audit filters current-branch paths before historical count','Audit launcher preserved virtualenv symlink entrypoint after initial missing-numpy failure'],'execution_code_unchanged':True,'training':{},'evaluation':{},'candidate_count':0,'evaluation_cells':0,'prediction_vectors':0}
for r in range(1,6):
 folder=ROOT/f'repeat{r}'
 metrics['candidate_count']+=json.loads((folder/'training_audit.json').read_text())['candidate_count']
 ev=json.loads((folder/'independent_audit.json').read_text());metrics['evaluation_cells']+=ev['complete_cells'];metrics['prediction_vectors']+=ev['head_prediction_vectors_recomputed']
 for seed in range(301,306):
  f=json.loads((folder/f'seed{seed}/fits.json').read_text());e=json.loads((folder/f'seed{seed}/results.json').read_text())
  metrics['training'][f'{r}/{seed}']={k:f[k] for k in ['seconds','peak_rss_kib']};metrics['evaluation'][f'{r}/{seed}']=e['seconds']
assert [metrics[k] for k in ['candidate_count','evaluation_cells','prediction_vectors']]==[250,1100,33000]
(ROOT/'execution_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
(ROOT/'completion_metadata.json').write_text(json.dumps({'completed_at':time.time(),'planning_pointer_sha256':sha('NEXT_EXPERIMENT_PLAN.md'),'integrity_scope':'Scientific integrity checked before completion document update; mutable planning pointer updated at completion. Archived original pointer preserved.'},indent=2)+'\n')
with tarfile.open(ROOT/'final_source_snapshot.tar.gz','w:gz') as t:
 for d in ['scripts','src/crowbrain','tests']:
  for p in sorted(Path(d).glob('*.py')):t.add(p)
 for p in ['requirements-lock.txt','pyproject.toml']:t.add(p)
paths=[]
for folder in [ROOT,Path('reports/branch_bz'),Path('docs/decisions/branch_bz')]:paths += [p for p in folder.rglob('*') if p.is_file() and p.name!='reproducibility_manifest.json']
paths+=list(Path('scripts').glob('branch_bz*.py'))+[Path('tests/test_branch_bz.py'),Path('NEXT_EXPERIMENT_PLAN.md'),Path('docs/branches/BZ_REFIT_REPLICATION.md')]
# All logs must be closed before calling this finalizer; its output goes to the terminal.
manifest=dict(status='complete_fixed_sample',quality_passed=True,panel=panel,individual=individual,created_at=time.time(),files={str(p.resolve()):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(set(paths))})
(ROOT/'reproducibility_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
assert all(sha(p)==v['sha256'] for p,v in manifest['files'].items())
print(msg,'Sealed',len(manifest['files']),'files.')
