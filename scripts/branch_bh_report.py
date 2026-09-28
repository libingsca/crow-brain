import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from branch_bh_round import ROOT,CFG,HEADS,DELAYS,REFERENCE_DELAYS
from branch_b_round1 import save
r=json.loads((ROOT/'summary.json').read_text());s=np.load(ROOT/'analysis_arrays.npz')['scores'];means=s.mean(1);out=Path('reports/branch_bh')/f'round{CFG["round"]}';out.mkdir(exist_ok=True);hi=HEADS.index(CFG['candidate']);ncheck=0
# Independent aggregation of every main interval directly from bank records.
for seed in range(301,306):
 rows=json.loads((ROOT/f'seed{seed}/results.json').read_text())['cells']
 for d in DELAYS:
  a=np.array([x['scores'] for x in sorted([x for x in rows if x['delay']==d],key=lambda x:x['bank'])]);assert a.shape[0]==20
  for ci,c in enumerate(['clean','noise']):
   v=a[:,hi,0] if ci==0 else a[:,hi,1:].sum(1)/10;mu=sum(v)/20;half=t.ppf(1-.05/220,19)*np.sqrt(sum((v-mu)**2)/19)/np.sqrt(20);expected=r['primary'][f'{seed}_d{d}_{c}'];np.testing.assert_allclose([mu,mu-half,mu+half],[expected[k] for k in ['mean','lower','upper']],atol=1e-12,rtol=0);ncheck+=1
assert ncheck==110
save(ROOT/'summary_crosscheck.json',{'passed':True,'intervals_independently_reaggregated':110})
lines=[f'# B-H 第{CFG["round"]}轮完整结果','','阶段：'+CFG['stage']+'。110项同一家族校正区间，20库、df19；单位BA%。','','| 模型 | 时长 | 条件 | '+' | '.join(HEADS)+' | 候选区间 | 判定 |','|---|---|---|'+ '---:|'*len(HEADS)+'---|---|']
for i,seed in enumerate(range(301,306)):
 for d in DELAYS:
  for ci,c in enumerate(['clean','noise']):
   k=f'{seed}_d{d}_{c}';v=r['primary'][k];status='高于' if k in r['supported'] else '低于' if k in r['excluded'] else '未决';lines.append(f'| {seed} | {d} | {c} | '+' | '.join(f'{x*100:.3f}' for x in means[i,d-5,:,ci])+f' | [{v["lower"]*100:.3f}, {v["upper"]*100:.3f}] | {status} |')
lines+=['','## 正对照','','| 条件 | 均值 | 区间 |','|---|---:|---|']
for k,v in r['positive_controls'].items():lines.append(f'| {k} | {v["mean"]*100:.3f} | [{v["lower"]*100:.3f}, {v["upper"]*100:.3f}] |')
lines+=['','## 负对照','','| 时长 | 方法 | 条件 | 清零 | 标签置换 | test数量内状态错配 |','|---|---|---|---:|---:|---:|']
for di,d in enumerate(DELAYS):
 for h,name in enumerate(HEADS):
  for ci,c in enumerate(['clean','noise']):lines.append(f'| {d} | {name} | {c} | '+' | '.join(f'{v*100:.3f}' for v in r['negative_means'][di][h][ci])+' |')
for di,d in enumerate(REFERENCE_DELAYS):
 for ci,c in enumerate(['clean','noise']):lines.append(f'| {d} | reference | {c} | '+' | '.join(f'{v*100:.3f}' for v in r['reference_negative_means'][di][ci])+' |')
lines+=['','## 优化','','| 模型 | 方法 | 梯度∞ | 验证最低BA% |','|---|---|---:|---:|']
for seed in range(301,306):
 fit=json.loads((ROOT/f'seed{seed}/fits.json').read_text())
 for h,v in fit['heads'].items():lines.append(f'| {seed} | {h} | {v["history"][-1]["gradient_inf"]:.3g} | {min(x["ba"] for x in v["validation"].values())*100:.3f} |')
(out/'FULL_RESULTS.md').write_text('\n'.join(lines)+'\n');print('Full results and independent 110-interval crosscheck complete.')
