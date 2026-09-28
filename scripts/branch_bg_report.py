import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from branch_bg_round1 import ROOT,HEADS,DELAYS,REFERENCE_DELAYS
r=json.loads((ROOT/'summary.json').read_text());scores=np.load(ROOT/'analysis_arrays.npz')['scores'].mean(1);out=Path('reports/branch_bg')
lines=['# B-G 完整结果','','BA、增益与区间单位均为百分数／百分点。主8项99.375%区间；次要110项99.954545…%区间，属于单独描述性家族，不与主家族声称联合5%错误覆盖。','','## 主家族','','| 条件与指标 | 均值 | 区间 | 阈值 | 判定 |','|---|---:|---|---:|---|']
for k,v in r['primary'].items():
 status='支持' if k in r['supported'] else '排除' if k in r['excluded'] else '未决';lines.append(f'| {k} | {100*v["mean"]:.3f} | [{100*v["lower"]:.3f}, {100*v["upper"]:.3f}] | {100*v["threshold"]:g} | {status} |')
lines+=['','## 全个体矩阵','','| 模型 | 时长 | 条件 | baseline | weak | sharp | joint | joint描述性区间 | 相对85% |','|---|---|---|---:|---:|---:|---:|---|---|']
for si,seed in enumerate(range(301,306)):
 for di,d in enumerate(DELAYS):
  for ci,c in enumerate(['clean','noise']):
   v=r['secondary_full'][f'{seed}_d{d}_{c}'];status='下界超过' if v['lower']>.85 else '上界低于' if v['upper']<.85 else '未决'
   lines.append(f'| {seed} | {d} | {c} | '+' | '.join(f'{x*100:.3f}' for x in scores[si,di,:,ci])+f' | [{v["lower"]*100:.3f}, {v["upper"]*100:.3f}] | {status} |')
lines+=['','## 预定四点因素效应（配对描述性点估计）','','| 条件 | 正则化/soft | 正则化/sharp | 温度/strong | 温度/weak | 交互差分 |','|---|---:|---:|---:|---:|---:|']
for k,v in r['effects'].items():lines.append('| '+k+' | '+' | '.join(f'{v[key]*100:+.3f}' for key in ['regularization_soft','regularization_sharp','temperature_strong','temperature_weak','interaction'])+' |')
lines+=['','## 正对照','','| 条件 | 均值 | 区间 |','|---|---:|---|']
for k,v in r['positive_controls'].items():lines.append(f'| {k} | {v["mean"]*100:.3f} | [{v["lower"]*100:.3f}, {v["upper"]*100:.3f}] |')
lines+=['','## 所有负对照','','先平均64置换，再平均20库和五模型，清零、标签置换、test数量内状态错配分别保留。','','| 时长 | 方法 | 条件 | 清零 | 标签置换 | 状态错配 |','|---|---|---|---:|---:|---:|']
for di,d in enumerate(DELAYS):
 for hi,h in enumerate(HEADS):
  for ci,c in enumerate(['clean','noise']):lines.append(f'| {d} | {h} | {c} | '+' | '.join(f'{v*100:.3f}' for v in r['negative_means'][di][hi][ci])+' |')
for di,d in enumerate(REFERENCE_DELAYS):
 for ci,c in enumerate(['clean','noise']):lines.append(f'| {d} | reference | {c} | '+' | '.join(f'{v*100:.3f}' for v in r['reference_negative_means'][di][ci])+' |')
lines+=['','## 优化诊断','','| 模型 | 方法 | C | tau | 梯度∞ | 最小验证BA% | 最大验证损失 |','|---|---|---:|---:|---:|---:|---:|']
for seed in range(301,306):
 f=json.loads((ROOT/f'seed{seed}/fits.json').read_text())
 for h,v in f['heads'].items():lines.append(f'| {seed} | {h} | {v["C"]:g} | {v["tau"]:g} | {v["history"][-1]["gradient_inf"]:.3g} | {min(z["ba"] for z in v["validation"].values())*100:.3f} | {max(z["loss"] for z in v["validation"].values()):.6f} |')
lines+=['','baseline复用B-F冻结头，其余零初始化；不按验证选择方法。保留未截断区间；区间退化到100%不等于总体无误差。完整优化轨迹与逐条件验证数值在fits.json。']
(out/'FULL_RESULTS.md').write_text('\n'.join(lines)+'\n')
fig,axes=plt.subplots(1,4,figsize=(13,3.6),sharey=True)
for ax,(k,v) in zip(axes,r['effects'].items()):
 ax.bar(range(4),[v['means'][h]*100 for h in HEADS],color=['#888888','#d99732','#4384b0','#4d9760']);ax.axhline(85,color='black',ls='--');ax.set(title=k,xticks=range(4),xticklabels=HEADS,ylim=(0,102))
axes[0].set_ylabel('Balanced accuracy (%)');fig.suptitle('B-G: fixed regularization and risk-temperature interventions');fig.tight_layout();fig.savefig(out/'factorial_results.png',dpi=170);plt.close(fig)
print('Full matrix, controls, optimization diagnostics and figure written.')
