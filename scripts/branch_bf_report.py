import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from branch_bf_round1 import ROOT,HEADS,DELAYS
r=json.loads((ROOT/'summary.json').read_text());out=Path('reports/branch_bf');s=np.load(ROOT/'analysis_arrays.npz')['scores'];m=s.mean(1)
fig,axes=plt.subplots(1,4,figsize=(14,3.6),sharey=True)
for ax,(key,v) in zip(axes,r['targets'].items()):
 vals=[v['means'][h] for h in HEADS];ax.bar(range(4),vals,color=['#777777','#3b80bc','#e49435','#4c9855']);ax.axhline(.85,color='black',ls='--');ax.set(title=key.replace('_',' '),xticks=range(4),xticklabels=['base\nmix','base\nsplit','cover\nmix','cover\nsplit'],ylim=(0,1.02))
axes[0].set_ylabel('Balanced accuracy: fixed model');fig.suptitle('B-F: four prespecified persistent weak conditions');fig.tight_layout();fig.savefig(out/'factorial_targets.png',dpi=170);plt.close(fig)
lines=['# B-F完整结果','','主指标为cover_split的110个固定个体条件，各区间99.954545…%，20库、df19；其余比较为配对描述。BA和区间单位%。','','| 模型 | 时长 | 条件 | base_mix | base_split | cover_mix | cover_split | 主区间 | 相对85% |','|---|---|---|---:|---:|---:|---:|---|---|']
for si,seed in enumerate(range(301,306)):
 for di,d in enumerate(DELAYS):
  for ci,c in enumerate(['clean','noise']):
   v=r['primary'][f'{seed}_d{d}_{c}'];status='支持' if v['lower']>.85 else ('排除' if v['upper']<.85 else '未决')
   lines.append(f'| {seed} | {d} | {c} | '+' | '.join(f'{100*x:.3f}' for x in m[si,di,:,ci])+f' | [{100*v["lower"]:.3f}, {100*v["upper"]:.3f}] | {status} |')
lines+=['','## 预定薄弱条件的配对效应（百分点）','','| 条件 | 覆盖效应mix | 覆盖效应split | 分组效应base | 分组效应cover | 交互差分 |','|---|---:|---:|---:|---:|---:|']
for key,v in r['targets'].items():lines.append('| '+key+' | '+' | '.join(f'{100*v[k]:+.3f}' for k in ['coverage_mix','coverage_split','split_base','split_cover','interaction'])+' |')
lines+=['','效应均为同库配对点估计，不提供新的确认性机制裁决。C选择和验证规则属于完整流程；覆盖同时改变基础权重。','','## 正对照','','| 条件 | 均值 | 区间 |','|---|---:|---|']
for k,v in r['positive_controls'].items():lines.append(f'| {k} | {100*v["mean"]:.3f} | [{100*v["lower"]:.3f}, {100*v["upper"]:.3f}] |')
lines+=['','## 负对照','','每头时长条件的清零／标签置换／test数量内状态错配，先库内平均置换，再平均库与五模型。','','| 时长 | 方法 | 输入 | 清零 | 标签置换 | 状态错配 |','|---|---|---|---:|---:|---:|']
neg=np.array(r['negative_means'])
for di,d in enumerate(DELAYS):
 for hi,h in enumerate(HEADS):
  for ci,c in enumerate(['clean','noise']):lines.append(f'| {d} | {h} | {c} | '+' | '.join(f'{100*x:.3f}' for x in neg[di,hi,ci])+' |')
from branch_bf_round1 import REFERENCE_DELAYS
for di,d in enumerate(REFERENCE_DELAYS):
 for ci,c in enumerate(['clean','noise']):lines.append(f'| {d} | reference | {c} | '+' | '.join(f'{100*x:.3f}' for x in r['reference_negative_means'][di][ci])+' |')
lines+=['','## 最终选参与优化','','| 模型 | 方法 | C | 最大验证条件损失 | 梯度∞范数 |','|---|---|---:|---:|---:|']
for seed in range(301,306):
 fits=json.loads((ROOT/f'seed{seed}/fits.json').read_text())
 for h,v in fits['heads'].items():
  row=next(x for x in v['grid'] if x['C']==v['C']);lines.append(f'| {seed} | {h} | {v["C"]:g} | {v["validation_loss"]:.6f} | {row["history"][-1]["gradient_inf"]:.3g} |')
lines+=['','全部100候选、组权重与逐时长拟合诊断在各模型fits.json和candidates.pkl。保留未截断t区间；天花板处退化／越界不是总体100%准确证据。']
(out/'FULL_RESULTS.md').write_text('\n'.join(lines)+'\n')
