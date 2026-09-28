"""Rebuild complete B-C tables and static scientific figure from audited summary."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from branch_bc_round1 import ROOT,DELAYS,TRAIN_DELAYS,HEADS,BASE_DELAYS
r=json.loads((ROOT/'summary.json').read_text());out=Path('reports/branch_bc');a=np.load(ROOT/'analysis_arrays.npz');scores=a['scores'];means=scores.mean(1)
fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
labels=['Five-delay clean','Five-delay augmented','Eight-delay clean','Eight-delay augmented']
for ci,ax in enumerate(axes):
 for name,label in zip(HEADS,labels):
  cells=[r['panel_all_cells'][f'{name}_d{d}_{["clean","noise"][ci]}'] for d in DELAYS]
  y=np.array([v['mean'] for v in cells]);lo=np.array([v['lower'] for v in cells]);hi=np.array([v['upper'] for v in cells])
  ax.errorbar(DELAYS,y,yerr=np.stack([y-lo,hi-y]),marker='o',capsize=2,label=label,lw=1.5,ms=4)
 for d in [8,11,12]:ax.axvspan(d-.18,d+.18,color='gray',alpha=.13)
 ax.axhline(.85,color='black',ls='--',lw=1);ax.set(title=['Clean input','Fixed 1% noise pulse'][ci],xlabel='Delay steps',xticks=DELAYS,ylim=(.5,1.03))
axes[0].set_ylabel('Balanced accuracy: fixed five-model panel');axes[1].legend(fontsize=8,loc='lower left')
fig.suptitle('B-C: matched training coverage intervention (shaded: added delays)');fig.tight_layout();fig.savefig(out/'coverage_intervention.png',dpi=180);plt.close(fig)
lines=['# B-C完整数值结果','','BA为百分比；配对增益为百分点。主终点28项使用99.82142857%双侧t区间，20个新测试库。其余同公式区间仅描述，不构成额外确认性结论。','','## 全时长表现','','| 时长 | 输入 | 五时长干净头 | 五时长增强头 | 八时长干净头 | 八时长增强头 | 八时长增强区间 | 八−五增强头（百分点） |','|---|---|---:|---:|---:|---:|---|---:|']
for di,d in enumerate(DELAYS):
 for ci,c in enumerate(['clean','noise']):
  cells=[r['panel_all_cells'][f'{h}_d{d}_{c}'] for h in HEADS];v=cells[3]
  lines.append(f'| {d} | {c} | '+' | '.join(f'{100*x["mean"]:.3f}' for x in cells)+f' | [{100*v["lower"]:.3f}, {100*v["upper"]:.3f}] | {100*r["paired_expanded_aug_minus_base_aug"][di][ci]:+.3f} |')
lines+=['','## 全部28项主要判据','','| 指标 | 均值 | 校正区间 | 门槛 | 裁决 |','|---|---:|---|---:|---|']
for name,v in r['primary'].items():
 status='支持' if v['lower']>v['threshold'] else '排除' if v['upper']<v['threshold'] else '未决'
 lines.append(f'| {name} | {100*v["mean"]:.3f} | [{100*v["lower"]:.3f}, {100*v["upper"]:.3f}] | {100*v["threshold"]:.1f} | {status} |')
lines+=['','## 全部440项单模型成绩','','| 模型 | 时长 | 输入 | 五时长干净 | 五时长增强 | 八时长干净 | 八时长增强 |','|---|---|---|---:|---:|---:|---:|']
for si,seed in enumerate(range(301,306)):
 for di,d in enumerate(DELAYS):
  for ci,c in enumerate(['clean','noise']):lines.append(f'| {seed} | {d} | {c} | '+' | '.join(f'{100*v:.3f}' for v in means[si,di,:,ci])+' |')
lines+=['','## 正对照','','| 条件 | 均值 | 区间 |','|---|---:|---|']
for name,v in r['positive_controls'].items():lines.append(f'| {name} | {100*v["mean"]:.3f} | [{100*v["lower"]:.3f}, {100*v["upper"]:.3f}] |')
lines+=['','## 全部负对照','','| 时长 | 头 | 输入 | 清零 | 标签置换 | test数量内状态错配 |','|---|---|---|---:|---:|---:|']
for di,d in enumerate(DELAYS):
 for hi,h in enumerate(HEADS):
  for ci,c in enumerate(['clean','noise']):lines.append(f'| {d} | {h} | {c} | '+' | '.join(f'{100*v:.3f}' for v in r['negative_means_delay_head_condition_control'][di][hi][ci])+' |')
for ti,d in enumerate(TRAIN_DELAYS):
 for ci,c in enumerate(['clean','noise']):lines.append(f'| {d} | 专用参照 | {c} | '+' | '.join(f'{100*v:.3f}' for v in r['reference_negative_means'][ti][ci])+' |')
lines+=['','## 选定头拟合诊断','','| 模型 | 头 | C | 训练损失 | 验证损失 | 梯度∞范数 |','|---|---|---:|---:|---:|---:|']
for seed in range(301,306):
 fits=json.loads((ROOT/f'seed{seed}/fits.json').read_text())
 for name,m in fits['heads'].items():
  row=next(x for x in m['grid'] if x['C']==m['C'])
  lines.append(f'| {seed} | {name} | {m["C"]:g} | {row["training_loss"]:.6f} | {row["validation_loss"]:.6f} | {row["history"][-1]["gradient_inf"]:.3g} |')
lines+=['','## 选定增强头的逐时长训练／验证诊断','','验证均为共同八时长；训练为各自覆盖集合。混合损失与BA按干净／噪声各半计算。','','| 模型 | 头 | 分区 | 时长 | 损失 | BA |','|---|---|---|---|---:|---:|']
for seed in range(301,306):
 fits=json.loads((ROOT/f'seed{seed}/fits.json').read_text())
 for name in ['base_aug','expanded_aug']:
  m=fits['heads'][name];row=next(x for x in m['grid'] if x['C']==m['C'])
  for part,ds in row['per_delay'].items():
   for d,v in ds.items():lines.append(f'| {seed} | {name} | {part} | {d} | {v["loss"]:.6f} | {100*v["ba"]:.3f} |')
lines+=['','全部100候选、逐时长损失及BA见各模型fits.json及candidates.pkl。保持未截断t区间；天花板处退化或越界不证明总体无误差。负对照是流程校准，不是等效检验。推断条件化于固定核心及本次拟合。']
(out/'FULL_RESULTS.md').write_text('\n'.join(lines)+'\n');print('Full tables and coverage_intervention.png written.')
