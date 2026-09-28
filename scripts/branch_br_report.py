"""Full tables and scientific figure for the completed B-R round."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from branch_br_round1 import ROOT,DELAYS
r=json.loads((ROOT/'summary.json').read_text());out=Path('reports/branch_br');a=np.load(ROOT/'analysis_arrays.npz');scores=a['scores'];means=scores.mean(1)
fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
for ci,ax in enumerate(axes):
 for hi,(name,label) in enumerate([('old','Frozen 15-step augmented'),('clean','Delay-specific clean'),('aug','Delay-specific augmented')]):
  cells=[r['panel_all_cells'][f'{name}_d{d}_{["clean","noise"][ci]}'] for d in DELAYS]
  y=np.array([c['mean'] for c in cells]);lower=np.array([c['lower'] for c in cells]);upper=np.array([c['upper'] for c in cells])
  ax.errorbar(DELAYS,y,yerr=np.stack([y-lower,upper-y]),marker='o',capsize=3,label=label)
 ax.axhline(.85,color='black',ls='--',lw=1);ax.set(title=['Clean input','Fixed 1% noise pulse'][ci],xlabel='Delay steps',xticks=DELAYS,ylim=(0,1.03))
axes[0].set_ylabel('Balanced accuracy: fixed five-model panel');axes[1].legend(fontsize=8,loc='lower right');fig.suptitle('B-R: delay-specific linear readout recovery');fig.tight_layout();fig.savefig(out/'readout_recovery.png',dpi=180);plt.close(fig)
lines=['# B-R 完整数值结果','','BA 均为百分比；区间为20个独立测试库上的99.375%双侧 t 区间。主指标仅为新专用增强头；旧头与专用干净头为次要参照。所有推断条件化于已固定的核心、训练／验证数据和最终头。','','| 延迟 | 输入 | 旧15步增强头 | 新专用干净头 | 新专用增强头 | 增强头区间 | 增强−旧头（百分点） | 增强−干净头（百分点） |','|---|---|---:|---:|---:|---|---:|---:|']
for di,d in enumerate(DELAYS):
 for ci,c in enumerate(['clean','noise']):
  cells=[r['panel_all_cells'][f'{h}_d{d}_{c}'] for h in ['old','clean','aug']];v=cells[2]
  lines.append(f'| {d} | {c} | '+ ' | '.join(f'{100*x["mean"]:.3f}' for x in cells)+f' | [{100*v["lower"]:.3f}, {100*v["upper"]:.3f}] | {100*r["paired_aug_minus_old"][di][ci]:+.3f} | {100*r["paired_aug_minus_clean"][di][ci]:+.3f} |')
lines+=['','## 全部单模型成绩','','| 模型 | 延迟 | 输入 | 旧增强头 | 专用干净头 | 专用增强头 |','|---|---|---|---:|---:|---:|']
for si,seed in enumerate(range(301,306)):
 for di,d in enumerate(DELAYS):
  for ci,c in enumerate(['clean','noise']):lines.append(f'| {seed} | {d} | {c} | '+' | '.join(f'{100*v:.3f}' for v in means[si,di,:,ci])+' |')
lines+=['','## 各模型增强头八项中最低成绩','']
for seed,v in r['per_model_aug_minimum'].items():lines.append(f'- {seed}: {100*v:.3f}%')
lines+=['','## 正对照','','| 条件 | 均值 | 区间 |','|---|---:|---|']
for name,v in r['positive_controls'].items():lines.append(f'| {name} | {100*v["mean"]:.3f} | [{100*v["lower"]:.3f}, {100*v["upper"]:.3f}] |')
lines+=['','原任务头的库间方差若为零，退化区间不证明总体准确率恰为100%。','','## 全部主要条件负对照','','各头／时长／输入条件的库内置换平均后，再平均五模型和20库。噪声置换用重复0。','','| 延迟 | 头 | 输入 | 清零 | 标签置换 | 测试数量内状态错配 |','|---|---|---|---:|---:|---:|']
neg=np.array(r['negative_means_delay_head_condition_control'])
for di,d in enumerate(DELAYS):
 for hi,h in enumerate(['old','clean','aug']):
  for ci,c in enumerate(['clean','noise']):lines.append(f'| {d} | {h} | {c} | '+' | '.join(f'{100*v:.3f}' for v in neg[di,hi,ci])+' |')
lines+=['','## 拟合与选择诊断','','训练／验证损失按对应方法分布的权重计算；其分布不同，不能直接把不同方法的损失高低等同为方法优劣。完整200候选的历史保存在各模型 fits.json 和 candidates.pkl。','','| 模型 | 头 | C | 训练损失 | 验证损失 | 最终梯度∞范数 |','|---|---|---:|---:|---:|---:|']
for seed in range(301,306):
 fits=json.loads((ROOT/f'seed{seed}/fits.json').read_text())
 for name,m in fits['heads'].items():
  row=next(x for x in m['grid'] if x['C']==m['C'])
  lines.append(f'| {seed} | {name} | {m["C"]:g} | {row["training_loss"]:.6f} | {row["validation_loss"]:.6f} | {row["history"][-1]["gradient_inf"]:.3g} |')
lines+=['','区间说明：按预定普通 t 公式保留未截断区间；接近天花板时上界可能略高于100%，这不是准确率超过100%。零样本方差可导致退化区间，也不构成总体无误差保证。覆盖陈述依赖库级抽样与 t 近似。']
(out/'FULL_RESULTS.md').write_text('\n'.join(lines)+'\n')
