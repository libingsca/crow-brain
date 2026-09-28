"""Export B-L complete tables and a static scientific figure from audited results."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from branch_bl_round1 import ROOT,DELAYS
from branch_bl_analyze import HEADS
r=json.loads((ROOT/'summary.json').read_text());out=Path('reports/branch_bl');a=np.load(ROOT/'analysis_arrays.npz');scores=a['scores'];means=scores.mean(1)
labels=['Frozen shared clean','Frozen shared augmented','Local clean','Local augmented']
fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
for ci,ax in enumerate(axes):
 for h,label in zip(HEADS,labels):
  cells=[r['panel_all_cells'][f'{h}_d{d}_{["clean","noise"][ci]}'] for d in DELAYS]
  y=np.array([c['mean'] for c in cells]);lo=np.array([c['lower'] for c in cells]);hi=np.array([c['upper'] for c in cells])
  ax.errorbar(DELAYS,y,yerr=np.stack([y-lo,hi-y]),marker='o',capsize=3,label=label)
 ax.axhline(.85,color='black',ls='--',lw=1);ax.set(title=['Clean input','Fixed 1% noise pulse'][ci],xlabel='Delay steps',xticks=DELAYS,ylim=(.5,1.03))
axes[0].set_ylabel('Balanced accuracy: fixed five-model panel');axes[1].legend(fontsize=8,loc='lower right')
fig.suptitle('B-L: local readout recovery at shared-readout failure delays');fig.tight_layout();fig.savefig(out/'local_recovery.png',dpi=180);plt.close(fig)
lines=['# B-L 完整数值结果','','BA单位为百分比。六项主要指标为专用增强头；主区间99.166667%，20个新测试库，固定五模型。其他同公式区间仅描述，不增加确认性结论。','','| 时长 | 输入 | 共享干净头 | 共享增强头 | 专用干净头 | 专用增强头 | 专用增强头区间 | 增强−共享增强（百分点） | 增强−专用干净（百分点） |','|---|---|---:|---:|---:|---:|---|---:|---:|']
for di,d in enumerate(DELAYS):
 for ci,c in enumerate(['clean','noise']):
  cells=[r['panel_all_cells'][f'{h}_d{d}_{c}'] for h in HEADS];v=cells[-1]
  lines.append(f'| {d} | {c} | '+' | '.join(f'{100*x["mean"]:.3f}' for x in cells)+f' | [{100*v["lower"]:.3f}, {100*v["upper"]:.3f}] | {100*r["paired_aug_minus_shared_aug"][di][ci]:+.3f} | {100*r["paired_aug_minus_clean"][di][ci]:+.3f} |')
lines+=['','## 全部单模型成绩','','| 模型 | 时长 | 输入 | 共享干净 | 共享增强 | 专用干净 | 专用增强 |','|---|---|---|---:|---:|---:|---:|']
for si,seed in enumerate(range(301,306)):
 for di,d in enumerate(DELAYS):
  for ci,c in enumerate(['clean','noise']):lines.append(f'| {seed} | {d} | {c} | '+' | '.join(f'{100*v:.3f}' for v in means[si,di,:,ci])+' |')
lines+=['','## 正对照','','| 条件 | 均值 | 区间 |','|---|---:|---|']
for name,v in r['positive_controls'].items():lines.append(f'| {name} | {100*v["mean"]:.3f} | [{100*v["lower"]:.3f}, {100*v["upper"]:.3f}] |')
lines+=['','## 全部负对照','','| 时长 | 头 | 输入 | 清零 | 标签置换 | 测试数量内状态错配 |','|---|---|---|---:|---:|---:|']
neg=np.array(r['negative_means_delay_head_condition_control'])
for di,d in enumerate(DELAYS):
 for hi,h in enumerate(HEADS):
  for ci,c in enumerate(['clean','noise']):lines.append(f'| {d} | {h} | {c} | '+' | '.join(f'{100*v:.3f}' for v in neg[di,hi,ci])+' |')
lines+=['','## 选定头诊断','','所有150候选及各条件BA／损失保存在各模型fits.json；此处列选定的30头。','','| 模型 | 头 | C | 训练损失 | 验证损失 | 梯度∞范数 |','|---|---|---:|---:|---:|---:|']
for seed in range(301,306):
 fits=json.loads((ROOT/f'seed{seed}/fits.json').read_text())
 for name,m in fits['heads'].items():
  row=next(x for x in m['grid'] if x['C']==m['C'])
  lines.append(f'| {seed} | {name} | {m["C"]:g} | {row["training_loss"]:.6f} | {row["validation_loss"]:.6f} | {row["history"][-1]["gradient_inf"]:.3g} |')
lines+=['','保留未截断t区间，接近天花板的退化或越界不证明总体100%准确。质量负对照为流程校准，不是统计等效证明。区间不涵盖新核心或重新拟合的不确定性。']
(out/'FULL_RESULTS.md').write_text('\n'.join(lines)+'\n')
print('Exported full tables and local_recovery.png')
