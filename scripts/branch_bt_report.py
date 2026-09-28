"""Render the complete B-T results; scientific decision text is separately reviewed."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from branch_bt_round1 import ROOT,DELAYS,HEADS
r=json.loads((ROOT/'summary.json').read_text());out=Path('reports/branch_bt')
s=np.load(ROOT/'analysis_arrays.npz')['scores'];means=s.mean(1)
fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
for ci,ax in enumerate(axes):
    for si,seed in enumerate(range(301,306)):
        ax.plot(DELAYS,means[si,:,1,ci],marker='o',label=str(seed))
    ax.axhline(.85,c='black',ls='--',lw=1,label='Useful-transfer threshold')
    ax.axhline(.5,c='grey',ls=':',lw=1)
    ax.set(title=['Clean input','Fixed 1% noise pulse'][ci],xlabel='Delay steps',xticks=DELAYS,ylim=(0,1.02))
axes[0].set_ylabel('Balanced accuracy (20-bank mean)');axes[1].legend(fontsize=8)
fig.suptitle('Frozen 15-step noise-augmented heads: all five fixed models');fig.tight_layout();fig.savefig(out/'time_generalization.png',dpi=180);plt.close(fig)
lines=['# B-T 首轮完整数值结果','','所有数值为 BA 百分比；主区间为99.375%双侧 t 区间，20图像库、固定五模型。','', '| 延迟 | 条件 | 增强头集合均值 | 区间 | 干净训练头均值 | 增强−干净（百分点） |','|---|---|---:|---|---:|---:|']
for di,d in enumerate(DELAYS):
 for ci,c in enumerate(['clean','noise']):
  v=r['panel_all_cells'][f'aug_d{d}_{c}'];base=r['panel_all_cells'][f'clean_d{d}_{c}']['mean']
  lines.append(f"| {d}{'（锚点）' if d==15 else ''} | {c} | {100*v['mean']:.3f} | [{100*v['lower']:.3f}, {100*v['upper']:.3f}] | {100*base:.3f} | {100*(v['mean']-base):+.3f} |")
lines+=['','## 全部单模型结果','','| 模型 | 延迟 | 干净训练头／无噪 | 干净训练头／有噪 | 增强头／无噪 | 增强头／有噪 |','|---|---|---:|---:|---:|---:|']
for si,seed in enumerate(range(301,306)):
 for di,d in enumerate(DELAYS):lines.append(f'| {seed} | {d} | '+' | '.join(f'{100*v:.3f}' for v in means[si,di].flatten())+' |')
lines+=['','## 每个模型在八项迁移条件中的最低成绩（增强头）','']
for seed,v in r['per_model_aug_minimum_new_delays'].items(): lines.append(f'- {seed}: {100*v:.3f}%')
lines+=['','## 正对照','','| 条件 | 均值 | 区间 |','|---|---:|---|']
for name,v in r['positive_controls'].items():lines.append(f"| {name} | {100*v['mean']:.3f} | [{100*v['lower']:.3f}, {100*v['upper']:.3f}] |")
lines+=['','## 负对照','','每库先平均64次置换，再平均模型和库；噪声使用重复0。清零仅一次。15步另列，不参与预定主要负对照门槛。','','| 延迟 | 头 | 条件 | 清零 | 标签置换 | 测试数量内状态错配 |','|---|---|---|---:|---:|---:|']
neg=np.asarray(r['negative_means_delay_head_condition_control'])
for di,d in enumerate(DELAYS):
 for hi,h in enumerate(HEADS):
  for ci,c in enumerate(['clean','noise']):lines.append(f'| {d} | {h} | {c} | '+' | '.join(f'{100*v:.3f}' for v in neg[di,hi,ci])+' |')
(out/'FULL_RESULTS.md').write_text('\n'.join(lines)+'\n')
