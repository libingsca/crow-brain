"""Render complete B-S tables and fixed-panel scientific plot."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from branch_bs_round1 import ROOT,DELAYS,TRAIN_DELAYS,HELDOUT
r=json.loads((ROOT/'summary.json').read_text());out=Path('reports/branch_bs');data=np.load(ROOT/'analysis_arrays.npz');scores=data['scores'];means=scores.mean(1)
fig,axes=plt.subplots(1,2,figsize=(12,4.6),sharey=True)
for ci,ax in enumerate(axes):
 for hi,(name,label) in enumerate([('clean','Shared clean-trained'),('aug','Shared noise-augmented')]):
  cells=[r['panel_all_cells'][f'{name}_d{d}_{["clean","noise"][ci]}'] for d in DELAYS];y=np.array([c['mean'] for c in cells]);lo=np.array([c['lower'] for c in cells]);hiarr=np.array([c['upper'] for c in cells])
  line=ax.errorbar(DELAYS,y,yerr=np.stack([y-lo,hiarr-y]),marker='o',capsize=2,label=label)
  ids=[DELAYS.index(d) for d in HELDOUT];ax.scatter(HELDOUT,y[ids],s=35,facecolors='white',edgecolors=line[0].get_color(),zorder=4)
 ax.scatter(TRAIN_DELAYS,data['references'].mean((0,1))[:,ci],marker='x',color='green',label='Frozen delay-specific reference',zorder=5)
 ax.axhline(.85,color='black',ls='--',lw=1,label='85% threshold');ax.set(title=['Clean input','Fixed 1% noise pulse'][ci],xlabel='Delay steps (hollow circles: held out)',xticks=DELAYS,ylim=(0,1.03))
axes[0].set_ylabel('Balanced accuracy: fixed five-model panel');axes[1].legend(fontsize=8,loc='lower right');fig.suptitle('B-S: one fixed readout per model across delays');fig.tight_layout();fig.savefig(out/'shared_readout.png',dpi=180);plt.close(fig)
lines=['# B-S 完整数值结果','','BA为百分比。每项主区间为99.772727…%双侧 t 区间，基于20个库的五模型集合均值。主检验仅为增强共享头22项，干净共享头和参考头为次要比较／质量对照。','','| 时长 | 角色 | 输入 | 共享干净头 | 共享增强头 | 增强头主区间 | 相对85%判据 | 增强−干净（百分点） |','|---|---|---|---:|---:|---|---|---:|']
for di,d in enumerate(DELAYS):
 for ci,c in enumerate(['clean','noise']):
  v=r['primary'][f'd{d}_{c}'];b=r['panel_all_cells'][f'clean_d{d}_{c}']['mean'];status='支持' if v['lower']>.85 else ('排除' if v['upper']<.85 else '未决')
  lines.append(f'| {d} | {"留出" if d in HELDOUT else "拟合"} | {c} | {100*b:.3f} | {100*v["mean"]:.3f} | [{100*v["lower"]:.3f}, {100*v["upper"]:.3f}] | {status} | {100*(v["mean"]-b):+.3f} |')
lines+=['','## 全部单模型成绩','','| 模型 | 时长 | 角色 | 共享干净头／无噪 | 共享干净头／有噪 | 共享增强头／无噪 | 共享增强头／有噪 |','|---|---|---|---:|---:|---:|---:|']
for si,seed in enumerate(range(301,306)):
 for di,d in enumerate(DELAYS):lines.append(f'| {seed} | {d} | {"留出" if d in HELDOUT else "拟合"} | '+' | '.join(f'{100*v:.3f}' for v in means[si,di].flatten())+' |')
lines+=['','## 各模型增强头22条件中最低均值','']
for seed,v in r['per_model_aug_minimum'].items():lines.append(f'- {seed}: {100*v:.3f}%')
lines+=['','## 正对照及共享增强头相对专用头差距','','| 对照 | 均值 | 区间 | 共享增强−专用参考（百分点） |','|---|---:|---|---:|']
for name,v in r['positive_controls'].items():
 diff='—'
 if name.startswith('reference_'):
  _,dd,c=name.split('_');di=TRAIN_DELAYS.index(int(dd[1:]));ci=['clean','noise'].index(c);diff=f'{100*r["paired_aug_minus_reference"][di][ci]:+.3f}'
 lines.append(f'| {name} | {100*v["mean"]:.3f} | [{100*v["lower"]:.3f}, {100*v["upper"]:.3f}] | {diff} |')
lines+=['','## 所有共享头负对照','','| 时长 | 头 | 输入 | 清零 | 标签置换 | 测试数量内状态错配 |','|---|---|---|---:|---:|---:|']
neg=np.array(r['negative_means_delay_head_condition_control']);rn=np.array(r['reference_negative_means'])
for di,d in enumerate(DELAYS):
 for hi,h in enumerate(['clean','aug']):
  for ci,c in enumerate(['clean','noise']):lines.append(f'| {d} | {h} | {c} | '+' | '.join(f'{100*v:.3f}' for v in neg[di,hi,ci])+' |')
for ti,d in enumerate(TRAIN_DELAYS):
 for ci,c in enumerate(['clean','noise']):lines.append(f'| {d} | reference | {c} | '+' | '.join(f'{100*v:.3f}' for v in rn[ti,ci])+' |')
lines+=['','## 选择与各拟合时长诊断','','验证损失按五时长平均用于选C；下表逐时长损失／加权BA仅用于描述。增强方法的拟合／验证BA是50%干净+50%噪声的混合分布值，非单独噪声BA。完整50候选参数和诊断在各模型 candidates.pkl／fits.json。','','| 模型 | 头 | 选定C | 时长 | 训练损失 | 验证损失 | 训练BA | 验证BA |','|---|---|---:|---|---:|---:|---:|---:|']
for seed in range(301,306):
 fits=json.loads((ROOT/f'seed{seed}/fits.json').read_text())
 for h,m in fits['heads'].items():
  row=next(x for x in m['grid'] if x['C']==m['C'])
  for d in TRAIN_DELAYS:
   a=row['per_delay']['fit'][str(d)];b=row['per_delay']['validation'][str(d)]
   lines.append(f'| {seed} | {h} | {m["C"]:g} | {d} | {a["loss"]:.6f} | {b["loss"]:.6f} | {100*a["ba"]:.3f} | {100*b["ba"]:.3f} |')
lines+=['','区间原样保留，不截断至[0%,100%]。天花板附近区间可能退化或越界，不能据此推断总体100%准确。主区间条件化于这次固定模型和拟合，依赖库级独立采样及 t 近似；不覆盖新核心或重新拟合的不确定性。次要配对差异不作为新的确认性机制检验。']
(out/'FULL_RESULTS.md').write_text('\n'.join(lines)+'\n')
