"""Scientific reporting, with refit-level uncertainty and all individual results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('artifacts/branch_bz');out=Path('reports/branch_bz');s=json.loads((root/'summary.json').read_text());assert s['quality_passed'];assert json.loads((root/'crosscheck.json').read_text())['passed']
a=np.load(root/'analysis_arrays.npz');per=a['per_refit'];ds=list(range(5,16));labels=['clean','noise'];methods=['mean_aug','worst_aug']
state={'supported':'支持','excluded':'排除','inconclusive':'未决'}
lines=['# B-Z：独立重拟合重复实验结果','','2026-09-25。全部5个独立批次完成，质量审计通过。','',f'**平均功能假设：{state[s["panel_classification"]["decision"]]}；个体最差表现假设：{state[s["individual_classification"]["decision"]]}。**','','本轮复用固定五核心与B-W增强流程，仅更换独立拟合、验证和测试数据；每批10头、50候选，合计50头、250候选。训练全部冻结后生成20个新测试库，共70,000新图像。未调整温度、时长覆盖或C网格。','','统计单位为5个独立批次，每批含4库。以下27项区间为99.81481481%双侧t区间，df=4，联合Bonferroni5%。平均问题针对固定五核心的期望平均；个体问题针对每核心每批22条件的有限测试观测最小值之期望，存在取最小值的向下偏差。5批的正态近似和估计精度是限制。','','## 27项确认性结果','','| 家族／条件 | 均值% | 校正区间% | 相对85% |','|---|---:|---|---|']
for family in ['panel','individual']:
 for name,v in s[family].items():
  st='支持' if v['lower']>.85 else '排除' if v['upper']<.85 else '未决'
  lines.append(f'| {family}/{name} | {100*v["mean"]:.3f} | [{100*v["lower"]:.3f}, {100*v["upper"]:.3f}] | {st} |')
lines+=['','## 目标时长增益：仅描述','','这些未校正95%区间不构成新增确认性结论，也不改变B-W的>2个百分点判据和结案。','','| 条件 | worst−mean，百分点 | 95%区间 | 五批增益 |','|---|---:|---|---|']
for name,v in s['gains_descriptive'].items():lines.append(f'| {name} | {100*v["mean"]:.3f} | [{100*v["lower"]:.3f}, {100*v["upper"]:.3f}] | '+', '.join(f'{100*x:.3f}' for x in v['values'])+' |')
lines+=['','## 个体薄弱条件稳定性','','旧最差条件在B-W既有结果上固定。本轮排序为1最差，允许并列；全部为描述性诊断。','','| 核心 | B-W最差条件 | 本轮同条件五批BA% | 五批排名 | 本轮五个最差条件 |','|---|---|---|---|---|']
for seed,v in s['stability'].items():
 old=v['bw_minimum'];mins=v['new_minima']
 lines.append(f'| {seed} | d{old["delay"]}/{old["condition"]} | '+', '.join(f'{100*x:.2f}' for x in v['same_condition_values'])+' | '+str(v['same_condition_ranks'])+' | '+', '.join(f'd{x["delay"]}/{x["condition"]}: {100*x["value"]:.2f}%' for x in mins)+' |')
lines+=['','## 质量与执行','','58项实现／回归测试通过。250候选全部独立重算组损失、目标、梯度、权重和选参；1,100测试单元、33,000组线性预测独立重算，包含全部噪声重复、清零、64次标签置换及64次分层错配。27项统计结果从原始单元再次独立聚合。审计复用模型定义，不是完全独立软件实现。','',f'正对照每批平均最低{100*s["positive_batch_minimum"]:.3f}%，跨批校正区间下界最低{100*min(v["lower"] for v in s["positive_controls"].values()):.3f}%；负对照范围{100*s["negative_range"][0]:.3f}%–{100*s["negative_range"][1]:.3f}%，在47%–53%流程门槛内。负对照通过不是等效证明。','','新图像相互及相对224,000张历史留存图像无像素重复；未留存在线图像不在核查范围。所有50头先审计冻结，随后生成任何测试数据。最多两个进程，每进程一线程。历史科学产物未修改；当前规划入口原文单独归档。','','## 停止和边界','','本轮按执行前固定的5批完成。任何未决项如实保留，不追加批次来改变显著性。本轮完成不等于所有科学问题已有确定答案。结论仍限于固定五核心和既定图像生成分布，不外推新核心；所有时长在历史开发中均已知。有限批次数无法可靠描述重拟合尾部失败概率。未截断t区间超出[0,1]不代表总体无误差。','','![独立重拟合结果](refit_replication.png)','','[执行前协议](../../artifacts/branch_bz/protocol.md) · [全部批次与个体数值](FULL_RESULTS.md) · [机器汇总](../../artifacts/branch_bz/summary.json) · [完整性审计](../../artifacts/branch_bz/integrity.json) · [复现清单](../../artifacts/branch_bz/reproducibility_manifest.json)']
(out/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
full=['# B-Z全部重拟合结果','','BA为百分比。各行先平均本批4库；有噪再先平均每库10次噪声。','','| 批次 | 核心 | 时长 | 输入 | mean_aug | worst_aug |','|---|---|---|---|---:|---:|']
for ri in range(5):
 for si,seed in enumerate(range(301,306)):
  for di,d in enumerate(ds):
   for ci,c in enumerate(labels):full.append(f'| {ri+1} | {seed} | {d} | {c} | {100*per[ri,si,di,0,ci]:.3f} | {100*per[ri,si,di,1,ci]:.3f} |')
full+=['','## 确认性指标的五批原值','','| 家族／条件 | 五批BA% |','|---|---|']
for fam in ['panel','individual']:
 for name,v in s[fam].items():full.append(f'| {fam}/{name} | '+', '.join(f'{100*x:.3f}' for x in v['values'])+' |')
full+=['','## 选定C','','| 批次 | 核心 | mean_aug | worst_aug |','|---|---|---:|---:|']
for r in range(1,6):
 for seed in range(301,306):
  f=json.loads((root/f'repeat{r}/seed{seed}/fits.json').read_text())['heads'];full.append(f'| {r} | {seed} | {f["mean_aug"]["C"]} | {f["worst_aug"]["C"]} |')
(out/'FULL_RESULTS.md').write_text('\n'.join(full)+'\n')
fig,axs=plt.subplots(1,3,figsize=(15,4.5))
for ci,ax in enumerate(axs[:2]):
 for ri in range(5):ax.plot(ds,per[ri,:,:,1,ci].mean(0),alpha=.5,marker='.',label=f'Refit {ri+1}')
 ax.axhline(.85,color='black',ls='--');ax.set(title=labels[ci]+' / fixed-panel mean',xlabel='Delay',ylim=(.75,1));ax.set_xticks(ds)
axs[0].set_ylabel('Balanced accuracy');axs[1].legend(fontsize=8)
for si,seed in enumerate(range(301,306)):
 v=s['individual'][str(seed)];axs[2].scatter(np.full(5,si),v['values'],alpha=.6);axs[2].errorbar(si,v['mean'],yerr=[[v['mean']-v['lower']],[v['upper']-v['mean']]],color='black',capsize=3)
axs[2].axhline(.85,color='black',ls='--');axs[2].set(title='Individual minimum over 22 conditions',xticks=range(5),xticklabels=range(301,306),xlabel='Frozen core',ylim=(.6,1))
fig.tight_layout();fig.savefig(out/'refit_replication.png',dpi=180);plt.close(fig)
print('Report and full refit tables written')
