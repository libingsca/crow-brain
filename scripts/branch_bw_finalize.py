"""Build B-W report after audited results; no experimental choices are made here."""
from pathlib import Path
import json
from branch_bw_round1 import ROOT, DELAYS
r=json.loads((ROOT/'summary.json').read_text())
assert r['quality_passed'] and r['decision'] in ['supported','excluded']
for name in ['independent_audit','data_chronology_integrity','summary_crosscheck']:
 assert json.loads((ROOT/f'{name}.json').read_text())['passed']
out=Path('reports/branch_bw');stats=r['primary_classification'];ab=r['absolute_classification'];gain=r['gain_classification'];panel=r['panel_all_cells']
verdict='获支持' if r['decision']=='supported' else '被限定排除'
lines=[f'# B-W结案：固定覆盖的平滑最差损失目标{verdict}','','2026-09-25。**H-BW1'+verdict+'；全部质量门槛通过，按预定判据结案。**','',f'26项主要要求：{len(stats["supported"])}项支持、{len(stats["excluded"])}项排除、{len(stats["unresolved"])}项未决。绝对功能22项：{len(ab["supported"])}支持／{len(ab["excluded"])}排除／{len(ab["unresolved"])}未决；配对增益4项：{len(gain["supported"])}支持／{len(gain["excluded"])}排除／{len(gain["unresolved"])}未决。','','## 实验设计','','B-C的缺口位于14、15步。本轮冻结五个核心，保持八时长拟合／验证覆盖[5,7,8,10,11,12,13,15]，比较平均时长对数损失与固定tau=0.05的平滑最大时长损失。各有干净／噪声增强版本，共20个新共享线性头、100个候选。四方法共享相同scaler来源、基础权重、C网格和求解器；均以八时长验证损失的最大值选C。14步未参与本轮拟合或验证。','','平滑最大目标是有限温度的损失折中，不是精确minimax，也不直接优化最差准确率。mean为本轮匹配基线，不是历史B-C的逐字复制。C可随目标改变，结论针对完整预定流程。','','成功要求worst_aug在5–15步干净／有噪22条件全部BA>85%，并在14、15步四条件相对mean_aug全部提升>2个百分点。每项使用99.80769231%双侧t区间、df=19，26项Bonferroni联合错误额度5%。先在模型内平均噪声重复，再平均固定五模型，以20个新测试库为推断单位。任一要求上界低于门槛即可排除该合取假设。','','## 全时长结果','','表中两方法均为噪声增强头；无噪／有噪指测试输入条件。BA以百分比表示。','','| 时长 | 平均目标无噪 | 最差目标无噪 | 平均目标有噪 | 最差目标有噪 | 最差−平均，无噪／有噪（百分点） |','|---|---:|---:|---:|---:|---:|']
for di,d in enumerate(DELAYS):
 vals=[100*panel[f'{h}_d{d}_{c}']['mean'] for c in ['clean','noise'] for h in ['mean_aug','worst_aug']]
 delta=r['paired_worst_aug_minus_mean_aug'][di]
 lines.append(f'| {d} | '+' | '.join(f'{v:.3f}%' for v in vals)+f' | {100*delta[0]:+.3f}／{100*delta[1]:+.3f} |')
lines+=['','![平均与平滑最差目标比较](worst_objective.png)','','## 四项预定配对改善','','| 指标 | 增益（百分点） | 校正区间 | >2个百分点 |','|---|---:|---|---|']
for name,v in r['gains'].items():
 state='支持' if v['lower']>.02 else '排除' if v['upper']<.02 else '未决'
 lines.append(f'| {name} | {100*v["mean"]:+.3f} | [{100*v["lower"]:.3f}, {100*v["upper"]:.3f}] | {state} |')
lines+=['','## 未满足或未决的要求','','| 指标 | 均值 | 校正区间 | 门槛 | 裁决 |','|---|---:|---|---:|---|']
for name in stats['excluded']+stats['unresolved']:
 v=r['primary'][name];state='排除' if name in stats['excluded'] else '未决'
 lines.append(f'| {name} | {100*v["mean"]:.3f} | [{100*v["lower"]:.3f}, {100*v["upper"]:.3f}] | {100*v["threshold"]:.1f} | {state} |')
lines+=['','绝对功能以百分比表示，增益以百分点表示。不满足预定2个百分点门槛不等于完全没有改善。全部26项区间、四方法与440个单模型成绩见[完整结果](FULL_RESULTS.md)。','','各模型worst_aug在22条件中的最低均值：'+'；'.join(f'{s}：{100*v:.3f}%' for s,v in r['per_model_worst_aug_minimum'].items())+'。集合判据不代表每个模型均满足门槛。','','## 拟合诊断','','以下为选定增强头的描述性诊断，干净／噪声各半，不新增确认性检验。','','| 头 | 时长 | 训练BA | 验证BA |','|---|---|---:|---:|']
fits={seed:json.loads((ROOT/f'seed{seed}/fits.json').read_text()) for seed in range(301,306)}
for name in ['mean_aug','worst_aug']:
 for d in [5,7,8,10,11,12,13,15]:
  vals={part:[] for part in ['fit','validation']}
  for f in fits.values():
   h=f['heads'][name];row=next(x for x in h['grid'] if x['C']==h['C'])
   for part in vals:vals[part].append(row['per_delay'][part][str(d)]['ba'])
  lines.append(f'| {name} | {d} | {100*sum(vals["fit"])/5:.3f}% | {100*sum(vals["validation"])/5:.3f}% |')
lines+=['','选定C（模型301→305）：','']
for name in ['mean_clean','mean_aug','worst_clean','worst_aug']:
 lines.append(f'- {name}: '+', '.join(str(f['heads'][name]['C']) for f in fits.values()))
lines+=['','## 审计与复现','','- 100候选的目标、组损失、梯度、自适应权重及最大验证损失选参均独立复核。最大梯度∞范数'+f'{json.loads((ROOT/"training_audit.json").read_text())["max_gradient_inf"]:.3g}'+'，低于1e-6。','- 拟合／验证各5,000张图像；冻结20头后生成20个测试库各1,000张。本轮30,000张相互无像素重复，并与B至B-C的190,000张及A显式库4,000张核查无重复；不涵盖未留存在线图像。','- 完成1,100个模型×库×时长单元，独立重建全部噪声重复和续算，核对57,200组线性头预测、原任务头及负对照。审计复用模型定义，不是完全独立的软件实现。','- 专用参照及原任务头正对照全部通过，最低区间下界'+f'{100*min(v["lower"] for v in r["positive_controls"].values()):.3f}%'+'。负对照均值范围'+f'{100*r["negative_range"][0]:.3f}%–{100*r["negative_range"][1]:.3f}%'+'，在47%–53%校准门槛内；这不是等效检验。','- 55项实现／回归测试通过；26项终点另行逐单元重算区间并交叉核对。最初两处测试引用旧分支别名的问题已在生成数据前修复并保留失败日志，不是科学拟合失败。','- 训练、冻结、测试顺序，核心与读出冻结，历史科学产物完整性全部通过。科学执行代码自归档起未改变；最多两个计算进程，每进程一线程。','','## 解释边界与停止','','本结论条件化于固定五核心和本次拟合／验证数据，不覆盖新核心或重拟合不确定性。所有测试时长均为历史已知条件；6、9、14步仅未参与本轮拟合，不能称为从未参与方法开发的新时长。未截断的t区间在天花板处退化或越界不证明总体无误差。','','按预定支持／排除判据结案，不据结果追加温度或C调参。具体有限温度流程的结果不证明共享线性边界不存在，也不排除其他最差损失目标。专用参照有多套参数与时长条件资源，不能当成资源相同的方法。','','## 材料','','- [执行前协议](../../artifacts/branch_bw/round1/protocol.md)、[完整数值](FULL_RESULTS.md)、[停止决定](../../docs/decisions/branch_bw/BW001_STOP.md)','- [机器汇总](../../artifacts/branch_bw/round1/summary.json)、[训练审计](../../artifacts/branch_bw/round1/training_audit.json)、[预测审计](../../artifacts/branch_bw/round1/independent_audit.json)、[数据与顺序审计](../../artifacts/branch_bw/round1/data_chronology_integrity.json)','- [哈希清单](../../artifacts/branch_bw/reproducibility_manifest.json)','','按prepare→train(301–305)→training审计→lock→prepare_test→evaluate(301–305)→evaluation及integrity审计→analyze→crosscheck→report→finalize顺序复现；使用独立空输出目录，避免覆盖本轮归档。']
# Interpretation added only after the complete, audited primary results.
assert r['decision']=='excluded' and stats['excluded']==['gain_d15_clean']
assert ab['decision']=='supported' and len(stats['supported'])==25 and not stats['unresolved']
lines[0]='# B-W结案：全时长平均功能过线，但15步无噪增益未达预定门槛'
lines.insert(4,'**主要发现：固定五模型平均的22项绝对功能全部获得>85%的支持；14步无噪／有噪和15步有噪三项增益超过2个百分点。15步无噪增益为1.883个百分点，校正区间[1.787, 1.979]，未达到预定>2个百分点要求，因此严格合取假设仍被排除。**')
interpret=['## 结果意味着什么、下一步优先问题','','目标改变明显调整了跨时长权衡：14、15步均值功能改善，但8、9步无噪均值分别下降4.750、4.890个百分点；本轮85%绝对门槛不要求所有位置不降分。15步训练／验证混合BA从83.641%／83.749%提高到86.362%／86.430%，提示改善已出现在拟合与验证分布上；这不能排除C选择变化的贡献。','','全时长过线是固定五模型平均的结论。五个模型中四个仍有低于85%的条件，最低观测均值分别约76.15%、77.94%、85.74%、79.77%、78.04%。不能称为每个模型都获得通用读出。','','后续优先保持当前目标、温度、覆盖与选参规则，另行预注册独立拟合／验证数据的重复实验，检查平均功能改善能否重现，以及各模型薄弱时长是否稳定。应把跨重拟合稳定性与单模型最差条件分别列为新问题，不能沿用本轮跨测试库区间来代表重拟合不确定性。更换核心的外部验证可作为随后独立问题。当前不为追过2个百分点门槛追加温度搜索，也不同时加入14步；这些均会开启不同干预。此处为下一问题建议，尚未执行。','']
idx=lines.index('## 材料');lines[idx:idx]=interpret
diag=['## 最差目标的实际梯度权重','','训练的基础时长权重仍相等；下表是选定worst_aug最终损失对应的softmax梯度权重，顺序与拟合时长一致。不是额外手工加权。','','| 模型 | 5 | 7 | 8 | 10 | 11 | 12 | 13 | 15 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for seed,f in fits.items():
 h=f['heads']['worst_aug'];row=next(x for x in h['grid'] if x['C']==h['C'])
 diag.append(f'| {seed} | '+' | '.join(f'{v:.3f}' for v in row['history'][-1]['group_weights'])+' |')
diag+=['']
idx=lines.index('## 审计与复现');lines[idx:idx]=diag
(out/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
stop=Path('docs/decisions/branch_bw/BW001_STOP.md');stop.parent.mkdir(parents=True,exist_ok=True)
stop.write_text(f'# BW001：按预定26项合取判据结案\n\nH-BW1{verdict}。全部质量门槛通过；主要判据：{stats}。\n\n绝对功能：{ab}。\n\n配对增益：{gain}。\n\n结果限定于本轮固定覆盖、温度与选参流程，不外推所有共享线性读出。完整报告见[最终报告](../../../reports/branch_bw/FINAL_REPORT.md)。\n')
p=Path('docs/branches/BW_WORST_DELAY_OBJECTIVE.md');s=p.read_text().replace('**执行前协议：尚未生成本分支数据、拟合或观察结果。**',f'**首轮完成：H-BW1{verdict}，按预定判据结案。**\n\n[最终报告](../../reports/branch_bw/FINAL_REPORT.md)。下文保留原设计；执行前原文另存artifacts/branch_bw/round1/protocol.md。');p.write_text(s)
p=Path('NEXT_EXPERIMENT_PLAN.md');s=p.read_text().replace('**协议已固定，开始执行。**',f'**B-W已完成，H-BW1{verdict}，已结案。**\n\n[最终报告](reports/branch_bw/FINAL_REPORT.md)。以下为本轮设计摘要。');p.write_text(s)
print('B-W final report:',r['decision'])
