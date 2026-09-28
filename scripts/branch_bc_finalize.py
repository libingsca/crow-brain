"""Write B-C decision report after all preregistered checks are complete."""
from pathlib import Path
import json
from branch_bc_round1 import ROOT,DELAYS,TRAIN_DELAYS,HEADS
r=json.loads((ROOT/'summary.json').read_text());assert r['quality_passed'] and r['decision'] in ['supported','excluded']
for name in ['independent_audit','data_chronology_integrity','summary_crosscheck']:assert json.loads((ROOT/f'{name}.json').read_text())['passed']
out=Path('reports/branch_bc');decision='获支持' if r['decision']=='supported' else '被限定排除'
stats=r['primary_classification'];ab=r['absolute_classification'];gain=r['gain_classification'];panel=r['panel_all_cells']
lines=[f'# B-C结案：训练覆盖修复假设{decision}','','2026-09-25。**H-BC1'+decision+'，全部质量门槛通过，首轮后结案。**','',f'28项预定要求中，{len(stats["supported"])}项支持、{len(stats["excluded"])}项排除、{len(stats["unresolved"])}项未决。绝对功能22项：{len(ab["supported"])}支持／{len(ab["excluded"])}排除／{len(ab["unresolved"])}未决；目标位置配对增益6项：{len(gain["supported"])}支持／{len(gain["excluded"])}排除／{len(gain["unresolved"])}未决。','','## 设计与归因边界','','固定五核心，用同批新图像比较五时长基线与加入8、11、12步后的八时长覆盖。四个共享头使用同一个scaler，仅由五个base时长的干净拟合状态计算；每原试次总权重1。四方法均用共同八时长验证规则选C。','', '干预包含训练时长支持与各时长权重重分配。模型301两个base头选C=10，其余18头选C=100，均在预定网格内；故结论针对整个预定流程，不证明固定C下同样成立。base采用共同八时长验证，是本轮匹配基线，不是历史B-S逐字复制。','','成功要求：expanded_aug在全部5–15步干净／有噪22条件BA均>85%，并且三个目标时长六条件相对base_aug均提升>2个百分点。85%是绝对功能保持，不代表其他位置不降分。所有28项区间为99.82142857%双侧t区间，df=19，Bonferroni本轮合计5%错误额度。','','## 全时长功能与配对变化','','BA为百分比；每库先在模型内平均10次噪声，再平均五模型；干净评价一次，以20个库为推断单位。','','| 时长 | 角色 | 基线无噪 | 扩展无噪 | 基线有噪 | 扩展有噪 | 扩展−基线，无噪／有噪（百分点） |','|---|---|---:|---:|---:|---:|---:|']
for di,d in enumerate(DELAYS):
 vals=[100*panel[f'{h}_d{d}_{c}']['mean'] for c in ['clean','noise'] for h in ['base_aug','expanded_aug']]
 role='新增拟合' if d in [8,11,12] else '原拟合' if d in [5,7,10,13,15] else '未拟合的已知时长'
 delta=r['paired_expanded_aug_minus_base_aug'][di]
 lines.append(f'| {d} | {role} | '+' | '.join(f'{v:.3f}%' for v in vals)+f' | {100*delta[0]:+.3f}／{100*delta[1]:+.3f} |')
lines+=['','![训练覆盖干预](coverage_intervention.png)','','## 三个目标位置的预定改善判据','','| 指标 | 配对增益（百分点） | 校正区间 | >2个百分点判据 |','|---|---:|---|---|']
for name,v in r['gains'].items():
 state='支持' if v['lower']>.02 else '排除' if v['upper']<.02 else '未决'
 lines.append(f'| {name} | {100*v["mean"]:+.3f} | [{100*v["lower"]:.3f}, {100*v["upper"]:.3f}] | {state} |')
lines+=['','## 未满足或未决的完整要求','']
if not stats['excluded'] and not stats['unresolved']:lines+=['全部28项下界超过各自门槛，无失败或未决项。']
else:
 lines+=['| 指标 | 均值 | 校正区间 | 门槛 | 裁决 |','|---|---:|---|---:|---|']
 for name in stats['excluded']+stats['unresolved']:
  v=r['primary'][name];state='排除' if name in stats['excluded'] else '未决'
  lines.append(f'| {name} | {100*v["mean"]:.3f} | [{100*v["lower"]:.3f}, {100*v["upper"]:.3f}] | {100*v["threshold"]:.1f} | {state} |')
lines+=['','绝对指标单位是百分比，增益单位是百分点。任一要求上界低于门槛即可排除合取假设；其他条件的成功不会被抹去。未达到2个百分点也不等于没有任何改善。全部28项区间和440个单模型成绩见[完整结果](FULL_RESULTS.md)。','','## 个体、参照与诊断','','各模型expanded_aug在22条件中的最低观测均值：'+'；'.join(f'{s}：{100*v:.3f}%' for s,v in r['per_model_expanded_aug_minimum'].items())+'。集合判据不保证每个个体均达到85%。','','四种共享头及八个专用增强参照完整保留；专用参照的正对照全部通过，具有多套参数和时长条件资源，不是资源相同的竞争方法。共享干净头为次要参照，不把其与增强头的差异当作增强必要性或等效性结论。每候选逐时长训练／验证损失与BA保留，可用于决定下一项可辨识干预，不在本轮据此更改目标或扩网格。','','## 执行与审计','','- 20新头、100候选全部完成，验证损失、选参、原试次／时长／噪声权重、共同scaler与梯度均独立重算。最大梯度∞范数为'+f'{json.loads((ROOT/"training_audit.json").read_text())["max_gradient_inf"]:.3g}'+'，低于1e-6。','- 拟合／验证各5,000张图像；读出冻结后才生成20个测试库各1,000张。本轮30,000张相互无像素重复，并与B至B-L的160,000张历史图像及A显式库4,000张核查无重复；不涵盖未留存在线训练图像。','- 1,100个模型×库×时长单元全部完成。独立重建全部噪声重复和test续算，复核57,200组线性头预测、原任务头、清零、64次标签置换与64次分层状态错配。审计复用模型定义，不是完全独立软件实现。','- 八个专用参照及原任务头5步的正对照全部通过，下界最低'+f'{100*min(v["lower"] for v in r["positive_controls"].values()):.3f}%'+'；负对照均值范围'+f'{100*r["negative_range"][0]:.3f}%–{100*r["negative_range"][1]:.3f}%'+'，在47%–53%流程门槛内。这不是等效性证明。','- 42项实现／回归测试通过；28项统计终点另用结果单元与独立区间计算交叉核对。训练先于冻结和测试的顺序、头及核心冻结、历史科学产物完整性通过。当前规划入口的合法更新单独归档，不重写历史清单。','- 每计算进程CPU一线程，最多两个并行。科学条件、数据量与总权重未因资源改变。所有候选、源码、环境、日志与原始预测归档。','','## 结论边界与停止','','推断只针对既有五核心和本次拟合／验证数据；不覆盖新核心或重新拟合的不确定性。5–15步已在历史中观察过，本轮是已知时长的新图像确认；6、9、14步未用于本轮拟合／验证，也不能据此称为从未参与方法开发的新时长。原始t区间未截断，天花板处退化或越界不证明总体无误差。','','依据预定合取判据，H-BC1'+decision+'，本分支结案。停止不是由耗时、预算、轮数或暂时不显著触发；不将具体流程的失败扩大为共享线性边界数学上不存在。后续应根据本轮功能与逐时长训练／验证诊断另立问题，本次不追加调参以改变结论。','','## 复现材料','','- [执行前协议](../../artifacts/branch_bc/round1/protocol.md)、[停止决定](../../docs/decisions/branch_bc/BC001_STOP.md)、[完整数值](FULL_RESULTS.md)','- [机器汇总](../../artifacts/branch_bc/round1/summary.json)、[训练审计](../../artifacts/branch_bc/round1/training_audit.json)、[预测审计](../../artifacts/branch_bc/round1/independent_audit.json)、[数据顺序完整性](../../artifacts/branch_bc/round1/data_chronology_integrity.json)','- [最终哈希清单](../../artifacts/branch_bc/reproducibility_manifest.json)','','重算现存汇总与图表：scripts/branch_bc_analyze.py、scripts/branch_bc_crosscheck.py、scripts/branch_bc_report.py。完整复制使用独立空输出目录，按prepare→train(301–305)→training审计→lock→prepare_test→evaluate(301–305)→evaluation及integrity审计→analyze/crosscheck/report执行，避免覆盖归档。']
# Observed B-C interpretation, derived only after complete audited results.
assert r['decision']=='excluded' and set(stats['excluded'])=={'absolute_d14_clean','absolute_d14_noise','absolute_d15_clean','absolute_d15_noise'}
assert gain['decision']=='supported'
lines[0]='# B-C结案：目标位置改善，但14、15步出现功能缺口'
lines.insert(4,'**主要发现：增加覆盖使三个目标位置的六项增益全部超过2个百分点，但14、15步的干净／有噪四项区间上界均低于85%。这套流程把部分功能缺口转移到了后段时长，没有完成全部时长修复。**')
idx=lines.index('## 结论边界与停止')
diag=['## 对下一步有用的诊断','','15步已包含在扩展训练中，不能将其失败仅解释为未拟合时长的插值不足。选定增强头的训练／验证混合BA如下（干净／噪声各半；描述性诊断，不是新增确认性检验）。','','| 方法 | 15步训练BA | 15步验证BA |','|---|---:|---:|']
for name in ['base_aug','expanded_aug']:
 values={part:[] for part in ['fit','validation']}
 for seed in range(301,306):
  fit=json.loads((ROOT/f'seed{seed}/fits.json').read_text())['heads'][name]
  row=next(x for x in fit['grid'] if x['C']==fit['C'])
  for part in values:values[part].append(row['per_delay'][part]['15']['ba'])
 diag.append(f'| {name} | {100*sum(values["fit"])/5:.3f}% | {100*sum(values["validation"])/5:.3f}% |')
diag+=['','扩展头的15步训练与验证表现已约84%，且专用15步参照正常。因此下一步优先考虑固定当前八时长覆盖、共同scaler和噪声定义，对比平均损失与更关注最差时长的目标，检验当前目标的权衡是否限制了功能。该方向尚未执行，失败也可能涉及正则化或共享线性表达限制。若同时把14步加入训练，会混合覆盖和目标两项变化，应单独设臂，不把效果单独归因于目标。','','本轮匹配基线在11步已经超过85%，不应把三个历史失败位置都称为本轮基线失败；六项配对增益仍按原要求全部检验。','']
lines[idx:idx]=diag
(out/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
Path('docs/decisions/branch_bc/BC001_STOP.md').write_text(f'''# BC001：按预定28项合取判据结案

2026-09-25。H-BC1{decision}。质量全部通过；{len(stats['supported'])}项支持、{len(stats['excluded'])}项排除、{len(stats['unresolved'])}项未决。

绝对功能：{ab}。

目标位置增益：{gain}。

完整1,100单元、20头／100候选、57,200组预测、42项测试及28项区间交叉核对通过。以明确支持／排除判据结案，不继续调参，不把具体流程结果推广为所有共享线性读出的可能性结论。完整报告见reports/branch_bc/FINAL_REPORT.md。
''')
p=Path('docs/branches/BC_TRAINING_COVERAGE.md');s=p.read_text().replace('**执行前协议：尚未生成数据、拟合或观察本分支结果。**',f'**首轮完成：H-BC1{decision}，按预定判据结案。**\n\n[最终报告](../../reports/branch_bc/FINAL_REPORT.md)。下文保留原规划；执行前原文独立归档在artifacts/branch_bc/round1/protocol.md。');p.write_text(s)
p=Path('NEXT_EXPERIMENT_PLAN.md');s=p.read_text().replace('**已制定协议，开始执行。**',f'**B-C首轮已完成，H-BC1{decision}，已结案。**\n\n[最终报告](reports/branch_bc/FINAL_REPORT.md)。以下保留执行前设计摘要；未启动后续修复。');p.write_text(s)
print('B-C final report and stopping decision:',r['decision'])
