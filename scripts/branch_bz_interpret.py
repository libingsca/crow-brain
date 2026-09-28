"""Add interpretation only after completed and audited B-Z results."""
from pathlib import Path
import json
r=Path('artifacts/branch_bz');s=json.loads((r/'summary.json').read_text())
assert s['quality_passed'] and s['panel_classification']['decision']=='supported'
assert s['individual_classification']['supported']==['303'] and s['individual_classification']['excluded']==['301','302','304','305']
assert all(v['same_condition_ranks']==[1]*5 for v in s['stability'].values())
p=Path('reports/branch_bz/FINAL_REPORT.md');text=p.read_text()
text=text.replace('# B-Z：独立重拟合重复实验结果','# B-Z：平均功能可重复，四个模型的个体短板持续存在')
lead='''**核心发现：独立更换5组拟合／验证／测试数据后，五模型平均的22项校正区间下界仍全部>85%，最低下界85.582%。但四个模型的期望观测最差表现区间上界仍<85%，只有303支持>85%；27项中23项支持、4项排除、0项未决。五个模型在全部5次重复中都保留了B-W的同一个最差条件。**

'''
text=text.replace('## 27项确认性结果',lead+'## 27项确认性结果')
extra='''## 科学解释与下一问题

平均功能恢复并非仅见于B-W的一次拟合；在本轮固定核心、生成分布与流程下具有可重复性。与此同时，个体薄弱点并未随独立重拟合消失。最差条件位置25/25次与B-W一致是描述性稳定性证据，不证明对所有可能重拟合或新核心都成立。

四个失败模型的短板可分为两类：301／302的15步已经参与训练；304的9步与305的14步未参与本轮训练。下一项机制实验应分别区分已训练时长的损失权衡与未拟合时长的覆盖问题，不能把两类失败都归因于采样波动，也不能据此宣布共享线性读出数学上不可能。后续干预须单独预注册；本轮没有执行额外温度、覆盖或特征搜索。

15步无噪的描述性配对增益为2.165个百分点，跨批未校正95%区间[1.974, 2.357]仍包含2个百分点。该描述性结果不改变B-W按原协议作出的严格增益结论，也不充当本轮新增确认性通过。

## 实现修正与复现

审计历史图像计数时补充排除了当前分支路径；原始审计源码保留。自动调度器首次调用审计时误将虚拟环境入口解析为系统Python，因缺少numpy未开始审计计算；修正后所有审计通过。原失败日志保留。两处修正均未改变已固定的数据、方法、样本数与统计判据，详见[实现记录](../../artifacts/branch_bz/implementation_notes.md)。

原执行代码、补充审计与调度代码的哈希分别归档。完整复现需在独立空输出位置按prepare(批次1–5)→train(全部25模型批次)→finish_execution（独立训练审计、全局冻结、测试生成、评价与审计、统计、图表）→interpret→finalize运行，避免覆盖现有产物。5批原始数据、全部候选及预测、环境版本、日志与源码快照均保留。

'''
text=text.replace('## 停止和边界',extra+'## 停止和边界');p.write_text(text)
print('Audited interpretation added; no experimental changes')
