import json
from pathlib import Path
R=Path('artifacts/branch_bi/round2');OUT=Path('reports/branch_bi');OUT.mkdir(exist_ok=True)
s=json.loads((R/'round_summary.json').read_text());lines=['# B-I 第二轮：冻结头精度补测','',f"640个共同新库，0个新拟合批次，K仍为6。联合分类 S={s['S']} / F={s['F']} / U={s['U']}；p保守区间 [{s['lower']:.6%}, {s['upper']:.6%}]；判定 **{s['decision']}**。",'', '首轮639个已支持条件保持其原联合区间，21个未决条件仅用本轮新库区间更新。测量相关以Bonferroni控制；共享库不增加独立拟合样本数。前两轮保守总错误额度≤.0375，库级t近似及去重有限依赖限制不变。','', '| 批次 | 分类 | 支持 | 排除 | 未决 |','|---|---|---:|---:|---:|']
for b,x in s['batches'].items():lines.append(f"| {b} | {x['state']} | {len(x['supported'])} | {len(x['excluded'])} | {len(x['unresolved'])} |")
lines+=['','## 预定21项新测量','', '| 批次 | 条件 | 首轮BA | 新BA | 新下界 | 新上界 |','|---|---|---:|---:|---:|---:|']
for b,items in s['new_intervals'].items():
 old=json.loads(Path(f'artifacts/branch_bi/round1/batch{b}/summary.json').read_text())['primary']
 for k,v in items.items():lines.append(f"| {b} | {k} | {old[k]['mean']:.5%} | {v['mean']:.5%} | {v['lower']:.5%} | {v['upper']:.5%} |")
lines+=['','## 审计与解释','', '六个头与原始参数逐项一致。3200个核心×库×时长单元、225280个候选/参考预测向量，以及原头、清零和两类置换均从像素独立重建。6项301专用正对照99%库级区间下界全部>90%，全部候选/参考负对照均值在47–53%。21项新测量与拼接后的660项另行聚合复核。','', '即便六批全部为S，此样本量仍不能支持p≥80%。后续应根据未决项比例和已分类情况，在测量精度和新独立拟合样本之间作出选择；本轮完成本身不是停止规则。','']
(OUT/'ROUND2_REPORT.md').write_text('\n'.join(lines))
