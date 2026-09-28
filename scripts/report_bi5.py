"""Write a human-readable report from completed R5 audited evidence."""
from pathlib import Path
import json
R=Path('artifacts/branch_bi/round5')
s=json.loads((R/'round_summary.json').read_text());a=json.loads((R/'independent_audit.json').read_text());g=s['gate']
assert s['quality_passed'] and s['aggregation_crosschecked'] and a['passed']
f=lambda v:f'{100*v:.6f}%'
lines=['# B-I 第五轮：完整未决条件族的证据综合','',f"完整查看点：{s['n_banks']}个共同测试库；第三轮同一96次独立拟合，无新拟合。",'',f"核心301必要条件：S={g['S']}，F={g['F']}，U={g['U']}；q区间[{g['lower']:.12f}, {g['upper']:.12f}]。原五核心成功概率p≤q，故p上界为{s['main_p_upper']:.12f}；p下界仍为0。",'',('原H-BI（p≥80%）已在登记假设和误差控制下被限定排除，满足约定停止规则。' if s['decision']=='excluded' else '原H-BI仍未决；本查看点完成本身不是停止依据，须根据完整结果继续规划。'),'', '## 数据与推断边界','',f"复用第四轮已有2048库，并明确承认这些数据已有部分结果被查看。本轮新增图像{s['new_images_generated']:,}张；不把补测头、条件、查看点或噪声重复计为新拟合。",'',f"同时覆盖完整68条件×4固定查看点；测量错误额度{s['measurement_alpha']}，每条件/点{s['pointwise_alpha']:.12g}。取已完成查看点的最小上界，依据完整族联合界判定失败。沿用第三轮同一潜在96次成功计数的CP覆盖事件（alpha={s['reused_rate_alpha']}），新增测量分类形成该区间的外包区间，不重复花费率误差。详见第五轮协议中的证明。",'', '原五核心事件蕴含核心301事件，因此必要条件失败可排除原假设；必要条件支持不能支持原五核心假设。结论仅针对固定核心、生成器、训练流程及误差模型。单项库级t覆盖是近似的；拟合近似IID、全局去重的有限依赖和伪随机实现仍是限制。此结论不推翻B-H某一固定拟合成功的存在性结论。','', '## 完整核验','',f"独立审计通过：{a['cells']:,}个时长单元，{a['prediction_vectors']:,}个完整候选/参考预测向量；从像素独立重建状态、全部噪声、reset及完整置换。原始Probe切片交叉核验{a['original_probe_slice_checks']:,}次。与第四轮重叠的{s['old_overlap_prediction_vectors']:,}个预测向量逐位相同。",'',f"全部正对照99%区间下界超过90%；负对照范围{f(s['negative_range'][0])}–{f(s['negative_range'][1])}。全部测量区间经第二套计算核对，Beta端点经二项CDF恒等式复核。",'',f"新增确定失败批次：{', '.join(s['newly_certified_heads']) or '无'}。批次按集合合并，重复条件不重复计数。",'', '## 全部68个条件','', '| 批次 | 时长 | 输入 | 当前均值 | 最优上界 | 使用库数 | 失败证书 |','|---|---:|---|---:|---:|---:|---|']
for e in s['target_evidence']:
 v=e['bounds_by_n'][str(s['n_banks'])]
 lines.append(f"| {e['batch']} | {e['delay']} | {e['input']} | {f(v['mean'])} | {f(e['best_upper'])} | {e['best_n']} | {'是' if e['certified_failure'] else '否'} |")
lines+=['','协议：`docs/decisions/branch_bi/ROUND5_PLAN.md`；完整数值：`artifacts/branch_bi/round5/round_summary.json`；各查看点证据清单：`look*/evidence_manifest.json`。','']
p=Path('reports/branch_bi/ROUND5_REPORT.md');p.write_text('\n'.join(lines));print(p)
