"""Report only completed, audited precision looks."""
import json
from pathlib import Path
R=Path('artifacts/branch_bi/round4');O=Path('reports/branch_bi')
def main():
 s=json.loads((R/'round_summary.json').read_text());assert s['quality_passed'] and s['aggregation_crosschecked'];g=s['gate'];cfg=json.loads((R/'config.json').read_text());lines=['# B-I 第四轮：固定次序精度检验','',f"已完成{s['look']}个预登记查看点，累计{s['n_banks']}个新测试库，无新拟合，K仍96。S={g['S']} / F={g['F']} / U={g['U']}；核心301必要条件q区间[{g['lower']:.6%}, {g['upper']:.6%}]，原p≤q，因此p上界为{s['main_p_upper']:.6%}。",'',f"主判定：**{s['decision']}**；后续动作：`{s['next_action']}`。12个固定靶点中前{s['certified_prefix']}项已获得失败证书。q下界不能当作原p下界。",'','## 每阶段结果','', '| 查看点 | 累计新库 | 已认证前缀 | 确定失败总数 | q上界 | 下一动作 |','|---|---:|---:|---:|---:|---|']
 for j in range(1,s['look']+1):
  x=json.loads((R/f'stage{j}/round_summary.json').read_text());lines.append(f"| {j} | {x['n_banks']} | {x['certified_prefix']} | {x['gate']['F']} | {x['gate']['upper']:.6%} | {x['next_action']} |")
 lines+=['','## 冻结靶点与当前测量','', '单项上界是各固定查看点的单侧t上界，不是12项联合区间；只有按固定次序获得的连续前缀可作为确认失败。旧阶段已获证书保留，后续某个单项上界变化不撤销原证书。','', '| 顺序 | 原批次／条件 | 当前BA均值 | 当前单侧上界 | 已认证 |','|---|---|---:|---:|---|']
 for i,x in enumerate(s['target_evidence']):lines.append(f"| {i+1} | {x['batch']} / {x['condition']} | {x['mean']:.6%} | {x['upper']:.6%} | {'是' if i<s['certified_prefix'] else '否'} |")
 a=json.loads((R/'independent_audit.json').read_text());lines+=['','## 证据与限制','',f"本轮实际measurement额度已花费{s['measurement_alpha_spent']:.8f}，rate额度{cfg['rate_alpha']:.8f}；四轮总额度最大.046875。固定序列中首个真实零假设若被错误认证，必在某个预登记查看点被错误拒绝；对查看点求和控制本轮任意错误前缀。固定96次拟合的潜在成功总数只使用同一个CP覆盖事件，重复查看不增加拟合样本数。详见协议中的完整证明。",'',f"已完成{a['cells']}个新增科学单元和{a['prediction_vectors']}个候选／参考预测向量的全量独立重建。原头、reset、全部噪声和64次置换完整复算，全部正负对照通过。原模型及头冻结不变，原图、几何、预测及完整置换均可无损恢复。累计留存图像{cfg['prior_images']+s['n_banks']*1000}张。",'',f"用于决定是否改换精度路线的诊断99%区间出现反向迹象的靶点：{s['diagnostic_futility_targets']}。这些区间不作联合确认主张，不能据此宣布原假设得到支持或分支科学停止。",'', '库级t覆盖仍是近似，拟合IID和固定生成机制是解释前提；全局拒绝去重有有限依赖，伪随机实现亦有限。不外推新核心、其他任务或训练算法。旧S/F与本轮有序证书合并属于明确登记的两层误差控制，不把第三轮旧测试数值混入本轮新均值。','', '协议：`docs/decisions/branch_bi/ROUND4_PLAN.md`；各阶段决策、原始输出、审计、统计与证据清单：`artifacts/branch_bi/round4/stage*/`。','']
 ledger={};previous=0
 from branch_b_round1 import sha
 for j in range(1,s['look']+1):
  source=R/f'stage{j}/round_summary.json';look=json.loads(source.read_text());assert look['previous_prefix']==previous
  for i in range(previous,look['certified_prefix']):
   evidence=look['target_evidence'][i];assert evidence['upper']<.85
   ledger[evidence['head']]=dict(first_certified_look=j,condition=evidence['condition'],upper=evidence['upper'],alpha=evidence['alpha'],n_banks=evidence['n_banks'],source=str(source),sha256=sha(source),fixed_sequence_rank=i+1)
  previous=look['certified_prefix']
 (R/'first_certification_ledger.json').write_text(json.dumps(dict(certificates=ledger,meaning='First successful registered fixed-sequence certificate; later summaries may retain it'),indent=2)+'\n')
 O.mkdir(exist_ok=True);(O/'ROUND4_REPORT.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
