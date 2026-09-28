"""Descriptive report from completed necessary-gate experiment."""
import json,pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from branch_b_round1 import save
R=Path('artifacts/branch_bi/round3');OUT=Path('reports/branch_bi');OUT.mkdir(exist_ok=True)
def main():
 s=json.loads((R/'round_summary.json').read_text());assert s['quality_passed'] and s['aggregation_crosschecked'];g=s['gate'];K=g['K'];fit={};directions=[]
 for b in range(1,K+1):
  p=R/f'batch{b}/seed301'
  with (p/'heads.pkl').open('rb') as f:h=pickle.load(f)['eleven']
  r=json.loads((p/'fits.json').read_text())['heads']['eleven'];raw=h.clf.coef_[0]/h.scaler.scale_;directions.append(raw/np.linalg.norm(raw));fit[str(b)]=dict(raw_norm=float(np.linalg.norm(raw)),standardized_norm=float(np.linalg.norm(h.clf.coef_)),training_group_losses=r['history'][-1]['group_losses'],validation=r['validation'],gradient_inf=r['history'][-1]['gradient_inf'])
 cosine=np.array(directions)@np.array(directions).T;off=cosine[np.triu_indices(K,1)];save(R/'descriptive_diagnostics.json',dict(fits=fit,cosine_min=float(off.min()),cosine_median=float(np.median(off)),cosine_max=float(off.max()),interpretation='Descriptive only; fixed core 301 raw-state coordinates'))
 np.savez_compressed(R/'direction_cosines.npz',cosine=cosine)
 lines=['# B-I 第三轮：核心301必要条件可靠性','',f"96套全新独立拟合数据，640个共同新测量库。必要条件分类 S={g['S']} / F={g['F']} / U={g['U']}。",'',f"q的保守区间为 **[{g['lower']:.6%}, {g['upper']:.6%}]**。由原五核心事件E蕴含301的22条件事件A，p≤q，故本轮给原p的上界为 **{s['main_p_upper']:.6%}**。原假设判定：**{s['decision']}**。q的下界不能作为p的下界。",'', '旧六批没有并入本轮二项区间。共同测试库的跨头测量相关由联合覆盖处理，独立抽样单位仍是训练数据产生的拟合。2112项双侧库级t区间共享measurement alpha=.003125，概率层alpha=.003125；前三轮总额度≤.04375。t覆盖近似、图像拒绝去重的有限依赖以及伪随机实现限制不变，不宣称无条件精确覆盖或外推新核心总体。','', '![各独立拟合的最差条件与联合区间](/workspace/CrowBrain/reports/branch_bi/round3_minimum_ba.png)','', '图中区间由每头22项区间的最小下界与最小上界构成，在原联合覆盖事件上覆盖该头的真实最差BA；点为最小条件点估计，不能当作无偏估计。','', '| 批次 | 必要门槛分类 | 支持项 | 排除项 | 未决项 | 最低BA点估计 |','|---|---|---:|---:|---:|---:|']
 for b,x in s['batches'].items():lines.append(f"| {b} | {x['state']} | {len(x['supported'])} | {len(x['excluded'])} | {len(x['unresolved'])} | {x['minimum_point']:.5%} |")
 lines+=['','## 失败与未决位置','', '| 条件 | 排除批次数 | 未决批次数 |','|---|---:|---:|']
 for d in range(5,16):
  for c in ['clean','noise']:
   key=f'301_d{d}_{c}';f=sum(key in x['excluded'] for x in s['batches'].values());u=sum(key in x['unresolved'] for x in s['batches'].values())
   if f or u:lines.append(f'| {key} | {f} | {u} |')
 audit=json.loads((R/'independent_audit.json').read_text());from branch_bi_codec import load_gzip_json
 evaluation=load_gzip_json(R/'seed301/results.json.gz')
 fa=json.loads((R/'fit_data_manifest.json').read_text());te=json.loads((R/'data_manifest.json').read_text());lines+=['','## 质量与复现','', '96头均按固定eleven方法拟合、审计后才生成新测试库。18项301正对照的99%区间下界均>90%，全部候选与参考负对照在47–53%。7040个科学单元、7490560个候选/参考预测向量，以及全部原头、reset与置换，均已独立重建；统计聚合另行复算。','', f"批量线性读出加速保留原float32缩放舍入语义，边界处回退原接口；完整旧实现快照的{evaluation['original_partial_prediction_vectors_compared']}个候选/参考预测向量逐位一致。审计首次因遗漏StandardScaler对参数的输入dtype转换而在边界标签处失败，错误日志和原实现完整保留；修订后全部新预测另以原标准化空间及原参数转换语义独立复算，并额外完成{audit['original_probe_slice_checks']}次原Probe接口固定切片核对；数值边界整头回退共{evaluation['fallback_head_calls']}次。原执行清单和训练锁保持字节不变，新增实现单独冻结，详见engineering_revision_03.md与engineering_revision_04.md。全部64次置换另用精确整数计数复核。",'', f"新图1600000张，累计留存3154000张；训练/验证拒绝重采样记录{len(fa['duplicate_rejections'])}项，测试记录{len(te['duplicate_rejections'])}项。几何表示与未改动原生成器逐图比较，解码像素逐字节一致；预测位编码与gzip不量化数值，置换完整整数数组以LZMA保存。",'', f"跨拟合原始状态读出方向余弦范围[{off.min():.6f}, {off.max():.6f}]，中位数{np.median(off):.6f}。范数、22组训练损失、验证诊断及全部优化轨迹均保留；这些描述不自动支持机制结论。",'']
 (OUT/'ROUND3_REPORT.md').write_text('\n'.join(lines))
 bounds=[]
 for b in range(1,K+1):
  vals=list(s['primary'][str(b)].values());bounds.append((min(x['mean'] for x in vals),min(x['lower'] for x in vals),min(x['upper'] for x in vals),s['batches'][str(b)]['state']))
 bounds.sort(key=lambda x:x[0]);fig,ax=plt.subplots(figsize=(14,6));colors=dict(S='#238b45',F='#cb181d',U='#e69500')
 for j,(m,lo,hi,st) in enumerate(bounds):ax.errorbar(j+1,m*100,yerr=[[100*(m-lo)],[100*(hi-m)]],fmt='o',ms=3,color=colors[st],elinewidth=.7,capsize=1)
 ax.axhline(85,color='black',ls='--',lw=1);ax.set(xlabel='Independent refits sorted by estimated minimum BA',ylabel='Minimum balanced accuracy (%)',title=f"Core 301 necessary gate: S={g['S']}, F={g['F']}, U={g['U']}\nq interval [{g['lower']:.4f}, {g['upper']:.4f}]; original p <= q")
 ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(OUT/'round3_minimum_ba.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
