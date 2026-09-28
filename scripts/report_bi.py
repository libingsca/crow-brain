"""Post-hoc descriptive reporting; does not alter fits, tests, or decisions."""
import json,pickle
from pathlib import Path
import numpy as np
from scipy.ndimage import label
from branch_b_round1 import sha,save
R=Path('artifacts/branch_bi/round1');OUT=Path('reports/branch_bi');OUT.mkdir(parents=True,exist_ok=True)
def diagnostics():
 records={};directions={str(s):[] for s in range(301,306)};checked=0
 for b in range(1,7):
  root=R/f'batch{b}'
  for part,n in [('fit',8000),('validation',2000)]:
   with np.load(root/f'{part}.npz') as d:
    assert d['images'].shape==(5,1000,1,32,32)
    np.testing.assert_array_equal([label(x[0]>.499,structure=np.ones((3,3)))[1] for x in d['images'].reshape(-1,1,32,32)],np.repeat(range(2,7),1000))
    assert np.bincount(d['y']).tolist()==[n//2]*2;np.testing.assert_array_equal(d['y'],d['a']==d['b']);assert not np.any((d['a']==d['b'])&(d['ia']==d['ib']))
    for a in range(5):
     for c in range(5):assert np.sum((d['a']==a)&(d['b']==c))==n//40*(4 if a==c else 1)
    checked+=1
  for seed in range(301,306):
   p=root/f'seed{seed}';f=json.loads((p/'fits.json').read_text())['heads']['eleven'];last=f['history'][-1]
   with (p/'heads.pkl').open('rb') as fh:h=pickle.load(fh)['eleven']
   beta=h.clf.coef_[0];raw=beta/h.scaler.scale_;directions[str(seed)].append(raw/np.linalg.norm(raw))
   records[f'{b}_{seed}']=dict(standardized_coefficient_norm=float(np.linalg.norm(beta)),raw_state_coefficient_norm=float(np.linalg.norm(raw)),raw_state_intercept=float(h.clf.intercept_[0]-raw@h.scaler.mean_),training_group_losses=last['group_losses'],training_group_weights=last['group_weights'],validation=f['validation'],gradient_inf=last['gradient_inf'])
 cos={s:(np.array(v)@np.array(v).T).tolist() for s,v in directions.items()};save(R/'descriptive_diagnostics.json',dict(fits=records,raw_state_direction_cosines=cos,fit_validation_banks_independently_checked=checked,interpretation='Descriptive only; raw state coefficients remove batch-specific standardization.'))
 return records,cos

def main():
 result=json.loads((R/'round_summary.json').read_text());records,cos=diagnostics();lines=['# B-I 首轮结果','',f"独立重拟合批次 K={result['K']}；确定成功 S={result['S']}、确定失败 F={result['F']}、未决 U={result['U']}。",'',f"全条件成功概率 p 的保守区间为 [{result['lower']:.6%}, {result['upper']:.6%}]；本轮判定：**{result['decision']}**。确定成功比例 {result['certified_fraction']:.2%}，可能成功比例 {result['possible_fraction']:.2%}，均不能直接当作真实成功率点估计。",'', '| 批次 | 分类 | 达标条件数 | 排除条件数 | 未决条件数 | 最低 BA 点估计 | 最低区间下界 |','|---|---|---:|---:|---:|---:|---:|']
 for b in range(1,7):
  x=result['batches'][str(b)];ss=json.loads((R/f'batch{b}/summary.json').read_text());low=min(v['lower'] for v in ss['primary'].values());lines.append(f"| {b} | {x['state']} | {x['supported_conditions']} | {len(x['excluded'])} | {len(x['unresolved'])} | {x['min_mean']:.4%} | {low:.4%} |")
 lines+=['','## 失败和未决条件','']
 for b in range(1,7):
  s=json.loads((R/f'batch{b}/summary.json').read_text());lines.append(f'### 批次 {b}');lines+=['','| 条件 | BA | 区间下界 | 区间上界 | 判定 |','|---|---:|---:|---:|---|']
  for status in ['excluded','unresolved']:
   for k in s[status]:
    v=s['primary'][k];lines.append(f"| {k} | {v['mean']:.5%} | {v['lower']:.5%} | {v['upper']:.5%} | {status} |")
  lines.append('')
 lines+=['## 拟合诊断','', '方向相似性在去除各批标准化后、同一核心的原始状态坐标中计算；不是机制检验。完整22组训练与验证损失、系数范数及相似性矩阵见工件 descriptive_diagnostics.json。','', '| 核心 | 原始系数范数范围 | 跨批方向余弦范围 |','|---|---:|---:|']
 for seed in range(301,306):
  norms=[records[f'{b}_{seed}']['raw_state_coefficient_norm'] for b in range(1,7)];c=np.array(cos[str(seed)])[np.triu_indices(6,1)];lines.append(f'| {seed} | {min(norms):.3f}–{max(norms):.3f} | {c.min():.6f}–{c.max():.6f} |')
 lines+=['','## 质量、范围与限制','', '30个头全部审计冻结后才生成测试图像；6批×40库×5核心×11时长共13200个科学单元和250800个候选/参考预测向量，均从像素独立重建，包括所有噪声、原头、清零及置换对照。660个条件区间与批次概率界另行复算。正对照每批18项99%区间下界均>90%，负对照均在47–53%。','', '首轮测量与概率层分别使用alpha=.0125；后续轮如有则按既定alpha花费。条件测量使用库级t近似，批次级使用对未决保守的Clopper–Pearson界。覆盖依赖于近似有效性及固定生成机制下批次独立同分布；图像拒绝去重带来有限依赖，不声称无条件精确保证。结论仅适用于固定五核心与固定读出拟合流程，不外推新核心总体。','', '共新增300000张图像，留存图像总数914000。历史工件保留；测试种子随机流冲突在任何新拟合和测试前修订为1140100+1000*(批次−1)起始，原协议与修订并存。3个拟合进程因内存不足终止后按原种子和原数据串行重跑，不替换种子或计作科学失败。','']
 (OUT/'ROUND1_REPORT.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
