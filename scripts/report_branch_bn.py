"""Final fixed-panel BN1 report and standalone scientific figure."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('artifacts/branch_bn/round1');D=Path('reports/branch_bn');S=['301','302','305']
def main():
    r=json.loads((R/'summary.json').read_text());assert r['status']=='supported' and (R/'audit.json').exists()
    ms=r['per_model'];ints=r['primary_intervals']
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    for i,(s,m) in enumerate(ms.items()):
        color=plt.get_cmap('tab10')(i)
        ax[0].plot([0,1],[m['noise']['primary']['clean'],m['noise']['primary']['aug']],marker='o',ls='-' if s in S else '--',color=color,alpha=.8,label=s+(' (primary)' if s in S else ' (reference)'))
    ax[0].legend(loc='lower right',fontsize=7)
    ax[0].axhline(.8,color='black',ls='--',lw=1);ax[0].set(xticks=[0,1],xticklabels=['Clean-trained head','Noise-augmented head'],ylabel='Balanced accuracy',ylim=(.45,1.04),xlim=(-.1,1.18),title='1% noise: fixed cores, new linear heads')
    for name,label,color in [('clean','Clean-trained','tab:red'),('aug','Noise-augmented','tab:blue')]:
        ys=[np.mean([ms[s]['clean'][name] for s in S]),np.mean([ms[s]['noise']['primary'][name] for s in S]),np.mean([ms[s]['noise']['legacy5pct'][name] for s in S])]
        ax[1].plot([0,1,5],ys,'o-',label=label,color=color)
    ax[1].set(xticks=[0,1,5],xlabel='Noise SD (% of prior training 5-step state RMS)',ylabel='Balanced accuracy',ylim=(.45,1.04),title='Prespecified sensitive panel: 301, 302, 305');ax[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(D/'RESULTS.png',dpi=180)
    rows='\n'.join(f"| {s}{'（主集合）' if s in S else '（参照）'} | {m['noise']['primary']['old']*100:.2f}% | {m['noise']['primary']['clean']*100:.2f}% | {m['noise']['primary']['aug']*100:.2f}% | {(m['clean']['clean']-m['clean']['aug'])*100:+.3f} |" for s,m in ms.items())
    scales='\n'.join(f"| {s} | {m['scales']['primary']:.5f} | {m['scales']['rms15']:.5f} | {m['scales']['centered15']:.5f} | {m['noise']['rms15']['aug']*100:.2f}% |" for s,m in ms.items())
    mainrows='\n'.join(f"| {label} | {ints[k]['mean']*100:.4f} | [{ints[k]['low']*100:.4f}, {ints[k]['high']*100:.4f}] | {rule} |" for k,label,rule in [('aug_noisy','增强头带噪 BA（%）','下界 >80'),('noisy_gain','相对干净训练头的提升（百分点）','下界 >15'),('clean_loss','干净 BA 损失（百分点）','上界 <2')])
    five=json.loads((R/'five_step_reference.json').read_text());assert all(q['clean_mean']==1 and q['noisy_mean']==1 for q in five['models'])
    text=f'''# B-N 最终报告：噪声增强的线性读出足以修复既有敏感模型

2026-09-24。**首轮科学实验完成，限定 H-BN1 获支持，按预定条件结案。**包含一次数值求解修正；没有把优化失败记作科学反证，也没有启动新的科学假设轮次。

## 主要结论

对预先指定的三个敏感模型 301、302、305，保持视觉前端、归一化统计、输入层及循环核心不变，将线性头的训练状态改为 50% 干净、50% 带噪状态后，15 步延迟、原尺度 1% 噪声下的平均 BA 从 **58.1188% 提升到 97.5605%**。干净 BA 从 **98.8233% 变为 98.8125%**，平均损失仅 **0.0108 个百分点**。

| 预定主要终点 | 均值 | 98⅓% 双侧区间 | 支持条件 |
|---|---:|---:|---|
{mainrows}

三个条件全部通过。三条主要区间经 Bonferroni 对应合计至少 95% 覆盖，限于 t 区间相应假设。**独立单位是 20 个新图像库，推断对象是已经固定的三个模型及本次拟合的头。**区间不涵盖新模型种子、重新抽取的拟合库或不同训练方案；不能把很窄的区间解释成已经验证随机 RNN 群体的普遍效果。

## 全部五模型结果

没有根据本轮结果重新分组。下表保留旧头、新干净头及增强头的带噪表现；最后一列为新干净头减增强头的干净 BA，正值表示增强有损失。

| 模型 | 旧头带噪 BA | 新干净头带噪 BA | 增强头带噪 BA | 干净损失（百分点） |
|---|---:|---:|---:|---:|
{rows}

三个敏感模型分别均获明显改善；另两个原已较稳健的参照模型没有获得类似幅度的增益。新干净头没有普遍修复敏感模型，说明主要恢复不能归结为“重新拟合一次就好”。两主臂最终均选择 C=100，并共享干净拟合分区的标准化参数；本次实际选中的模型间，C 值和标准化尺度不是造成主效果的差别。

![读出训练条件与噪声敏感性](RESULTS.png)

## 设计、质量检查与数值修正

- 新拟合／验证各 5,000 张图像，20 个独立测试库各 1,000 张，共 **30,000 张**。数量、标签／数量对平衡、分区不重叠及不与 B 的四个图像库重叠均通过审计。
- 拟合 8,000 对、验证 2,000 对；每测试库 2,000 对，主噪声重复 10 次，先在模型内平均，再在固定敏感集合内平均。
- 每原拟合试次总权重均为 1。增强臂一个合并的干净状态权重 0.5，四个独立噪声状态各 0.125；干净臂重复样本合并为权重 1。单元测试验证合并前后的加权正则化目标和预测一致。
- 清零、带噪清零、分层状态错配及测试标签置换，两个新头全部五模型的总体 BA 在 **49.9825%–50.0695%**，均通过预定 [47%,53%] 流程门槛。这不是独立的统计等效证明。
- 所有已选拟合收敛；两个主头的解析目标梯度、C 选择、共享训练标准化、核心与旧头哈希、全部 20 库原始分数及每库首个扰动的状态／预测重算均通过独立审计。其余九次扰动预测的分数全部重算，扰动状态可用保存的随机流再生。

原 L-BFGS 在模型302、增强臂 C=1 上达到 3,000 和 15,000 次迭代后仍未收敛。随后统一改为 **Newton-Cholesky 求解同一加权逻辑回归目标**，数据、特征、C 网格、验证规则和科学判据未变。所有五模型的最终结果使用同一修正流程。

修正前先验证两个求解器在加权小型问题上的目标／预测一致；真实失败问题在修正后 11 次迭代收敛，平均目标的最大绝对梯度约 **2.59×10⁻¹¹**。原失败、301被中断的部分运行及其日志完整保留在 [原尝试目录](../../artifacts/branch_bn/round1/attempts_lbfgs/)。详见 [BN002](../../docs/decisions/branch_bn/002_solver_repair.md) 和 [数值审计](../../artifacts/branch_bn/round1/solver_audit.json)。这是数值方法修正，不是根据最终测试选择有利求解器。

**16 项测试全部通过。**原分支 B 的 118 个清单文件保持原样，旧结论没有被覆盖。

## 时间定位、幅度与单位边界

在主噪声下，五个模型的带噪测试前样本数量线性探针均为 100%；测试后为 **99.992%–100%**。不匹配子集也接近完整，按测试数量分层置换的基线约 31.26%。因此在本轮条件下，原噪声行为失败不能简单解释为样本数量信息已经不可线性读取。只改线性头就获得恢复，为读出训练条件的模型内贡献提供了直接干预证据；没有由此定位真实鸟脑机制。

增强只使用 1% 噪声；未参与拟合的 5% 幅度下，敏感集合增强头平均 BA 为 **88.60%**，个体为 86.33%、80.10%、99.37%。这只是次要外推结果，不能据此宣称对所有更强噪声有效。

还评估了两个替代尺度，头均不重新拟合：

| 模型 | 主噪声绝对 SD | 15步 RMS 的1% | 15步去均值 RMS 的1% | 15步 RMS 尺度下增强 BA |
|---|---:|---:|---:|---:|
{scales}

噪声单位明显影响难度，例如302在15步 RMS 尺度下增强 BA 为83.15%，低于主条件的94.63%。旧头的敏感／较稳健分组在所测三种尺度下仍可观察到，但这不等于跨模型结构优劣已经被公平、普遍地识别。所有尺度都是模型操作定义，未对应生理噪声。

25%、50%、100% 拟合数据的学习曲线完整保存。它们是诊断，不是独立的样本效率或最佳正则化比较；没有用测试曲线重选主要头。

原协议中的五步参照也在相同20个新测试库完成：五模型原任务头在干净及主尺度1%噪声下均为100%。该描述性计算在主要汇总后补齐，时间顺序记录于 [BN003](../../docs/decisions/branch_bn/003_five_step_reference.md)，不增加确认性结论，也不代表15步专用头已获得跨时间通用性。

## 停止理由与未解决问题

停止理由是预先登记的 H-BN1 三项条件均得到支持、质量与独立审计通过。不是时间、轮数或优化失败触发停止，也没有判断所有后续研究价值低。

本次只支持：**在这些固定核心、该图像域、15步延迟及指定噪声下，噪声增强训练能找到有效的线性读出。**未证明新随机种子的成功率、跨时间单一读出、跨外形数量抽象、稳定吸引子、持续／相关噪声抵抗或真实 NCL 机制。所用扰动是测试前一次性状态扰动，不是每步持续噪声。

新种子复制、跨时间共享读出或不同噪声过程需要另立可检验假设，不能把它们无声并入本次已完成目标。

## 产物与复现

- [初始协议](../../artifacts/branch_bn/round1/protocol.md)、[执行细化](../../docs/decisions/branch_bn/001_execution_details.md)
- [主要／次要汇总](../../artifacts/branch_bn/round1/summary.json)、[独立审计](../../artifacts/branch_bn/round1/audit.json)、[五步参照](../../artifacts/branch_bn/round1/five_step_reference.json)
- [停止决定](../../docs/decisions/branch_bn/004_stop_supported.md)、[最终完整性清单](../../artifacts/branch_bn/reproducibility_manifest.json)

完整保留图像库、原始预测、干净与首个主扰动状态、头及拟合状态、学习曲线、所有候选的收敛记录、原数值失败和源码版本。可重算汇总／审计：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/summarize_branch_bn.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_branch_bn.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/report_branch_bn.py
```

完整重新执行应在空的输出目录进行；最终执行入口为 `scripts/branch_bn_newton.py`，原 `branch_bn_round1.py` 保留以复核最初数值流程，不能把原求解器失败当成最终结果。
'''
    (D/'FINAL_REPORT.md').write_text(text)
if __name__=='__main__':main()
