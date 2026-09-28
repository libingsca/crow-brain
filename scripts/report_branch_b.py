"""Build final branch B report and scientific figure from saved summaries."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path('artifacts/branch_b');D=Path('reports/branch_b')
def main():
    r1=json.loads((B/'round1/summary.json').read_text());r2=json.loads((B/'round2/summary.json').read_text())
    assert r2['status']=='supported' and (B/'round2/audit.json').exists()
    raw=[json.loads((B/f'round1/seed{s}/result.json').read_text()) for s in range(301,306)]
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for q in r2['per_seed']:ax[0].plot([0,1],[q['old'],q['rescue']],'o-',label=str(q['seed']))
    ax[0].axhline(.65,ls='--',color='gray',lw=1);ax[0].set(xticks=[0,1],xticklabels=['Original head','Frozen refitted head'],ylabel='Balanced accuracy',ylim=(.3,1.04),title='B2: new images, 15-step delay');ax[0].legend(fontsize=7)
    for q in raw:
        ys=[np.mean([z['rescue_ba'] for z in q['noise'][f'delay15_fraction{f}']]) for f in ['0','0.01','0.05']]
        ax[1].plot([0,1,5],ys,'o-',label=str(q['seed']))
    ax[1].set(xlabel='Noise SD (% of training 5-step state RMS)',ylabel='Refitted-head balanced accuracy',ylim=(.3,1.04),xticks=[0,1,5],title='B1 secondary: heterogeneous noise sensitivity');ax[1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(D/'RESULTS.png',dpi=180)
    rows='\n'.join(f"| {q['seed']} | {q['old']*100:.2f}% | {q['rescue']*100:.2f}% | {(q['rescue']-q['old'])*100:.2f} |" for q in r2['per_seed'])
    nrows='\n'.join('| '+str(q['seed'])+' | '+' | '.join(f"{np.mean([z['rescue_ba'] for z in q['noise'][f'delay15_fraction{f}']])*100:.2f}%" for f in ['0','0.01','0.05'])+' |' for q in raw)
    text=f'''# 分支 B 最终报告：冻结核心的长延迟读出救援

2026-09-24。**完成两轮自适应实验，H-B1 在预定范围内得到支持，按此条件停止。**第二轮针对第一轮实际出现的随机对照质量失败制定，没有预先排定第二轮，也没有重新训练循环核心。

## 主要结论

在既有 5 个普通 RNN、原圆点图像域、15 步延迟且无状态扰动的条件下，仅更换线性决策读出，就能把新图像上的平均平衡准确率从 **48.365% 提升到 97.930%**。第二轮使用第一轮已经拟合并冻结的读出，未以新图像重新选参数。

| 模型种子 | 原头 BA | 冻结新头 BA | 提升（百分点） |
|---|---:|---:|---:|
{rows}

救援 BA 的 99.5% t 区间下界为 **93.972%**，高于预先指定的 65%；配对提升为 **49.565 个百分点**，99.5% t 区间为 **[33.656,65.474] 个百分点**，下界高于预设的 15 个百分点。两个终点按 Bonferroni 对应本轮合计 99% 覆盖，依赖 t 区间的模型层假设。样本单位是五个模型，不能把数千个试次或置换当成更多独立模型。

救援区间原始上界 101.888% 超过概率取值范围，完整保存于 JSON；正文报告决定判据的下界，不将超过 100% 的端点解释为可达到的性能。两轮复用同一批核心，因此是新图像确认，不是新模型群体复制，也不宣称整个自适应流程有未经证明的固定 5% 错误率。

![主要结果与噪声边界](RESULTS.png)

## 为什么没有在第一轮就宣布成功

第一轮救援 BA=97.78%，但训练标签置换头平均 BA=62.455%，违反原 [47%,53%] 质量门槛。**第一轮正式结果保留为 quality_gate_failed**；详见 [第一轮报告](ROUND1.md)。

独立审计确认没有主干更新、分区串用、测试集拟合标准化或错误重算预测。缺陷在对照设计：一次随机标签拟合并不保证在结构化测试数据上恰好表现为随机；多个模型还共享同一次置换，而且该对照使用真实验证标签选择 C，仍有任务监督信息。

据此制定 [第二轮协议](../../docs/decisions/branch_b/002_from_round1_control_failure.md)：保留所有原结果，冻结原主头，换独立图像；负对照改用 256 次独立训练标签置换、固定 C=0.01、不使用真实验证标签选参，并增加测试标签与状态错配对照。主效果阈值没有降低，确认区间反而更严格。

第二轮的均值如下，全部满足同一数值范围 [47%,53%]：

| 对照 | 平均 BA |
|---|---:|
| 清零记忆，原头 | 50.025% |
| 清零记忆，救援头 | 50.085% |
| 256 次训练标签置换，固定 C | 50.317% |
| 64 次独立测试标签置换 | 50.090% |
| 64 次按测试数量分层的状态行错配 | 49.966% |

256 个独立置换的五模型平均 BA 范围为 33.09%–72.575%；随机化均值的 Monte Carlo 99% 区间为 [49.287%,51.347%]。这说明单个随机方向可能表现很好或很差。原置换及其标签补集在固定 C 下给出互补结果，例如 seed302 为 72.95% 与 27.05%。这些观察与单次随机方向偶然对齐任务的解释一致；没有分别定量识别验证选参偏差和随机方向的全部贡献。新旧对照的选参方式不同，不能把旧单次结果直接放进新分布求精确 p 值。

这些校准检查不是严格的统计等效检验；对照通过与主效果支持是分别判断的。

## 第一轮的时间定位与噪声边界

第一轮的次要分析保留其探索性地位：

- 在 15 步测试前，按同一时刻拟合的样本数量探针准确率为 100%；测试输入后第 4 步仍为 99.99%。不匹配试次单独评价也接近满分。因此当前结果不支持“测试一进入就完全抹掉样本数量信息”的简单解释。
- 5 步拟合的样本探针直接迁移到 15 步，只有 13.6%；5 步线性匹配头迁移到 15 步为 48.225%。表征可在同时间读出而难以跨时间复用，与时间相关的表征漂移／读出错配一致；没有识别漂移是增益、平移、旋转还是更复杂变化。
- 样本和测试数量并不独立。条件置换基线约 31.23%，直接用测试数量猜样本的总体基线为 50%；因此没有把所有样本探针简单与 20% 比较来认定记忆。测试前解码测试数量的高于 20% 表现也不是预知未来输入。

扰动施于测试输入前，以训练分区 **5 步**末状态 RMS 为尺度。每种幅度重复 10 次后先在模型内平均：

| 模型种子 | 无扰动 | 1% RMS 噪声 | 5% RMS 噪声 |
|---|---:|---:|---:|
{nrows}

长延迟救援在 1% 噪声下平均降至 72.81%，但模型间差异很大：三个明显脆弱、两个较稳健。均值的描述性 95% 区间为 [45.83%,99.79%]，不能据此宣布已获得稳定的抗噪长时记忆，也不能说所有模型同样脆弱。标准 5 步的新读出在相同 1% 噪声下仍为 100%。噪声尺度是本模型的操作定义，没有映射为真实 NCL 的生理噪声水平。

## 本次能说和不能说的结论

**能说**：对这些模型，原长延迟行为失败不等于样本信息已不可利用。固定核心内仍有可供线性决策使用的状态，更换读出足以大幅恢复指定无扰动任务。这给出了读出瓶颈在模型内的干预证据。

**不能说**：已得到跨时间通用或稳健记忆机制；已发现吸引子；已解释真实乌鸦 NCL；已建立跨外形数量抽象；所有 RNN 或所有延迟均可由线性头修复。15 步专用头不自动解决训练时序变化，其他延迟曲线只作描述。

## 停止与复现

停止依据是 H-B1 已达到预定支持阈值并通过修正后独立确认与审计，**不是**时间耗尽、优化失败或认定所有后续问题价值低。进一步研究抗噪性或跨时间通用读出需要另立假设；本次不把目标无声扩大为必须造出通用工作记忆。

验证：13 项测试通过，包括原有 9 项和新增 4 个时长下的状态延续／清零一致性测试；B1、B2 全部五模型的独立产物审计均通过。

产物包括：20,000 张经数量与跨分区重复审计的图像（B1 三库+B2 新库）、原始预测和关键状态、所有探针与可移植系数、置换拟合记录、协议及源码归档。旧分支 A 的权重、报告与停止结论保留。

- [B1 主／次要结果](../../artifacts/branch_b/round1/summary.json) 与 [独立审计](../../artifacts/branch_b/round1/audit.json)
- [B2 主结果与置换分布](../../artifacts/branch_b/round2/summary.json) 与 [独立审计](../../artifacts/branch_b/round2/audit.json)
- [停止决定](../../docs/decisions/branch_b/003_stop_supported.md)
- [最终文件清单](../../artifacts/branch_b/reproducibility_manifest.json)

可在本工作区重算汇总与审计（不重新训练核心）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/summarize_branch_b.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_branch_b.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/summarize_branch_b_round2.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/audit_branch_b_round2.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/report_branch_b.py
```

训练／数据生成入口在已有目录中不应重跑覆盖原结果；独立完整复制应使用空输出目录并保留相应源码版本。
'''
    (D/'FINAL_REPORT.md').write_text(text)
if __name__=='__main__':main()
