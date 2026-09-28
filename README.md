# CrowBrain

分支 A 已完成六轮自适应实验，按证据停止架构扩展。**[先读最终报告](reports/FINAL_REPORT.md)**，再看 [停止依据](docs/decisions/009_stop_current_branch.md)、[假设状态](docs/hypotheses.md) 和 [研究规划](RESEARCH_PLAN.md)。结论限于本次合成任务和模型，不代表真实乌鸦 NCL 机制已被证实或证伪。

**分支 B 已完成两轮并按假设获支持结案**：[读出救援最终报告](reports/branch_b/FINAL_REPORT.md)。在全新图像上，冻结核心的 15 步准确率从 48.37% 提升到 97.93%；结论限于无扰动条件。原对照失败及第二轮修正均完整保留。

## 环境

Python 3.12、CPU PyTorch，运行时使用 1 torch 线程；批处理最多两个进程。现成环境在 `.venv`。完整依赖版本在 `requirements-lock.txt`，本项目通过 editable 安装；本目录没有 Git，运行保存源码哈希和各阶段源码归档。

标准环境可用以下方式安装（系统需支持 venv）：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e . --no-deps
```

本工作区的 venv 没有内置 pip，实际安装使用系统 `pip --python .venv/bin/python ...`；同样可以用这一形式管理现有环境。

## 验证和重建结果图

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/compare_cores.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/summarize.py round6
```

这些命令不重新训练。9 项测试包括图像数量、标签平衡、独立图像、清空记忆、分析正负对照、Dale 符号、归一化推理、匹配初始权重／缓存前向一致性，以及圆环像素化缺陷的回归测试。

## 重跑训练

先读 [原协议](docs/experiment-protocol.md) 和 [逐轮决策](docs/decisions/)，不要把后验方案当作第一轮就已确定。典型命令：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/batch_runs.py --config configs/round3.json --kind rnn --stage dev
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/batch_runs.py --config configs/round6.json --kind rnn --stage final --lr .001 --test-base 220000
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/batch_runs.py --config configs/round6.json --kind ei --stage final --lr .001 --test-base 220000
```

批处理跳过已有完整 result.json。若独立重跑，用配置中的新 `round` 名创建新输出目录，保留原证据；不要覆盖已保存的结果。Round 6 依赖预先存在的 Round 3 开发编码器，具体选择与哈希在配置／manifest 中。每次运行保存初末权重、配置、种子、数据生成流所需源码、训练轨迹和分析数组。

`src/crowbrain/legacy_task.py` 保存被拒绝的初始 1–5 刺激生成器；当前主任务为 2–6。原阶段的源码位于各 artifacts/round*/ 的 source_snapshot 归档。当前 runner 显式拒绝未实现的数量集合和时序配置，避免静默使用错误任务。

## 渲染修复与原始失败

Round 6 的训练全部完成，但原外形测试生成器在一个碎裂圆环上触发断言。10 个原 failure.json 均保留；随后统一在修复后的共享测试库上从原终点权重恢复评估，没有重训或挑 checkpoint。当前生成器会按原物体数量条件拒绝无效渲染，且已验证它与恢复评估的测试库逐像素相同。

重现这次恢复流程的脚本为 `scripts/prepare_valid_shapes.py` 和 `scripts/reevaluate_round6.py`；原诊断脚本 `diagnose_shape_failure.py` 故意在保存坏样本后报错。每次恢复的 `evaluation_repair.json` 保存权重、数据和脚本哈希。

## 生物参考数据

[data-audit.md](docs/data-audit.md) 记录可用性与局限；[ATTRIBUTION.md](data/ditz2022/ATTRIBUTION.md) 记录来源和 CC BY 4.0。原始下载文件 SHA-256 已校验，均未修改。它们提供背景与处理后观测，当前合成实验不是 NCL 定量拟合。
