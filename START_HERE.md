# 从这里接手 CrowBrain

在 Cursor、Antigravity 或其他工具中打开 **CrowBrain 根目录**，先让代理读取 `AGENTS.md` 和 `workspace/state.json`。无需原来的聊天历史。研究规范只在 AGENTS.md 维护；工具规则仅指向它。

当前交接状态：H01 当前实现分支已科学停止，较一般的机制假说仍未决；H02–H12 尚未启动。准确结论见 [H01 最终报告](reports/unified_cognition/H01_FINAL_REPORT.md)。旧 README、NEXT_EXPERIMENT_PLAN、CURRENT_RESEARCH_STATUS 等是历史材料，其中部分受封存清单保护，因此没有改写为新入口。

## 先做只读检查

在根目录运行（当前 Linux 环境）：

```bash
.venv/bin/python tools/workspace.py status
.venv/bin/python tools/workspace.py doctor
.venv/bin/python tools/workspace.py verify
```

`status` 检查交接引用并显示任务状态；`doctor` 检查当前解释器、依赖安装元数据和剩余磁盘，缺依赖返回非零；`verify` 按已登记的清单哈希逐文件校验 H01，发现缺失或改动返回非零。它们均不启动训练、不写回结果。verify 初始范围为 H01 的 2,585 个文件，不代表所有历史分支已经审计。清单和索引本身仍应纳入可信备份或版本管理。

入口只用标准库，Python 3.12+ 即可执行。没有虚拟环境时也可用 `python3 tools/workspace.py status`；doctor 会如实报告该解释器的依赖。脚本以自身位置定位根目录，不要求固定安装路径；从其他目录运行时使用脚本的绝对路径。

## 换机器时建立环境

不要直接搬用旧 `.venv`，也不要覆盖已有环境。新建环境的基础命令：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e . pytest
```

Windows 可用 `py -3.12 -m venv .venv`，把随后命令中的 `.venv/bin/python` 换为 `.venv\Scripts\python.exe`。用终端可用的 Python 3.12 或更新版本。编辑器也选择该虚拟环境解释器。

上面安装的是可用环境，不保证精确复现。`requirements-lock.txt` 保存旧环境版本（含 CPU 专用 Torch 构建），不保证每个操作系统或软件源均有这些版本。需要精确复现时参考 `artifacts/unified_cognition/h01/environment.json`，在兼容平台和软件源安装锁定版本；不可悄悄替换后声称精确复现。新环境须检查实际导入、设备和小规模运行，把包版本与设备写入新轮次记录。当前已有环境可直接使用；缺 pip 不影响运行，勿为此重建实验环境。

## 后续实验与换工具

1. 读取状态索引指向的报告、规划、停止证书和统计预算。新任务的用户授权范围决定要执行哪一分支；不要自动把 H02 当作当前任务。
2. 使用 `workspace/templates/ROUND_PROTOCOL.md` 写下一轮协议，明确上一轮证据如何决定本轮。为源码和结果选择新目录，冻结协议/配置/源码哈希后才运行。
3. 由代理实现该轮专属运行器并记录完整命令。旧 H01 运行器有固定结果路径，不能直接拿来执行新分支；本入口故意不提供会误重跑旧实验的“run next”。
4. 保留进度、日志、种子、检查点和恢复命令。用 `ROUND_REPORT.md` 写结果与继续/停止理由，更新 `workspace/state.json`。索引中的 `seals` 追加新的完整性清单和哈希，不移除历史封存项；本校验器要求清单字段为 `files[{path,bytes,sha256}]`、`file_count`、`total_bytes`，路径相对根目录。
5. 换工具前按 `HANDOFF.md` 记录活动进程及下一步，令 `state.json` 的 `handoff` 指向该文件。`active_runs` 记录主机、PID、命令、日志和恢复路径；索引不是进程探测器，新代理必须核查。不要让两个工具同时写同一分支。

复制目录时保留隐藏的规则目录、源码、配置、文档和所需科学产物；可排除 `.venv`、缓存。若仅迁移代码而未迁移结果，必须记录缺失范围；verify 报错是预期信号，不能改清单消除错误。大数据不需要全部加入模型上下文，按文件路径读取即可。

## 工具适配

- Cursor：`.cursor/rules/crowbrain.mdc` 使用 `alwaysApply: true`；也可直接读取根 AGENTS.md。
- Antigravity：`.agents/rules/crowbrain.md` 使用 `trigger: always_on`；根 AGENTS.md / GEMINI.md 提供简短入口。
- 其他工具：手动让代理读根 AGENTS.md 即可，不依赖工具专有 API。

规则格式依据 [Cursor 官方文档](https://cursor.com/docs/rules) 与 [Antigravity 官方文档](https://antigravity.google/docs/rules/)（2026-09-27 查阅）。如所用版本没有自动加载规则，直接在首条消息指定读取上述文件。不需要开启全局自动执行权限。

可复制的接手提示：

> 请先读取 AGENTS.md、START_HERE.md 和 workspace/state.json，执行只读状态、环境和完整性检查，说明当前研究结论及下一步依据。根据我本次授权的任务继续工作，每轮由上一轮结果制定；保留封存证据，不重复启动已停止分支。将进度和交接信息写回工作区。
