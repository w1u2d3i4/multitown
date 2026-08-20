# MultiTown

[English](README.md) | [简体中文](README_ZH.md)

MultiTown 是一个用于研究成本感知多智能体组织与序列控制的 Python
运行时工具包。项目包含确定性环境、组织控制器、路由与安全组件、轨迹工具和机器可读
Schema。

本仓库是仅包含代码和简明审计结果摘要的公开版本，刻意排除了原始实验记录、生成的
结果文件、模型 Checkpoint、Benchmark 数据集、内部过程文档、中断运行以及私有开发
Git 历史。

## 代表性 Benchmark 结果

MultiTown 当前最强的冻结结果来自 A8 确定性执行期控制器，它在 180 条 held-out
场景上的结果为：

| 控制器 | 成功率 | 每次决策 Token | 平均端到端延迟 | p95 端到端延迟 |
| --- | ---: | ---: | ---: | ---: |
| A8 | 142/180（78.89%） | 621.5 | 1.104 秒 | 2.275 秒 |

按照预注册的配对比较口径：

- A8 相对 A4 的成功率提高 **11.67 个百分点**，95% 配对 Bootstrap CI 为
  **+4.44 至 +18.89**，同时 Token 减少 **76.54%**；
- A8 相对 A7 的成功率提高 **11.11 个百分点**，95% 配对 Bootstrap CI 为
  **+4.44 至 +17.78**，同时 Token 减少 **54.58%**；
- 更早的 A6 预算路由器在观测到的 A4 准确率水平下，将每次决策 Token 减少
  **24.63%**、平均延迟减少 **31.79%**。其准确率区间跨零，因此这里只将它解释为
  效率结果，不能声称 A6 更准确。

A8 是确定性 heuristic controller，不是训练得到的 RL 策略。这些结果来自冻结的
MultiTown 合成任务集和本地 Qwen/llama.cpp 推理，不能外推为通用多智能体优势。

## Agentic RL 与安全实验发现

MultiTown 同时保留优化收益和安全失败：

| 实验 | 主要结果 | 必须保留的解释边界 |
| --- | --- | --- |
| A9-v1 offline fitted-Q | 成功率 75.00%，对应 A8 基线为 72.22% | 差值区间跨零，不能声称显著优于 A8 |
| A9-v2 masked PPO | 成功率 23.83% → 34.00%，提高 10.17 pp（95% CI +7.87 至 +12.47），Token −25.41% | 仅为 train-only 控制器结果；unsafe episode 从 15.73% 上升至 66.00% |
| A9-v3 hard-shield diagnostic | unsafe episode 从 66.00% 降至 5.84% | autonomous success 从 34.00% 降至 0%，属于安全—效用负结果 |
| A22 constrained-PPO follow-up | 60 次拟合、345,600 条 rollout、36,000 条 calibration row、9,000 条 outer row | 所测安全边界恢复，但 success noninferiority 不稳定且 Token 增加 |

历史 A10 long-horizon 结果没有作为正结果展示，因为后续审计发现策略可见字段能够
确定性泄露正确动作。A23 因 snapshot binding 失败而失效，Stage W 的 CR 轴为 inert。
这些失败继续保留在研究证据中，但原始记录不会进入本代码-only仓库。

## 安装

MultiTown 需要 Linux 和 Python 3.12 或更高版本。

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

控制器训练依赖需要单独安装：

```bash
python -m pip install -e '.[rl]'
```

公开 Benchmark 适配器使用独立的可选依赖：

```bash
python -m pip install -e '.[reproduction]'
```

## 查看运行入口

基础安装完成后，可以查看主要命令：

```bash
multitown-bench --help
multitown-validate-serving-trace --help
```

安装 `.[reproduction]` 后可以使用路由与 A8 命令；安装 `.[rl]` 后可以使用 PPO
命令：

```bash
multitown-run-a8 --help
multitown-a10-ppo --help
```

需要模型的命令使用由用户提供的 OpenAI 兼容推理端点和模型标识。该软件包不会自动
下载模型权重，也不应把任何凭据写入 Git 跟踪文件。

## 测试

```bash
pytest -q \
  tests/test_contracts.py \
  tests/test_a8_controller.py \
  tests/test_stateful_ops.py \
  tests/test_stateful_behavior.py \
  tests/test_stateful_groups.py \
  tests/test_stateful_pomdp.py \
  tests/test_serving_trace.py
```

本仓库中的测试只验证公开运行时代码，不复现或验证任何未公开实验。

## 仓库边界

公开内容包括：

- `multitown/`：运行时实现；
- `schemas/`：公开的机器可读协议；
- `tests/`：经过筛选的运行时测试。

明确排除：

- 实验记录和监控曲线；
- 原始结果表格和生成报告；
- 数据集、轨迹及模型输出；
- 模型或控制器 Checkpoint；
- 研究日志、计划和审查过程文档；
- 本地推理端点、凭据和机器相关 provenance。

## 许可证

MultiTown 使用 MIT License 发布，详见 [LICENSE](LICENSE)。
