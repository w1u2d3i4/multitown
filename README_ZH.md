# MultiTown

[English](README.md) | [简体中文](README_ZH.md)

> [!IMPORTANT]
> 当前是 **`public-bench` 分支**，新增了相互隔离的
> [TeamBench 公开数据评测](public_bench/)及紧凑正式记录。稳定 Arena 仍位于
> [`main`](https://github.com/w1u2d3i4/multitown/tree/main) 分支。

## 本分支的目的

`public-bench` 是 MultiTown 的外部证据分支。它不增加新的产品故事，也不继续调整小镇
合成 Benchmark，而是回答：当前组织控制方法放到公开、通用任务上后是否仍然成立。

证据矩阵在同一任务列表和确定性质量/Token 协议下比较五种系统：TeamBench 标准的单强 Agent
（Solo-TB）、只规划后执行（PlanExecute-TB）、先执行再独立复核（ExecuteReview-TB）、
固定 Planner–Executor–Verifier 流水线（FixedTeam-TB），以及 MultiTown 的弱模型优先
选择性组织（MultiTown-TB）。五种方法直接比较质量和 Token；只有来源版本兼容的修复后
运行才比较延迟与能耗。成本优势不会被包装成方法全面领先。

这里的多智能体价值主张是职责分离：与 Solo-TB 不同，角色隔离组织不会让同一个 Agent
同时读取完整要求、修改工作区并自我验收。A8-TB 要检验的是，能否保留这条治理边界，
同时避免每个任务都启用全部角色的成本。

<p align="center">
  <img src="demo/assets/multitown-arena.gif" alt="MultiTown Arena：固定组织与动态组织处理同一个任务" width="960" />
</p>

<p align="center">
  <strong>AI Agent 的模拟城市——建立一家 AI 公司，看它花钱、质检、升级并动态调整。</strong>
</p>

MultiTown 是一个 **AI 组织数字孪生** 与 Python 运行时。Arena 把原本不可见的任务
路由、模型调用、结果验证、升级处理、Token 消耗和最终结果变成一座可以观看和比较的
小镇。

本仓库是仅包含代码和简明审计结果摘要的公开版本，刻意排除了原始实验记录、生成的
结果文件、模型 Checkpoint、Benchmark 数据集、内部过程文档、中断运行以及私有开发
Git 历史。

## A0–A8 到底是什么

`A` 编号只是不同 AI 组织方案的实验编号，**不是模型版本，也不代表数字越大越强**。
下表中的“弱模型”和“强模型”，分别指冻结实验中承担经济型角色和高能力角色的模型。

| 编号 | 通俗名称 | 每个任务如何执行 |
| --- | --- | --- |
| A0 | 单个经济型 Agent | 一个弱模型 Agent 独立完成任务。 |
| A1 | 单个强 Agent | 一个强模型 Agent 独立完成任务。 |
| A2 | 小模型投票组 | 四个弱模型 Agent 独立作答，再由确定性投票选出答案。 |
| A3 | 经理带队组织 | 强模型负责规划，三个弱模型提出方案，再由强模型整合。 |
| A4 | 固定完整团队 | 每次都调用一个强 Planner、三个弱 Worker 和一个独立强 Verifier。 |
| A5 | 规则式动态团队 | 先让弱模型工作，再由固定规则决定是否升级给强模型并选择性复核。 |
| A6 | 任务前统计路由器 | 任务开始前，根据交叉拟合的场景统计和预算，从 A0–A5 中选择一套完整组织。 |
| A7 | 任务前学习型路由器 | 任务开始前，根据安全的任务特征预测质量、Token 和延迟，再从 A0–A5 中选择组织。 |
| A8 | 执行期动态控制器 | 先让一个经济型 Agent 尝试并验证；只有证据表明有必要时，才追加 Worker、强专家或复核。 |

A6、A7 都是在任务执行**之前**一次性选好组织；A8 可以在任务执行**过程中**根据验证
结果改变组织。A8 是确定性的选择性委派策略，不是训练得到的强化学习策略。

## 启动 Arena

内置回放不需要模型、API Key、前端构建或网络请求：

```bash
git clone https://github.com/w1u2d3i4/multitown.git
cd multitown
python3 -m http.server 8000 --directory demo
```

打开 <http://127.0.0.1:8000>。界面中的 Benchmark 聚合指标来自冻结实测结果；动画
工单是解释控制流程的确定性演示场景，不是某一条原始实验轨迹。自动生成 GIF 的命令
见 [`demo/`](demo/)。

## TeamBench 主流策略对比结果

89 个公开任务的配对矩阵现已包含一个强 Solo 锚点和四种角色隔离组织：

| 策略 | 完全通过 | 平均部分分 | 平均 Token/任务 |
| --- | ---: | ---: | ---: |
| Solo-TB | 16 / 89 | **0.64180** | 82,869 |
| PlanExecute-TB | **18 / 89** | 0.62434 | **49,166** |
| ExecuteReview-TB | 10 / 89 | 0.54940 | 67,011 |
| FixedTeam-TB | 14 / 89 | 0.63375 | 108,381 |
| MultiTown-TB | 11 / 89 | 0.58251 | 68,218 |

质量/Token Pareto 前沿由 **Solo-TB 和 PlanExecute-TB** 构成。PlanExecute-TB
比 Solo-TB 少用 **40.67% Token**，完全通过数最高，但部分分差为 -0.01745
（95% CI [-0.05573, +0.01861]），因此这是效率权衡，不能声称质量更强。
ExecuteReview-TB 的平均部分分显著低于 PlanExecute-TB（-0.07494，95% CI
[-0.11236, -0.04090]），同时多用 36.30% Token；FixedTeam-TB 多用
120.44% Token，却没有明确的部分分收益。

因此证据不是简单的“多 Agent 更好”，而是：**规划与交接可以很高效；没有修复闭环时，
盲目增加角色反而可能更差。** 当前 MultiTown-TB 动态控制器并非五种策略中的赢家，
应作为有价值的负结果和下一步重设计目标。详见[正式对比记录](public_bench/records/TEAMBENCH_STRATEGY_QUALITY_V2.md)
与[主流策略映射](public_bench/docs/RELATED_WORK_AND_EVIDENCE.md)。

## TeamBench 公开任务迁移结果（历史 v1.2）

`public_bench/` 子项目在 TeamBench 公开测试列表中当前可评测的 89 个任务上，对比固定
组织与执行期选择控制器。该结果与下方 MultiTown 合成任务结果严格分开，不合并分数。

在这里，Planner 负责编写计划，Executor 负责修改代码和调用工具，Verifier 负责独立
检查结果。**A4-TB 每次都启用全部三个角色**；**A8-TB 先让经济型 Executor 执行，
只有公开运行时验证器发现有必要时，才启用更强的 Planner 或 Verifier**。`-TB` 表示
“针对 TeamBench 角色协议的适配版”，它们和合成任务里的 A4/A8 不是同一实现，也不
共用分数。

| 指标 | A4-TB——固定完整团队 | A8-TB——选择性团队 |
| --- | ---: | ---: |
| 完全通过 | 14 / 89 | 11 / 89 |
| 平均部分分 | 0.63375 | 0.58251 |
| 平均 Token/任务 | 108,381 | 68,218 |
| 中位延迟/任务 | 72.20 秒 | 58.15 秒 |
| p95 延迟/任务 | 134.98 秒 | 165.11 秒 |
| 监控能耗 | 101.04 Wh | 86.41 Wh |

A8-TB 将平均 Token 减少 **37.06%**、监控能耗减少 **14.48%**，但 A8−A4
配对部分分差为 **−0.05124**，95% 配对 Bootstrap CI 为
**[−0.08951, −0.01678]**。成本门通过、质量非劣门失败，因此不能将当前 A8-TB
宣传为 A4-TB 的替代方案。通俗来说：选择性团队节省了算力，但损失的任务质量超过
冻结规则允许的范围。详见[正式记录](public_bench/records/TEAMBENCH_TEST_V1.2.md)。

后续沙箱审计发现，Docker 命令超时后其容器可能继续运行。因此 v1.2 的确定性任务分数
仍作为历史证据保留，但延迟/能耗不会与修复后的新方法混合比较。完整五策略运行时重跑
继续受 [v2 协议](public_bench/docs/TEAMBENCH_STRATEGY_MATRIX_V2_PROTOCOL.md)约束；当前公开
五策略桥接表只比较兼容的质量/Token 字段，修复后的延迟与能耗仅报告 Solo-TB、
PlanExecute-TB 和 ExecuteReview-TB。

## 代表性 Benchmark 结果

MultiTown 当前最强的冻结结果来自 A8，即“先用便宜方案，必要时才升级”的确定性
执行期控制器。它在 180 条 held-out 场景上的结果为：

| 控制器 | 成功率 | 每次决策 Token | 平均端到端延迟 | p95 端到端延迟 |
| --- | ---: | ---: | ---: | ---: |
| A8 | 142/180（78.89%） | 621.5 | 1.104 秒 | 2.275 秒 |

按照预注册的配对比较口径：

- A8 相对 A4 固定完整团队的成功率提高 **11.67 个百分点**，95% 配对 Bootstrap CI 为
  **+4.44 至 +18.89**，同时 Token 减少 **76.54%**；
- A8 相对 A7 任务前学习型路由器的成功率提高 **11.11 个百分点**，95% 配对 Bootstrap CI 为
  **+4.44 至 +17.78**，同时 Token 减少 **54.58%**；
- 更早的 A6 任务前统计路由器在观测到的 A4 准确率水平下，将每次决策 Token 减少
  **24.63%**、平均延迟减少 **31.79%**。其准确率区间跨零，因此这里只将它解释为
  效率结果，不能声称 A6 更准确。

A8 是确定性 heuristic controller，不是训练得到的 RL 策略。这些结果来自冻结的
MultiTown 合成任务集和本地 Qwen/llama.cpp 推理，不能外推为通用多智能体优势。

## Agentic RL 与安全实验发现

MultiTown 同时保留优化收益和安全失败：

TeamBench 上的 MT-Agentic-RL-v1 从 28 条完整反事实轨迹训练三阶段 fitted-Q
策略。在新 sampling seed 的 30 个开发任务上，策略全部选择 `stop`，因此与同轨迹
PlanExecute 的 6/30 通过、0.62299 部分分和 45,682 Token/任务完全一致。它验证了
真实训练与部署链路，但没有形成性能优势，也不会进入 seed 3。详见
[Agentic RL v1 正式记录](public_bench/records/TEAMBENCH_AGENTIC_RL_DEV_V1.md)。

MT-Agentic-RL-v2 将训练集扩展到 3 个 generator seed 的 85 条成对轨迹，并用
悲观 Bootstrap-Q 集成做动作门控。在未参与训练的 seed 3 上，v2 与同轨迹
PlanExecute 都是 7/30 通过，平均部分分从 0.68465 提高到 0.69343，任务级为
2 胜、28 平、0 负。代价是 Token 增加 18.98%；质量区间仍接触 0，没有新增完整
通过，而且有 2 个任务超过声明的 90k 预算。因此它是有正向点估计的研究候选，
还不是 benchmark 最优或严格预算结果。详见
[Agentic RL v2 正式记录](public_bench/records/TEAMBENCH_AGENTIC_RL_CONFIRM_V2.md)。

后续的 A9、A22 是学习型控制器尝试的时间顺序研究编号，不是大模型版本；只有同时
通过质量、成本和安全门，它们才能替代 A8。

| 实验 | 主要结果 | 必须保留的解释边界 |
| --- | --- | --- |
| TeamBench Agentic RL v1——fitted-Q 编排控制器 | 新 sampling seed 的 30/30 个状态均选择 PlanExecute 回退；质量与 Token 完全相同 | 属于真正训练的控制器，但性能结果为负，不能声称 benchmark 最优 |
| TeamBench Agentic RL v2——悲观集成编排控制器 | Seed 3 两组均 7/30 通过；部分分 0.68465 → 0.69343；2 胜 / 28 平 / 0 负 | 点估计门通过，但无新增完整通过、质量区间接触 0、Token +18.98%，且严格预算失败 |
| A9-v1——离线 fitted-Q 控制器 | 成功率 75.00%，对应 A8 基线为 72.22% | 差值区间跨零，不能声称显著优于 A8 |
| A9-v2——带动作屏蔽的 PPO 控制器 | 成功率 23.83% → 34.00%，提高 10.17 pp（95% CI +7.87 至 +12.47），Token −25.41% | 仅为 train-only 控制器结果；unsafe episode 从 15.73% 上升至 66.00% |
| A9-v3——强制复核安全盾 | unsafe episode 从 66.00% 降至 5.84% | autonomous success 从 34.00% 降至 0%，属于安全—效用负结果 |
| A22——约束 PPO 后续实验 | 60 次拟合、345,600 条 rollout、36,000 条 calibration row、9,000 条 outer row | 所测安全边界恢复，但 success noninferiority 不稳定且 Token 增加 |

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
- `tests/`：经过筛选的运行时测试；
- `public_bench/`：仅在本分支提供的 TeamBench 适配器、冻结任务 ID、紧凑配对结果、
  沙箱定义和测试。

明确排除：

- 实验记录和监控曲线；
- 原始结果表格和生成报告；
- 数据集、轨迹及模型输出；
- 模型或控制器 Checkpoint；
- 研究日志、计划和审查过程文档；
- 本地推理端点、凭据和机器相关 provenance。

## 许可证

MultiTown 使用 MIT License 发布，详见 [LICENSE](LICENSE)。
