# MultiTown

[English](README.md) | [简体中文](README_ZH.md)

<p align="center">
  <img src="demo/assets/multitown-arena.gif" alt="MultiTown Arena：A4 与 A8 两种 AI 组织处理同一个任务" width="960" />
</p>

<p align="center">
  <strong>一座可观看的 AI Agent 赛博小镇——建立 AI 公司，看它工作、花钱、质检、升级并动态调整。</strong>
</p>

MultiTown 是一座**可视化多智能体赛博小镇**、AI 组织数字孪生与 Python 运行时。
Planner、Worker、Specialist、Validator、Router 被具象为拥有不同信息、权限和成本的
建筑与居民。Arena 把原本不可见的模型调用、任务交接、验证、升级、Token 消耗和最终
结果变成一段可以观看、回放和比较的小镇运行过程。

## 这座赛博小镇能做什么

| 能力 | 在 MultiTown 中如何体现 |
| --- | --- |
| 可视化组织回放 | 在浏览器 Arena 中观看任务包穿过两座小镇，看到建筑激活、队列、告警、验证和最终交付。 |
| 角色与权限隔离 | Planner、Executor、Verifier 获得不同上下文、工具和权限，避免一个 Agent 同时提出、修改并自我验收。 |
| 成本感知调度 | 先启用经济型 Worker，只在证据和预算允许时追加强专家或独立复核。 |
| 验证与恢复 | 读取可观察运行证据，触发复核或升级，执行 Token 预算守卫，并保留确定性回退与回滚路径。 |
| 可比较实验 | 在冻结任务上回放固定组织与动态组织，同时记录成功率、Token、延迟、能耗和安全结果。 |
| 产品演进方向 | 在 `agentic-rpg` 上继续把同一套小镇抽象扩展为角色、剧情、任务和玩家干预。 |

当前 Arena 是确定性的可视化与组织对比界面，还不是自由游玩的游戏；赛博小镇的可玩化
由 `agentic-rpg` 分支持续推进。

## 为什么需要 Multi-Agent

MultiTown 并不假设“Agent 越多，准确率一定越高”。它把多智能体看成一个**组织与治理
问题**：不同角色获得不同信息和权限，由控制器判断什么时候值得增加规划、执行或复核
成本。

单个全权限 Agent 很简单，但它可以同时读取要求、修改工作并自我验收；固定完整团队
能够分离这些职责，却会在每个任务上支付全部协作成本。MultiTown 研究中间方案：先用
满足约束的最低成本组织，根据可观察的执行证据验证，只在必要时启用专家。`public-bench`
分支会在公开任务上同时对比单 Agent 和固定团队，检验这项主张。

## 项目分支分别做什么

新读者应从 `main` 开始。每个长期分支只承担一种职责，实验结论也只留在拥有对应证据
的分支中：

| 分支 | 目的 | 可以看到什么 |
| --- | --- | --- |
| [`main`](https://github.com/w1u2d3i4/multitown/tree/main) | 稳定成果展示 | 当前已经验证的 MultiTown 结果、Arena 演示、公开运行时代码和保守的核心结论。 |
| [`public-bench`](https://github.com/w1u2d3i4/multitown/tree/public-bench) | 通用公开数据证据 | 在公开通用 Benchmark 上，将当前方法与标准单 Agent、固定 Planner–Executor–Verifier 团队比较，同时报告质量、Token、延迟、能耗和负结果。 |
| [`agentic-rl`](https://github.com/w1u2d3i4/multitown/tree/agentic-rl) | 学习型控制研究 | 尝试在质量、预算和安全约束下，用训练得到的顺序策略替代确定性委派规则；同时为未来跑团 Agent 的动态控制打基础。 |
| [`agentic-rpg`](https://github.com/w1u2d3i4/multitown/tree/agentic-rpg) | 跑团产品方向 | 将“可观看的 Agent 小镇”继续发展为可玩的多智能体跑团雏形，逐步加入角色、剧情、任务和玩家干预。 |

这些分支不是性能排名：`public-bench` 回答当前方法能否迁移，`agentic-rl` 回答控制策略
能否学习，`agentic-rpg` 则探索这种控制如何变成实际玩法。

本仓库是仅包含代码和简明审计结果摘要的公开版本，刻意排除了原始实验记录、生成的
结果文件、模型 Checkpoint、Benchmark 数据集、内部过程文档、中断运行以及私有开发
Git 历史。

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

## 如何理解实验编号

`A` 编号是组织方案的实验编号，不是模型版本，也不代表性能排名。结果部分主要涉及
以下四种设计：

| 编号 | 通俗含义 |
| --- | --- |
| A4 | 固定完整团队：每次都调用一个强 Planner、三个弱 Worker 和一个独立强 Verifier。 |
| A6 | 任务前统计路由器：执行前根据交叉拟合的场景统计和预算，一次性选择一套完整组织。 |
| A7 | 任务前学习型路由器：根据安全任务特征预测质量、Token 和延迟，再在执行前选择组织。 |
| A8 | 执行期动态控制器：先让经济型 Agent 尝试，只有验证证据表明有必要时才委派、升级或复核。 |

A8 是确定性的选择性委派控制器，不是训练得到的 RL 策略；学习型控制器尝试隔离在
`agentic-rl` 分支。

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

## 公开任务迁移结果：TeamBench

`public-bench` 分支把 MultiTown 的组织方法迁移到 89 个公开 TeamBench 任务，并在
操作系统级角色隔离以及相同模型、任务、采样种子、容器和 Token 设置下进行配对比较：

| 策略 | 完全通过 | 平均部分分 | Token/任务 | p95 延迟 | 能耗 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PlanExecute-TB | 16/89 | 0.61603 | 49,579 | 151.96 秒 | 67.89 Wh |
| Solo-TB | 14/89 | 0.63989 | 84,085 | 237.06 秒 | 90.94 Wh |
| **MT-CapacityRoute-v1** | **20/89** | **0.64180** | **48,296** | **91.06 秒** | **63.32 Wh** |

MT-CapacityRoute 保留 Planner–Executor 职责分离，只在三个由开发集预先选定的任务
类别上使用强 Executor，其余类别使用经济型 Executor。它是当前冻结本地协议中最强的
实测工作点：相对 Solo 新增 6 个通过且没有丢失 Solo 已通过任务，Token 减少 42.56%，
监控能耗减少 30.37%。但它相对两个基线的平均部分分置信区间仍跨 0，因此不能宣称为
跨论文 SOTA。协议、负结果和可迁移运行时控制位于
[`public-bench`](https://github.com/w1u2d3i4/multitown/tree/public-bench) 分支。

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
