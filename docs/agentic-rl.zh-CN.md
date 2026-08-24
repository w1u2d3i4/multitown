# MultiTown 中的 Agentic RL

[English](agentic-rl.md)

> 研究状态：本分支公开控制器学习代码与经过审计的实验结论，但不声称学习策略已经可以
> 取代 A8。

## 一句话解释

MultiTown 不是用强化学习重新训练语言模型，而是训练语言模型外面的**组织控制器**：
何时补充证据、调用另一个 Agent、升级到更强模型、发起复核、执行结果、停止流程或转交
人工。

因此优化目标不只是准确率，还必须同时考虑任务成功、Token、延迟、人工成本和不安全
执行。

## 控制问题定义

MultiTown 目前包含两类相关的序列控制环境：

| 环境 | 公开状态 | 动作 | 目标 |
| --- | --- | --- | --- |
| A9 offline fitted-Q | 当前候选结果与验证状态、已用 Token 和延迟、是否委派/升级/复核、弱模型分歧 | `stop`、`delegate`、`escalate`、`review`、`human` | 成功收益减去 Token、延迟、安全和人工惩罚 |
| Long-horizon POMDP | 47 维公开观测、剩余预算、事件进度、工具与复核状态、合法动作 Mask | `observe`、`delegate`、`escalate`、`connect`、`review`、`execute`、`human`、`stop` | 在多事件任务中平衡完成度、行动成本、安全、预算和人工介入 |

策略看不到私有正确答案。合法动作 Mask 会阻止 Actor 采样当前状态下不可用的动作；
安全实验还比较了学习式约束与只依赖公开状态的 Review Shield。

A9 offline controller 的终止奖励可以概括为：

```text
成功收益
− Token 系数 × Token / 1000
− 延迟系数 × 延迟
− 安全系数 × 不安全执行
− 人工系数 × 人工升级
```

长序列环境进一步加入子目标进度、工具故障恢复、非法动作、预算超限和无效委派。准确
系数写在版本化环境代码里，而不是依赖未跟踪的实验配置。

## 证据路线

在已经公开的 180 条冻结场景对比中，确定性 A8 仍然是最强参考。RL 部分应理解为一组
逐步推进的机制和安全性发现：

| 实验 | 测试内容 | 审计结论 |
| --- | --- | --- |
| A9-v1 | 在反事实控制器转移上训练 offline fitted-Q | 75.00% 对匹配 A8 的 72.22%，但差值区间跨零 |
| A9-v2 | Train-only masked PPO | 成功率 23.83% → 34.00%、Token −25.41%，但 unsafe episode 15.73% → 66.00% |
| A9-v3 | Hard review shield 诊断 | unsafe episode 66.00% → 5.84%，但 autonomous success 34.00% → 0% |
| A22 | 60 次拟合的 constrained-PPO 后续实验 | 所测协议中的安全边界恢复，但成功率 noninferiority 不稳定且 Token 增加 |

这些结果说明只优化成本是不够的：Agentic controller 可能变得更便宜、表面成功率更高，
同时学会了危险的停止或执行策略。

## 代码入口

| 路径 | 作用 |
| --- | --- |
| `multitown/a9_fitted_q.py` | 五动作 offline fitted-Q controller |
| `multitown/long_horizon_env.py` | 20–50 步确定性 POMDP 与奖励协议 |
| `multitown/a9_long_horizon_env.py` | 防泄漏 train-only episode 生成与审计 |
| `multitown/ppo_controller.py` | 带动作 Mask 的 Actor-Critic PPO |
| `multitown/a9_ppo_oof.py` | Out-of-fold 训练协议 |
| `multitown/a9_safety_development.py` | Review Shield 安全诊断 |
| `multitown/a22_constrained_ppo.py` | Lagrangian 与 Shield 机制实现 |
| `multitown/a25_method_conformance.py` | 方法与声明一致性检查 |
| `multitown/a26_safe_router.py` | 非泄漏 Fixture 与风险校准策略改进基线 |

下一项冻结协议是 [A26 风险校准策略改进](A26_SAFE_POLICY_IMPROVEMENT_ZH.md)。
它先建立更安全的学习式路由基线，明确不冒充完整 Agentic RL。

## 查看公开命令

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,rl,reproduction]'

multitown-run-a9-offline --help
multitown-a10-ppo --help
multitown-a9-ppo-oof --help
multitown-a9-review-shield --help
multitown-a22-adaptive --help
multitown-a22-report --help
multitown-a26-safe-router --help
```

仓库有意排除了原始实验记录、私有 episode bank、Checkpoint 和模型输出。因此这些命令
可以查看并测试公开机制，但不能单独复现私有 headline 实验。

## 分支规则

- `main`：稳定的公开运行时与 Arena；
- `agentic-rl`：学习式控制器、公开复现 Fixture 和 RL 文档的研究分支；
- `agentic-rpg`：未来可玩的叙事产品分支；这里研究的委派、预算、复核和玩家接管机制
  只有通过证据与安全门后，才会进入跑团方向；
- 学习策略只有在冻结 held-out、与 A8 等预算比较、防泄漏审计、安全指标和公开输入测试
  全部完成后，才适合进入 `main`；
- 负结果仍然保留。差值区间跨零，或者安全收益完全破坏自主效用时，不把实验描述为
  性能提升。

## 下一项公开里程碑

下一步应提供一个小型公开 episode bank 和一条命令运行的 A8-vs-RL 回放，并输出 Arena
已经支持的 Trace Schema。在完成之前，本分支应准确称为 **Agentic RL 研究实现**，而
不是已经证明普遍优于 A8 的训练控制器。
