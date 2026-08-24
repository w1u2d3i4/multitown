# A28：一致性门控的专家优先路由

状态：在打开独立 OOD confirmation bank 前冻结的协议。

## 前序负结果与唯一结构修改

A26 在一次性 OOD 测试上是有效负结果：自主成功率 18.9%，A8 为 18.7%，配对 95% 区间为
−2.2 至 +2.6 个百分点；unsafe episode 为 18.3%，A8 为 16.2%。成功与安全门均未通过。

轨迹机制分析发现：传感器与学习到的专家不一致时，A26 先复核准确率更低的传感器，并可能在
一次假阳性复核后执行。A28 只改这一处：

```text
sensor != preferred specialist
    A26：review(sensor) -> 可能 execute(sensor)
    A28：connect(preferred) -> review(preferred) -> 可能 execute(preferred)
```

传感器与专家一致时，仍由 calibration 阈值决定能否跳过复核。复核拒绝后调用另一位专家、重新
连接并再次复核；人工接管保持最终兜底。

## 独立协议

- 非泄漏环境与 A26 相同；
- train：3,000 个 episode，新偏移 `51_000_000`；
- calibration：500 个 episode，新偏移 `52_000_000`；
- confirmation：1,000 个 OOD episode，新偏移 `53_000_000`；
- 阈值网格、选择顺序和相对 A8 的安全 margin 均不改变；
- `selection-lock.json` 落盘后才构造 confirmation bank；
- 已经使用过的 A26 test row 不参与 A28 阈值选择。

A28 额外加入更严格的 Token 门：每 episode 平均 Token 不得超过 A8。

```bash
multitown-a28-conservative-router \
  --output /仓库外路径/a28-confirmation
```

## 声明边界

A28 是学习式路由加确定性安全工作流，不是完整 Agentic RL，也不训练语言模型参数。即使通过，
也只能证明在该非泄漏合成 Fixture 上优于 A8；通用 TeamBench 证据仍由独立 `public-bench`
分支承担。

若 A28 通过，它将成为下一阶段离线顺序 RL 的不可变回退策略。只有训练数据支持、对 A28 的
悲观优势为正、且校准风险在预算内时，RL 策略才允许偏离 A28。
