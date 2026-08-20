# MultiTown

[English](README.md) | [简体中文](README_ZH.md)

MultiTown 是一个用于研究成本感知多智能体组织与序列控制的 Python
运行时工具包。项目包含确定性环境、组织控制器、路由与安全组件、轨迹工具和机器可读
Schema。

本仓库是仅包含代码的公开版本，刻意排除了实验记录、结果表格、生成产物、模型
Checkpoint、Benchmark 数据集、内部过程文档、中断运行以及私有开发 Git 历史。

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
- 数据集、轨迹及模型输出；
- 模型或控制器 Checkpoint；
- 研究日志、计划和审查过程文档；
- 本地推理端点、凭据和机器相关 provenance。

## 许可证

MultiTown 使用 MIT License 发布，详见 [LICENSE](LICENSE)。
