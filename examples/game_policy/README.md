# Agentick policy improvement with GRPO

这个示例训练纯文本 LLM 改进 Python 游戏策略。每个 rollout 只调用模型一次：

```text
游戏规则 + 上一轮完整策略 + 上一轮真实执行反馈
  -> LLM 生成下一轮完整策略
  -> 在多个固定 evaluation seeds 上执行
  -> 成功率作为 reward
  -> AReaL GRPO 更新模型
```

环境内部会执行多步，但过程中不会再次调用 LLM。初版固定使用 `GoToGoal-v0 / easy`，采用完全可观测的文字网格。它用于验证训练链路，不代表 Agentick
完整 benchmark；easy 地图很小，模型可能很快达到满分。

## 文件与接口

| 文件               | 作用                                                        |
| ------------------ | ----------------------------------------------------------- |
| `agentick_grpo.py` | 训练入口，沿用 `PPOTrainer` 和 `get_custom_dataset`         |
| `agentick.yaml`    | 从 Geometry3K 改出的纯文本 GRPO 配置                        |
| `agent.py`         | 构造 prompt，通过 AReaL 注入的 OpenAI client 调用一次 actor |
| `prepare_data.py`  | 执行弱初始策略，保存真实反馈和 train/test 数据              |
| `evaluator.py`     | 异步启动环境子进程，汇总多 seed 成功率                      |
| `env_runner.py`    | 运行 Agentick，过滤 observation，计算可信的任务结果         |
| `policy_worker.py` | 在独立进程执行生成的策略                                    |

原复制文件 `agentick.py` 改名为 `agentick_grpo.py`，避免直接执行脚本时遮蔽 第三方 `agentick` 包。所有适配代码位于当前目录，不修改
AReaL 核心。

模型需要返回如下接口的完整 Python 程序：

```python
def act(obs, memory):
    # 替换这里的行为；该示例只会等待。
    return 0, None
```

`obs` 只有 `grid`（字符串列表，`#` 墙、`.` 空地、`G` 目标）、`position` （`[x, y]`）、`step` 和
`max_steps`。动作是 `0` 等待、`1` 上、`2` 下、`3` 左、 `4` 右。`memory` 初始为 `None`，允许跨环境步保存最多 16 KiB 的
JSON 数据， 每个 seed 重置。策略可定义辅助函数；支持的 Python 子集见 `agent.py` 的 prompt。

环境使用 Agentick 的公开 `state_dict` observation，并只转发上述字段；
不会向策略传递环境对象、`info`、任务配置、参考策略或最短路径。模型输入也不包含 evaluation seeds。此 observation 转换仅适用于当前 easy
任务，扩展难度前需补齐 危险地形、实体等语义。

## 在云端准备数据

先完成仓库本身的 AReaL 安装，再在同一环境安装固定版本的 Agentick：

```bash
# 从 AReaL-Examples 仓库根目录执行。
python -m pip install -r examples/game_policy/requirements-agentick.txt

python -m examples.game_policy.prepare_data \
  --output examples/game_policy/data/goto_goal \
  --train-size 32 \
  --valid-size 8 \
  --seeds-per-policy 4
```

数据准备只需 CPU 和 Agentick、datasets 依赖，不调用模型。脚本先运行一个已知 导航策略检查环境接口，再执行四种弱策略收集反馈。保存格式为 Hugging
Face `DatasetDict`，包含 `train` 和 `test`，可由现有 `get_custom_dataset` 的 `load_from_disk`
分支直接读取。已有输出目录不会被覆盖。

每条样本包含 `previous_policy`、JSON 字符串 `previous_feedback`、 `feedback_seeds` 和
`evaluation_seeds`。反馈包含成功率、错误、步数、起始和最终 observation，均来自真实执行。反馈 seeds 与评分 seeds 分离，训练和验证的
seed 编号也互不重叠；相同 prompt 的 GRPO 候选使用相同评分 seeds。不同 seed 仍可能 生成相同小地图，因此这不是严格的地图去重评测。

这份数据在训练期间保持固定，不会自动把当前生成策略写回下一轮数据。 如需连续策略演化，后续再增加离线收集和数据刷新流程。默认 32/8 条样本只供
连通性验证，不能据此判断泛化表现。

## 启动训练

```bash
# 保证 examples.game_policy 对本机和 rollout workers 均可导入。
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python -m examples.game_policy.agentick_grpo \
  --config examples/game_policy/agentick.yaml \
  scheduler.type=local
```

配置保留原模板的单节点 8 GPU、rollout `d4p1t1` 和 actor `d4p1t1` 布局。
这是待匹配云端资源的模板，不是此任务的最低硬件要求；上云时一起调整 `cluster`、两个 backend 和 scheduling 配置。默认文本模型为
`Qwen/Qwen2.5-1.5B-Instruct`，也可覆盖 `actor.path`。多节点运行时应让全部 worker 具有相同源码、Agentick
依赖和可访问的数据路径。

默认每组 4 个候选、最多 4 个并发 rollout，每个候选顺序评估 4 个 seed。 候选分组与 reward normalization 由 AReaL 完成。原始
reward 是 `成功 seed 数 / 总 seed 数`，不使用环境 dense reward，也不减去旧策略得分。 验证时模型 temperature 为 0；训练时为
1。

## 执行边界与排错

语法错误、输出格式错误、策略异常、非法动作和策略超时都记作失败；Agentick 缺依赖、环境异常、运行器超时等基础设施问题会抛出异常，由训练框架处理， 不会伪装成 0
分。策略动作默认限时 1 秒、每个策略进程累计 CPU 时间 5 秒， Linux 下地址空间限制为 256 MiB；每个 seed 的整个环境进程限时 60 秒。
外层取消或超时会终止整个子进程组。

这里直接用异步子进程实现 reward，没有套用 `AsyncRewardWrapper`：需要在 策略超时时实际终止执行。环境进程负责计分，策略进程只收到
observation，并使用 精简环境变量、临时工作目录和受限 Python 接口。这些措施用于初版执行隔离， 不是对恶意 Python
的完整安全沙箱；对不可信模型代码做正式训练前，应在云端将 执行器放入无网络、无凭据、受文件系统和资源限制的容器。

本机未运行 Agentick、模型推理或 GRPO。轻量测试无需这些依赖：

```bash
python -m pytest -o addopts='' examples/game_policy/tests -q
```

实现按 Agentick
[`ddbad3b`](https://github.com/roger-creus/agentick/tree/ddbad3b205e6aa331a7d985873df6cbd5f381600)
的源码接口编写。云端应先完成数据准备中的环境检查，再启动训练。
