# Agentick Full：跨任务、多轮策略改进

本目录是独立 CPU 客户端，可整体复制到评测机器。训练端继续使用上一级的 [`train.py`](../train.py) 和
[`config.yaml`](../config.yaml)。客户端不导入 AReaL、torch、datasets 或 OpenAI SDK。

共享配置采用 v2 Megatron 训练、SGLang 推理和 AWEX 权重同步。模型参数和推理使用 BF16， Megatron 的分布式 Adam 保留 FP32
优化器主权重和状态，梯度归约使用 FP32。 单节点 8 GPU 分配为 4 张训练、4 张推理，TP 和 PP 均为 1。 训练端需具备仓库要求的 Megatron/CUDA
依赖，参见[安装说明](../../../docs/en/tutorial/installation.md)； 本目录的 CPU 客户端不依赖 Megatron。

## 任务与数据

Agentick 固定为
[`ddbad3b`](https://github.com/roger-creus/agentick/tree/ddbad3b205e6aa331a7d985873df6cbd5f381600)。
[`tasks.py`](tasks.py) 硬编码 30 个训练任务和 7 个测试任务，同一任务的四档难度全部属于同一个集合。

| Category       | 训练数 | 测试数 | 测试任务               |
| -------------- | -----: | -----: | ---------------------- |
| Navigation     |      7 |      1 | RecursiveRooms         |
| Planning       |      7 |      2 | ToolUse、PackingPuzzle |
| Reasoning      |      7 |      1 | ProgramSynthesis       |
| Memory         |      3 |      1 | SequenceMemory         |
| Generalization |      2 |      1 | FewShotAdaptation      |
| Multi-Agent    |      4 |      1 | CooperativeTransport   |

训练数据对 6 个类别 × 4 档难度分配相同数量的样本，在每个类别、难度内打乱任务并循环抽样。 默认 384 条，即每个类别、难度 16
条；各任务次数最多相差一次。最后打乱全体样本。 测试数据遍历 7 个任务 × 4 档难度 × 4 组 seeds，共 112 条。

每条数据保存官方 `get_task_description_structured()` 返回的 summary、objects、goal、actions 和当前难度描述。
游戏说明直接拼接原文；我们只编写通用的代码执行和 refine 协议。首轮让模型生成代码，没有手写初始策略。

每条样本有 4 个反馈 seeds 和 8 个最终评分 seeds。训练、测试、反馈、评分使用互不重叠的 seed 区间。 同组候选使用相同的 seeds；所有
checkpoint 和基线使用同一份数据。

## 多轮流程与 reward

```text
同一 task、difficulty、seed 集合
  ├─ session A：code₁ → feedback₁ → code₂ → … → STOP / 预算耗尽
  ├─ session B：code₁ → feedback₁ → code₂ → … → STOP / 预算耗尽
  ├─ session C：…
  └─ session D：…
       ↓ 最终 policy 在独立评分 seeds 上执行
  4 个最终成功率 → 客户端组归一化 → 每个 session 上报一次末次 reward
```

`--max-rounds` 是 LLM 调用总预算，默认 4，包含首次生成和 STOP。每次调用重建两条消息： 通用 system 协议，以及官方
instruction、当前难度、剩余预算、上一轮代码与反馈。 不传入更早的对话，也不向模型传入评分 seeds 或最终评分结果。

模型输出完整 Python 或精确的 `STOP`。STOP 保留当前代码；首次就 STOP 会消耗一次调用并收到格式错误反馈。
预算耗尽时使用最后一次代码，不挑历史最佳结果。最后一次替换无效也不回退。 非末轮代码用反馈 seeds 执行；停止或预算耗尽后，最终代码仅用评分 seeds 评分。
反馈保留成功率、步数、错误和有限的公开观测示例；游戏内的 memory 每个 seed 重置。

同一候选的所有调用共用一个 session；不同候选各有自己的 session。 整组候选都完成后，用总体标准差计算
`A[i] = (r[i] - mean(r)) / (std(r) + 1e-8)`。 同分组的 A 全为 0，仍正常提交。

仅在最后一个 completion ID（可能是 STOP）上调用 `/rl/set_reward`，中间不提交 reward。 服务需保持
`export_style=individual`、`turn_discount=1`、`set_reward_finish_timeout=0`。 AReaL
按调用顺序把末次 reward 传播给此前调用，每次调用独立导出；不依赖长对话前缀。 服务关闭 reward/advantage 二次归一化、KL reward 和
critic，保持 discount、GAE lambda 为 1。 不能给每轮重复提交同一个 reward，否则会提前结束 session 或重复累计。

这是一种共享终局回报的训练方案。候选调用次数不同，导出的训练行数也不同；没有额外的停止奖励或逐轮提升奖励。 原生 online
仍不保证客户端同组候选锁定相同权重版本、原子提交或落入同一个训练 batch。

## 启动训练

**CPU 客户端**：复制本目录并进入它，安装依赖、生成数据。输出目录必须不存在。

```bash
python -m pip install -r requirements.txt
python prepare_data.py --output data/full
```

**训练机器**：在 AReaL 仓库根目录运行。部署环境提供独立的 `AREAL_ROLLOUT_ADMIN_KEY` 和
`AREAL_ACTOR_ADMIN_KEY`；GPU 布局沿用基础配置，按云端资源调整。

```bash
python examples/game_policy/train.py --config examples/game_policy/config.yaml \
  experiment_name=agentick-policy-full total_train_steps=96 \
  gconfig.max_new_tokens=4096 +rollout.agent.session_timeout_seconds=7200
```

记录训练日志中的 inference gateway 地址。客户端和服务默认 temperature 都为 1；修改时保持一致。

默认训练量是 **384 组 × 4 个候选 = 1,536 个 session**，最多 6,144 次 LLM 调用。 当前 online 队列每次消费一个完整
session 的导出结果，因此 batch size 16 对应 16 个候选， 正常接收全部结果时共 96 次更新。每次更新包含 16–64 条独立 LLM
调用的训练行，显存开销随实际轮数变化。 最多执行 30,720 局游戏：每个候选最多 3 × 4 局反馈和 8 局评分。 更改组数、epochs、group-size
时，同时调整训练端总步数；服务拒收轨迹时可能需要补足候选。

**低权限客户端模式**：在受信任环境使用管理密钥预发 session，再仅交付 session 凭据文件。 完整训练耗时可能超过凭据有效期，建议分 16 批，每批 24
条样本、96 个 session，临近执行时发放。 例如第一批，在本目录执行：

```bash
# 受信任环境；从 AREAL_ROLLOUT_ADMIN_KEY 读取管理密钥。
python rl_client.py --gateway http://TRAINING_HOST:PORT --count 96 --output sessions-00.jsonl

# 低权限评测环境；只需上述凭据文件，不需要管理密钥。
python agent.py train --gateway http://TRAINING_HOST:PORT \
  --data data/full/train.jsonl --offset 0 --limit 24 \
  --sessions sessions-00.jsonl --output runs/train-00.jsonl
```

后续批次 offset 为 24、48、…、360；每批领取新凭据并写入新日志。整个过程中训练服务持续运行。 凭据文件以 `0600`
创建，交付时设置正确所有权。已评分或结果不明的 session 不得复用。

也可在受信任的驱动环境直接运行全量训练，让客户端逐组申请 session：

```bash
python agent.py train --gateway http://TRAINING_HOST:PORT \
  --data data/full/train.jsonl --admin-key-env AREAL_ROLLOUT_ADMIN_KEY \
  --output runs/train.jsonl
```

## 冻结评测与基线

用独立推理服务加载固定 checkpoint，提供 OpenAI 兼容 Chat Completions API。 评测地址包含 `/v1`，不要指向在线训练
gateway。可通过 `OPENAI_API_KEY` 提供该推理服务的密钥。

```bash
python agent.py eval --base-url http://FROZEN_HOST:PORT/v1 --model MODEL_NAME \
  --data data/full/test.jsonl --max-rounds 4 --output runs/eval-refine.jsonl
```

`eval` 只调用 chat API，没有申请 session 或提交 reward 的路径。模型必须由部署端保持冻结。
训练命令拒绝测试任务，评测命令只接受固定测试集任务。`--samples` 可增加每条数据的独立候选数， 结果取平均，不根据测试分数挑最佳候选。

比较三个条件：未训练 checkpoint + `--max-rounds 1`、未训练 checkpoint + `--max-rounds 4`、 训练后
checkpoint + `--max-rounds 4`。其余数据、采样参数、候选数和执行预算保持一致。 评测会输出各
task/difficulty、各类别和各难度的平均成功率，以及类别宏平均和平均调用次数。 需要调参时应另留训练任务的新 seeds 做开发集，避免用这 7 个测试任务选择
checkpoint 或超参数。

## 环境、日志与验证

`env_runner.py` 使用固定版本官方 `state_dict` 的公开 grid、agent、annotations 和 public task config。
不传递原始实体列表、环境对象或内部 task_config。Fog 单元格统一设为 -1；记忆任务的隐藏字段由官方 public config 接口排除。动作 ID
根据环境的实际 action space 提供，支持交互任务；不再限制为移动动作。

生成代码在独立进程执行，策略 memory 上限 64 KiB，每个动作最多 1 秒，每局策略进程最多 30 CPU 秒， Linux 下内存上限 256
MiB。整局包含环境初始化的外部超时为 120 秒。 Python 限制不是完整安全沙箱，仍应把评测端放在受限 OS 用户或容器里。任务子进程不会继承 API 凭据。

代码格式错误、策略执行异常和动作超时记为失败，可供下一轮修复；环境故障、整局外部超时或 HTTP 故障停止运行。
组内基础设施失败会取消未完成候选；开始提交前整组原始分数和归一化结果已写入日志。 若提交中途失败，可能已有部分 session
进入训练；日志记录已确认项，客户端不自动重试，也不提供自动恢复。

JSONL 日志包含数据快照、各轮代码、反馈、最终 policy、评分、reward 确认和汇总，拒绝覆盖已有文件，不记录凭据。
正常结束的训练日志表示客户端已提交完毕，服务是否消费完成仍以训练端为准。

```bash
# 从本目录执行；轻量测试无需 Agentick 或 GPU。
python -m pytest -o addopts='' tests -q

# 安装 requirements.txt 后，检查全部 148 个组合的真实环境执行。
python smoke.py --output runs/smoke.jsonl
```

smoke 使用 no-op 策略，验证环境、观测、动作接口和终止行为；游戏失败不表示 smoke 失败。 GPU 训练、真实模型的多轮生成和训练效果需要在云端联调。
