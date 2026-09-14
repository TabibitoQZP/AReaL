# Agentick：客户端组归一化

这个目录 `examples/game_policy/agentick_single/` 可以单独复制到 CPU 机器运行，无需 AReaL、torch、OpenAI SDK
或 datasets。训练端入口和配置位于上一级目录： [`train.py`](../train.py) 和 [`config.yaml`](../config.yaml)。

## 训练流程

```text
同一条旧策略＋旧反馈
  → 4 个独立 session，各调用 LLM 一次
  → 每个候选在相同的 4 个 seed 上执行
  → 收齐原始成功率 r[0..3]
  → 客户端计算 A[i] = (r[i] - mean(r)) / (std(r) + 1e-8)
  → 用各自的 session key、completion ID 上报 A[i]
```

`std` 使用总体标准差（除以组大小）。全组同分时明确返回全零，不跳过这些轨迹。原始成功率保留在日志的 `evaluation.reward`，实际上报的值保存在
`normalized_reward`，允许负数和大于 1 的值。

组之间顺序执行，组内候选并发执行；只有整组评测成功才开始上报。策略语法错误、非法动作或策略超时是有效的 0 分；环境缺依赖、HTTP 错误等基础设施问题会停止整组，不伪造 0
分。

服务配置保留 `n_samples=1`，因为一个 HTTP session 对应一条轨迹。客户端负责分组，服务设置
`reward_bias=0`、`reward_scaling=1`、`reward_norm=null`、`adv_norm=null`、`discount=1`、`gae_lambda=1`，无
critic、无 KL reward，使上报值成为生成 token 的优势。

原生 online 队列仍不提供组原子提交、同组进入同一个 batch 或采样权重版本锁定的保证。默认使用一个 driver，训练 batch size
设为组大小的整数倍；不要把这一实现理解为服务端已经增加了完整 GRPO 分组调度。

## 启动

**训练机器**：安装 AReaL，在仓库根目录运行。通过部署环境设置独立的 `AREAL_ROLLOUT_ADMIN_KEY` 和
`AREAL_ACTOR_ADMIN_KEY`，不把管理密钥交给低权限评测进程。

```bash
python examples/game_policy/train.py --config examples/game_policy/config.yaml
```

记录服务日志中的 inference gateway 地址。配置保留原来的单节点 8 GPU 布局，需按云端资源调整。

**评测机器**：复制本目录，进入目录后直接运行脚本。没有本地 `agentick` Python 包或 `agentick.py` 文件，以免遮蔽第三方 Agentick。

```bash
python -m pip install -r requirements.txt
python prepare_data.py --output data/goto_goal --train-size 32 --valid-size 8
```

任务暂时限定 `GoToGoal-v0 / easy`。数据准备执行四种弱策略，保存真实反馈及独立的评分 seeds，输出
`train.jsonl`、`valid.jsonl` 和
`preparation.json`。旧策略＋旧反馈在训练期间固定，不会自动变成上轮生成的策略。`valid.jsonl` 留作单独评测，不交给在线
driver，以免参与训练。

在持有 gateway 管理密钥的**受信任环境**预发 session：

```bash
python rl_client.py --gateway http://TRAINING_HOST:PORT --count 384 --output sessions.jsonl
```

申请脚本从 `AREAL_ROLLOUT_ADMIN_KEY` 读取密钥，通过 `POST /rl/start_session` 为每个候选创建独立 session。凭据文件以
`0600` 创建，控制台不打印 key。将文件交付给评测用户并设置适当所有权，不交付 admin key；凭据会过期，宜在开始评测前申请，大规模运行应分批处理。

然后在低权限评测环境运行：

```bash
python agent.py --gateway http://TRAINING_HOST:PORT --data data/goto_goal/train.jsonl --sessions sessions.jsonl --output runs/train.jsonl --group-size 4 --epochs 3
```

这一模式仅凭 session key 调用 `/v1/chat/completions` 和 `/rl/set_reward`。也支持在受信任的驱动环境中，用
`--admin-key-env AREAL_ROLLOUT_ADMIN_KEY` 替换 `--sessions sessions.jsonl`，由 driver
在每组开始时自行申请 session。

每个候选使用一个新 session，不复用已经评分的 key。`set_reward_finish_timeout=0` 使上报立即结束轨迹，客户端检查
`trajectory_ready=true`，不调用不存在的 `/rl/end_session`。

## 数量与文件

默认 32 条 sample × 3 次遍历 × 每组 4 个候选，共 **96 组、384 条轨迹**。服务全局 batch size 为 16，即默认每次更新消费 4
组，共 **24 次更新**。每个候选评测 4 个 seed，最多执行 1,536 局游戏。改变数据量、epochs 或组大小后，要相应调整训练总步数或 batch size。

| 文件               | 作用                             |
| ------------------ | -------------------------------- |
| `agent.py`         | 组调度、评测、归一化和上报       |
| `rl_client.py`     | HTTP 协议及预发 session 命令     |
| `prepare_data.py`  | 准备固定 JSONL 数据              |
| `evaluator.py`     | 异步环境执行与成功率汇总         |
| `env_runner.py`    | Agentick 循环和 observation 过滤 |
| `policy_worker.py` | 受限策略接口与独立进程           |

模型仍输出 `act(obs, memory) -> (action, next_memory)`。observation
包含文字网格、位置和步数，不包含环境对象、任务配置或最短路径。策略执行保留原有超时和资源限制；这些 Python
限制不是完整安全沙箱，评测用户／容器还应限制文件访问和网络权限。客户端本身是可信评分方。

## 日志和验证

JSONL 日志保存生成代码、原始反馈、完整组的归一化结果以及逐条 reward 确认，不保存 API
keys，拒绝覆盖已有文件。整组归一化结果在第一次上报前落盘。若上报中途失败，可能已有部分成员进入训练；驱动停止且不自动重试，需结合日志人工核对，不能直接重用旧 session
重跑。

`set_reward` 成功只表示轨迹就绪，服务消费是异步的。若有轨迹被服务拒收，可能需要用新 session 补足训练量。客户端温度默认为 1，修改时应同时调整服务的
`gconfig.temperature`／`actor.temperature`。

```bash
# 从本目录执行轻量测试，无需 Agentick、AReaL 或 GPU；需安装 pytest。
python -m pytest -o addopts='' tests -q
```

测试覆盖组归一化、同分组、全组评测屏障、组失败取消、负 reward 的 HTTP 上报、凭据隔离及独立目录运行。真实 Agentick 与 GPU
训推联调仍需在云端完成。Agentick 依赖固定到
[`ddbad3b`](https://github.com/roger-creus/agentick/tree/ddbad3b205e6aa331a7d985873df6cbd5f381600)。
