# Agentick policy refinement with inline GRPO

This example trains Qwen3.5-9B on one node with eight H200 GPUs using AReaL v1,
Megatron, SGLang and colocated AWEX. The **inline agent runs inside AReaL**. Agentick is
a standalone, concurrent **game execution service**.

```text
AReaL: dataset -> candidate sessions -> inline agent -> model inference
                                           |
                               task, difficulty, seed, code
                                           |
                                   Agentick /evaluate
                                           |
                                   one game's result
                                           |
AReaL: feedback -> next policy / STOP -> final score -> group normalization
                                                            |
                                                 Megatron update -> AWEX
```

## Responsibilities

| AReaL side                                                      | Agentick service                                                 |
| --------------------------------------------------------------- | ---------------------------------------------------------------- |
| Official instructions, task split, feedback/scoring seeds       | Execute the requested game and policy                            |
| Model calls and session credentials                             | Bound concurrent CPU game processes                              |
| Short-context construction, code extraction, refine and STOP    | Return public observations, success, steps, return and errors    |
| Aggregate final scoring results and normalize candidate rewards | Cancel execution when the request disconnects or expires         |
| Candidate-level traces and frozen-checkpoint evaluation         | No model client, agent loop, session API or reward normalization |

`agent.py` contains the inline agent and refinement loop. `client.py` contains the model
client and game HTTP client. `agentick_service/` contains only the execution server,
subprocess runner and restricted Python policy worker; that directory can be deployed
independently. Model URLs, model credentials, session IDs, instructions, conversation
state and seed roles never enter the game service.

Each authenticated `POST /evaluate` has exactly four JSON fields:

```json
{
  "task": "GoToGoal-v0",
  "difficulty": "easy",
  "seed": 0,
  "code": "def act(obs, memory):\n    return 0, memory"
}
```

The response identifies the task, difficulty and seed, and returns `success`, `steps`,
`episode_return`, public `initial`/`final` observations and `trace`, plus `error` for
policy failures. There is no aggregated training reward in the response. The service
also exposes `GET /health`.

## Refinement and rewards

The fixed split contains 30 training tasks and seven held-out tasks across six
categories and four difficulties. Preparation snapshots official Agentick instructions
and balances category/difficulty counts, cycling tasks within categories. A row contains
four feedback seeds and eight separate scoring seeds by default.

AReaL creates four candidate sessions per row. Each candidate's model calls share one
session. Every call receives only the instruction, immediately previous code and
feedback, and remaining call budget. The budget defaults to four calls, including the
initial generation and any `STOP` call. `STOP` retains the latest policy; a malformed
replacement never falls back to an earlier policy.

The inline agent sends one game request per seed, sequentially within a candidate to
bound request fan-out. Independent candidates and groups overlap under AReaL's
scheduler, while the service executes requests up to its CPU concurrency limit. The
inline agent aggregates feedback locally, makes the next model call, and finally
computes mean success over scoring seeds. It returns that raw float to AReaL.

AReaL normalizes terminal scores across each four-candidate group using population
standard deviation and propagates each candidate's outcome across its calls
(`individual`, `turn_discount=1`). Actor-side reward/advantage normalization is
disabled. More calls produce more training tokens; group statistics count each candidate
once. Infrastructure failures reject the entire group instead of assigning an invented
score. Invalid policies and policy execution errors produce game failure.

Qwen3 reasoning is separated by the proxy (`reasoning_parser: qwen3`,
`enable_thinking: true`). The inline agent executes final content and also handles one
leading `<think>...</think>` block from APIs returning raw text. Original tokens remain
in AReaL's training cache. This example uses text observations, without images.

The training client sends raw HTTP JSON to AReaL with
`extra_body: {chat_template_kwargs: {enable_thinking: true}}`. Frozen SGLang evaluation
keeps `chat_template_kwargs` at the JSON top level. These are different server
contracts; do not globally wrap the shared HTTP client's payload.

## Start the execution service

Copy `agentick_service/` to the CPU host or sandbox. From its parent directory, using
Python 3.12 or newer in a separate CPU environment:

```bash
python -m pip install -r agentick_service/requirements.txt
export AGENTICK_SERVICE_KEY='<shared execution-service secret>'
python -m agentick_service.server \
  --host '<service bind address>' --port 8080 \
  --evaluations 16 --max-pending 256 --timeout 540
```

Only the trainer's rollout workers need to reach this service. The service does not
connect back to AReaL or a model endpoint. The shared service key authenticates game
execution requests; it is separate from all model/session/admin credentials.

Run the service as the intended low-privilege user or inside the chosen sandbox. Game
subprocesses receive neither API credentials nor GPU visibility. Restricted Python and
process limits supplement that deployment boundary. Each game has a 120-second execution
deadline; the service's 540-second request deadline also includes CPU queueing.
Disconnects and service shutdown cancel outstanding games and terminate their
environment/policy process groups.

## Prepare data

From the repository root, in an environment with the pinned Agentick dependency:

```bash
python -m examples.game_policy_async.prepare_data \
  --output examples/game_policy_async/data
```

This writes 384 training rows, 112 held-out rows and a manifest. Existing
`agentick_full` JSONL data at the pinned revision can also be used. Dataset preparation
needs Agentick to read official descriptions; the training process only loads the
resulting JSONL and does not need Agentick installed. The execution service does not
need the dataset files or shared storage with the trainer.

## Start training

Use this AReaL checkout's GPU environment with compatible Qwen3.5 support in Megatron
Bridge, AWEX and SGLang. The recipe follows the upstream
[Qwen3.5 AWEX example](../math/qwen3_5_27b_grpo_awex_colocate.yaml), adjusted for one
node with eight H200 GPUs. From the repository root:

```bash
export FILEROOT='<absolute experiment directory>'
export MODEL_PATH='Qwen/Qwen3.5-9B'  # or a local checkpoint directory
export AGENTICK_TRAIN_DATA='<absolute path to train.jsonl>'
export AGENTICK_SERVICE_URL='http://<service address>:8080'
export AGENTICK_SERVICE_KEY='<same execution-service secret>'
export AREAL_PROXY_ADMIN_KEY='<separate trainer admin secret>'

python -m examples.game_policy_async.train \
  --config examples/game_policy_async/config.yaml
```

Actor uses DP4 × TP2 (eight rank workers); rollout uses DP8 × TP1 (eight inference
workers). Forked colocation requires equal worker counts, not merely equal total GPU
counts: each actor worker exposes one card, inherited by its inference worker. AWEX
supports different actor/inference TP layouts. Training uses BF16 parameters and FP32
optimizer masters, CP1, Megatron Bridge, and disabled MTP. SGLang's memory saver is
enabled; expandable-segment allocation is disabled for AWEX memory remapping. Training
and inference alternate on the shared GPUs. Each TP1 inference replica now holds the
full model; verify per-card memory headroom on the target hardware before a full run.
The entry point rejects incompatible worker counts or multi-GPU inference instances
before constructing the trainer.

The default batch is **eight groups × four candidates = 32 candidate sessions**, with
32–128 model calls per update. With 384 rows and one epoch, this is nominally 48 updates
and 1,536 candidates. Dropped groups, prefetch and early termination can change executed
work. Start with a short cloud run by adding
`total_train_steps=2 gconfig.max_new_tokens=1024` before the full epoch.

| Control                           | Default | Unit                                           |
| --------------------------------- | ------: | ---------------------------------------------- |
| `train_dataset.batch_size`        |       8 | Groups per update                              |
| `rollout.max_concurrent_rollouts` |      32 | Group slots across eight inference DP replicas |
| `gconfig.n_samples`               |       4 | Candidate sessions per group                   |
| `sglang.max_running_requests`     |      16 | Generation requests per inference replica      |
| Service `--evaluations`           |      16 | Simultaneously executing games                 |
| Service `--max-pending`           |     256 | Executing plus queued game requests            |

Batch eight and `max_head_offpolicyness: 3` allow up to `8 × (3 + 1) = 32` outstanding
groups, matching the group slots. Reducing either value lowers the ceiling even when
HTTP concurrency limits stay unchanged. Available work and prefetch can also limit
achieved concurrency.

`AGENTICK_MAX_ROUNDS` defaults to four. `AGENTICK_EVALUATION_TIMEOUT` defaults to 600
seconds per game HTTP request, including queueing. Model calls have a 1,800-second
timeout, and `AGENTICK_REQUEST_TIMEOUT` bounds the complete local candidate at 7,500
seconds, including AWEX pauses. Proxy session expiry is 14,400 seconds. Adjust these
together for different CPU capacity or refinement budgets. Candidate traces are written
on the AReaL workers under `FILEROOT/experiment_name/trial_name/agentick/`, without API
keys.

## Held-out evaluation

Run the same local agent against a separately served frozen model. This command needs
`httpx` and the example source, but neither AReaL's training runtime nor a local
Agentick installation. Set the service URL/key and the frozen model's optional
`OPENAI_API_KEY`, then run from the repository root:

```bash
python -m examples.game_policy_async.evaluate \
  --data examples/game_policy_async/data/test.jsonl \
  --output examples/game_policy_async/logs/eval.jsonl \
  --base-url 'http://<frozen model address>:<port>/v1' \
  --max-rounds 4 --max-tokens 8192 --concurrency 16
```

Model calls and refinement run in this local driver. The game service still receives
only the four execution fields. Evaluation averages repeats within task/difficulty and
then averages categories equally, without best-of-N selection. Full candidate traces are
saved beside the summary under `episodes/`.

Alternatively, replace `valid_dataset: null` in the training YAML with a mapping
containing `path`, `batch_size: 4`, `num_workers: 0`, and `type: rl`. The trainer loads
the fixed test split and runs the same inline agent in AReaL evaluation sessions.

## Local checks

With the CPU service dependencies, `httpx`, `pytest`, `pytest-asyncio` and `omegaconf`
installed, run from the repository root:

```bash
python -m pytest -q examples/game_policy_async/tests
```

Tests exercise the game-only HTTP contract, local model/refinement loop, credential
separation, concurrency, cancellation, task/seed partitions and real CPU games. GPU
training, Qwen3.5 generation, AWEX updates and throughput still require the eight-H200
server.
