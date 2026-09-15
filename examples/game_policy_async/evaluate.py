# SPDX-License-Identifier: Apache-2.0
"""Run the local refinement agent on held-out tasks against a frozen model API."""

import argparse
import asyncio
import json
import logging
import os
import uuid
from collections import defaultdict
from pathlib import Path

import httpx

from .agent import AgentickAgent, append_event
from .client import api_url
from .tasks import load_rows

logger = logging.getLogger("AgentickEvaluation")


def summarize(rows: list[dict], results: list[dict]) -> dict:
    cells = defaultdict(list)
    for row, result in zip(rows, results, strict=True):
        cells[row["category"], row["task"], row["difficulty"]].append(result["reward"])
    categories = defaultdict(list)
    by_cell = []
    for (category, task, difficulty), scores in cells.items():
        score = sum(scores) / len(scores)
        categories[category].append(score)
        by_cell.append({"task": task, "difficulty": difficulty, "reward": score})
    by_category = {key: sum(values) / len(values) for key, values in categories.items()}
    return {
        "rows": len(results),
        "category_macro_success": sum(by_category.values()) / len(by_category),
        "by_category": by_category,
        "by_task_difficulty": by_cell,
    }


async def evaluate(args: argparse.Namespace) -> dict:
    rows = await asyncio.to_thread(load_rows, args.data, "test")
    base_url = api_url(args.base_url)
    service_url = api_url(os.environ["AGENTICK_SERVICE_URL"])
    if min(args.concurrency, args.max_rounds, args.max_tokens) < 1:
        raise ValueError("Use positive concurrency and budgets")
    await asyncio.to_thread(args.output.parent.mkdir, parents=True, exist_ok=True)
    await asyncio.to_thread(args.output.touch, exist_ok=False)
    slots = asyncio.Semaphore(args.concurrency)
    log_lock = asyncio.Lock()
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(args.timeout, connect=30, pool=30),
        trust_env=False,
        limits=httpx.Limits(max_connections=args.concurrency),
    ) as http:

        async def run(row):
            async with slots:
                session_id = f"eval-{uuid.uuid4().hex}"
                agent = AgentickAgent(
                    service_url,
                    max_rounds=args.max_rounds,
                    timeout=args.timeout,
                    evaluation_timeout=args.evaluation_timeout,
                    log_dir=str(args.output.parent / "episodes"),
                    model=args.model,
                    max_completion_tokens=args.max_tokens,
                    temperature=args.temperature,
                    chat_template_kwargs={"enable_thinking": True},
                )
                reward = await agent.run(
                    row,
                    session_id=session_id,
                    base_url=base_url,
                    api_key=os.environ.get("OPENAI_API_KEY", "unused"),
                    http_client=http,
                )
                result = {"session_id": session_id, "reward": reward}

                async with log_lock:
                    await asyncio.to_thread(
                        append_event, args.output, {"id": row["id"], **result}
                    )
                return result

        # TaskGroup cancels siblings on infrastructure failure; disconnect then
        # cancels their jobs on the service. No candidate is silently omitted.
        async with asyncio.TaskGroup() as group:
            pending = [group.create_task(run(row)) for row in rows]
        summary = summarize(rows, [task.result() for task in pending])
        await asyncio.to_thread(
            append_event, args.output, {"status": "summary", **summary}
        )
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--base-url", required=True, help="Frozen model API, including /v1"
    )
    parser.add_argument("--model", default="default")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--timeout", type=float, default=7500)
    parser.add_argument("--evaluation-timeout", type=float, default=600)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logger.info("%s", json.dumps(asyncio.run(evaluate(parser.parse_args()))))


if __name__ == "__main__":
    main()
