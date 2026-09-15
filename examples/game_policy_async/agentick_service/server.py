# SPDX-License-Identifier: Apache-2.0
"""Standalone execution service: task + difficulty + seed + code -> game result."""

import argparse
import asyncio
import logging
import math
import os
import secrets

from aiohttp import web

from .evaluator import run_policy

logger = logging.getLogger("AgentickService")


def validate_request(data: dict, tasks: set[str]) -> None:
    if not isinstance(data, dict) or set(data) != {
        "task",
        "difficulty",
        "seed",
        "code",
    }:
        raise ValueError("Expected task, difficulty, seed and code only")
    if not isinstance(data["task"], str) or data["task"] not in tasks:
        raise ValueError("Unknown task")
    if data["difficulty"] not in ("easy", "medium", "hard", "expert"):
        raise ValueError("Unknown difficulty")
    if type(data["seed"]) is not int or data["seed"] < 0:
        raise ValueError("Expected a nonnegative integer seed")
    if not isinstance(data["code"], str) or len(data["code"].encode()) > 32768:
        raise ValueError("Expected policy code up to 32 KiB")


def create_app(
    key: str,
    evaluations: int = 16,
    max_pending: int = 256,
    timeout: float = 540,
) -> web.Application:
    import agentick

    if not key:
        raise ValueError("Set AGENTICK_SERVICE_KEY")
    if min(evaluations, max_pending, timeout) <= 0 or not math.isfinite(timeout):
        raise ValueError("Concurrency limits and timeout must be positive and finite")
    tasks = set(agentick.list_tasks())
    slots = asyncio.Semaphore(evaluations)
    active: set[asyncio.Task] = set()

    async def shutdown(app):
        pending = list(active)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    async def evaluate(request: web.Request) -> web.Response:
        if not secrets.compare_digest(
            request.headers.get("Authorization", ""), f"Bearer {key}"
        ):
            raise web.HTTPUnauthorized()
        if len(active) >= max_pending:
            raise web.HTTPServiceUnavailable(text="Evaluation queue is full")
        handler = asyncio.current_task()
        active.add(handler)
        try:
            async with asyncio.timeout(timeout):
                try:
                    data = await request.json()
                    validate_request(data, tasks)
                except (KeyError, TypeError, ValueError):
                    raise web.HTTPBadRequest(
                        text="Invalid game execution request"
                    ) from None
                async with slots:
                    result = await run_policy(
                        data["code"],
                        data["seed"],
                        data["task"],
                        data["difficulty"],
                    )
                    # Return observations and outcome for precisely one game.
                    # No knowledge of candidates, sessions, refinement or rewards.
                    return web.json_response(
                        {
                            **result,
                            "task": data["task"],
                            "difficulty": data["difficulty"],
                            "seed": data["seed"],
                        }
                    )
        except web.HTTPException:
            raise
        except asyncio.CancelledError:
            # Cancellation propagates into evaluator, killing its process group.
            raise
        except TimeoutError:
            raise web.HTTPGatewayTimeout(text="Game execution timed out") from None
        except Exception as exc:
            logger.error("Game execution failed (%s)", type(exc).__name__)
            raise web.HTTPBadGateway(text="Game execution failed") from None
        finally:
            active.discard(handler)

    async def health(request):
        return web.json_response({"status": "ok"})

    app = web.Application(
        client_max_size=262144, handler_args={"handler_cancellation": True}
    )
    app.on_shutdown.append(shutdown)
    app.router.add_post("/evaluate", evaluate)
    app.router.add_get("/health", health)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--evaluations", type=int, default=16)
    parser.add_argument("--max-pending", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=540)
    args = vars(parser.parse_args())
    host, port = args.pop("host"), args.pop("port")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    web.run_app(
        create_app(os.environ["AGENTICK_SERVICE_KEY"], **args),
        host=host,
        port=port,
        handler_cancellation=True,
        access_log=None,
    )


if __name__ == "__main__":
    main()
