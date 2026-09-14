# SPDX-License-Identifier: Apache-2.0
"""Standalone HTTP protocol client and trusted session-provisioning command."""

import argparse
import asyncio
import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger("GamePolicy")


class ProtocolError(RuntimeError):
    """A request failed or its outcome is unknown; never turn this into reward."""


def parse_completion(data: dict) -> tuple[str, str]:
    try:
        interaction_id = data["id"]
        choices = data["choices"]
        if len(choices) != 1:
            raise ValueError("Expected one completion")
        content = choices[0]["message"]["content"]
        if not isinstance(interaction_id, str) or not interaction_id:
            raise ValueError("Missing completion ID")
        if content is not None and not isinstance(content, str):
            raise ValueError("Completion must contain text")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ProtocolError("Invalid Chat Completions response") from exc
    return interaction_id, content or ""


class FrozenClient:
    """Evaluation-only chat client. Has no session or reward methods.

    Point this at a separately served, frozen checkpoint, never the online
    training gateway. No /rl requests or training session keys are involved.
    """

    def __init__(self, base_url: str, key: str, http: httpx.AsyncClient):
        url = urlsplit(base_url)
        if (
            url.scheme not in ("http", "https")
            or not url.netloc
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("base-url must be an HTTP(S) API URL without credentials")
        self.base_url = base_url.rstrip("/")
        self.key = key
        self.http = http

    async def generate(
        self,
        session: None,
        messages: list[dict],
        model: str = "default",
        temperature: float = 1.0,
        max_tokens: int = 4096,
    ) -> tuple[str, str]:
        if session is not None:
            raise ValueError("Evaluation must not use a training session")
        try:
            response = await self.http.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.key}"} if self.key else {},
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_completion_tokens": max_tokens,
                    "n": 1,
                    "stream": False,
                },
            )
        except httpx.RequestError as exc:
            raise ProtocolError(
                "Evaluation transport failure; request not retried"
            ) from exc
        if not response.is_success:
            raise ProtocolError(
                f"Evaluation HTTP {response.status_code}; request not retried"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ProtocolError("Evaluation response is not JSON") from exc
        if not isinstance(data, dict):
            raise ProtocolError("Evaluation response must be an object")
        return parse_completion(data)


@dataclass(frozen=True)
class Session:
    session_id: str
    session_api_key: str = field(repr=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        if not isinstance(data, dict) or not all(
            isinstance(data.get(k), str) and data[k]
            for k in (
                "session_id",
                "session_api_key",
            )
        ):
            raise ProtocolError("Invalid session credentials")
        return cls(data["session_id"], data["session_api_key"])


class RLClient:
    """No trainer imports, tensors, callbacks, or OpenAI SDK required."""

    def __init__(self, gateway: str, http: httpx.AsyncClient):
        url = urlsplit(gateway)
        if (
            url.scheme not in ("http", "https")
            or not url.netloc
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in ("", "/")
        ):
            raise ValueError(
                "gateway must be an HTTP(S) origin, without /v1 or credentials"
            )
        self.gateway = gateway.rstrip("/")
        self.http = http

    async def _post(self, path: str, key: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self.http.post(
                f"{self.gateway}{path}",
                json=body,
                headers={"Authorization": f"Bearer {key}"},
            )
        except httpx.RequestError as exc:
            # No automatic replay: generation or reward may already have succeeded.
            raise ProtocolError(
                f"{path}: transport failure; request outcome unknown"
            ) from exc
        if not response.is_success:
            # Do not log response bodies, request headers, or credential values.
            raise ProtocolError(
                f"{path}: HTTP {response.status_code}; request not retried"
            )
        try:
            result = response.json()
        except ValueError as exc:
            raise ProtocolError(f"{path}: response is not JSON") from exc
        if not isinstance(result, dict):
            raise ProtocolError(f"{path}: expected a JSON object")
        return result

    async def start_session(self, task_id: str, admin_key: str) -> Session:
        data = await self._post(
            "/rl/start_session",
            admin_key,
            {
                "task_id": task_id,
                "group_size": 1,
            },
        )
        sessions = data.get("sessions")
        if not isinstance(sessions, list) or len(sessions) != 1:
            raise ProtocolError("Expected exactly one session from /rl/start_session")
        return Session.from_dict(sessions[0])

    async def generate(
        self,
        session: Session,
        messages: list[dict[str, str]],
        model: str = "default",
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> tuple[str, str]:
        data = await self._post(
            "/v1/chat/completions",
            session.session_api_key,
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_completion_tokens": max_tokens,
                "n": 1,
                "stream": False,
            },
        )
        return parse_completion(data)

    async def set_reward(
        self,
        session: Session,
        interaction_id: str,
        reward: float,
    ) -> dict[str, Any]:
        if not math.isfinite(reward):
            raise ValueError("Normalized reward must be finite")
        data = await self._post(
            "/rl/set_reward",
            session.session_api_key,
            {
                "interaction_id": interaction_id,
                "reward": reward,
            },
        )
        if (
            data.get("session_id") != session.session_id
            or data.get("trajectory_ready") is not True
        ):
            raise ProtocolError(
                "Reward did not acknowledge a ready trajectory for this session; "
                "require rollout.agent.set_reward_finish_timeout=0"
            )
        return data


def read_sessions(path: Path) -> list[Session]:
    with path.open(encoding="utf-8") as stream:
        sessions = [
            Session.from_dict(json.loads(line)) for line in stream if line.strip()
        ]
    if len({s.session_id for s in sessions}) != len(sessions) or len(
        {s.session_api_key for s in sessions}
    ) != len(sessions):
        raise ValueError("Each candidate must have its own unique session credentials")
    return sessions


async def provision(args: argparse.Namespace) -> None:
    admin_key = os.environ.get(args.admin_key_env)
    if not admin_key:
        raise ValueError(f"Set the gateway admin key in {args.admin_key_env}")
    if args.count <= 0:
        raise ValueError("count must be positive")
    # Exclusive, private output. Keep partial files on failure for operator review.
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        async with httpx.AsyncClient(timeout=args.timeout, trust_env=False) as http:
            client = RLClient(args.gateway, http)
            for index in range(args.count):
                session = await client.start_session(
                    f"{args.task_prefix}-{index}", admin_key
                )
                await asyncio.to_thread(
                    stream.write, json.dumps(asdict(session)) + "\n"
                )
                await asyncio.to_thread(stream.flush)
    logger.info("Provisioned %d sessions in %s", args.count, args.output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--admin-key-env", default="AREAL_ROLLOUT_ADMIN_KEY")
    parser.add_argument("--task-prefix", default="game-policy")
    parser.add_argument("--timeout", type=float, default=120.0)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(provision(parser.parse_args()))


if __name__ == "__main__":
    main()
