# SPDX-License-Identifier: Apache-2.0
"""Training-side clients for model generation and single-game execution."""

from urllib.parse import urlsplit

import httpx


def api_url(value: str) -> str:
    url = urlsplit(value)
    if (
        url.scheme not in ("http", "https")
        or not url.netloc
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError("Expected an HTTP(S) API URL without embedded credentials")
    return value.rstrip("/")


class ChatClient:
    def __init__(self, http: httpx.AsyncClient, base_url: str, api_key: str):
        self.http = http
        self.base_url = api_url(base_url)
        self.api_key = api_key

    async def generate(self, messages: list[dict], **generation) -> tuple[str, str]:
        # Never retry an ambiguous generation: it may already be in the session.
        try:
            response = await self.http.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={**generation, "messages": messages, "n": 1, "stream": False},
                timeout=httpx.Timeout(1800, connect=30, pool=30),
            )
        except httpx.RequestError:
            raise RuntimeError("Model API transport failure; no retry") from None
        if not response.is_success:
            raise RuntimeError(f"Model API HTTP {response.status_code}; no retry")
        try:
            data = response.json()
            choices = data["choices"]
            content = choices[0]["message"]["content"]
            if (
                len(choices) != 1
                or not isinstance(data["id"], str)
                or not data["id"]
                or (content is not None and not isinstance(content, str))
            ):
                raise ValueError
        except (KeyError, IndexError, TypeError, ValueError):
            raise RuntimeError("Invalid Chat Completions response") from None
        # reasoning_content is already separated by the model API. Only the final
        # content is executable; AReaL retains the original generation tokens.
        return data["id"], content or ""


class PolicyEvaluator:
    """One request per seed; aggregate feedback/scoring on the agent side."""

    def __init__(
        self, http: httpx.AsyncClient, base_url: str, key: str, timeout: float = 600
    ):
        self.http = http
        self.base_url = api_url(base_url)
        self.key = key
        self.timeout = timeout

    async def evaluate(
        self, code: str, seeds: list[int], task: str, difficulty: str
    ) -> dict:
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("Use nonempty distinct evaluation seeds")
        trials = []
        # Bound fan-out to one in-flight evaluation per candidate. Independent
        # candidates/groups still execute concurrently under AReaL's scheduler.
        for seed in seeds:
            payload = {
                "task": task,
                "difficulty": difficulty,
                "seed": seed,
                "code": code,
            }
            try:
                response = await self.http.post(
                    f"{self.base_url}/evaluate",
                    headers={"Authorization": f"Bearer {self.key}"},
                    json=payload,
                    timeout=httpx.Timeout(self.timeout, connect=30, pool=30),
                )
            except httpx.RequestError:
                raise RuntimeError("Game execution transport failure") from None
            if not response.is_success:
                raise RuntimeError(f"Game execution HTTP {response.status_code}")
            try:
                trial = response.json()
                if (
                    trial["task"] != task
                    or trial["difficulty"] != difficulty
                    or type(trial["seed"]) is not int
                    or trial["seed"] != seed
                    or type(trial["success"]) is not bool
                ):
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                raise RuntimeError("Invalid game execution result") from None
            trials.append(trial)
        return {
            "reward": sum(t["success"] for t in trials) / len(trials),
            "trials": trials,
        }
