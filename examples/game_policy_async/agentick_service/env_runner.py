# SPDX-License-Identifier: Apache-2.0
"""Trusted Agentick runner; generated Python lives in a separate process."""

import contextlib
import json
import os
import selectors
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any


class PolicyError(Exception):
    """A candidate failed, so this trial receives zero reward."""


class PolicyProcess:
    def __init__(self, code: str, timeout: float):
        self.code = code
        self.timeout = timeout

    def __enter__(self) -> "PolicyProcess":
        self.directory = tempfile.TemporaryDirectory(prefix="agentick-policy-")
        try:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    # Exclude cwd/user packages while honoring the fixed hash seed.
                    "-P",
                    "-s",
                    str(Path(__file__).with_name("policy_worker.py")),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=self.directory.name,
                env={"PATH": os.defpath, "PYTHONHASHSEED": "0"},
                bufsize=0,
            )
        except BaseException:
            self.directory.cleanup()
            raise
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            response = self.request({"code": self.code})
            if response != {"ready": True}:
                raise PolicyError("Policy initialization failed")
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def request(self, message: dict[str, Any]) -> dict[str, Any]:
        try:
            self.process.stdin.write((json.dumps(message) + "\n").encode())
            deadline = time.monotonic() + self.timeout
            line = b""
            while not line.endswith(b"\n"):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self.selector.select(remaining):
                    raise PolicyError("Policy action timed out")
                chunk = os.read(self.process.stdout.fileno(), 4096)
                if not chunk or len(line) + len(chunk) > 4096:
                    raise PolicyError(
                        "Policy process exited or returned invalid output"
                    )
                line += chunk
            result = json.loads(line)
            if "error" in result:
                raise PolicyError(result["error"])
            return result
        except (BrokenPipeError, json.JSONDecodeError) as exc:
            raise PolicyError("Policy process failed") from exc

    def __exit__(self, *_: Any) -> None:
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()
        self.selector.close()
        self.directory.cleanup()


def policy_observation(
    obs: dict[str, Any],
    step: int,
    max_steps: int,
    actions: dict[str, int],
    legend: dict,
) -> dict[str, Any]:
    """Only rendered observation and upstream's public config, never env internals.

    Omit raw entities (which can include hidden positions). Mark all fogged grid
    layers unknown, even though the pinned renderer already masks their contents.
    """
    result = json.loads(
        json.dumps(
            {
                "grid": obs["grid"],
                "agent": obs["agent"],
                "annotations": obs["annotations"],
                "task_state": obs["info"].get("task_config", {}),
                "step": step,
                "max_steps": max_steps,
                "actions": actions,
                "legend": legend,
            },
            allow_nan=False,
        )
    )
    grid = result["grid"]
    for y, row in enumerate(grid["metadata"]):
        for x, value in enumerate(row):
            if value == -1:
                for field in ("terrain", "objects", "agents"):
                    grid[field][y][x] = -1
    return result


def run_trial(request: dict[str, Any]) -> dict[str, Any]:
    import agentick
    from agentick.core.types import AgentType, CellType, ObjectType

    DIFFICULTIES = ("easy", "medium", "hard", "expert")
    TASKS = agentick.list_tasks()

    if request["task"] not in TASKS or request["difficulty"] not in DIFFICULTIES:
        raise ValueError("Unknown task or difficulty")
    seed = request["seed"]
    env = agentick.make(
        request["task"],
        difficulty=request["difficulty"],
        seed=seed,
        render_mode="state_dict",
        reward_mode="sparse",
    )
    actions = {
        env.action_space_obj.get_action_name(i): i for i in range(env.action_space.n)
    }
    legend = {
        key: {value.name: int(value) for value in enum}
        for key, enum in (
            ("terrain", CellType),
            ("objects", ObjectType),
            ("agents", AgentType),
        )
    }
    steps = 0
    episode_return = 0.0
    trace: deque = deque(maxlen=4)
    try:
        obs, _ = env.reset(seed=seed)
        initial = policy_observation(obs, 0, env.max_steps, actions, legend)
        final = initial
        try:
            with PolicyProcess(request["code"], request["action_timeout"]) as policy:
                for _ in range(env.max_steps):
                    action = policy.request(final)["action"]
                    if type(action) is not int or action not in actions.values():
                        raise PolicyError("Invalid action")
                    before = final
                    obs, reward, terminated, truncated, info = env.step(action)
                    steps += 1
                    episode_return += float(reward)
                    final = policy_observation(
                        obs, steps, env.max_steps, actions, legend
                    )
                    trace.append(
                        {
                            "observation": before,
                            "action": action,
                            "reward": float(reward),
                        }
                    )
                    if terminated or truncated:
                        if type(info.get("success")) is not bool:
                            raise RuntimeError("Agentick did not return a success flag")
                        return {
                            "success": info["success"],
                            "steps": steps,
                            "error": "",
                            "initial": initial,
                            "final": final,
                            "trace": list(trace),
                            "episode_return": episode_return,
                        }
        except PolicyError as exc:
            return {
                "success": False,
                "steps": steps,
                "error": str(exc)[:300],
                "initial": initial,
                "final": final,
                "trace": list(trace),
                "episode_return": episode_return,
            }
        raise RuntimeError("Agentick did not terminate within max_steps")
    finally:
        env.close()


def main() -> None:
    request = json.load(sys.stdin)
    # Keep dependency banners/logging out of the machine-readable result.
    with contextlib.redirect_stdout(sys.stderr):
        result = run_trial(request)
    sys.stdout.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
