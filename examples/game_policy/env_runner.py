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
    obs: dict[str, Any], step: int, max_steps: int
) -> dict[str, Any]:
    """Whitelist public state_dict fields for the fully observed easy task."""
    from agentick.core.types import CellType, ObjectType

    terrain, objects = obs["grid"]["terrain"], obs["grid"]["objects"]
    grid = [
        "".join(
            "#" if cell == CellType.WALL else "G" if obj == ObjectType.GOAL else "."
            for cell, obj in zip(row, object_row, strict=True)
        )
        for row, object_row in zip(terrain, objects, strict=True)
    ]
    return {
        "grid": grid,
        "position": list(obs["agent"]["position"]),
        "step": step,
        "max_steps": max_steps,
    }


def run_trial(request: dict[str, Any]) -> dict[str, Any]:
    import agentick

    if request["task"] != "GoToGoal-v0" or request["difficulty"] != "easy":
        raise ValueError("This observation adapter supports only GoToGoal-v0 / easy")
    seed = request["seed"]
    env = agentick.make(
        request["task"],
        difficulty=request["difficulty"],
        seed=seed,
        render_mode="state_dict",
        reward_mode="sparse",
    )
    steps = 0
    try:
        obs, _ = env.reset(seed=seed)
        initial = policy_observation(obs, 0, env.max_steps)
        final = initial
        try:
            with PolicyProcess(request["code"], request["action_timeout"]) as policy:
                for _ in range(env.max_steps):
                    action = policy.request(final)["action"]
                    if type(action) is not int or action not in range(5):
                        raise PolicyError("Invalid action")
                    obs, _, terminated, truncated, info = env.step(action)
                    steps += 1
                    final = policy_observation(obs, steps, env.max_steps)
                    if terminated or truncated:
                        if type(info.get("success")) is not bool:
                            raise RuntimeError("Agentick did not return a success flag")
                        return {
                            "success": info["success"],
                            "steps": steps,
                            "error": "",
                            "initial": initial,
                            "final": final,
                        }
        except PolicyError as exc:
            return {
                "success": False,
                "steps": steps,
                "error": str(exc)[:300],
                "initial": initial,
                "final": final,
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
