# SPDX-License-Identifier: Apache-2.0
"""Async, bounded environment evaluation shared by preparation and training."""

import asyncio
import json
import os
import re
import signal
import sys
from pathlib import Path
from typing import Any


def extract_policy(output: str) -> str:
    """Accept plain Python or exactly one fenced Python code block."""
    output = output.strip()
    if "```" in output:
        match = re.fullmatch(r"```(?:python|py)?\s*\n(.*?)\n```", output, re.DOTALL)
        if match is None:
            raise ValueError("Return plain Python or one Python block, without prose")
        output = match[1].strip()
    if not output or len(output.encode()) > 32_768:
        raise ValueError("Empty or oversized policy")
    return output


async def evaluate_policy(
    code: str,
    seeds: list[int],
    task: str = "GoToGoal-v0",
    difficulty: str = "easy",
    action_timeout: float = 1.0,
    trial_timeout: float = 60.0,
) -> dict[str, Any]:
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Use a nonempty list of distinct evaluation seeds")
    if action_timeout <= 0 or trial_timeout <= action_timeout:
        raise ValueError("Require 0 < action_timeout < trial_timeout")
    trials = []
    for seed in seeds:
        # Sequential seeds bound CPU use; the standalone driver limits candidates.
        request = {
            "code": code,
            "task": task,
            "difficulty": difficulty,
            "seed": int(seed),
            "action_timeout": action_timeout,
        }
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__).with_name("env_runner.py")),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            env={
                # Do not pass API credentials or CUDA visibility to task processes.
                "PATH": os.environ.get("PATH", os.defpath),
                "LANG": os.environ.get("LANG", "C.UTF-8"),
                "CUDA_VISIBLE_DEVICES": "",
                "SDL_VIDEODRIVER": "dummy",
                "SDL_AUDIODRIVER": "dummy",
                "PYGAME_HIDE_SUPPORT_PROMPT": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(json.dumps(request).encode()),
                timeout=trial_timeout,
            )
        except BaseException:
            # Also terminate the policy child on cancellation/outer deadline.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
            raise
        if process.returncode:
            raise RuntimeError(
                f"Agentick runner failed on seed {seed}: {stderr.decode(errors='replace')[-2000:]}"
            )
        trial = json.loads(stdout)
        if type(trial.get("success")) is not bool:
            raise RuntimeError("Agentick runner returned an invalid result")
        trials.append({"seed": int(seed), **trial})
    return {
        "reward": sum(t["success"] for t in trials) / len(trials),
        "trials": trials,
    }
