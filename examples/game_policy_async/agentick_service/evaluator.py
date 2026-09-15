# SPDX-License-Identifier: Apache-2.0
"""Execute one game in an isolated subprocess and return its outcome."""

import asyncio
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any


async def run_policy(
    code: str,
    seed: int,
    task: str = "GoToGoal-v0",
    difficulty: str = "easy",
    action_timeout: float = 1.0,
    trial_timeout: float = 120.0,
) -> dict[str, Any]:
    if action_timeout <= 0 or trial_timeout <= action_timeout:
        raise ValueError("Require 0 < action_timeout < trial_timeout")
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
    return {"seed": seed, **trial}
