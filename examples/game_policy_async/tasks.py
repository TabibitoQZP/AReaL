# SPDX-License-Identifier: Apache-2.0
"""Fixed task-level split; no task or difficulty crosses the train/test boundary."""

import json
import random
from pathlib import Path
from typing import Any

AGENTICK_REVISION = "ddbad3b205e6aa331a7d985873df6cbd5f381600"
DIFFICULTIES = ("easy", "medium", "hard", "expert")
# Each category explicitly lists (train tasks, test tasks). Keep this split fixed
# across checkpoints and baselines; never choose held-out tasks by their scores.
TASK_SPLIT = {
    "navigation": (
        (
            "GoToGoal-v0",
            "MazeNavigation-v0",
            "ShortestPath-v0",
            "DynamicObstacles-v0",
            "CuriosityMaze-v0",
            "TimingChallenge-v0",
            "InstructionFollowing-v0",
        ),
        ("RecursiveRooms-v0",),
    ),
    "planning": (
        (
            "KeyDoorPuzzle-v0",
            "BacktrackPuzzle-v0",
            "SokobanPush-v0",
            "RecipeAssembly-v0",
            "ResourceManagement-v0",
            "PreciseNavigation-v0",
            "TileSorting-v0",
        ),
        ("ToolUse-v0", "PackingPuzzle-v0"),
    ),
    "reasoning": (
        (
            "SwitchCircuit-v0",
            "RuleInduction-v0",
            "SymbolMatching-v0",
            "LightsOut-v0",
            "GraphColoring-v0",
            "DeceptiveReward-v0",
            "TaskInterference-v0",
        ),
        ("ProgramSynthesis-v0",),
    ),
    "memory": (
        ("FogOfWarExploration-v0", "TreasureHunt-v0", "DelayedGratification-v0"),
        ("SequenceMemory-v0",),
    ),
    "generalization": (
        ("NoisyObservation-v0", "DistributionShift-v0"),
        ("FewShotAdaptation-v0",),
    ),
    "multi_agent": (
        ("EmergentStrategy-v0", "Herding-v0", "ChaseEvade-v0", "TagHunt-v0"),
        ("CooperativeTransport-v0",),
    ),
}
TASKS = {
    task: (category, split)
    for category, partitions in TASK_SPLIT.items()
    for split, names in zip(("train", "test"), partitions, strict=True)
    for task in names
}


def training_schedule(count: int, seed: int) -> list[tuple[str, str]]:
    """Equal category/difficulty counts; shuffled round-robin tasks within each."""
    if count <= 0 or count % (len(TASK_SPLIT) * len(DIFFICULTIES)):
        raise ValueError("train-groups must be a positive multiple of 24")
    rng = random.Random(seed)
    schedule = []
    for _, (names, _) in TASK_SPLIT.items():
        for difficulty in DIFFICULTIES:
            bag = []
            for _ in range(count // 24):
                if not bag:
                    bag = list(names)
                    rng.shuffle(bag)
                schedule.append((bag.pop(), difficulty))
    rng.shuffle(schedule)
    return schedule


def official_instructions() -> dict[tuple[str, str], str]:
    """Use upstream text verbatim, selecting only the current difficulty."""
    import agentick
    from agentick.tasks.descriptions import get_task_description_structured

    if set(agentick.list_tasks()) != set(TASKS):
        raise RuntimeError(
            "Agentick task registry differs from the pinned 37-task split"
        )
    result = {}
    for task, (category, _) in TASKS.items():
        desc = get_task_description_structured(task)
        if desc is None or desc.category != category:
            raise RuntimeError(f"Missing or changed official description: {task}")
        difficulties = {d.level: d.description for d in desc.difficulty_descriptions}
        if set(difficulties) != set(DIFFICULTIES):
            raise RuntimeError(f"Expected four difficulty descriptions: {task}")
        for difficulty in DIFFICULTIES:
            result[task, difficulty] = "\n".join(
                [
                    desc.summary,
                    f"Objects: {desc.objects}",
                    f"Goal: {desc.goal}",
                    f"Actions: {desc.actions}",
                    f"Difficulty ({difficulty}): {difficulties[difficulty]}",
                ]
            )
    return result


def load_rows(path: Path, split: str) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return validate_rows(rows, split)


def validate_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split not in ("train", "test"):
        raise ValueError("Unknown split")
    if not rows:
        raise ValueError("Dataset is empty")
    ids: set[str] = set()
    feedback_seeds: set[int] = set()
    score_seeds: set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Dataset rows must be objects")
        if (row.get("category"), split) != TASKS.get(row.get("task")):
            raise ValueError(f"Task is not in the fixed {split} split")
        if row.get("split") != split or row.get("revision") != AGENTICK_REVISION:
            raise ValueError("Dataset split or Agentick revision does not match")
        if row.get("difficulty") not in DIFFICULTIES:
            raise ValueError("Unknown difficulty")
        if (
            not isinstance(row.get("instruction"), str)
            or not row["instruction"].strip()
        ):
            raise ValueError("Missing official instruction snapshot")
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in ids:
            raise ValueError("Sample IDs must be nonempty and unique")
        ids.add(row["id"])
        for field, collected in (
            ("feedback_seeds", feedback_seeds),
            ("score_seeds", score_seeds),
        ):
            seeds = row.get(field)
            if (
                not isinstance(seeds, list)
                or not seeds
                or any(type(s) is not int or s < 0 for s in seeds)
                or len(set(seeds)) != len(seeds)
            ):
                raise ValueError(f"{field} must contain distinct nonnegative integers")
            collected.update(seeds)
    if feedback_seeds & score_seeds:
        raise ValueError("Feedback and final scoring seeds must be disjoint")
    return rows
