# SPDX-License-Identifier: Apache-2.0
"""Task/seed partition regressions for standalone preparation."""

import json
from collections import Counter

import pytest

from examples.game_policy_async import prepare_data
from examples.game_policy_async.tasks import (
    AGENTICK_REVISION,
    DIFFICULTIES,
    TASK_SPLIT,
    TASKS,
    load_rows,
    training_schedule,
)


def row(split="train"):
    task = "GoToGoal-v0" if split == "train" else "RecursiveRooms-v0"
    return {
        "id": "case",
        "split": split,
        "revision": AGENTICK_REVISION,
        "task": task,
        "category": "navigation",
        "difficulty": "easy",
        "instruction": "Official fixture instruction",
        "feedback_seeds": [10, 11],
        "score_seeds": [20, 21],
    }


async def ignore(event):
    pass


def test_fixed_split_and_balanced_schedule():
    assert len(TASKS) == 37
    assert Counter(split for _, split in TASKS.values()) == {"train": 30, "test": 7}
    assert sum(len(a) + len(b) for a, b in TASK_SPLIT.values()) == 37
    schedule = training_schedule(384, 7)
    assert schedule == training_schedule(384, 7)
    assert schedule != training_schedule(384, 8)
    counts = Counter((TASKS[t][0], d) for t, d in schedule)
    assert len(counts) == 24 and set(counts.values()) == {16}
    for category, (names, _) in TASK_SPLIT.items():
        for difficulty in DIFFICULTIES:
            task_counts = Counter(
                t for t, d in schedule if d == difficulty and TASKS[t][0] == category
            )
            assert set(task_counts) == set(names)
            assert max(task_counts.values()) - min(task_counts.values()) <= 1


def test_data_generation_partitions_and_seed_roles(monkeypatch, tmp_path):
    monkeypatch.setattr(
        prepare_data,
        "official_instructions",
        lambda: {(t, d): f"{t}: {d}" for t in TASKS for d in DIFFICULTIES},
    )
    output = tmp_path / "data"
    prepare_data.prepare(output, train_groups=24, test_repeats=1)
    train, test = [load_rows(output / f"{s}.jsonl", s) for s in ("train", "test")]
    assert len(train) == 24 and len(test) == 28
    sets = [
        {seed for r in rows for seed in r[field]}
        for rows in (train, test)
        for field in ("feedback_seeds", "score_seeds")
    ]
    for i, seeds in enumerate(sets):
        assert all(not seeds & other for other in sets[i + 1 :])
    with pytest.raises(FileExistsError):
        prepare_data.prepare(output, train_groups=24)


@pytest.mark.parametrize(
    "change",
    [
        {"task": "RecursiveRooms-v0"},
        {"split": "test"},
        {"difficulty": "invalid"},
        {"revision": "other"},
        {"score_seeds": [10]},
        {"score_seeds": [True]},
    ],
)
def test_loader_rejects_split_leakage_and_invalid_data(tmp_path, change):
    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({**row(), **change}) + "\n")
    with pytest.raises(ValueError):
        load_rows(data, "train")
