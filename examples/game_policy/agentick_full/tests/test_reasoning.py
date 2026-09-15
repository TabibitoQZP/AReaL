# SPDX-License-Identifier: Apache-2.0
"""Qwen3 final-answer extraction without changing the training response."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import agent
import httpx
import pytest
from evaluator import extract_policy
from rl_client import FrozenClient, RLClient, Session

CODE = "def act(obs, memory):\n    return 0, memory"
FENCED = f"```python\n{CODE}\n```"
ROW = {
    "task": "GoToGoal-v0",
    "difficulty": "easy",
    "instruction": "Reach the goal",
    "feedback_seeds": [10],
    "score_seeds": [20],
}


@pytest.mark.parametrize("answer", [CODE, FENCED, "STOP"])
@pytest.mark.parametrize(
    "prefix", ["", "<think>Plan a route.</think>\n", "\n<think>\n</think>\n"]
)
def test_policy_answer_preserves_final_answer(answer, prefix):
    """Both raw thinking and already-separated content use the same protocol."""
    assert agent.policy_answer(prefix + answer) == answer


@pytest.mark.parametrize(
    "output",
    [
        "<think>unfinished",
        "<think>```python\n" + CODE + "\n```",
        "<think>STOP",
        "<think>done</think>\n",
        "<think><think>nested</think>" + FENCED,
        "<think>one</think><think>two</think>" + FENCED,
        "<think>one</think></think>" + FENCED,
    ],
)
def test_policy_answer_rejects_missing_or_ambiguous_final_answer(output):
    """Do not salvage code or STOP from unfinished reasoning."""
    with pytest.raises(ValueError):
        agent.policy_answer(output)


def test_policy_answer_preserves_code_literals_and_indentation():
    """Thinking-like strings inside the actual policy are not stripped."""
    code = 'def act(obs, memory):\n    return 0, "<think>literal</think>"'
    for prefix in ("", "<think>Plan.</think>\n"):
        assert extract_policy(agent.policy_answer(prefix + code)) == code


def test_reasoning_is_not_counted_toward_policy_byte_limit():
    """Only the extracted code is subject to the existing 32 KiB code limit."""
    assert (
        extract_policy(
            agent.policy_answer("<think>" + "x" * 40000 + "</think>" + FENCED)
        )
        == CODE
    )
    with pytest.raises(ValueError, match="oversized"):
        extract_policy(
            agent.policy_answer("<think>ok</think>" + CODE + " " * 32768 + "# end")
        )


@pytest.mark.parametrize("client_kind", ["train", "frozen"])
@pytest.mark.parametrize("separated", [False, True])
def test_http_refinement_preserves_raw_response_but_executes_only_answer(
    monkeypatch, client_kind, separated
):
    """Real client parsing, feedback, and STOP work for both response formats."""
    requests, events = [], []
    answers = [FENCED, "STOP"]
    outputs = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        answer = answers[len(requests) - 1]
        output = answer if separated else "<think>private plan</think>\n" + answer
        outputs.append(output)
        message = {"content": output}
        if separated:
            message["reasoning_content"] = "private plan"
        return httpx.Response(
            200, json={"id": f"c{len(requests)}", "choices": [{"message": message}]}
        )

    async def record(event):
        events.append(event)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
            if client_kind == "train":
                client, session = (
                    RLClient("http://gateway.test", http),
                    Session("s", "test-key"),
                )
            else:
                client, session = (
                    FrozenClient("http://frozen.test/v1", "test-key", http),
                    None,
                )
            return await agent.refine(client, session, ROW, record, 4)

    evaluate = AsyncMock(return_value={"reward": 1.0, "trials": []})
    monkeypatch.setattr(agent, "evaluate_policy", evaluate)
    result = asyncio.run(run())
    assert result["policy"] == CODE and result["stop_reason"] == "stop"
    assert result["calls"] == 2 and result["interaction_ids"] == ["c1", "c2"]
    assert [c.args[:2] for c in evaluate.await_args_list] == [
        (CODE, [10]),
        (CODE, [20]),
    ]
    assert [e["output"] for e in events if e["status"] == "generated"] == outputs
    state = json.loads(requests[1]["messages"][-1]["content"])
    assert state["previous_policy"] == FENCED
    assert "private plan" not in json.dumps(state)


@pytest.mark.parametrize("invalid", ["<think>unfinished", "<think>done</think>", ""])
def test_invalid_final_reasoning_does_not_reuse_older_code(monkeypatch, invalid):
    """Reasoning-only final turns produce zero, never best-so-far reward."""
    client = SimpleNamespace(
        generate=AsyncMock(side_effect=[("c1", CODE), ("c2", invalid)])
    )
    evaluate = AsyncMock(return_value={"reward": 1.0, "trials": []})
    monkeypatch.setattr(agent, "evaluate_policy", evaluate)
    result = asyncio.run(agent.refine(client, None, ROW, AsyncMock(), 2))
    assert result["policy"] is None and result["evaluation"]["reward"] == 0.0
    assert result["evaluation"]["format_error"]
    assert evaluate.await_count == 1


@pytest.mark.parametrize("first", ["<think>unfinished", "<think>done</think>STOP"])
def test_invalid_initial_reasoning_is_repairable_with_feedback(monkeypatch, first):
    """A failed first call consumes a round and receives format feedback."""
    client = SimpleNamespace(
        generate=AsyncMock(side_effect=[("c1", first), ("c2", FENCED)])
    )
    evaluate = AsyncMock(return_value={"reward": 1.0, "trials": []})
    monkeypatch.setattr(agent, "evaluate_policy", evaluate)
    result = asyncio.run(agent.refine(client, None, ROW, AsyncMock(), 2))
    state = json.loads(client.generate.await_args_list[-1].args[1][-1]["content"])
    assert "<think>" not in state["previous_policy"]
    assert state["previous_feedback"]["format_error"]
    assert result["policy"] == CODE and result["calls"] == 2
    assert evaluate.await_args_list[-1].args[:2] == (CODE, [20])
