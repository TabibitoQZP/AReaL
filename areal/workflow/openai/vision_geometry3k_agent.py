# SPDX-License-Identifier: Apache-2.0

import copy
import os
import re

from mathruler.grader import extract_boxed_content, grade_answer
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from areal.api import AsyncRewardWrapper
from areal.utils.image import image2base64

_FORMAT_RE = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
_FORMAT_WEIGHT = 0.1


def _format_reward(predict_str: str) -> float:
    return 1.0 if _FORMAT_RE.fullmatch(predict_str) else 0.0


def _acc_reward(predict_str: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict_str)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def geometry3k_reward_fn(completions: str, answer: str) -> float:
    format_val = _format_reward(completions)
    acc_val = _acc_reward(completions, answer)
    return (1.0 - _FORMAT_WEIGHT) * acc_val + _FORMAT_WEIGHT * format_val


def _fill_image_urls(messages_chat: list, images_b64: list[str]) -> list:
    """Fill Geometry3K image placeholders with OpenAI-compatible data URLs."""
    filled = copy.deepcopy(messages_chat)
    images = iter(images_b64)
    for message in filled:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not (isinstance(part, dict) and part.get("type") == "image_url"):
                continue
            image_url = part.setdefault("image_url", {})
            if image_url.get("url"):
                continue
            image_b64 = next(images, None)
            if image_b64 is None:
                raise ValueError("Geometry3K message has more image slots than images.")
            image_url["url"] = f"data:image/png;base64,{image_b64}"
    return filled


class VisionGeometry3KAgent:
    """Single-turn Geometry3K agent for v2 OpenAI-compatible rollouts."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs.copy()
        self.kwargs.pop("max_tokens", None)
        self.kwargs.pop("max_turns", None)
        self._reward_fn = AsyncRewardWrapper(geometry3k_reward_fn)

    async def run(self, data: dict, **extra_kwargs):
        http_client = extra_kwargs.get("http_client")
        base_url = extra_kwargs.get("base_url") or os.getenv("OPENAI_BASE_URL")
        api_key = extra_kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")

        messages = _fill_image_urls(data["messages_chat"], image2base64(data["images"]))
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
            max_retries=0,
        )
        completion: ChatCompletion = await client.chat.completions.create(
            messages=messages,
            model="default",
            **self.kwargs,
        )
        return await self._reward_fn(
            completions=completion.choices[0].message.content,
            answer=data["answer"],
        )
