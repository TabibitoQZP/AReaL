# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

_DATA_URI_RE = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,(.+)$", re.DOTALL)


def iter_image_urls(messages: list[dict[str, Any]]) -> list[str]:
    """Return image URLs from normalized OpenAI-format messages."""
    urls: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url", {})
            url = image_url.get("url", "") if isinstance(image_url, dict) else ""
            if not url:
                raise ValueError(
                    "image_url content part has an empty or missing URL. "
                    "Provide a valid data URI or HTTP(S) URL in image_url.url."
                )
            urls.append(url)
    return urls


def extract_images_from_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract backend image data and tokenizer/vLLM message variants."""
    urls = iter(iter_image_urls(messages))
    image_data: list[str] = []
    messages_for_tokenizer: list[dict[str, Any]] = []
    vision_messages_for_vllm: list[dict[str, Any]] = []

    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            messages_for_tokenizer.append(deepcopy(message))
            vision_messages_for_vllm.append(deepcopy(message))
            continue

        tokenizer_parts: list[Any] = []
        vllm_parts: list[Any] = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                tokenizer_parts.append(deepcopy(part))
                vllm_parts.append(deepcopy(part))
                continue

            url = next(urls)
            match = _DATA_URI_RE.match(url)
            image_data.append(match.group(1) if match else url)
            tokenizer_parts.append({"type": "image"})
            vllm_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": "placeholder"},
                }
            )

        messages_for_tokenizer.append({**message, "content": tokenizer_parts})
        vision_messages_for_vllm.append({**message, "content": vllm_parts})

    return image_data, messages_for_tokenizer, vision_messages_for_vllm


def parse_tool_call_arguments(messages: list[dict]) -> list[dict]:
    """Parse JSON tool-call arguments for templates that expect mappings."""
    result = []
    for message in messages:
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not tool_calls:
            result.append(message)
            continue

        new_tool_calls = []
        modified = False
        for tool_call in tool_calls:
            function = (
                tool_call.get("function") if isinstance(tool_call, dict) else None
            )
            if function is None:
                new_tool_calls.append(tool_call)
                continue

            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                    tool_call = {
                        **tool_call,
                        "function": {**function, "arguments": arguments},
                    }
                    modified = True
                except (json.JSONDecodeError, TypeError):
                    pass
            new_tool_calls.append(tool_call)

        result.append(
            {**message, "tool_calls": new_tool_calls} if modified else message
        )
    return result
