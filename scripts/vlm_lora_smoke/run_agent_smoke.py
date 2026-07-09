# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os

from PIL import Image, ImageDraw

from areal.workflow.openai.vision_geometry3k_agent import VisionGeometry3KAgent


async def main() -> None:
    base_url = os.environ["DATA_PROXY_BASE_URL"]
    api_key = os.environ.get("ADMIN_API_KEY", "areal-admin-key")

    img = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, 184, 184], fill="red")

    data = {
        "images": [img],
        "messages_chat": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": ""}},
                    {
                        "type": "text",
                        "text": (
                            "What is the color of the large square in the image? "
                            "Answer with reasoning in <think></think> and put the "
                            "final answer in \\boxed{}."
                        ),
                    },
                ],
            }
        ],
        "answer": "red",
    }

    agent = VisionGeometry3KAgent(max_completion_tokens=64, temperature=0.0)
    reward = await agent.run(data, base_url=base_url, api_key=api_key)
    print(f"agent_reward: {reward}")


if __name__ == "__main__":
    asyncio.run(main())
