"""LLM API adapter for DeepSeek via OpenAI-compatible SDK."""

from __future__ import annotations

import os


def request_patch(
    system_prompt: str,
    user_prompt: str,
    model: str,
    reasoning_effort: str = "high",
    thinking_enabled: bool = True,
    base_url: str = "https://api.deepseek.com",
    api_key_env: str = "DEEPSEEK_API_KEY",
) -> str:
    """Request one patch proposal from DeepSeek API."""
    from openai import OpenAI

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Please set it in your environment variables."
        )

    client = OpenAI(api_key=api_key, base_url=base_url)

    extra_body = {"thinking": {"type": "enabled"}} if thinking_enabled else {}
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        reasoning_effort=reasoning_effort,
        extra_body=extra_body,
    )
    return (response.choices[0].message.content or "").strip()
