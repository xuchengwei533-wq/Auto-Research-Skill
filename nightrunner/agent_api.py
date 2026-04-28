"""LLM API adapter for DeepSeek via OpenAI-compatible SDK."""

from __future__ import annotations

import os
import time


def request_patch(
    system_prompt: str,
    user_prompt: str,
    model: str,
    reasoning_effort: str = "high",
    thinking_enabled: bool = True,
) -> str:
    """Request one experiment proposal from DeepSeek API."""
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set. Please configure it in environment variables.")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    extra_body = {"thinking": {"type": "enabled"}} if thinking_enabled else None
    retry_delays = [0, 5, 15, 30]
    last_exc: Exception | None = None
    for delay in retry_delays:
        if delay > 0:
            time.sleep(delay)
        try:
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
        except Exception as exc:  # pragma: no cover
            last_exc = exc
    raise RuntimeError(f"DeepSeek API request failed after retries: {last_exc}") from last_exc
