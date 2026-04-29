"""LLM API adapter for DeepSeek via OpenAI-compatible SDK."""

from __future__ import annotations

import os
import time
from time import perf_counter
from typing import Callable


def request_patch(
    system_prompt: str,
    user_prompt: str,
    model: str,
    reasoning_effort: str = "high",
    thinking_enabled: bool = True,
    on_start: Callable[[], None] | None = None,
    on_success: Callable[[float], None] | None = None,
    on_retry: Callable[[int, int, int, str], None] | None = None,
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
    started = False
    t0 = perf_counter()
    for idx, delay in enumerate(retry_delays):
        if not started:
            started = True
            if on_start:
                on_start()
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
            if on_success:
                on_success(perf_counter() - t0)
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            if idx < len(retry_delays) - 1 and on_retry:
                next_wait = retry_delays[idx + 1]
                on_retry(idx + 1, 3, next_wait, str(exc))
    raise RuntimeError(f"DeepSeek API request failed after retries: {last_exc}") from last_exc
