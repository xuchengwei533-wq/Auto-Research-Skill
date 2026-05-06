"""LLM API adapter for DeepSeek via OpenAI-compatible SDK."""

from __future__ import annotations

import os
import time
from time import perf_counter
from typing import Callable

from .auth_store import load_api_key, load_auth_config


def resolve_api_settings(
    api_key_env: str = "DEEPSEEK_API_KEY",
    provider: str = "deepseek",
    base_url: str = "https://api.deepseek.com",
) -> tuple[str, str]:
    """Resolve API key and base URL from environment first, then user config."""
    env_api_key = os.environ.get(api_key_env)
    if env_api_key:
        return env_api_key, base_url

    provider_name = (provider or "deepseek").lower()
    if provider_name == "deepseek":
        stored_api_key = load_api_key("deepseek")
    else:
        stored_api_key = None
    if stored_api_key:
        auth_cfg = load_auth_config()
        provider_cfg = auth_cfg.get("providers", {}).get("deepseek", {})
        stored_base_url = (
            provider_cfg.get("base_url")
            if isinstance(provider_cfg, dict)
            else None
        )
        return stored_api_key, str(stored_base_url or base_url)

    raise RuntimeError(
        "No API key found.\n"
        "Run `nightrunner auth login` or set DEEPSEEK_API_KEY."
    )


def request_patch(
    system_prompt: str,
    user_prompt: str,
    model: str,
    base_url: str = "https://api.deepseek.com",
    api_key_env: str = "DEEPSEEK_API_KEY",
    reasoning_effort: str = "high",
    thinking_enabled: bool = True,
    on_start: Callable[[], None] | None = None,
    on_success: Callable[[float], None] | None = None,
    on_retry: Callable[[int, int, int, str], None] | None = None,
) -> str:
    """Request one experiment proposal from DeepSeek API."""
    from openai import OpenAI

    api_key, resolved_base_url = resolve_api_settings(
        api_key_env=api_key_env,
        provider="deepseek",
        base_url=base_url,
    )

    client = OpenAI(api_key=api_key, base_url=resolved_base_url)
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
