"""LLM API adapter for DeepSeek via OpenAI-compatible SDK."""

from __future__ import annotations

import os


def request_patch(
    system_prompt: str,
    user_prompt: str,
    model: str,
    reasoning_effort: str = "high",
    thinking_enabled: bool = True,
) -> str:
    """向 DeepSeek API 请求一次实验提案。"""
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("未检测到 DEEPSEEK_API_KEY，请先在环境变量中设置。")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    extra_body = {"thinking": {"type": "enabled"}} if thinking_enabled else None
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
    except Exception as exc:
        raise RuntimeError(f"DeepSeek API 请求失败: {exc}") from exc
    return (response.choices[0].message.content or "").strip()
