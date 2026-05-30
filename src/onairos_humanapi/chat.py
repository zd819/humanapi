"""High-level chat helpers with optional LangSmith traceable spans."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from openai import OpenAI
from openai.types.chat import ChatCompletion

from onairos_humanapi.client import (
    HumanApiConfig,
    build_chat_request_kwargs,
    create_human_api_client,
    create_traced_human_api_client,
)


def chat_completion(
    client: OpenAI,
    *,
    messages: Sequence[Mapping[str, Any]],
    model: str = "openai/gpt-4o-mini",
    user_context: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> ChatCompletion:
    """Run a single Human API chat completion."""
    request_kwargs = build_chat_request_kwargs(
        messages=messages,
        model=model,
        user_context=user_context,
    )
    request_kwargs.update(kwargs)
    return client.chat.completions.create(**request_kwargs)


def chat_completion_traced(
    *,
    api_key: str,
    user_jwt: str,
    messages: Sequence[Mapping[str, Any]],
    model: str = "openai/gpt-4o-mini",
    user_context: Mapping[str, Any] | None = None,
    run_name: str = "human_api_chat",
    config: HumanApiConfig | None = None,
    **kwargs: Any,
) -> ChatCompletion:
    """
    Chat completion inside a LangSmith ``@traceable`` span (when tracing is enabled).

    Creates a traced client automatically.
    """
    if config is None:
        config = HumanApiConfig(api_key=api_key, user_jwt=user_jwt)

    client = create_traced_human_api_client(config, enable_tracing=True)

    try:
        from langsmith import traceable
    except ImportError:
        return chat_completion(
            client,
            messages=messages,
            model=model,
            user_context=user_context,
            **kwargs,
        )

    @traceable(name=run_name, run_type="llm")
    def _run() -> ChatCompletion:
        return chat_completion(
            client,
            messages=messages,
            model=model,
            user_context=user_context,
            **kwargs,
        )

    return _run()
