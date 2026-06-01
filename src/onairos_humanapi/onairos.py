"""Client interface for the Onairos Human API."""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

from openai import OpenAI
from openai.types.chat import ChatCompletion

from onairos_humanapi.chat import chat_completion
from onairos_humanapi.client import (
    DEFAULT_BASE_URL,
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_USER_JWT,
    HumanApiConfig,
    UserContext,
    create_human_api_client,
)


class _ChatCompletions:
    def __init__(self, owner: "Onairos") -> None:
        self._owner = owner

    def create(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        model: str = "openai/gpt-4o-mini",
        user_context: UserContext | None = None,
        enable_tracing: bool | None = None,
        run_name: str = "human_api_chat",
        trace_metadata: Mapping[str, Any] | None = None,
        trace_tags: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Create a Human API chat completion with optional Onairos context."""
        resolved_tracing = self._owner.enable_tracing if enable_tracing is None else enable_tracing
        return chat_completion(
            self._owner.raw_client,
            messages=messages,
            model=model,
            user_context=user_context,
            enable_tracing=resolved_tracing,
            run_name=run_name,
            trace_metadata=trace_metadata,
            trace_tags=trace_tags,
            **kwargs,
        )


class _Chat:
    def __init__(self, owner: "Onairos") -> None:
        self.completions = _ChatCompletions(owner)


class Onairos:
    """
    Human API client.

    Provides ``client.chat.completions.create(...)`` with optional Onairos
    ``user_context`` support.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        user_jwt: str | None = None,
        base_url: str | None = None,
        config: HumanApiConfig | None = None,
        extra_headers: Mapping[str, str] | None = None,
        enable_tracing: bool | None = None,
        **openai_kwargs: Any,
    ) -> None:
        if config is None:
            resolved_key = api_key or os.environ.get(ENV_API_KEY)
            resolved_jwt = user_jwt or os.environ.get(ENV_USER_JWT)
            resolved_base = base_url or os.environ.get(ENV_BASE_URL) or DEFAULT_BASE_URL
            if not resolved_key:
                raise ValueError(f"Missing developer API key: set {ENV_API_KEY} or pass api_key=")
            if not resolved_jwt:
                raise ValueError(f"Missing user JWT: set {ENV_USER_JWT} or pass user_jwt=")
            config = HumanApiConfig(
                api_key=resolved_key,
                user_jwt=resolved_jwt,
                base_url=resolved_base,
                extra_headers=extra_headers or {},
            )

        self.config = config
        self.enable_tracing = enable_tracing
        self.raw_client = create_human_api_client(config, **openai_kwargs)
        self.chat = _Chat(self)

    @classmethod
    def from_env(
        cls,
        *,
        api_key: str | None = None,
        user_jwt: str | None = None,
        base_url: str | None = None,
        enable_tracing: bool | None = None,
        **openai_kwargs: Any,
    ) -> "Onairos":
        """Create an Onairos client from environment variables."""
        return cls(
            api_key=api_key,
            user_jwt=user_jwt,
            base_url=base_url,
            enable_tracing=enable_tracing,
            **openai_kwargs,
        )


__all__ = ["Onairos"]
