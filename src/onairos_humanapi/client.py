"""Human API OpenAI client factory and configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Sequence

from openai import OpenAI

DEFAULT_BASE_URL = "https://human-api.onairos.io/v1"
JWT_HEADER_NAME = "x-jwt-token"
ENV_API_KEY = "ONAIROS_API_KEY"
ENV_USER_JWT = "ONAIROS_USER_JWT"
ENV_BASE_URL = "ONAIROS_HUMAN_API_BASE_URL"


@dataclass(frozen=True)
class HumanApiConfig:
    """Credentials and endpoint for Human API."""

    api_key: str
    user_jwt: str
    base_url: str = DEFAULT_BASE_URL
    extra_headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.api_key or not self.api_key.strip():
            raise ValueError("api_key is required (Onairos developer key, e.g. ona_…)")
        if not self.user_jwt or not self.user_jwt.strip():
            raise ValueError("user_jwt is required (from SDK consent / handoff token)")


def _merge_headers(config: HumanApiConfig) -> dict[str, str]:
    headers: dict[str, str] = {JWT_HEADER_NAME: config.user_jwt.strip()}
    headers.update(config.extra_headers)
    return headers


def create_human_api_client(
    config: HumanApiConfig | None = None,
    *,
    api_key: str | None = None,
    user_jwt: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    extra_headers: Mapping[str, str] | None = None,
    **openai_kwargs: Any,
) -> OpenAI:
    """
    Build a standard OpenAI SDK client pointed at Human API.

    Uses the Onairos **developer** API key as ``api_key`` — not an OpenAI key.
    Passes the end-user JWT via ``x-jwt-token``.
    """
    if config is None:
        if api_key is None or user_jwt is None:
            raise ValueError("Provide HumanApiConfig or both api_key and user_jwt")
        config = HumanApiConfig(
            api_key=api_key,
            user_jwt=user_jwt,
            base_url=base_url,
            extra_headers=extra_headers or {},
        )

    return OpenAI(
        api_key=config.api_key.strip(),
        base_url=config.base_url.strip().rstrip("/"),
        default_headers=_merge_headers(config),
        **openai_kwargs,
    )


def create_traced_human_api_client(
    config: HumanApiConfig | None = None,
    *,
    enable_tracing: bool | None = None,
    **kwargs: Any,
) -> OpenAI:
    """
    Human API client wrapped with LangSmith ``wrap_openai`` when tracing is on.

    Tracing is enabled when ``enable_tracing`` is True, or when env
    ``LANGSMITH_TRACING`` is ``true``/``1`` (and ``langsmith`` is installed).
    """
    client = create_human_api_client(config, **kwargs)

    if enable_tracing is False:
        return client

    if enable_tracing is None:
        raw = os.environ.get("LANGSMITH_TRACING", "").strip().lower()
        enable_tracing = raw in ("true", "1", "yes")

    if not enable_tracing:
        return client

    try:
        from langsmith import wrappers
    except ImportError as exc:
        raise ImportError(
            "LangSmith tracing requested but langsmith is not installed. "
            "Install with: pip install onairos-humanapi"
        ) from exc

    return wrappers.wrap_openai(client)


def from_env(
    *,
    api_key: str | None = None,
    user_jwt: str | None = None,
    base_url: str | None = None,
    traced: bool = False,
) -> OpenAI:
    """Load config from environment and return a Human API client."""
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
    )
    if traced:
        return create_traced_human_api_client(config)
    return create_human_api_client(config)


def build_user_context_payload(
    user_context: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not user_context:
        return None
    return {"userContext": dict(user_context)}


def build_chat_request_kwargs(
    *,
    messages: Sequence[Mapping[str, Any]],
    model: str = "openai/gpt-4o-mini",
    user_context: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build kwargs for ``client.chat.completions.create`` including Human API extensions.

    The ``onairos`` block is sent via ``extra_body`` for broad SDK compatibility.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
    }
    onairos = build_user_context_payload(user_context)
    extra_body: MutableMapping[str, Any] = {}
    if onairos:
        extra_body["onairos"] = onairos
    if extra:
        extra_body.update(extra)
    if extra_body:
        payload["extra_body"] = dict(extra_body)
    return payload
