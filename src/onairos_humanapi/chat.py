"""High-level chat helpers with optional LangSmith traceable spans."""

from __future__ import annotations

import os
import time
from typing import Any, Mapping, Sequence

from openai import OpenAI
from openai.types.chat import ChatCompletion

from onairos_humanapi.client import (
    HumanApiConfig,
    UserContext,
    build_chat_request_kwargs,
    create_human_api_client,
)


def _compact_text(value: str, limit: int = 360) -> str:
    return " ".join(value.split())[:limit]


def _get_value(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _get_header(headers: Any, name: str) -> str | None:
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if callable(getter):
        return getter(name)
    return None


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_bool(value: Any) -> bool:
    return str(value).lower() == "true"


def _uses_onairos_memory(messages: Sequence[Mapping[str, Any]]) -> bool:
    return any("{onairos_memory}" in str(message.get("content", "")) for message in messages)


def _user_context_type(user_context: UserContext | None) -> str:
    if user_context is None:
        return "none"
    if isinstance(user_context, str):
        return "string"
    return "json"


def _completion_text(completion: ChatCompletion) -> str:
    choices = _get_value(completion, "choices", []) or []
    if not choices:
        return ""
    message = _get_value(choices[0], "message")
    return _get_value(message, "content", "") or ""


def _completion_usage(completion: ChatCompletion) -> dict[str, int]:
    usage = _get_value(completion, "usage")
    return {
        "promptTokens": _as_int(_get_value(usage, "prompt_tokens")),
        "completionTokens": _as_int(_get_value(usage, "completion_tokens")),
        "totalTokens": _as_int(_get_value(usage, "total_tokens")),
    }


def _onairos_headers(headers: Any) -> dict[str, str | None]:
    return {
        "memoryInjected": _get_header(headers, "X-Onairos-Memory-Injected"),
        "memoryCount": _get_header(headers, "X-Onairos-Memory-Count"),
        "messagesEnriched": _get_header(headers, "X-Onairos-Messages-Enriched"),
        "userContextInjected": _get_header(headers, "X-Onairos-User-Context-Injected"),
        "userContextSource": _get_header(headers, "X-Onairos-User-Context-Source"),
        "userContextType": _get_header(headers, "X-Onairos-User-Context-Type"),
        "userContextDeduped": _get_header(headers, "X-Onairos-User-Context-Deduped"),
        "provider": _get_header(headers, "X-Onairos-Provider"),
        "model": _get_header(headers, "X-Onairos-Model"),
        "authMethod": _get_header(headers, "X-Onairos-Auth-Method"),
        "promptTokens": _get_header(headers, "X-Onairos-Prompt-Tokens"),
        "completionTokens": _get_header(headers, "X-Onairos-Completion-Tokens"),
        "totalTokens": _get_header(headers, "X-Onairos-Total-Tokens"),
    }


def _status_code(raw_response: Any) -> int | None:
    status = getattr(raw_response, "status_code", None)
    if status is not None:
        return status
    http_response = getattr(raw_response, "http_response", None)
    return getattr(http_response, "status_code", None)


def _trace_input(
    *,
    messages: Sequence[Mapping[str, Any]],
    model: str,
    user_context: UserContext | None,
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model": model,
        "messageCount": len(messages),
        "usesOnairosMemory": _uses_onairos_memory(messages),
        "hasUserContext": user_context is not None,
        "userContextType": _user_context_type(user_context),
        "maxTokens": kwargs.get("max_tokens") or kwargs.get("max_completion_tokens"),
        "temperature": kwargs.get("temperature"),
        "stream": bool(kwargs.get("stream", False)),
    }


def _trace_output(
    *,
    completion: ChatCompletion,
    headers: Any,
    elapsed_ms: int,
    status: int | None,
    test_case: str | None = None,
) -> dict[str, Any]:
    onairos_headers = _onairos_headers(headers)
    usage = _completion_usage(completion)
    usage = {
        "promptTokens": usage["promptTokens"] or _as_int(onairos_headers["promptTokens"]),
        "completionTokens": usage["completionTokens"] or _as_int(onairos_headers["completionTokens"]),
        "totalTokens": usage["totalTokens"] or _as_int(onairos_headers["totalTokens"]),
    }
    return {
        **({"testCase": test_case} if test_case else {}),
        "status": status,
        "ok": status is None or 200 <= status < 300,
        "elapsedMs": elapsed_ms,
        "provider": onairos_headers["provider"],
        "model": onairos_headers["model"] or _get_value(completion, "model"),
        "memoryInjected": _as_bool(onairos_headers["memoryInjected"]),
        "memoryCount": _as_int(onairos_headers["memoryCount"]),
        "messagesEnriched": _as_int(onairos_headers["messagesEnriched"]),
        "userContextInjected": _as_bool(onairos_headers["userContextInjected"]),
        "userContextSource": onairos_headers["userContextSource"],
        "userContextType": onairos_headers["userContextType"],
        "userContextDeduped": _as_bool(onairos_headers["userContextDeduped"]),
        "authMethod": onairos_headers["authMethod"],
        "usage": usage,
        "headers": onairos_headers,
        "assistantPreview": _compact_text(_completion_text(completion)),
    }


def _tracing_enabled(enable_tracing: bool | None) -> bool:
    if enable_tracing is not None:
        return enable_tracing
    raw = os.environ.get("LANGSMITH_TRACING", "").strip().lower()
    return raw in ("true", "1", "yes", "on")


def _resolve_client(
    client: OpenAI | None,
    *,
    config: HumanApiConfig | None,
    api_key: str | None,
    user_jwt: str | None,
) -> OpenAI:
    if client is not None:
        return client
    if config is None:
        if api_key is None or user_jwt is None:
            raise ValueError("Provide client, HumanApiConfig, or both api_key and user_jwt")
        config = HumanApiConfig(api_key=api_key, user_jwt=user_jwt)
    return create_human_api_client(config)


def _merge_request_kwargs(
    base_kwargs: dict[str, Any],
    override_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(base_kwargs)
    base_headers = merged.get("extra_headers")
    override_headers = override_kwargs.get("extra_headers")
    merged.update(override_kwargs)
    if isinstance(base_headers, Mapping) or isinstance(override_headers, Mapping):
        extra_headers: dict[str, Any] = {}
        if isinstance(base_headers, Mapping):
            extra_headers.update(base_headers)
        if isinstance(override_headers, Mapping):
            extra_headers.update(override_headers)
        merged["extra_headers"] = extra_headers
    return merged


def _run_traced_completion(
    *,
    client: OpenAI,
    request_kwargs: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
    model: str,
    user_context: UserContext | None,
    run_name: str,
    trace_metadata: Mapping[str, Any] | None,
    trace_tags: Sequence[str] | None,
) -> ChatCompletion:
    trace_input = _trace_input(
        messages=messages,
        model=model,
        user_context=user_context,
        kwargs=request_kwargs,
    )
    extra_metadata = dict(trace_metadata or {})
    if extra_metadata.get("test_case"):
        trace_input["testCase"] = extra_metadata["test_case"]

    try:
        from langsmith import traceable
    except ImportError:
        return client.chat.completions.create(**request_kwargs)

    trace_state: dict[str, Any] = {}

    def process_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
        return dict(inputs.get("trace_input", inputs))

    def process_outputs(output: ChatCompletion) -> dict[str, Any]:
        return dict(
            trace_state.get("output")
            or {
                "model": _get_value(output, "model"),
                "usage": _completion_usage(output),
                "assistantPreview": _compact_text(_completion_text(output)),
            }
        )

    tags = ["onairos", "human-api", "openrouter"]
    for tag in trace_tags or []:
        if tag not in tags:
            tags.append(str(tag))

    @traceable(
        name=run_name,
        run_type="llm",
        metadata={
            "service": "onairos-human-api",
            "router": "openrouter",
            "requested_model": model,
            "uses_onairos_memory": trace_input["usesOnairosMemory"],
            "has_user_context": trace_input["hasUserContext"],
            **extra_metadata,
        },
        tags=tags,
        process_inputs=process_inputs,
        process_outputs=process_outputs,
    )
    def _run(trace_input: Mapping[str, Any]) -> ChatCompletion:
        started_at = time.perf_counter()
        raw_response = client.chat.completions.with_raw_response.create(
            **request_kwargs
        )
        completion = raw_response.parse()
        trace_state["output"] = _trace_output(
            completion=completion,
            headers=getattr(raw_response, "headers", None),
            elapsed_ms=round((time.perf_counter() - started_at) * 1000),
            status=_status_code(raw_response),
            test_case=str(trace_input.get("testCase")) if trace_input.get("testCase") else None,
        )
        return completion

    return _run(trace_input)


def chat_completion(
    client: OpenAI | None = None,
    *,
    config: HumanApiConfig | None = None,
    api_key: str | None = None,
    user_jwt: str | None = None,
    messages: Sequence[Mapping[str, Any]],
    model: str = "openai/gpt-4o-mini",
    user_context: UserContext | None = None,
    enable_tracing: bool | None = None,
    run_name: str = "human_api_chat",
    trace_metadata: Mapping[str, Any] | None = None,
    trace_tags: Sequence[str] | None = None,
    **kwargs: Any,
) -> ChatCompletion:
    """
    Run a Human API chat completion.

    Tracing is enabled when ``enable_tracing`` is True, or when
    ``LANGSMITH_TRACING`` is ``true``/``1``/``yes``/``on``.
    """
    resolved_client = _resolve_client(
        client,
        config=config,
        api_key=api_key,
        user_jwt=user_jwt,
    )
    request_kwargs = build_chat_request_kwargs(
        messages=messages,
        model=model,
        user_context=user_context,
    )
    request_kwargs = _merge_request_kwargs(request_kwargs, kwargs)
    if _tracing_enabled(enable_tracing):
        return _run_traced_completion(
            client=resolved_client,
            request_kwargs=request_kwargs,
            messages=messages,
            model=model,
            user_context=user_context,
            run_name=run_name,
            trace_metadata=trace_metadata,
            trace_tags=trace_tags,
        )
    return resolved_client.chat.completions.create(**request_kwargs)


def chat_completion_traced(
    *,
    api_key: str | None = None,
    user_jwt: str | None = None,
    messages: Sequence[Mapping[str, Any]],
    model: str = "openai/gpt-4o-mini",
    user_context: UserContext | None = None,
    run_name: str = "human_api_chat",
    trace_metadata: Mapping[str, Any] | None = None,
    trace_tags: Sequence[str] | None = None,
    config: HumanApiConfig | None = None,
    **kwargs: Any,
) -> ChatCompletion:
    """
    Chat completion with LangSmith tracing forced on.

    Returns the normal OpenAI ``ChatCompletion`` while logging sanitized Human API
    metadata and output fields to LangSmith.
    """
    return chat_completion(
        config=config,
        api_key=api_key,
        user_jwt=user_jwt,
        messages=messages,
        model=model,
        user_context=user_context,
        enable_tracing=True,
        run_name=run_name,
        trace_metadata=trace_metadata,
        trace_tags=trace_tags,
        **kwargs,
    )
