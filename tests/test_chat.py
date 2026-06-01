"""Tests for chat completion helpers."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from onairos_humanapi.chat import chat_completion, chat_completion_traced
from onairos_humanapi.client import HumanApiConfig


def test_chat_completion_calls_create_with_extra_body(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    client = MagicMock()
    completion = MagicMock()
    client.chat.completions.create.return_value = completion

    result = chat_completion(
        client,
        messages=[{"role": "user", "content": "Use {onairos_memory}"}],
        model="openai/gpt-4o-mini",
        user_context={"currentPage": "workout_builder"},
    )

    assert result is completion
    client.chat.completions.create.assert_called_once()
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "openai/gpt-4o-mini"
    assert call_kwargs["extra_body"]["onairos"]["userContext"]["currentPage"] == "workout_builder"


def test_chat_completion_accepts_string_user_context(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    client = MagicMock()
    completion = MagicMock()
    client.chat.completions.create.return_value = completion

    result = chat_completion(
        client,
        messages=[{"role": "user", "content": "Use current context"}],
        user_context="The user is in a developer console.",
    )

    assert result is completion
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["extra_body"]["onairos"]["userContext"] == "The user is in a developer console."


@patch("onairos_humanapi.chat.create_human_api_client")
def test_chat_completion_auto_tracing_uses_clean_trace_output(mock_create_client, monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    mock_client = MagicMock()
    mock_create_client.return_value = mock_client
    mock_completion = MagicMock()
    mock_completion.model = "openai/gpt-4o-mini"
    mock_completion.usage = MagicMock(
        prompt_tokens=10,
        completion_tokens=3,
        total_tokens=13,
    )
    mock_completion.choices = [
        MagicMock(
            message=MagicMock(content="A useful assistant response."),
        )
    ]
    mock_raw_response = MagicMock(
        status_code=200,
        headers={
            "X-Onairos-Memory-Injected": "true",
            "X-Onairos-Memory-Count": "15",
            "X-Onairos-Messages-Enriched": "1",
            "X-Onairos-User-Context-Injected": "true",
            "X-Onairos-User-Context-Source": "request",
            "X-Onairos-User-Context-Type": "json",
            "X-Onairos-User-Context-Deduped": "false",
            "X-Onairos-Provider": "openai",
            "X-Onairos-Model": "openai/gpt-4o-mini",
            "X-Onairos-Auth-Method": "dual-credential",
        },
    )
    mock_raw_response.parse.return_value = mock_completion
    mock_client.chat.completions.with_raw_response.create.return_value = mock_raw_response
    captured = {}

    def traceable_decorator(**_decorator_kwargs):
        captured.update(_decorator_kwargs)

        def wrapper(fn):
            def wrapped(*args, **kwargs):
                result = fn(*args, **kwargs)
                captured["processed_inputs"] = _decorator_kwargs["process_inputs"](
                    {"trace_input": args[0]}
                )
                captured["processed_outputs"] = _decorator_kwargs["process_outputs"](
                    result
                )
                return result

            return wrapped

        return wrapper

    mock_langsmith = MagicMock()
    mock_langsmith.traceable.side_effect = traceable_decorator

    with patch.dict(sys.modules, {"langsmith": mock_langsmith}):
        result = chat_completion(
            config=HumanApiConfig(api_key="ona_key", user_jwt="user_jwt"),
            messages=[{"role": "user", "content": "Hello"}],
            user_context={"currentGoal": "cardio"},
            trace_metadata={"test_case": "unit-test"},
            trace_tags=["e2e", "unit-test"],
        )

    assert result is mock_completion
    mock_create_client.assert_called_once()
    mock_client.chat.completions.with_raw_response.create.assert_called_once()

    call_kwargs = mock_client.chat.completions.with_raw_response.create.call_args.kwargs
    assert call_kwargs["extra_body"]["onairos"]["userContext"]["currentGoal"] == "cardio"

    assert captured["metadata"]["service"] == "onairos-human-api"
    assert captured["metadata"]["router"] == "openrouter"
    assert captured["metadata"]["test_case"] == "unit-test"
    assert captured["tags"] == ["onairos", "human-api", "openrouter", "e2e", "unit-test"]
    assert captured["processed_inputs"] == {
        "model": "openai/gpt-4o-mini",
        "messageCount": 1,
        "usesOnairosMemory": False,
        "hasUserContext": True,
        "userContextType": "json",
        "maxTokens": None,
        "temperature": None,
        "stream": False,
        "testCase": "unit-test",
    }
    assert captured["processed_outputs"]["provider"] == "openai"
    assert captured["processed_outputs"]["memoryInjected"] is True
    assert captured["processed_outputs"]["memoryCount"] == 15
    assert captured["processed_outputs"]["userContextInjected"] is True
    assert captured["processed_outputs"]["usage"] == {
        "promptTokens": 10,
        "completionTokens": 3,
        "totalTokens": 13,
    }


@patch("onairos_humanapi.chat.chat_completion")
def test_chat_completion_traced_forces_tracing(mock_chat_completion):
    completion = MagicMock()
    mock_chat_completion.return_value = completion

    result = chat_completion_traced(
        config=HumanApiConfig(api_key="ona_key", user_jwt="user_jwt"),
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert result is completion
    assert mock_chat_completion.call_args.kwargs["enable_tracing"] is True
