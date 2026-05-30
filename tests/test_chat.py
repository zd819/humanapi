"""Tests for chat completion helpers."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from onairos_humanapi.chat import chat_completion, chat_completion_traced
from onairos_humanapi.client import HumanApiConfig


def test_chat_completion_calls_create_with_extra_body():
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


@patch("onairos_humanapi.chat.create_traced_human_api_client")
def test_chat_completion_traced_uses_traceable(mock_create_client):
    mock_client = MagicMock()
    mock_create_client.return_value = mock_client
    mock_completion = MagicMock()
    mock_client.chat.completions.create.return_value = mock_completion

    def traceable_decorator(**_decorator_kwargs):
        def wrapper(fn):
            return fn

        return wrapper

    mock_langsmith = MagicMock()
    mock_langsmith.traceable.side_effect = traceable_decorator

    with patch.dict(sys.modules, {"langsmith": mock_langsmith}):
        result = chat_completion_traced(
            api_key="ona_key",
            user_jwt="user_jwt",
            messages=[{"role": "user", "content": "Hello"}],
            user_context={"currentGoal": "cardio"},
        )

    assert result is mock_completion
    mock_create_client.assert_called_once()
