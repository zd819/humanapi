"""Tests for Human API client factory."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from onairos_humanapi.client import (
    DEFAULT_BASE_URL,
    ENV_API_KEY,
    ENV_USER_JWT,
    JWT_HEADER_NAME,
    HumanApiConfig,
    build_chat_request_kwargs,
    create_human_api_client,
    create_traced_human_api_client,
    from_env,
)


@pytest.fixture
def sample_config():
    return HumanApiConfig(
        api_key="ona_test_developer_key",
        user_jwt="eyJhbG.test.user.token",
    )


def test_human_api_config_requires_api_key():
    with pytest.raises(ValueError, match="api_key"):
        HumanApiConfig(api_key="", user_jwt="jwt")


def test_human_api_config_requires_user_jwt():
    with pytest.raises(ValueError, match="user_jwt"):
        HumanApiConfig(api_key="ona_x", user_jwt="")


@patch("onairos_humanapi.client.OpenAI")
def test_create_human_api_client(mock_openai, sample_config):
    create_human_api_client(sample_config)

    mock_openai.assert_called_once()
    call_kwargs = mock_openai.call_args.kwargs
    assert call_kwargs["api_key"] == "ona_test_developer_key"
    assert call_kwargs["base_url"] == DEFAULT_BASE_URL
    assert call_kwargs["default_headers"][JWT_HEADER_NAME] == "eyJhbG.test.user.token"


@patch("onairos_humanapi.client.OpenAI")
def test_create_human_api_client_keyword_args(mock_openai):
    create_human_api_client(
        api_key="ona_key",
        user_jwt="user_jwt_value",
        base_url="https://human-api.onairos.io/v1",
    )

    call_kwargs = mock_openai.call_args.kwargs
    assert call_kwargs["api_key"] == "ona_key"
    assert call_kwargs["default_headers"][JWT_HEADER_NAME] == "user_jwt_value"


def test_build_chat_request_kwargs_includes_onairos_extra_body():
    kwargs = build_chat_request_kwargs(
        messages=[{"role": "user", "content": "Hello"}],
        model="openai/gpt-4o-mini",
        user_context={"currentPage": "home", "currentGoal": "cardio"},
    )

    assert kwargs["model"] == "openai/gpt-4o-mini"
    assert kwargs["messages"] == [{"role": "user", "content": "Hello"}]
    assert kwargs["extra_body"]["onairos"]["userContext"] == {
        "currentPage": "home",
        "currentGoal": "cardio",
    }


def test_build_chat_request_kwargs_without_user_context():
    kwargs = build_chat_request_kwargs(
        messages=[{"role": "user", "content": "Hi"}],
    )
    assert "extra_body" not in kwargs


@patch("onairos_humanapi.client.OpenAI")
def test_from_env(mock_openai, monkeypatch):
    monkeypatch.setenv(ENV_API_KEY, "ona_from_env")
    monkeypatch.setenv(ENV_USER_JWT, "jwt_from_env")

    from_env()

    call_kwargs = mock_openai.call_args.kwargs
    assert call_kwargs["api_key"] == "ona_from_env"
    assert call_kwargs["default_headers"][JWT_HEADER_NAME] == "jwt_from_env"


def test_from_env_missing_api_key(monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    monkeypatch.setenv(ENV_USER_JWT, "jwt")

    with pytest.raises(ValueError, match=ENV_API_KEY):
        from_env()


@patch("onairos_humanapi.client.OpenAI")
def test_create_traced_human_api_client_disabled(mock_openai, sample_config):
    client = create_traced_human_api_client(sample_config, enable_tracing=False)
    assert client is mock_openai.return_value
    mock_openai.assert_called_once()


@patch("onairos_humanapi.client.OpenAI")
def test_create_traced_human_api_client_enabled(mock_openai, sample_config):
    wrapped = MagicMock()
    mock_wrap_openai = MagicMock(return_value=wrapped)
    mock_langsmith = MagicMock()
    mock_langsmith.wrappers.wrap_openai = mock_wrap_openai

    with patch.dict(sys.modules, {"langsmith": mock_langsmith}):
        client = create_traced_human_api_client(sample_config, enable_tracing=True)

    mock_wrap_openai.assert_called_once_with(mock_openai.return_value)
    assert client is wrapped


@patch("onairos_humanapi.client.OpenAI")
def test_create_traced_from_env_flag(mock_openai, monkeypatch):
    monkeypatch.setenv(ENV_API_KEY, "ona_x")
    monkeypatch.setenv(ENV_USER_JWT, "jwt_x")

    with patch("onairos_humanapi.client.create_traced_human_api_client") as mock_traced:
        mock_traced.return_value = MagicMock()
        from_env(traced=True)
        mock_traced.assert_called_once()
