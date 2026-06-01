"""Tests for the Onairos client interface."""

from unittest.mock import MagicMock, patch

import pytest

from onairos_humanapi import Onairos
from onairos_humanapi.client import ENV_API_KEY, ENV_USER_JWT, HumanApiConfig


@patch("onairos_humanapi.onairos.chat_completion")
@patch("onairos_humanapi.onairos.create_human_api_client")
def test_onairos_chat_completions_create(mock_create_client, mock_chat_completion):
    raw_client = MagicMock()
    completion = MagicMock()
    mock_create_client.return_value = raw_client
    mock_chat_completion.return_value = completion

    client = Onairos(
        api_key="ona_key",
        user_jwt="user_jwt",
        enable_tracing=False,
    )
    result = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Use {onairos_memory}"}],
        user_context={"currentPage": "dashboard"},
        temperature=0.2,
    )

    assert result is completion
    config = mock_create_client.call_args.args[0]
    assert isinstance(config, HumanApiConfig)
    assert config.api_key == "ona_key"
    assert config.user_jwt == "user_jwt"

    assert mock_chat_completion.call_args.args == (raw_client,)
    call_kwargs = mock_chat_completion.call_args.kwargs
    assert call_kwargs["model"] == "openai/gpt-4o-mini"
    assert call_kwargs["messages"] == [{"role": "user", "content": "Use {onairos_memory}"}]
    assert call_kwargs["user_context"] == {"currentPage": "dashboard"}
    assert call_kwargs["enable_tracing"] is False
    assert call_kwargs["temperature"] == 0.2


@patch("onairos_humanapi.onairos.create_human_api_client")
def test_onairos_from_env(mock_create_client, monkeypatch):
    monkeypatch.setenv(ENV_API_KEY, "ona_from_env")
    monkeypatch.setenv(ENV_USER_JWT, "jwt_from_env")

    client = Onairos.from_env()

    config = mock_create_client.call_args.args[0]
    assert client.config is config
    assert config.api_key == "ona_from_env"
    assert config.user_jwt == "jwt_from_env"


@patch("onairos_humanapi.onairos.create_human_api_client")
def test_onairos_requires_credentials(mock_create_client, monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    monkeypatch.delenv(ENV_USER_JWT, raising=False)

    with pytest.raises(ValueError, match=ENV_API_KEY):
        Onairos()

    mock_create_client.assert_not_called()
