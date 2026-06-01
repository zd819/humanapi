"""Onairos Human API — OpenAI-compatible client with LangSmith tracing."""

from onairos_humanapi.client import (
    DEFAULT_BASE_URL,
    JWT_HEADER_NAME,
    HumanApiConfig,
    build_chat_request_kwargs,
    create_human_api_client,
    create_traced_human_api_client,
    from_env,
)
from onairos_humanapi.chat import chat_completion, chat_completion_traced
from onairos_humanapi.onairos import Onairos

__all__ = [
    "Onairos",
    "DEFAULT_BASE_URL",
    "JWT_HEADER_NAME",
    "HumanApiConfig",
    "build_chat_request_kwargs",
    "create_human_api_client",
    "create_traced_human_api_client",
    "from_env",
    "chat_completion",
    "chat_completion_traced",
]

__version__ = "0.1.0"
