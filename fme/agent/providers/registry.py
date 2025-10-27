"""Factory helpers for constructing provider clients."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Type

from .base import LLMClient
from .claude_client import ClaudeClient
from .errors import ConfigurationError
from .gemini_client import GeminiClient
from .openai_client import OpenAIClient

_PROVIDER_MAP: Dict[str, Type[LLMClient]] = {
    OpenAIClient.provider_name: OpenAIClient,
    ClaudeClient.provider_name: ClaudeClient,
    GeminiClient.provider_name: GeminiClient,
}


def get_available_providers() -> Iterable[str]:
    """Return the canonical provider names available for construction."""

    return tuple(sorted(_PROVIDER_MAP))


def build_client(provider: str, model: str, **kwargs: Any) -> LLMClient:
    """Instantiate a provider specific :class:`LLMClient` implementation."""

    try:
        client_cls = _PROVIDER_MAP[provider.lower()]
    except KeyError as exc:
        raise ConfigurationError(
            f"Unknown LLM provider '{provider}'. Available providers: {', '.join(get_available_providers())}",
            provider=provider,
        ) from exc
    return client_cls(model, **kwargs)


def register_provider(name: str, client_cls: Type[LLMClient]) -> None:
    """Register a custom provider at runtime."""

    if not issubclass(client_cls, LLMClient):
        raise TypeError("client_cls must be a subclass of LLMClient")
    _PROVIDER_MAP[name.lower()] = client_cls
