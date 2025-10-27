"""Custom exceptions raised by provider clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(slots=True)
class ProviderError(Exception):
    """Base error type for provider failures."""

    message: str
    provider: str | None = None
    status_code: int | None = None
    payload: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:  # pragma: no cover - trivial
        super().__init__(self.message)


class ConfigurationError(ProviderError):
    """Raised when a provider has been misconfigured."""


class AuthenticationError(ProviderError):
    """Raised when a provider rejects supplied credentials."""


class RetryableProviderError(ProviderError):
    """Raised for errors that should be retried with backoff."""


class RateLimitError(RetryableProviderError):
    """Raised when a provider indicates a throttling or quota issue."""


class StreamNotSupportedError(ProviderError):
    """Raised when streaming is requested but unsupported by the provider."""

    def __init__(self, provider: str | None = None) -> None:
        super().__init__("Streaming is not supported by this provider.", provider=provider)
