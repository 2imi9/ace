"""OpenAI provider client implementation."""

from __future__ import annotations

import json
import os
from typing import Iterator, Mapping, MutableMapping, Sequence
from urllib import error, request

from .base import ChatMessage, LLMClient
from .errors import (
    AuthenticationError,
    ConfigurationError,
    ProviderError,
    RateLimitError,
    RetryableProviderError,
)
from .utils import RetryConfig, retry_with_backoff


class OpenAIClient(LLMClient):
    """Thin wrapper around the OpenAI chat completions endpoint."""

    provider_name = "openai"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        super().__init__(model)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ConfigurationError(
                "OpenAI API key is required. Set OPENAI_API_KEY in the environment.",
                provider=self.provider_name,
            )
        self.api_base = (api_base or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1").rstrip(
            "/"
        )
        self.retry_config = retry_config or RetryConfig()

    def chat(
        self,
        messages: Sequence[ChatMessage | Mapping[str, str]],
        *,
        extra: MutableMapping[str, object] | None = None,
    ) -> str:
        normalised = self.normalise_messages(messages)
        payload: dict[str, object] = {"model": self.model, "messages": list(normalised)}
        if extra:
            payload.update(dict(extra))
        response = retry_with_backoff(
            lambda: self._post_json("/chat/completions", payload),
            retry_config=self.retry_config,
        )
        choices = response.get("choices", [])
        if not choices:
            raise ProviderError(
                "OpenAI response did not contain any choices.",
                provider=self.provider_name,
                payload=response,
            )
        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise ProviderError(
                "OpenAI response did not contain textual content.",
                provider=self.provider_name,
                payload=response,
            )
        return content

    def stream(
        self,
        messages: Sequence[ChatMessage | Mapping[str, str]],
        *,
        extra: MutableMapping[str, object] | None = None,
    ) -> Iterator[str]:
        # The lightweight implementation performs a non-streaming call and
        # yields the resulting message as a single chunk. This keeps the
        # interface uniform for orchestrators that expect an iterator.
        yield self.chat(messages, extra=extra)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _post_json(self, path: str, payload: Mapping[str, object]) -> Mapping[str, object]:
        data = json.dumps(payload).encode("utf-8")
        url = f"{self.api_base}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        req = request.Request(url, data=data, headers=headers, method="POST")
        try:
            with request.urlopen(req) as response:
                response_data = response.read().decode("utf-8")
                return json.loads(response_data) if response_data else {}
        except error.HTTPError as exc:
            self._handle_http_error(exc)
            raise ProviderError(str(exc), provider=self.provider_name) from exc
        except error.URLError as exc:
            raise RetryableProviderError(str(exc.reason), provider=self.provider_name) from exc

    def _handle_http_error(self, exc: error.HTTPError) -> None:
        status = exc.code
        body = exc.read().decode("utf-8") if hasattr(exc, "read") else ""
        try:
            payload = json.loads(body) if body else None
        except json.JSONDecodeError:
            payload = None
        message = payload.get("error", {}).get("message") if isinstance(payload, dict) else body
        if status == 401:
            raise AuthenticationError(
                message or "Authentication failed for OpenAI.",
                provider=self.provider_name,
                status_code=status,
                payload=payload,
            ) from exc
        if status in {429, 500, 502, 503}:
            raise RateLimitError(
                message or "OpenAI service is temporarily unavailable.",
                provider=self.provider_name,
                status_code=status,
                payload=payload,
            ) from exc
        raise ProviderError(
            message or f"OpenAI request failed with status {status}.",
            provider=self.provider_name,
            status_code=status,
            payload=payload,
        ) from exc
