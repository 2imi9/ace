"""Anthropic Claude provider client."""

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


class ClaudeClient(LLMClient):
    """Client for Anthropic's Claude Messages API."""

    provider_name = "claude"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_version: str | None = None,
        api_base: str | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        super().__init__(model)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ConfigurationError(
                "Anthropic API key is required. Set ANTHROPIC_API_KEY in the environment.",
                provider=self.provider_name,
            )
        self.api_version = api_version or os.getenv("ANTHROPIC_API_VERSION") or "2023-06-01"
        self.api_base = (api_base or os.getenv("ANTHROPIC_API_URL") or "https://api.anthropic.com").rstrip(
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
        system_messages = [msg["content"] for msg in normalised if msg.get("role") == "system"]
        conversation = [
            {
                "role": msg["role"],
                "content": [{"type": "text", "text": msg["content"]}],
            }
            for msg in normalised
            if msg.get("role") != "system"
        ]
        payload: dict[str, object] = {
            "model": self.model,
            "messages": conversation,
            "max_tokens": 1024,
        }
        if system_messages:
            payload["system"] = "\n\n".join(system_messages)
        if extra:
            payload.update(dict(extra))
        response = retry_with_backoff(
            lambda: self._post_json("/v1/messages", payload),
            retry_config=self.retry_config,
        )
        content = response.get("content")
        if not isinstance(content, list) or not content:
            raise ProviderError(
                "Anthropic response did not contain content.",
                provider=self.provider_name,
                payload=response,
            )
        first_item = content[0]
        if not isinstance(first_item, dict):
            raise ProviderError(
                "Anthropic response content was not structured as expected.",
                provider=self.provider_name,
                payload=response,
            )
        text = first_item.get("text")
        if not isinstance(text, str):
            raise ProviderError(
                "Anthropic response did not include text content.",
                provider=self.provider_name,
                payload=response,
            )
        return text

    def stream(
        self,
        messages: Sequence[ChatMessage | Mapping[str, str]],
        *,
        extra: MutableMapping[str, object] | None = None,
    ) -> Iterator[str]:
        yield self.chat(messages, extra=extra)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _post_json(self, path: str, payload: Mapping[str, object]) -> Mapping[str, object]:
        data = json.dumps(payload).encode("utf-8")
        url = f"{self.api_base}{path}"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "content-type": "application/json",
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
        message = None
        if isinstance(payload, dict):
            message = payload.get("error") or payload.get("message")
        if status == 401:
            raise AuthenticationError(
                message or "Authentication failed for Anthropic.",
                provider=self.provider_name,
                status_code=status,
                payload=payload,
            ) from exc
        if status in {429, 500, 502, 503}:
            raise RateLimitError(
                message or "Anthropic service is temporarily unavailable.",
                provider=self.provider_name,
                status_code=status,
                payload=payload,
            ) from exc
        raise ProviderError(
            message or f"Anthropic request failed with status {status}.",
            provider=self.provider_name,
            status_code=status,
            payload=payload,
        ) from exc
