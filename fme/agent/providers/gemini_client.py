"""Google Gemini provider client."""

from __future__ import annotations

import json
import os
from typing import Iterator, Mapping, MutableMapping, Sequence
from urllib import error, parse, request

from .base import ChatMessage, LLMClient
from .errors import (
    AuthenticationError,
    ConfigurationError,
    ProviderError,
    RateLimitError,
    RetryableProviderError,
)
from .utils import RetryConfig, retry_with_backoff


class GeminiClient(LLMClient):
    """Client for Google's Gemini Generative Language API."""

    provider_name = "gemini"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        super().__init__(model)
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ConfigurationError(
                "Gemini API key is required. Set GEMINI_API_KEY or GOOGLE_API_KEY.",
                provider=self.provider_name,
            )
        self.api_key = key
        self.api_base = (api_base or os.getenv("GEMINI_API_URL") or "https://generativelanguage.googleapis.com").rstrip(
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
        contents = [
            {
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [{"text": msg["content"]}],
            }
            for msg in normalised
            if msg.get("role") != "system"
        ]
        payload: dict[str, object] = {"contents": contents}
        if system_messages:
            payload["system_instruction"] = {
                "parts": [{"text": "\n\n".join(system_messages)}]
            }
        if extra:
            payload.update(dict(extra))
        response = retry_with_backoff(
            lambda: self._post_json(payload),
            retry_config=self.retry_config,
        )
        candidates = response.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ProviderError(
                "Gemini response did not contain candidates.",
                provider=self.provider_name,
                payload=response,
            )
        first = candidates[0]
        if not isinstance(first, dict):
            raise ProviderError(
                "Gemini candidate had unexpected structure.",
                provider=self.provider_name,
                payload=response,
            )
        content = first.get("content")
        if not isinstance(content, dict):
            raise ProviderError(
                "Gemini response missing content.",
                provider=self.provider_name,
                payload=response,
            )
        parts = content.get("parts")
        if not isinstance(parts, list) or not parts:
            raise ProviderError(
                "Gemini content missing parts array.",
                provider=self.provider_name,
                payload=response,
            )
        text = parts[0].get("text")
        if not isinstance(text, str):
            raise ProviderError(
                "Gemini part missing text field.",
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
    def _post_json(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        data = json.dumps(payload).encode("utf-8")
        endpoint = f"{self.api_base}/v1beta/models/{self.model}:generateContent"
        query = parse.urlencode({"key": self.api_key})
        url = f"{endpoint}?{query}"
        headers = {"Content-Type": "application/json"}
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
            error_info = payload.get("error")
            if isinstance(error_info, dict):
                message = error_info.get("message")
        if status in {401, 403}:
            raise AuthenticationError(
                message or "Authentication failed for Gemini.",
                provider=self.provider_name,
                status_code=status,
                payload=payload,
            ) from exc
        if status in {429, 500, 502, 503}:
            raise RateLimitError(
                message or "Gemini service is temporarily unavailable.",
                provider=self.provider_name,
                status_code=status,
                payload=payload,
            ) from exc
        raise ProviderError(
            message or f"Gemini request failed with status {status}.",
            provider=self.provider_name,
            status_code=status,
            payload=payload,
        ) from exc
