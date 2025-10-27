"""Base classes and shared types for LLM provider clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, MutableMapping, Sequence

from .errors import ProviderError, StreamNotSupportedError


@dataclass(frozen=True)
class ChatMessage:
    """Represents a single chat message exchanged with an LLM."""

    role: str
    content: str

    def to_mapping(self) -> Mapping[str, str]:
        """Return a serialisable mapping representation."""

        return {"role": self.role, "content": self.content}


class LLMClient(ABC):
    """Abstract interface implemented by concrete LLM providers."""

    provider_name: str

    def __init__(self, model: str) -> None:
        self.model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @abstractmethod
    def chat(
        self,
        messages: Sequence[ChatMessage | Mapping[str, str]],
        *,
        extra: MutableMapping[str, object] | None = None,
    ) -> str:
        """Return a full completion for ``messages``.

        Concrete implementations must consume an ordered sequence of chat
        messages and return the aggregated assistant response as a single
        string. ``extra`` can be used to pass provider specific overrides.
        """

    def stream(
        self,
        messages: Sequence[ChatMessage | Mapping[str, str]],
        *,
        extra: MutableMapping[str, object] | None = None,
    ) -> Iterator[str]:
        """Yield the completion for ``messages`` incrementally.

        Providers that do not implement streaming can raise
        :class:`StreamNotSupportedError` which will be propagated to the
        caller.
        """

        raise StreamNotSupportedError(self.provider_name)

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    @staticmethod
    def normalise_messages(
        messages: Sequence[ChatMessage | Mapping[str, str]]
    ) -> Sequence[Mapping[str, str]]:
        """Coerce heterogeneous message objects to a uniform mapping."""

        normalised: list[Mapping[str, str]] = []
        for message in messages:
            if isinstance(message, ChatMessage):
                normalised.append(message.to_mapping())
            else:
                normalised.append(dict(message))
        return normalised

    def _chat(
        self,
        messages: Sequence[ChatMessage | Mapping[str, str]],
        *,
        extra: MutableMapping[str, object] | None = None,
    ) -> str:
        """Internal helper that defers to :meth:`chat` with validation."""

        try:
            return self.chat(messages, extra=extra)
        except ProviderError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise ProviderError(str(exc), provider=self.provider_name) from exc

    def _stream(
        self,
        messages: Sequence[ChatMessage | Mapping[str, str]],
        *,
        extra: MutableMapping[str, object] | None = None,
    ) -> Iterable[str]:
        """Internal helper that wraps :meth:`stream` with validation."""

        try:
            return self.stream(messages, extra=extra)
        except ProviderError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise ProviderError(str(exc), provider=self.provider_name) from exc
