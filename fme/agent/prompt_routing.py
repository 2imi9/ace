"""Command routing infrastructure for ACE agent prompts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Dict


class PromptRouter:
    """Simple command router that dispatches prompts to handlers."""

    def __init__(
        self,
        handlers: Mapping[str, Callable[..., Any]] | None = None,
    ) -> None:
        self._handlers: Dict[str, Callable[..., Any]] = {}
        if handlers is not None:
            for command, handler in handlers.items():
                self.register(command, handler)

    def register(self, command: str, handler: Callable[..., Any]) -> None:
        """Register a handler for the provided command."""

        key = command.strip().lower()
        if not key:
            raise ValueError("Command name must be a non-empty string.")
        self._handlers[key] = handler

    def unregister(self, command: str) -> None:
        """Remove a registered command handler."""

        self._handlers.pop(command.strip().lower(), None)

    def available_commands(self) -> list[str]:
        """Return a sorted list of the registered commands."""

        return sorted(self._handlers.keys())

    def route(self, command: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke the handler associated with ``command``."""

        key = command.strip().lower()
        try:
            handler = self._handlers[key]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise KeyError(
                f"Unknown command '{command}'. Known commands: {self.available_commands()}"
            ) from exc
        return handler(*args, **kwargs)
