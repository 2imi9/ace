"""Utilities for tracking conversational and simulation state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SessionEvent:
    """Lightweight record describing an action taken during the session."""

    command: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionState:
    """In-memory state for interactive ACE agent sessions."""

    events: List[SessionEvent] = field(default_factory=list)
    conversation: List[Dict[str, str]] = field(default_factory=list)
    last_simulation_output: Any = None
    last_analysis: Optional[Dict[str, Any]] = None

    def record_event(self, command: str, **payload: Any) -> None:
        """Record that a command was executed with optional metadata."""

        self.events.append(SessionEvent(command=command, payload=payload))

    def append_conversation(self, role: str, content: str) -> None:
        """Append a conversational turn to the history."""

        self.conversation.append({"role": role, "content": content})

    def reset(self) -> None:
        """Reset session state while keeping configuration intact."""

        self.events.clear()
        self.conversation.clear()
        self.last_simulation_output = None
        self.last_analysis = None
