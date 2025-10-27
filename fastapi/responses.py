"""Simplified response objects for the FastAPI stub."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class JSONResponse:
    content: Any
    status_code: int = 200


@dataclass
class PlainTextResponse:
    content: str
    status_code: int = 200


__all__ = ["JSONResponse", "PlainTextResponse"]
