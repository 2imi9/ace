"""Utility helpers shared between provider clients."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Iterable, TypeVar

from .errors import RetryableProviderError

T = TypeVar("T")


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for retry_with_backoff."""

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    multiplier: float = 2.0
    jitter: float = 0.25

    def __post_init__(self) -> None:
        if self.max_attempts < 1:  # pragma: no cover - defensive programming
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay <= 0:
            raise ValueError("base_delay must be positive")
        if self.max_delay < self.base_delay:
            raise ValueError("max_delay must be >= base_delay")
        if self.multiplier <= 1.0:
            raise ValueError("multiplier must be greater than 1.0")
        if self.jitter < 0:
            raise ValueError("jitter cannot be negative")


def retry_with_backoff(
    operation: Callable[[], T],
    *,
    retry_config: RetryConfig | None = None,
    retry_on: Iterable[type[BaseException]] | None = (RetryableProviderError,),
) -> T:
    """Execute ``operation`` with exponential backoff retry logic."""

    config = retry_config or RetryConfig()
    retry_types = tuple(retry_on or (RetryableProviderError,))
    attempt = 0
    delay = config.base_delay
    while True:
        try:
            return operation()
        except retry_types as exc:
            attempt += 1
            if attempt >= config.max_attempts:
                raise
            jitter = random.uniform(0, config.jitter)
            time.sleep(delay + jitter)
            delay = min(delay * config.multiplier, config.max_delay)
        except Exception:
            raise
