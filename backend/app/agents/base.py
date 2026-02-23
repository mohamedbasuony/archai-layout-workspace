from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Base interface for ArchAI pipeline agents."""

    name: str = "base-agent"

    @abstractmethod
    def run(self, payload: Any) -> Any:
        raise NotImplementedError
