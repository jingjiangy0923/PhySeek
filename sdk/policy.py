"""Policy interfaces implemented by external model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from .message import Action, Observation, WireConfig


@runtime_checkable
class Policy(Protocol):
    """Structural policy API consumed by :class:`sdk.PolicyClient`."""

    def reset(self) -> None:
        ...

    def update_obs(self, observation: Observation) -> None:
        ...

    def get_action(self) -> Action:
        ...

    def get_wire_config(self) -> WireConfig:
        ...

    def set_instruction(self, instruction: str) -> None:
        ...

    def get_policy_spec(self) -> dict:
        ...


class BasePolicy(ABC):
    """Base class for stateful policy adapters."""

    @abstractmethod
    def reset(self) -> None:
        """Clear policy state before a new SEP run or explicit reset."""

    @abstractmethod
    def update_obs(self, observation: Observation) -> None:
        """Cache or preprocess the latest observation."""

    @abstractmethod
    def get_action(self) -> Action:
        """Return the next action chunk for the latest observation."""

    def get_wire_config(self) -> WireConfig:
        return WireConfig()

    def set_instruction(self, instruction: str) -> None:
        return None

    def get_policy_spec(self) -> dict:
        return {}


__all__ = ["Action", "BasePolicy", "Observation", "Policy", "WireConfig"]
