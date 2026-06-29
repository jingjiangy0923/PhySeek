"""Minimal external policy adapter for PhySeek."""

from __future__ import annotations

import numpy as np

from sdk import Action, BasePolicy, Observation, WireConfig


class MinimalPolicy(BasePolicy):
    def __init__(self, action_dim: int = 14, chunk_len: int = 8, action_space: str = "eef"):
        self.action_dim = int(action_dim)
        self.chunk_len = int(chunk_len)
        self.action_space = action_space
        self._instruction = ""
        self._obs: Observation | None = None

    def reset(self) -> None:
        self._obs = None

    def set_instruction(self, instruction: str) -> None:
        self._instruction = instruction

    def update_obs(self, observation: Observation) -> None:
        self._obs = observation

    def get_action(self) -> Action:
        if self._obs is None:
            raise RuntimeError("No observation received")
        action = np.zeros((self.chunk_len, self.action_dim), dtype=np.float32)
        return Action(action=action)

    def get_policy_spec(self) -> dict:
        return {
            "output_action_space": self.action_space,
            "output_action_dim": self.action_dim,
            "chunk_len": self.chunk_len,
        }

    def get_wire_config(self) -> WireConfig:
        return WireConfig(video_action_rate=1, video_length=1, image_format="jpeg", jpeg_quality=90)
