"""Mock policy useful for SEP integration smoke tests."""

from __future__ import annotations

import time

import numpy as np

from sdk import Action, BasePolicy, Observation, WireConfig


class MockPolicy(BasePolicy):
    def __init__(
        self,
        action_dim: int = 14,
        chunk_len: int = 8,
        hz: int = 10,
        action_space: str = "eef",
        return_video: bool = False,
    ):
        self.action_dim = int(action_dim)
        self.chunk_len = int(chunk_len)
        self.hz = int(hz)
        self.action_space = action_space
        self.return_video = bool(return_video)
        self.step = 0
        self.instruction = ""
        self.obs: Observation | None = None

    def reset(self) -> None:
        self.step = 0
        self.obs = None

    def set_instruction(self, instruction: str) -> None:
        self.instruction = instruction

    def update_obs(self, observation: Observation) -> None:
        self.obs = observation

    def get_action(self) -> Action:
        self.step += 1
        t = np.linspace(0, 2 * np.pi, self.chunk_len, dtype=np.float32)
        action = np.stack(
            [np.sin(t + self.step * 0.2 + i * 0.15) for i in range(self.action_dim)],
            axis=1,
        ).astype(np.float32)
        kwargs = {"frame_count": self.chunk_len, "hz": self.hz}
        if self.return_video:
            kwargs["generated_video"] = self._prediction_video()
        return Action(action=action, timestamp=time.time(), **kwargs)

    def get_policy_spec(self) -> dict:
        return {
            "output_action_space": self.action_space,
            "output_action_dim": self.action_dim,
            "action_space": self.action_space,
            "action_dim": self.action_dim,
            "chunk_len": self.chunk_len,
            "hz": self.hz,
            "camera_views": ["front"],
        }

    def get_wire_config(self) -> WireConfig:
        return WireConfig(video_action_rate=1, video_length=1, image_format="jpeg", jpeg_quality=80)

    def _prediction_video(self, frames: int = 12, height: int = 96, width: int = 144) -> np.ndarray:
        video = np.zeros((frames, height, width, 3), dtype=np.uint8)
        for i in range(frames):
            phase = (self.step * 31 + i * 11) % 255
            video[i, :, :, 0] = phase
            video[i, :, :, 1] = (phase + 80) % 255
            video[i, :, :, 2] = (phase + 160) % 255
            bar = max(1, int(width * (i + 1) / frames))
            video[i, height - 10 : height, :bar, :] = 255
        return video
