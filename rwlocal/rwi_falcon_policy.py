"""Adapt the existing RWI Falcon policy to the PhySeek SDK interface.

This file is the important example:
an external user owns a local model, wraps it as ``BasePolicy``, and starts it
with ``physeek run rwlocal.rwi_falcon_policy:RwiFalconPolicy ...``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from sdk import Action, BasePolicy, Observation, WireConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
RWI_ROOT = REPO_ROOT / "RealWorldInference"
if str(RWI_ROOT) not in sys.path:
    sys.path.insert(0, str(RWI_ROOT))

from policy.policies import load_policy  # noqa: E402


_ROBOT_CONFIGS = {
    "xingchen": "../ss-inference/configs/embodied/action_stream_xingchen_eef.py",
    "moz": "../ss-inference/configs/embodied/action_stream_qianxun.py",
}


class RwiFalconPolicy(BasePolicy):
    """SDK adapter around the existing RWI Falcon policy."""

    def __init__(
        self,
        robot_type: str,
        checkpoint_path: str,
        device: str = "cuda",
        return_video: bool = True,
    ):
        config_path = _ROBOT_CONFIGS.get(robot_type)
        if config_path is None:
            raise ValueError(f"unsupported robot_type={robot_type!r}; expected one of {sorted(_ROBOT_CONFIGS)}")

        self._policy = load_policy(
            "falcon",
            {
                "type": "falcon",
                "config_path": config_path,
                "checkpoint_path": checkpoint_path,
                "device": device,
                "return_video": return_video,
            },
        )

    def reset(self) -> None:
        self._policy.reset()

    def set_instruction(self, instruction: str) -> None:
        self._policy.set_instruction(instruction)

    def update_obs(self, observation: Observation) -> None:
        self._policy.update_obs(observation)

    def get_action(self) -> Action:
        action = self._policy.get_action()
        return Action(action=action.action, timestamp=action.timestamp, **(action.kwargs or {}))

    def get_policy_spec(self) -> dict:
        return dict(self._policy.get_policy_spec() or {})

    def get_wire_config(self) -> WireConfig:
        wire = self._policy.get_wire_config()
        return WireConfig.from_dict(wire.to_dict())
