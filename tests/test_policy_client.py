from __future__ import annotations

import asyncio

import numpy as np
import pytest

from sdk import Action, BasePolicy, MsgType, Observation, PolicyClient, WireConfig, WsMessage
from sdk.client import build_policy_ws_url


class _FakePolicy(BasePolicy):
    def __init__(self):
        self.instruction = ""
        self.reset_count = 0
        self.obs = None

    def reset(self) -> None:
        self.reset_count += 1

    def set_instruction(self, instruction: str) -> None:
        self.instruction = instruction

    def update_obs(self, observation: Observation) -> None:
        self.obs = observation

    def get_action(self) -> Action:
        return Action(np.zeros((2, 3), dtype=np.float32))

    def get_wire_config(self) -> WireConfig:
        return WireConfig(image_format="raw")

    def get_policy_spec(self) -> dict:
        return {"output_action_space": "eef", "output_action_dim": 3}


class _FakeWs:
    def __init__(self, frames):
        self.frames = list(frames)
        self.sent = []

    async def recv(self):
        return self.frames.pop(0)

    async def send(self, frame):
        self.sent.append(frame)


def test_hello_contains_policy_spec_and_wire_config():
    client = PolicyClient(
        policy=_FakePolicy(),
        policy_id="mock",
        policy_type="test",
        sep_url="wss://sep",
        policy_spec={"robot_preset": "demo"},
    )

    hello = client._hello()

    assert hello.type == MsgType.POLICY_HELLO
    assert hello.payload["policy_id"] == "mock"
    assert hello.payload["policy_spec"]["output_action_dim"] == 3
    assert hello.payload["policy_spec"]["robot_preset"] == "demo"
    assert hello.payload["wire_config"]["image_format"] == "raw"


def test_handle_infer_sends_action_and_infer_end():
    async def _run():
        policy = _FakePolicy()
        client = PolicyClient(policy=policy, policy_id="mock", policy_type="test", sep_url="wss://sep")
        obs = Observation(
            images={"front": np.zeros((2, 2, 3), dtype=np.uint8)},
            image_format="raw",
            timestamp=1.0,
        )
        ws = _FakeWs([obs.to_bytes()])

        await client._handle_infer(ws, session_id="sid", instruction="move")

        headers = [WsMessage.from_json(frame) for frame in ws.sent if isinstance(frame, str)]
        binary_frames = [frame for frame in ws.sent if isinstance(frame, bytes)]
        assert policy.instruction == ""
        assert policy.obs.instruction == "move"
        assert headers[0].type == MsgType.POLICY_ACTION
        assert headers[-1].type == MsgType.POLICY_INFER_END
        assert len(binary_frames) == 1
        decoded = Action.from_bytes(binary_frames[0])
        assert decoded.action.shape == (2, 3)

    asyncio.run(_run())


def test_policy_ws_url_requires_wss():
    assert build_policy_ws_url("wss://sep", "mock") == "wss://sep/api/execution/ws/policy/mock"
    with pytest.raises(ValueError, match="wss://"):
        build_policy_ws_url("ws://sep", "mock")
