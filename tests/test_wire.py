from __future__ import annotations

import numpy as np

from sdk import Action, MsgType, Observation, WireConfig, WsMessage


def test_observation_roundtrip_raw():
    image = np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 3)
    obs = Observation(
        images={"front": image},
        state={"eef": [1.0, 2.0]},
        instruction="pick",
        image_format="raw",
        timestamp=12.5,
        obs_step=3,
    )

    decoded = Observation.from_bytes(obs.to_bytes())

    assert decoded.instruction == "pick"
    assert decoded.timestamp == 12.5
    assert decoded.kwargs["obs_step"] == 3
    np.testing.assert_array_equal(decoded.images["front"], image)


def test_action_roundtrip_with_timestamp():
    action = np.ones((8, 14), dtype=np.float32)
    decoded = Action.from_bytes(Action(action, timestamp=7.0, hz=10).to_bytes())

    np.testing.assert_array_equal(decoded.action, action)
    assert decoded.timestamp == 7.0
    assert decoded.kwargs["hz"] == 10


def test_wire_config_filters_unknown_fields():
    config = WireConfig.from_dict({"image_format": "png", "unknown": "ignored"})

    assert config.image_format == "png"
    assert "unknown" not in config.to_dict()


def test_ws_message_ack_roundtrip():
    msg = WsMessage.ack(MsgType.POLICY_SESSION_START, session_id="sid")
    decoded = WsMessage.from_json(msg.to_json())

    assert decoded.type == MsgType.ACK
    assert decoded.session_id == "sid"
    assert decoded.payload["ack_for"] == MsgType.POLICY_SESSION_START
