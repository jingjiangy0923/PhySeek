"""Shared RWI / SEP realtime WebSocket protocol."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class MsgType(str, Enum):
    """WebSocket message types shared by SEP, robot slaves, and policies."""

    SLAVE_HELLO = "slave.hello"
    SLAVE_OBS = "slave.obs"
    SLAVE_STATUS = "slave.status"
    SLAVE_HEARTBEAT = "slave.heartbeat"
    SLAVE_REVIEW_OBS0 = "slave.review.obs0"
    SLAVE_REVIEW_FRAME = "slave.review.frame"
    SLAVE_REVIEW_READY = "slave.review.ready"
    CMD_START = "cmd.start"
    CMD_PAUSE = "cmd.pause"
    CMD_RESUME = "cmd.resume"
    CMD_STOP = "cmd.stop"
    CMD_ACTION = "cmd.action"
    CMD_REVIEW_DISPATCH = "cmd.review.dispatch"
    CMD_REVIEW_APPROVE = "cmd.review.approve"
    CMD_REVIEW_REJECT = "cmd.review.reject"
    CMD_EXPORT_UPLOAD = "cmd.export.upload"
    EXPORT_PROGRESS = "export.progress"
    EXPORT_DONE = "export.done"
    EXPORT_FAILED = "export.failed"
    SLAVE_EXPORT_PENDING = "slave.export_pending"
    STREAM_FRAME = "stream.frame"
    STREAM_ACTION = "stream.action"
    STREAM_GEN_FRAME = "stream.gen_frame"
    STREAM_REVIEW_OBS0 = "stream.review.obs0"
    STREAM_REVIEW_FRAME = "stream.review.frame"
    STREAM_REVIEW_READY = "stream.review.ready"
    UI_START = "ui.start"
    UI_PAUSE = "ui.pause"
    UI_RESUME = "ui.resume"
    UI_STOP = "ui.stop"
    UI_UPDATE_INSTRUCTION = "ui.update_instruction"
    UI_REVIEW_APPROVE = "ui.review.approve"
    UI_REVIEW_REJECT = "ui.review.reject"
    POLICY_HELLO = "policy.hello"
    POLICY_SESSION_START = "session.start"
    POLICY_INFER = "infer"
    POLICY_ACTION = "action"
    POLICY_PREDICTION_VIDEO = "prediction.video"
    POLICY_INFER_END = "infer_end"
    POLICY_SET_INSTRUCTION = "set_instruction"
    POLICY_RESET = "reset"
    POLICY_PING = "ping"
    POLICY_PONG = "pong"
    ACK = "ack"
    ERROR = "error"


def derive_policy_ws_url(http_url: str) -> str:
    """Derive the legacy policy ``/ws/infer`` URL using secure websockets."""

    base = (http_url or "").rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base[8:]
    elif not base.startswith("wss://"):
        raise ValueError("PhySeek only supports secure WebSocket URLs starting with wss://")
    if base.endswith("/ws/infer"):
        return base
    return f"{base}/ws/infer"


@dataclass
class WsMessage:
    """JSON text envelope for lightweight WebSocket metadata.

    Large payloads use split frames: send this JSON header first, then the
    adjacent binary frame.
    """

    type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    ts: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "WsMessage":
        data = json.loads(raw)
        return cls(
            type=data["type"],
            payload=data.get("payload", {}),
            session_id=data.get("session_id"),
            ts=data.get("ts", time.time()),
        )

    @classmethod
    def ack(
        cls,
        original_type: str,
        session_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> "WsMessage":
        payload: Dict[str, Any] = {"ack_for": original_type}
        if extra:
            for key, value in extra.items():
                if key != "ack_for":
                    payload[key] = value
        return cls(type=MsgType.ACK, payload=payload, session_id=session_id)

    @classmethod
    def error(cls, message: str, session_id: Optional[str] = None) -> "WsMessage":
        return cls(type=MsgType.ERROR, payload={"message": message}, session_id=session_id)


__all__ = ["MsgType", "WsMessage", "derive_policy_ws_url"]
