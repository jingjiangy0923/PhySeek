"""Wire-format data objects exchanged between slaves, SEP, and policies."""

from __future__ import annotations

import io
import pickle
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional

import numpy as np

from .image import decode_view, encode_view


class _CrossNumpyUnpickler(pickle.Unpickler):
    """Unpickler compatible with numpy 1.x and 2.x internal module paths."""

    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            if module.startswith("numpy._core"):
                module = "numpy.core" + module[len("numpy._core") :]
            elif module.startswith("numpy.core"):
                module = "numpy._core" + module[len("numpy.core") :]
            else:
                raise
            return super().find_class(module, name)


def _loads(data: bytes) -> Any:
    """``pickle.loads`` with numpy cross-version compatibility."""

    return _CrossNumpyUnpickler(io.BytesIO(data)).load()


@dataclass
class WireConfig:
    """How the robot executor should prepare observations for this policy."""

    video_action_rate: int = 1
    video_length: int = 1
    image_format: str = "jpeg"
    jpeg_quality: int = 95

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "WireConfig":
        field_names = {field.name for field in fields(WireConfig)}
        return WireConfig(**{key: data[key] for key in data if key in field_names})


class Observation:
    """Observation sent from a robot slave to a policy."""

    def __init__(
        self,
        images: Dict[str, np.ndarray],
        state=None,
        instruction: Optional[str] = None,
        t5_embeddings: Optional[np.ndarray] = None,
        action_constraint: Optional[np.ndarray] = None,
        joint_state: Optional[np.ndarray] = None,
        image_format: str = "jpeg",
        jpeg_quality: int = 95,
        timestamp: Optional[float] = None,
        **kwargs,
    ):
        self.images = images
        self.state = state
        self.instruction = instruction
        self.t5_embeddings = t5_embeddings
        self.action_constraint = action_constraint
        self.joint_state = joint_state
        self.image_format = image_format
        self.jpeg_quality = jpeg_quality
        self.timestamp = timestamp
        self.kwargs = kwargs

    def to_bytes(self) -> bytes:
        encoded_images = {
            name: encode_view(arr, self.image_format, self.jpeg_quality)
            for name, arr in self.images.items()
        }
        payload = {
            "image_format": self.image_format,
            "images": encoded_images,
            "state": self.state,
            "instruction": self.instruction,
            "t5_embeddings": self.t5_embeddings,
            "action_constraint": self.action_constraint,
            "joint_state": self.joint_state,
            "timestamp": self.timestamp,
            **(self.kwargs or {}),
        }
        return pickle.dumps(payload)

    @staticmethod
    def from_bytes(data: bytes) -> "Observation":
        payload = _loads(data)
        fmt = payload.get("image_format", "png")
        images = {name: decode_view(view, fmt) for name, view in payload["images"].items()}
        reserved = {
            "action_constraint",
            "image_format",
            "images",
            "instruction",
            "joint_state",
            "state",
            "t5_embeddings",
            "timestamp",
        }
        kwargs = {key: value for key, value in payload.items() if key not in reserved}
        return Observation(
            images=images,
            state=payload.get("state"),
            instruction=payload.get("instruction"),
            t5_embeddings=payload.get("t5_embeddings"),
            action_constraint=payload.get("action_constraint"),
            joint_state=payload.get("joint_state"),
            image_format=fmt,
            timestamp=payload.get("timestamp"),
            **kwargs,
        )


class Action:
    """Action chunk returned by a policy."""

    def __init__(self, action: np.ndarray, timestamp: Optional[float] = None, **kwargs):
        self.action = action
        self.timestamp = timestamp
        self.kwargs = kwargs

    def to_bytes(self) -> bytes:
        return pickle.dumps({"action": self.action, "timestamp": self.timestamp, **(self.kwargs or {})})

    @staticmethod
    def from_bytes(data: bytes) -> "Action":
        payload = _loads(data)
        kwargs = {key: value for key, value in payload.items() if key not in ("action", "timestamp")}
        return Action(action=np.asarray(payload["action"]), timestamp=payload.get("timestamp"), **kwargs)


__all__ = ["Action", "Observation", "WireConfig", "_loads"]
