"""Image encoding helpers used by the PhySeek wire format."""

from __future__ import annotations

import base64
import io
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image


def encode_frame(frame: np.ndarray, fmt: str = "jpeg", quality: int = 95) -> bytes:
    """Encode one RGB uint8 frame to ``raw``, ``png``, or ``jpeg`` bytes."""

    arr = np.asarray(frame, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"frame must have shape [H, W, 3], got {arr.shape}")
    if fmt == "raw":
        return arr.tobytes()
    if fmt not in {"png", "jpeg"}:
        raise ValueError(f"unknown image format: {fmt!r}")

    image = Image.fromarray(arr, mode="RGB")
    buffer = io.BytesIO()
    if fmt == "png":
        image.save(buffer, format="PNG", compress_level=1)
    else:
        image.save(buffer, format="JPEG", quality=int(quality))
    return buffer.getvalue()


def decode_frame(data: bytes, fmt: str = "jpeg", shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    """Decode one frame from ``raw``, ``png``, or ``jpeg`` bytes."""

    if fmt == "raw":
        if shape is None:
            raise ValueError("shape is required for raw frame decoding")
        return np.frombuffer(data, dtype=np.uint8).reshape(shape)
    if fmt not in {"png", "jpeg"}:
        raise ValueError(f"unknown image format: {fmt!r}")
    with Image.open(io.BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def encode_view(arr: np.ndarray, fmt: str = "jpeg", quality: int = 95) -> dict:
    """Encode a view ndarray with shape ``[H,W,3]`` or ``[T,H,W,3]``."""

    view = np.asarray(arr, dtype=np.uint8)
    if view.ndim == 3:
        return {"shape": view.shape, "frames": [encode_frame(view, fmt, quality)]}
    if view.ndim == 4:
        return {
            "shape": view.shape,
            "frames": [encode_frame(view[i], fmt, quality) for i in range(view.shape[0])],
        }
    raise ValueError(f"image view must be 3D or 4D, got shape {view.shape}")


def decode_view(data: dict, fmt: str = "jpeg") -> np.ndarray:
    """Decode a wire-format view dict back to an ndarray."""

    shape = tuple(data["shape"])
    frames_bytes: Sequence[bytes] = data["frames"]
    if fmt == "raw":
        frame_shape = shape if len(shape) == 3 else shape[1:]
        frames = [decode_frame(frame, fmt, shape=frame_shape) for frame in frames_bytes]
    else:
        frames = [decode_frame(frame, fmt) for frame in frames_bytes]
    if len(shape) == 3:
        return frames[0]
    return np.stack(frames, axis=0)


def rgb_to_jpeg_b64(frame: np.ndarray, quality: int = 85) -> str:
    return base64.b64encode(encode_frame(frame, "jpeg", quality)).decode("ascii")


def rgb_to_jpeg_data_uri(frame: np.ndarray, quality: int = 85) -> str:
    return f"data:image/jpeg;base64,{rgb_to_jpeg_b64(frame, quality)}"


__all__ = [
    "decode_frame",
    "decode_view",
    "encode_frame",
    "encode_view",
    "rgb_to_jpeg_b64",
    "rgb_to_jpeg_data_uri",
]
