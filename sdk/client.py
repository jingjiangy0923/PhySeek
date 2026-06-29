"""Reverse WebSocket client that registers a local policy with SEP."""

from __future__ import annotations

import asyncio
import logging
import pickle
import ssl
import time
from typing import Optional

import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed

from .image import encode_view
from .message import Action, Observation
from .policy import Policy
from .protocol import MsgType, WsMessage

logger = logging.getLogger(__name__)

WS_MAX_SIZE = 64 * 1024 * 1024
PING_TIMEOUT = 30.0
RECONNECT_DELAY = 5.0
PREDICTION_VIDEO_FORMAT = "jpeg_stack"
PREDICTION_VIDEO_JPEG_QUALITY = 90


def build_policy_ws_url(base_url: str, policy_id: str) -> str:
    """Build SEP reverse policy websocket URL from a SEP base URL."""

    base = base_url.rstrip("/")
    if not base.startswith("wss://"):
        raise ValueError("PhySeek only supports secure WebSocket SEP URLs starting with wss://")
    return f"{base}/api/execution/ws/policy/{policy_id}"


def _encode_prediction_video(video_arr: np.ndarray) -> tuple[str, bytes]:
    """Encode optional prediction video for SEP browser preview."""

    if PREDICTION_VIDEO_FORMAT == "jpeg_stack":
        try:
            encoded = encode_view(video_arr, "jpeg", PREDICTION_VIDEO_JPEG_QUALITY)
            return "jpeg_stack", pickle.dumps(encoded)
        except Exception as exc:
            logger.warning("prediction.video jpeg_stack encode failed; falling back to raw: %s", exc)
    return "raw", pickle.dumps(video_arr)


class PolicyClient:
    """Expose a local policy to SEP as a reverse-connected policy endpoint."""

    def __init__(
        self,
        *,
        policy: Policy,
        policy_id: str,
        policy_type: str,
        sep_url: str,
        policy_spec: Optional[dict] = None,
        insecure: bool = False,
        reconnect_delay: float = RECONNECT_DELAY,
        ping_timeout: float = PING_TIMEOUT,
        max_size: int = WS_MAX_SIZE,
    ):
        self.policy = policy
        self.policy_id = policy_id
        self.policy_type = policy_type
        self.sep_url = sep_url
        self.policy_spec = dict(policy_spec or {})
        self.insecure = insecure
        self.reconnect_delay = float(reconnect_delay)
        self.ping_timeout = float(ping_timeout)
        self.max_size = int(max_size)
        self._infer_lock = asyncio.Lock()

    def _hello(self) -> WsMessage:
        spec = dict(self.policy.get_policy_spec() or {})
        spec.update(self.policy_spec)
        return WsMessage(
            type=MsgType.POLICY_HELLO,
            payload={
                "policy_id": self.policy_id,
                "policy_type": self.policy_type,
                "policy_spec": spec,
                "wire_config": self.policy.get_wire_config().to_dict(),
            },
        )

    def _ssl_context(self, sep_ws_url: str) -> Optional[ssl.SSLContext]:
        if not sep_ws_url.startswith("wss://"):
            raise ValueError("PhySeek only supports secure WebSocket SEP URLs starting with wss://")
        ctx = ssl.create_default_context()
        if self.insecure:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            logger.warning("TLS verification disabled for policy websocket")
        return ctx

    def _run_infer_sync(self, obs: Observation) -> Action:
        self.policy.update_obs(obs)
        return self.policy.get_action()

    def _set_instruction(self, instruction: str) -> None:
        try:
            self.policy.set_instruction(instruction)
        except Exception as exc:
            logger.warning("policy.set_instruction failed: %s", exc)

    async def _send_infer_end(self, ws, session_id: Optional[str]) -> None:
        await ws.send(WsMessage(type=MsgType.POLICY_INFER_END, session_id=session_id).to_json())

    async def _handle_infer(
        self,
        ws,
        *,
        session_id: Optional[str],
        instruction: Optional[str],
    ) -> None:
        server_start = time.time()
        try:
            obs_frame = await asyncio.wait_for(ws.recv(), timeout=self.ping_timeout)
        except asyncio.TimeoutError:
            logger.error("timed out waiting for observation bytes session=%s", session_id)
            await ws.send(WsMessage.error("missing obs binary frame", session_id).to_json())
            await self._send_infer_end(ws, session_id)
            return

        if not isinstance(obs_frame, (bytes, bytearray)):
            logger.error("expected observation bytes, got %s", type(obs_frame).__name__)
            await ws.send(WsMessage.error("expected binary obs frame", session_id).to_json())
            await self._send_infer_end(ws, session_id)
            return

        try:
            obs = Observation.from_bytes(bytes(obs_frame))
        except Exception as exc:
            logger.error("observation decode failed: %s", exc)
            await ws.send(WsMessage.error(f"obs decode failed: {exc}", session_id).to_json())
            await self._send_infer_end(ws, session_id)
            return

        obs_plan_ts = getattr(obs, "timestamp", None)
        obs_kwargs = obs.kwargs or {}
        obs_capture_ts = obs_kwargs.get("capture_timestamp")
        obs_send_ts = obs_kwargs.get("obs_send_timestamp")
        obs_step = obs_kwargs.get("obs_step")

        if instruction:
            try:
                obs.instruction = instruction
            except Exception:
                pass

        inference_start = time.time()
        try:
            loop = asyncio.get_running_loop()
            async with self._infer_lock:
                action = await loop.run_in_executor(None, self._run_infer_sync, obs)
        except Exception as exc:
            logger.error("inference failed: %s", exc, exc_info=True)
            await ws.send(WsMessage.error(f"inference failed: {exc}", session_id).to_json())
            await self._send_infer_end(ws, session_id)
            return

        if getattr(action, "timestamp", None) is None:
            action.timestamp = obs_plan_ts
        action.kwargs = dict(action.kwargs or {})
        if obs_send_ts is not None:
            action.kwargs["obs_send_timestamp"] = obs_send_ts
        if obs_step is not None:
            action.kwargs["obs_step"] = obs_step
        if obs_capture_ts is not None:
            action.kwargs["capture_timestamp"] = obs_capture_ts

        generated_video = action.kwargs.pop("generated_video", None)
        action_bytes = action.to_bytes()
        inference_time = time.time() - inference_start
        server_total = time.time() - server_start
        has_video = generated_video is not None

        await ws.send(
            WsMessage(
                type=MsgType.POLICY_ACTION,
                session_id=session_id,
                payload={
                    "inference_time": inference_time,
                    "server_total_time": server_total,
                    "has_video": has_video,
                    "action_bytes": len(action_bytes),
                },
            ).to_json()
        )
        await ws.send(action_bytes)

        if has_video:
            try:
                video_arr = np.asarray(generated_video, dtype=np.uint8)
                loop = asyncio.get_running_loop()
                video_format, video_bytes = await loop.run_in_executor(
                    None, _encode_prediction_video, video_arr
                )
                await ws.send(
                    WsMessage(
                        type=MsgType.POLICY_PREDICTION_VIDEO,
                        session_id=session_id,
                        payload={
                            "shape": list(video_arr.shape),
                            "dtype": str(video_arr.dtype),
                            "video_format": video_format,
                            "video_bytes": len(video_bytes),
                        },
                    ).to_json()
                )
                await ws.send(video_bytes)
            except Exception as exc:
                logger.error("prediction.video send failed: %s", exc, exc_info=True)

        await self._send_infer_end(ws, session_id)
        logger.info(
            "infer done session=%s inference_time=%.3fs total=%.3fs action_bytes=%d has_video=%s",
            session_id,
            inference_time,
            server_total,
            len(action_bytes),
            has_video,
        )

    async def _serve_one_connection(self, sep_ws_url: str, ssl_ctx: Optional[ssl.SSLContext]) -> None:
        logger.info("connecting to SEP policy websocket: %s", sep_ws_url)
        async with websockets.connect(
            sep_ws_url,
            max_size=self.max_size,
            open_timeout=15,
            close_timeout=5,
            ping_interval=self.ping_timeout / 2,
            ping_timeout=self.ping_timeout,
            ssl=ssl_ctx,
        ) as ws:
            await ws.send(self._hello().to_json())
            logger.info("policy hello sent policy_id=%s", self.policy_id)

            current_instruction: Optional[str] = None
            while True:
                raw = await ws.recv()
                if isinstance(raw, (bytes, bytearray)):
                    logger.warning("dropped isolated binary frame (%d bytes)", len(raw))
                    continue
                try:
                    msg = WsMessage.from_json(raw)
                except Exception as exc:
                    logger.warning("failed to decode ws header: %s text=%r", exc, raw[:200])
                    continue

                if msg.type == MsgType.POLICY_PING:
                    await ws.send(WsMessage(type=MsgType.POLICY_PONG).to_json())
                    continue

                if msg.type == MsgType.POLICY_SESSION_START:
                    current_instruction = str((msg.payload or {}).get("instruction") or "")
                    try:
                        self.policy.reset()
                    except Exception as exc:
                        logger.warning("policy.reset on session.start failed: %s", exc)
                    self._set_instruction(current_instruction)
                    await ws.send(
                        WsMessage.ack(MsgType.POLICY_SESSION_START, session_id=msg.session_id).to_json()
                    )
                    continue

                if msg.type == MsgType.POLICY_SET_INSTRUCTION:
                    current_instruction = str((msg.payload or {}).get("instruction") or "")
                    self._set_instruction(current_instruction)
                    await ws.send(
                        WsMessage.ack(MsgType.POLICY_SET_INSTRUCTION, session_id=msg.session_id).to_json()
                    )
                    continue

                if msg.type == MsgType.POLICY_RESET:
                    try:
                        self.policy.reset()
                    except Exception as exc:
                        logger.warning("policy.reset failed: %s", exc)
                    await ws.send(WsMessage.ack(MsgType.POLICY_RESET, session_id=msg.session_id).to_json())
                    continue

                if msg.type == MsgType.POLICY_INFER:
                    await self._handle_infer(
                        ws,
                        session_id=msg.session_id,
                        instruction=current_instruction,
                    )
                    continue

                logger.warning("unknown policy message type: %s", msg.type)

    async def run_forever(self) -> None:
        sep_ws_url = build_policy_ws_url(self.sep_url, self.policy_id)
        ssl_ctx = self._ssl_context(sep_ws_url)
        while True:
            try:
                await self._serve_one_connection(sep_ws_url, ssl_ctx)
            except asyncio.CancelledError:
                raise
            except (ConnectionClosed, OSError, asyncio.TimeoutError) as exc:
                logger.warning("policy websocket lost: %s; reconnecting in %.1fs", exc, self.reconnect_delay)
            except Exception as exc:
                logger.error("unexpected policy websocket error: %s", exc, exc_info=True)
            await asyncio.sleep(self.reconnect_delay)

    def run(self) -> None:
        asyncio.run(self.run_forever())


__all__ = ["PolicyClient", "build_policy_ws_url"]
