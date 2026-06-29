"""Lightweight mock slave for SEP smoke tests.

This script is intentionally outside ``sdk``. It is a debug helper that mimics
the robot-side websocket protocol just enough to test SEP + policy connectivity.

Example:
    python PhySeek/examples/mock_slave.py --sep wss://127.0.0.1:9668 --slave-id mock-slave
"""

from __future__ import annotations

import argparse
import asyncio
import ssl
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import websockets

PHYSEEK_ROOT = Path(__file__).resolve().parents[1]
if str(PHYSEEK_ROOT) not in sys.path:
    sys.path.insert(0, str(PHYSEEK_ROOT))

from sdk import Action, MsgType, Observation, WsMessage


def _now() -> float:
    return time.time()


def _frame(step: int, height: int, width: int) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    phase = (step * 37) % 255
    image[:, :, 0] = phase
    image[:, :, 1] = (phase + 70) % 255
    image[:, :, 2] = (phase + 140) % 255
    bar = max(1, int(width * ((step % 20) + 1) / 20))
    image[height - 10 : height, :bar, :] = 255
    return image


def _make_obs(args: argparse.Namespace, step: int) -> bytes:
    obs = Observation(
        images={"front": _frame(step, args.height, args.width)},
        state=np.zeros(args.action_dim, dtype=np.float32),
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
        timestamp=_now(),
        capture_timestamp=_now(),
        obs_step=step,
    )
    return obs.to_bytes()


async def _send_obs(ws: Any, args: argparse.Namespace, session_id: str, run_id: str, step: int) -> None:
    obs_bytes = _make_obs(args, step)
    await ws.send(
        WsMessage(
            type=MsgType.SLAVE_OBS,
            session_id=session_id,
            payload={"run_id": run_id, "step": step, "obs_bytes": len(obs_bytes)},
        ).to_json()
    )
    await ws.send(obs_bytes)
    print(f"[slave:{args.slave_id}] slave.obs step={step} bytes={len(obs_bytes)}")


async def _heartbeat(ws: Any) -> None:
    while True:
        await asyncio.sleep(20)
        await ws.send(WsMessage(type=MsgType.SLAVE_HEARTBEAT).to_json())


async def _execute_action(args: argparse.Namespace, action_bytes: bytes) -> None:
    try:
        action = Action.from_bytes(action_bytes)
        frame_count = int(np.asarray(action.action).shape[0]) if np.asarray(action.action).ndim >= 2 else 1
    except Exception:
        frame_count = args.chunk_len
    duration = frame_count / args.hz if args.hz > 0 else 0.0
    await asyncio.sleep(min(duration, args.max_action_sleep))


def _ssl_context(url: str, insecure: bool) -> ssl.SSLContext | None:
    if not url.startswith("wss://"):
        raise ValueError("mock_slave only supports secure WebSocket SEP URLs starting with wss://")
    ctx = ssl.create_default_context()
    if insecure:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


async def run(args: argparse.Namespace) -> None:
    url = f"{args.sep.rstrip('/')}/api/execution/ws/slave/{args.slave_id}"
    connect_kwargs: dict[str, Any] = {
        "max_size": None,
        "ping_interval": 20,
        "ping_timeout": 30,
        "ssl": _ssl_context(url, args.insecure_tls),
    }

    print(f"[slave:{args.slave_id}] connect -> {url}")
    async with websockets.connect(url, **connect_kwargs) as ws:
        hello = WsMessage(
            type=MsgType.SLAVE_HELLO,
            payload={
                "slave_id": args.slave_id,
                "controller": "mock",
                "executor": "mock",
                "action_space": args.action_space,
                "action_dim": args.action_dim,
                "robot_spec": {
                    "robot_preset": args.robot_preset,
                    "action_space": args.action_space,
                    "action_dim": args.action_dim,
                },
                "cameras": ["front"],
            },
        )
        await ws.send(hello.to_json())
        print(f"[slave:{args.slave_id}] slave.hello sent")

        hb_task = asyncio.create_task(_heartbeat(ws))
        session_id = ""
        run_id = ""
        step = 0
        running = False
        paused = False

        try:
            async for raw in ws:
                if isinstance(raw, (bytes, bytearray)):
                    print(f"[slave:{args.slave_id}] dropped isolated binary frame bytes={len(raw)}")
                    continue

                msg = WsMessage.from_json(raw)
                payload = msg.payload or {}

                if msg.type == MsgType.CMD_START:
                    session_id = msg.session_id or ""
                    run_id = str(payload.get("run_id") or "run_001")
                    step = 0
                    running = True
                    paused = False
                    print(f"[slave:{args.slave_id}] cmd.start session={session_id} run_id={run_id}")
                    await _send_obs(ws, args, session_id, run_id, step)
                    continue

                if msg.type == MsgType.CMD_ACTION:
                    action_frame = await ws.recv()
                    if not isinstance(action_frame, (bytes, bytearray)):
                        print(f"[slave:{args.slave_id}] cmd.action expected bytes, got text")
                        continue
                    print(f"[slave:{args.slave_id}] cmd.action bytes={len(action_frame)}")
                    await _execute_action(args, bytes(action_frame))
                    if running and not paused and session_id:
                        step += 1
                        await _send_obs(ws, args, session_id, run_id, step)
                    continue

                if msg.type == MsgType.CMD_PAUSE:
                    paused = True
                    print(f"[slave:{args.slave_id}] cmd.pause")
                    continue

                if msg.type == MsgType.CMD_RESUME:
                    paused = False
                    print(f"[slave:{args.slave_id}] cmd.resume")
                    if running and session_id:
                        step += 1
                        await _send_obs(ws, args, session_id, run_id, step)
                    continue

                if msg.type == MsgType.CMD_STOP:
                    running = False
                    paused = False
                    print(f"[slave:{args.slave_id}] cmd.stop")
                    await ws.send(
                        WsMessage.ack(
                            MsgType.CMD_STOP,
                            session_id=msg.session_id,
                            extra={"run_id": run_id, "episode_dir": "", "mock": True},
                        ).to_json()
                    )
                    continue

                if msg.type == MsgType.CMD_EXPORT_UPLOAD:
                    print(f"[slave:{args.slave_id}] cmd.export.upload ignored for lightweight mock")
                    await ws.send(
                        WsMessage(
                            type=MsgType.EXPORT_DONE,
                            session_id=msg.session_id,
                            payload={"files": 0, "bytes": 0, "mock": True},
                        ).to_json()
                    )
                    continue

                if msg.type == MsgType.ACK:
                    continue

                print(f"[slave:{args.slave_id}] ignored msg type={msg.type}")
        finally:
            hb_task.cancel()
            await asyncio.gather(hb_task, return_exceptions=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight mock slave for SEP smoke tests.")
    parser.add_argument(
        "--sep",
        required=True,
        help="SEP secure websocket base URL",
    )
    parser.add_argument("--slave-id", default="mock-slave")
    parser.add_argument("--robot-preset", default="mock")
    parser.add_argument("--action-space", default="eef")
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--chunk-len", type=int, default=8)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--image-format", choices=["raw", "jpeg", "png"], default="jpeg")
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--max-action-sleep", type=float, default=1.0)
    parser.add_argument(
        "--insecure-tls",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable TLS certificate verification for local self-signed SEP debugging.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(run(parse_args()))
    except KeyboardInterrupt:
        print("\n[slave] interrupted")
