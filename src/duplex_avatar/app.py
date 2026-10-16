from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from duplex_avatar.backends.h3_diffsynth import H3DiffSynthBackend
from duplex_avatar.backends.mock import MockBackend
from duplex_avatar.backends.stableavatar import StableAvatarBackend
from duplex_avatar.backends.minicpm_o import MiniCPMOBackend, MiniCPMOConfig
from duplex_avatar.brains.mock import MockBrain
from duplex_avatar.config import AppConfig
from duplex_avatar.schemas import AudioChunk
from duplex_avatar.session import SessionManager


def _make_brain(cfg: AppConfig):
    brain_cfg = cfg.brain
    mode = brain_cfg.name.lower().replace("_", "-")
    if mode == "mock":
        return MockBrain(MiniCPMOConfig(
            mode="mock",
            force_listen_count=0,
            mock_text=brain_cfg.prompt or "你好，我在听。",
            audio_sample_rate=cfg.stream.input_sample_rate,
        ))
    if mode in {"minicpm-o", "minicpm", "local", "remote", "realtime-ws", "realtime"}:
        selected_mode = "remote" if mode in {"remote", "realtime-ws", "realtime", "minicpm-o", "minicpm"} else "local"
        return MiniCPMOBackend(MiniCPMOConfig(
            mode=selected_mode,
            model_path=brain_cfg.model_path or brain_cfg.model_id,
            remote_url=brain_cfg.endpoint or "https://minicpmo45.modelbest.cn",
            remote_mode=brain_cfg.remote_mode,
            api_key=brain_cfg.api_key,
            system_prompt=brain_cfg.prompt,
            audio_sample_rate=cfg.stream.input_sample_rate,
        ))
    raise ValueError(f"unknown brain: {brain_cfg.name}")


def _make_avatar(cfg: AppConfig):
    name = cfg.avatar.name.lower()
    if name == "mock":
        return MockBackend(cfg.server.artifacts_dir)
    if name in {"stableavatar", "stableavatar-1.3b"}:
        return StableAvatarBackend(cfg.server.artifacts_dir, cfg.avatar)
    if name in {"h3", "h3-diffsynth"}:
        return H3DiffSynthBackend(cfg.server.artifacts_dir, cfg.avatar)
    raise ValueError(f"unknown avatar backend: {cfg.avatar.name}")


def create_app(
    config_path: str | None = None,
    backend_override: str | None = None,
    brain_override: str | None = None,
) -> FastAPI:
    cfg = AppConfig.load(config_path)
    if backend_override is not None:
        cfg.avatar.name = backend_override
    if brain_override is not None:
        cfg.brain.name = brain_override
    Path(cfg.server.artifacts_dir).mkdir(parents=True, exist_ok=True)
    avatar = _make_avatar(cfg)
    manager = SessionManager(cfg, lambda: _make_brain(cfg), avatar)
    app = FastAPI(title="MiniCPM-o + StableAvatar Duplex", version="0.2.0")
    app.mount("/artifacts", StaticFiles(directory=cfg.server.artifacts_dir), name="artifacts")

    @app.get("/health")
    async def health():
        brain = await _make_brain(cfg).health()
        renderer = await avatar.health()
        return {"ok": bool(brain.get("ok")) and bool(renderer.get("ok")), "brain": brain, "avatar": renderer}

    @app.get("/sessions/{session_id}")
    async def session_state(session_id: str):
        session = manager.sessions.get(session_id)
        if session is None:
            return {"session_id": session_id, "state": "missing"}
        return session.snapshot.__dict__

    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str):
        await websocket.accept()
        outbound: asyncio.Queue = asyncio.Queue()

        async def sink(event: dict) -> None:
            await outbound.put(event)

        session = manager.sessions.get(session_id) or manager.create(sink, session_id)

        async def sender():
            while True:
                event = await outbound.get()
                await websocket.send_text(json.dumps(event, ensure_ascii=False))

        sender_task = asyncio.create_task(sender())
        try:
            while True:
                message = json.loads(await websocket.receive_text())
                kind = message.get("type")
                if kind == "start":
                    await session.start(
                        message.get("prompt"),
                        message.get("identity_image_path"),
                        message.get("ref_audio_path"),
                    )
                elif kind == "audio":
                    await session.add_audio(AudioChunk.from_message(message))
                elif kind == "flush":
                    await session.flush()
                elif kind == "interrupt":
                    await session.interrupt(message.get("reason", "client"))
                elif kind == "stop":
                    await session.stop()
                    break
                else:
                    await sink({"type": "error", "message": f"unknown message type: {kind}"})
        except WebSocketDisconnect:
            await session.stop()
        finally:
            sender_task.cancel()

    return app
