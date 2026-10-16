from __future__ import annotations

import asyncio
import time
import uuid
import wave
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np

from duplex_avatar.artifacts import write_pcm16_wav, write_wav
from duplex_avatar.brains.base import Brain
from duplex_avatar.schemas import AudioChunk, GenerationRequest, SessionSnapshot


EventSink = Callable[[dict], Awaitable[None]]


def _read_pcm16_wav(path: str | Path) -> tuple[bytes, int, int]:
    """Read the raw PCM payload written by ``write_pcm16_wav``."""
    with wave.open(str(path), "rb") as reader:
        return reader.readframes(reader.getnframes()), reader.getframerate(), reader.getnchannels()


class DuplexSession:
    """A small full-duplex scheduler with explicit listen/speak interruption.

    The brain owns the conversational state and produces target audio. The
    avatar backend only sees that target audio, so microphone input can keep
    arriving while a previous video block is rendering. A turn id protects the
    client from late results after an interruption.
    """

    def __init__(
        self,
        session_id: str,
        cfg,
        brain_or_backend,
        event_sink: EventSink,
        avatar_backend=None,
    ):
        # Backwards compatibility: old callers passed one H3/mock backend. It
        # still works by using a mock brain and treating that object as avatar.
        if avatar_backend is None:
            from duplex_avatar.brains.mock import MockBrain

            self.brain: Brain = MockBrain()
            self.avatar_backend = brain_or_backend
        else:
            self.brain = brain_or_backend
            self.avatar_backend = avatar_backend
        self.session_id = session_id
        self.cfg = cfg
        self.event_sink = event_sink
        self.snapshot = SessionSnapshot(session_id=session_id)
        self.prompt = cfg.avatar.prompt
        self.identity_image_path: str | None = None
        self.audio_buffer = bytearray()
        self.audio_ms = 0
        self.block_index = 0
        self.cancel_event = asyncio.Event()
        self.generation_task: asyncio.Task | None = None
        self.created_at = time.perf_counter()
        self._lock = asyncio.Lock()
        self._started = False

    async def emit(self, event: dict) -> None:
        event.setdefault("session_id", self.session_id)
        event.setdefault("turn_id", self.snapshot.turn_id)
        self.snapshot.events.append(dict(event))
        # Keep snapshots bounded for clients that inspect state after a long
        # session.
        if len(self.snapshot.events) > 100:
            del self.snapshot.events[:-100]
        await self.event_sink(event)

    async def start(
        self,
        prompt: str | None = None,
        identity_image_path: str | None = None,
        ref_audio_path: str | None = None,
    ) -> None:
        self.prompt = prompt or self.prompt or self.cfg.brain.prompt
        self.identity_image_path = identity_image_path
        await self.brain.start(system_prompt=self.prompt, ref_audio_path=ref_audio_path)
        self._started = True
        self.snapshot.state = "listening"
        await self.emit({"type": "state", "state": self.snapshot.state})

    async def add_audio(self, chunk: AudioChunk) -> None:
        async with self._lock:
            if chunk.sample_rate != self.cfg.stream.input_sample_rate or chunk.channels != self.cfg.stream.input_channels:
                await self.emit({
                    "type": "error",
                    "message": (
                        f"audio format must be {self.cfg.stream.input_sample_rate} Hz / "
                        f"{self.cfg.stream.input_channels} channel(s); got {chunk.sample_rate} Hz / {chunk.channels}"
                    ),
                })
                return
            # Only interrupt once when a user starts talking over an avatar.
            # Subsequent chunks are part of the new turn and must accumulate.
            if self.snapshot.state == "speaking":
                await self.interrupt(reason="user_audio")
            self.audio_buffer.extend(chunk.pcm)
            bytes_per_ms = max(1, chunk.sample_rate * chunk.channels * 2 // 1000)
            self.audio_ms += len(chunk.pcm) // bytes_per_ms
            self.snapshot.input_cutoff_ms = chunk.timestamp_ms + self.audio_ms
            if self.audio_ms >= self.cfg.stream.block_ms and self.generation_task is None:
                await self._launch_generation_locked()

    async def flush(self) -> None:
        async with self._lock:
            if self.audio_buffer and self.generation_task is None:
                await self._launch_generation_locked()

    async def _launch_generation_locked(self) -> None:
        if not self.audio_buffer or self.snapshot.state == "stopped":
            return
        self.snapshot.state = "thinking"
        turn_id = self.snapshot.turn_id
        block_index = self.block_index
        self.block_index += 1
        turn_dir = Path(self.cfg.server.artifacts_dir) / self.session_id / f"turn_{turn_id:04d}"
        user_audio_path = turn_dir / f"user_audio_{block_index:04d}.wav"
        write_pcm16_wav(
            user_audio_path,
            bytes(self.audio_buffer),
            self.cfg.stream.input_sample_rate,
            self.cfg.stream.input_channels,
        )
        self.audio_buffer.clear()
        self.audio_ms = 0
        request = GenerationRequest(
            session_id=self.session_id,
            turn_id=turn_id,
            block_index=block_index,
            prompt=self.prompt,
            identity_image_path=self.identity_image_path,
            user_audio_path=str(user_audio_path),
            num_frames=self.cfg.stream.num_frames,
            fps=self.cfg.stream.fps,
            height=self.cfg.avatar.height,
            width=self.cfg.avatar.width,
            num_inference_steps=self.cfg.avatar.num_inference_steps,
            cfg_scale=self.cfg.avatar.cfg_scale,
            seed=self.cfg.avatar.seed,
            cutoff_ms=self.snapshot.input_cutoff_ms,
        )
        self.cancel_event = asyncio.Event()
        self.generation_task = asyncio.create_task(self._run_brain_and_render(request, self.cancel_event))
        await self.emit({"type": "state", "state": self.snapshot.state, "block_index": block_index})

    async def _run_brain_and_render(self, request: GenerationRequest, cancel_event: asyncio.Event) -> None:
        started = time.perf_counter()
        try:
            pcm, sample_rate, channels = _read_pcm16_wav(request.user_audio_path)
            events = await self.brain.feed_audio(pcm, sample_rate=sample_rate, channels=channels)
            if cancel_event.is_set() or request.turn_id != self.snapshot.turn_id:
                return

            text_parts: list[str] = []
            target_audio: list[np.ndarray] = []
            target_rate = self.cfg.brain.output_sample_rate
            saw_listen = False
            for brain_event in events:
                payload = brain_event.as_message() if hasattr(brain_event, "as_message") else {
                    "type": "brain", "kind": getattr(brain_event, "kind", "unknown")
                }
                payload["type"] = "brain"
                if getattr(brain_event, "kind", None) == "audio":
                    # The audio delta is persisted as target_audio_*.wav below;
                    # the WebSocket only needs metadata before block_ready.
                    payload.pop("audio_b64", None)
                    payload["audio_bytes"] = len(getattr(brain_event, "audio_pcm", b"") or b"")
                await self.emit(payload)
                kind = getattr(brain_event, "kind", None)
                if kind == "listen":
                    saw_listen = True
                elif kind == "text":
                    text = str(getattr(brain_event, "text", "") or "")
                    if text:
                        text_parts.append(text)
                elif kind == "audio" and getattr(brain_event, "audio_pcm", None):
                    audio_bytes = brain_event.audio_pcm
                    if getattr(brain_event, "audio_format", "float32") == "float32":
                        array = np.frombuffer(audio_bytes, dtype="<f4").astype(np.float32, copy=False)
                    else:
                        array = np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32) / 32768.0
                    target_audio.append(np.ascontiguousarray(array))
                    target_rate = int(getattr(brain_event, "audio_sample_rate", target_rate))

            # A listen decision is a valid full-duplex outcome: the user gets
            # no avatar block, and the next microphone unit continues the turn.
            if not target_audio:
                self.snapshot.state = "listening"
                if saw_listen or not events:
                    await self.emit({"type": "state", "state": self.snapshot.state, "reason": "brain_listen"})
                return

            target_dir = Path(self.cfg.server.artifacts_dir) / self.session_id / f"turn_{request.turn_id:04d}"
            target_audio_path = target_dir / f"target_audio_{request.block_index:04d}.wav"
            target = np.concatenate(target_audio)
            write_wav(target_audio_path, target, target_rate)
            request.target_audio_path = str(target_audio_path)
            if text_parts:
                request.prompt = " ".join(text_parts)
            if cancel_event.is_set() or request.turn_id != self.snapshot.turn_id:
                return
            block = await self.avatar_backend.generate(request, cancel_event)
            if cancel_event.is_set() or request.turn_id != self.snapshot.turn_id:
                return
            self.snapshot.state = "speaking"
            await self.emit({"type": "state", "state": self.snapshot.state})
            await self.emit({
                "type": "block_ready",
                "block_index": block.block_index,
                "video_url": f"/artifacts/{self.session_id}/turn_{block.turn_id:04d}/{Path(block.video_path).name}",
                "audio_url": f"/artifacts/{self.session_id}/turn_{block.turn_id:04d}/{Path(block.audio_path).name}",
                "frames": block.frames,
                "fps": block.fps,
                "latency_ms": block.latency_ms,
                "mode": block.mode,
                "text": " ".join(text_parts),
                "ttfc_ms": round((time.perf_counter() - self.created_at) * 1000),
            })
        except asyncio.CancelledError:
            return
        except Exception as exc:
            await self.emit({"type": "error", "message": str(exc), "elapsed_ms": round((time.perf_counter() - started) * 1000)})
            if request.turn_id == self.snapshot.turn_id and self.snapshot.state != "stopped":
                self.snapshot.state = "listening"
        finally:
            if self.generation_task is asyncio.current_task():
                self.generation_task = None
            # Audio that arrived while the previous block was rendering should
            # start promptly after it completes.
            if self.audio_ms >= self.cfg.stream.block_ms and self.snapshot.state not in {"stopped", "speaking"}:
                async with self._lock:
                    if self.generation_task is None and self.audio_ms >= self.cfg.stream.block_ms:
                        await self._launch_generation_locked()

    async def interrupt(self, reason: str = "user_audio") -> None:
        self.cancel_event.set()
        if self.generation_task and not self.generation_task.done():
            self.generation_task.cancel()
        self.generation_task = None
        await self.brain.interrupt()
        self.snapshot.turn_id += 1
        self.snapshot.state = "interrupted"
        await self.emit({"type": "interrupt", "reason": reason})
        self.snapshot.state = "listening"
        await self.emit({"type": "state", "state": self.snapshot.state})

    async def stop(self) -> None:
        self.cancel_event.set()
        if self.generation_task and not self.generation_task.done():
            self.generation_task.cancel()
        self.generation_task = None
        await self.brain.close()
        self.snapshot.state = "stopped"
        await self.emit({"type": "state", "state": self.snapshot.state})


class SessionManager:
    def __init__(self, cfg, brain_factory, avatar_backend=None):
        self.cfg = cfg
        # Accept either a factory or a prebuilt brain for small integrations.
        self.brain_factory = brain_factory if callable(brain_factory) else (lambda: brain_factory)
        self.avatar_backend = avatar_backend
        self.sessions: dict[str, DuplexSession] = {}

    def create(self, event_sink: EventSink, session_id: str | None = None) -> DuplexSession:
        session_id = session_id or uuid.uuid4().hex[:12]
        session = DuplexSession(session_id, self.cfg, self.brain_factory(), event_sink, self.avatar_backend)
        self.sessions[session_id] = session
        return session
