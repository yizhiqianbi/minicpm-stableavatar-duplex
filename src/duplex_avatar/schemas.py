from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Literal


SessionState = Literal["idle", "listening", "thinking", "speaking", "interrupted", "stopped"]


@dataclass(slots=True)
class AudioChunk:
    seq: int
    timestamp_ms: int
    pcm: bytes
    sample_rate: int = 16000
    channels: int = 1

    @classmethod
    def from_message(cls, message: dict) -> "AudioChunk":
        encoded = message.get("pcm_b64", "")
        if not encoded:
            raise ValueError("audio message requires pcm_b64")
        return cls(
            seq=int(message["seq"]),
            timestamp_ms=int(message.get("timestamp_ms", 0)),
            pcm=base64.b64decode(encoded),
            sample_rate=int(message.get("sample_rate", 16000)),
            channels=int(message.get("channels", 1)),
        )


@dataclass(slots=True)
class GenerationRequest:
    session_id: str
    turn_id: int
    block_index: int
    prompt: str
    identity_image_path: str | None
    user_audio_path: str | None
    history: tuple[str, ...] = ()
    num_frames: int = 56
    fps: int = 24
    height: int = 480
    width: int = 832
    num_inference_steps: int = 50
    cfg_scale: float = 1.0
    seed: int = 42
    cutoff_ms: int = 0
    # Target/agent audio is the audio StableAvatar should animate.  During
    # the initial baseline the session only has user_audio_path, so the
    # adapter falls back to it; the MiniCPM-o integration can populate this
    # field without changing the backend protocol.
    target_audio_path: str | None = None


@dataclass(slots=True)
class GeneratedBlock:
    session_id: str
    turn_id: int
    block_index: int
    video_path: str
    audio_path: str
    frames: int
    fps: int
    latency_ms: int
    mode: str = "offline_block"


@dataclass
class SessionSnapshot:
    session_id: str
    state: SessionState = "idle"
    turn_id: int = 0
    block_index: int = 0
    input_cutoff_ms: int = 0
    events: list[dict] = field(default_factory=list)

