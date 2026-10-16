from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    artifacts_dir: str = "artifacts"


@dataclass
class StreamConfig:
    fps: int = 25
    num_frames: int = 81
    block_ms: int = 1000
    max_buffer_ms: int = 6000
    input_sample_rate: int = 16000
    input_channels: int = 1
    output_sample_rate: int = 24000
    max_inflight_blocks: int = 1

    def validate(self) -> None:
        if self.fps <= 0 or self.num_frames <= 0:
            raise ValueError("stream fps and num_frames must be positive")
        if self.block_ms <= 0 or self.max_buffer_ms < self.block_ms:
            raise ValueError("max_buffer_ms must be at least block_ms")
        if self.input_sample_rate <= 0 or self.input_channels <= 0:
            raise ValueError("input audio format must be positive")


@dataclass
class BrainConfig:
    name: str = "mock"
    model_id: str = "openbmb/MiniCPM-o-4_5"
    model_path: str | None = None
    endpoint: str | None = None
    api_key: str | None = None
    remote_mode: str = "audio"
    python_executable: str = "python"
    output_sample_rate: int = 24000
    listen_probability: float = 0.0
    prompt: str = "You are a concise, warm digital human. Answer the user naturally."
    max_response_ms: int = 6000


@dataclass
class BackendConfig:
    name: str = "mock"
    prompt: str = "A friendly digital human answers naturally with subtle head motion."
    height: int = 480
    width: int = 832
    num_inference_steps: int = 30
    cfg_scale: float = 1.0
    seed: int = 42
    model_id: str = "Wan2.1-Fun-V1.1-1.3B-InP"
    variant: str = "rec_vec"
    model_base_path: str = "models"
    processor_path: str | None = None
    lora_path: str | None = None
    # StableAvatar-1.3B subprocess adapter.  These fields are ignored by
    # mock/H3 backends and keep the heavyweight StableAvatar dependencies
    # out of the web server process until the first generation request.
    stableavatar_root: str = "stableavatar"
    stableavatar_python: str = "python"
    stableavatar_inference_script: str = "inference.py"
    stableavatar_config_path: str = "deepspeed_config/wan2.1/wan_civitai.yaml"
    stableavatar_pretrained_model_path: str = "checkpoints/Wan2.1-Fun-V1.1-1.3B"
    stableavatar_transformer_path: str = "checkpoints/StableAvatar-1.3B/transformer3d-rec-vec.pt"
    stableavatar_wav2vec_path: str = "checkpoints/wav2vec2-base-960h"
    stableavatar_gpu_memory_mode: str = "model_cpu_offload"
    stableavatar_sample_steps: int = 30
    stableavatar_overlap_window_length: int = 5
    # 0 means use the session block length.  StableAvatar's upstream demo
    # uses 81, but the duplex scheduler must keep generated audio/video
    # blocks aligned with request.num_frames.
    stableavatar_clip_sample_n_frames: int = 0
    stableavatar_motion_frame: int = 25
    stableavatar_text_guide_scale: float = 3.0
    stableavatar_audio_guide_scale: float = 5.0
    stableavatar_ulysses_degree: int = 1
    stableavatar_ring_degree: int = 1
    stableavatar_cuda_visible_devices: str | None = None


@dataclass
class AppConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    avatar: BackendConfig = field(default_factory=BackendConfig)

    @property
    def backend(self) -> BackendConfig:
        """Legacy alias used by the original H3 adapter and old clients."""
        return self.avatar

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AppConfig":
        data: dict[str, Any] = {}
        if path is not None:
            with Path(path).open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        cfg = cls(
            server=ServerConfig(**data.get("server", {})),
            stream=StreamConfig(**data.get("stream", {})),
            brain=BrainConfig(**data.get("brain", {})),
            avatar=BackendConfig(**(data.get("avatar") or data.get("backend") or {})),
        )
        cfg.stream.validate()
        return cfg

