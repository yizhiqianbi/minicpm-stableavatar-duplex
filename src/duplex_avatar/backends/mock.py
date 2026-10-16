from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from duplex_avatar.artifacts import write_video, write_wav
from duplex_avatar.schemas import GeneratedBlock, GenerationRequest


class MockBackend:
    """A deterministic backend for testing the session protocol before downloading H3."""

    name = "mock"

    def __init__(self, artifacts_dir: str | Path):
        self.artifacts_dir = Path(artifacts_dir)

    async def health(self) -> dict:
        return {"ok": True, "backend": self.name, "mode": "synthetic"}

    async def generate(self, request: GenerationRequest, cancel_event: asyncio.Event) -> GeneratedBlock:
        started = time.perf_counter()
        out_dir = self.artifacts_dir / request.session_id / f"turn_{request.turn_id:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for i in range(request.num_frames):
            if cancel_event.is_set():
                raise asyncio.CancelledError
            # A visible turn/block colour makes stale blocks obvious in a browser.
            phase = (request.block_index * 29 + i * 3) % 255
            image = Image.new("RGB", (request.width, request.height), (30, 45 + phase // 5, 80 + phase // 2))
            draw = ImageDraw.Draw(image)
            draw.ellipse((request.width // 2 - 90, 95, request.width // 2 + 90, 275), fill=(225, 176, 145))
            draw.rectangle((request.width // 2 - 135, 260, request.width // 2 + 135, 430), fill=(40, 120, 145))
            draw.ellipse((request.width // 2 - 48, 170, request.width // 2 - 30, 188), fill="black")
            draw.ellipse((request.width // 2 + 30, 170, request.width // 2 + 48, 188), fill="black")
            draw.text((18, 18), f"MOCK  turn={request.turn_id} block={request.block_index}", fill="white")
            frames.append(image)
            await asyncio.sleep(0)
        video_path = write_video(out_dir / f"block_{request.block_index:04d}.mp4", frames, request.fps)
        audio_path = out_dir / f"block_{request.block_index:04d}.wav"
        if request.target_audio_path and Path(request.target_audio_path).is_file():
            # The mock renderer follows the same contract as StableAvatar: the
            # target (assistant) audio drives the avatar and is returned to the
            # client unchanged.
            shutil.copy2(request.target_audio_path, audio_path)
        else:
            duration = request.num_frames / request.fps
            t = np.arange(int(duration * 32000), dtype=np.float32) / 32000.0
            audio = 0.08 * np.sin(2 * np.pi * (220 + request.block_index * 15) * t)
            audio = np.stack([audio, audio], axis=1)
            write_wav(audio_path, audio, 32000)
        return GeneratedBlock(
            session_id=request.session_id,
            turn_id=request.turn_id,
            block_index=request.block_index,
            video_path=video_path,
            audio_path=str(audio_path),
            frames=request.num_frames,
            fps=request.fps,
            latency_ms=round((time.perf_counter() - started) * 1000),
        )

