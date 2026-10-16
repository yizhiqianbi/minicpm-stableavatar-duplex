from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2 and samples.shape[0] <= 4 and samples.shape[1] > samples.shape[0]:
        samples = samples.T
    sf.write(str(path), samples, sample_rate, subtype="PCM_16")
    return str(path)


def write_video(path: str | Path, frames: Iterable, fps: int) -> str:
    """Write PIL frames to MP4 through ffmpeg, keeping the backend dependency small."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_list = list(frames)
    if not frame_list:
        raise ValueError("cannot write an empty video")
    frame_dir = path.parent / f".{path.stem}_frames"
    frame_dir.mkdir(exist_ok=True)
    try:
        for index, frame in enumerate(frame_list):
            frame.save(frame_dir / f"{index:06d}.png")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required to export MP4 artifacts")
        command = [
            ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
            "-i", str(frame_dir / "%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path),
        ]
        subprocess.run(command, check=True)
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)
    return str(path)


def write_pcm16_wav(path: str | Path, pcm: bytes, sample_rate: int, channels: int) -> str:
    """Write raw little-endian PCM16 from the WebSocket protocol."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm)
    return str(path)

