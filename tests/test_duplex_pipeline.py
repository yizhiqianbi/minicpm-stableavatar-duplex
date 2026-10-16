import asyncio
from pathlib import Path

import pytest

from duplex_avatar.backends.minicpm_o import MiniCPMOConfig
from duplex_avatar.backends.mock import MockBackend
from duplex_avatar.brains.mock import MockBrain
from duplex_avatar.config import AppConfig
from duplex_avatar.schemas import AudioChunk
from duplex_avatar.session import DuplexSession


@pytest.mark.asyncio
async def test_brain_target_audio_drives_avatar_block(tmp_path: Path):
    cfg = AppConfig.load()
    cfg.server.artifacts_dir = str(tmp_path)
    events = []

    async def sink(event):
        events.append(event)

    brain = MockBrain(MiniCPMOConfig(mode="mock", force_listen_count=0, mock_text="ok"))
    session = DuplexSession("s1", cfg, brain, sink, MockBackend(tmp_path))
    await session.start(identity_image_path=None)
    await session.add_audio(AudioChunk(seq=0, timestamp_ms=0, pcm=b"\x00\x00" * 16000))
    for _ in range(100):
        if any(event["type"] == "block_ready" for event in events):
            break
        await asyncio.sleep(0.02)

    block = next(event for event in events if event["type"] == "block_ready")
    assert block["mode"] == "offline_block"
    assert (tmp_path / "s1" / "turn_0000" / "target_audio_0000.wav").is_file()
    assert block["audio_url"].endswith("block_0000.wav")
    await session.stop()
