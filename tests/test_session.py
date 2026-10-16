import asyncio
import base64
import json
from pathlib import Path

import pytest

from duplex_avatar.backends.mock import MockBackend
from duplex_avatar.config import AppConfig
from duplex_avatar.schemas import AudioChunk
from duplex_avatar.session import DuplexSession


@pytest.mark.asyncio
async def test_interrupt_increments_turn_and_cancels(tmp_path: Path):
    cfg = AppConfig.load()
    cfg.server.artifacts_dir = str(tmp_path)
    cfg.stream.block_ms = 1
    events = []

    async def sink(event):
        events.append(event)

    session = DuplexSession("s1", cfg, MockBackend(tmp_path), sink)
    await session.start()
    await session.add_audio(AudioChunk(seq=0, timestamp_ms=0, pcm=b"\x00\x00" * 160, sample_rate=16000, channels=1))
    await asyncio.sleep(0.01)
    previous_turn = session.snapshot.turn_id
    await session.interrupt()
    assert session.snapshot.turn_id == previous_turn + 1
    assert session.snapshot.state == "listening"
    assert any(e["type"] == "interrupt" for e in events)

