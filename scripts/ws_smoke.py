#!/usr/bin/env python3
"""Send one synthetic 16 kHz microphone block to the local mock server."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import struct

import websockets


async def main(url: str) -> None:
    sample_rate = 16_000
    duration_ms = 2_400
    samples = bytearray()
    for index in range(sample_rate * duration_ms // 1000):
        value = int(0.1 * 32767 * math.sin(2 * math.pi * 220 * index / sample_rate))
        samples.extend(struct.pack("<h", value))
    async with websockets.connect(url) as socket:
        await socket.send('{"type":"start","prompt":"Audio 1 is user speech. The avatar listens and responds naturally."}')
        print(await socket.recv())
        await socket.send(json.dumps({
            "type": "audio",
            "seq": 0,
            "timestamp_ms": 0,
            "sample_rate": sample_rate,
            "channels": 1,
            "pcm_b64": base64.b64encode(samples).decode("ascii"),
        }))
        for _ in range(12):
            event = json.loads(await socket.recv())
            if event.get("type") == "brain" and event.get("kind") == "audio":
                event.pop("audio_b64", None)
            print(json.dumps(event, ensure_ascii=False))
            if event.get("type") == "block_ready":
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws/demo")
    args = parser.parse_args()
    asyncio.run(main(args.url))
