from __future__ import annotations

import asyncio
from typing import Protocol

from duplex_avatar.schemas import GeneratedBlock, GenerationRequest


class AvatarBackend(Protocol):
    name: str

    async def generate(self, request: GenerationRequest, cancel_event: asyncio.Event) -> GeneratedBlock:
        ...

    async def health(self) -> dict:
        ...

