"""Small protocol shared by local, remote, and mock conversational brains."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from duplex_avatar.backends.minicpm_o import MiniCPMEvent as BrainEvent


class Brain(Protocol):
    """Stateful one-second-unit duplex brain interface.

    A brain owns the conversational KV cache.  The video renderer should only
    consume the returned audio events; it must not feed target audio back as
    the user's next input.
    """

    name: str

    async def start(
        self,
        *,
        system_prompt: str | None = None,
        ref_audio_path: str | None = None,
        tts_ref_audio_path: str | None = None,
    ) -> None:
        ...

    async def feed_audio(
        self,
        audio: bytes | bytearray | memoryview | Any,
        *,
        sample_rate: int | None = None,
        channels: int = 1,
        force_listen: bool = False,
        video_frames: Sequence[Any] | None = None,
    ) -> list[BrainEvent]:
        ...

    async def interrupt(self) -> None:
        ...

    async def close(self) -> None:
        ...

    async def health(self) -> dict[str, Any]:
        ...


__all__ = ["Brain", "BrainEvent"]
