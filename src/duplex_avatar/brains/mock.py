"""Deterministic brain for the duplex scheduler and WebSocket smoke tests."""

from __future__ import annotations

from typing import Any, Sequence

from duplex_avatar.backends.minicpm_o import MiniCPMOBackend, MiniCPMOConfig, MiniCPMEvent


class MockBrain(MiniCPMOBackend):
    """A MiniCPM-o-shaped mock that never imports torch or opens a socket."""

    name = "mock-brain"

    def __init__(self, config: Any = None, **overrides: Any):
        if config is None:
            config = MiniCPMOConfig(mode="mock")
        elif isinstance(config, dict):
            config = dict(config)
            config["mode"] = "mock"
        else:
            # Keep a supplied config but force the backend mode.  This prevents
            # a mock unit test from accidentally loading a local model.
            config.mode = "mock"
        super().__init__(config, **overrides)


__all__ = ["MockBrain"]
