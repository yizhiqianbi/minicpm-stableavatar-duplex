"""Public brain names for the MiniCPM-o adapter.

The implementation lives in ``backends.minicpm_o`` so older callers that use
the backend namespace keep working.  This module is the preferred import for
new duplex session code.
"""

from duplex_avatar.backends.minicpm_o import (
    MiniCPMOBackend,
    MiniCPMOConfig,
    MiniCPMEvent,
    MockMiniCPMOBackend,
)


class MiniCPMOBrain(MiniCPMOBackend):
    """Alias with a brain-oriented name for session wiring."""

    name = "minicpm-o-brain"


# Short aliases make the adapter convenient to import in small experiments.
MiniCPMBrain = MiniCPMOBrain
MiniCPMBackend = MiniCPMOBackend


__all__ = [
    "MiniCPMOBrain",
    "MiniCPMBrain",
    "MiniCPMOBackend",
    "MiniCPMBackend",
    "MiniCPMOConfig",
    "MiniCPMEvent",
    "MockMiniCPMOBackend",
]
