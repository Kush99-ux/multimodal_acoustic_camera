"""
fusion
======

Fusion subsystem for the
Real-Time Multimodal Acoustic Camera.

Provides:

- Timestamp synchronization
- Synchronized frame pairing
- Multimodal result models
- Multimodal fusion engine

Author: Kush Sahu
Project: Real-Time Multimodal Acoustic Camera
"""

from .synchronization import (
    SYNC_TOLERANCE_MS,
    SYNC_TOLERANCE_SECONDS,
    SynchronizedFramePair,
    TimestampSynchronizer,
)

from .models import (
    MultimodalResult,
)

from .engine import (
    FusionEngine,
)


__all__ = [
    # Synchronization
    "SYNC_TOLERANCE_MS",
    "SYNC_TOLERANCE_SECONDS",
    "SynchronizedFramePair",
    "TimestampSynchronizer",

    # Models
    "MultimodalResult",

    # Engine
    "FusionEngine",
]