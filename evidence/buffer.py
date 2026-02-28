"""WitnessAI - Pre-Crime Rolling Frame Buffer"""
from __future__ import annotations
from collections import deque
from datetime import datetime
from dataclasses import dataclass
import logging; logger = logging.getLogger(__name__)
import numpy as np


@dataclass
class BufferedFrame:
    frame: np.ndarray
    timestamp: datetime
    frame_number: int


class RollingBuffer:
    """
    Maintains a fixed-duration rolling buffer of video frames in memory.
    On trigger, returns the buffered frames as the pre-crime window.
    """

    def __init__(self, duration_seconds: int = 30, fps: float = 15.0):
        self.duration_seconds = duration_seconds
        self.fps = fps
        self._max_frames = int(duration_seconds * fps)
        self._buffer: deque[BufferedFrame] = deque(maxlen=self._max_frames)

    def push(self, frame: np.ndarray, timestamp: datetime, frame_number: int) -> None:
        """Add a frame to the rolling buffer."""
        self._buffer.append(BufferedFrame(
            frame=frame,
            timestamp=timestamp,
            frame_number=frame_number,
        ))

    def snapshot(self) -> list[BufferedFrame]:
        """Return a copy of all buffered frames (pre-crime window)."""
        return list(self._buffer)

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def duration_covered(self) -> float:
        """Actual seconds of footage in buffer."""
        if len(self._buffer) < 2:
            return 0.0
        delta = (self._buffer[-1].timestamp - self._buffer[0].timestamp).total_seconds()
        return delta

    def is_ready(self) -> bool:
        """True when buffer holds at least half of the target duration."""
        return self.size >= (self._max_frames // 2)

    def clear(self) -> None:
        self._buffer.clear()

    def update_fps(self, fps: float) -> None:
        """Dynamically adjust buffer capacity when FPS changes."""
        if fps > 0 and fps != self.fps:
            self.fps = fps
            new_max = int(self.duration_seconds * fps)
            if new_max != self._max_frames:
                self._max_frames = new_max
                # Rebuild deque with new maxlen, preserving recent frames
                recent = list(self._buffer)[-new_max:]
                self._buffer = deque(recent, maxlen=new_max)
