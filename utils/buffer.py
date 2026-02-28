"""
WitnessAI — Ring Buffer
Maintains a rolling window of video frames for pre-crime clip extraction.
"""
from collections import deque
from datetime import datetime
from typing import Optional, List, Tuple
import threading
import numpy as np


class FrameRingBuffer:
    """
    Thread-safe ring buffer that holds the last N video frames.
    Used to extract pre-crime footage when an anomaly is triggered.
    """

    def __init__(self, maxlen: int = 900):
        """
        Args:
            maxlen: Maximum number of frames to retain (~30s at 30fps).
        """
        self._buffer: deque[Tuple[int, datetime, np.ndarray]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self.maxlen = maxlen

    def push(self, frame_number: int, timestamp: datetime, frame: np.ndarray) -> None:
        """Add a frame to the buffer."""
        with self._lock:
            self._buffer.append((frame_number, timestamp, frame))

    def get_last_n_seconds(self, seconds: float, fps: float = 15.0) -> List[Tuple[int, datetime, np.ndarray]]:
        """
        Return frames from the last `seconds` seconds.

        Args:
            seconds: How many seconds of history to return.
            fps: Frames per second — used to estimate frame count if timestamps
                 are not available.

        Returns:
            List of (frame_number, timestamp, frame) tuples, oldest first.
        """
        n_frames = int(seconds * fps)
        with self._lock:
            frames = list(self._buffer)

        if not frames:
            return []

        # Filter by timestamp if possible
        cutoff_ts = datetime.utcnow()
        result = []
        for fn, ts, frame in reversed(frames):
            delta = (cutoff_ts - ts).total_seconds()
            if delta <= seconds:
                result.append((fn, ts, frame))
            else:
                break

        return list(reversed(result)) or frames[-min(n_frames, len(frames)):]

    def get_all(self) -> List[Tuple[int, datetime, np.ndarray]]:
        """Return all buffered frames."""
        with self._lock:
            return list(self._buffer)

    def clear(self) -> None:
        """Empty the buffer."""
        with self._lock:
            self._buffer.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    def latest_frame_number(self) -> Optional[int]:
        with self._lock:
            if self._buffer:
                return self._buffer[-1][0]
        return None
