"""Tests for Rolling Evidence Buffer"""
import pytest
import numpy as np
from datetime import datetime, timedelta
from evidence.buffer import RollingBuffer, BufferedFrame


def make_frame(h=100, w=100):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestRollingBuffer:
    def test_push_adds_frame(self):
        buf = RollingBuffer(duration_seconds=10, fps=5.0)
        buf.push(make_frame(), datetime.utcnow(), 1)
        assert buf.size == 1

    def test_max_size_enforced(self):
        buf = RollingBuffer(duration_seconds=2, fps=5.0)  # max 10 frames
        for i in range(20):
            buf.push(make_frame(), datetime.utcnow(), i)
        assert buf.size == 10  # capped at maxlen

    def test_snapshot_returns_all_frames(self):
        buf = RollingBuffer(duration_seconds=10, fps=5.0)
        for i in range(5):
            buf.push(make_frame(), datetime.utcnow(), i)
        snap = buf.snapshot()
        assert len(snap) == 5

    def test_snapshot_does_not_clear_buffer(self):
        buf = RollingBuffer(duration_seconds=10, fps=5.0)
        buf.push(make_frame(), datetime.utcnow(), 1)
        buf.snapshot()
        assert buf.size == 1

    def test_buffered_frame_has_correct_number(self):
        buf = RollingBuffer(duration_seconds=10, fps=5.0)
        buf.push(make_frame(), datetime.utcnow(), 42)
        snap = buf.snapshot()
        assert snap[0].frame_number == 42

    def test_is_ready_half_full(self):
        buf = RollingBuffer(duration_seconds=2, fps=10.0)  # max 20 frames
        assert not buf.is_ready()
        for i in range(10):  # exactly half
            buf.push(make_frame(), datetime.utcnow(), i)
        assert buf.is_ready()

    def test_clear_empties_buffer(self):
        buf = RollingBuffer(duration_seconds=5, fps=5.0)
        for i in range(10):
            buf.push(make_frame(), datetime.utcnow(), i)
        buf.clear()
        assert buf.size == 0

    def test_old_frames_evicted_when_full(self):
        buf = RollingBuffer(duration_seconds=1, fps=5.0)  # max 5 frames
        for i in range(10):
            buf.push(make_frame(), datetime.utcnow(), i)
        snap = buf.snapshot()
        # Should contain frame numbers 5-9 (the most recent 5)
        frame_numbers = [f.frame_number for f in snap]
        assert frame_numbers == [5, 6, 7, 8, 9]

    def test_duration_covered(self):
        buf = RollingBuffer(duration_seconds=30, fps=15.0)
        now = datetime.utcnow()
        buf.push(make_frame(), now, 1)
        buf.push(make_frame(), now + timedelta(seconds=10), 2)
        assert buf.duration_covered == pytest.approx(10.0, abs=0.1)
