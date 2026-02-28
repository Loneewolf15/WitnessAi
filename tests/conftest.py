"""
WitnessAI — Test Fixtures
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# ── Path Setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Async Support ─────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()


# ── Frame Fixtures ────────────────────────────────────────────────────────────
@pytest.fixture
def blank_frame():
    """480x640 black frame (BGR)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def white_frame():
    return np.ones((480, 640, 3), dtype=np.uint8) * 255


@pytest.fixture
def person_frame():
    """Frame with a white rectangle simulating a person."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:400, 280:360] = 255  # tall rectangle = upright person
    return frame


@pytest.fixture
def fallen_person_frame():
    """Frame with a wide rectangle simulating a fallen person."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[300:360, 100:500] = 255  # wide rectangle = fallen person
    return frame


# ── Schema Fixtures ───────────────────────────────────────────────────────────
@pytest.fixture
def sample_bbox():
    from models.schemas import BoundingBox
    return BoundingBox(x1=100.0, y1=50.0, x2=200.0, y2=400.0)


@pytest.fixture
def wide_bbox():
    """Bounding box wider than tall — simulates a fall."""
    from models.schemas import BoundingBox
    return BoundingBox(x1=50.0, y1=300.0, x2=500.0, y2=360.0)


@pytest.fixture
def sample_detection(sample_bbox):
    from models.schemas import Detection
    return Detection(
        track_id=1,
        class_name="person",
        confidence=0.92,
        bbox=sample_bbox,
        frame_number=42,
    )


@pytest.fixture
def sample_tracked_object(sample_bbox):
    from models.schemas import TrackedObject
    return TrackedObject(
        track_id=1,
        class_name="person",
        confidence=0.92,
        bbox=sample_bbox,
        frame_history=[1, 2, 3],
        bbox_history=[sample_bbox],
    )


@pytest.fixture
def loitering_tracked_object(sample_bbox):
    """Tracked object that has been in scene for >65 seconds."""
    from models.schemas import TrackedObject
    old_time = datetime.utcnow() - timedelta(seconds=65)
    obj = TrackedObject(
        track_id=2,
        class_name="person",
        confidence=0.88,
        bbox=sample_bbox,
        frame_history=list(range(975)),
        bbox_history=[sample_bbox],
    )
    obj.first_seen = old_time
    return obj


@pytest.fixture
def sample_frame_result(sample_detection):
    from models.schemas import FrameResult
    return FrameResult(
        feed_id="test-feed",
        frame_number=42,
        detections=[sample_detection],
        scene_description="A person is visible in the scene.",
    )


@pytest.fixture
def sample_anomaly_event():
    from models.schemas import AnomalyEvent
    from models.enums import AnomalyType
    return AnomalyEvent(
        feed_id="test-feed",
        anomaly_type=AnomalyType.LOITERING,
        track_ids=[1],
        frame_number=42,
        confidence=0.85,
        description="Track-1 has been loitering for 65 seconds.",
    )


@pytest.fixture
def sample_narrative_entries():
    from models.schemas import NarrativeEntry
    return [
        NarrativeEntry(
            feed_id="test-feed",
            text="Track-1 entered the scene.",
            timestamp=datetime.utcnow() - timedelta(seconds=30),
        ),
        NarrativeEntry(
            feed_id="test-feed",
            text="Track-1 has been loitering for 65 seconds.",
            is_anomaly=True,
            timestamp=datetime.utcnow(),
        ),
    ]


# ── Settings Override for Tests ───────────────────────────────────────────────
@pytest.fixture(autouse=True)
def mock_settings(tmp_path):
    """Override incidents_dir to use a temp directory in all tests."""
    with patch("config.settings") as mock_cfg:
        from config import Config
        real_cfg = Config()
        real_cfg.incidents_dir = tmp_path / "incidents"
        real_cfg.incidents_dir.mkdir(parents=True, exist_ok=True)
        real_cfg.llm_mock_mode = True
        real_cfg.moondream_mock_mode = True
        # Copy all attributes
        for attr in vars(real_cfg):
            setattr(mock_cfg, attr, getattr(real_cfg, attr))
        yield mock_cfg
