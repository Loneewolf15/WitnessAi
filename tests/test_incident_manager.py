"""
Tests for core.incident_manager
"""
import asyncio
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
import numpy as np
import pytest

from core.incident_manager import IncidentManager
from models.schemas import AnomalyEvent, NarrativeEntry
from models.enums import AnomalyType, IncidentStatus


@pytest.fixture
def manager(tmp_path):
    with patch("core.incident_manager.settings") as mock_settings:
        mock_settings.incidents_dir = tmp_path / "incidents"
        mock_settings.incidents_dir.mkdir(parents=True, exist_ok=True)
        mock_settings.pre_event_seconds = 5
        mock_settings.post_event_seconds = 2
        mock_settings.target_fps = 15
        mock_settings.max_buffer_frames = 450
        yield IncidentManager("test-feed", "Test Camera")


@pytest.fixture
def anomaly(anomaly=None):
    return AnomalyEvent(
        feed_id="test-feed",
        anomaly_type=AnomalyType.LOITERING,
        track_ids=[1],
        frame_number=100,
        confidence=0.87,
        description="Loitering detected.",
    )


@pytest.fixture
def narrative():
    return [
        NarrativeEntry(feed_id="test-feed", text="Track-1 entered the scene."),
        NarrativeEntry(feed_id="test-feed", text="Loitering detected.", is_anomaly=True),
    ]


class TestIncidentManagerBuffer:

    def test_ingest_frame_increases_buffer(self, manager):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert len(manager.buffer) == 0
        manager.ingest_frame(frame, frame_number=0)
        assert len(manager.buffer) == 1

    def test_ingest_multiple_frames(self, manager):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(10):
            manager.ingest_frame(frame, frame_number=i)
        assert len(manager.buffer) == 10

    def test_buffer_respects_maxlen(self):
        with patch("core.incident_manager.settings") as s:
            s.incidents_dir = Path("/tmp/wi_test")
            s.incidents_dir.mkdir(exist_ok=True)
            s.max_buffer_frames = 5
            s.pre_event_seconds = 1
            s.post_event_seconds = 1
            s.target_fps = 5
            mgr = IncidentManager("f1")
            frame = np.zeros((10, 10, 3), dtype=np.uint8)
            for i in range(20):
                mgr.ingest_frame(frame, i)
            assert len(mgr.buffer) == 5


class TestIncidentManagerIncidents:

    def test_no_incidents_initially(self, manager):
        assert manager.get_incident_count() == 0
        assert manager.get_all_incidents() == []

    def test_get_nonexistent_incident(self, manager):
        assert manager.get_incident("nonexistent") is None

    @pytest.mark.asyncio
    async def test_trigger_incident_creates_report(self, manager, tmp_path):
        with patch("core.incident_manager.settings") as s:
            s.incidents_dir = tmp_path / "incidents"
            s.incidents_dir.mkdir(parents=True, exist_ok=True)
            s.pre_event_seconds = 1
            s.post_event_seconds = 1
            s.target_fps = 5
            s.max_buffer_frames = 50

            mgr = IncidentManager("test-feed", "Test Camera")
            # Populate buffer with some frames
            frame = np.zeros((10, 10, 3), dtype=np.uint8)
            for i in range(5):
                mgr.ingest_frame(frame, i)

            anomaly = AnomalyEvent(
                feed_id="test-feed",
                anomaly_type=AnomalyType.LOITERING,
                track_ids=[1],
                frame_number=4,
                confidence=0.87,
                description="Test loitering.",
            )
            narrative = [NarrativeEntry(feed_id="test-feed", text="Track-1 entered.")]

            # Patch finalize to avoid waiting
            with patch.object(mgr, "_finalize_incident", new=AsyncMock()):
                with patch("asyncio.create_task"):
                    incident = await mgr.trigger_incident(anomaly, narrative, "Test summary")

            assert incident.feed_id == "test-feed"
            assert incident.anomaly_type == AnomalyType.LOITERING
            assert incident.status == IncidentStatus.PROCESSING
            assert incident.confidence_score == pytest.approx(0.87)
            assert len(incident.narrative_log) >= 1

    @pytest.mark.asyncio
    async def test_no_duplicate_active_incident(self, manager, tmp_path):
        with patch("core.incident_manager.settings") as s:
            s.incidents_dir = tmp_path / "i2"
            s.incidents_dir.mkdir(parents=True, exist_ok=True)
            s.pre_event_seconds = 1
            s.post_event_seconds = 1
            s.target_fps = 5
            s.max_buffer_frames = 50

            mgr = IncidentManager("f2")
            anomaly = AnomalyEvent(
                feed_id="f2", anomaly_type=AnomalyType.RUNNING,
                track_ids=[1], frame_number=1, confidence=0.9, description="Running."
            )
            with patch.object(mgr, "_finalize_incident", new=AsyncMock()):
                with patch("asyncio.create_task"):
                    inc1 = await mgr.trigger_incident(anomaly, [], "")
            # Second trigger while first is active
            assert mgr._active_incident_id is not None


class TestIncidentManagerPersistence:

    def test_save_json(self, manager, tmp_path):
        from models.schemas import IncidentReport
        from models.enums import IncidentStatus
        inc = IncidentReport(
            feed_id="test-feed",
            feed_name="Test Camera",
            anomaly_type=AnomalyType.FALL,
        )
        path = tmp_path / "test_incident.json"
        IncidentManager._save_json(inc, path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["feed_id"] == "test-feed"

    def test_load_from_json(self, manager, tmp_path):
        from models.schemas import IncidentReport
        inc = IncidentReport(
            feed_id="test-feed",
            anomaly_type=AnomalyType.FALL,
        )
        path = tmp_path / "test_load.json"
        IncidentManager._save_json(inc, path)
        loaded = IncidentManager.load_incident_from_json(path)
        assert loaded is not None
        assert loaded.feed_id == "test-feed"
        assert loaded.anomaly_type == AnomalyType.FALL

    def test_load_nonexistent_json(self):
        result = IncidentManager.load_incident_from_json(Path("/nonexistent/path.json"))
        assert result is None
