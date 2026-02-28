"""
Tests for core.narrative_engine
"""
import asyncio
import pytest
from datetime import datetime

from core.narrative_engine import NarrativeEngine
from models.schemas import FrameResult, Detection, BoundingBox, TrackedObject, AnomalyEvent, NarrativeEntry
from models.enums import AnomalyType


def make_detection(track_id=1, class_name="person"):
    bbox = BoundingBox(x1=100, y1=50, x2=200, y2=400)
    return Detection(track_id=track_id, class_name=class_name, confidence=0.9,
                     bbox=bbox, frame_number=1)


def make_tracked(track_id=1, class_name="person", n_frames=1):
    bbox = BoundingBox(x1=100, y1=50, x2=200, y2=400)
    return TrackedObject(
        track_id=track_id, class_name=class_name, confidence=0.9,
        bbox=bbox, frame_history=list(range(n_frames)), bbox_history=[bbox]
    )


def make_frame_result(feed_id="feed-1", frame_number=0, detections=None, scene_desc=None):
    return FrameResult(
        feed_id=feed_id,
        frame_number=frame_number,
        detections=detections or [],
        scene_description=scene_desc,
    )


class TestNarrativeEngine:

    def test_log_frame_returns_entries(self, sample_anomaly_event):
        engine = NarrativeEngine()
        det = make_detection()
        tracked = {1: make_tracked(n_frames=1)}  # first appearance
        fr = make_frame_result(detections=[det])
        entries = engine.log_frame(fr, tracked, [])
        assert isinstance(entries, list)

    def test_scene_description_logged(self):
        engine = NarrativeEngine()
        fr = make_frame_result(scene_desc="A person is walking.")
        entries = engine.log_frame(fr, {}, [])
        texts = [e.text for e in entries]
        assert "A person is walking." in texts

    def test_first_appearance_logged(self):
        engine = NarrativeEngine()
        det = make_detection(track_id=5)
        tracked = {5: make_tracked(track_id=5, n_frames=1)}
        fr = make_frame_result(detections=[det])
        entries = engine.log_frame(fr, tracked, [])
        assert any("Track-5" in e.text for e in entries)

    def test_subsequent_frame_no_entry_event(self):
        """Second+ frame for same track should not create another 'entered' log."""
        engine = NarrativeEngine()
        det = make_detection(track_id=5)
        tracked = {5: make_tracked(track_id=5, n_frames=10)}
        fr = make_frame_result(detections=[det])
        entries = engine.log_frame(fr, tracked, [])
        assert not any("entered frame" in e.text and "Track-5" in e.text for e in entries)

    def test_anomaly_event_logged_as_anomaly(self, sample_anomaly_event):
        engine = NarrativeEngine()
        fr = make_frame_result()
        entries = engine.log_frame(fr, {}, [sample_anomaly_event])
        anomaly_entries = [e for e in entries if e.is_anomaly]
        assert len(anomaly_entries) == 1
        assert anomaly_entries[0].anomaly_type == AnomalyType.LOITERING

    def test_get_log_returns_all(self, sample_anomaly_event):
        engine = NarrativeEngine()
        fr = make_frame_result()
        engine.log_frame(fr, {}, [sample_anomaly_event])
        engine.log_frame(fr, {}, [sample_anomaly_event])
        log = engine.get_log("feed-1")
        assert len(log) >= 2

    def test_get_recent_log_limits_entries(self, sample_anomaly_event):
        engine = NarrativeEngine()
        fr = make_frame_result()
        for _ in range(30):
            engine.log_frame(fr, {}, [sample_anomaly_event])
        recent = engine.get_recent_log("feed-1", last_n=10)
        assert len(recent) == 10

    def test_clear_log(self, sample_anomaly_event):
        engine = NarrativeEngine()
        fr = make_frame_result()
        engine.log_frame(fr, {}, [sample_anomaly_event])
        engine.clear_log("feed-1")
        assert engine.get_log("feed-1") == []

    def test_multiple_feeds_isolated(self, sample_anomaly_event):
        engine = NarrativeEngine()
        fr_a = make_frame_result(feed_id="feed-A")
        fr_b = make_frame_result(feed_id="feed-B")
        engine.log_frame(fr_a, {}, [])
        engine.log_frame(fr_b, {}, [sample_anomaly_event])
        log_a = engine.get_log("feed-A")
        log_b = engine.get_log("feed-B")
        assert log_a is not log_b

    @pytest.mark.asyncio
    async def test_mock_summary_returns_string(self, sample_anomaly_event):
        engine = NarrativeEngine()
        summary = await engine.generate_incident_summary(
            "feed-1", sample_anomaly_event
        )
        assert isinstance(summary, str)
        assert len(summary) > 10

    @pytest.mark.asyncio
    async def test_mock_summary_contains_anomaly_type(self, sample_anomaly_event):
        engine = NarrativeEngine()
        summary = await engine.generate_incident_summary("feed-1", sample_anomaly_event)
        assert "loitering" in summary.lower()

    def test_build_context_empty(self):
        engine = NarrativeEngine()
        result = engine._build_context([])
        assert "No prior" in result

    def test_build_context_with_entries(self, sample_narrative_entries):
        engine = NarrativeEngine()
        result = engine._build_context(sample_narrative_entries)
        assert "Track-1" in result
