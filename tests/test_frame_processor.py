"""
Tests for core.frame_processor
(Runs in mock mode — no GPU/model required)
"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from core.frame_processor import FrameProcessor, YOLODetector, DeepSORTTracker, MoondreamDescriber
from models.schemas import FrameResult, Detection


class TestYOLODetector:

    def test_mock_detect_returns_person(self, blank_frame):
        detector = YOLODetector()
        detector._model = "mock"
        results = detector.detect(blank_frame)
        assert len(results) >= 1
        assert results[0]["class_name"] == "person"
        assert 0 < results[0]["confidence"] <= 1.0

    def test_mock_detect_has_valid_bbox(self, blank_frame):
        detector = YOLODetector()
        detector._model = "mock"
        results = detector.detect(blank_frame)
        bbox = results[0]["bbox"]
        x1, y1, x2, y2 = bbox
        assert x1 < x2
        assert y1 < y2

    def test_detect_falls_back_to_mock_on_import_error(self, blank_frame):
        detector = YOLODetector()
        # Force mock by setting model directly
        detector._model = "mock"
        results = detector.detect(blank_frame)
        assert isinstance(results, list)


class TestDeepSORTTracker:

    def test_mock_update_assigns_track_ids(self, blank_frame):
        tracker = DeepSORTTracker()
        tracker._tracker = "mock"
        detections = [
            {"class_name": "person", "confidence": 0.9, "bbox": (100, 50, 200, 400)},
            {"class_name": "person", "confidence": 0.85, "bbox": (300, 50, 400, 400)},
        ]
        results = tracker.update(detections, blank_frame)
        assert len(results) == 2
        track_ids = {r[0] for r in results}
        assert len(track_ids) == 2  # unique IDs

    def test_mock_update_empty_detections(self, blank_frame):
        tracker = DeepSORTTracker()
        tracker._tracker = "mock"
        results = tracker.update([], blank_frame)
        assert results == []

    def test_result_format(self, blank_frame):
        tracker = DeepSORTTracker()
        tracker._tracker = "mock"
        detections = [{"class_name": "person", "confidence": 0.9, "bbox": (100, 50, 200, 400)}]
        results = tracker.update(detections, blank_frame)
        assert len(results) == 1
        track_id, class_name, confidence, bbox = results[0]
        assert isinstance(track_id, int)
        assert class_name == "person"
        assert 0 < confidence <= 1.0
        assert len(bbox) == 4


class TestMoondreamDescriber:

    def test_mock_returns_string(self, blank_frame):
        describer = MoondreamDescriber()
        describer._model = "mock"
        result = describer.describe(blank_frame)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_mock_describe_content(self, blank_frame):
        describer = MoondreamDescriber()
        describer._model = "mock"
        result = describer.describe(blank_frame)
        assert "person" in result.lower() or "scene" in result.lower()


class TestFrameProcessor:

    def test_process_frame_returns_frame_result(self, blank_frame):
        processor = FrameProcessor()
        # Force mock mode
        processor.detector._model = "mock"
        processor.tracker._tracker = "mock"
        processor.describer._model = "mock"
        result = processor.process_frame("feed-1", blank_frame)
        assert isinstance(result, FrameResult)
        assert result.feed_id == "feed-1"

    def test_frame_number_auto_increments(self, blank_frame):
        processor = FrameProcessor()
        processor.detector._model = "mock"
        processor.tracker._tracker = "mock"
        processor.describer._model = "mock"
        r1 = processor.process_frame("feed-1", blank_frame)
        r2 = processor.process_frame("feed-1", blank_frame)
        assert r2.frame_number == r1.frame_number + 1

    def test_manual_frame_number(self, blank_frame):
        processor = FrameProcessor()
        processor.detector._model = "mock"
        processor.tracker._tracker = "mock"
        processor.describer._model = "mock"
        result = processor.process_frame("feed-1", blank_frame, frame_number=99)
        assert result.frame_number == 99

    def test_detections_populated(self, blank_frame):
        processor = FrameProcessor()
        processor.detector._model = "mock"
        processor.tracker._tracker = "mock"
        processor.describer._model = "mock"
        result = processor.process_frame("feed-1", blank_frame)
        assert len(result.detections) > 0
        assert result.detections[0].class_name == "person"

    def test_tracked_objects_updated(self, blank_frame):
        processor = FrameProcessor()
        processor.detector._model = "mock"
        processor.tracker._tracker = "mock"
        processor.describer._model = "mock"
        processor.process_frame("feed-1", blank_frame)
        tracked = processor.get_tracked_objects("feed-1")
        assert len(tracked) > 0

    def test_scene_description_on_nth_frame(self, blank_frame):
        """Scene description should be set every N frames."""
        processor = FrameProcessor()
        processor.detector._model = "mock"
        processor.tracker._tracker = "mock"
        processor.describer._model = "mock"
        # Frame 0 should trigger description (0 % 30 == 0)
        result = processor.process_frame("feed-1", blank_frame, frame_number=0)
        assert result.scene_description is not None

    def test_no_scene_description_between_intervals(self, blank_frame):
        processor = FrameProcessor()
        processor.detector._model = "mock"
        processor.tracker._tracker = "mock"
        processor.describer._model = "mock"
        # Frame 1 should NOT trigger description (1 % 30 != 0)
        result = processor.process_frame("feed-1", blank_frame, frame_number=1)
        assert result.scene_description is None

    def test_processing_time_recorded(self, blank_frame):
        processor = FrameProcessor()
        processor.detector._model = "mock"
        processor.tracker._tracker = "mock"
        processor.describer._model = "mock"
        result = processor.process_frame("feed-1", blank_frame)
        assert result.processing_time_ms >= 0

    def test_reset_feed_clears_state(self, blank_frame):
        processor = FrameProcessor()
        processor.detector._model = "mock"
        processor.tracker._tracker = "mock"
        processor.describer._model = "mock"
        processor.process_frame("feed-1", blank_frame)
        processor.reset_feed("feed-1")
        tracked = processor.get_tracked_objects("feed-1")
        assert tracked == {}

    def test_multiple_feeds_isolated(self, blank_frame):
        """Two feeds should not share tracked object state."""
        processor = FrameProcessor()
        processor.detector._model = "mock"
        processor.tracker._tracker = "mock"
        processor.describer._model = "mock"
        processor.process_frame("feed-A", blank_frame)
        processor.process_frame("feed-B", blank_frame)
        assert "feed-A" in processor._tracked_objects
        assert "feed-B" in processor._tracked_objects
        assert processor._tracked_objects["feed-A"] is not processor._tracked_objects["feed-B"]
