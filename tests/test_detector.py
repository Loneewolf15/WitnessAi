"""Tests for YOLOv8 Detector"""
import pytest
import numpy as np
from datetime import datetime
from detection.detector import Detector, MockYOLOModel
from models.schemas import FrameDetections, DetectedObject


@pytest.fixture
def mock_detector():
    d = Detector(mock=True, camera_id="cam-test", confidence_threshold=0.5)
    d.load()
    return d


@pytest.fixture
def sample_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestDetector:
    def test_load_mock(self, mock_detector):
        assert mock_detector._model is not None
        assert mock_detector.mock is True

    def test_detect_returns_frame_detections(self, mock_detector, sample_frame):
        result = mock_detector.detect(sample_frame)
        assert isinstance(result, FrameDetections)

    def test_detect_returns_persons(self, mock_detector, sample_frame):
        result = mock_detector.detect(sample_frame)
        assert len(result.objects) == 2
        assert all(o.class_name == "person" for o in result.objects)

    def test_detect_confidence_above_threshold(self, mock_detector, sample_frame):
        result = mock_detector.detect(sample_frame)
        for obj in result.objects:
            assert obj.confidence >= mock_detector.confidence_threshold

    def test_frame_count_increments(self, mock_detector, sample_frame):
        assert mock_detector.frame_count == 0
        mock_detector.detect(sample_frame)
        assert mock_detector.frame_count == 1
        mock_detector.detect(sample_frame)
        assert mock_detector.frame_count == 2

    def test_reset_clears_frame_count(self, mock_detector, sample_frame):
        mock_detector.detect(sample_frame)
        mock_detector.reset()
        assert mock_detector.frame_count == 0

    def test_detect_without_load_raises(self):
        d = Detector(mock=True)
        with pytest.raises(RuntimeError, match="not loaded"):
            d.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_detect_assigns_camera_id(self, mock_detector, sample_frame):
        result = mock_detector.detect(sample_frame)
        for obj in result.objects:
            assert obj.camera_id == "cam-test"

    def test_detect_assigns_minus_one_track_id(self, mock_detector, sample_frame):
        """Track IDs should be -1 before tracker assigns them."""
        result = mock_detector.detect(sample_frame)
        for obj in result.objects:
            assert obj.track_id == -1

    def test_detect_bounding_box_valid(self, mock_detector, sample_frame):
        result = mock_detector.detect(sample_frame)
        for obj in result.objects:
            assert obj.bbox.x2 > obj.bbox.x1
            assert obj.bbox.y2 > obj.bbox.y1
            assert obj.bbox.width > 0
            assert obj.bbox.height > 0

    def test_confidence_threshold_filters(self):
        """High threshold should still return objects above it."""
        d = Detector(mock=True, confidence_threshold=0.99)
        d.load()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = d.detect(frame)
        # Mock confidences are 0.92 and 0.87, both below 0.99
        assert len(result.objects) == 0
