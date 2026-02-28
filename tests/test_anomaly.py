"""Tests for Anomaly Detection Rules Engine"""
import pytest
from datetime import datetime
from detection.anomaly import AnomalyDetector, AnomalyConfig
from detection.tracker import Tracker
from models.schemas import (
    AnomalyType, ConfidenceLevel, FrameDetections, DetectedObject, BoundingBox
)


def make_tracked_obj(track_id=1, x1=10, y1=10, x2=60, y2=160, speed=0.0):
    vx = speed / (2 ** 0.5)
    return DetectedObject(
        track_id=track_id,
        class_name="person",
        confidence=0.9,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        camera_id="cam1",
        velocity_x=vx,
        velocity_y=vx,
    )


def make_frame(objects):
    return FrameDetections(
        camera_id="cam1",
        frame_number=1,
        timestamp=datetime.utcnow(),
        objects=objects,
    )


class TestLoiteringDetection:
    def test_loitering_triggers_above_threshold(self):
        config = AnomalyConfig(loitering_threshold=5)
        detector = AnomalyDetector(camera_id="cam1", config=config)
        tracker = Tracker()

        obj = make_tracked_obj(track_id=1)
        frame = make_frame([obj])
        tracker.update(frame)

        # Manually force stationary frames
        tracker._tracks[1].stationary_frames = 90  # 90 / 15fps = 6s > threshold 5s

        events = detector.evaluate(frame, tracker)
        loitering = [e for e in events if e.anomaly_type == AnomalyType.LOITERING]
        assert len(loitering) == 1

    def test_loitering_not_triggered_below_threshold(self):
        config = AnomalyConfig(loitering_threshold=30)
        detector = AnomalyDetector(camera_id="cam1", config=config)
        tracker = Tracker()

        obj = make_tracked_obj(track_id=1)
        frame = make_frame([obj])
        tracker.update(frame)
        tracker._tracks[1].stationary_frames = 5  # Only 0.3 seconds

        events = detector.evaluate(frame, tracker)
        assert not any(e.anomaly_type == AnomalyType.LOITERING for e in events)

    def test_loitering_deduplication(self):
        """Same track should not fire loitering twice."""
        config = AnomalyConfig(loitering_threshold=5)
        detector = AnomalyDetector(camera_id="cam1", config=config)
        tracker = Tracker()

        obj = make_tracked_obj(track_id=1)
        frame = make_frame([obj])
        tracker.update(frame)
        tracker._tracks[1].stationary_frames = 90

        events1 = detector.evaluate(frame, tracker)
        events2 = detector.evaluate(frame, tracker)

        loitering1 = [e for e in events1 if e.anomaly_type == AnomalyType.LOITERING]
        loitering2 = [e for e in events2 if e.anomaly_type == AnomalyType.LOITERING]
        assert len(loitering1) == 1
        assert len(loitering2) == 0  # Deduplicated


class TestRunningDetection:
    def test_running_triggers_above_threshold(self):
        config = AnomalyConfig(running_velocity_threshold=100.0)
        detector = AnomalyDetector(camera_id="cam1", config=config)
        tracker = Tracker()

        obj = make_tracked_obj(track_id=2, speed=200.0)
        frame = make_frame([obj])
        tracker.update(frame)

        events = detector.evaluate(frame, tracker)
        running = [e for e in events if e.anomaly_type == AnomalyType.RUNNING]
        assert len(running) == 1

    def test_running_not_triggered_below_threshold(self):
        config = AnomalyConfig(running_velocity_threshold=200.0)
        detector = AnomalyDetector(camera_id="cam1", config=config)
        tracker = Tracker()

        obj = make_tracked_obj(track_id=2, speed=50.0)
        frame = make_frame([obj])
        tracker.update(frame)

        events = detector.evaluate(frame, tracker)
        assert not any(e.anomaly_type == AnomalyType.RUNNING for e in events)


class TestCrowdSurgeDetection:
    def test_crowd_surge_triggers_above_threshold(self):
        config = AnomalyConfig(crowd_density_threshold=3)
        detector = AnomalyDetector(camera_id="cam1", config=config)
        tracker = Tracker()

        objects = [make_tracked_obj(track_id=i) for i in range(5)]
        frame = make_frame(objects)
        tracker.update(frame)

        events = detector.evaluate(frame, tracker)
        surge = [e for e in events if e.anomaly_type == AnomalyType.CROWD_SURGE]
        assert len(surge) == 1
        assert events[0].metadata["person_count"] == 5

    def test_crowd_surge_not_triggered_below_threshold(self):
        config = AnomalyConfig(crowd_density_threshold=10)
        detector = AnomalyDetector(camera_id="cam1", config=config)
        tracker = Tracker()

        objects = [make_tracked_obj(track_id=i) for i in range(3)]
        frame = make_frame(objects)
        tracker.update(frame)

        events = detector.evaluate(frame, tracker)
        assert not any(e.anomaly_type == AnomalyType.CROWD_SURGE for e in events)


class TestFallDetection:
    def test_fall_triggers_when_prone(self):
        """Person lying down has wide bbox (width >> height)."""
        config = AnomalyConfig(fall_aspect_ratio_threshold=2.0)
        detector = AnomalyDetector(camera_id="cam1", config=config)
        tracker = Tracker()

        # Wide bounding box: width=200, height=50 → aspect=4.0
        prone_obj = DetectedObject(
            track_id=10,
            class_name="person",
            confidence=0.9,
            bbox=BoundingBox(x1=0, y1=100, x2=200, y2=150),
            camera_id="cam1",
        )
        frame = make_frame([prone_obj])
        tracker.update(frame)

        events = detector.evaluate(frame, tracker)
        falls = [e for e in events if e.anomaly_type == AnomalyType.FALL_DETECTED]
        assert len(falls) == 1

    def test_fall_not_triggered_for_standing(self):
        config = AnomalyConfig(fall_aspect_ratio_threshold=2.0)
        detector = AnomalyDetector(camera_id="cam1", config=config)
        tracker = Tracker()

        # Tall bbox: width=50, height=200 → aspect=0.25
        standing = make_tracked_obj(x1=0, y1=0, x2=50, y2=200)
        frame = make_frame([standing])
        tracker.update(frame)

        events = detector.evaluate(frame, tracker)
        assert not any(e.anomaly_type == AnomalyType.FALL_DETECTED for e in events)


class TestAnomalyEventProperties:
    def test_event_has_uuid_id(self):
        config = AnomalyConfig(crowd_density_threshold=1)
        detector = AnomalyDetector(camera_id="cam1", config=config)
        tracker = Tracker()
        frame = make_frame([make_tracked_obj()])
        tracker.update(frame)
        events = detector.evaluate(frame, tracker)
        assert len(events) > 0
        import uuid
        for e in events:
            uuid.UUID(e.id)  # Should not raise

    def test_reset_alerts_clears_state(self):
        config = AnomalyConfig(crowd_density_threshold=1)
        detector = AnomalyDetector(camera_id="cam1", config=config)
        tracker = Tracker()
        frame = make_frame([make_tracked_obj()])
        tracker.update(frame)
        detector.evaluate(frame, tracker)
        detector.reset_alerts()
        events = detector.evaluate(frame, tracker)
        assert len(events) > 0  # Should fire again after reset
