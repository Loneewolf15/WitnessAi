"""
Tests for core.anomaly_engine
"""
from datetime import datetime, timedelta
from unittest.mock import patch
import pytest
import numpy as np

from models.schemas import BoundingBox, TrackedObject, FrameResult, Detection
from models.enums import AnomalyType
from core.anomaly_engine import (
    LoiteringDetector,
    RunningDetector,
    CrowdSurgeDetector,
    FallDetector,
    AbandonedObjectDetector,
    AnomalyEngine,
)


def make_frame_result(feed_id="test-feed", frame_number=100, detections=None):
    return FrameResult(
        feed_id=feed_id,
        frame_number=frame_number,
        detections=detections or [],
    )


def make_person(track_id=1, x1=100, y1=50, x2=160, y2=400, age_seconds=0):
    bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    obj = TrackedObject(
        track_id=track_id,
        class_name="person",
        confidence=0.9,
        bbox=bbox,
        frame_history=list(range(10)),
        bbox_history=[bbox],
    )
    if age_seconds > 0:
        obj.first_seen = datetime.utcnow() - timedelta(seconds=age_seconds)
    return obj


def make_bag(track_id=10, age_seconds=0):
    bbox = BoundingBox(x1=200, y1=300, x2=260, y2=360)
    obj = TrackedObject(
        track_id=track_id,
        class_name="backpack",
        confidence=0.85,
        bbox=bbox,
        frame_history=[1],
        bbox_history=[bbox],
    )
    if age_seconds > 0:
        obj.first_seen = datetime.utcnow() - timedelta(seconds=age_seconds)
    return obj


class TestLoiteringDetector:

    def test_no_anomaly_below_threshold(self):
        detector = LoiteringDetector(threshold_seconds=60.0)
        person = make_person(age_seconds=30)
        fr = make_frame_result()
        events = detector.check(fr, {1: person})
        assert events == []

    def test_detects_loitering_above_threshold(self):
        detector = LoiteringDetector(threshold_seconds=60.0)
        person = make_person(age_seconds=65)
        fr = make_frame_result()
        events = detector.check(fr, {1: person})
        assert len(events) == 1
        assert events[0].anomaly_type == AnomalyType.LOITERING
        assert 1 in events[0].track_ids

    def test_no_duplicate_alert_same_track(self):
        detector = LoiteringDetector(threshold_seconds=60.0)
        person = make_person(age_seconds=65)
        fr = make_frame_result()
        events1 = detector.check(fr, {1: person})
        events2 = detector.check(fr, {1: person})
        assert len(events1) == 1
        assert len(events2) == 0  # already alerted

    def test_ignores_non_person(self):
        detector = LoiteringDetector(threshold_seconds=60.0)
        bag = make_bag(age_seconds=65)
        fr = make_frame_result()
        events = detector.check(fr, {10: bag})
        assert events == []


class TestRunningDetector:

    def test_no_anomaly_low_velocity(self):
        detector = RunningDetector(velocity_threshold=80.0)
        bbox1 = BoundingBox(x1=100, y1=100, x2=160, y2=400)
        bbox2 = BoundingBox(x1=110, y1=100, x2=170, y2=400)  # 10px displacement
        person = TrackedObject(
            track_id=1, class_name="person", confidence=0.9,
            bbox=bbox2, frame_history=[0, 1], bbox_history=[bbox1, bbox2]
        )
        fr = make_frame_result()
        events = detector.check(fr, {1: person})
        assert events == []

    def test_detects_high_velocity(self):
        detector = RunningDetector(velocity_threshold=80.0)
        bbox1 = BoundingBox(x1=100, y1=100, x2=160, y2=400)
        bbox2 = BoundingBox(x1=200, y1=100, x2=260, y2=400)  # 100px displacement
        person = TrackedObject(
            track_id=1, class_name="person", confidence=0.9,
            bbox=bbox2, frame_history=[0, 1], bbox_history=[bbox1, bbox2]
        )
        fr = make_frame_result(frame_number=100)
        events = detector.check(fr, {1: person})
        assert len(events) == 1
        assert events[0].anomaly_type == AnomalyType.RUNNING

    def test_velocity_with_single_bbox_history(self):
        """No velocity can be computed with < 2 bbox history."""
        detector = RunningDetector(velocity_threshold=80.0)
        bbox = BoundingBox(x1=100, y1=100, x2=160, y2=400)
        person = TrackedObject(
            track_id=1, class_name="person", confidence=0.9,
            bbox=bbox, frame_history=[0], bbox_history=[bbox]
        )
        fr = make_frame_result()
        events = detector.check(fr, {1: person})
        assert events == []


class TestCrowdSurgeDetector:

    def test_no_anomaly_gradual_increase(self):
        detector = CrowdSurgeDetector(surge_count=5)
        persons = {i: make_person(track_id=i) for i in range(3)}
        fr = make_frame_result(frame_number=0)
        events = detector.check(fr, persons)
        assert events == []

    def test_detects_sudden_crowd(self):
        detector = CrowdSurgeDetector(surge_count=5)
        # First call: 0 people
        detector.check(make_frame_result(frame_number=0), {})
        # Second call: 8 people suddenly
        persons = {i: make_person(track_id=i) for i in range(8)}
        fr = make_frame_result(frame_number=1)
        events = detector.check(fr, persons)
        assert len(events) == 1
        assert events[0].anomaly_type == AnomalyType.CROWD_SURGE

    def test_no_duplicate_crowd_alert_within_cooldown(self):
        detector = CrowdSurgeDetector(surge_count=5)
        detector.check(make_frame_result(frame_number=0), {})
        persons = {i: make_person(track_id=i) for i in range(8)}
        events1 = detector.check(make_frame_result(frame_number=1), persons)
        events2 = detector.check(make_frame_result(frame_number=2), persons)
        assert len(events1) == 1
        assert len(events2) == 0


class TestFallDetector:

    def test_no_anomaly_standing_person(self):
        detector = FallDetector(aspect_ratio_threshold=1.5)
        # width=60, height=350 → aspect=0.17 (standing)
        person = make_person(x1=100, y1=50, x2=160, y2=400)
        fr = make_frame_result()
        events = detector.check(fr, {1: person})
        assert events == []

    def test_detects_fallen_person(self):
        detector = FallDetector(aspect_ratio_threshold=1.5)
        # width=400, height=60 → aspect=6.67 (fallen)
        bbox = BoundingBox(x1=50, y1=300, x2=450, y2=360)
        person = TrackedObject(
            track_id=1, class_name="person", confidence=0.9,
            bbox=bbox, frame_history=[0], bbox_history=[bbox]
        )
        fr = make_frame_result(frame_number=100)
        events = detector.check(fr, {1: person})
        assert len(events) == 1
        assert events[0].anomaly_type == AnomalyType.FALL

    def test_zero_height_bbox_no_crash(self):
        detector = FallDetector()
        bbox = BoundingBox(x1=100, y1=200, x2=200, y2=200)  # height=0
        person = TrackedObject(
            track_id=1, class_name="person", confidence=0.9,
            bbox=bbox, frame_history=[0], bbox_history=[bbox]
        )
        fr = make_frame_result()
        events = detector.check(fr, {1: person})
        assert events == []


class TestAbandonedObjectDetector:

    def test_no_anomaly_object_with_nearby_person(self):
        detector = AbandonedObjectDetector(stillness_seconds=30.0)
        bag = make_bag(age_seconds=35)
        # Person right next to the bag
        person_bbox = BoundingBox(x1=210, y1=290, x2=270, y2=380)
        person = TrackedObject(
            track_id=1, class_name="person", confidence=0.9,
            bbox=person_bbox, frame_history=[0], bbox_history=[person_bbox]
        )
        fr = make_frame_result()
        events = detector.check(fr, {10: bag, 1: person})
        assert events == []

    def test_detects_abandoned_object_no_person(self):
        detector = AbandonedObjectDetector(stillness_seconds=30.0)
        bag = make_bag(age_seconds=35)
        fr = make_frame_result()
        events = detector.check(fr, {10: bag})
        assert len(events) == 1
        assert events[0].anomaly_type == AnomalyType.ABANDONED_OBJECT

    def test_no_anomaly_object_too_new(self):
        detector = AbandonedObjectDetector(stillness_seconds=30.0)
        bag = make_bag(age_seconds=5)
        fr = make_frame_result()
        events = detector.check(fr, {10: bag})
        assert events == []


class TestAnomalyEngine:

    def test_engine_runs_all_detectors(self):
        engine = AnomalyEngine()
        assert len(engine.detectors) == 5

    def test_engine_returns_list(self, sample_frame_result, sample_tracked_object):
        engine = AnomalyEngine()
        events = engine.analyze(sample_frame_result, {1: sample_tracked_object})
        assert isinstance(events, list)

    def test_engine_handles_detector_exception_gracefully(self, sample_frame_result):
        engine = AnomalyEngine()
        broken_detector = LoiteringDetector()

        def raise_exc(*args):
            raise RuntimeError("Detector exploded!")

        broken_detector.check = raise_exc
        engine.detectors = [broken_detector]
        # Should not raise
        events = engine.analyze(sample_frame_result, {})
        assert events == []

    def test_engine_detects_loitering_end_to_end(self):
        engine = AnomalyEngine()
        # Replace with a fresh loitering detector with low threshold
        from core.anomaly_engine import LoiteringDetector
        engine.detectors = [LoiteringDetector(threshold_seconds=60.0)]
        person = make_person(age_seconds=65)
        fr = make_frame_result()
        events = engine.analyze(fr, {1: person})
        assert len(events) == 1
        assert events[0].anomaly_type == AnomalyType.LOITERING
