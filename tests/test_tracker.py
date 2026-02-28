"""Tests for Multi-Object Tracker"""
import pytest
import numpy as np
from datetime import datetime, timedelta
from detection.tracker import Tracker, Track, iou
from detection.detector import Detector
from models.schemas import FrameDetections, DetectedObject, BoundingBox


def make_frame_detections(camera_id="cam1", objects=None):
    return FrameDetections(
        camera_id=camera_id,
        frame_number=1,
        timestamp=datetime.utcnow(),
        objects=objects or [],
    )


def make_object(x1=10, y1=10, x2=100, y2=200, cls="person", conf=0.9):
    return DetectedObject(
        track_id=-1,
        class_name=cls,
        confidence=conf,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        camera_id="cam1",
    )


class TestIoU:
    def test_perfect_overlap(self):
        box = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        assert iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = BoundingBox(x1=0, y1=0, x2=50, y2=50)
        b = BoundingBox(x1=100, y1=100, x2=200, y2=200)
        assert iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        b = BoundingBox(x1=50, y1=50, x2=150, y2=150)
        result = iou(a, b)
        assert 0 < result < 1

    def test_contained_box(self):
        outer = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        inner = BoundingBox(x1=25, y1=25, x2=75, y2=75)
        result = iou(outer, inner)
        assert result > 0


class TestTracker:
    def test_new_detection_creates_track(self):
        tracker = Tracker()
        obj = make_object()
        frame = make_frame_detections(objects=[obj])
        result = tracker.update(frame)
        assert len(tracker.active_tracks) == 1
        assert len(result.objects) == 1

    def test_track_id_assigned(self):
        tracker = Tracker()
        obj = make_object()
        frame = make_frame_detections(objects=[obj])
        result = tracker.update(frame)
        assert result.objects[0].track_id >= 1

    def test_same_object_keeps_track_id(self):
        tracker = Tracker()
        obj = make_object(x1=10, y1=10, x2=100, y2=200)
        f1 = make_frame_detections(objects=[obj])
        r1 = tracker.update(f1)
        tid1 = r1.objects[0].track_id

        # Move slightly
        obj2 = make_object(x1=12, y1=12, x2=102, y2=202)
        f2 = make_frame_detections(objects=[obj2])
        r2 = tracker.update(f2)
        tid2 = r2.objects[0].track_id

        assert tid1 == tid2

    def test_multiple_objects_get_different_ids(self):
        tracker = Tracker()
        objs = [
            make_object(x1=0, y1=0, x2=50, y2=100),
            make_object(x1=300, y1=0, x2=400, y2=100),
        ]
        result = tracker.update(make_frame_detections(objects=objs))
        ids = [o.track_id for o in result.objects]
        assert len(set(ids)) == 2

    def test_missing_object_increments_misses(self):
        tracker = Tracker()
        obj = make_object()
        tracker.update(make_frame_detections(objects=[obj]))
        # Next frame: no detections
        tracker.update(make_frame_detections(objects=[]))
        track = list(tracker.active_tracks.values())[0]
        assert track.misses == 1

    def test_stale_track_removed(self):
        tracker = Tracker()
        tracker.MAX_MISSES = 2
        obj = make_object()
        tracker.update(make_frame_detections(objects=[obj]))
        for _ in range(3):
            tracker.update(make_frame_detections(objects=[]))
        assert len(tracker.active_tracks) == 0

    def test_reset_clears_tracks(self):
        tracker = Tracker()
        tracker.update(make_frame_detections(objects=[make_object()]))
        assert len(tracker.active_tracks) == 1
        tracker.reset()
        assert len(tracker.active_tracks) == 0

    def test_velocity_computed(self):
        tracker = Tracker()
        obj1 = make_object(x1=0, y1=0, x2=100, y2=200)
        r1 = tracker.update(make_frame_detections(objects=[obj1]))
        tid = r1.objects[0].track_id

        obj2 = make_object(x1=20, y1=0, x2=120, y2=200)
        import time; time.sleep(0.05)
        r2 = tracker.update(make_frame_detections(objects=[obj2]))
        track = tracker.get_track(tid)
        assert track is not None
        assert track.velocity_x != 0
