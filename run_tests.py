#!/usr/bin/env python3
"""
WitnessAI — Self-Contained Test Runner
Uses Python's built-in unittest — no external dependencies needed.
Run: python run_tests.py
"""
import sys
import os
import unittest
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import numpy as np
import tempfile
import json

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── Colour output ─────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def run_async(coro):
    """Run a coroutine in the current event loop."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ═══════════════════════════════════════════════════════════════════════════════
# BUFFER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFrameRingBuffer(unittest.TestCase):

    def setUp(self):
        from utils.buffer import FrameRingBuffer
        self.FrameRingBuffer = FrameRingBuffer
        self.blank = np.zeros((48, 64, 3), dtype=np.uint8)

    def test_push_increases_len(self):
        buf = self.FrameRingBuffer(maxlen=10)
        self.assertEqual(len(buf), 0)
        buf.push(0, datetime.utcnow(), self.blank)
        self.assertEqual(len(buf), 1)

    def test_maxlen_evicts_old_frames(self):
        buf = self.FrameRingBuffer(maxlen=5)
        for i in range(10):
            buf.push(i, datetime.utcnow(), self.blank)
        self.assertEqual(len(buf), 5)

    def test_get_all_order(self):
        buf = self.FrameRingBuffer(maxlen=10)
        for i in range(5):
            buf.push(i, datetime.utcnow(), self.blank)
        frames = buf.get_all()
        self.assertEqual([f[0] for f in frames], [0, 1, 2, 3, 4])

    def test_is_empty(self):
        buf = self.FrameRingBuffer(maxlen=10)
        self.assertTrue(buf.is_empty)
        buf.push(0, datetime.utcnow(), self.blank)
        self.assertFalse(buf.is_empty)

    def test_clear(self):
        buf = self.FrameRingBuffer(maxlen=10)
        buf.push(0, datetime.utcnow(), self.blank)
        buf.clear()
        self.assertTrue(buf.is_empty)

    def test_latest_frame_number(self):
        buf = self.FrameRingBuffer(maxlen=10)
        self.assertIsNone(buf.latest_frame_number())
        buf.push(7, datetime.utcnow(), self.blank)
        buf.push(8, datetime.utcnow(), self.blank)
        self.assertEqual(buf.latest_frame_number(), 8)

    def test_get_last_n_seconds(self):
        buf = self.FrameRingBuffer(maxlen=100)
        old_ts = datetime.utcnow() - timedelta(seconds=60)
        recent_ts = datetime.utcnow() - timedelta(seconds=5)
        buf.push(1, old_ts, self.blank)
        buf.push(2, recent_ts, self.blank)
        buf.push(3, datetime.utcnow(), self.blank)
        result = buf.get_last_n_seconds(seconds=10)
        frame_nums = [f[0] for f in result]
        self.assertIn(2, frame_nums)
        self.assertIn(3, frame_nums)
        self.assertNotIn(1, frame_nums)

    def test_get_last_n_seconds_empty(self):
        buf = self.FrameRingBuffer(maxlen=10)
        self.assertEqual(buf.get_last_n_seconds(30), [])

    def test_thread_safety(self):
        import threading
        buf = self.FrameRingBuffer(maxlen=1000)
        errors = []
        def pusher(start_i):
            try:
                for i in range(50):
                    buf.push(start_i + i, datetime.utcnow(), self.blank)
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=pusher, args=(i * 50,)) for i in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(errors, [])
        self.assertLessEqual(len(buf), 1000)

    def test_frame_data_preserved(self):
        buf = self.FrameRingBuffer(maxlen=10)
        frame = np.full((10, 10, 3), 42, dtype=np.uint8)
        buf.push(0, datetime.utcnow(), frame.copy())
        retrieved = buf.get_all()[0][2]
        self.assertTrue(np.all(retrieved == 42))


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchemas(unittest.TestCase):

    def test_bounding_box_properties(self):
        from models.schemas import BoundingBox
        bbox = BoundingBox(x1=100, y1=50, x2=200, y2=300)
        self.assertAlmostEqual(bbox.width, 100.0)
        self.assertAlmostEqual(bbox.height, 250.0)
        self.assertAlmostEqual(bbox.center_x, 150.0)
        self.assertAlmostEqual(bbox.center_y, 175.0)
        self.assertAlmostEqual(bbox.area, 25000.0)

    def test_tracked_object_duration(self):
        from models.schemas import TrackedObject, BoundingBox
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=200)
        obj = TrackedObject(track_id=1, class_name="person", confidence=0.9,
                            bbox=bbox, frame_history=[0], bbox_history=[bbox])
        obj.first_seen = datetime.utcnow() - timedelta(seconds=30)
        obj.last_seen = datetime.utcnow()
        self.assertGreaterEqual(obj.duration_seconds(), 29.0)

    def test_tracked_object_velocity_single_bbox(self):
        from models.schemas import TrackedObject, BoundingBox
        bbox = BoundingBox(x1=100, y1=50, x2=200, y2=400)
        obj = TrackedObject(track_id=1, class_name="person", confidence=0.9,
                            bbox=bbox, frame_history=[0], bbox_history=[bbox])
        self.assertEqual(obj.velocity(), 0.0)

    def test_tracked_object_velocity_two_bboxes(self):
        from models.schemas import TrackedObject, BoundingBox
        bbox1 = BoundingBox(x1=100, y1=100, x2=160, y2=400)
        bbox2 = BoundingBox(x1=200, y1=100, x2=260, y2=400)  # 100px right
        obj = TrackedObject(track_id=1, class_name="person", confidence=0.9,
                            bbox=bbox2, frame_history=[0, 1], bbox_history=[bbox1, bbox2])
        v = obj.velocity()
        self.assertAlmostEqual(v, 100.0)

    def test_incident_report_defaults(self):
        from models.schemas import IncidentReport
        from models.enums import IncidentStatus
        inc = IncidentReport(feed_id="f1")
        self.assertEqual(inc.status, IncidentStatus.DETECTED)
        self.assertIsNotNone(inc.incident_id)
        self.assertIsInstance(inc.narrative_log, list)


# ═══════════════════════════════════════════════════════════════════════════════
# ANOMALY ENGINE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def _make_person(track_id=1, x1=100, y1=50, x2=160, y2=400, age_seconds=0):
    from models.schemas import BoundingBox, TrackedObject
    bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    obj = TrackedObject(track_id=track_id, class_name="person", confidence=0.9,
                        bbox=bbox, frame_history=list(range(5)), bbox_history=[bbox])
    if age_seconds:
        obj.first_seen = datetime.utcnow() - timedelta(seconds=age_seconds)
    return obj


def _make_bag(track_id=10, age_seconds=0):
    from models.schemas import BoundingBox, TrackedObject
    bbox = BoundingBox(x1=200, y1=300, x2=260, y2=360)
    obj = TrackedObject(track_id=track_id, class_name="backpack", confidence=0.85,
                        bbox=bbox, frame_history=[1], bbox_history=[bbox])
    if age_seconds:
        obj.first_seen = datetime.utcnow() - timedelta(seconds=age_seconds)
    return obj


def _make_fr(feed_id="test-feed", frame_number=100):
    from models.schemas import FrameResult
    return FrameResult(feed_id=feed_id, frame_number=frame_number, detections=[])


class TestLoiteringDetector(unittest.TestCase):

    def test_no_anomaly_below_threshold(self):
        from core.anomaly_engine import LoiteringDetector
        d = LoiteringDetector(threshold_seconds=60.0)
        person = _make_person(age_seconds=30)
        events = d.check(_make_fr(), {1: person})
        self.assertEqual(events, [])

    def test_detects_loitering(self):
        from core.anomaly_engine import LoiteringDetector
        from models.enums import AnomalyType
        d = LoiteringDetector(threshold_seconds=60.0)
        person = _make_person(age_seconds=65)
        events = d.check(_make_fr(), {1: person})
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].anomaly_type, AnomalyType.LOITERING)
        self.assertIn(1, events[0].track_ids)

    def test_no_duplicate_alert(self):
        from core.anomaly_engine import LoiteringDetector
        d = LoiteringDetector(threshold_seconds=60.0)
        person = _make_person(age_seconds=65)
        e1 = d.check(_make_fr(), {1: person})
        e2 = d.check(_make_fr(), {1: person})
        self.assertEqual(len(e1), 1)
        self.assertEqual(len(e2), 0)

    def test_ignores_non_person(self):
        from core.anomaly_engine import LoiteringDetector
        d = LoiteringDetector(threshold_seconds=60.0)
        bag = _make_bag(age_seconds=65)
        events = d.check(_make_fr(), {10: bag})
        self.assertEqual(events, [])


class TestRunningDetector(unittest.TestCase):

    def test_no_anomaly_low_velocity(self):
        from core.anomaly_engine import RunningDetector
        from models.schemas import BoundingBox, TrackedObject
        d = RunningDetector(velocity_threshold=80.0)
        b1 = BoundingBox(x1=100, y1=100, x2=160, y2=400)
        b2 = BoundingBox(x1=110, y1=100, x2=170, y2=400)
        obj = TrackedObject(track_id=1, class_name="person", confidence=0.9,
                            bbox=b2, frame_history=[0, 1], bbox_history=[b1, b2])
        events = d.check(_make_fr(frame_number=0), {1: obj})
        self.assertEqual(events, [])

    def test_detects_running(self):
        from core.anomaly_engine import RunningDetector
        from models.schemas import BoundingBox, TrackedObject
        from models.enums import AnomalyType
        d = RunningDetector(velocity_threshold=80.0)
        b1 = BoundingBox(x1=100, y1=100, x2=160, y2=400)
        b2 = BoundingBox(x1=200, y1=100, x2=260, y2=400)  # 100px
        obj = TrackedObject(track_id=1, class_name="person", confidence=0.9,
                            bbox=b2, frame_history=[0, 1], bbox_history=[b1, b2])
        events = d.check(_make_fr(frame_number=100), {1: obj})
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].anomaly_type, AnomalyType.RUNNING)

    def test_zero_velocity_single_bbox(self):
        from core.anomaly_engine import RunningDetector
        from models.schemas import BoundingBox, TrackedObject
        d = RunningDetector(velocity_threshold=80.0)
        bbox = BoundingBox(x1=100, y1=100, x2=160, y2=400)
        obj = TrackedObject(track_id=1, class_name="person", confidence=0.9,
                            bbox=bbox, frame_history=[0], bbox_history=[bbox])
        events = d.check(_make_fr(), {1: obj})
        self.assertEqual(events, [])


class TestFallDetector(unittest.TestCase):

    def test_no_anomaly_standing(self):
        from core.anomaly_engine import FallDetector
        d = FallDetector(aspect_ratio_threshold=1.5)
        person = _make_person(x1=100, y1=50, x2=160, y2=400)  # w=60, h=350
        events = d.check(_make_fr(frame_number=100), {1: person})
        self.assertEqual(events, [])

    def test_detects_fall(self):
        from core.anomaly_engine import FallDetector
        from models.schemas import BoundingBox, TrackedObject
        from models.enums import AnomalyType
        d = FallDetector(aspect_ratio_threshold=1.5)
        bbox = BoundingBox(x1=50, y1=300, x2=450, y2=360)  # w=400, h=60
        person = TrackedObject(track_id=1, class_name="person", confidence=0.9,
                               bbox=bbox, frame_history=[0], bbox_history=[bbox])
        events = d.check(_make_fr(frame_number=100), {1: person})
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].anomaly_type, AnomalyType.FALL)

    def test_zero_height_no_crash(self):
        from core.anomaly_engine import FallDetector
        from models.schemas import BoundingBox, TrackedObject
        d = FallDetector()
        bbox = BoundingBox(x1=100, y1=200, x2=200, y2=200)
        person = TrackedObject(track_id=1, class_name="person", confidence=0.9,
                               bbox=bbox, frame_history=[0], bbox_history=[bbox])
        events = d.check(_make_fr(), {1: person})
        self.assertEqual(events, [])


class TestCrowdSurgeDetector(unittest.TestCase):

    def test_no_anomaly_gradual(self):
        from core.anomaly_engine import CrowdSurgeDetector
        d = CrowdSurgeDetector(surge_count=5)
        persons = {i: _make_person(track_id=i) for i in range(3)}
        events = d.check(_make_fr(frame_number=0), persons)
        self.assertEqual(events, [])

    def test_detects_surge(self):
        from core.anomaly_engine import CrowdSurgeDetector
        from models.enums import AnomalyType
        d = CrowdSurgeDetector(surge_count=5)
        d.check(_make_fr(frame_number=0), {})
        persons = {i: _make_person(track_id=i) for i in range(8)}
        events = d.check(_make_fr(frame_number=1), persons)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].anomaly_type, AnomalyType.CROWD_SURGE)


class TestAbandonedObjectDetector(unittest.TestCase):

    def test_no_anomaly_with_nearby_person(self):
        from core.anomaly_engine import AbandonedObjectDetector
        from models.schemas import BoundingBox, TrackedObject
        d = AbandonedObjectDetector(stillness_seconds=30.0)
        bag = _make_bag(age_seconds=35)
        person_bbox = BoundingBox(x1=210, y1=290, x2=270, y2=380)
        person = TrackedObject(track_id=1, class_name="person", confidence=0.9,
                               bbox=person_bbox, frame_history=[0], bbox_history=[person_bbox])
        events = d.check(_make_fr(), {10: bag, 1: person})
        self.assertEqual(events, [])

    def test_detects_abandoned(self):
        from core.anomaly_engine import AbandonedObjectDetector
        from models.enums import AnomalyType
        d = AbandonedObjectDetector(stillness_seconds=30.0)
        bag = _make_bag(age_seconds=35)
        events = d.check(_make_fr(), {10: bag})
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].anomaly_type, AnomalyType.ABANDONED_OBJECT)

    def test_too_new_object_ignored(self):
        from core.anomaly_engine import AbandonedObjectDetector
        d = AbandonedObjectDetector(stillness_seconds=30.0)
        bag = _make_bag(age_seconds=5)
        events = d.check(_make_fr(), {10: bag})
        self.assertEqual(events, [])


class TestAnomalyEngine(unittest.TestCase):

    def test_engine_has_five_detectors(self):
        from core.anomaly_engine import AnomalyEngine
        e = AnomalyEngine()
        self.assertEqual(len(e.detectors), 5)

    def test_engine_returns_list(self):
        from core.anomaly_engine import AnomalyEngine
        e = AnomalyEngine()
        person = _make_person()
        events = e.analyze(_make_fr(), {1: person})
        self.assertIsInstance(events, list)

    def test_engine_handles_exception_gracefully(self):
        from core.anomaly_engine import AnomalyEngine, LoiteringDetector
        e = AnomalyEngine()
        broken = LoiteringDetector()
        broken.check = lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
        e.detectors = [broken]
        events = e.analyze(_make_fr(), {})
        self.assertEqual(events, [])

    def test_end_to_end_loitering_detection(self):
        from core.anomaly_engine import AnomalyEngine, LoiteringDetector
        from models.enums import AnomalyType
        e = AnomalyEngine()
        e.detectors = [LoiteringDetector(threshold_seconds=60.0)]
        person = _make_person(age_seconds=65)
        events = e.analyze(_make_fr(), {1: person})
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].anomaly_type, AnomalyType.LOITERING)


# ═══════════════════════════════════════════════════════════════════════════════
# FRAME PROCESSOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestYOLODetector(unittest.TestCase):

    def setUp(self):
        from core.frame_processor import YOLODetector
        self.detector = YOLODetector()
        self.detector._model = "mock"
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_mock_returns_list(self):
        results = self.detector.detect(self.frame)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

    def test_mock_returns_person(self):
        results = self.detector.detect(self.frame)
        self.assertEqual(results[0]["class_name"], "person")

    def test_mock_confidence_valid(self):
        results = self.detector.detect(self.frame)
        conf = results[0]["confidence"]
        self.assertGreater(conf, 0)
        self.assertLessEqual(conf, 1.0)

    def test_mock_bbox_valid(self):
        results = self.detector.detect(self.frame)
        x1, y1, x2, y2 = results[0]["bbox"]
        self.assertLess(x1, x2)
        self.assertLess(y1, y2)


class TestDeepSORTTracker(unittest.TestCase):

    def setUp(self):
        from core.frame_processor import DeepSORTTracker
        self.tracker = DeepSORTTracker()
        self.tracker._tracker = "mock"
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_empty_detections(self):
        results = self.tracker.update([], self.frame)
        self.assertEqual(results, [])

    def test_two_detections_two_tracks(self):
        dets = [
            {"class_name": "person", "confidence": 0.9, "bbox": (100, 50, 200, 400)},
            {"class_name": "person", "confidence": 0.85, "bbox": (300, 50, 400, 400)},
        ]
        results = self.tracker.update(dets, self.frame)
        self.assertEqual(len(results), 2)
        ids = {r[0] for r in results}
        self.assertEqual(len(ids), 2)

    def test_result_structure(self):
        dets = [{"class_name": "person", "confidence": 0.9, "bbox": (100, 50, 200, 400)}]
        results = self.tracker.update(dets, self.frame)
        track_id, cls, conf, bbox = results[0]
        self.assertIsInstance(track_id, int)
        self.assertEqual(cls, "person")
        self.assertEqual(len(bbox), 4)


class TestMoondreamDescriber(unittest.TestCase):

    def setUp(self):
        from core.frame_processor import MoondreamDescriber
        self.desc = MoondreamDescriber()
        self.desc._model = "mock"
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_returns_non_empty_string(self):
        result = self.desc.describe(self.frame)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


class TestFrameProcessor(unittest.TestCase):

    def setUp(self):
        from core.frame_processor import FrameProcessor
        self.proc = FrameProcessor()
        self.proc.detector._model = "mock"
        self.proc.tracker._tracker = "mock"
        self.proc.describer._model = "mock"
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_returns_frame_result(self):
        from models.schemas import FrameResult
        result = self.proc.process_frame("feed-1", self.frame)
        self.assertIsInstance(result, FrameResult)
        self.assertEqual(result.feed_id, "feed-1")

    def test_frame_number_increments(self):
        r1 = self.proc.process_frame("feed-1", self.frame)
        r2 = self.proc.process_frame("feed-1", self.frame)
        self.assertEqual(r2.frame_number, r1.frame_number + 1)

    def test_manual_frame_number(self):
        result = self.proc.process_frame("feed-1", self.frame, frame_number=99)
        self.assertEqual(result.frame_number, 99)

    def test_detections_populated(self):
        result = self.proc.process_frame("feed-1", self.frame)
        self.assertGreater(len(result.detections), 0)

    def test_tracked_objects_updated(self):
        self.proc.process_frame("feed-1", self.frame)
        tracked = self.proc.get_tracked_objects("feed-1")
        self.assertGreater(len(tracked), 0)

    def test_scene_description_on_frame_zero(self):
        result = self.proc.process_frame("feed-1", self.frame, frame_number=0)
        self.assertIsNotNone(result.scene_description)

    def test_no_scene_description_frame_one(self):
        result = self.proc.process_frame("feed-1", self.frame, frame_number=1)
        self.assertIsNone(result.scene_description)

    def test_processing_time_non_negative(self):
        result = self.proc.process_frame("feed-1", self.frame)
        self.assertGreaterEqual(result.processing_time_ms, 0)

    def test_reset_clears_state(self):
        self.proc.process_frame("feed-1", self.frame)
        self.proc.reset_feed("feed-1")
        self.assertEqual(self.proc.get_tracked_objects("feed-1"), {})

    def test_multiple_feeds_isolated(self):
        self.proc.process_frame("feed-A", self.frame)
        self.proc.process_frame("feed-B", self.frame)
        self.assertIn("feed-A", self.proc._tracked_objects)
        self.assertIn("feed-B", self.proc._tracked_objects)
        self.assertIsNot(
            self.proc._tracked_objects["feed-A"],
            self.proc._tracked_objects["feed-B"]
        )


# ═══════════════════════════════════════════════════════════════════════════════
# NARRATIVE ENGINE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestNarrativeEngine(unittest.TestCase):

    def setUp(self):
        from core.narrative_engine import NarrativeEngine
        self.engine = NarrativeEngine()

    def _make_fr(self, feed_id="feed-1", frame_number=0, scene_desc=None):
        from models.schemas import FrameResult
        return FrameResult(feed_id=feed_id, frame_number=frame_number,
                           detections=[], scene_description=scene_desc)

    def _make_anomaly(self):
        from models.schemas import AnomalyEvent
        from models.enums import AnomalyType
        return AnomalyEvent(feed_id="feed-1", anomaly_type=AnomalyType.LOITERING,
                            track_ids=[1], frame_number=42, confidence=0.85,
                            description="Loitering detected.")

    def test_log_frame_returns_list(self):
        fr = self._make_fr()
        entries = self.engine.log_frame(fr, {}, [])
        self.assertIsInstance(entries, list)

    def test_scene_description_logged(self):
        fr = self._make_fr(scene_desc="A person is walking.")
        entries = self.engine.log_frame(fr, {}, [])
        texts = [e.text for e in entries]
        self.assertIn("A person is walking.", texts)

    def test_anomaly_logged_as_anomaly(self):
        from models.enums import AnomalyType
        fr = self._make_fr()
        anomaly = self._make_anomaly()
        entries = self.engine.log_frame(fr, {}, [anomaly])
        anomaly_entries = [e for e in entries if e.is_anomaly]
        self.assertEqual(len(anomaly_entries), 1)
        self.assertEqual(anomaly_entries[0].anomaly_type, AnomalyType.LOITERING)

    def test_get_recent_log_limits(self):
        fr = self._make_fr()
        anomaly = self._make_anomaly()
        for _ in range(30):
            self.engine.log_frame(fr, {}, [anomaly])
        recent = self.engine.get_recent_log("feed-1", last_n=10)
        self.assertEqual(len(recent), 10)

    def test_clear_log(self):
        fr = self._make_fr()
        self.engine.log_frame(fr, {}, [self._make_anomaly()])
        self.engine.clear_log("feed-1")
        self.assertEqual(self.engine.get_log("feed-1"), [])

    def test_multiple_feeds_isolated(self):
        from models.schemas import FrameResult
        fr_a = FrameResult(feed_id="feed-A", frame_number=0, detections=[])
        fr_b = FrameResult(feed_id="feed-B", frame_number=0, detections=[])
        self.engine.log_frame(fr_a, {}, [])
        self.engine.log_frame(fr_b, {}, [])
        self.assertIsNot(self.engine._logs.get("feed-A"), self.engine._logs.get("feed-B"))

    def test_mock_summary_returns_string(self):
        anomaly = self._make_anomaly()
        summary = run_async(self.engine.generate_incident_summary("feed-1", anomaly))
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 10)

    def test_mock_summary_mentions_anomaly_type(self):
        anomaly = self._make_anomaly()
        summary = run_async(self.engine.generate_incident_summary("feed-1", anomaly))
        self.assertIn("loitering", summary.lower())

    def test_build_context_empty(self):
        result = self.engine._build_context([])
        self.assertIn("No prior", result)

    def test_build_context_with_entries(self):
        from models.schemas import NarrativeEntry
        entries = [NarrativeEntry(feed_id="f1", text="Track-1 entered the scene.")]
        result = self.engine._build_context(entries)
        self.assertIn("Track-1", result)


# ═══════════════════════════════════════════════════════════════════════════════
# INCIDENT MANAGER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestIncidentManager(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.patch_settings = patch("core.incident_manager.settings")
        self.mock_settings = self.patch_settings.start()
        self.mock_settings.incidents_dir = Path(self.tmp) / "incidents"
        self.mock_settings.incidents_dir.mkdir(parents=True, exist_ok=True)
        self.mock_settings.pre_event_seconds = 2
        self.mock_settings.post_event_seconds = 1
        self.mock_settings.target_fps = 5
        self.mock_settings.max_buffer_frames = 50

        from core.incident_manager import IncidentManager
        self.mgr = IncidentManager("test-feed", "Test Camera")
        self.frame = np.zeros((10, 10, 3), dtype=np.uint8)

    def tearDown(self):
        self.patch_settings.stop()

    def test_ingest_frame_increases_buffer(self):
        self.assertEqual(len(self.mgr.buffer), 0)
        self.mgr.ingest_frame(self.frame, 0)
        self.assertEqual(len(self.mgr.buffer), 1)

    def test_no_incidents_initially(self):
        self.assertEqual(self.mgr.get_incident_count(), 0)
        self.assertEqual(self.mgr.get_all_incidents(), [])

    def test_get_nonexistent_incident(self):
        self.assertIsNone(self.mgr.get_incident("nope"))

    def test_save_and_load_json(self):
        from models.schemas import IncidentReport
        from models.enums import AnomalyType
        from core.incident_manager import IncidentManager
        inc = IncidentReport(feed_id="f1", anomaly_type=AnomalyType.FALL)
        path = Path(self.tmp) / "test.json"
        IncidentManager._save_json(inc, path)
        self.assertTrue(path.exists())
        loaded = IncidentManager.load_incident_from_json(path)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.feed_id, "f1")

    def test_load_nonexistent_json(self):
        from core.incident_manager import IncidentManager
        result = IncidentManager.load_incident_from_json(Path("/tmp/nonexistent.json"))
        self.assertIsNone(result)

    def test_trigger_incident_creates_report(self):
        from models.schemas import AnomalyEvent, NarrativeEntry
        from models.enums import AnomalyType, IncidentStatus
        for i in range(5):
            self.mgr.ingest_frame(self.frame, i)
        anomaly = AnomalyEvent(feed_id="test-feed", anomaly_type=AnomalyType.LOITERING,
                               track_ids=[1], frame_number=4, confidence=0.87,
                               description="Loitering.")
        with patch("asyncio.create_task"):
            with patch.object(self.mgr, "_finalize_incident", new=AsyncMock()):
                inc = run_async(self.mgr.trigger_incident(anomaly, [], "Summary"))
        self.assertEqual(inc.feed_id, "test-feed")
        self.assertEqual(inc.status, IncidentStatus.PROCESSING)
        self.assertAlmostEqual(inc.confidence_score, 0.87)


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATOR TEST
# ═══════════════════════════════════════════════════════════════════════════════

class TestReportGenerator(unittest.TestCase):

    def test_generate_pdf(self):
        from utils.report_generator import generate_incident_pdf
        from models.schemas import IncidentReport, NarrativeEntry
        from models.enums import AnomalyType, IncidentStatus
        import tempfile

        inc = IncidentReport(
            feed_id="f1",
            feed_name="Front Door",
            anomaly_type=AnomalyType.LOITERING,
            status=IncidentStatus.COMPLETE,
            confidence_score=0.88,
            scene_summary="A person was seen loitering near the entrance.",
            narrative_log=[
                NarrativeEntry(feed_id="f1", text="Track-1 entered the scene."),
                NarrativeEntry(feed_id="f1", text="Loitering detected.", is_anomaly=True),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.pdf"
            result = generate_incident_pdf(inc, out)
            self.assertTrue(result)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 1000)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestFrameRingBuffer,
        TestSchemas,
        TestLoiteringDetector,
        TestRunningDetector,
        TestFallDetector,
        TestCrowdSurgeDetector,
        TestAbandonedObjectDetector,
        TestAnomalyEngine,
        TestYOLODetector,
        TestDeepSORTTracker,
        TestMoondreamDescriber,
        TestFrameProcessor,
        TestNarrativeEngine,
        TestIncidentManager,
        TestReportGenerator,
    ]

    for tc in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(tc))

    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  🔍 WitnessAI — Test Suite{RESET}")
    print(f"{BOLD}{'='*65}{RESET}\n")

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total - failures - errors

    print(f"\n{BOLD}{'='*65}{RESET}")
    print(f"{BOLD}  Results:{RESET}")
    print(f"  {GREEN}✓ Passed: {passed}{RESET}")
    if failures:
        print(f"  {RED}✗ Failed: {failures}{RESET}")
    if errors:
        print(f"  {RED}✗ Errors: {errors}{RESET}")
    print(f"  Total:   {total}")
    print(f"{BOLD}{'='*65}{RESET}\n")

    sys.exit(0 if result.wasSuccessful() else 1)
