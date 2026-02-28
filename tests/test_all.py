"""
WitnessAI - Full Test Suite (unittest, no external deps)
Run: python3 -m pytest tests/test_all.py -v
  OR: python3 -m unittest tests.test_all -v
"""
import sys
import os
import unittest
import numpy as np
import json
import tempfile
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.schemas import (
    BoundingBox, DetectedObject, FrameDetections, AnomalyEvent,
    AnomalyType, ConfidenceLevel, Incident, IncidentStatus,
    IncidentNarrative, NarrativeEntry, CameraConfig
)
from detection.detector import Detector
from detection.tracker import Tracker, iou
from detection.anomaly import AnomalyDetector, AnomalyConfig
from narration.narrator import Narrator, MockLLM, build_llm
from evidence.buffer import RollingBuffer, BufferedFrame
from evidence.packager import Packager
from core.correlator import Correlator
from core.agent import WitnessAgent


# ─────────────────────────── HELPERS ──────────────────────────────

def make_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)

def make_obj(track_id=1, x1=10, y1=10, x2=60, y2=160, cls="person", speed=0.0, cam="cam1"):
    vx = speed / (2 ** 0.5)
    return DetectedObject(
        track_id=track_id, class_name=cls, confidence=0.9,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        camera_id=cam, velocity_x=vx, velocity_y=vx
    )

def make_detections(objects=None, cam="cam1", fn=1):
    return FrameDetections(
        camera_id=cam, frame_number=fn,
        timestamp=datetime.utcnow(), objects=objects or []
    )

def make_anomaly(cam="cam1", atype=AnomalyType.LOITERING, tid=1):
    return AnomalyEvent(
        id=f"ev-{cam}-{atype.value}", camera_id=cam,
        anomaly_type=atype, confidence=ConfidenceLevel.HIGH,
        description=f"{atype.value} on {cam}", involved_track_ids=[tid]
    )

def make_camera_config(cam_id="cam-test"):
    return CameraConfig(
        camera_id=cam_id, name="Test Camera", source="mock://0",
        loitering_threshold=5, running_velocity_threshold=50.0,
        crowd_density_threshold=2
    )

def make_agent(cam_id="cam-test", output_dir="/tmp/witnessai_test"):
    config = make_camera_config(cam_id)
    narrator = Narrator(llm=MockLLM())
    packager = Packager(output_dir=output_dir)
    correlator = Correlator()
    agent = WitnessAgent(
        config=config, narrator=narrator, packager=packager,
        correlator=correlator, mock=True
    )
    agent.load()
    return agent


# ─────────────────────── BOUNDING BOX ─────────────────────────────

class TestBoundingBox(unittest.TestCase):
    def test_width_height(self):
        box = BoundingBox(x1=10, y1=20, x2=110, y2=220)
        self.assertEqual(box.width, 100)
        self.assertEqual(box.height, 200)

    def test_center(self):
        box = BoundingBox(x1=0, y1=0, x2=100, y2=200)
        self.assertEqual(box.center_x, 50)
        self.assertEqual(box.center_y, 100)

    def test_area(self):
        box = BoundingBox(x1=0, y1=0, x2=10, y2=20)
        self.assertEqual(box.area, 200)


# ─────────────────────── DETECTED OBJECT ──────────────────────────

class TestDetectedObject(unittest.TestCase):
    def test_speed(self):
        obj = make_obj(speed=0)
        obj.velocity_x = 3.0
        obj.velocity_y = 4.0
        self.assertAlmostEqual(obj.speed, 5.0)

    def test_zero_speed(self):
        obj = make_obj()
        self.assertEqual(obj.speed, 0.0)


# ──────────────────────────── IOu ─────────────────────────────────

class TestIoU(unittest.TestCase):
    def test_perfect_overlap(self):
        box = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        self.assertAlmostEqual(iou(box, box), 1.0)

    def test_no_overlap(self):
        a = BoundingBox(x1=0, y1=0, x2=50, y2=50)
        b = BoundingBox(x1=100, y1=100, x2=200, y2=200)
        self.assertAlmostEqual(iou(a, b), 0.0)

    def test_partial_overlap(self):
        a = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        b = BoundingBox(x1=50, y1=50, x2=150, y2=150)
        result = iou(a, b)
        self.assertGreater(result, 0)
        self.assertLess(result, 1)


# ─────────────────────────── DETECTOR ─────────────────────────────

class TestDetector(unittest.TestCase):
    def setUp(self):
        self.detector = Detector(mock=True, camera_id="cam-test", confidence_threshold=0.5)
        self.detector.load()
        self.frame = make_frame()

    def test_loaded(self):
        self.assertIsNotNone(self.detector._model)

    def test_returns_frame_detections(self):
        result = self.detector.detect(self.frame)
        self.assertIsInstance(result, FrameDetections)

    def test_returns_persons(self):
        result = self.detector.detect(self.frame)
        self.assertEqual(len(result.objects), 2)
        for obj in result.objects:
            self.assertEqual(obj.class_name, "person")

    def test_confidence_above_threshold(self):
        result = self.detector.detect(self.frame)
        for obj in result.objects:
            self.assertGreaterEqual(obj.confidence, 0.5)

    def test_frame_count(self):
        self.detector.detect(self.frame)
        self.detector.detect(self.frame)
        self.assertEqual(self.detector.frame_count, 2)

    def test_reset(self):
        self.detector.detect(self.frame)
        self.detector.reset()
        self.assertEqual(self.detector.frame_count, 0)

    def test_not_loaded_raises(self):
        d = Detector(mock=True)
        with self.assertRaises(RuntimeError):
            d.detect(self.frame)

    def test_high_threshold_filters_all(self):
        d = Detector(mock=True, confidence_threshold=0.99)
        d.load()
        result = d.detect(self.frame)
        self.assertEqual(len(result.objects), 0)

    def test_camera_id_assigned(self):
        result = self.detector.detect(self.frame)
        for obj in result.objects:
            self.assertEqual(obj.camera_id, "cam-test")

    def test_bbox_valid(self):
        result = self.detector.detect(self.frame)
        for obj in result.objects:
            self.assertGreater(obj.bbox.x2, obj.bbox.x1)
            self.assertGreater(obj.bbox.y2, obj.bbox.y1)


# ─────────────────────────── TRACKER ──────────────────────────────

class TestTracker(unittest.TestCase):
    def test_new_detection_creates_track(self):
        tracker = Tracker()
        tracker.update(make_detections(objects=[make_obj()]))
        self.assertEqual(len(tracker.active_tracks), 1)

    def test_track_id_assigned(self):
        tracker = Tracker()
        result = tracker.update(make_detections(objects=[make_obj()]))
        self.assertGreaterEqual(result.objects[0].track_id, 1)

    def test_same_object_keeps_id(self):
        tracker = Tracker()
        r1 = tracker.update(make_detections(objects=[make_obj(x1=10, y1=10, x2=100, y2=200)]))
        tid1 = r1.objects[0].track_id
        import time; time.sleep(0.05)
        r2 = tracker.update(make_detections(objects=[make_obj(x1=12, y1=12, x2=102, y2=202)]))
        tid2 = r2.objects[0].track_id
        self.assertEqual(tid1, tid2)

    def test_multiple_objects_different_ids(self):
        tracker = Tracker()
        objs = [make_obj(x1=0, y1=0, x2=50, y2=100), make_obj(x1=300, y1=0, x2=400, y2=100)]
        result = tracker.update(make_detections(objects=objs))
        ids = [o.track_id for o in result.objects]
        self.assertEqual(len(set(ids)), 2)

    def test_missing_increments_misses(self):
        tracker = Tracker()
        tracker.update(make_detections(objects=[make_obj()]))
        tracker.update(make_detections(objects=[]))
        track = list(tracker.active_tracks.values())[0]
        self.assertEqual(track.misses, 1)

    def test_stale_track_removed(self):
        tracker = Tracker()
        tracker.MAX_MISSES = 2
        tracker.update(make_detections(objects=[make_obj()]))
        for _ in range(3):
            tracker.update(make_detections(objects=[]))
        self.assertEqual(len(tracker.active_tracks), 0)

    def test_reset(self):
        tracker = Tracker()
        tracker.update(make_detections(objects=[make_obj()]))
        tracker.reset()
        self.assertEqual(len(tracker.active_tracks), 0)


# ─────────────────────────── ANOMALY ──────────────────────────────

class TestLoitering(unittest.TestCase):
    def _setup(self, threshold=5):
        config = AnomalyConfig(loitering_threshold=threshold)
        detector = AnomalyDetector(camera_id="cam1", config=config)
        tracker = Tracker()
        obj = make_obj(track_id=1)
        tracker.update(make_detections(objects=[obj]))
        return detector, tracker

    def test_triggers_above_threshold(self):
        detector, tracker = self._setup(threshold=5)
        tracker._tracks[1].stationary_frames = 90  # 6s @ 15fps
        frame = make_detections(objects=[make_obj(track_id=1)])
        events = detector.evaluate(frame, tracker)
        types = [e.anomaly_type for e in events]
        self.assertIn(AnomalyType.LOITERING, types)

    def test_not_triggered_below_threshold(self):
        detector, tracker = self._setup(threshold=30)
        tracker._tracks[1].stationary_frames = 5  # 0.3s
        frame = make_detections(objects=[make_obj(track_id=1)])
        events = detector.evaluate(frame, tracker)
        types = [e.anomaly_type for e in events]
        self.assertNotIn(AnomalyType.LOITERING, types)

    def test_deduplication(self):
        detector, tracker = self._setup(threshold=5)
        tracker._tracks[1].stationary_frames = 90
        frame = make_detections(objects=[make_obj(track_id=1)])
        e1 = detector.evaluate(frame, tracker)
        e2 = detector.evaluate(frame, tracker)
        loitering1 = [e for e in e1 if e.anomaly_type == AnomalyType.LOITERING]
        loitering2 = [e for e in e2 if e.anomaly_type == AnomalyType.LOITERING]
        self.assertEqual(len(loitering1), 1)
        self.assertEqual(len(loitering2), 0)

    def test_reset_clears_dedup(self):
        detector, tracker = self._setup(threshold=5)
        tracker._tracks[1].stationary_frames = 90
        frame = make_detections(objects=[make_obj(track_id=1)])
        detector.evaluate(frame, tracker)
        detector.reset_alerts()
        events = detector.evaluate(frame, tracker)
        self.assertTrue(any(e.anomaly_type == AnomalyType.LOITERING for e in events))


class TestRunning(unittest.TestCase):
    def test_triggers_above_threshold(self):
        config = AnomalyConfig(running_velocity_threshold=100.0)
        detector = AnomalyDetector(camera_id="cam1", config=config)
        tracker = Tracker()
        obj = make_obj(track_id=2, speed=200.0)
        frame = make_detections(objects=[obj])
        tracker.update(frame)
        events = detector.evaluate(frame, tracker)
        self.assertTrue(any(e.anomaly_type == AnomalyType.RUNNING for e in events))

    def test_not_triggered_below_threshold(self):
        config = AnomalyConfig(running_velocity_threshold=200.0)
        detector = AnomalyDetector(camera_id="cam1", config=config)
        tracker = Tracker()
        obj = make_obj(track_id=2, speed=50.0)
        frame = make_detections(objects=[obj])
        tracker.update(frame)
        events = detector.evaluate(frame, tracker)
        self.assertFalse(any(e.anomaly_type == AnomalyType.RUNNING for e in events))


class TestCrowdSurge(unittest.TestCase):
    def test_triggers_above_threshold(self):
        config = AnomalyConfig(crowd_density_threshold=3)
        detector = AnomalyDetector(camera_id="cam1", config=config)
        tracker = Tracker()
        objects = [make_obj(track_id=i, x1=i*100, x2=i*100+50) for i in range(5)]
        frame = make_detections(objects=objects)
        tracker.update(frame)
        events = detector.evaluate(frame, tracker)
        self.assertTrue(any(e.anomaly_type == AnomalyType.CROWD_SURGE for e in events))

    def test_metadata_has_count(self):
        config = AnomalyConfig(crowd_density_threshold=2)
        detector = AnomalyDetector(camera_id="cam1", config=config)
        tracker = Tracker()
        objects = [make_obj(track_id=i, x1=i*100, x2=i*100+50) for i in range(4)]
        frame = make_detections(objects=objects)
        tracker.update(frame)
        events = detector.evaluate(frame, tracker)
        surge = [e for e in events if e.anomaly_type == AnomalyType.CROWD_SURGE]
        self.assertEqual(surge[0].metadata["person_count"], 4)


class TestFall(unittest.TestCase):
    def test_triggers_when_prone(self):
        config = AnomalyConfig(fall_aspect_ratio_threshold=2.0)
        detector = AnomalyDetector(camera_id="cam1", config=config)
        tracker = Tracker()
        prone = DetectedObject(
            track_id=10, class_name="person", confidence=0.9,
            bbox=BoundingBox(x1=0, y1=100, x2=200, y2=150),  # W=200, H=50 → aspect=4
            camera_id="cam1"
        )
        frame = make_detections(objects=[prone])
        tracker.update(frame)
        events = detector.evaluate(frame, tracker)
        self.assertTrue(any(e.anomaly_type == AnomalyType.FALL_DETECTED for e in events))

    def test_not_triggered_standing(self):
        config = AnomalyConfig(fall_aspect_ratio_threshold=2.0)
        detector = AnomalyDetector(camera_id="cam1", config=config)
        tracker = Tracker()
        standing = make_obj(x1=0, y1=0, x2=50, y2=200)  # W=50, H=200 → aspect=0.25
        frame = make_detections(objects=[standing])
        tracker.update(frame)
        events = detector.evaluate(frame, tracker)
        self.assertFalse(any(e.anomaly_type == AnomalyType.FALL_DETECTED for e in events))


# ──────────────────────────── NARRATOR ────────────────────────────

class TestMockLLM(unittest.TestCase):
    def test_returns_string(self):
        llm = MockLLM()
        result = llm.generate("test prompt")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_loitering_response(self):
        llm = MockLLM()
        result = llm.generate("loitering detected")
        self.assertTrue(len(result) > 0)

    def test_fall_response(self):
        llm = MockLLM()
        result = llm.generate("fall detected prone")
        self.assertTrue(len(result) > 0)


class TestNarrator(unittest.TestCase):
    def setUp(self):
        self.narrator = Narrator(llm=MockLLM())

    def test_narrate_creates_entry(self):
        entry = self.narrator.narrate_anomaly("inc-001", make_anomaly())
        self.assertIsNotNone(entry)
        self.assertGreater(len(entry.text), 0)

    def test_narrate_links_anomaly_id(self):
        anomaly = make_anomaly()
        entry = self.narrator.narrate_anomaly("inc-001", anomaly)
        self.assertEqual(entry.anomaly_id, anomaly.id)

    def test_narrative_accumulates_entries(self):
        self.narrator.narrate_anomaly("inc-multi", make_anomaly(atype=AnomalyType.LOITERING))
        self.narrator.narrate_anomaly("inc-multi", make_anomaly(atype=AnomalyType.RUNNING))
        narrative = self.narrator.get_narrative("inc-multi")
        self.assertEqual(len(narrative.entries), 2)

    def test_unknown_incident_returns_none(self):
        self.assertIsNone(self.narrator.get_narrative("does-not-exist"))

    def test_full_text_has_timestamp_separator(self):
        self.narrator.narrate_anomaly("inc-ts", make_anomaly())
        narrative = self.narrator.get_narrative("inc-ts")
        self.assertIn("—", narrative.full_text())

    def test_llm_failure_fallback(self):
        class FailingLLM:
            def generate(self, prompt): raise RuntimeError("down")
        narrator = Narrator(llm=FailingLLM())
        entry = narrator.narrate_anomaly("inc-fail", make_anomaly())
        self.assertIn("[AUTO]", entry.text)

    def test_separate_incidents_isolated(self):
        self.narrator.narrate_anomaly("inc-A", make_anomaly(atype=AnomalyType.LOITERING))
        self.narrator.narrate_anomaly("inc-B", make_anomaly(atype=AnomalyType.RUNNING))
        self.assertEqual(len(self.narrator.get_narrative("inc-A").entries), 1)
        self.assertEqual(len(self.narrator.get_narrative("inc-B").entries), 1)

    def test_build_llm_mock(self):
        llm = build_llm("mock")
        self.assertIsInstance(llm, MockLLM)


# ─────────────────────────── BUFFER ───────────────────────────────

class TestRollingBuffer(unittest.TestCase):
    def test_push_adds_frame(self):
        buf = RollingBuffer(duration_seconds=10, fps=5.0)
        buf.push(make_frame(), datetime.utcnow(), 1)
        self.assertEqual(buf.size, 1)

    def test_max_size_enforced(self):
        buf = RollingBuffer(duration_seconds=2, fps=5.0)  # max 10
        for i in range(20):
            buf.push(make_frame(), datetime.utcnow(), i)
        self.assertEqual(buf.size, 10)

    def test_snapshot_returns_all(self):
        buf = RollingBuffer(duration_seconds=10, fps=5.0)
        for i in range(5):
            buf.push(make_frame(), datetime.utcnow(), i)
        self.assertEqual(len(buf.snapshot()), 5)

    def test_snapshot_non_destructive(self):
        buf = RollingBuffer(duration_seconds=10, fps=5.0)
        buf.push(make_frame(), datetime.utcnow(), 1)
        buf.snapshot()
        self.assertEqual(buf.size, 1)

    def test_is_ready_half_full(self):
        buf = RollingBuffer(duration_seconds=2, fps=10.0)  # max 20
        self.assertFalse(buf.is_ready())
        for i in range(10):
            buf.push(make_frame(), datetime.utcnow(), i)
        self.assertTrue(buf.is_ready())

    def test_clear(self):
        buf = RollingBuffer(duration_seconds=5, fps=5.0)
        for i in range(5):
            buf.push(make_frame(), datetime.utcnow(), i)
        buf.clear()
        self.assertEqual(buf.size, 0)

    def test_eviction_order(self):
        buf = RollingBuffer(duration_seconds=1, fps=5.0)  # max 5
        for i in range(10):
            buf.push(make_frame(), datetime.utcnow(), i)
        snap = buf.snapshot()
        frame_numbers = [f.frame_number for f in snap]
        self.assertEqual(frame_numbers, [5, 6, 7, 8, 9])

    def test_duration_covered(self):
        buf = RollingBuffer(duration_seconds=30, fps=15.0)
        now = datetime.utcnow()
        buf.push(make_frame(), now, 1)
        buf.push(make_frame(), now + timedelta(seconds=10), 2)
        self.assertAlmostEqual(buf.duration_covered, 10.0, delta=0.1)


# ─────────────────────────── PACKAGER ─────────────────────────────

class TestPackager(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.packager = Packager(output_dir=self.tmp)
        self.anomaly = make_anomaly()

    def _make_narrative(self, incident_id="inc-001"):
        narrative = IncidentNarrative(
            incident_id=incident_id, camera_id="cam1", started_at=datetime.utcnow()
        )
        narrative.entries.append(NarrativeEntry(
            timestamp=datetime.utcnow(), text="Test narrative.", camera_id="cam1"
        ))
        return narrative

    def _make_frames(self, n=3):
        return [BufferedFrame(frame=make_frame(100, 100), timestamp=datetime.utcnow(), frame_number=i) for i in range(n)]

    def test_build_returns_package(self):
        pkg = self.packager.build("inc-001", "cam1", [], [], [self.anomaly])
        self.assertIsNotNone(pkg)
        self.assertEqual(pkg.incident_id, "inc-001")

    def test_report_created(self):
        pkg = self.packager.build("inc-report", "cam1", [], [], [self.anomaly])
        import os
        self.assertTrue(os.path.exists(pkg.report_path))

    def test_report_valid_json(self):
        pkg = self.packager.build("inc-json", "cam1", [], [], [self.anomaly])
        with open(pkg.report_path) as f:
            report = json.load(f)
        self.assertEqual(report["incident_id"], "inc-json")
        self.assertEqual(report["anomaly_count"], 1)

    def test_narrative_in_report(self):
        narrative = self._make_narrative("inc-narr")
        pkg = self.packager.build("inc-narr", "cam1", [], [], [self.anomaly], narrative=narrative)
        with open(pkg.report_path) as f:
            report = json.load(f)
        self.assertGreater(len(report["narrative"]), 0)

    def test_empty_frames_no_video(self):
        pkg = self.packager.build("inc-novid", "cam1", [], [], [self.anomaly])
        self.assertIsNone(pkg.video_path)

    def test_unique_package_ids(self):
        p1 = self.packager.build("inc-u1", "cam1", [], [], [self.anomaly])
        p2 = self.packager.build("inc-u2", "cam1", [], [], [self.anomaly])
        self.assertNotEqual(p1.package_id, p2.package_id)

    def test_list_packages(self):
        self.packager.build("inc-list-A", "cam1", [], [], [self.anomaly])
        self.packager.build("inc-list-B", "cam1", [], [], [self.anomaly])
        packages = self.packager.list_packages()
        self.assertIn("inc-list-A", packages)
        self.assertIn("inc-list-B", packages)


# ─────────────────────────── CORRELATOR ───────────────────────────

class TestCorrelator(unittest.TestCase):
    def test_register_incident(self):
        correlator = Correlator()
        incident = Incident(incident_id="inc-1", camera_id="cam1",
                            anomaly_events=[make_anomaly(cam="cam1")])
        correlator.register_incident(incident)
        self.assertEqual(len(correlator._incidents), 1)

    def test_unified_timeline_empty(self):
        correlator = Correlator()
        self.assertEqual(correlator.unified_timeline(), [])

    def test_unified_timeline_populated(self):
        correlator = Correlator()
        for i, cam in enumerate(["cam1", "cam2"]):
            incident = Incident(incident_id=f"inc-{i}", camera_id=cam,
                                anomaly_events=[make_anomaly(cam=cam)])
            correlator.register_incident(incident)
        self.assertEqual(len(correlator.unified_timeline()), 2)

    def test_cross_camera_match_same_type(self):
        correlator = Correlator()
        correlator.ingest_anomaly(make_anomaly(cam="cam1", atype=AnomalyType.RUNNING))
        matches = correlator.ingest_anomaly(make_anomaly(cam="cam2", atype=AnomalyType.RUNNING))
        self.assertGreater(len(matches), 0)

    def test_no_match_same_camera(self):
        correlator = Correlator()
        correlator.ingest_anomaly(make_anomaly(cam="cam1", atype=AnomalyType.RUNNING))
        matches = correlator.ingest_anomaly(make_anomaly(cam="cam1", atype=AnomalyType.RUNNING))
        self.assertEqual(len(matches), 0)

    def test_no_match_different_types(self):
        correlator = Correlator()
        correlator.ingest_anomaly(make_anomaly(cam="cam1", atype=AnomalyType.RUNNING))
        matches = correlator.ingest_anomaly(make_anomaly(cam="cam2", atype=AnomalyType.LOITERING))
        self.assertEqual(len(matches), 0)

    def test_summary_keys(self):
        correlator = Correlator()
        summary = correlator.summary()
        self.assertIn("total_incidents", summary)
        self.assertIn("total_cross_matches", summary)
        self.assertIn("cameras_tracked", summary)

    def test_cross_match_multi_camera(self):
        correlator = Correlator()
        correlator.ingest_anomaly(make_anomaly(cam="cam1", atype=AnomalyType.CROWD_SURGE))
        matches = correlator.ingest_anomaly(make_anomaly(cam="cam2", atype=AnomalyType.CROWD_SURGE))
        self.assertGreater(len(matches), 0)
        self.assertEqual(len(matches[0].cameras_seen()), 2)


# ─────────────────────────── AGENT ────────────────────────────────

class TestWitnessAgent(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.agent = make_agent(output_dir=self.tmp)
        self.frame = make_frame()

    def test_loads(self):
        self.assertTrue(self.agent._running)

    def test_process_returns_list(self):
        result = self.agent.process_frame(self.frame)
        self.assertIsInstance(result, list)

    def test_frame_count_increments(self):
        self.agent.process_frame(self.frame)
        self.assertEqual(self.agent._total_frames, 1)

    def test_status(self):
        status = self.agent.status()
        self.assertEqual(status.camera_id, "cam-test")
        self.assertTrue(status.is_running)

    def test_stop(self):
        self.agent.stop()
        self.assertFalse(self.agent._running)

    def test_stopped_ignores_frames(self):
        self.agent.stop()
        result = self.agent.process_frame(self.frame)
        self.assertEqual(result, [])
        self.assertEqual(self.agent._total_frames, 0)

    def test_crowd_surge_detected(self):
        """Mock returns 2 persons, threshold=2 → crowd surge expected."""
        anomalies_found = []
        for _ in range(5):
            anomalies_found.extend(self.agent.process_frame(self.frame))
        types = [a.anomaly_type for a in anomalies_found]
        self.assertIn(AnomalyType.CROWD_SURGE, types)

    def test_detections_accumulate(self):
        for _ in range(3):
            self.agent.process_frame(self.frame)
        self.assertGreater(self.agent._total_detections, 0)

    def test_anomalies_accumulate(self):
        for _ in range(10):
            self.agent.process_frame(self.frame)
        self.assertGreater(self.agent._total_anomalies, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
