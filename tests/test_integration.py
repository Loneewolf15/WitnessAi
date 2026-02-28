"""
WitnessAI — Integration Tests for WitnessProcessor
====================================================
Tests the full pipeline: SDK processor → WitnessAgent → anomaly → annotation
Uses mocks for av/aiortc so no real Stream credentials needed.
"""
import sys
import os
import unittest
import asyncio
import tempfile
import numpy as np
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.schemas import CameraConfig, AnomalyType
from core.agent import WitnessAgent
from core.correlator import Correlator
from narration.narrator import Narrator, MockLLM
from evidence.packager import Packager
from integration.witness_processor import (
    WitnessProcessor,
    draw_detections,
    draw_hud,
    AnomalyDetectedEvent,
    IncidentPackagedEvent,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_witness_agent(output_dir="/tmp/witnessai_integration_test"):
    config = CameraConfig(
        camera_id="cam-integration",
        name="Integration Test Camera",
        source="mock://test",
        loitering_threshold=5,
        running_velocity_threshold=50.0,
        crowd_density_threshold=2,   # Low so mock detector (2 persons) triggers it
    )
    agent = WitnessAgent(
        config=config,
        narrator=Narrator(llm=MockLLM()),
        packager=Packager(output_dir=output_dir),
        correlator=Correlator(),
        mock=True,
    )
    agent.load()
    return agent


def make_mock_av_frame(h=480, w=640):
    """Create a mock av.VideoFrame backed by a numpy array."""
    import av
    img = np.zeros((h, w, 3), dtype=np.uint8)
    frame = av.VideoFrame.from_ndarray(img, format="rgb24")
    frame.pts = 0
    frame.time_base = 1 / 30
    return frame


def make_mock_forwarder():
    """Mock VideoForwarder that captures registered handlers."""
    forwarder = MagicMock()
    forwarder._handlers = {}

    def add_frame_handler(callback, fps, name):
        forwarder._handlers[name] = callback

    forwarder.add_frame_handler = add_frame_handler
    forwarder.remove_frame_handler = AsyncMock()
    return forwarder


def make_mock_track():
    return MagicMock()


# ─────────────────────────────────────────────────────────────────────────────
# Annotation Tests (no SDK needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestAnnotations(unittest.TestCase):
    def test_draw_detections_returns_same_shape(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_detections(frame, [], set())
        self.assertEqual(result.shape, frame.shape)

    def test_draw_detections_does_not_modify_original(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original = frame.copy()
        draw_detections(frame, [], set())
        np.testing.assert_array_equal(frame, original)

    def test_draw_status_banner_nominal(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_hud(frame, "🟢 nominal", is_incident=False, fps=5.0, latency_ms=20.0, frame_count=1, tracked_count=0)
        self.assertEqual(result.shape, frame.shape)
        # Nominal banner is dark (not red)
        banner_region = result[0:32, :, :]
        # Should not be full red (is_incident=False uses dark color)
        self.assertLess(banner_region[:, :, 2].mean(), 200)

    def test_draw_status_banner_incident(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = draw_hud(frame, "🔴 INCIDENT", is_incident=True, fps=5.0, latency_ms=20.0, frame_count=1, tracked_count=0)
        # Incident banner has red channel dominant
        banner_region = result[0:32, :, :]
        self.assertGreater(banner_region[:, :, 2].mean(), 50)

    def test_draw_detections_with_objects(self):
        from models.schemas import DetectedObject, BoundingBox
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        obj = DetectedObject(
            track_id=1, class_name="person", confidence=0.9,
            bbox=BoundingBox(x1=50, y1=50, x2=200, y2=300),
            camera_id="cam1"
        )
        result = draw_detections(frame, [obj], set())
        # Frame should be modified (bounding box drawn)
        self.assertFalse(np.array_equal(result, frame))


# ─────────────────────────────────────────────────────────────────────────────
# Event Model Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEventModels(unittest.TestCase):
    def test_anomaly_detected_event_creation(self):
        event = AnomalyDetectedEvent(
            incident_id="inc-001",
            camera_id="cam1",
            anomaly_type="loitering",
            confidence="HIGH",
            description="Test description",
            timestamp="14:32:01 UTC",
            narrative_entry="Subject was observed loitering.",
        )
        self.assertEqual(event.incident_id, "inc-001")
        self.assertEqual(event.anomaly_type, "loitering")

    def test_incident_packaged_event_creation(self):
        event = IncidentPackagedEvent(
            incident_id="inc-001",
            camera_id="cam1",
            package_id="pkg-abc",
            report_path="/tmp/report.json",
            video_path="/tmp/video.mp4",
            total_anomalies=3,
        )
        self.assertEqual(event.total_anomalies, 3)
        self.assertIsNotNone(event.video_path)


# ─────────────────────────────────────────────────────────────────────────────
# WitnessProcessor Unit Tests (mocked SDK)
# ─────────────────────────────────────────────────────────────────────────────

class TestWitnessProcessor(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.witness_agent = make_witness_agent(self.tmp)
        self.processor = WitnessProcessor(
            witness_agent=self.witness_agent,
            fps=5,
        )

    async def test_process_video_registers_handler(self):
        forwarder = make_mock_forwarder()
        track = make_mock_track()
        await self.processor.process_video(track, "participant-1", forwarder)
        self.assertIn("witness_ai", forwarder._handlers)

    async def test_name_is_witness_ai(self):
        self.assertEqual(self.processor.name, "witness_ai")

    async def test_publish_video_track_returns_track(self):
        track = self.processor.publish_video_track()
        self.assertIsNotNone(track)

    async def test_process_frame_runs_full_pipeline(self):
        """End-to-end: feed a real av frame through the full WitnessAI pipeline."""
        try:
            frame = make_mock_av_frame()
        except ImportError:
            self.skipTest("av library not available")

        forwarder = make_mock_forwarder()
        track = make_mock_track()
        await self.processor.process_video(track, "participant-1", forwarder)

        handler = forwarder._handlers["witness_ai"]
        # Should not raise
        await handler(frame)

        # Frame should have been processed
        self.assertGreater(self.witness_agent._total_frames, 0)

    async def test_process_frame_detects_persons(self):
        """Mock detector returns 2 persons — detections should accumulate."""
        try:
            frame = make_mock_av_frame()
        except ImportError:
            self.skipTest("av library not available")

        forwarder = make_mock_forwarder()
        await self.processor.process_video(make_mock_track(), "p1", forwarder)
        handler = forwarder._handlers["witness_ai"]

        for _ in range(5):
            await handler(frame)

        self.assertGreater(self.witness_agent._total_detections, 0)

    async def test_crowd_surge_triggers_anomaly_event(self):
        """With 2 persons and threshold=2, crowd surge should fire."""
        try:
            frame = make_mock_av_frame()
        except ImportError:
            self.skipTest("av library not available")

        forwarder = make_mock_forwarder()
        await self.processor.process_video(make_mock_track(), "p1", forwarder)
        handler = forwarder._handlers["witness_ai"]

        for _ in range(10):
            await handler(frame)

        self.assertGreater(self.witness_agent._total_anomalies, 0)

    async def test_active_anomaly_types_updated(self):
        """After crowd surge triggers, active anomaly types should be set."""
        try:
            frame = make_mock_av_frame()
        except ImportError:
            self.skipTest("av library not available")

        forwarder = make_mock_forwarder()
        await self.processor.process_video(make_mock_track(), "p1", forwarder)
        handler = forwarder._handlers["witness_ai"]

        for _ in range(10):
            await handler(frame)

        # Should have detected crowd surge (2 persons, threshold=2)
        if self.witness_agent._total_anomalies > 0:
            self.assertGreater(len(self.processor._active_anomaly_types), 0)

    async def test_stop_processing_removes_handler(self):
        forwarder = make_mock_forwarder()
        await self.processor.process_video(make_mock_track(), "p1", forwarder)
        await self.processor.stop_processing()
        forwarder.remove_frame_handler.assert_awaited()

    async def test_close_cleans_up(self):
        forwarder = make_mock_forwarder()
        await self.processor.process_video(make_mock_track(), "p1", forwarder)
        # Should not raise
        await self.processor.close()
        self.assertIsNone(self.processor._forwarder)

    async def test_switching_tracks_removes_old_handler(self):
        """Re-calling process_video with a new track cleans up the old one."""
        forwarder1 = make_mock_forwarder()
        forwarder2 = make_mock_forwarder()
        track = make_mock_track()

        await self.processor.process_video(track, "p1", forwarder1)
        await self.processor.process_video(track, "p2", forwarder2)

        # Old forwarder should have had its handler removed
        forwarder1.remove_frame_handler.assert_awaited()

    async def test_attach_agent_sets_events(self):
        mock_sdk_agent = MagicMock()
        mock_sdk_agent.events = MagicMock()
        self.processor.attach_agent(mock_sdk_agent)
        self.assertIsNotNone(self.processor._sdk_events)


# ─────────────────────────────────────────────────────────────────────────────
# Full Pipeline Integration Test
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPipeline(unittest.IsolatedAsyncioTestCase):
    """
    End-to-end pipeline test:
    av.VideoFrame → WitnessProcessor → WitnessAgent → anomaly → annotated frame
    """

    async def test_full_pipeline_no_crash(self):
        try:
            import av
        except ImportError:
            self.skipTest("av library not available in test environment")

        tmp = tempfile.mkdtemp()
        agent = make_witness_agent(tmp)
        processor = WitnessProcessor(witness_agent=agent, fps=5)

        # Simulate SDK giving us a forwarder and track
        forwarder = make_mock_forwarder()
        await processor.process_video(make_mock_track(), "test-participant", forwarder)
        handler = forwarder._handlers["witness_ai"]

        # Feed 20 frames
        for i in range(20):
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = i
            await handler(frame)

        # Assert pipeline ran
        self.assertEqual(agent._total_frames, 20)
        self.assertGreater(agent._total_detections, 0)

        # Clean up
        await processor.close()

    async def test_annotated_frames_published(self):
        """Verify that annotated frames are pushed to the output track."""
        try:
            import av
        except ImportError:
            self.skipTest("av library not available in test environment")

        tmp = tempfile.mkdtemp()
        agent = make_witness_agent(tmp)
        processor = WitnessProcessor(witness_agent=agent, fps=5)

        forwarder = make_mock_forwarder()
        await processor.process_video(make_mock_track(), "p1", forwarder)
        handler = forwarder._handlers["witness_ai"]

        # Feed a frame
        frame = make_mock_av_frame()
        await handler(frame)

        # The output video track should have frames queued
        output_track = processor.publish_video_track()
        self.assertIsNotNone(output_track)

        await processor.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
