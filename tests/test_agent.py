"""Tests for WitnessAgent orchestrator"""
import pytest
import numpy as np
from datetime import datetime
from core.agent import WitnessAgent
from core.correlator import Correlator
from narration.narrator import Narrator, MockLLM
from evidence.packager import Packager
from models.schemas import CameraConfig, AnomalyType


def make_config(cam_id="cam-test"):
    return CameraConfig(
        camera_id=cam_id,
        name="Test Camera",
        source="mock://0",
        loitering_threshold=5,
        running_velocity_threshold=50.0,
        crowd_density_threshold=2,
    )


def make_agent(cam_id="cam-test", tmp_path=None):
    config = make_config(cam_id)
    narrator = Narrator(llm=MockLLM())
    packager = Packager(output_dir=str(tmp_path) if tmp_path else "/tmp/witnessai_test")
    correlator = Correlator()
    agent = WitnessAgent(
        config=config,
        narrator=narrator,
        packager=packager,
        correlator=correlator,
        mock=True,
    )
    agent.load()
    return agent


class TestWitnessAgent:
    def test_agent_loads(self, tmp_path):
        agent = make_agent(tmp_path=tmp_path)
        assert agent._running is True

    def test_process_frame_returns_list(self, tmp_path):
        agent = make_agent(tmp_path=tmp_path)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = agent.process_frame(frame)
        assert isinstance(result, list)

    def test_frame_count_increments(self, tmp_path):
        agent = make_agent(tmp_path=tmp_path)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        agent.process_frame(frame)
        assert agent._total_frames == 1

    def test_status_returns_agent_status(self, tmp_path):
        agent = make_agent(tmp_path=tmp_path)
        status = agent.status()
        assert status.camera_id == "cam-test"
        assert status.is_running is True

    def test_stop_sets_running_false(self, tmp_path):
        agent = make_agent(tmp_path=tmp_path)
        agent.stop()
        assert agent._running is False

    def test_stopped_agent_ignores_frames(self, tmp_path):
        agent = make_agent(tmp_path=tmp_path)
        agent.stop()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = agent.process_frame(frame)
        assert result == []
        assert agent._total_frames == 0

    def test_crowd_surge_detected_with_mock(self, tmp_path):
        """Mock detector returns 2 persons, threshold is 2 — should trigger."""
        agent = make_agent(tmp_path=tmp_path)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        anomalies_found = []
        for _ in range(5):
            anomalies = agent.process_frame(frame)
            anomalies_found.extend(anomalies)

        types = [a.anomaly_type for a in anomalies_found]
        assert AnomalyType.CROWD_SURGE in types

    def test_detections_accumulate(self, tmp_path):
        agent = make_agent(tmp_path=tmp_path)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            agent.process_frame(frame)
        assert agent._total_detections > 0

    def test_incident_created_on_anomaly(self, tmp_path):
        agent = make_agent(tmp_path=tmp_path)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(10):
            agent.process_frame(frame)
        # At least one incident should be active (crowd_surge threshold=2)
        total = agent._total_anomalies
        assert total > 0
