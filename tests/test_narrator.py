"""Tests for LLM Narrator"""
import pytest
from datetime import datetime
from narration.narrator import Narrator, MockLLM, build_llm
from models.schemas import AnomalyEvent, AnomalyType, ConfidenceLevel


def make_anomaly(anomaly_type=AnomalyType.LOITERING, cam="cam1"):
    return AnomalyEvent(
        id="test-id-123",
        camera_id=cam,
        anomaly_type=anomaly_type,
        confidence=ConfidenceLevel.HIGH,
        description="Test anomaly description",
        involved_track_ids=[1],
        metadata={"test_key": "test_value"},
    )


class TestMockLLM:
    def test_generates_string(self):
        llm = MockLLM()
        result = llm.generate("test prompt")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_loitering_prompt(self):
        llm = MockLLM()
        result = llm.generate("This involves loitering behavior")
        assert "stationary" in result.lower() or "zone" in result.lower()

    def test_running_prompt(self):
        llm = MockLLM()
        result = llm.generate("A person is running rapidly")
        assert len(result) > 0

    def test_crowd_prompt(self):
        llm = MockLLM()
        result = llm.generate("crowd surge detected")
        assert len(result) > 0

    def test_fall_prompt(self):
        llm = MockLLM()
        result = llm.generate("fall detected person prone")
        assert len(result) > 0


class TestBuildLLM:
    def test_mock_provider_returns_mock_llm(self):
        llm = build_llm("mock")
        assert isinstance(llm, MockLLM)

    def test_unknown_provider_returns_mock_llm(self):
        llm = build_llm("unknown_provider")
        assert isinstance(llm, MockLLM)


class TestNarrator:
    def test_narrate_creates_entry(self):
        narrator = Narrator(llm=MockLLM())
        anomaly = make_anomaly()
        entry = narrator.narrate_anomaly("incident-001", anomaly)
        assert entry is not None
        assert len(entry.text) > 0

    def test_narrate_links_anomaly_id(self):
        narrator = Narrator(llm=MockLLM())
        anomaly = make_anomaly()
        entry = narrator.narrate_anomaly("incident-001", anomaly)
        assert entry.anomaly_id == anomaly.id

    def test_get_narrative_returns_none_for_unknown(self):
        narrator = Narrator(llm=MockLLM())
        assert narrator.get_narrative("does-not-exist") is None

    def test_get_narrative_after_narrate(self):
        narrator = Narrator(llm=MockLLM())
        anomaly = make_anomaly()
        narrator.narrate_anomaly("incident-abc", anomaly)
        narrative = narrator.get_narrative("incident-abc")
        assert narrative is not None
        assert narrative.incident_id == "incident-abc"

    def test_multiple_entries_accumulate(self):
        narrator = Narrator(llm=MockLLM())
        anomaly1 = make_anomaly(AnomalyType.LOITERING)
        anomaly2 = make_anomaly(AnomalyType.RUNNING)
        narrator.narrate_anomaly("incident-multi", anomaly1)
        narrator.narrate_anomaly("incident-multi", anomaly2)
        narrative = narrator.get_narrative("incident-multi")
        assert len(narrative.entries) == 2

    def test_full_text_is_timestamped(self):
        narrator = Narrator(llm=MockLLM())
        anomaly = make_anomaly()
        narrator.narrate_anomaly("inc-ts", anomaly)
        narrative = narrator.get_narrative("inc-ts")
        full = narrative.full_text()
        assert "—" in full  # Timestamp separator

    def test_llm_failure_falls_back_to_description(self):
        class FailingLLM:
            def generate(self, prompt): raise RuntimeError("API down")

        narrator = Narrator(llm=FailingLLM())
        anomaly = make_anomaly()
        entry = narrator.narrate_anomaly("inc-fail", anomaly)
        assert "[AUTO]" in entry.text

    def test_separate_incidents_isolated(self):
        narrator = Narrator(llm=MockLLM())
        a1 = make_anomaly(AnomalyType.LOITERING)
        a2 = make_anomaly(AnomalyType.RUNNING)
        narrator.narrate_anomaly("incident-A", a1)
        narrator.narrate_anomaly("incident-B", a2)
        assert len(narrator.get_narrative("incident-A").entries) == 1
        assert len(narrator.get_narrative("incident-B").entries) == 1
