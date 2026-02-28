"""Tests for Pydantic schemas"""
import pytest
from datetime import datetime
from models.schemas import (
    BoundingBox, DetectedObject, FrameDetections, AnomalyEvent,
    AnomalyType, ConfidenceLevel, IncidentNarrative, NarrativeEntry,
    EvidencePackage, Incident, IncidentStatus
)


class TestBoundingBox:
    def test_width_height_computed(self):
        box = BoundingBox(x1=10, y1=20, x2=110, y2=220)
        assert box.width == 100
        assert box.height == 200

    def test_center_computed(self):
        box = BoundingBox(x1=0, y1=0, x2=100, y2=200)
        assert box.center_x == 50
        assert box.center_y == 100

    def test_area_computed(self):
        box = BoundingBox(x1=0, y1=0, x2=10, y2=20)
        assert box.area == 200


class TestDetectedObject:
    def test_speed_computed(self):
        obj = DetectedObject(
            track_id=1, class_name="person", confidence=0.9,
            bbox=BoundingBox(x1=0, y1=0, x2=50, y2=100),
            velocity_x=3.0, velocity_y=4.0
        )
        assert obj.speed == pytest.approx(5.0)

    def test_zero_velocity_zero_speed(self):
        obj = DetectedObject(
            track_id=1, class_name="person", confidence=0.9,
            bbox=BoundingBox(x1=0, y1=0, x2=50, y2=100),
        )
        assert obj.speed == 0.0


class TestIncidentNarrative:
    def test_full_text_format(self):
        narrative = IncidentNarrative(
            incident_id="inc-1",
            camera_id="cam1",
            started_at=datetime.utcnow(),
        )
        narrative.entries.append(NarrativeEntry(
            timestamp=datetime(2026, 2, 26, 14, 32, 1),
            text="Subject entered the frame.",
            camera_id="cam1",
        ))
        full = narrative.full_text()
        assert "14:32:01" in full
        assert "Subject entered" in full


class TestAnomalyEvent:
    def test_anomaly_event_creation(self):
        event = AnomalyEvent(
            id="ev-001",
            camera_id="cam1",
            anomaly_type=AnomalyType.RUNNING,
            confidence=ConfidenceLevel.HIGH,
            description="Running detected",
        )
        assert event.anomaly_type == AnomalyType.RUNNING
        assert event.confidence == ConfidenceLevel.HIGH

    def test_all_anomaly_types_valid(self):
        for atype in AnomalyType:
            event = AnomalyEvent(
                id="ev", camera_id="cam", anomaly_type=atype,
                confidence=ConfidenceLevel.LOW, description="test"
            )
            assert event.anomaly_type == atype


class TestIncident:
    def test_default_status_detecting(self):
        incident = Incident(incident_id="inc-1", camera_id="cam1")
        assert incident.status == IncidentStatus.DETECTING

    def test_status_transitions(self):
        incident = Incident(incident_id="inc-1", camera_id="cam1")
        incident.status = IncidentStatus.PACKAGED
        assert incident.status == IncidentStatus.PACKAGED
