"""Tests for Multi-Camera Correlator"""
import pytest
from datetime import datetime
from core.correlator import Correlator
from models.schemas import AnomalyEvent, AnomalyType, ConfidenceLevel, Incident, IncidentStatus


def make_anomaly(cam="cam1", atype=AnomalyType.LOITERING, track_id=1):
    return AnomalyEvent(
        id=f"ev-{cam}-{atype.value}",
        camera_id=cam,
        anomaly_type=atype,
        confidence=ConfidenceLevel.HIGH,
        description=f"{atype.value} on {cam}",
        involved_track_ids=[track_id],
    )


def make_incident(incident_id="inc-001", cam="cam1"):
    return Incident(
        incident_id=incident_id,
        camera_id=cam,
        status=IncidentStatus.CONFIRMED,
        anomaly_events=[make_anomaly(cam=cam)],
    )


class TestCorrelator:
    def test_register_incident(self):
        correlator = Correlator()
        incident = make_incident()
        correlator.register_incident(incident)
        assert len(correlator._incidents) == 1

    def test_unified_timeline_empty(self):
        correlator = Correlator()
        timeline = correlator.unified_timeline()
        assert timeline == []

    def test_unified_timeline_with_incidents(self):
        correlator = Correlator()
        correlator.register_incident(make_incident("inc-1", "cam1"))
        correlator.register_incident(make_incident("inc-2", "cam2"))
        timeline = correlator.unified_timeline()
        assert len(timeline) == 2

    def test_cross_camera_correlation_detected(self):
        correlator = Correlator()
        ev1 = make_anomaly(cam="cam1", atype=AnomalyType.RUNNING)
        ev2 = make_anomaly(cam="cam2", atype=AnomalyType.RUNNING)

        correlator.ingest_anomaly(ev1)
        matches = correlator.ingest_anomaly(ev2)
        assert len(matches) >= 1

    def test_no_cross_camera_same_camera(self):
        correlator = Correlator()
        ev1 = make_anomaly(cam="cam1", atype=AnomalyType.RUNNING)
        ev2 = make_anomaly(cam="cam1", atype=AnomalyType.RUNNING)
        correlator.ingest_anomaly(ev1)
        matches = correlator.ingest_anomaly(ev2)
        assert len(matches) == 0

    def test_no_cross_camera_different_types(self):
        correlator = Correlator()
        ev1 = make_anomaly(cam="cam1", atype=AnomalyType.RUNNING)
        ev2 = make_anomaly(cam="cam2", atype=AnomalyType.LOITERING)
        correlator.ingest_anomaly(ev1)
        matches = correlator.ingest_anomaly(ev2)
        assert len(matches) == 0

    def test_summary_structure(self):
        correlator = Correlator()
        summary = correlator.summary()
        assert "total_incidents" in summary
        assert "total_cross_matches" in summary
        assert "cameras_tracked" in summary

    def test_cross_match_has_multiple_cameras(self):
        correlator = Correlator()
        ev1 = make_anomaly(cam="cam1", atype=AnomalyType.CROWD_SURGE)
        ev2 = make_anomaly(cam="cam2", atype=AnomalyType.CROWD_SURGE)
        correlator.ingest_anomaly(ev1)
        matches = correlator.ingest_anomaly(ev2)
        assert len(matches) > 0
        assert len(matches[0].cameras_seen()) == 2
