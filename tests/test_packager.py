"""Tests for Evidence Packager"""
import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
import numpy as np
from evidence.packager import Packager
from evidence.buffer import BufferedFrame
from models.schemas import AnomalyEvent, AnomalyType, ConfidenceLevel, IncidentNarrative, NarrativeEntry


def make_buffered_frame(n=1):
    return BufferedFrame(
        frame=np.zeros((100, 100, 3), dtype=np.uint8),
        timestamp=datetime.utcnow(),
        frame_number=n,
    )


def make_anomaly():
    return AnomalyEvent(
        id="anomaly-001",
        camera_id="cam1",
        anomaly_type=AnomalyType.LOITERING,
        confidence=ConfidenceLevel.HIGH,
        description="Test loitering event",
        involved_track_ids=[1],
        metadata={"stationary_seconds": 35.0},
    )


def make_narrative():
    narrative = IncidentNarrative(
        incident_id="incident-001",
        camera_id="cam1",
        started_at=datetime.utcnow(),
    )
    narrative.entries.append(NarrativeEntry(
        timestamp=datetime.utcnow(),
        text="Subject loitered for extended period.",
        anomaly_id="anomaly-001",
        camera_id="cam1",
    ))
    return narrative


class TestPackager:
    def test_build_creates_package(self, tmp_path):
        packager = Packager(output_dir=str(tmp_path))
        pre = [make_buffered_frame(i) for i in range(3)]
        post = [make_buffered_frame(i) for i in range(3, 6)]
        package = packager.build(
            incident_id="incident-001",
            camera_id="cam1",
            pre_crime_frames=pre,
            post_crime_frames=post,
            anomaly_events=[make_anomaly()],
            narrative=make_narrative(),
        )
        assert package is not None
        assert package.incident_id == "incident-001"
        assert package.camera_id == "cam1"

    def test_report_file_created(self, tmp_path):
        packager = Packager(output_dir=str(tmp_path))
        package = packager.build(
            incident_id="inc-report",
            camera_id="cam1",
            pre_crime_frames=[make_buffered_frame()],
            post_crime_frames=[],
            anomaly_events=[make_anomaly()],
        )
        assert package.report_path is not None
        assert Path(package.report_path).exists()

    def test_report_json_valid(self, tmp_path):
        packager = Packager(output_dir=str(tmp_path))
        package = packager.build(
            incident_id="inc-json",
            camera_id="cam1",
            pre_crime_frames=[make_buffered_frame()],
            post_crime_frames=[],
            anomaly_events=[make_anomaly()],
            narrative=make_narrative(),
        )
        with open(package.report_path) as f:
            report = json.load(f)
        assert report["incident_id"] == "inc-json"
        assert report["anomaly_count"] == 1
        assert "narrative" in report

    def test_narrative_in_report(self, tmp_path):
        packager = Packager(output_dir=str(tmp_path))
        package = packager.build(
            incident_id="inc-narr",
            camera_id="cam1",
            pre_crime_frames=[],
            post_crime_frames=[],
            anomaly_events=[make_anomaly()],
            narrative=make_narrative(),
        )
        with open(package.report_path) as f:
            report = json.load(f)
        assert len(report["narrative"]) > 0

    def test_list_packages(self, tmp_path):
        packager = Packager(output_dir=str(tmp_path))
        packager.build("inc-A", "cam1", [], [], [make_anomaly()])
        packager.build("inc-B", "cam1", [], [], [make_anomaly()])
        packages = packager.list_packages()
        assert "inc-A" in packages
        assert "inc-B" in packages

    def test_package_id_unique(self, tmp_path):
        packager = Packager(output_dir=str(tmp_path))
        p1 = packager.build("inc-u1", "cam1", [], [], [make_anomaly()])
        p2 = packager.build("inc-u2", "cam1", [], [], [make_anomaly()])
        assert p1.package_id != p2.package_id

    def test_empty_frames_no_video(self, tmp_path):
        packager = Packager(output_dir=str(tmp_path))
        package = packager.build("inc-novid", "cam1", [], [], [make_anomaly()])
        assert package.video_path is None
