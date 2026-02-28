"""WitnessAI - Evidence Package Builder"""
from __future__ import annotations
import os
import uuid
import json
from datetime import datetime
from pathlib import Path
import logging; logger = logging.getLogger(__name__)
from models.schemas import AnomalyEvent, EvidencePackage, IncidentNarrative
from evidence.buffer import BufferedFrame


class Packager:
    """
    Assembles complete evidence packages:
    - Pre/post-crime video clip (MP4 via OpenCV)
    - JSON incident report
    - AI narrative text
    """

    def __init__(self, output_dir: str = "./evidence_packages"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build(
        self,
        incident_id: str,
        camera_id: str,
        pre_crime_frames: list[BufferedFrame],
        post_crime_frames: list[BufferedFrame],
        anomaly_events: list[AnomalyEvent],
        narrative: IncidentNarrative | None = None,
    ) -> EvidencePackage:
        """
        Build and persist a complete evidence package.

        Returns an EvidencePackage with paths to saved artifacts.
        """
        package_id = str(uuid.uuid4())
        incident_dir = self.output_dir / incident_id
        incident_dir.mkdir(parents=True, exist_ok=True)

        video_path = self._write_video(
            incident_dir, package_id, pre_crime_frames + post_crime_frames
        )
        report_path = self._write_report(
            incident_dir, package_id, incident_id, camera_id,
            anomaly_events, narrative
        )

        package = EvidencePackage(
            package_id=package_id,
            incident_id=incident_id,
            camera_id=camera_id,
            video_path=str(video_path) if video_path else None,
            report_path=str(report_path),
            narrative=narrative,
            anomaly_events=anomaly_events,
            metadata={
                "pre_crime_frames": len(pre_crime_frames),
                "post_crime_frames": len(post_crime_frames),
                "total_anomalies": len(anomaly_events),
            },
        )

        logger.info(
            f"[{camera_id}] Evidence package {package_id} created → {incident_dir}"
        )
        return package

    def _write_video(
        self,
        incident_dir: Path,
        package_id: str,
        frames: list[BufferedFrame],
    ) -> Path | None:
        """Write frames to an MP4 file using OpenCV."""
        if not frames:
            return None

        try:
            import cv2
            video_path = incident_dir / f"{package_id}.mp4"
            h, w = frames[0].frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(video_path), fourcc, 15.0, (w, h))
            for bf in frames:
                out.write(bf.frame)
            out.release()
            logger.debug(f"Video written: {video_path} ({len(frames)} frames)")
            return video_path
        except Exception as e:
            logger.warning(f"Video write skipped (OpenCV unavailable or error): {e}")
            return None

    def _write_report(
        self,
        incident_dir: Path,
        package_id: str,
        incident_id: str,
        camera_id: str,
        anomaly_events: list[AnomalyEvent],
        narrative: IncidentNarrative | None,
    ) -> Path:
        """Write JSON incident report."""
        report = {
            "report_version": "1.0",
            "generated_by": "WitnessAI",
            "package_id": package_id,
            "incident_id": incident_id,
            "camera_id": camera_id,
            "generated_at": datetime.utcnow().isoformat(),
            "anomaly_count": len(anomaly_events),
            "anomalies": [
                {
                    "id": e.id,
                    "type": e.anomaly_type.value,
                    "confidence": e.confidence.value,
                    "timestamp": e.timestamp.isoformat(),
                    "description": e.description,
                    "involved_track_ids": e.involved_track_ids,
                    "metadata": e.metadata,
                }
                for e in anomaly_events
            ],
            "narrative": narrative.full_text() if narrative else "",
            "narrative_entries": [
                {
                    "timestamp": entry.timestamp.isoformat(),
                    "text": entry.text,
                    "anomaly_id": entry.anomaly_id,
                }
                for entry in (narrative.entries if narrative else [])
            ],
        }

        report_path = incident_dir / f"{package_id}_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.debug(f"Report written: {report_path}")
        return report_path

    def list_packages(self) -> list[str]:
        """List all incident IDs with saved packages."""
        return [d.name for d in self.output_dir.iterdir() if d.is_dir()]
