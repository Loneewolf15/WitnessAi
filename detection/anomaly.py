"""WitnessAI - Behavioral Anomaly Detection Rules Engine"""
from __future__ import annotations
import uuid
from datetime import datetime
import logging; logger = logging.getLogger(__name__)
from models.schemas import (
    AnomalyEvent, AnomalyType, ConfidenceLevel, FrameDetections, DetectedObject
)
from detection.tracker import Tracker


class AnomalyConfig:
    def __init__(
        self,
        loitering_threshold: int = 30,
        running_velocity_threshold: float = 150.0,
        crowd_density_threshold: int = 8,
        object_abandonment_seconds: int = 20,
        fall_aspect_ratio_threshold: float = 2.5,  # width/height > this = prone
    ):
        self.loitering_threshold = loitering_threshold
        self.running_velocity_threshold = running_velocity_threshold
        self.crowd_density_threshold = crowd_density_threshold
        self.object_abandonment_seconds = object_abandonment_seconds
        self.fall_aspect_ratio_threshold = fall_aspect_ratio_threshold


class AnomalyDetector:
    """
    Rules-based behavioral anomaly engine.
    Evaluates tracked detections against configured thresholds.
    """

    def __init__(self, camera_id: str = "default", config: AnomalyConfig | None = None):
        self.camera_id = camera_id
        self.config = config or AnomalyConfig()
        self._alerted_tracks: set[str] = set()  # "track_id:anomaly_type" dedup keys

    def evaluate(
        self, frame: FrameDetections, tracker: Tracker
    ) -> list[AnomalyEvent]:
        """Evaluate all anomaly rules for the current frame."""
        events: list[AnomalyEvent] = []
        tracks = tracker.active_tracks
        persons = [o for o in frame.objects if o.class_name == "person"]

        events.extend(self._check_loitering(tracks))
        events.extend(self._check_running(persons))
        events.extend(self._check_crowd_surge(persons))
        events.extend(self._check_falls(persons, tracks))

        return events

    def _dedup_key(self, track_id: int, anomaly_type: AnomalyType) -> str:
        return f"{track_id}:{anomaly_type.value}"

    def _check_loitering(self, tracks: dict) -> list[AnomalyEvent]:
        events = []
        for tid, track in tracks.items():
            if track.class_name != "person":
                continue
            stationary_secs = track.stationary_frames / max(1, 15)  # assume 15fps
            if stationary_secs >= self.config.loitering_threshold:
                key = self._dedup_key(tid, AnomalyType.LOITERING)
                if key not in self._alerted_tracks:
                    self._alerted_tracks.add(key)
                    events.append(AnomalyEvent(
                        id=str(uuid.uuid4()),
                        camera_id=self.camera_id,
                        anomaly_type=AnomalyType.LOITERING,
                        confidence=ConfidenceLevel.HIGH,
                        description=(
                            f"Person (ID:{tid}) has been stationary for "
                            f"{stationary_secs:.0f} seconds in monitored zone."
                        ),
                        involved_track_ids=[tid],
                        metadata={"stationary_seconds": round(stationary_secs, 1)},
                    ))
        return events

    def _check_running(self, persons: list[DetectedObject]) -> list[AnomalyEvent]:
        events = []
        for obj in persons:
            if obj.speed >= self.config.running_velocity_threshold:
                key = self._dedup_key(obj.track_id, AnomalyType.RUNNING)
                if key not in self._alerted_tracks:
                    self._alerted_tracks.add(key)
                    events.append(AnomalyEvent(
                        id=str(uuid.uuid4()),
                        camera_id=self.camera_id,
                        anomaly_type=AnomalyType.RUNNING,
                        confidence=ConfidenceLevel.MEDIUM,
                        description=(
                            f"Person (ID:{obj.track_id}) moving rapidly at "
                            f"{obj.speed:.0f} px/s — possible panic or pursuit."
                        ),
                        involved_track_ids=[obj.track_id],
                        metadata={"speed_px_per_sec": round(obj.speed, 1)},
                    ))
        return events

    def _check_crowd_surge(self, persons: list[DetectedObject]) -> list[AnomalyEvent]:
        events = []
        count = len(persons)
        if count >= self.config.crowd_density_threshold:
            key = f"crowd:{count}"
            if key not in self._alerted_tracks:
                self._alerted_tracks.add(key)
                events.append(AnomalyEvent(
                    id=str(uuid.uuid4()),
                    camera_id=self.camera_id,
                    anomaly_type=AnomalyType.CROWD_SURGE,
                    confidence=ConfidenceLevel.HIGH,
                    description=(
                        f"Crowd surge detected — {count} persons in frame "
                        f"(threshold: {self.config.crowd_density_threshold})."
                    ),
                    involved_track_ids=[p.track_id for p in persons],
                    metadata={"person_count": count},
                ))
        return events

    def _check_falls(self, persons: list[DetectedObject], tracks: dict) -> list[AnomalyEvent]:
        events = []
        for obj in persons:
            bbox = obj.bbox
            if bbox.height == 0:
                continue
            aspect = bbox.width / bbox.height
            if aspect >= self.config.fall_aspect_ratio_threshold:
                key = self._dedup_key(obj.track_id, AnomalyType.FALL_DETECTED)
                if key not in self._alerted_tracks:
                    self._alerted_tracks.add(key)
                    events.append(AnomalyEvent(
                        id=str(uuid.uuid4()),
                        camera_id=self.camera_id,
                        anomaly_type=AnomalyType.FALL_DETECTED,
                        confidence=ConfidenceLevel.HIGH,
                        description=(
                            f"Possible fall detected — Person (ID:{obj.track_id}) "
                            f"appears to be prone (aspect ratio: {aspect:.2f})."
                        ),
                        involved_track_ids=[obj.track_id],
                        metadata={"aspect_ratio": round(aspect, 2)},
                    ))
        return events

    def reset_alerts(self) -> None:
        """Clear dedup state (e.g., between incidents)."""
        self._alerted_tracks.clear()
