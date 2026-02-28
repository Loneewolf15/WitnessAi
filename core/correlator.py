"""WitnessAI - Multi-Camera Event Correlator"""
from __future__ import annotations
import uuid
from datetime import datetime, timedelta
from collections import defaultdict
import logging; logger = logging.getLogger(__name__)
from models.schemas import AnomalyEvent, Incident, IncidentStatus


class CrossCameraMatch(object):
    """Represents a correlated observation of the same entity across cameras."""

    def __init__(self, entity_id: str):
        self.entity_id = entity_id  # Synthetic ID for correlated subject
        self.observations: list[dict] = []  # [{camera_id, track_id, timestamp, bbox_center}]
        self.created_at = datetime.utcnow()

    def add_observation(
        self, camera_id: str, track_id: int, timestamp: datetime, cx: float, cy: float
    ) -> None:
        self.observations.append({
            "camera_id": camera_id,
            "track_id": track_id,
            "timestamp": timestamp.isoformat(),
            "center": (cx, cy),
        })

    def cameras_seen(self) -> set[str]:
        return {o["camera_id"] for o in self.observations}

    def timeline(self) -> list[dict]:
        return sorted(self.observations, key=lambda x: x["timestamp"])


class Correlator:
    """
    Multi-feed event correlator.
    Detects when the same anomaly pattern occurs across cameras
    and links incidents into a unified correlated timeline.
    """

    TIME_WINDOW_SECONDS = 120  # Events within 2 minutes are candidates for correlation
    MAX_CROSS_MATCHES = 500

    def __init__(self):
        # incident_id -> Incident
        self._incidents: dict[str, Incident] = {}
        # camera_id -> list of recent AnomalyEvent
        self._recent_events: dict[str, list[AnomalyEvent]] = defaultdict(list)
        # Correlated cross-camera matches
        self._cross_matches: list[CrossCameraMatch] = []

    def register_incident(self, incident: Incident) -> None:
        self._incidents[incident.incident_id] = incident
        logger.info(
            f"Correlator: registered incident {incident.incident_id} "
            f"from camera {incident.camera_id}"
        )

    def ingest_anomaly(self, event: AnomalyEvent) -> list[CrossCameraMatch]:
        """
        Ingest a new anomaly event and check for cross-camera correlations.
        Returns any new cross-camera matches found.
        """
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self.TIME_WINDOW_SECONDS)

        # Clean old events
        for cam_id in list(self._recent_events.keys()):
            self._recent_events[cam_id] = [
                e for e in self._recent_events[cam_id] if e.timestamp >= cutoff
            ]

        self._recent_events[event.camera_id].append(event)

        # Look for same anomaly type from different cameras
        new_matches = []
        for other_cam, events in self._recent_events.items():
            if other_cam == event.camera_id:
                continue
            matching = [
                e for e in events if e.anomaly_type == event.anomaly_type
            ]
            if matching:
                match = CrossCameraMatch(entity_id=str(uuid.uuid4()))
                match.add_observation(
                    camera_id=event.camera_id,
                    track_id=event.involved_track_ids[0] if event.involved_track_ids else -1,
                    timestamp=event.timestamp,
                    cx=0.0,
                    cy=0.0,
                )
                for m_event in matching:
                    match.add_observation(
                        camera_id=other_cam,
                        track_id=m_event.involved_track_ids[0] if m_event.involved_track_ids else -1,
                        timestamp=m_event.timestamp,
                        cx=0.0,
                        cy=0.0,
                    )
                self._cross_matches.append(match)
                new_matches.append(match)
                logger.info(
                    f"Cross-camera correlation: {event.anomaly_type.value} "
                    f"on {event.camera_id} ↔ {other_cam}"
                )

        # Trim cross-matches list
        if len(self._cross_matches) > self.MAX_CROSS_MATCHES:
            self._cross_matches = self._cross_matches[-self.MAX_CROSS_MATCHES:]

        return new_matches

    def unified_timeline(self) -> list[dict]:
        """Return all incidents sorted by time as a unified cross-camera timeline."""
        events = []
        for incident in self._incidents.values():
            for anomaly in incident.anomaly_events:
                events.append({
                    "timestamp": anomaly.timestamp.isoformat(),
                    "camera_id": anomaly.camera_id,
                    "incident_id": incident.incident_id,
                    "anomaly_type": anomaly.anomaly_type.value,
                    "description": anomaly.description,
                    "confidence": anomaly.confidence.value,
                })
        return sorted(events, key=lambda x: x["timestamp"])

    def cross_matches(self) -> list[CrossCameraMatch]:
        return self._cross_matches.copy()

    def summary(self) -> dict:
        return {
            "total_incidents": len(self._incidents),
            "total_cross_matches": len(self._cross_matches),
            "cameras_tracked": list(self._recent_events.keys()),
        }
