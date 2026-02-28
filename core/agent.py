"""WitnessAI - Main Orchestration Agent"""
from __future__ import annotations
import uuid
import asyncio
from datetime import datetime
import logging; logger = logging.getLogger(__name__)
import numpy as np

from models.schemas import (
    AnomalyEvent, Incident, IncidentStatus, CameraConfig,
    AgentStatus, WebSocketMessage
)
from detection.detector import Detector
from detection.tracker import Tracker
from detection.anomaly import AnomalyDetector, AnomalyConfig
from narration.narrator import Narrator
from evidence.buffer import RollingBuffer
from evidence.packager import Packager
from core.correlator import Correlator


class WitnessAgent:
    """
    Core orchestration agent for a single camera feed.
    Coordinates detection → tracking → anomaly → narration → evidence pipeline.
    """

    POST_CRIME_FRAMES = 450  # ~30s at 15fps

    def __init__(
        self,
        config: CameraConfig,
        narrator: Narrator,
        packager: Packager,
        correlator: Correlator,
        ws_broadcast=None,  # async callable(WebSocketMessage)
        mock: bool = False,
    ):
        self.config = config
        self.camera_id = config.camera_id
        self._narrator = narrator
        self._packager = packager
        self._correlator = correlator
        self._ws_broadcast = ws_broadcast
        self._mock = mock

        # Sub-components
        self._detector = Detector(mock=mock, camera_id=self.camera_id)
        self._tracker = Tracker(camera_id=self.camera_id)
        self._anomaly_detector = AnomalyDetector(
            camera_id=self.camera_id,
            config=AnomalyConfig(
                loitering_threshold=config.loitering_threshold,
                running_velocity_threshold=config.running_velocity_threshold,
                crowd_density_threshold=config.crowd_density_threshold,
                object_abandonment_seconds=config.object_abandonment_seconds,
            ),
        )
        self._buffer = RollingBuffer(duration_seconds=30)

        # State
        self._running = False
        self._total_frames = 0
        self._total_detections = 0
        self._total_anomalies = 0
        self._active_incidents: dict[str, Incident] = {}

        # Post-crime capture state
        self._capturing_post: dict[str, list] = {}  # incident_id -> post-crime frames

    def load(self) -> None:
        """Initialize detector. Must be called before process_frame()."""
        self._detector.load()
        self._running = True
        logger.info(f"[{self.camera_id}] WitnessAgent initialized")

    def process_frame(self, frame: np.ndarray) -> list[AnomalyEvent]:
        """
        Process a single video frame through the full pipeline.
        Returns any anomaly events detected in this frame.
        """
        if not self._running:
            return []

        now = datetime.utcnow()
        self._total_frames += 1

        # Buffer the raw frame
        self._buffer.push(frame, now, self._total_frames)

        # --- Post-crime capture ---
        for inc_id in list(self._capturing_post.keys()):
            self._capturing_post[inc_id].append(
                self._buffer.snapshot()[-1] if self._buffer.size else None
            )
            if len(self._capturing_post[inc_id]) >= self.POST_CRIME_FRAMES:
                self._finalize_incident(inc_id)

        # --- Detection + Tracking ---
        detections = self._detector.detect(frame)
        tracked = self._tracker.update(detections)
        self._total_detections += len(tracked.objects)

        # --- Anomaly Detection ---
        anomalies = self._anomaly_detector.evaluate(tracked, self._tracker)
        self._total_anomalies += len(anomalies)

        for anomaly in anomalies:
            self._handle_anomaly(anomaly, now)

        return anomalies

    def _handle_anomaly(self, anomaly: AnomalyEvent, now: datetime) -> None:
        """Create/update an incident and trigger narration."""
        # Create or find active incident
        incident_id = self._get_or_create_incident(anomaly)
        incident = self._active_incidents[incident_id]
        incident.anomaly_events.append(anomaly)
        incident.updated_at = now
        incident.status = IncidentStatus.CONFIRMED

        # Correlate across cameras
        self._correlator.register_incident(incident)
        self._correlator.ingest_anomaly(anomaly)

        # Generate narrative
        entry = self._narrator.narrate_anomaly(incident_id, anomaly)

        # Start post-crime capture
        if incident_id not in self._capturing_post:
            self._capturing_post[incident_id] = []
            logger.info(
                f"[{self.camera_id}] Incident {incident_id} confirmed — "
                f"capturing post-crime footage"
            )

        # Broadcast to WebSocket
        if self._ws_broadcast:
            msg = WebSocketMessage(
                event_type="anomaly",
                camera_id=self.camera_id,
                payload={
                    "incident_id": incident_id,
                    "anomaly_type": anomaly.anomaly_type.value,
                    "confidence": anomaly.confidence.value,
                    "description": anomaly.description,
                    "narrative_entry": entry.text,
                },
            )
            try:
                asyncio.get_event_loop().create_task(self._ws_broadcast(msg))
            except RuntimeError:
                pass  # No event loop in sync context

    def _get_or_create_incident(self, anomaly: AnomalyEvent) -> str:
        """Find an open incident or create a new one for this anomaly."""
        # Simple strategy: one active incident per camera at a time
        if self._active_incidents:
            return list(self._active_incidents.keys())[-1]

        incident_id = str(uuid.uuid4())
        incident = Incident(
            incident_id=incident_id,
            camera_id=self.camera_id,
            status=IncidentStatus.DETECTING,
        )
        self._active_incidents[incident_id] = incident
        return incident_id

    def _finalize_incident(self, incident_id: str) -> None:
        """Package and close an incident after post-crime capture."""
        incident = self._active_incidents.get(incident_id)
        if not incident:
            return

        post_frames = [f for f in self._capturing_post.pop(incident_id, []) if f]
        pre_frames = self._buffer.snapshot()
        narrative = self._narrator.get_narrative(incident_id)

        package = self._packager.build(
            incident_id=incident_id,
            camera_id=self.camera_id,
            pre_crime_frames=pre_frames,
            post_crime_frames=post_frames,
            anomaly_events=incident.anomaly_events,
            narrative=narrative,
        )

        incident.evidence_package = package
        incident.status = IncidentStatus.PACKAGED
        incident.updated_at = datetime.utcnow()

        logger.info(
            f"[{self.camera_id}] Incident {incident_id} packaged "
            f"({len(incident.anomaly_events)} anomalies)"
        )

        del self._active_incidents[incident_id]

    def status(self) -> AgentStatus:
        return AgentStatus(
            camera_id=self.camera_id,
            is_running=self._running,
            total_frames=self._total_frames,
            total_detections=self._total_detections,
            total_anomalies=self._total_anomalies,
            active_incidents=len(self._active_incidents),
        )

    def stop(self) -> None:
        self._running = False
        logger.info(f"[{self.camera_id}] Agent stopped")
