"""
WitnessAI — Incident Manager
Handles the full lifecycle of an incident:
  1. Buffer frames continuously
  2. On anomaly trigger: extract pre/post clip
  3. Generate JSON incident report + PDF
  4. Persist to disk
"""
import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from config import settings
from models.schemas import (
    AnomalyEvent, IncidentReport, NarrativeEntry, FrameResult
)
from models.enums import IncidentStatus
from utils.buffer import FrameRingBuffer
from utils.ffmpeg import save_frames_to_clip
from utils.report_generator import generate_incident_pdf

logger = logging.getLogger(__name__)


class IncidentManager:
    """
    Per-feed incident manager.
    Maintains a rolling frame buffer and generates full incident packages on trigger.
    """

    def __init__(self, feed_id: str, feed_name: str = ""):
        self.feed_id = feed_id
        self.feed_name = feed_name
        self.buffer = FrameRingBuffer(maxlen=settings.max_buffer_frames)
        self._incidents: Dict[str, IncidentReport] = {}
        self._post_event_frames: Dict[str, List] = {}
        self._active_incident_id: Optional[str] = None
        self._post_frame_target: int = 0

    # ── Frame Ingestion ───────────────────────────────────────────────────────

    def ingest_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Push a frame into the rolling buffer."""
        ts = timestamp or datetime.utcnow()
        self.buffer.push(frame_number, ts, frame)

        # Collect post-event frames if an incident is in progress
        if self._active_incident_id:
            inc = self._incidents.get(self._active_incident_id)
            if inc and inc.status == IncidentStatus.PROCESSING:
                if self._active_incident_id not in self._post_event_frames:
                    self._post_event_frames[self._active_incident_id] = []
                self._post_event_frames[self._active_incident_id].append(
                    (frame_number, ts, frame.copy())
                )

    # ── Incident Trigger ──────────────────────────────────────────────────────

    async def trigger_incident(
        self,
        anomaly_event: AnomalyEvent,
        narrative_log: List[NarrativeEntry],
        scene_summary: str = "",
    ) -> IncidentReport:
        """
        Trigger a new incident from an anomaly event.
        Extracts pre-event frames, initializes the incident report,
        and starts collecting post-event frames.

        Args:
            anomaly_event: The triggering anomaly.
            narrative_log: Current narrative log entries for this feed.
            scene_summary: LLM-generated scene summary.

        Returns:
            The initialized IncidentReport.
        """
        incident_id = str(uuid.uuid4())
        pre_frames = self.buffer.get_last_n_seconds(
            settings.pre_event_seconds,
            fps=settings.target_fps,
        )

        incident = IncidentReport(
            incident_id=incident_id,
            feed_id=self.feed_id,
            feed_name=self.feed_name,
            status=IncidentStatus.PROCESSING,
            anomaly_type=anomaly_event.anomaly_type,
            triggered_at=anomaly_event.timestamp,
            track_ids_involved=anomaly_event.track_ids,
            narrative_log=narrative_log[-50:],  # last 50 entries
            scene_summary=scene_summary,
            confidence_score=anomaly_event.confidence,
            frame_start=pre_frames[0][0] if pre_frames else 0,
            frame_end=anomaly_event.frame_number,
        )

        self._incidents[incident_id] = incident
        self._active_incident_id = incident_id
        self._post_event_frames[incident_id] = []
        self._post_frame_target = int(settings.post_event_seconds * settings.target_fps)

        logger.info(
            f"[{self.feed_id}] Incident triggered: {incident_id} "
            f"({anomaly_event.anomaly_type.value}) — {len(pre_frames)} pre-event frames buffered"
        )

        # Schedule async finalization after post-event window
        asyncio.create_task(
            self._finalize_incident(incident_id, pre_frames)
        )

        return incident

    async def _finalize_incident(
        self,
        incident_id: str,
        pre_frames: List,
    ) -> None:
        """Wait for post-event frames, then finalize the incident."""
        # Wait for post-event window
        post_wait = settings.post_event_seconds + 2  # +2s buffer
        await asyncio.sleep(post_wait)

        incident = self._incidents.get(incident_id)
        if not incident:
            return

        post_frames = self._post_event_frames.pop(incident_id, [])
        all_frames = pre_frames + post_frames

        # Save video clip
        clip_path = None
        if all_frames:
            ts_str = incident.triggered_at.strftime("%Y%m%d_%H%M%S")
            clip_filename = f"incident_{ts_str}_{incident_id[:8]}.mp4"
            clip_path = settings.incidents_dir / clip_filename
            success = save_frames_to_clip(
                all_frames, clip_path, fps=settings.target_fps
            )
            if success:
                incident.clip_path = str(clip_path)
                logger.info(f"[{self.feed_id}] Clip saved: {clip_path}")
            else:
                logger.warning(f"[{self.feed_id}] Clip save failed for {incident_id}")

        # Save PDF report
        pdf_path = settings.incidents_dir / f"report_{incident_id[:8]}.pdf"
        pdf_ok = generate_incident_pdf(incident, pdf_path)
        if pdf_ok:
            incident.pdf_report_path = str(pdf_path)

        # Save JSON
        json_path = settings.incidents_dir / f"incident_{incident_id[:8]}.json"
        self._save_json(incident, json_path)

        incident.status = IncidentStatus.COMPLETE
        incident.completed_at = datetime.utcnow()
        self._active_incident_id = None

        logger.info(
            f"[{self.feed_id}] Incident finalized: {incident_id} "
            f"({len(all_frames)} frames, clip={clip_path})"
        )

    # ── Accessors ──────────────────────────────────────────────────────────────

    def get_incident(self, incident_id: str) -> Optional[IncidentReport]:
        return self._incidents.get(incident_id)

    def get_all_incidents(self) -> List[IncidentReport]:
        return list(self._incidents.values())

    def get_incident_count(self) -> int:
        return len(self._incidents)

    # ── Persistence ───────────────────────────────────────────────────────────

    @staticmethod
    def _save_json(incident: IncidentReport, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write(incident.model_dump_json(indent=2))
        except Exception as e:
            logger.error(f"Failed to save incident JSON: {e}")

    @staticmethod
    def load_incident_from_json(path: Path) -> Optional[IncidentReport]:
        try:
            with open(path) as f:
                data = json.load(f)
            return IncidentReport(**data)
        except Exception as e:
            logger.error(f"Failed to load incident JSON from {path}: {e}")
            return None
