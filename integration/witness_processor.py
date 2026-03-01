"""
WitnessAI — Vision Agents SDK Processor
========================================
The single bridge between Stream's WebRTC edge and our entire backend.

Frame flow per tick:
  av.VideoFrame → numpy BGR → WitnessAgent pipeline → annotated RGB → Stream
  + latency measured every frame
  + anomaly events broadcast to WebSocket dashboard
  + annotated feed published back into the call
"""
from __future__ import annotations

import base64
import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional SDK imports — guarded so tests run without full install ──
try:
    import aiortc
    import av
    from vision_agents.core.processors import VideoProcessorPublisher
    from vision_agents.core.utils.video_forwarder import VideoForwarder
    from vision_agents.core.utils.video_track import QueuedVideoTrack
    SDK_AVAILABLE = True
except ImportError:
    aiortc = None
    av = None
    SDK_AVAILABLE = False

    class VideoProcessorPublisher:
        pass
    class VideoForwarder:
        pass
    class QueuedVideoTrack:
        def stop(self): pass

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────
# Events (broadcast via WebSocket to the React dashboard)
# ─────────────────────────────────────────────────────────────────

@dataclass
class AnomalyDetectedEvent:
    incident_id: str
    camera_id: str
    anomaly_type: str
    confidence: str
    description: str
    timestamp: str
    narrative_entry: str


@dataclass
class IncidentPackagedEvent:
    incident_id: str
    camera_id: str
    package_id: str
    report_path: str
    video_path: Optional[str]
    total_anomalies: int


# ─────────────────────────────────────────────────────────────────
# Frame annotation
# ─────────────────────────────────────────────────────────────────

ANOMALY_COLORS = {
    "loitering":        (0, 165, 255),
    "running":          (0, 0, 255),
    "crowd_surge":      (147, 20, 255),
    "abandoned_object": (255, 255, 0),
    "fall_detected":    (0, 0, 200),
    "intrusion":        (0, 0, 255),
    "normal":           (0, 200, 80),
}


def draw_detections(frame: np.ndarray, tracked_objects: list, anomaly_types: set) -> np.ndarray:
    if not CV2_AVAILABLE:
        return frame.copy()
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for obj in tracked_objects:
        b = obj.bbox
        x1 = max(0, int(b.x1)); y1 = max(0, int(b.y1))
        x2 = min(w - 1, int(b.x2)); y2 = min(h - 1, int(b.y2))
        color = ANOMALY_COLORS["normal"]
        label = f"#{obj.track_id} {obj.class_name} {obj.confidence:.0%}"
        for atype in anomaly_types:
            color = ANOMALY_COLORS.get(atype, ANOMALY_COLORS["normal"])
            label = f"#{obj.track_id} \u26a0 {atype.replace('_', ' ').upper()}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, font, 0.42, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4), font, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return annotated


def draw_hud(
    frame: np.ndarray,
    status: str,
    is_incident: bool,
    fps: float,
    latency_ms: float,
    frame_count: int,
    tracked_count: int,
) -> np.ndarray:
    """Draw full HUD: top banner, bottom metrics bar, watermark."""
    if not CV2_AVAILABLE:
        return frame.copy()
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Top banner
    banner_color = (0, 0, 160) if is_incident else (20, 20, 20)
    cv2.rectangle(annotated, (0, 0), (w, 34), banner_color, -1)
    cv2.putText(annotated, status, (8, 23), font, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # Bottom metrics bar
    cv2.rectangle(annotated, (0, h - 28), (w, h), (10, 10, 10), -1)
    ts = datetime.utcnow().strftime("%H:%M:%S UTC")
    left = f"\U0001f441 WitnessAI  |  {ts}"
    right = f"FPS:{fps:.1f}  LAT:{latency_ms:.0f}ms  TRK:{tracked_count}  FRM:{frame_count}"
    cv2.putText(annotated, left,  (8,      h - 10), font, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(annotated, right, (w - 310, h - 10), font, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    # Latency dot (green < 50ms, yellow < 100ms, red otherwise)
    dot_color = (0, 220, 80) if latency_ms < 50 else (0, 220, 220) if latency_ms < 100 else (0, 0, 220)
    cv2.circle(annotated, (w - 320, h - 14), 5, dot_color, -1)

    return annotated


# ─────────────────────────────────────────────────────────────────
# Processor
# ─────────────────────────────────────────────────────────────────

class WitnessProcessor(VideoProcessorPublisher):
    """
    Vision Agents SDK processor — plugs WitnessAI into a Stream WebRTC call.
    Publishes annotated video back into the call with live HUD.
    Broadcasts all events to the React dashboard via WebSocket.
    """

    name = "witness_ai"

    def __init__(self, witness_agent, fps: int = 5, ws_manager=None):
        self.fps = fps
        self._agent = witness_agent
        self._ws_manager = ws_manager
        self._forwarder = None
        self._video_track = QueuedVideoTrack()
        self._sdk_events = None
        self._frame_count = 0
        self._active_anomaly_types: set[str] = set()
        self._last_status = "\U0001f7e2 NOMINAL  \u2014  WitnessAI monitoring"

        # Latency tracking — rolling 30-frame window
        self._latencies: deque[float] = deque(maxlen=30)
        self._fps_window: deque[float] = deque(maxlen=30)

        logger.info(f"[{witness_agent.camera_id}] WitnessProcessor initialised @ {fps}fps")

    def attach_agent(self, agent) -> None:
        self._sdk_events = agent.events

    async def process_video(self, track, participant_id, shared_forwarder=None) -> None:
        if self._forwarder:
            await self._forwarder.remove_frame_handler(self._process_frame)
        self._forwarder = shared_forwarder
        self._forwarder.add_frame_handler(
            self._process_frame, fps=float(self.fps), name=self.name
        )
        logger.info(f"[{self._agent.camera_id}] Track registered (participant={participant_id})")

    async def _process_frame(self, frame) -> None:
        t_start = time.perf_counter()
        self._frame_count += 1

        # ── Decode ──
        if av is not None:
            img_rgb = frame.to_ndarray(format="rgb24")
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR) if CV2_AVAILABLE else img_rgb
        else:
            img_bgr = np.zeros((480, 640, 3), dtype=np.uint8)

        # ── WitnessAI full pipeline ──
        anomalies = self._agent.process_frame(img_bgr)

        # ── Update anomaly state ──
        if anomalies:
            self._active_anomaly_types = {a.anomaly_type.value for a in anomalies}
            self._last_status = (
                f"\U0001f534 INCIDENT  \u2014  "
                f"{', '.join(self._active_anomaly_types).replace('_',' ').upper()}"
            )
            for anomaly in anomalies:
                await self._on_anomaly(anomaly)
        elif self._frame_count % (self.fps * 15) == 0:
            self._active_anomaly_types.clear()
            self._last_status = "\U0001f7e2 NOMINAL  \u2014  WitnessAI monitoring"

        # ── Compute metrics ──
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000
        self._latencies.append(latency_ms)
        self._fps_window.append(t_end)

        avg_latency = sum(self._latencies) / len(self._latencies)
        actual_fps = 0.0
        if len(self._fps_window) >= 2:
            span = self._fps_window[-1] - self._fps_window[0]
            actual_fps = (len(self._fps_window) - 1) / span if span > 0 else 0.0

        # ── Broadcast metrics to dashboard every 10 frames ──
        if self._frame_count % 10 == 0 and self._ws_manager:
            status = self._agent.status()
            await self._ws_manager.broadcast_raw({
                "event_type": "metrics",
                "fps": round(actual_fps, 1),
                "latency_ms": round(avg_latency, 1),
                "frame_count": self._frame_count,
                "tracked_persons": len(self._agent._tracker.active_tracks),
                "total_anomalies": status.total_anomalies,
                "active_incidents": status.active_incidents,
                "is_incident": bool(self._active_anomaly_types),
                "anomaly_types": list(self._active_anomaly_types),
            })

        # ── Annotate + publish ──
        tracked = [t.to_detected_object() for t in self._agent._tracker.active_tracks.values()]
        annotated = draw_detections(img_bgr, tracked, self._active_anomaly_types)
        annotated = draw_hud(
            annotated,
            status=self._last_status,
            is_incident=bool(self._active_anomaly_types),
            fps=actual_fps,
            latency_ms=avg_latency,
            frame_count=self._frame_count,
            tracked_count=len(tracked),
        )

        if av is not None and CV2_AVAILABLE:
            out_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            out_frame = av.VideoFrame.from_ndarray(out_rgb, format="rgb24")
            out_frame.pts = frame.pts
            out_frame.time_base = frame.time_base
            await self._video_track.add_frame(out_frame)

        # ── Broadcast annotated frame to dashboard WebSocket ──
        if CV2_AVAILABLE and self._ws_manager:
            _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"
            await self._ws_manager.broadcast_frame(b64)

    async def _on_anomaly(self, anomaly) -> None:
        """Broadcast anomaly to dashboard + emit SDK event for LLM."""
        incident_id = (
            list(self._agent._active_incidents.keys())[-1]
            if self._agent._active_incidents else "unknown"
        )
        narrative = self._agent._narrator.get_narrative(incident_id)
        latest_entry = narrative.entries[-1].text if narrative and narrative.entries else ""

        payload = {
            "event_type": "anomaly",
            "incident_id": incident_id,
            "camera_id": anomaly.camera_id,
            "anomaly_type": anomaly.anomaly_type.value,
            "confidence": anomaly.confidence.value,
            "description": anomaly.description,
            "timestamp": anomaly.timestamp.strftime("%H:%M:%S UTC"),
            "narrative_entry": latest_entry,
        }

        # → WebSocket dashboard
        if self._ws_manager:
            await self._ws_manager.broadcast_raw(payload)

        # → SDK event bus (so Gemini hears it and speaks aloud)
        if self._sdk_events:
            try:
                event = AnomalyDetectedEvent(
                    incident_id=incident_id,
                    camera_id=anomaly.camera_id,
                    anomaly_type=anomaly.anomaly_type.value,
                    confidence=anomaly.confidence.value,
                    description=anomaly.description,
                    timestamp=anomaly.timestamp.strftime("%H:%M:%S UTC"),
                    narrative_entry=latest_entry,
                )
                await self._sdk_events.emit(event)
            except Exception as e:
                logger.debug(f"SDK event emit: {e}")

    def publish_video_track(self):
        return self._video_track

    async def stop_processing(self) -> None:
        if self._forwarder:
            await self._forwarder.remove_frame_handler(self._process_frame)
            self._forwarder = None

    async def close(self) -> None:
        await self.stop_processing()
        self._video_track.stop()
