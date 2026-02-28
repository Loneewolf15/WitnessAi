"""WitnessAI — WebSocket handler for real-time dashboard updates"""
from __future__ import annotations
import base64
import json
import logging
import os
import time
from collections import deque

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False

logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)
        logger.info(f"WS client connected. Total: {len(self._connections)}")

    def disconnect(self, ws: WebSocket) -> None:
        self._connections = [c for c in self._connections if c is not ws]
        logger.info(f"WS client disconnected. Total: {len(self._connections)}")

    async def broadcast_raw(self, data: dict) -> None:
        """Broadcast raw dict as JSON to all connected clients."""
        if not self._connections:
            return
        text = json.dumps(data, default=str)
        dead = []
        for ws in self._connections:
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    @property
    def connection_count(self) -> int:
        return len(self._connections)


# Global singleton — imported by main.py and witness_processor.py
manager = ConnectionManager()


@router.websocket("/live")
async def websocket_live(ws: WebSocket):
    """WebSocket endpoint consumed by the React dashboard."""
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()  # keep-alive; server pushes via manager
    except WebSocketDisconnect:
        manager.disconnect(ws)


@router.websocket("/camera")
async def websocket_camera(ws: WebSocket):
    """WebSocket endpoint that receives live camera frames from the browser.

    Uses frame-skipping: browser sends at high FPS, but the server only
    processes the latest frame when it finishes the previous one.
    In between, it echoes back the last annotated frame to keep the
    video feed smooth on the frontend.
    """
    await ws.accept()
    logger.info("Camera WS connected from dashboard.")

    # Get agent from app.state (set in api/main.py create_app)
    _witness_agent = ws.app.state.witness_agent
    if _witness_agent is None:
        logger.error("WitnessAgent not initialized — closing camera WS")
        await ws.close(code=1011, reason="Agent not ready")
        return
    from integration.witness_processor import draw_detections, draw_hud

    # ── State ──
    processed_count = 0
    latencies: deque = deque(maxlen=30)
    fps_window: deque = deque(maxlen=30)
    active_anomaly_types: set = set()
    last_status = "\U0001f7e2 NOMINAL  \u2014  WitnessAI monitoring"
    last_annotated_b64: str | None = None
    fps_cfg = int(os.getenv("PROCESSING_FPS", "5"))

    try:
        while True:
            # ── 1. Receive frame (blocks until browser sends one) ──
            data = await ws.receive_text()
            if not data.startswith("data:image/"):
                continue

            # ── 2. If we have a cached annotated frame, send it immediately ──
            #    This keeps the frontend feed alive even while we process.
            if last_annotated_b64:
                await ws.send_text(last_annotated_b64)

            # ── 3. Drain any queued frames — keep only the LATEST ──
            #    This is the key frame-skipping trick: discard stale frames.
            import asyncio
            while True:
                try:
                    newer = await asyncio.wait_for(ws.receive_text(), timeout=0.005)
                    if newer.startswith("data:image/"):
                        data = newer  # overwrite with newer frame
                except asyncio.TimeoutError:
                    break  # no more queued frames

            # ── 4. Process this (latest) frame through YOLOv8 ──
            t_start = time.perf_counter()
            processed_count += 1

            try:
                base64_data = data.split(",", 1)[1]
                img_data = base64.b64decode(base64_data)
                np_arr = np.frombuffer(img_data, np.uint8)
                img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception:
                continue

            if img_bgr is None:
                continue

            # Run YOLO + tracker + anomaly detection
            anomalies = _witness_agent.process_frame(img_bgr)

            if anomalies:
                active_anomaly_types = {a.anomaly_type.value for a in anomalies}
                last_status = (
                    f"\U0001f534 INCIDENT  \u2014  "
                    f"{', '.join(active_anomaly_types).replace('_',' ').upper()}"
                )
                # Broadcast anomalies to /ws/live
                for anomaly in anomalies:
                    incident_id = (
                        list(_witness_agent._active_incidents.keys())[-1]
                        if _witness_agent._active_incidents else "unknown"
                    )
                    narrative = _witness_agent._narrator.get_narrative(incident_id)
                    latest_entry = (
                        narrative.entries[-1].text
                        if narrative and narrative.entries else ""
                    )
                    await manager.broadcast_raw({
                        "event_type": "anomaly",
                        "incident_id": incident_id,
                        "camera_id": anomaly.camera_id,
                        "anomaly_type": anomaly.anomaly_type.value,
                        "confidence": anomaly.confidence.value,
                        "description": anomaly.description,
                        "timestamp": anomaly.timestamp.strftime("%H:%M:%S UTC"),
                        "narrative_entry": latest_entry,
                    })
            elif processed_count % (fps_cfg * 15) == 0:
                active_anomaly_types.clear()
                last_status = "\U0001f7e2 NOMINAL  \u2014  WitnessAI monitoring"

            # ── 5. Compute metrics ──
            t_end = time.perf_counter()
            latency_ms = (t_end - t_start) * 1000
            latencies.append(latency_ms)
            fps_window.append(t_end)

            avg_latency = sum(latencies) / len(latencies)
            actual_fps = 0.0
            if len(fps_window) >= 2:
                span = fps_window[-1] - fps_window[0]
                actual_fps = (len(fps_window) - 1) / span if span > 0 else 0.0

            # Broadcast metrics every 5 processed frames
            if processed_count % 5 == 0:
                status = _witness_agent.status()
                await manager.broadcast_raw({
                    "event_type": "metrics",
                    "fps": round(actual_fps, 1),
                    "latency_ms": round(avg_latency, 1),
                    "frame_count": processed_count,
                    "tracked_persons": len(_witness_agent._tracker.active_tracks),
                    "total_anomalies": status.total_anomalies,
                    "active_incidents": status.active_incidents,
                    "is_incident": bool(active_anomaly_types),
                    "anomaly_types": list(active_anomaly_types),
                })

            # ── 6. Annotate and cache ──
            tracked = [
                t.to_detected_object()
                for t in _witness_agent._tracker.active_tracks.values()
            ]
            if CV2_AVAILABLE:
                annotated = draw_detections(img_bgr, tracked, active_anomaly_types)
                annotated = draw_hud(
                    annotated,
                    status=last_status,
                    is_incident=bool(active_anomaly_types),
                    fps=actual_fps,
                    latency_ms=avg_latency,
                    frame_count=processed_count,
                    tracked_count=len(tracked),
                )
                _, buf = cv2.imencode(
                    '.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70]
                )
                last_annotated_b64 = (
                    f"data:image/jpeg;base64,"
                    f"{base64.b64encode(buf).decode('utf-8')}"
                )

                # Send the freshly annotated frame
                await ws.send_text(last_annotated_b64)

    except WebSocketDisconnect:
        logger.info("Camera WS disconnected.")
    except Exception as e:
        logger.error(f"Camera WS error: {e}", exc_info=True)
