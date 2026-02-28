"""
WitnessAI — Camera Manager
Orchestrates multiple simultaneous video feeds.
Each feed runs its own processing loop in an async task.
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

from config import settings
from models.schemas import FeedConfig, FeedResponse, AnomalyEvent
from models.enums import FeedStatus
from core.frame_processor import FrameProcessor
from core.anomaly_engine import AnomalyEngine
from core.narrative_engine import NarrativeEngine
from core.incident_manager import IncidentManager

logger = logging.getLogger(__name__)


class FeedWorker:
    """
    Manages the processing loop for a single camera feed.
    """

    def __init__(self, config: FeedConfig):
        self.config = config
        self.feed_id = config.feed_id
        self.frame_processor = FrameProcessor()
        self.anomaly_engine = AnomalyEngine()
        self.narrative_engine = NarrativeEngine()
        self.incident_manager = IncidentManager(config.feed_id, config.name)
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._frame_number = 0
        self._latest_annotated_frame: Optional[np.ndarray] = None

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.config.status = FeedStatus.ACTIVE
        self._task = asyncio.create_task(self._processing_loop())
        logger.info(f"Feed worker started: {self.feed_id} — source: {self.config.source}")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.config.status = FeedStatus.STOPPED
        logger.info(f"Feed worker stopped: {self.feed_id}")

    async def _processing_loop(self) -> None:
        """Main frame processing loop for this feed."""
        cap = self._open_capture()
        if cap is None:
            logger.warning(f"[{self.feed_id}] Could not open source; running in mock mode")
            await self._mock_processing_loop()
            return

        try:
            import cv2  # type: ignore
            frame_interval = 1.0 / settings.target_fps

            while self._running:
                loop_start = asyncio.get_event_loop().time()
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"[{self.feed_id}] End of stream or read error")
                    # For video files: loop; for live feeds: break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                await self._process_single_frame(frame)

                # Throttle to target FPS
                elapsed = asyncio.get_event_loop().time() - loop_start
                sleep_time = max(0.0, frame_interval - elapsed)
                await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[{self.feed_id}] Processing loop error: {e}", exc_info=True)
            self.config.status = FeedStatus.ERROR
        finally:
            if cap:
                cap.release()

    async def _mock_processing_loop(self) -> None:
        """Generates synthetic frames for testing without a real camera."""
        frame_interval = 1.0 / settings.target_fps
        while self._running:
            mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add a moving rectangle to simulate a person
            x = (self._frame_number * 3) % 600
            import cv2  # type: ignore
            cv2.rectangle(mock_frame, (x, 150), (x + 60, 420), (255, 255, 255), -1)
            await self._process_single_frame(mock_frame)
            await asyncio.sleep(frame_interval)

    async def _process_single_frame(self, frame: np.ndarray) -> None:
        """Run the full pipeline on a single frame."""
        fn = self._frame_number
        now = datetime.utcnow()
        self._frame_number += 1

        # 1. Buffer frame
        self.incident_manager.ingest_frame(frame.copy(), fn, now)

        # 2. Detect + track
        frame_result = self.frame_processor.process_frame(self.feed_id, frame, fn)

        # 3. Get current tracked objects
        tracked = self.frame_processor.get_tracked_objects(self.feed_id)

        # 4. Anomaly detection
        anomalies = self.anomaly_engine.analyze(frame_result, tracked)
        frame_result.anomalies_detected = anomalies

        # 5. Narrative logging
        new_entries = self.narrative_engine.log_frame(frame_result, tracked, anomalies)

        # 6. Trigger incident if anomaly detected and not already processing
        if anomalies and self.incident_manager._active_incident_id is None:
            primary_anomaly = anomalies[0]
            recent_log = self.narrative_engine.get_recent_log(self.feed_id, last_n=50)
            scene_summary = await self.narrative_engine.generate_incident_summary(
                self.feed_id, primary_anomaly, recent_log
            )
            await self.incident_manager.trigger_incident(
                primary_anomaly, recent_log, scene_summary
            )

    def _open_capture(self):
        """Try to open the video source with OpenCV."""
        try:
            import cv2  # type: ignore
            source = self.config.source
            # Convert string "0", "1" etc. to int for webcam
            try:
                source = int(source)
            except (ValueError, TypeError):
                pass
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                return None
            return cap
        except ImportError:
            logger.warning("OpenCV (cv2) not installed")
            return None

    @property
    def latest_annotated_frame(self) -> Optional[np.ndarray]:
        return self._latest_annotated_frame


class CameraManager:
    """
    Top-level manager for all active camera feeds.
    Provides the interface used by the FastAPI routes.
    """

    def __init__(self):
        self._workers: Dict[str, FeedWorker] = {}

    async def add_feed(self, config: FeedConfig) -> FeedWorker:
        worker = FeedWorker(config)
        self._workers[config.feed_id] = worker
        await worker.start()
        return worker

    async def remove_feed(self, feed_id: str) -> bool:
        worker = self._workers.pop(feed_id, None)
        if worker:
            await worker.stop()
            return True
        return False

    def get_worker(self, feed_id: str) -> Optional[FeedWorker]:
        return self._workers.get(feed_id)

    def list_feeds(self) -> List[FeedResponse]:
        results = []
        for worker in self._workers.values():
            cfg = worker.config
            results.append(FeedResponse(
                feed_id=cfg.feed_id,
                name=cfg.name,
                source=cfg.source,
                status=cfg.status,
                created_at=cfg.created_at,
                incident_count=worker.incident_manager.get_incident_count(),
            ))
        return results

    def get_all_incidents(self):
        incidents = []
        for worker in self._workers.values():
            incidents.extend(worker.incident_manager.get_all_incidents())
        return sorted(incidents, key=lambda i: i.triggered_at, reverse=True)

    def get_incident(self, incident_id: str):
        for worker in self._workers.values():
            inc = worker.incident_manager.get_incident(incident_id)
            if inc:
                return inc
        return None

    async def shutdown(self) -> None:
        for worker in list(self._workers.values()):
            await worker.stop()
        self._workers.clear()


# Singleton — shared across the FastAPI app
camera_manager = CameraManager()
