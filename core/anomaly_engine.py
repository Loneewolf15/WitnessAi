"""
WitnessAI — Anomaly Engine
Behavioral anomaly detectors that operate on tracked object state.
Each detector is stateless with respect to the engine — state lives in TrackedObject.
"""
import logging
from datetime import datetime
from typing import List, Dict, Optional
from abc import ABC, abstractmethod

from config import settings
from models.schemas import FrameResult, TrackedObject, AnomalyEvent
from models.enums import AnomalyType

logger = logging.getLogger(__name__)


class BaseDetector(ABC):
    """Abstract base class for anomaly detectors."""

    @abstractmethod
    def check(
        self,
        frame_result: FrameResult,
        tracked_objects: Dict[int, TrackedObject],
    ) -> List[AnomalyEvent]:
        """Return list of anomaly events detected in this frame."""
        ...


class LoiteringDetector(BaseDetector):
    """
    Detects when a person has been in the scene beyond a time threshold.
    """

    def __init__(self, threshold_seconds: Optional[float] = None):
        self.threshold = threshold_seconds or settings.loiter_threshold_seconds
        self._alerted: Dict[int, bool] = {}  # track_id → already alerted

    def check(self, frame_result: FrameResult, tracked_objects: Dict[int, TrackedObject]) -> List[AnomalyEvent]:
        events = []
        for track_id, obj in tracked_objects.items():
            if obj.class_name != "person":
                continue
            duration = obj.duration_seconds()
            if duration >= self.threshold and not self._alerted.get(track_id, False):
                self._alerted[track_id] = True
                events.append(AnomalyEvent(
                    feed_id=frame_result.feed_id,
                    anomaly_type=AnomalyType.LOITERING,
                    track_ids=[track_id],
                    frame_number=frame_result.frame_number,
                    confidence=0.85,
                    description=(
                        f"Track-{track_id} ({obj.class_name}) has been in the scene "
                        f"for {duration:.0f}s — exceeds threshold of {self.threshold:.0f}s."
                    ),
                    metadata={"duration_seconds": duration},
                ))
        return events

    def reset_track(self, track_id: int) -> None:
        self._alerted.pop(track_id, None)


class RunningDetector(BaseDetector):
    """
    Detects rapid movement based on bounding box displacement between frames.
    """

    def __init__(self, velocity_threshold: Optional[float] = None):
        self.threshold = velocity_threshold or settings.run_velocity_threshold
        self._alerted: Dict[int, int] = {}  # track_id → last alerted frame

    def check(self, frame_result: FrameResult, tracked_objects: Dict[int, TrackedObject]) -> List[AnomalyEvent]:
        events = []
        fn = frame_result.frame_number
        for track_id, obj in tracked_objects.items():
            if obj.class_name != "person":
                continue
            velocity = obj.velocity()
            last_alerted = self._alerted.get(track_id, -999)
            # Only alert once per 30 frames to avoid spam
            if velocity >= self.threshold and (fn - last_alerted) > 30:
                self._alerted[track_id] = fn
                events.append(AnomalyEvent(
                    feed_id=frame_result.feed_id,
                    anomaly_type=AnomalyType.RUNNING,
                    track_ids=[track_id],
                    frame_number=fn,
                    confidence=min(velocity / self.threshold, 1.0),
                    description=(
                        f"Track-{track_id} moving at high velocity ({velocity:.1f} px/frame). "
                        f"Threshold: {self.threshold:.1f} px/frame."
                    ),
                    metadata={"velocity": velocity},
                ))
        return events


class CrowdSurgeDetector(BaseDetector):
    """
    Detects sudden increase in person count within the scene.
    """

    def __init__(self, surge_count: Optional[int] = None):
        self.surge_count = surge_count or settings.crowd_surge_count
        self._prev_count: int = 0
        self._last_alerted_frame: int = -999

    def check(self, frame_result: FrameResult, tracked_objects: Dict[int, TrackedObject]) -> List[AnomalyEvent]:
        events = []
        person_count = sum(1 for o in tracked_objects.values() if o.class_name == "person" and o.is_active)
        fn = frame_result.frame_number
        delta = person_count - self._prev_count
        if delta >= self.surge_count and (fn - self._last_alerted_frame) > 60:
            self._last_alerted_frame = fn
            events.append(AnomalyEvent(
                feed_id=frame_result.feed_id,
                anomaly_type=AnomalyType.CROWD_SURGE,
                track_ids=[tid for tid, o in tracked_objects.items() if o.class_name == "person"],
                frame_number=fn,
                confidence=0.75,
                description=(
                    f"Sudden increase of {delta} persons detected. "
                    f"Current count: {person_count}. Previous: {self._prev_count}."
                ),
                metadata={"person_count": person_count, "delta": delta},
            ))
        self._prev_count = person_count
        return events


class FallDetector(BaseDetector):
    """
    Detects a fall by monitoring the aspect ratio of a person's bounding box.
    When a person falls, their bounding box becomes wider than it is tall.
    """

    def __init__(self, aspect_ratio_threshold: Optional[float] = None):
        self.threshold = aspect_ratio_threshold or settings.fall_aspect_ratio_threshold
        self._alerted: Dict[int, int] = {}

    def check(self, frame_result: FrameResult, tracked_objects: Dict[int, TrackedObject]) -> List[AnomalyEvent]:
        events = []
        fn = frame_result.frame_number
        for track_id, obj in tracked_objects.items():
            if obj.class_name != "person":
                continue
            bbox = obj.bbox
            if bbox.height == 0:
                continue
            aspect = bbox.width / bbox.height
            last_alerted = self._alerted.get(track_id, -999)
            if aspect >= self.threshold and (fn - last_alerted) > 45:
                self._alerted[track_id] = fn
                events.append(AnomalyEvent(
                    feed_id=frame_result.feed_id,
                    anomaly_type=AnomalyType.FALL,
                    track_ids=[track_id],
                    frame_number=fn,
                    confidence=min(aspect / self.threshold, 1.0),
                    description=(
                        f"Possible fall detected for Track-{track_id}. "
                        f"Bounding box aspect ratio: {aspect:.2f} (threshold: {self.threshold:.2f})."
                    ),
                    metadata={"aspect_ratio": aspect},
                ))
        return events


class AbandonedObjectDetector(BaseDetector):
    """
    Detects objects (bags, backpacks, suitcases) that remain stationary
    after no person is nearby.
    """

    OBJECT_CLASSES = {"backpack", "handbag", "suitcase", "bag"}
    PROXIMITY_THRESHOLD = 150  # pixels

    def __init__(self, stillness_seconds: Optional[float] = None):
        self.threshold = stillness_seconds or settings.abandon_stillness_seconds
        self._alerted: Dict[int, bool] = {}

    def check(self, frame_result: FrameResult, tracked_objects: Dict[int, TrackedObject]) -> List[AnomalyEvent]:
        events = []
        objects = {tid: o for tid, o in tracked_objects.items() if o.class_name in self.OBJECT_CLASSES}
        persons = {tid: o for tid, o in tracked_objects.items() if o.class_name == "person"}

        for track_id, obj in objects.items():
            if self._alerted.get(track_id, False):
                continue
            if obj.duration_seconds() < self.threshold:
                continue
            # Check if any person is nearby
            near_person = any(
                self._distance(obj.bbox, p.bbox) < self.PROXIMITY_THRESHOLD
                for p in persons.values()
            )
            if not near_person:
                self._alerted[track_id] = True
                events.append(AnomalyEvent(
                    feed_id=frame_result.feed_id,
                    anomaly_type=AnomalyType.ABANDONED_OBJECT,
                    track_ids=[track_id],
                    frame_number=frame_result.frame_number,
                    confidence=0.80,
                    description=(
                        f"Unattended {obj.class_name} detected (Track-{track_id}). "
                        f"Stationary for {obj.duration_seconds():.0f}s with no person in proximity."
                    ),
                    metadata={"object_class": obj.class_name, "duration": obj.duration_seconds()},
                ))
        return events

    @staticmethod
    def _distance(bbox_a, bbox_b) -> float:
        dx = bbox_a.center_x - bbox_b.center_x
        dy = bbox_a.center_y - bbox_b.center_y
        return (dx ** 2 + dy ** 2) ** 0.5


class AnomalyEngine:
    """
    Orchestrates all detectors and produces AnomalyEvents per frame.
    """

    def __init__(self):
        self.detectors: List[BaseDetector] = [
            LoiteringDetector(),
            RunningDetector(),
            CrowdSurgeDetector(),
            FallDetector(),
            AbandonedObjectDetector(),
        ]

    def analyze(
        self,
        frame_result: FrameResult,
        tracked_objects: Dict[int, TrackedObject],
    ) -> List[AnomalyEvent]:
        """
        Run all detectors on the current frame state.

        Returns:
            All anomaly events detected across all detectors.
        """
        all_events = []
        for detector in self.detectors:
            try:
                events = detector.check(frame_result, tracked_objects)
                all_events.extend(events)
            except Exception as e:
                logger.error(f"Detector {detector.__class__.__name__} error: {e}")
        if all_events:
            logger.info(f"[{frame_result.feed_id}] Frame {frame_result.frame_number}: {len(all_events)} anomaly events")
        return all_events
