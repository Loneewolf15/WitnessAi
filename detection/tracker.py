"""WitnessAI - Multi-Object Tracker (DeepSORT-inspired, pure Python/NumPy)"""
from __future__ import annotations
import numpy as np
from datetime import datetime
from collections import defaultdict
import logging; logger = logging.getLogger(__name__)
from models.schemas import DetectedObject, FrameDetections, BoundingBox


def iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Intersection over Union for two bounding boxes."""
    xa = max(box_a.x1, box_b.x1)
    ya = max(box_a.y1, box_b.y1)
    xb = min(box_a.x2, box_b.x2)
    yb = min(box_a.y2, box_b.y2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = box_a.area + box_b.area - inter
    return inter / union if union > 0 else 0.0


class Track:
    """Single tracked object with Kalman-like state prediction."""

    def __init__(self, track_id: int, obj: DetectedObject):
        self.track_id = track_id
        self.class_name = obj.class_name
        self.camera_id = obj.camera_id
        self.bbox = obj.bbox
        self.confidence = obj.confidence
        self.hits = 1
        self.misses = 0
        self.last_seen: datetime = obj.timestamp
        self.history: list[BoundingBox] = [obj.bbox]
        self._prev_center: tuple[float, float] = (obj.bbox.center_x, obj.bbox.center_y)
        self.velocity_x: float = 0.0
        self.velocity_y: float = 0.0
        # Time-at-location for loitering detection
        self.stationary_frames: int = 0
        self.first_seen: datetime = obj.timestamp

    def update(self, obj: DetectedObject, dt: float = 1.0) -> None:
        """Update track with new detection."""
        cx, cy = obj.bbox.center_x, obj.bbox.center_y
        prev_cx, prev_cy = self._prev_center

        if dt > 0:
            self.velocity_x = (cx - prev_cx) / dt
            self.velocity_y = (cy - prev_cy) / dt
        
        # Detect stationarity (threshold: 5 pixels movement)
        movement = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
        if movement < 5:
            self.stationary_frames += 1
        else:
            self.stationary_frames = 0

        self._prev_center = (cx, cy)
        self.bbox = obj.bbox
        self.confidence = obj.confidence
        self.last_seen = obj.timestamp
        self.hits += 1
        self.misses = 0
        self.history.append(obj.bbox)
        if len(self.history) > 100:
            self.history.pop(0)

    def predict(self) -> BoundingBox:
        """Simple linear prediction of next position."""
        return BoundingBox(
            x1=self.bbox.x1 + self.velocity_x,
            y1=self.bbox.y1 + self.velocity_y,
            x2=self.bbox.x2 + self.velocity_x,
            y2=self.bbox.y2 + self.velocity_y,
        )

    def to_detected_object(self) -> DetectedObject:
        return DetectedObject(
            track_id=self.track_id,
            class_name=self.class_name,
            confidence=self.confidence,
            bbox=self.bbox,
            timestamp=self.last_seen,
            camera_id=self.camera_id,
            velocity_x=self.velocity_x,
            velocity_y=self.velocity_y,
        )

    @property
    def speed(self) -> float:
        return (self.velocity_x ** 2 + self.velocity_y ** 2) ** 0.5

    @property
    def age_seconds(self) -> float:
        return (datetime.utcnow() - self.first_seen).total_seconds()


class Tracker:
    """
    IoU-based multi-object tracker.
    Assigns persistent track IDs across frames.
    """

    IOU_THRESHOLD = 0.3
    MAX_MISSES = 10  # Frames before a track is dropped

    def __init__(self, camera_id: str = "default"):
        self.camera_id = camera_id
        self._tracks: dict[int, Track] = {}
        self._next_id = 1
        self._last_timestamp: datetime | None = None

    def update(self, detections: FrameDetections) -> FrameDetections:
        """
        Match detections to existing tracks using IoU.
        Returns FrameDetections with assigned track IDs.
        """
        now = detections.timestamp
        dt = 0.0
        if self._last_timestamp:
            dt = max(0.001, (now - self._last_timestamp).total_seconds())
        self._last_timestamp = now

        # Predict new locations for existing tracks
        predicted = {tid: t.predict() for tid, t in self._tracks.items()}

        # Build IoU matrix: tracks x detections
        track_ids = list(self._tracks.keys())
        dets = detections.objects

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        if track_ids and dets:
            iou_matrix = np.zeros((len(track_ids), len(dets)))
            for i, tid in enumerate(track_ids):
                for j, det in enumerate(dets):
                    iou_matrix[i, j] = iou(predicted[tid], det.bbox)

            # Greedy matching (Hungarian-lite)
            while True:
                if iou_matrix.size == 0:
                    break
                max_val = iou_matrix.max()
                if max_val < self.IOU_THRESHOLD:
                    break
                i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                tid = track_ids[i]
                self._tracks[tid].update(dets[j], dt)
                matched_tracks.add(tid)
                matched_dets.add(j)
                iou_matrix[i, :] = 0
                iou_matrix[:, j] = 0

        # Create new tracks for unmatched detections
        for j, det in enumerate(dets):
            if j not in matched_dets:
                new_track = Track(self._next_id, det)
                self._tracks[self._next_id] = new_track
                matched_tracks.add(self._next_id)
                self._next_id += 1

        # Increment misses for unmatched tracks; remove stale ones
        stale = []
        for tid in list(self._tracks.keys()):
            if tid not in matched_tracks:
                self._tracks[tid].misses += 1
                if self._tracks[tid].misses > self.MAX_MISSES:
                    stale.append(tid)
        for tid in stale:
            del self._tracks[tid]

        # Build output detections with assigned track IDs
        tracked_objects = [t.to_detected_object() for t in self._tracks.values()]

        return FrameDetections(
            camera_id=detections.camera_id,
            frame_number=detections.frame_number,
            timestamp=now,
            objects=tracked_objects,
            fps=detections.fps,
        )

    @property
    def active_tracks(self) -> dict[int, Track]:
        return self._tracks.copy()

    def get_track(self, track_id: int) -> Track | None:
        return self._tracks.get(track_id)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._last_timestamp = None
