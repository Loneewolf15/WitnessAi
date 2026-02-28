"""
WitnessAI — Frame Processor
Integrates YOLOv8 (detection) + DeepSORT (tracking) + Moondream (scene description).
Designed to be called per-frame in a processing loop.
"""
import logging
import time
from datetime import datetime
from typing import List, Optional, Dict, Tuple
import numpy as np

from config import settings
from models.schemas import Detection, BoundingBox, FrameResult, TrackedObject

logger = logging.getLogger(__name__)


class YOLODetector:
    """
    Wraps Ultralytics YOLOv8 for object detection.
    Lazy-loads the model on first use to keep startup fast.
    """

    TARGET_CLASSES = {"person", "car", "truck", "backpack", "handbag", "suitcase", "cell phone", "laptop"}

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO  # type: ignore
            self._model = YOLO(settings.yolo_model)
            logger.info(f"YOLOv8 loaded: {settings.yolo_model}")
        except ImportError:
            logger.warning("ultralytics not installed; using mock detector")
            self._model = "mock"

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run detection on a single frame.

        Returns:
            List of dicts with keys: class_name, confidence, bbox (x1,y1,x2,y2).
        """
        self._load()

        if self._model == "mock":
            return self._mock_detect(frame)

        results = self._model(
            frame,
            conf=settings.yolo_confidence,
            device=settings.yolo_device,
            verbose=False,
        )
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                class_name = self._model.names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "class_name": class_name,
                    "confidence": float(box.conf[0]),
                    "bbox": (x1, y1, x2, y2),
                })
        return detections

    def _mock_detect(self, frame: np.ndarray) -> List[Dict]:
        """Return a deterministic mock detection for testing."""
        h, w = frame.shape[:2] if len(frame.shape) >= 2 else (480, 640)
        return [
            {
                "class_name": "person",
                "confidence": 0.92,
                "bbox": (w * 0.3, h * 0.2, w * 0.5, h * 0.9),
            }
        ]


class DeepSORTTracker:
    """
    Wraps deep-sort-realtime for multi-object tracking across frames.
    """

    def __init__(self):
        self._tracker = None

    def _load(self):
        if self._tracker is not None:
            return
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort  # type: ignore
            self._tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)
            logger.info("DeepSORT tracker initialized")
        except ImportError:
            logger.warning("deep-sort-realtime not installed; using mock tracker")
            self._tracker = "mock"
            self._mock_id_counter = 0

    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Tuple[int, str, float, Tuple]]:
        """
        Update tracker with new detections.

        Returns:
            List of (track_id, class_name, confidence, (x1,y1,x2,y2))
        """
        self._load()

        if self._tracker == "mock":
            return self._mock_update(detections)

        # DeepSort expects [[x1,y1,w,h], confidence, class]
        ds_input = []
        class_map = {}
        conf_map = {}
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            w, h = x2 - x1, y2 - y1
            ds_input.append(([x1, y1, w, h], det["confidence"], det["class_name"]))

        tracks = self._tracker.update_tracks(ds_input, frame=frame)

        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            results.append((
                track.track_id,
                track.det_class or "unknown",
                track.det_conf or 1.0,
                tuple(ltrb),
            ))
        return results

    def _mock_update(self, detections: List[Dict]) -> List[Tuple]:
        results = []
        for i, det in enumerate(detections):
            results.append((i + 1, det["class_name"], det["confidence"], det["bbox"]))
        return results


class MoondreamDescriber:
    """
    Uses Moondream2 to generate natural language scene descriptions.
    Runs asynchronously every N frames to avoid blocking the detection loop.
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        if settings.moondream_mock_mode:
            self._model = "mock"
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            import torch
            self._tokenizer = AutoTokenizer.from_pretrained(
                settings.moondream_model, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                settings.moondream_model, trust_remote_code=True
            )
            logger.info("Moondream2 loaded")
        except ImportError:
            logger.warning("transformers not installed; using mock describer")
            self._model = "mock"

    def describe(self, frame: np.ndarray, question: str = "Describe what is happening in this scene.") -> str:
        """Generate a scene description for the given frame."""
        self._load()

        if self._model == "mock":
            return self._mock_describe(frame)

        try:
            from PIL import Image  # type: ignore
            import torch
            img = Image.fromarray(frame[..., ::-1])  # BGR → RGB
            enc_image = self._model.encode_image(img)
            answer = self._model.answer_question(enc_image, question, self._tokenizer)
            return answer
        except Exception as e:
            logger.error(f"Moondream describe failed: {e}")
            return "Scene analysis unavailable."

    def _mock_describe(self, frame: np.ndarray) -> str:
        return "A person is visible in the scene, moving through an indoor environment."


class FrameProcessor:
    """
    Orchestrates the full per-frame analysis pipeline:
    Detection → Tracking → (optional) Scene Description → FrameResult
    """

    def __init__(self):
        self.detector = YOLODetector()
        self.tracker = DeepSORTTracker()
        self.describer = MoondreamDescriber()
        self._frame_counter: Dict[str, int] = {}
        self._tracked_objects: Dict[str, Dict[int, TrackedObject]] = {}

    def process_frame(
        self,
        feed_id: str,
        frame: np.ndarray,
        frame_number: Optional[int] = None,
    ) -> FrameResult:
        """
        Full pipeline for one frame.

        Args:
            feed_id: Identifier for the camera feed.
            frame: BGR numpy array from OpenCV/Stream SDK.
            frame_number: Sequential frame index (auto-incremented if None).

        Returns:
            FrameResult with detections, tracked objects, and optional scene description.
        """
        t0 = time.perf_counter()

        # Track frame counter per feed
        if feed_id not in self._frame_counter:
            self._frame_counter[feed_id] = 0
        if frame_number is None:
            frame_number = self._frame_counter[feed_id]
        self._frame_counter[feed_id] += 1

        # Step 1: Detect
        raw_detections = self.detector.detect(frame)

        # Step 2: Track
        tracks = self.tracker.update(raw_detections, frame)

        # Step 3: Build Detection objects + update tracked objects
        detections = []
        now = datetime.utcnow()
        if feed_id not in self._tracked_objects:
            self._tracked_objects[feed_id] = {}

        for track_id, class_name, confidence, bbox_tuple in tracks:
            x1, y1, x2, y2 = bbox_tuple
            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            det = Detection(
                track_id=track_id,
                class_name=class_name,
                confidence=confidence,
                bbox=bbox,
                frame_number=frame_number,
                timestamp=now,
            )
            detections.append(det)

            # Update TrackedObject registry
            if track_id in self._tracked_objects[feed_id]:
                obj = self._tracked_objects[feed_id][track_id]
                obj.last_seen = now
                obj.bbox = bbox
                obj.confidence = confidence
                obj.frame_history.append(frame_number)
                obj.bbox_history.append(bbox)
                # Keep history bounded
                if len(obj.frame_history) > 300:
                    obj.frame_history = obj.frame_history[-300:]
                    obj.bbox_history = obj.bbox_history[-300:]
            else:
                self._tracked_objects[feed_id][track_id] = TrackedObject(
                    track_id=track_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=bbox,
                    frame_history=[frame_number],
                    bbox_history=[bbox],
                )

        # Step 4: Optional scene description (every N frames)
        scene_desc = None
        if frame_number % settings.moondream_every_n_frames == 0:
            scene_desc = self.describer.describe(frame)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return FrameResult(
            feed_id=feed_id,
            frame_number=frame_number,
            timestamp=now,
            detections=detections,
            scene_description=scene_desc,
            processing_time_ms=elapsed_ms,
        )

    def get_tracked_objects(self, feed_id: str) -> Dict[int, TrackedObject]:
        return self._tracked_objects.get(feed_id, {})

    def reset_feed(self, feed_id: str) -> None:
        self._frame_counter.pop(feed_id, None)
        self._tracked_objects.pop(feed_id, None)
