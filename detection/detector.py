"""WitnessAI - YOLOv8 Detection Wrapper"""
from __future__ import annotations
import numpy as np
from datetime import datetime
import logging; logger = logging.getLogger(__name__)
from models.schemas import DetectedObject, BoundingBox, FrameDetections


class MockYOLOModel:
    """Lightweight mock for testing without GPU/model download."""

    def __call__(self, frame: np.ndarray):
        return [MockResult(frame)]


class MockResult:
    def __init__(self, frame: np.ndarray):
        h, w = (frame.shape[0], frame.shape[1]) if len(frame.shape) == 3 else (480, 640)
        self.boxes = MockBoxes(w, h)


class MockBoxes:
    def __init__(self, w: int, h: int):
        self._data = [
            [0.1 * w, 0.1 * h, 0.3 * w, 0.8 * h, 0.92, 0],
            [0.6 * w, 0.15 * h, 0.85 * w, 0.85 * h, 0.87, 0],
        ]

    def __iter__(self):
        return iter(self._data)


class Detector:
    """YOLOv8-based object detector with mock fallback for testing."""

    SUPPORTED_CLASSES = {0: "person", 24: "backpack", 26: "handbag", 28: "suitcase"}

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        camera_id: str = "default",
        mock: bool = False,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.camera_id = camera_id
        self.mock = mock
        self._model = None
        self._frame_count = 0

    def load(self) -> None:
        """Load the YOLO model. Call once before detect()."""
        if self.mock:
            self._model = MockYOLOModel()
            logger.info(f"[{self.camera_id}] Detector loaded in MOCK mode")
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            logger.info(f"[{self.camera_id}] YOLOv8 model loaded: {self.model_path}")
        except ImportError:
            logger.warning("ultralytics not available — falling back to mock mode")
            self._model = MockYOLOModel()
            self.mock = True

    def detect(self, frame: np.ndarray) -> FrameDetections:
        """Run inference on a single BGR frame."""
        if self._model is None:
            raise RuntimeError("Detector not loaded. Call load() first.")

        self._frame_count += 1
        now = datetime.utcnow()
        objects: list[DetectedObject] = []

        results = self._model(frame, imgsz=320, verbose=False)

        for result in results:
            for box in result.boxes:
                if self.mock:
                    x1, y1, x2, y2, conf, cls_id = box
                else:
                    data = box.data[0].tolist()
                    x1, y1, x2, y2, conf, cls_id = data

                cls_id = int(float(cls_id))
                conf = float(conf)

                if conf < self.confidence_threshold:
                    continue
                if cls_id not in self.SUPPORTED_CLASSES:
                    continue

                obj = DetectedObject(
                    track_id=-1,
                    class_name=self.SUPPORTED_CLASSES[cls_id],
                    confidence=conf,
                    bbox=BoundingBox(
                        x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)
                    ),
                    timestamp=now,
                    camera_id=self.camera_id,
                )
                objects.append(obj)

        return FrameDetections(
            camera_id=self.camera_id,
            frame_number=self._frame_count,
            timestamp=now,
            objects=objects,
        )

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def reset(self) -> None:
        self._frame_count = 0
