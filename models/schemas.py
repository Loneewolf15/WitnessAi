"""WitnessAI - Core data models using Python dataclasses (no external deps)"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime
import uuid


class AnomalyType(str, Enum):
    LOITERING = "loitering"
    RUNNING = "running"
    CROWD_SURGE = "crowd_surge"
    ABANDONED_OBJECT = "abandoned_object"
    FALL_DETECTED = "fall_detected"
    INTRUSION = "intrusion"


class ConfidenceLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class IncidentStatus(str, Enum):
    DETECTING = "detecting"
    CONFIRMED = "confirmed"
    PACKAGED = "packaged"
    REVIEWED = "reviewed"


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class DetectedObject:
    track_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    timestamp: datetime = field(default_factory=datetime.utcnow)
    camera_id: str = "default"
    velocity_x: float = 0.0
    velocity_y: float = 0.0

    @property
    def speed(self) -> float:
        return (self.velocity_x ** 2 + self.velocity_y ** 2) ** 0.5


@dataclass
class FrameDetections:
    camera_id: str
    frame_number: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    objects: list = field(default_factory=list)
    fps: float = 0.0


@dataclass
class AnomalyEvent:
    id: str
    camera_id: str
    anomaly_type: AnomalyType
    confidence: ConfidenceLevel
    description: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    involved_track_ids: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class NarrativeEntry:
    timestamp: datetime
    text: str
    anomaly_id: Optional[str] = None
    camera_id: str = "default"


@dataclass
class IncidentNarrative:
    incident_id: str
    camera_id: str
    started_at: datetime
    entries: list = field(default_factory=list)

    def full_text(self) -> str:
        return "\n".join(
            f"{e.timestamp.strftime('%H:%M:%S')} — {e.text}"
            for e in sorted(self.entries, key=lambda x: x.timestamp)
        )


@dataclass
class EvidencePackage:
    package_id: str
    incident_id: str
    camera_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    video_path: Optional[str] = None
    report_path: Optional[str] = None
    narrative: Optional[IncidentNarrative] = None
    anomaly_events: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class Incident:
    incident_id: str
    camera_id: str
    status: IncidentStatus = IncidentStatus.DETECTING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    anomaly_events: list = field(default_factory=list)
    narrative: Optional[IncidentNarrative] = None
    evidence_package: Optional[EvidencePackage] = None


@dataclass
class CameraConfig:
    camera_id: str
    name: str
    source: str
    location: str = ""
    loitering_threshold: int = 30
    running_velocity_threshold: float = 150.0
    crowd_density_threshold: int = 8
    object_abandonment_seconds: int = 20


@dataclass
class AgentStatus:
    camera_id: str
    is_running: bool
    fps: float = 0.0
    total_frames: int = 0
    total_detections: int = 0
    total_anomalies: int = 0
    active_incidents: int = 0


@dataclass
class WebSocketMessage:
    event_type: str
    camera_id: str
    payload: dict
    timestamp: datetime = field(default_factory=datetime.utcnow)
