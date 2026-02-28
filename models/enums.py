"""
WitnessAI — Enums
"""
from enum import Enum


class AnomalyType(str, Enum):
    LOITERING = "loitering"
    RUNNING = "running"
    CROWD_SURGE = "crowd_surge"
    FALL = "fall"
    ABANDONED_OBJECT = "abandoned_object"
    RAPID_DEPARTURE = "rapid_departure"
    UNKNOWN = "unknown"


class FeedStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


class IncidentStatus(str, Enum):
    DETECTED = "detected"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    MOCK = "mock"
