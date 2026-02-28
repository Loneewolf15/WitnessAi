"""
WitnessAI — Central Configuration
All settings loaded from environment variables with sensible defaults.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


BASE_DIR = Path(__file__).parent


@dataclass
class Config:
    # ── API Keys ────────────────────────────────────────────────────────────
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    stream_api_key: str = field(default_factory=lambda: os.getenv("STREAM_API_KEY", ""))
    stream_api_secret: str = field(default_factory=lambda: os.getenv("STREAM_API_SECRET", ""))

    # ── LLM Settings ────────────────────────────────────────────────────────
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "anthropic"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "claude-3-haiku-20240307"))
    llm_max_tokens: int = 512
    llm_mock_mode: bool = field(default_factory=lambda: os.getenv("LLM_MOCK_MODE", "false").lower() == "true")

    # ── Vision / Detection ───────────────────────────────────────────────────
    yolo_model: str = field(default_factory=lambda: os.getenv("YOLO_MODEL", "yolov8n.pt"))
    yolo_confidence: float = 0.45
    yolo_device: str = field(default_factory=lambda: os.getenv("YOLO_DEVICE", "cpu"))
    moondream_model: str = "vikhyatk/moondream2"
    moondream_every_n_frames: int = 30  # run moondream every N frames
    moondream_mock_mode: bool = field(
        default_factory=lambda: os.getenv("MOONDREAM_MOCK_MODE", "true").lower() == "true"
    )

    # ── Anomaly Thresholds ───────────────────────────────────────────────────
    loiter_threshold_seconds: float = 60.0
    run_velocity_threshold: float = 80.0   # pixels/frame displacement
    crowd_surge_count: int = 5              # persons appearing suddenly
    fall_aspect_ratio_threshold: float = 1.5  # width > height * factor
    abandon_stillness_seconds: float = 30.0

    # ── Buffer / Clips ───────────────────────────────────────────────────────
    buffer_duration_seconds: int = 30
    pre_event_seconds: int = 30
    post_event_seconds: int = 30
    incidents_dir: Path = field(default_factory=lambda: BASE_DIR / "incidents")
    max_buffer_frames: int = 900   # ~30s at 30fps

    # ── Server ───────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")

    # ── Targeting FPS ────────────────────────────────────────────────────────
    target_fps: int = 15

    def __post_init__(self):
        self.incidents_dir.mkdir(parents=True, exist_ok=True)


# Singleton
settings = Config()
