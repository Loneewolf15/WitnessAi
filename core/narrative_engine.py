"""
WitnessAI — Narrative Engine
Transforms raw tracking events and scene descriptions into
timestamped, human-readable incident narratives via LLM.
"""
import logging
import asyncio
from datetime import datetime
from typing import List, Optional, Dict
from collections import deque

from config import settings
from models.schemas import (
    AnomalyEvent, TrackedObject, NarrativeEntry, FrameResult
)
from models.enums import AnomalyType, LLMProvider

logger = logging.getLogger(__name__)


class NarrativeEngine:
    """
    Maintains a narrative log per feed and generates LLM-powered
    natural language summaries of incidents.
    """

    def __init__(self):
        self._logs: Dict[str, List[NarrativeEntry]] = {}
        self._event_queue: Dict[str, deque] = {}
        self._llm_client = None

    # ── Public API ────────────────────────────────────────────────────────────

    def log_frame(
        self,
        frame_result: FrameResult,
        tracked_objects: Dict[int, TrackedObject],
        anomaly_events: List[AnomalyEvent],
    ) -> List[NarrativeEntry]:
        """
        Process a frame result and return new narrative entries.
        Called every frame by the camera manager.
        """
        feed_id = frame_result.feed_id
        if feed_id not in self._logs:
            self._logs[feed_id] = []

        entries = []

        # Log scene description (from Moondream) if available
        if frame_result.scene_description:
            entry = NarrativeEntry(
                feed_id=feed_id,
                text=frame_result.scene_description,
                timestamp=frame_result.timestamp,
            )
            entries.append(entry)

        # Log tracking updates (entry / exit)
        for det in frame_result.detections:
            obj = tracked_objects.get(det.track_id)
            if obj and len(obj.frame_history) == 1:
                # First appearance
                entry = NarrativeEntry(
                    feed_id=feed_id,
                    text=(
                        f"Track-{det.track_id} ({det.class_name}) entered frame. "
                        f"Confidence: {det.confidence:.2f}."
                    ),
                    timestamp=det.timestamp,
                    track_ids=[det.track_id],
                )
                entries.append(entry)

        # Log anomaly events with extra emphasis
        for event in anomaly_events:
            entry = NarrativeEntry(
                feed_id=feed_id,
                text=event.description,
                timestamp=event.timestamp,
                is_anomaly=True,
                anomaly_type=event.anomaly_type,
                track_ids=event.track_ids,
            )
            entries.append(entry)

        self._logs[feed_id].extend(entries)
        return entries

    def get_log(self, feed_id: str) -> List[NarrativeEntry]:
        return self._logs.get(feed_id, [])

    def get_recent_log(self, feed_id: str, last_n: int = 50) -> List[NarrativeEntry]:
        return self._logs.get(feed_id, [])[-last_n:]

    def clear_log(self, feed_id: str) -> None:
        self._logs.pop(feed_id, None)

    async def generate_incident_summary(
        self,
        feed_id: str,
        anomaly_event: AnomalyEvent,
        recent_entries: Optional[List[NarrativeEntry]] = None,
    ) -> str:
        """
        Call the LLM to generate a concise incident summary.
        Returns a plain-text paragraph suitable for the PDF report.
        """
        if settings.llm_mock_mode or settings.llm_provider == LLMProvider.MOCK:
            return self._mock_summary(anomaly_event)

        entries = recent_entries or self.get_recent_log(feed_id)
        context = self._build_context(entries)

        prompt = (
            f"You are WitnessAI, a real-time security intelligence system. "
            f"Generate a concise, factual incident summary paragraph (3-5 sentences) "
            f"for a security report based on the following real-time observations:\n\n"
            f"{context}\n\n"
            f"Anomaly detected: {anomaly_event.anomaly_type.value} — {anomaly_event.description}\n\n"
            f"Write in third person, past tense, and include only observable facts. "
            f"Do not speculate about intent."
        )

        try:
            if settings.llm_provider == LLMProvider.ANTHROPIC:
                return await self._call_anthropic(prompt)
            else:
                return await self._call_gemini(prompt)
        except Exception as e:
            logger.error(f"LLM summary generation failed: {e}")
            return self._mock_summary(anomaly_event)

    # ── LLM Clients ───────────────────────────────────────────────────────────

    async def _call_anthropic(self, prompt: str) -> str:
        try:
            import anthropic  # type: ignore
            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            message = client.messages.create(
                model=settings.llm_model,
                max_tokens=settings.llm_max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except ImportError:
            logger.warning("anthropic SDK not installed")
            return self._mock_summary(None)

    async def _call_gemini(self, prompt: str) -> str:
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=settings.gemini_api_key)
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt)
            return response.text
        except ImportError:
            logger.warning("google-generativeai not installed")
            return self._mock_summary(None)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_context(self, entries: List[NarrativeEntry]) -> str:
        if not entries:
            return "No prior observations recorded."
        lines = []
        for e in entries[-20:]:  # last 20 entries for context
            ts = e.timestamp.strftime("%H:%M:%S")
            lines.append(f"[{ts}] {e.text}")
        return "\n".join(lines)

    def _mock_summary(self, event: Optional[AnomalyEvent]) -> str:
        anomaly_type = event.anomaly_type.value if event else "unknown"
        return (
            f"At {datetime.utcnow().strftime('%H:%M:%S UTC')}, WitnessAI detected a "
            f"{anomaly_type.replace('_', ' ')} event on the monitored feed. "
            f"One or more individuals were observed exhibiting behavior consistent with the anomaly classification. "
            f"A pre-crime buffer clip has been preserved for review. "
            f"This report was generated automatically and should be reviewed by a qualified security professional."
        )
