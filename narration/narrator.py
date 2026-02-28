"""WitnessAI - LLM Incident Narrative Engine"""
from __future__ import annotations
import uuid
from datetime import datetime
import logging; logger = logging.getLogger(__name__)
from models.schemas import (
    AnomalyEvent, IncidentNarrative, NarrativeEntry
)


class MockLLM:
    """Mock LLM for testing — returns deterministic narrative text."""

    def generate(self, prompt: str) -> str:
        # Extract anomaly type from prompt for realistic mock output
        if "loitering" in prompt.lower():
            return "Subject has remained stationary in the monitored zone for an extended period. Behavior is consistent with surveillance or pre-criminal observation. Recommend immediate operator review."
        elif "running" in prompt.lower():
            return "Rapid movement detected from tracked individual. Speed and trajectory suggest panic, pursuit, or unauthorized access attempt. Corroborating feeds should be reviewed."
        elif "crowd" in prompt.lower():
            return "Unusual crowd density observed in frame. Surge may indicate an incident in progress or an organized gathering. Crowd composition and behavior warrant immediate attention."
        elif "fall" in prompt.lower():
            return "Individual appears to have fallen or is in a prone position. Medical emergency protocols should be initiated. Area should be cleared for emergency responder access."
        else:
            return "Anomalous behavior detected. The scene has deviated from established baseline patterns. Operator review is recommended to assess the nature and severity of the event."


class GeminiLLM:
    """Google Gemini API wrapper (uses google-genai SDK)."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str) -> str:
        client = self._get_client()
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return response.text.strip()


class AnthropicLLM:
    """Anthropic Claude API wrapper."""

    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def generate(self, prompt: str) -> str:
        client = self._get_client()
        message = client.messages.create(
            model=self.model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()


def build_llm(provider: str = "mock", api_key: str = ""):
    """Factory: returns the configured LLM backend."""
    if provider == "gemini":
        return GeminiLLM(api_key=api_key)
    elif provider == "anthropic":
        return AnthropicLLM(api_key=api_key)
    else:
        return MockLLM()


NARRATION_PROMPT_TEMPLATE = """You are WitnessAI, an intelligent security witness system.
Generate a concise, professional, factual incident narrative entry (2-3 sentences max).
Write in past tense as if documenting for a legal incident report.

Anomaly Type: {anomaly_type}
Camera: {camera_id}
Timestamp: {timestamp}
Description: {description}
Additional Context: {metadata}

Generate the narrative entry:"""


class Narrator:
    """
    Generates timestamped, LLM-powered incident narrative entries.
    Maintains a rolling narrative per incident.
    """

    def __init__(self, llm=None):
        self._llm = llm or MockLLM()
        self._narratives: dict[str, IncidentNarrative] = {}

    def get_or_create_narrative(self, incident_id: str, camera_id: str) -> IncidentNarrative:
        if incident_id not in self._narratives:
            self._narratives[incident_id] = IncidentNarrative(
                incident_id=incident_id,
                camera_id=camera_id,
                started_at=datetime.utcnow(),
            )
        return self._narratives[incident_id]

    def narrate_anomaly(
        self, incident_id: str, anomaly: AnomalyEvent
    ) -> NarrativeEntry:
        """Generate a narrative entry for an anomaly event."""
        prompt = NARRATION_PROMPT_TEMPLATE.format(
            anomaly_type=anomaly.anomaly_type.value,
            camera_id=anomaly.camera_id,
            timestamp=anomaly.timestamp.strftime("%H:%M:%S UTC"),
            description=anomaly.description,
            metadata=str(anomaly.metadata),
        )

        try:
            text = self._llm.generate(prompt)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            text = f"[AUTO] {anomaly.description}"

        entry = NarrativeEntry(
            timestamp=anomaly.timestamp,
            text=text,
            anomaly_id=anomaly.id,
            camera_id=anomaly.camera_id,
        )

        narrative = self.get_or_create_narrative(incident_id, anomaly.camera_id)
        narrative.entries.append(entry)

        logger.info(f"[{anomaly.camera_id}] Narrative entry added: {text[:80]}...")
        return entry

    def get_narrative(self, incident_id: str) -> IncidentNarrative | None:
        return self._narratives.get(incident_id)

    def all_narratives(self) -> dict[str, IncidentNarrative]:
        return self._narratives.copy()
