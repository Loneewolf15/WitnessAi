"""
WitnessAI — Main Entry Point
==============================
Fill in .env, then run:  python main.py join

What this does:
  - Joins Stream's WebRTC edge network (sub-30ms latency)
  - Runs YOLOv8 detection + IoU tracking on every frame
  - Detects behavioral anomalies (loitering, running, falls, crowd surge)
  - Gemini Realtime watches the video at 5fps AND listens to the operator
  - Deepgram STT converts operator speech → text questions for the agent
  - ElevenLabs TTS speaks incident narrations aloud in real time
  - SDK tool calling lets Gemini trigger evidence packaging + status queries
  - FastAPI + WebSocket server feeds the React dashboard live data
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

# ── Vision Agents SDK (only needed in --sdk mode) ─────────────────
try:
    from vision_agents.core import Agent, AgentLauncher, User, Runner
    from vision_agents.core.tools import function_tool
    from vision_agents.plugins import getstream, gemini, deepgram, elevenlabs
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

    # Dummy decorator for when SDK is not installed (e.g., on Railway)
    def function_tool(func):
        return func


# ── Our backend ───────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from models.schemas import CameraConfig
from core.agent import WitnessAgent
from core.correlator import Correlator
from narration.narrator import Narrator, build_llm
from evidence.packager import Packager
from integration.witness_processor import WitnessProcessor
from api.websocket import manager as ws_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("witnessai.main")


# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────
CALL_TYPE            = os.getenv("CALL_TYPE", "default")
CALL_ID              = os.getenv("CALL_ID", "witnessai-demo-001")
PROCESSING_FPS       = int(os.getenv("PROCESSING_FPS", "5"))
LOITERING_THRESHOLD  = int(os.getenv("LOITERING_THRESHOLD_SECONDS", "30"))
RUNNING_VELOCITY     = float(os.getenv("RUNNING_VELOCITY_THRESHOLD", "150"))
CROWD_DENSITY        = int(os.getenv("CROWD_DENSITY_THRESHOLD", "8"))
OBJECT_ABANDONMENT   = int(os.getenv("OBJECT_ABANDONMENT_SECONDS", "20"))
EVIDENCE_OUTPUT_DIR  = os.getenv("EVIDENCE_OUTPUT_DIR", "./evidence_packages")
GOOGLE_API_KEY       = os.getenv("GOOGLE_API_KEY", "")
DASHBOARD_PORT       = int(os.getenv("PORT", os.getenv("DASHBOARD_PORT", "8000")))


# ─────────────────────────────────────────────────────────────────
# Build backend (once, shared across all calls)
# ─────────────────────────────────────────────────────────────────
def build_witness_agent() -> WitnessAgent:
    config = CameraConfig(
        camera_id="cam-main",
        name="WitnessAI Primary Feed",
        source="stream://webrtc",
        loitering_threshold=LOITERING_THRESHOLD,
        running_velocity_threshold=RUNNING_VELOCITY,
        crowd_density_threshold=CROWD_DENSITY,
        object_abandonment_seconds=OBJECT_ABANDONMENT,
    )
    llm_backend = build_llm(
        "gemini" if GOOGLE_API_KEY else "mock",
        GOOGLE_API_KEY
    )
    agent = WitnessAgent(
        config=config,
        narrator=Narrator(llm=llm_backend),
        packager=Packager(output_dir=EVIDENCE_OUTPUT_DIR),
        correlator=Correlator(),
        mock=False,
    )
    agent.load()
    logger.info("WitnessAI backend ready")
    return agent

_witness_agent = build_witness_agent()


# ─────────────────────────────────────────────────────────────────
# SDK Tool Definitions
# These are callable by Gemini mid-conversation. This is what makes
# the LLM actually connected to our backend, not just talking at it.
# ─────────────────────────────────────────────────────────────────

@function_tool
def get_agent_status() -> str:
    """
    Returns the current live status of WitnessAI:
    active incidents, total anomalies detected, frame count,
    and performance metrics.
    """
    status = _witness_agent.status()
    incidents = list(_witness_agent._active_incidents.values())
    return json.dumps({
        "is_running": status.is_running,
        "total_frames_processed": status.total_frames,
        "total_detections": status.total_detections,
        "total_anomalies": status.total_anomalies,
        "active_incidents": status.active_incidents,
        "active_incident_ids": [i.incident_id for i in incidents],
        "tracked_persons": len(_witness_agent._tracker.active_tracks),
        "fps": status.fps,
    })


@function_tool
def get_incident_narrative(incident_id: str) -> str:
    """
    Returns the full AI-generated timestamped narrative for a specific incident.
    Use this when the operator asks 'what happened?' or 'give me the report'.

    Args:
        incident_id: The incident ID to retrieve narrative for.
                     Use 'latest' to get the most recent incident.
    """
    if incident_id == "latest":
        if not _witness_agent._active_incidents:
            return json.dumps({"error": "No active incidents"})
        incident_id = list(_witness_agent._active_incidents.keys())[-1]

    narrative = _witness_agent._narrator.get_narrative(incident_id)
    if not narrative:
        return json.dumps({"error": f"No narrative found for incident {incident_id}"})

    return json.dumps({
        "incident_id": incident_id,
        "camera_id": narrative.camera_id,
        "started_at": narrative.started_at.isoformat(),
        "entry_count": len(narrative.entries),
        "full_narrative": narrative.full_text(),
    })


@function_tool
def package_incident_evidence(incident_id: str) -> str:
    """
    Manually triggers evidence packaging for an incident.
    Compiles video clip + AI narrative + metadata into a downloadable package.
    Call this when the operator says 'package the evidence' or 'save the incident'.

    Args:
        incident_id: The incident to package. Use 'latest' for most recent.
    """
    if incident_id == "latest":
        if not _witness_agent._active_incidents:
            return json.dumps({"error": "No active incidents to package"})
        incident_id = list(_witness_agent._active_incidents.keys())[-1]

    incident = _witness_agent._active_incidents.get(incident_id)
    if not incident:
        return json.dumps({"error": f"Incident {incident_id} not found or already packaged"})

    narrative = _witness_agent._narrator.get_narrative(incident_id)
    pre_frames = _witness_agent._buffer.snapshot()

    package = _witness_agent._packager.build(
        incident_id=incident_id,
        camera_id=incident.camera_id,
        pre_crime_frames=pre_frames,
        post_crime_frames=[],
        anomaly_events=incident.anomaly_events,
        narrative=narrative,
    )

    # Broadcast to dashboard
    asyncio.create_task(ws_manager.broadcast_raw({
        "event_type": "package_ready",
        "incident_id": incident_id,
        "package_id": package.package_id,
        "report_path": package.report_path,
        "total_anomalies": len(incident.anomaly_events),
        "narrative": narrative.full_text() if narrative else "",
    }))

    return json.dumps({
        "success": True,
        "package_id": package.package_id,
        "report_path": package.report_path,
        "video_path": package.video_path,
        "anomaly_count": len(incident.anomaly_events),
        "narrative_entries": len(narrative.entries) if narrative else 0,
    })


@function_tool
def list_evidence_packages() -> str:
    """
    Lists all saved evidence packages on disk.
    Returns incident IDs and their package locations.
    """
    packages = _witness_agent._packager.list_packages()
    return json.dumps({
        "total_packages": len(packages),
        "incident_ids": packages,
        "output_dir": EVIDENCE_OUTPUT_DIR,
    })


@function_tool
def get_scene_description() -> str:
    """
    Returns the current state of the scene:
    how many people are being tracked, their positions,
    speeds, and whether any are in anomalous states.
    """
    tracks = _witness_agent._tracker.active_tracks
    persons = []
    for tid, track in tracks.items():
        persons.append({
            "track_id": tid,
            "class": track.class_name,
            "speed_px_per_sec": round(track.speed, 1),
            "stationary_seconds": round(track.stationary_frames / 15.0, 1),
            "age_seconds": round(track.age_seconds, 1),
            "position": {
                "center_x": round(track.bbox.center_x, 1),
                "center_y": round(track.bbox.center_y, 1),
            },
        })
    return json.dumps({
        "tracked_persons": len(persons),
        "scene": persons,
        "active_anomaly_types": list(
            set(
                a.anomaly_type.value
                for inc in _witness_agent._active_incidents.values()
                for a in inc.anomaly_events
            )
        ),
    })


# ─────────────────────────────────────────────────────────────────
# Agent factory
# ─────────────────────────────────────────────────────────────────
async def create_agent(**kwargs) -> Agent:
    processor = WitnessProcessor(
        witness_agent=_witness_agent,
        fps=PROCESSING_FPS,
        ws_manager=ws_manager,
    )

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(
            name="WitnessAI",
            id="witness-ai-agent",
        ),
        instructions=_system_prompt(),
        # ── Real-time video at 5fps (was 1fps — that was embarrassing) ──
        llm=gemini.Realtime(fps=5),
        # ── Operator can SPEAK to the agent ──
        stt=deepgram.STT(),
        # ── Agent SPEAKS incident narrations aloud ──
        tts=elevenlabs.TTS(),
        # ── Our full detection + tracking + anomaly pipeline ──
        processors=[processor],
        # ── SDK tool calling — Gemini can trigger backend actions ──
        tools=[
            get_agent_status,
            get_incident_narrative,
            package_incident_evidence,
            list_evidence_packages,
            get_scene_description,
        ],
    )

    logger.info("Vision Agents SDK agent created — Gemini 5fps | Deepgram STT | ElevenLabs TTS | 5 tools")
    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    logger.info(f"Joining call: {call_type}/{call_id}")
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        logger.info("✅ WitnessAI is live")

        # Announce — now spoken aloud via ElevenLabs
        await agent.simple_response(
            "WitnessAI is now active and monitoring. "
            "I am watching the feed in real time for loitering, running, "
            "crowd surges, and falls. "
            "You can ask me what I see, request a scene description, "
            "or say 'package the evidence' at any time."
        )

        await agent.finish()

    _witness_agent.stop()
    logger.info("WitnessAI shut down cleanly")


def _system_prompt() -> str:
    return """You are WitnessAI — a real-time security intelligence agent with access to live video.

CAPABILITIES:
- You watch the live video feed at 5 frames per second via Gemini Realtime
- You receive automatic alerts when behavioral anomalies are detected
- You have 5 tools to query status, get narratives, and package evidence

TOOLS — use these proactively:
- get_scene_description(): Call this every 30s to give scene updates
- get_agent_status(): Call when asked about system status
- get_incident_narrative(incident_id): Call when asked 'what happened?'
- package_incident_evidence(incident_id): Call when operator says 'save' or 'package'
- list_evidence_packages(): Call when asked about saved reports

ANOMALY RESPONSE — when you receive an anomaly alert, immediately:
1. Call get_incident_narrative('latest') to get the AI-generated report
2. Speak it aloud in this format:
   "[TIME] [TYPE] — [description]. Confidence: [level]."
3. Offer to package the evidence

OPERATOR INTERACTION:
- Respond to voice questions naturally
- If asked "what do you see?" — call get_scene_description() and narrate it
- If asked "package the evidence" — call package_incident_evidence('latest')
- Keep responses under 3 sentences unless giving a full narrative

TONE: Professional, calm, factual. You are a legal witness, not an alarm system."""


# ─────────────────────────────────────────────────────────────────
# FastAPI Dashboard Server (runs alongside the agent)
# ─────────────────────────────────────────────────────────────────
async def start_dashboard_server():
    """Start the FastAPI server wired to the live agent."""
    try:
        import uvicorn
        from api.main import create_app
        app = create_app(witness_agent=_witness_agent)
        config = uvicorn.Config(app, host="0.0.0.0", port=DASHBOARD_PORT, log_level="warning")
        server = uvicorn.Server(config)
        logger.info(f"Dashboard + API: http://localhost:{DASHBOARD_PORT}")
        logger.info(f"API docs:        http://localhost:{DASHBOARD_PORT}/docs")
        await server.serve()
    except ImportError:
        logger.warning("uvicorn/fastapi not installed — dashboard disabled")


# ─────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("""
  ╔══════════════════════════════════════════════════════════╗
  ║   👁  WitnessAI — Real-Time Crime Scene Intelligence    ║
  ║   Vision Possible: Agent Protocol Hackathon 2026        ║
  ╠══════════════════════════════════════════════════════════╣
  ║  Gemini Realtime 5fps | Deepgram STT | ElevenLabs TTS  ║
  ║  YOLOv8 Detection | IoU Tracking | 5 SDK Tools         ║
  ╚══════════════════════════════════════════════════════════╝
    """)

    use_sdk = "join" in sys.argv or "--sdk" in sys.argv

    if use_sdk and SDK_AVAILABLE:
        # ── LOCAL MODE: Run the heavy AI agent and connect via WebRTC ──
        logger.info("Starting in Agent SDK mode (handling 'join')")
        
        # Start the dashboard server in the background so local UI works too
        import threading
        def _run_dashboard():
            # Create a new event loop for this thread since uvicorn needs one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(start_dashboard_server())

        t = threading.Thread(target=_run_dashboard, daemon=True)
        t.start()
        
        # Give the Stream SDK control over the command line arguments
        # This is what actually parses 'python main.py join' and connects
        # to the CALL_ID defined in your .env file!
        Runner(AgentLauncher(
            create_agent=create_agent,
            join_call=join_call,
        )).cli()

    else:
        # ── RAILWAY / SERVER MODE: Just run the web server and WebSockets ──
        logger.info("Starting in Server mode (Dashboard + API only)")
        import uvicorn
        from api.main import create_app
        app = create_app(witness_agent=_witness_agent)
        
        logger.info(f"Dashboard + API: http://0.0.0.0:{DASHBOARD_PORT}")
        logger.info(f"API docs:        http://0.0.0.0:{DASHBOARD_PORT}/docs")
        
        # If railway_ws is set, start relaying to the central server
        railway_ws = os.getenv("RAILWAY_WS_URL", "")
        if railway_ws:
            @app.on_event("startup")
            async def _start_relay():
                try:
                    from api.websocket import manager as ws_mgr
                    ws_mgr.start_relay(railway_ws)
                    logger.info(f"Relaying live events → {railway_ws}")
                except AttributeError:
                    pass
                
        uvicorn.run(app, host="0.0.0.0", port=DASHBOARD_PORT, log_level="info")