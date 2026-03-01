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

# ── Vision Agents SDK ─────────────────────────────────────────────
try:
    from vision_agents.core import Agent, AgentLauncher, User, Runner
    from vision_agents.core.utils.audio_filter import FirstSpeakerWinsFilter
    from vision_agents.plugins import getstream, gemini, deepgram, elevenlabs
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

# Passthrough decorator — real registration happens on the llm instance in create_agent
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
# Build backend
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
# ─────────────────────────────────────────────────────────────────

@function_tool
def get_agent_status() -> str:
    """Returns current live status of WitnessAI."""
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
    Returns the full AI-generated narrative for an incident.
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
    Packages evidence for an incident. Use 'latest' for most recent.
    Call when operator says 'package the evidence' or 'save the incident'.
    """
    if incident_id == "latest":
        if not _witness_agent._active_incidents:
            return json.dumps({"error": "No active incidents to package"})
        incident_id = list(_witness_agent._active_incidents.keys())[-1]

    incident = _witness_agent._active_incidents.get(incident_id)
    if not incident:
        return json.dumps({"error": f"Incident {incident_id} not found"})

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
    """Lists all saved evidence packages on disk."""
    packages = _witness_agent._packager.list_packages()
    return json.dumps({
        "total_packages": len(packages),
        "incident_ids": packages,
        "output_dir": EVIDENCE_OUTPUT_DIR,
    })


@function_tool
def get_scene_description() -> str:
    """Returns the current scene state — tracked persons, positions, speeds."""
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

    # fps=0 disables video forwarding to Gemini — native-audio models don't support
    # video frame input. WitnessProcessor + YOLO handle video analysis locally.
    llm = gemini.Realtime(fps=0)
    for fn in [
        get_agent_status,
        get_incident_narrative,
        package_incident_evidence,
        list_evidence_packages,
        get_scene_description,
    ]:
        llm.register_function()(fn)

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(
            name="WitnessAI",
            id="witness-ai-agent",
        ),
        instructions=_system_prompt(),
        llm=llm,
        stt=deepgram.STT(),
        tts=elevenlabs.TTS(),
        processors=[processor],
        multi_speaker_filter=FirstSpeakerWinsFilter(model_dir=os.path.join(os.path.dirname(__file__), ".cache", "vad"))
    )

    logger.info("Agent ready — Gemini native-audio | Deepgram STT | ElevenLabs TTS | YOLO vision | 5 tools registered")
    return agent


# ─────────────────────────────────────────────────────────────────
# Monkey-patch the SDK's open_demo to use VP8 at 720p
# H264 at 12Mbps/1080p causes aiortc decode failures; VP8 is reliable.
# ─────────────────────────────────────────────────────────────────
from urllib.parse import urlencode
import webbrowser

_orig_open_demo = getstream.Edge.open_demo

async def _patched_open_demo(self, call):
    """Open demo with VP8 at 720p instead of H264 at 1080p."""
    client = call.client.stream
    human_id = "user-demo-agent"
    name = "Human User"

    await client.create_user(name=name, id=human_id)

    # Ensure channel exists
    from getstream.models import ChannelInput, ChannelMemberRequest
    channel = client.chat.channel(self.channel_type, call.id)
    await channel.get_or_create(
        data=ChannelInput(
            created_by_id=self.agent_user_id,
            members=[ChannelMemberRequest(user_id=human_id)],
        )
    )

    token = client.create_token(human_id, expiration=3600)
    base_url = f"{os.getenv('EXAMPLE_BASE_URL', 'https://getstream.io/video/demos')}/join/"
    params = {
        "api_key": client.api_key,
        "token": token,
        "skip_lobby": "true",
        "user_name": name,
        "video_encoder": "vp8",      # VP8 decodes reliably in aiortc
        "bitrate": 2000000,           # 2 Mbps — much less overhead
        "w": 1280,
        "h": 720,
        "channel_type": self.channel_type,
    }
    url = f"{base_url}{call.id}?{urlencode(params)}"
    logger.info(f"🌐 Opening browser to: {url}")
    try:
        await asyncio.to_thread(webbrowser.open, url)
        logger.info("✅ Browser opened successfully!")
    except Exception as e:
        logger.warning(f"Please manually open: {url}")
    return url

getstream.Edge.open_demo = _patched_open_demo


# ─────────────────────────────────────────────────────────────────
# Join call with auto-retry for Gemini transient failures
# ─────────────────────────────────────────────────────────────────
MAX_RETRIES = 3

async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    logger.info(f"Joining call: {call_type}/{call_id}")
    
    # Retry loop for agent.create_call to handle ConnectTimeouts
    call = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            call = await agent.create_call(call_type, call_id)
            break
        except Exception as e:
            logger.warning(f"Error creating call (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2 * attempt)
            else:
                raise

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with agent.join(call):
                logger.info("WitnessAI is live in the call")
                await agent.simple_response(
                    "WitnessAI is now active and monitoring. "
                    "I am watching the feed in real time for loitering, running, "
                    "crowd surges, and falls. "
                    "You can ask me what I see, request a scene description, "
                    "or say package the evidence at any time."
                )
                await agent.finish()
            break  # clean exit
        except Exception as e:
            err_msg = str(e)
            if "1011" in err_msg or "unavailable" in err_msg.lower():
                logger.warning(f"Gemini transient error (attempt {attempt}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(2 * attempt)  # backoff
                    continue
            raise  # non-transient or out of retries

    _witness_agent.stop()
    logger.info("WitnessAI shut down cleanly")


def _system_prompt() -> str:
    return """You are WitnessAI — a real-time security intelligence agent.

CAPABILITIES:
- YOLOv8 computer vision processes the live video feed locally for detections
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
2. Speak it aloud in this format: "[TIME] [TYPE] — [description]. Confidence: [level]."
3. Offer to package the evidence

OPERATOR INTERACTION:
- Respond to voice questions naturally
- If asked "what do you see?" — call get_scene_description() and narrate it
- If asked "package the evidence" — call package_incident_evidence('latest')
- Keep responses under 3 sentences unless giving a full narrative

TONE: Professional, calm, factual. You are a legal witness, not an alarm system."""


# ─────────────────────────────────────────────────────────────────
# FastAPI Dashboard Server
# ─────────────────────────────────────────────────────────────────
async def start_dashboard_server():
    try:
        import uvicorn
        from api.main import create_app
        app = create_app(witness_agent=_witness_agent)
        config = uvicorn.Config(
            app, host="0.0.0.0", port=DASHBOARD_PORT, log_level="warning"
        )
        server = uvicorn.Server(config)
        logger.info(f"Dashboard: http://localhost:{DASHBOARD_PORT}")
        logger.info(f"API docs:  http://localhost:{DASHBOARD_PORT}/docs")
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

    use_sdk = "join" in sys.argv or "run" in sys.argv or "--sdk" in sys.argv

    if use_sdk and SDK_AVAILABLE:
        logger.info(f"SDK mode — joining call: {CALL_TYPE}/{CALL_ID}")

        # Runner uses 'run' not 'join' — inject correct args from .env
        sys.argv = [sys.argv[0], "run", "--call-type", CALL_TYPE, "--call-id", CALL_ID]
        logger.info(f"Injected args: {sys.argv}")

        # Start dashboard in background thread
        import threading
        def _run_dashboard():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(start_dashboard_server())
        threading.Thread(target=_run_dashboard, daemon=True).start()

        # Start relay to Railway if configured
        railway_ws = os.getenv("RAILWAY_WS_URL", "")
        if railway_ws:
            async def _relay():
                ws_manager.start_relay(railway_ws)
            asyncio.run(_relay())
            logger.info(f"Relaying events to Railway: {railway_ws}")

        # Hand control to the SDK Runner — this blocks until the call ends
        Runner(AgentLauncher(
            create_agent=create_agent,
            join_call=join_call,
        )).cli()

    else:
        # Server mode — Railway or local dashboard only
        logger.info("Server mode — Dashboard + API only (no SDK)")
        import uvicorn
        from api.main import create_app
        app = create_app(witness_agent=_witness_agent)

        railway_ws = os.getenv("RAILWAY_WS_URL", "")
        if railway_ws:
            @app.on_event("startup")
            async def _start_relay():
                try:
                    ws_manager.start_relay(railway_ws)
                    logger.info(f"Relaying live events to Railway: {railway_ws}")
                except AttributeError:
                    pass

        logger.info(f"Dashboard: http://0.0.0.0:{DASHBOARD_PORT}")
        uvicorn.run(app, host="0.0.0.0", port=DASHBOARD_PORT, log_level="info")