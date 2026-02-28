# 👁 WitnessAI
### Real-Time Crime Scene Intelligence Agent
**Vision Possible: Agent Protocol Hackathon 2026 — Built to Win**

---

## What It Does

WitnessAI transforms any live video feed into an **intelligent, always-on legal witness**. It watches, listens, narrates aloud, and documents — automatically.

- 🎥 **YOLOv8 + IoU Tracker** detects and tracks every person in the scene
- 🧠 **Gemini Realtime at 5fps** watches the video and responds to voice questions
- 🎙 **Deepgram STT** — operator speaks naturally: *"What do you see? Package the evidence."*
- 🔊 **ElevenLabs TTS** — WitnessAI narrates incidents aloud the moment they happen
- ⚙️ **5 SDK Tools** — Gemini calls backend functions mid-conversation to query status, retrieve narratives, and trigger evidence packaging
- 📊 **Live Operator Dashboard** at `http://localhost:8000` — anomaly log, metrics, narrative feed, evidence download
- 📦 **Auto Evidence Packages** — JSON report + MP4 clip saved on every confirmed incident

---

## Architecture

```
Stream WebRTC Edge (sub-30ms)
         │
         ▼
┌─────────────────────────────────────┐
│        Vision Agents SDK            │
│  ┌──────────────────────────────┐   │
│  │  WitnessProcessor            │   │  ← VideoProcessorPublisher
│  │  (witness_processor.py)      │   │
│  │  • av.VideoFrame decode      │   │
│  │  • WitnessAgent pipeline     │   │
│  │  • Frame annotation + HUD    │   │
│  │  • Latency measurement       │   │
│  │  • WebSocket broadcast       │   │
│  └──────────────────────────────┘   │
│                                      │
│  gemini.Realtime(fps=5)  ←→ video   │  ← Watches & understands at 5fps
│  deepgram.STT()          ←→ audio   │  ← Operator speaks to the agent
│  elevenlabs.TTS()        ←→ audio   │  ← Agent speaks alerts aloud
│                                      │
│  Tools (5 SDK function_tools):       │  ← Gemini calls these mid-conversation
│    get_agent_status()               │
│    get_scene_description()          │
│    get_incident_narrative()         │
│    package_incident_evidence()      │
│    list_evidence_packages()         │
└─────────────────────────────────────┘
         │                   │
         ▼                   ▼
  Annotated feed      FastAPI + WS Server
  (back to call)      http://localhost:8000
                      /api/v1/incidents
                      /api/v1/packages/{id}/report
                      /ws/live  ← React dashboard
```

---

## Setup (3 steps)

### Step 1 — Get API keys (all free tiers available)

| Key | Where | Used for |
|---|---|---|
| `STREAM_API_KEY` + `STREAM_API_SECRET` | [getstream.io/try-for-free](https://getstream.io/try-for-free) | WebRTC video edge |
| `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com) | Gemini Realtime 5fps |
| `DEEPGRAM_API_KEY` | [deepgram.com](https://deepgram.com) | Operator voice input |
| `ELEVENLABS_API_KEY` | [elevenlabs.io](https://elevenlabs.io) | Agent voice output |

### Step 2 — Install

```bash
bash setup.sh
# OR manually:
pip install "vision-agents[getstream,gemini,ultralytics,deepgram,elevenlabs]" \
            fastapi uvicorn opencv-python-headless python-dotenv scipy
```

### Step 3 — Configure & run

```bash
# Edit .env — add your 4 API keys
python main.py join

# Dashboard opens at http://localhost:8000
# API docs at       http://localhost:8000/docs
```

---

## What the Judge Sees

| Criterion | What we do | Score |
|---|---|---|
| **Potential Impact** | Replaces missing documentation layer in ALL existing security systems | ✅ |
| **Creativity** | "AI Legal Witness" — documents, not just detects. Pre-crime buffer. | ✅ |
| **Technical Excellence** | YOLOv8 + IoU tracker + anomaly rules engine + 98 tests | ✅ |
| **Real-Time Performance** | 5fps Gemini, <50ms pipeline latency shown live on HUD | ✅ |
| **User Experience** | Live operator dashboard + voice in/out + annotated video feed | ✅ |
| **Best Use of Vision Agents** | `Realtime`, `STT`, `TTS`, `VideoProcessorPublisher`, 5 `function_tool`s, `Edge` | ✅ |

---

## Anomaly Detection

| Anomaly | Detection Method | Default Threshold |
|---|---|---|
| Loitering | Stationary frames counter | 30 seconds |
| Running | Track velocity (px/sec) | 150 px/s |
| Crowd Surge | Person count in frame | 8 persons |
| Fall Detected | Bounding box aspect ratio (prone) | width/height > 2.5 |

All configurable in `.env`.

---

## Voice Interaction Examples

Operator says → WitnessAI responds:

> *"What do you see?"* → calls `get_scene_description()`, narrates scene aloud
> *"What happened?"* → calls `get_incident_narrative('latest')`, reads full report
> *"Package the evidence"* → calls `package_incident_evidence('latest')`, confirms via voice
> *"How many packages do we have?"* → calls `list_evidence_packages()`, gives count

---

## Evidence Package Contents

```
evidence_packages/
└── {incident_id}/
    ├── {package_id}.mp4           ← 30s pre + post crime video
    └── {package_id}_report.json  ← Full report:
                                      - Timestamped anomaly log
                                      - AI-generated narrative (LLM)
                                      - Track IDs + velocities
                                      - Confidence levels
                                      - Camera metadata
```

Download via: `GET /api/v1/packages/{incident_id}/report`

---

## Tests

```bash
python -m unittest tests.test_all tests.test_integration -v
# 98 tests, 0 failures, 6 skipped (SDK integration — need real keys)
```

---

*WitnessAI — The AI that doesn't just watch. It testifies.*
