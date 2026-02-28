"""WitnessAI — Agent status endpoints (wired to live WitnessAgent)"""
from __future__ import annotations
import logging
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)
router = APIRouter()

_witness_agent = None

def set_agent(agent) -> None:
    global _witness_agent
    _witness_agent = agent


@router.get("/agents")
async def list_agents():
    if _witness_agent is None:
        return {"agents": [], "count": 0}
    status = _witness_agent.status()
    return {
        "agents": [
            {
                "camera_id": status.camera_id,
                "is_running": status.is_running,
                "fps": status.fps,
                "total_frames": status.total_frames,
                "total_detections": status.total_detections,
                "total_anomalies": status.total_anomalies,
                "active_incidents": status.active_incidents,
                "tracked_persons": len(_witness_agent._tracker.active_tracks),
            }
        ],
        "count": 1,
    }


@router.get("/agents/{camera_id}/status")
async def agent_status(camera_id: str):
    if _witness_agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised")
    if camera_id != _witness_agent.camera_id:
        raise HTTPException(status_code=404, detail=f"Agent {camera_id} not found")
    status = _witness_agent.status()
    return {
        "camera_id": status.camera_id,
        "is_running": status.is_running,
        "fps": status.fps,
        "total_frames": status.total_frames,
        "total_detections": status.total_detections,
        "total_anomalies": status.total_anomalies,
        "active_incidents": status.active_incidents,
        "tracked_persons": len(_witness_agent._tracker.active_tracks),
    }
