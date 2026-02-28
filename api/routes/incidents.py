"""WitnessAI — Incident + Evidence API routes (wired to live WitnessAgent)"""
from __future__ import annotations
import os
import json
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# Injected at startup by main.py — avoids circular import
_witness_agent = None

def set_agent(agent) -> None:
    global _witness_agent
    _witness_agent = agent


@router.get("/incidents")
async def list_incidents():
    """List all active incidents from the live agent."""
    if _witness_agent is None:
        return {"incidents": [], "total": 0}

    result = []
    for inc_id, incident in _witness_agent._active_incidents.items():
        narrative = _witness_agent._narrator.get_narrative(inc_id)
        result.append({
            "incident_id": inc_id,
            "camera_id": incident.camera_id,
            "status": incident.status.value,
            "created_at": incident.created_at.isoformat(),
            "anomaly_count": len(incident.anomaly_events),
            "anomaly_types": list({a.anomaly_type.value for a in incident.anomaly_events}),
            "narrative_entries": len(narrative.entries) if narrative else 0,
            "narrative_preview": (
                narrative.entries[-1].text[:120] if narrative and narrative.entries else ""
            ),
        })
    return {"incidents": result, "total": len(result)}


@router.get("/incidents/{incident_id}")
async def get_incident(incident_id: str):
    if _witness_agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised")
    incident = _witness_agent._active_incidents.get(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    narrative = _witness_agent._narrator.get_narrative(incident_id)
    return {
        "incident_id": incident_id,
        "camera_id": incident.camera_id,
        "status": incident.status.value,
        "created_at": incident.created_at.isoformat(),
        "updated_at": incident.updated_at.isoformat(),
        "anomaly_events": [
            {
                "id": a.id,
                "type": a.anomaly_type.value,
                "confidence": a.confidence.value,
                "description": a.description,
                "timestamp": a.timestamp.isoformat(),
                "track_ids": a.involved_track_ids,
                "metadata": a.metadata,
            }
            for a in incident.anomaly_events
        ],
        "narrative": {
            "full_text": narrative.full_text() if narrative else "",
            "entries": [
                {"timestamp": e.timestamp.isoformat(), "text": e.text}
                for e in (narrative.entries if narrative else [])
            ],
        },
    }


@router.get("/incidents/{incident_id}/narrative")
async def get_narrative(incident_id: str):
    if _witness_agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised")
    narrative = _witness_agent._narrator.get_narrative(incident_id)
    if not narrative:
        raise HTTPException(status_code=404, detail="Narrative not found")
    return {
        "incident_id": incident_id,
        "full_text": narrative.full_text(),
        "entry_count": len(narrative.entries),
        "entries": [
            {"timestamp": e.timestamp.isoformat(), "text": e.text}
            for e in narrative.entries
        ],
    }


@router.get("/packages")
async def list_packages():
    """List all saved evidence packages on disk."""
    if _witness_agent is None:
        return {"packages": [], "total": 0}
    packages = _witness_agent._packager.list_packages()
    return {"packages": packages, "total": len(packages)}


@router.get("/packages/{incident_id}/report")
async def download_report(incident_id: str):
    """Download the JSON evidence report for an incident."""
    if _witness_agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised")
    output_dir = Path(_witness_agent._packager.output_dir)
    incident_dir = output_dir / incident_id
    if not incident_dir.exists():
        raise HTTPException(status_code=404, detail="Package not found")
    reports = list(incident_dir.glob("*_report.json"))
    if not reports:
        raise HTTPException(status_code=404, detail="Report not yet generated")
    return FileResponse(
        reports[0],
        media_type="application/json",
        filename=f"witnessai_report_{incident_id[:8]}.json",
    )
