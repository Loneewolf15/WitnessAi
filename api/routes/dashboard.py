"""Serve the operator dashboard HTML — no npm build step required."""
from fastapi import APIRouter
from fastapi.responses import FileResponse
import os

router = APIRouter()
_DASHBOARD = os.path.join(os.path.dirname(__file__), "..", "..", "dashboard", "index.html")

@router.get("/")
async def dashboard():
    return FileResponse(_DASHBOARD, media_type="text/html")

@router.get("/talk")
async def talk():
    _TALK = os.path.join(os.path.dirname(__file__), "..", "..", "dashboard", "talk.html")
    return FileResponse(_TALK, media_type="text/html")
