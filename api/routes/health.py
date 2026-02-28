"""WitnessAI - Health check endpoints"""
from fastapi import APIRouter
from datetime import datetime

router = APIRouter()


@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "WitnessAI",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }
