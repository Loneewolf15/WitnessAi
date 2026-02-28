"""
WitnessAI — Feed Routes
CRUD endpoints for managing camera feeds.
"""
from fastapi import APIRouter, HTTPException, status
from typing import List

from models.schemas import FeedCreateRequest, FeedResponse, FeedConfig
from core.camera_manager import camera_manager

router = APIRouter(prefix="/feeds", tags=["feeds"])


@router.get("/", response_model=List[FeedResponse])
async def list_feeds():
    """List all active camera feeds."""
    return camera_manager.list_feeds()


@router.post("/", response_model=FeedResponse, status_code=status.HTTP_201_CREATED)
async def create_feed(body: FeedCreateRequest):
    """Add a new camera feed and start processing."""
    config = FeedConfig(
        name=body.name,
        source=body.source,
        zones=body.zones,
    )
    worker = await camera_manager.add_feed(config)
    return FeedResponse(
        feed_id=config.feed_id,
        name=config.name,
        source=config.source,
        status=config.status,
        created_at=config.created_at,
        incident_count=0,
    )


@router.get("/{feed_id}", response_model=FeedResponse)
async def get_feed(feed_id: str):
    """Get details for a specific feed."""
    worker = camera_manager.get_worker(feed_id)
    if not worker:
        raise HTTPException(status_code=404, detail=f"Feed '{feed_id}' not found")
    cfg = worker.config
    return FeedResponse(
        feed_id=cfg.feed_id,
        name=cfg.name,
        source=cfg.source,
        status=cfg.status,
        created_at=cfg.created_at,
        incident_count=worker.incident_manager.get_incident_count(),
    )


@router.delete("/{feed_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_feed(feed_id: str):
    """Stop and remove a camera feed."""
    removed = await camera_manager.remove_feed(feed_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Feed '{feed_id}' not found")
