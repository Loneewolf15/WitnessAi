"""WitnessAI — WebSocket handler for real-time dashboard updates"""
from __future__ import annotations
import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)
        logger.info(f"WS client connected. Total: {len(self._connections)}")

    def disconnect(self, ws: WebSocket) -> None:
        self._connections = [c for c in self._connections if c is not ws]
        logger.info(f"WS client disconnected. Total: {len(self._connections)}")

    async def broadcast_raw(self, data: dict) -> None:
        """Broadcast raw dict as JSON to all connected clients."""
        if not self._connections:
            return
        text = json.dumps(data, default=str)
        dead = []
        for ws in self._connections:
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    @property
    def connection_count(self) -> int:
        return len(self._connections)


# Global singleton — imported by main.py and witness_processor.py
manager = ConnectionManager()


@router.websocket("/live")
async def websocket_live(ws: WebSocket):
    """WebSocket endpoint consumed by the React dashboard."""
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()  # keep-alive; server pushes via manager
    except WebSocketDisconnect:
        manager.disconnect(ws)
