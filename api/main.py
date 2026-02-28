"""WitnessAI — FastAPI Application"""
from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import incidents, agents, health, dashboard
from api.websocket import router as ws_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("WitnessAI API starting")
    yield
    logger.info("WitnessAI API stopping")


def create_app(witness_agent=None) -> FastAPI:
    app = FastAPI(
        title="WitnessAI",
        description="Real-Time Crime Scene Intelligence Agent API",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Wire live agent into route modules
    if witness_agent is not None:
        incidents.set_agent(witness_agent)
        agents.set_agent(witness_agent)

    app.include_router(health.router,    prefix="/api/v1", tags=["Health"])
    app.include_router(incidents.router, prefix="/api/v1", tags=["Incidents"])
    app.include_router(agents.router,    prefix="/api/v1", tags=["Agents"])
    app.include_router(ws_router,        prefix="/ws",     tags=["WebSocket"])
    app.include_router(dashboard.router, prefix="",        tags=["Dashboard"])

    return app


app = create_app()
