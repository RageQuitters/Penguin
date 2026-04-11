"""
SentinelOps — FastAPI Application Entry Point
"""
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.api.routes import router
from app.api.websocket import ws_router, trace_broadcaster

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan: start background WebSocket broadcaster on startup,
    cancel it cleanly on shutdown.
    """
    task = asyncio.create_task(trace_broadcaster())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Smart Multi-Agent AI System for Industrial Plant Monitoring",
    lifespan=lifespan,
)

# CORS — allow the React frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(router, prefix="/api")
app.include_router(ws_router)


@app.get("/health")
async def health():
    return {"status": "ok", "app": settings.app_name, "version": settings.app_version}
