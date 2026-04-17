"""
WebSocket Route — /ws
---------------------
Streams agent trace entries in real-time to connected clients.
Each entry is broadcast as a JSON-serialised TraceEntry.

Multiple clients can connect simultaneously; all receive the same broadcast.
"""
import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.core.trace import _trace_queue

ws_router = APIRouter()

# Active WebSocket connections
_connections: set[WebSocket] = set()


async def broadcast(message: str) -> None:
    """Send a message to all connected WebSocket clients."""
    dead = set()
    for ws in _connections:
        try:
            await ws.send_text(message)
        except Exception:
            dead.add(ws)
    _connections.difference_update(dead)


async def trace_broadcaster() -> None:
    """
    Background task: continuously pulls from the trace queue and
    broadcasts to all connected WebSocket clients.
    Uses get() with no timeout so each entry is broadcast the instant it's queued —
    no artificial delay between entry generation and WS push.
    """
    while True:
        try:
            entry = await asyncio.wait_for(_trace_queue.get(), timeout=1.0)
            await broadcast(entry.model_dump_json())
            _trace_queue.task_done()
        except asyncio.TimeoutError:
            pass  # nothing queued — loop and wait again
        except asyncio.CancelledError:
            break


@ws_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket endpoint. Clients connect here to receive live trace entries.
    Also serves a ping/pong heartbeat so clients can detect disconnection.
    """
    await websocket.accept()
    _connections.add(websocket)
    try:
        # Send a welcome message
        await websocket.send_text(json.dumps({
            "type": "connected",
            "message": "SentinelOps WebSocket connected. Awaiting analysis runs.",
        }))
        # Keep alive — handle incoming pings from client
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_text(json.dumps({"type": "heartbeat"}))
    except WebSocketDisconnect:
        pass
    finally:
        _connections.discard(websocket)
