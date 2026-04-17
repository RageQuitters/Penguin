"""
Trace / Observability System
----------------------------
Every agent action is logged as a TraceEntry and accumulated in a TraceLog.
The TraceLog is:
  - returned in full in POST /analyze responses
  - streamed entry-by-entry over the WebSocket /ws channel
"""
import asyncio
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable
from uuid import uuid4

from app.models.schemas import TraceEntry, TraceAction


# Global async queue — WebSocket consumers subscribe to this
_trace_queue: asyncio.Queue[TraceEntry] = asyncio.Queue()

# Registered WebSocket broadcast callback (set by the WS route)
_broadcast_cb: Callable[[TraceEntry], Awaitable[None]] | None = None


def register_broadcast(cb: Callable[[TraceEntry], Awaitable[None]]) -> None:
    global _broadcast_cb
    _broadcast_cb = cb


class TraceLog:
    """
    Collects trace entries for a single analysis run.
    Each agent writes to this log via add_entry().
    """

    def __init__(self, session_id: str | None = None):
        self.session_id: str = session_id or str(uuid4())
        self.entries: list[TraceEntry] = []

    def add_entry(
        self,
        agent: str,
        action: TraceAction,
        input_data: Any = None,
        output_data: Any = None,
        explanation: str = "",
    ) -> TraceEntry:
        entry = TraceEntry(
            id=str(uuid4()),
            session_id=self.session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            agent=agent,
            action=action,
            input_data=input_data,
            output_data=output_data,
            explanation=explanation,
        )
        self.entries.append(entry)
        # Push to global queue for WebSocket streaming
        _trace_queue.put_nowait(entry)
        return entry

    def to_list(self) -> list[dict]:
        return [e.model_dump() for e in self.entries]


async def get_next_trace_entry(timeout: float = 0.1) -> TraceEntry | None:
    """Pull next entry from queue (used by WebSocket broadcaster)."""
    try:
        return await asyncio.wait_for(_trace_queue.get(), timeout=timeout)
    except asyncio.TimeoutError:
        return None
