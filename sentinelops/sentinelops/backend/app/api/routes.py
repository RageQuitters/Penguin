"""
API Routes
POST /api/analyze    — run multi-agent analysis pipeline
GET  /api/machines   — current machine states for dashboard
POST /api/breakdown  — simulate a machine breakdown (sets Critical status)
POST /api/reset      — reset a machine back to Normal
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.schemas import AnalyzeRequest, AnalyzeResponse, MachineState
from app.core.trace import TraceLog
from app.orchestrator.orchestrator import run_orchestrator
from app.tools.data_store import get_all_machine_states, simulate_breakdown, reset_breakdown

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Run the full multi-agent analysis pipeline for a single sensor reading.
    Falls back to mock responses when the inference server is not configured.
    """
    trace = TraceLog()
    try:
        decision = await run_orchestrator(
            machine_id=request.machine_id,
            reading=request.reading,
            trace=trace,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent pipeline error: {str(e)}")
    return AnalyzeResponse(
        session_id=trace.session_id,
        machine_id=request.machine_id,
        decision=decision,
        trace=trace.to_list(),
    )


@router.get("/machines", response_model=list[MachineState])
async def get_machines() -> list[MachineState]:
    """Return all machine states for dashboard polling."""
    return [MachineState(**s) for s in get_all_machine_states()]


class BreakdownRequest(BaseModel):
    machine_id: str


@router.post("/breakdown")
async def breakdown(req: BreakdownRequest):
    """
    Simulate a machine breakdown.
    Sets the machine to Critical status with a TWF fault so the operator
    can then click ANALYZE to run the full agent pipeline.
    """
    if req.machine_id not in {s["machine_id"] for s in get_all_machine_states()}:
        raise HTTPException(status_code=404, detail=f"Machine {req.machine_id} not found")
    simulate_breakdown(req.machine_id)
    return {"status": "ok", "machine_id": req.machine_id, "simulated": "breakdown"}


@router.post("/reset")
async def reset(req: BreakdownRequest):
    """Reset a machine back to Normal status."""
    reset_breakdown(req.machine_id)
    return {"status": "ok", "machine_id": req.machine_id, "simulated": "reset"}
