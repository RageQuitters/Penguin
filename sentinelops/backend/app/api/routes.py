"""
API Routes
POST /api/analyze  — run multi-agent analysis pipeline
GET  /api/machines — current machine states for dashboard
"""
from fastapi import APIRouter, HTTPException
from app.models.schemas import AnalyzeRequest, AnalyzeResponse, MachineState
from app.core.trace import TraceLog
from app.orchestrator.orchestrator import run_orchestrator
from app.tools.data_store import get_all_machine_states

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Run the full Pangu LLM multi-agent pipeline for a single sensor reading.
    Falls back to mock responses when PANGU_API_BASE is not configured.
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
    """
    Return all machine states for dashboard. Reads from GaussDB in production.
    """
    return [MachineState(**s) for s in get_all_machine_states()]
