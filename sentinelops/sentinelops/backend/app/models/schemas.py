"""
Pydantic schemas — single source of truth for all data shapes.
"""
from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TraceAction(str, Enum):
    TOOL_CALL = "tool_call"
    REASONING = "reasoning"
    DECISION = "decision"
    HANDOFF = "handoff"


class AnomalyClassification(str, Enum):
    NORMAL = "normal"
    TRANSIENT_SPIKE = "transient_spike"
    DEVELOPING_FAULT = "developing_fault"


class Urgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MachineStatus(str, Enum):
    NORMAL = "Normal"
    WARNING = "Warning"
    CRITICAL = "Critical"


# ---------------------------------------------------------------------------
# Sensor reading
# ---------------------------------------------------------------------------

class SensorReading(BaseModel):
    air_temperature: float = Field(..., description="Air temperature in Kelvin", example=298.1)
    process_temperature: float = Field(..., description="Process temperature in Kelvin", example=308.6)
    rotational_speed: float = Field(..., description="Rotational speed in RPM", example=1551.0)
    torque: float = Field(..., description="Torque in Nm", example=42.8)
    tool_wear: float = Field(..., description="Tool wear in minutes", example=187.0)

    def to_tuple(self) -> tuple:
        return (
            self.air_temperature,
            self.process_temperature,
            self.rotational_speed,
            self.torque,
            self.tool_wear,
        )


# ---------------------------------------------------------------------------
# API input
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    machine_id: str = Field(..., example="U-07")
    reading: SensorReading


# ---------------------------------------------------------------------------
# Agent outputs
# ---------------------------------------------------------------------------

class AnomalyResult(BaseModel):
    anomaly_score: float
    classification: AnomalyClassification
    urgency: Urgency
    reasoning: str
    baseline_trend: list[float] = []


class FaultResult(BaseModel):
    active_faults: list[str]
    severity: Urgency
    enriched_analysis: dict[str, Any]  # per-fault explanation
    reasoning: str
    procurement_flag: bool = False


class PredictiveResult(BaseModel):
    rul_hours: float
    degradation_rate: float
    parts_available: dict[str, bool]
    urgency: Urgency
    procurement_flag: bool
    reasoning: str


class OrchestratorDecision(BaseModel):
    machine_id: str
    final_status: MachineStatus
    overall_urgency: Urgency
    work_order: str
    agents_called: list[str]
    anomaly: AnomalyResult | None = None
    fault: FaultResult | None = None
    predictive: PredictiveResult | None = None


# ---------------------------------------------------------------------------
# Trace entry
# ---------------------------------------------------------------------------

class TraceEntry(BaseModel):
    id: str
    session_id: str
    timestamp: str
    agent: str
    action: TraceAction
    input_data: Any = None
    output_data: Any = None
    explanation: str = ""


# ---------------------------------------------------------------------------
# API response
# ---------------------------------------------------------------------------

class AnalyzeResponse(BaseModel):
    session_id: str
    machine_id: str
    decision: OrchestratorDecision
    trace: list[dict]


# ---------------------------------------------------------------------------
# Machine status (for dashboard polling)
# ---------------------------------------------------------------------------

class MachineState(BaseModel):
    machine_id: str
    machine_type: str
    status: MachineStatus
    anomaly_score: float
    rul_hours: float | None = None
    active_faults: list[str] = []
    tool_wear: float = 0.0
