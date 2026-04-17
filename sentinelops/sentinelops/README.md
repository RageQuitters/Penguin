# SentinelOps

**Smart Multi-Agent AI System for Industrial Plant Monitoring**
Team Penguin · Huawei Cloud Challenge 2026

---

## Architecture

```
L1  Real-time ingestion   IoT Device Access → DMS for Kafka → Cloud Stream Storage
L2  Processing & storage  FunctionGraph → GaussDB → OBS
L3  Multi-agent AI        Pangu LLM (Orchestrator) + ModelArts (ML inference)
L4  Operator interfaces   FastAPI + React Dashboard + Telegram / SMSMSG alerts
```

### Agent Flow

```
Sensor reading → Orchestrator (Pangu LLM)
                     │
                     ▼
             Anomaly Detection Agent
               ├─ run_isolation_forest()   [ModelArts / local joblib]
               ├─ query_baseline()          [Cloud Stream Storage / GaussDB]
               └─ get_machine_profile()    [GaussDB]
                     │
               score ≥ 0.4?  ──NO──▶ Monitor only. Stop.
                     │YES
                     ▼
             Fault Classification Agent
               ├─ run_fault_classifier()   [ModelArts / local joblib]
               ├─ get_fault_history()      [GaussDB]
               └─ get_maintenance_log()    [GaussDB]
                     │
           fault OR score ≥ 0.7?  ──NO──▶ Skip predictive.
                     │YES
                     ▼
             Predictive Maintenance Agent
               ├─ get_wear_trend()         [GaussDB]
               ├─ estimate_rul()
               └─ check_parts_inventory()
                     │
                     ▼
             Orchestrator synthesises → Work Order (Pangu LLM)
```

---

## Folder Structure

```
sentinelops/
├── backend/
│   ├── app/
│   │   ├── agents/
│   │   │   ├── anomaly_agent.py       # Isolation Forest + Pangu LLM reasoning
│   │   │   ├── fault_agent.py         # Random Forest + contextual Pangu reasoning
│   │   │   └── predictive_agent.py    # RUL + parts + Pangu urgency decision
│   │   ├── api/
│   │   │   ├── routes.py              # POST /api/analyze, GET /api/machines
│   │   │   └── websocket.py           # /ws real-time trace streaming
│   │   ├── core/
│   │   │   ├── config.py              # All Huawei Cloud env vars (pydantic-settings)
│   │   │   ├── pangu_client.py        # Pangu LLM client (IAM + API key auth)
│   │   │   └── trace.py              # Observability: TraceLog → async queue → WS
│   │   ├── models/
│   │   │   └── schemas.py             # All Pydantic schemas
│   │   ├── orchestrator/
│   │   │   └── orchestrator.py        # Conditional agent routing + synthesis
│   │   ├── tools/
│   │   │   ├── data_store.py          # Mock GaussDB / Cloud Stream Storage queries
│   │   │   └── ml_inference.py        # Joblib model wrapper (→ ModelArts in prod)
│   │   └── main.py                    # FastAPI app entry point
│   ├── dataset/
│   │   └── train_test.csv
│   ├── joblib_files/
│   │   ├── urgency_model.joblib       # Trained Isolation Forest
│   │   └── fault_model.joblib         # Trained Random Forest (multi-label)
│   ├── telegram/
│   │   └── bot.py                     # Telegram alert bot
│   ├── requirements.txt
│   └── .env.example
│
└── frontend/
    ├── public/
    │   └── index.html
    ├── src/
    │   ├── components/
    │   │   ├── MachineCard.jsx         # Color-coded machine status card
    │   │   ├── AgentTracePanel.jsx     # Live agent reasoning timeline
    │   │   ├── DashboardWidgets.jsx    # SensorGrid, FaultGrid, AnomalyChart, KpiGrid
    │   │   ├── WorkOrderPanel.jsx      # Orchestrator decision + agent summaries
    │   │   └── AnalyzeModal.jsx        # Custom sensor reading input
    │   ├── hooks/
    │   │   └── useWebSocket.js         # Auto-reconnect WS hook
    │   ├── pages/
    │   │   └── Dashboard.jsx           # Main 3-column layout
    │   ├── services/
    │   │   └── api.js                  # fetchMachines(), analyzeReading()
    │   ├── App.jsx
    │   ├── index.js
    │   └── index.css                   # Dark terminal theme (IBM Plex Mono)
    └── package.json
```

---

## Setup

### 1. Backend

```bash
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — at minimum set PANGU_API_BASE + PANGU_API_KEY
# Leave PANGU_API_BASE empty to run in mock mode (no cloud credentials needed)

# Run
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend endpoints:
- `GET  /health`           — health check
- `GET  /api/machines`     — all machine states
- `POST /api/analyze`      — run multi-agent analysis
- `WS   /ws`               — real-time agent trace stream

### 2. Frontend

```bash
cd frontend
npm install
npm start        # opens http://localhost:3000
```

The React dev server proxies `/api/*` and `/ws` to `localhost:8000`.

---

## Huawei Cloud Configuration

### Pangu LLM (L3 — Orchestrator)

1. Deploy Pangu model via **ModelArts** → Dedicated Resource Pool
2. Note the inference endpoint URL
3. Set in `.env`:
   ```
   PANGU_API_BASE=https://infer-modelarts-ap-southeast-3.myhuaweicloud.com/modelarts/v1/infers/<id>
   PANGU_API_KEY=<your_key>
   PANGU_AUTH_MODE=apikey   # or "iam"
   ```

### ModelArts ML Inference (L3 — Anomaly + Fault models)

1. Upload `joblib_files/*.joblib` to **OBS**
2. Create inference services in **ModelArts**
3. Set `MODELARTS_URGENCY_ENDPOINT` and `MODELARTS_FAULT_ENDPOINT` in `.env`
4. Update `app/tools/ml_inference.py` — the functions have clear production swap comments

### GaussDB (L2 — operational database)

Replace mock functions in `app/tools/data_store.py` with real SQL queries:
```python
import asyncpg
# Each function has a clear "Production swap:" comment
```

### DMS for Kafka (L1 — real-time ingestion)

The system is designed for Kafka consumption. Add a Kafka consumer in `app/main.py`:
```python
# On new message in processed-readings topic → call run_orchestrator()
```

---

## Mock Mode

When `PANGU_API_BASE` is not set, `pangu_client.py` returns realistic mock
LLM responses based on the prompt content. The full agent pipeline still runs
— Isolation Forest and Random Forest models score the readings, tools query
the mock data store, and the trace system streams over WebSocket. Only the LLM
reasoning step is simulated. This allows full local development without
Huawei Cloud credentials.

---

## Sensor Data Format

```json
{
  "machine_id": "U-07",
  "reading": {
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1543.0,
    "torque": 42.8,
    "tool_wear": 187.0
  }
}
```

### Example curl

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "machine_id": "U-07",
    "reading": {
      "air_temperature": 298.1,
      "process_temperature": 308.6,
      "rotational_speed": 1543.0,
      "torque": 42.8,
      "tool_wear": 187.0
    }
  }'
```
