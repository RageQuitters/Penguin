# SentinelOps — Enhanced Edition

Industrial machine-fleet monitoring with ML anomaly detection, AI orchestration, Telegram notifications, and rolling data simulation.

## New Features (this edition)

### 📋 Machine Logs
- **Anomaly History** — Line graph of anomaly scores over time (10-min intervals), plus raw reading table
- **Fault Log** — Past fault events with resolution status and engineer notes
- **Engineer Fixes** — History of engineer visits, actions taken, and outcomes

### 🤖 Telegram AI Agent
- Autonomously **notifies engineers** when machines become critical or high urgency
- Notifies on: manual "Notify Engineers" button, "Analyze All Machines" (for critical dispatch), orchestrate calls
- Engineers can **text the bot** and get AI-powered answers about machine status, anomalies, and faults
- Bot commands: "status", "severe", any natural-language question

### 🔄 Rolling Fake Database
- Seeds 12 historical readings per machine (2h of 10-min intervals) if Firestore is empty
- A ticker adds a new anomaly log every 10 minutes with realistic sensor drift
- Seeds fake engineer visit logs and fault logs for demo purposes

---

## Setup

### 1. Environment variables

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Fill in:
```env
LLM_API_KEY=your_deepseek_key
TELEGRAM_BOT_TOKEN=your_bot_token    # from @BotFather on Telegram
TELEGRAM_ENGINEER_CHAT_IDS=12345,67890  # comma-separated chat IDs
```

**Getting a Telegram Bot:**
1. Message `@BotFather` on Telegram → `/newbot`
2. Copy the token into `TELEGRAM_BOT_TOKEN`
3. Message `@userinfobot` to find your chat ID → add to `TELEGRAM_ENGINEER_CHAT_IDS`

### 2. Install dependencies

```bash
npm install
```

### 3. Start the Python ML sidecar

```bash
cd ../  # repo root
pip install fastapi uvicorn joblib scikit-learn pandas numpy
python sentinelops/server/ml_server.py
```

### 4. Start the Node.js server + frontend

```bash
# Terminal 1: frontend
npm run dev

# Terminal 2: backend  
PORT=3001 npm run dev:server
```

---

## Architecture

```
Browser (React + Firebase)
  └─ Dashboard
       ├─ Machine Carousel (sidebar)
       ├─ Machine Detail Tabs: Sensors | Faults | Health | Logs
       │    └─ MachineLogs (anomaly chart, fault log, engineer log)
       └─ AI Chat

Node.js Express Server (:3001)
  ├─ /api/chat           — conversational LLM
  ├─ /api/orchestrate    — single machine (LOF + RF + DeepSeek)
  ├─ /api/orchestrate/fleet — all machines (auto-notifies Telegram)
  ├─ /api/predict-all    — ML batch scoring on page load
  ├─ /api/rolling-tick   — advances rolling DB (also runs on 10-min timer)
  └─ /api/telegram/notify — broadcast message to engineers
  
  Telegram Bot (long-polling)
  └─ Listens for engineer messages → AI reply
  └─ Sends alerts on orchestration

Python FastAPI (:5001)
  └─ /predict            — LOF anomaly score + RF fault classifiers (joblib)

Firebase Firestore
  ├─ machines/           — machine state
  ├─ anomaly_logs/       — 10-min rolling readings
  ├─ fault_logs/         — fault events
  └─ engineer_logs/      — engineer visit history
```

---

## Firestore Indexes Required

For the anomaly_logs and fault_logs queries you'll need composite indexes.
Firebase will print a direct link to create them on first query — just click the link.

Fields needed:
- `anomaly_logs`: `machine_id` ASC + `timestamp` DESC
- `fault_logs`: `machine_id` ASC + `timestamp` DESC  
- `engineer_logs`: `machine_id` ASC + `timestamp` DESC
