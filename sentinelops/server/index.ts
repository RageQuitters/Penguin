import express from "express";
import { createServer } from "http";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
dotenv.config();

import {
  startTelegramBot,
  notifyEngineers,
  updateAgentContext,
} from "./telegramAgent.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ── LLM config ────────────────────────────────────────────────────────────────
const LLM_BASE_URL =
  process.env.LLM_BASE_URL ||
  "https://api-ap-southeast-1.modelarts-maas.com/openai/v1";
const LLM_URL = `${LLM_BASE_URL}/chat/completions`;
const LLM_API_KEY = process.env.LLM_API_KEY;
const LLM_MODEL = process.env.LLM_MODEL || "DeepSeek-V3";

if (!LLM_API_KEY) console.warn("[WARN] LLM_API_KEY not set.");

// ── ML sidecar config ─────────────────────────────────────────────────────────
const ML_URL = process.env.ML_URL || "http://localhost:5001";

// ── In-memory rolling state ───────────────────────────────────────────────────
// Server keeps a shadow of the machine list for Telegram context and ticker
let _machines: any[] = [];
let _anomalyHistory: Record<string, any[]> = {};
let _faultHistory: Record<string, any[]> = {};

// ── ML helpers ─────────────────────────────────────────────────────────────────

async function callMLPredict(machine: any) {
  const res = await fetch(`${ML_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      air_temperature: machine.air_temperature,
      process_temperature: machine.process_temperature,
      rotational_speed: machine.rotational_speed,
      torque: machine.torque,
      tool_wear: machine.tool_wear,
    }),
  });
  if (!res.ok) throw new Error(`ML sidecar ${res.status}: ${await res.text()}`);
  return res.json() as Promise<{
    anomaly_score: number;
    failure_vector: Record<string, number>;
    active_faults: string[];
    decision: "NORMAL" | "WARNING" | "FAILURE";
  }>;
}

// ── LLM helpers ────────────────────────────────────────────────────────────────

async function callLLM(body: Record<string, unknown>) {
  const res = await fetch(LLM_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${LLM_API_KEY}` },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`LLM ${res.status}: ${await res.text()}`);
  return res.json();
}

async function callLLMJson(systemPrompt: string, userPrompt: string) {
  const data = await callLLM({
    model: LLM_MODEL,
    temperature: 0.1,
    max_tokens: 1024,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: userPrompt },
    ],
  });
  const text = data?.choices?.[0]?.message?.content ?? "{}";
  try { return JSON.parse(text); } catch { return {}; }
}

// ── Classification ─────────────────────────────────────────────────────────────

function getClassification(anomalyScore: number, activeFaults: string[]): "severe" | "moderate" | "normal" {
  if (activeFaults.length > 0) return "severe";
  if (anomalyScore > 0.6) return "moderate";
  return "normal";
}

// ── Sub-agents ─────────────────────────────────────────────────────────────────

async function anomalyAgent(machine: any) {
  const ml = await callMLPredict(machine);
  const score = ml.anomaly_score;
  const classification = getClassification(score, ml.active_faults);
  return { anomaly_score: score, classification, reasoning: `LOF model: score=${score.toFixed(3)}, decision=${ml.decision}` };
}

async function faultAgent(machine: any) {
  const ml = await callMLPredict(machine);
  const active_faults = ml.active_faults;
  const severity = active_faults.length >= 3 ? "high" : active_faults.length >= 1 ? "medium" : "low";
  const procurement_flag = active_faults.includes("TWF") || active_faults.includes("HDF");
  return {
    active_faults, severity, procurement_flag,
    reasoning: `RF classifiers predicted faults: [${active_faults.join(", ") || "none"}]. Raw vector: TWF=${ml.failure_vector.TWF} HDF=${ml.failure_vector.HDF} PWF=${ml.failure_vector.PWF} OSF=${ml.failure_vector.OSF} RNF=${ml.failure_vector.RNF}`,
  };
}

async function predictiveAgent(machine: any) {
  const system = `You are the Predictive Maintenance Agent. Estimate remaining useful life and degradation. Return ONLY JSON: { "rul_hours": number, "degradation_rate": number between 0 and 1, "urgency": "low"|"medium"|"high"|"critical", "procurement_flag": boolean, "reasoning": "one sentence" }`;
  const user = `Machine data:\n${JSON.stringify({ machine_id: machine.machine_id, tool_wear: machine.tool_wear, existing_rul: machine.rul_hours, anomaly_score: machine.anomaly_score })}`;
  const out = await callLLMJson(system, user);
  return {
    rul_hours: typeof out.rul_hours === "number" ? out.rul_hours : machine.rul_hours ?? 0,
    degradation_rate: out.degradation_rate ?? 0,
    urgency: out.urgency ?? "low",
    procurement_flag: !!out.procurement_flag,
    reasoning: out.reasoning ?? "No reasoning provided.",
  };
}

// ── Orchestrator ───────────────────────────────────────────────────────────────

interface OrchestratorResult {
  machine_id: string;
  agents_called: string[];
  anomaly?: any;
  fault?: any;
  predictive?: any;
  routing_reason: string;
  work_order: string;
  overall_urgency: "low" | "medium" | "high" | "critical";
}

async function orchestrate(machine: any): Promise<OrchestratorResult> {
  const agents_called: string[] = [];
  const routing_log: string[] = [];

  const anomaly = await anomalyAgent(machine);
  agents_called.push("anomaly");
  routing_log.push(`Anomaly Agent (LOF) returned score=${anomaly.anomaly_score.toFixed(3)}, classification=${anomaly.classification}.`);

  let fault: any, predictive: any;

  if (anomaly.classification === "normal") {
    routing_log.push("Classification normal → skip Fault & Predictive agents.");
  } else {
    fault = await faultAgent(machine);
    agents_called.push("fault");
    routing_log.push(`Classification ${anomaly.classification} → Fault Agent returned [${fault.active_faults.join(", ") || "none"}], severity=${fault.severity}.`);

    if (fault.active_faults.length > 0 || anomaly.classification === "moderate") {
      predictive = await predictiveAgent(machine);
      agents_called.push("predictive");
      routing_log.push(`Predictive Agent returned RUL=${predictive.rul_hours.toFixed(1)}h, urgency=${predictive.urgency}.`);
    }
  }

  const synthesis = await synthesizeWorkOrder(machine, { anomaly, fault, predictive, routing_reason: routing_log.join(" ") });

  // After orchestration for a severe/critical machine, notify engineers via Telegram
  if (synthesis.overall_urgency === "critical" || synthesis.overall_urgency === "high") {
    const faultStr = fault?.active_faults?.join(", ") || "anomaly detected";
    notifyEngineers(
      `🚨 *SentinelOps Alert*\n\n*Machine:* ${machine.machine_id}\n*Urgency:* ${synthesis.overall_urgency.toUpperCase()}\n*Issues:* ${faultStr}\n*RUL:* ${predictive?.rul_hours?.toFixed(1) ?? machine.rul_hours?.toFixed(1)}h\n\n📋 *Work Order:*\n${synthesis.work_order}`
    ).catch(() => {});
  }

  return { machine_id: machine.machine_id, agents_called, anomaly, fault, predictive, routing_reason: routing_log.join(" "), work_order: synthesis.work_order, overall_urgency: synthesis.overall_urgency };
}

async function synthesizeWorkOrder(machine: any, signals: any) {
  const system = `You are the SentinelOps Orchestrator. Synthesize sub-agent outputs into a final work order.\nRULES: 1. overall_urgency = highest severity across all agents. 2. If ANY agent indicates critical failure risk → urgency = "critical". 3. If classification = "normal" → urgency = "low", work_order = monitor only. 4. If RUL < 24h OR severe faults → urgency = "critical". 5. Do not invent numbers.\nWork order: 2-5 sentences, actionable for technicians. Include machine ID, fault types, ETA to failure, required parts, action. If monitoring only, specify review interval.\nReturn ONLY JSON: { "overall_urgency": "low"|"medium"|"high"|"critical", "work_order": "..." }`;
  const user = `Machine: ${machine.machine_id}\n\nROUTING LOG:\n${signals.routing_reason}\n\nSUB-AGENT OUTPUTS:\n${JSON.stringify({ anomaly: signals.anomaly, fault: signals.fault ?? "not called", predictive: signals.predictive ?? "not called" }, null, 2)}`;
  const out = await callLLMJson(system, user);
  return { overall_urgency: out.overall_urgency ?? "medium", work_order: out.work_order ?? "Unable to generate decision. Manual inspection required." };
}

// ── Rolling data ticker ────────────────────────────────────────────────────────
// Every 10 minutes: for each machine, fetch ML scores and write a new anomaly log
// This is done via the /api/rolling-tick endpoint (called by a cron/setInterval on startup)

async function runRollingTick(machines: any[]) {
  console.log(`[RollingTick] Running for ${machines.length} machines at ${new Date().toISOString()}`);
  const results = await Promise.allSettled(
    machines.map(async (m) => {
      try {
        const ml = await callMLPredict(m);
        return { machine_id: m.machine_id, anomaly_score: ml.anomaly_score, failure_vector: ml.failure_vector, decision: ml.decision, ...m };
      } catch {
        return { machine_id: m.machine_id, anomaly_score: m.anomaly_score, failure_vector: { TWF: m.TWF, HDF: m.HDF, PWF: m.PWF, OSF: m.OSF, RNF: m.RNF }, decision: m.anomaly_score > 0.7 ? 'FAILURE' : m.anomaly_score > 0.4 ? 'WARNING' : 'NORMAL', ...m };
      }
    })
  );

  const readings = results
    .filter((r) => r.status === 'fulfilled')
    .map((r: any) => r.value);

  // Update in-memory context for Telegram agent
  for (const r of readings) {
    if (!_anomalyHistory[r.machine_id]) _anomalyHistory[r.machine_id] = [];
    _anomalyHistory[r.machine_id].push({ timestamp: new Date().toISOString(), anomaly_score: r.anomaly_score, decision: r.decision });
    if (_anomalyHistory[r.machine_id].length > 24) _anomalyHistory[r.machine_id].shift();
  }
  updateAgentContext(_machines, _anomalyHistory, _faultHistory);

  return readings;
}

// ── HTTP Server ────────────────────────────────────────────────────────────────

const CHAT_SYSTEM_PROMPT = `
You are SentinelOps AI for a machine-fleet monitoring dashboard.

You help plant managers:
- understand current machine status using the correct three-tier classification:
    • Severe   — at least 1 predicted fault (HDF, OSF, PWF, RNF, or TWF = 1). Requires immediate engineer dispatch.
    • Moderate — no predicted faults, but anomaly_score > 0.6. Monitor closely and schedule maintenance.
    • Normal   — no predicted faults and anomaly_score ≤ 0.6. Operating within acceptable parameters.
- explain anomaly scores, RUL, tool wear, and active faults (HDF, OSF, PWF, RNF, TWF)
- recommend maintenance and dispatch actions

Rules:
- ALWAYS use the Severe / Moderate / Normal classification above.
- Base answers ONLY on the live machine data provided.
- Be concise. Use markdown with **bold** for machine_ids.
`.trim();

async function startServer() {
  const app = express();
  const server = createServer(app);

  app.use((req, res, next) => {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
    if (req.method === "OPTIONS") return res.sendStatus(200);
    next();
  });
  app.use(express.json({ limit: "2mb" }));

  // ── /api/chat ───────────────────────────────────────────────────────────────
  app.post("/api/chat", async (req, res) => {
    try {
      const { userMessage, machines, history = [] } = req.body ?? {};
      if (typeof userMessage !== "string" || !Array.isArray(machines))
        return res.status(400).json({ error: "Invalid body" });

      const safeHistory = history
        .filter((h: any) => h && (h.role === "user" || h.role === "assistant") && typeof h.content === "string")
        .map(({ role, content }: any) => ({ role, content }))
        .slice(-20);

      const data = await callLLM({
        model: LLM_MODEL, temperature: 0.3, max_tokens: 2048,
        messages: [
          { role: "system", content: `${CHAT_SYSTEM_PROMPT}\n\nLive machine data (JSON):\n${JSON.stringify(machines)}` },
          ...safeHistory,
          { role: "user", content: userMessage },
        ],
      });
      const msg = data?.choices?.[0]?.message ?? {};
      res.json({ reply: msg.content ?? "", reasoning: msg.reasoning_content ?? null, model: LLM_MODEL });
    } catch (err: any) {
      console.error("[/api/chat]", err?.message);
      res.status(500).json({ error: "AI call failed", detail: err?.message });
    }
  });

  // ── /api/orchestrate ────────────────────────────────────────────────────────
  app.post("/api/orchestrate", async (req, res) => {
    try {
      const { machine } = req.body ?? {};
      if (!machine?.machine_id) return res.status(400).json({ error: "Missing machine" });
      console.log(`[orchestrate] starting for ${machine.machine_id}`);
      const result = await orchestrate(machine);
      console.log(`[orchestrate] ${machine.machine_id} → ${result.overall_urgency}`);
      res.json(result);
    } catch (err: any) {
      console.error("[/api/orchestrate]", err?.message);
      res.status(500).json({ error: "Orchestrate failed", detail: err?.message });
    }
  });

  // ── /api/orchestrate/fleet ──────────────────────────────────────────────────
  app.post("/api/orchestrate/fleet", async (req, res) => {
    try {
      const { machines } = req.body ?? {};
      if (!Array.isArray(machines)) return res.status(400).json({ error: "machines must be an array" });
      const results = await Promise.all(
        machines.map((m) =>
          orchestrate(m).catch((err) => ({
            machine_id: m.machine_id, error: err?.message, agents_called: [],
            routing_reason: "orchestration failed",
            work_order: "Error during orchestration — manual inspection required.",
            overall_urgency: "medium" as const,
          }))
        )
      );
      // After fleet orchestration, build summary and notify engineers
      const severe = results.filter((r: any) => r.overall_urgency === 'critical' || r.overall_urgency === 'high');
      if (severe.length > 0) {
        const summary = severe.map((r: any) => `• *${r.machine_id}* (${r.overall_urgency}): ${(r.work_order ?? '').slice(0, 100)}`).join('\n');
        await notifyEngineers(`📊 *SentinelOps Fleet Report*\n\n${severe.length} machine(s) require attention:\n\n${summary}`).catch(() => {});
      }
      res.json({ results });
    } catch (err: any) {
      console.error("[/api/orchestrate/fleet]", err?.message);
      res.status(500).json({ error: "Fleet orchestrate failed", detail: err?.message });
    }
  });

  // ── /api/predict-all ────────────────────────────────────────────────────────
  app.post("/api/predict-all", async (req, res) => {
    try {
      const { machines } = req.body ?? {};
      if (!Array.isArray(machines)) return res.status(400).json({ error: "machines must be an array" });
      _machines = machines; // update in-memory shadow

      const results = await Promise.all(
        machines.map(async (m: any) => {
          try {
            const mlRes = await fetch(`${ML_URL}/predict`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ air_temperature: m.air_temperature, process_temperature: m.process_temperature, rotational_speed: m.rotational_speed, torque: m.torque, tool_wear: m.tool_wear }),
            });
            if (!mlRes.ok) throw new Error(`ML sidecar ${mlRes.status}`);
            const ml = await mlRes.json();
            return { machine_id: m.machine_id, ...ml };
          } catch {
            return {
              machine_id: m.machine_id, anomaly_score: m.anomaly_score,
              failure_vector: { TWF: m.TWF, HDF: m.HDF, PWF: m.PWF, OSF: m.OSF, RNF: m.RNF },
              active_faults: [],
              decision: (() => { const f = [m.TWF, m.HDF, m.PWF, m.OSF, m.RNF]; if (f.some((v) => v === 1)) return 'FAILURE'; if (m.anomaly_score > 0.6) return 'WARNING'; return 'NORMAL'; })(),
            };
          }
        })
      );
      updateAgentContext(_machines, _anomalyHistory, _faultHistory);
      res.json({ results });
    } catch (err: any) {
      console.error("[/api/predict-all]", err?.message);
      res.status(500).json({ error: "predict-all failed", detail: err?.message });
    }
  });

  // ── /api/rolling-tick ───────────────────────────────────────────────────────
  // Called internally every 10 minutes to advance the rolling fake DB
  app.post("/api/rolling-tick", async (req, res) => {
    try {
      const { machines } = req.body ?? {};
      if (!Array.isArray(machines)) return res.status(400).json({ error: "machines required" });
      const readings = await runRollingTick(machines);
      res.json({ readings, ticked_at: new Date().toISOString() });
    } catch (err: any) {
      res.status(500).json({ error: err?.message });
    }
  });

  // ── /api/telegram/notify ────────────────────────────────────────────────────
  app.post("/api/telegram/notify", async (req, res) => {
    try {
      const { message } = req.body ?? {};
      if (!message) return res.status(400).json({ error: "message required" });
      await notifyEngineers(message);
      res.json({ sent: true });
    } catch (err: any) {
      res.status(500).json({ error: err?.message });
    }
  });

  // ── /api/logs/anomaly ───────────────────────────────────────────────────────
  app.get("/api/logs/anomaly/:machineId", (req, res) => {
    const { machineId } = req.params;
    res.json({ logs: _anomalyHistory[machineId] ?? [] });
  });

  app.get("/api/health", (_req, res) => res.json({ ok: true }));

  const port = Number(process.env.PORT) || 3001;
  server.listen(port, "0.0.0.0", () => {
    console.log(`SentinelOps orchestrator on :${port} (${LLM_MODEL})`);
    console.log(`ML sidecar expected at: ${ML_URL}`);
  });

  // Start Telegram bot (non-blocking)
  startTelegramBot();

  // Rolling tick every 10 minutes (kicks off if machines are available)
  setInterval(async () => {
    if (_machines.length > 0) {
      await runRollingTick(_machines).catch((e) => console.warn('[RollingTick] error:', e?.message));
    }
  }, 10 * 60 * 1000);
}

startServer().catch((e) => { console.error(e); process.exit(1); });
