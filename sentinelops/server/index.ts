import express from "express";
import { createServer } from "http";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// ---- LLM config (server-side only) --------------------------------------
const LLM_BASE_URL =
  process.env.LLM_BASE_URL ||
  "https://api-ap-southeast-1.modelarts-maas.com/openai/v1";
const LLM_URL = `${LLM_BASE_URL}/chat/completions`;
const LLM_API_KEY = process.env.LLM_API_KEY;
const LLM_MODEL = process.env.LLM_MODEL || "DeepSeek-V3";

if (!LLM_API_KEY) {
  console.warn("[WARN] LLM_API_KEY is not set. /api/* will return 500.");
}

// ---- ML sidecar config --------------------------------------------------
// Python FastAPI server that loads the joblib models and runs predict()
const ML_URL = process.env.ML_URL || "http://localhost:5001";

// =========================================================================
// ML SIDECAR CALLER
// Calls your Python ml_server.py which runs the exact predict() pipeline
// from your Jupyter notebook (LOF anomaly score + RF fault classifiers).
// =========================================================================

async function callMLPredict(machine: any) {
  const res = await fetch(`${ML_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      air_temperature:     machine.air_temperature,
      process_temperature: machine.process_temperature,
      rotational_speed:    machine.rotational_speed,
      torque:              machine.torque,
      tool_wear:           machine.tool_wear,
    }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`ML sidecar ${res.status}: ${text}`);
  }
  return res.json() as Promise<{
    anomaly_score:  number;
    failure_vector: Record<string, number>;  // { TWF: 0, HDF: 1, ... }
    active_faults:  string[];                // ["HDF"]
    decision:       "NORMAL" | "WARNING" | "FAILURE";
  }>;
}

// =========================================================================
// LLM helpers (still used by predictiveAgent + synthesizeWorkOrder)
// =========================================================================

async function callLLM(body: Record<string, unknown>) {
  const res = await fetch(LLM_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${LLM_API_KEY}`,
    },
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
  try {
    return JSON.parse(text);
  } catch {
    return {};
  }
}

// =========================================================================
// SUB-AGENTS
// =========================================================================

// ----- Agent 1: Anomaly  ← NOW ML-POWERED (LOF model) -------------------
async function anomalyAgent(machine: any) {
  const ml = await callMLPredict(machine);

  const score = ml.anomaly_score;
  const classification =
    score > 0.7 ? "severe" :
    score > 0.4 ? "moderate" :
                  "normal";

  return {
    anomaly_score:  score,
    classification,
    // decision mirrors your notebook's final_decision() thresholds
    reasoning: `LOF model: score=${score.toFixed(3)}, decision=${ml.decision}`,
  };
}

// ----- Agent 2: Fault Classifier  ← NOW ML-POWERED (RF models) ----------
async function faultAgent(machine: any) {
  const ml = await callMLPredict(machine);

  const active_faults = ml.active_faults;  // e.g. ["HDF", "TWF"]
  const severity =
    active_faults.length >= 3 ? "high"   :
    active_faults.length >= 1 ? "medium" :
                                "low";

  // Flag procurement if tool-wear-related faults are active
  const procurement_flag =
    active_faults.includes("TWF") || active_faults.includes("HDF");

  return {
    active_faults,
    severity,
    procurement_flag,
    reasoning: `RF classifiers predicted faults: [${active_faults.join(", ") || "none"}]. ` +
               `Raw vector: TWF=${ml.failure_vector.TWF} HDF=${ml.failure_vector.HDF} ` +
               `PWF=${ml.failure_vector.PWF} OSF=${ml.failure_vector.OSF} RNF=${ml.failure_vector.RNF}`,
  };
}

// ----- Agent 3: Predictive Maintenance  ← DeepSeek (no RUL model yet) ---
async function predictiveAgent(machine: any) {
  const system = `You are the Predictive Maintenance Agent.
Estimate remaining useful life and degradation.
Return ONLY JSON:
{
  "rul_hours": number (hours remaining before failure),
  "degradation_rate": number between 0 and 1,
  "urgency": "low" | "medium" | "high" | "critical",
  "procurement_flag": boolean,
  "reasoning": "one sentence"
}`;

  const user = `Machine data:
${JSON.stringify({
  machine_id:    machine.machine_id,
  tool_wear:     machine.tool_wear,
  existing_rul:  machine.rul_hours,
  anomaly_score: machine.anomaly_score,
})}`;

  const out = await callLLMJson(system, user);
  return {
    rul_hours:        typeof out.rul_hours === "number" ? out.rul_hours : machine.rul_hours ?? 0,
    degradation_rate: out.degradation_rate ?? 0,
    urgency:          out.urgency ?? "low",
    procurement_flag: !!out.procurement_flag,
    reasoning:        out.reasoning ?? "No reasoning provided.",
  };
}

// =========================================================================
// ORCHESTRATOR — routing logic unchanged, agents now ML-backed
// =========================================================================

interface OrchestratorResult {
  machine_id: string;
  agents_called: string[];
  anomaly?: Awaited<ReturnType<typeof anomalyAgent>>;
  fault?: Awaited<ReturnType<typeof faultAgent>>;
  predictive?: Awaited<ReturnType<typeof predictiveAgent>>;
  routing_reason: string;
  work_order: string;
  overall_urgency: "low" | "medium" | "high" | "critical";
}

async function orchestrate(machine: any): Promise<OrchestratorResult> {
  const agents_called: string[] = [];
  const routing_log: string[] = [];

  // Step 1: ALWAYS call Anomaly Agent (ML)
  const anomaly = await anomalyAgent(machine);
  agents_called.push("anomaly");
  routing_log.push(`Anomaly Agent (LOF) returned score=${anomaly.anomaly_score.toFixed(3)}, classification=${anomaly.classification}.`);

  let fault: Awaited<ReturnType<typeof faultAgent>> | undefined;
  let predictive: Awaited<ReturnType<typeof predictiveAgent>> | undefined;

  // Step 2: score < 0.4 → monitor only, stop
  if (anomaly.anomaly_score < 0.4) {
    routing_log.push("Score below 0.4 → skip Fault & Predictive agents (monitor only).");
  } else {
    // Step 3: score >= 0.4 → call Fault Classifier (ML)
    fault = await faultAgent(machine);
    agents_called.push("fault");
    routing_log.push(
      `Score >= 0.4 → Fault Agent (RF) returned [${fault.active_faults.join(", ") || "none"}], severity=${fault.severity}.`
    );

    // Step 4: any fault OR score >= 0.7 → call Predictive Agent (DeepSeek)
    const hasFault    = fault.active_faults.length > 0;
    const highAnomaly = anomaly.anomaly_score >= 0.7;

    if (hasFault || highAnomaly) {
      predictive = await predictiveAgent(machine);
      agents_called.push("predictive");
      routing_log.push(
        `${hasFault ? "Fault detected" : "Score >= 0.7"} → Predictive Agent returned RUL=${predictive.rul_hours.toFixed(1)}h, urgency=${predictive.urgency}.`
      );
    } else {
      routing_log.push("No faults and score < 0.7 → skip Predictive Agent.");
    }
  }

  // Step 5: Synthesize work order (DeepSeek)
  const synthesis = await synthesizeWorkOrder(machine, {
    anomaly, fault, predictive,
    routing_reason: routing_log.join(" "),
  });

  return {
    machine_id: machine.machine_id,
    agents_called,
    anomaly, fault, predictive,
    routing_reason: routing_log.join(" "),
    work_order:      synthesis.work_order,
    overall_urgency: synthesis.overall_urgency,
  };
}

async function synthesizeWorkOrder(
  machine: any,
  signals: { anomaly: any; fault?: any; predictive?: any; routing_reason: string }
) {
  const system = `You are the SentinelOps Orchestrator.
Synthesize sub-agent outputs into a final work order.

RULES:
1. overall_urgency = highest severity across all agents (low < medium < high < critical).
2. If ANY agent indicates critical failure risk → urgency = "critical".
3. If anomaly_score < 0.4 AND no active faults → urgency = "low", work_order = monitor only.
4. If RUL < 24h OR severe faults → urgency = "critical".
5. Do not invent any new numbers.

Work order rules:
- 2 to 5 sentences, actionable for technicians.
- Include: machine ID, fault types (if any), ETA to failure (if available),
  required parts (if applicable), action (inspect / replace / shutdown / monitor).
- If monitoring only, specify review interval (e.g., "recheck in 12 hours").
- Keep format compact — the AI Assistant panel is small.

Return ONLY JSON:
{ "overall_urgency": "low"|"medium"|"high"|"critical", "work_order": "..." }`;

  const user = `Machine: ${machine.machine_id}

ROUTING DECISION LOG:
${signals.routing_reason}

SUB-AGENT OUTPUTS:
${JSON.stringify({ anomaly: signals.anomaly, fault: signals.fault ?? "not called", predictive: signals.predictive ?? "not called" }, null, 2)}`;

  const out = await callLLMJson(system, user);
  return {
    overall_urgency: out.overall_urgency ?? "medium",
    work_order:      out.work_order ?? "Unable to generate decision. Manual inspection required.",
  };
}

// =========================================================================
// HTTP SERVER
// =========================================================================

const CHAT_SYSTEM_PROMPT = `
You are SentinelOps AI for a machine-fleet monitoring dashboard.

You help plant managers:
- understand current machine status (Normal / Warning / Critical)
- explain anomaly scores, RUL, tool wear, and active faults (HDF, OSF, PWF, RNF, TWF)
- recommend maintenance and dispatch actions

Rules:
- Base answers ONLY on the live machine data provided. Do not invent machines or readings.
- Be concise. Use markdown with **bold** for machine_ids.
- If the user asks to "orchestrate" or wants a work order for a specific machine,
  tell them to use the Orchestrate button — the orchestrator tool is more accurate
  than this conversational endpoint.
`.trim();

async function startServer() {
  const app = express();
  const server = createServer(app);
  app.use(express.json({ limit: "2mb" }));

  // --- /api/chat — conversational Q&A ------------------------------------
  app.post("/api/chat", async (req, res) => {
    try {
      const { userMessage, machines, history = [] } = req.body ?? {};
      if (typeof userMessage !== "string" || !Array.isArray(machines)) {
        return res.status(400).json({ error: "Invalid body" });
      }

      const safeHistory = history
        .filter((h: any) => h && (h.role === "user" || h.role === "assistant") && typeof h.content === "string")
        .map(({ role, content }: any) => ({ role, content }))
        .slice(-20);

      const data = await callLLM({
        model: LLM_MODEL,
        temperature: 0.3,
        max_tokens: 2048,
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

  // --- /api/orchestrate — single machine ---------------------------------
  app.post("/api/orchestrate", async (req, res) => {
    try {
      const { machine } = req.body ?? {};
      if (!machine || !machine.machine_id) {
        return res.status(400).json({ error: "Missing machine" });
      }
      console.log(`[orchestrate] starting for ${machine.machine_id}`);
      const result = await orchestrate(machine);
      console.log(`[orchestrate] ${machine.machine_id} → ${result.overall_urgency} (agents: ${result.agents_called.join(", ")})`);
      res.json(result);
    } catch (err: any) {
      console.error("[/api/orchestrate]", err?.message);
      res.status(500).json({ error: "Orchestrate failed", detail: err?.message });
    }
  });

  // --- /api/orchestrate/fleet — all machines in parallel -----------------
  app.post("/api/orchestrate/fleet", async (req, res) => {
    try {
      const { machines } = req.body ?? {};
      if (!Array.isArray(machines)) {
        return res.status(400).json({ error: "machines must be an array" });
      }
      const results = await Promise.all(
        machines.map((m) =>
          orchestrate(m).catch((err) => ({
            machine_id:      m.machine_id,
            error:           err?.message,
            agents_called:   [],
            routing_reason:  "orchestration failed",
            work_order:      "Error during orchestration — manual inspection required.",
            overall_urgency: "medium" as const,
          }))
        )
      );
      res.json({ results });
    } catch (err: any) {
      console.error("[/api/orchestrate/fleet]", err?.message);
      res.status(500).json({ error: "Fleet orchestrate failed", detail: err?.message });
    }
  });

  // --- /api/predict-all — run ML predict() for every machine at once ------
  // Called by the frontend on page load to enrich all machines with real
  // LOF anomaly scores + RF fault flags before the chat sees them.
  app.post("/api/predict-all", async (req, res) => {
    try {
      const { machines } = req.body ?? {};
      if (!Array.isArray(machines)) {
        return res.status(400).json({ error: "machines must be an array" });
      }

      const results = await Promise.all(
        machines.map(async (m: any) => {
          try {
            const mlRes = await fetch(`${ML_URL}/predict`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                air_temperature:     m.air_temperature,
                process_temperature: m.process_temperature,
                rotational_speed:    m.rotational_speed,
                torque:              m.torque,
                tool_wear:           m.tool_wear,
              }),
            });
            if (!mlRes.ok) throw new Error(`ML sidecar ${mlRes.status}`);
            const ml = await mlRes.json();
            return { machine_id: m.machine_id, ...ml };
          } catch (err: any) {
            // If ML sidecar is down, return the stored values unchanged
            return {
              machine_id:     m.machine_id,
              anomaly_score:  m.anomaly_score,
              failure_vector: { TWF: m.TWF, HDF: m.HDF, PWF: m.PWF, OSF: m.OSF, RNF: m.RNF },
              active_faults:  [],
              decision:       m.status === 'Critical' ? 'FAILURE' : m.status === 'Warning' ? 'WARNING' : 'NORMAL',
            };
          }
        })
      );

      res.json({ results });
    } catch (err: any) {
      console.error("[/api/predict-all]", err?.message);
      res.status(500).json({ error: "predict-all failed", detail: err?.message });
    }
  });
  
  app.get("/api/health", (_req, res) => res.json({ ok: true }));

  const staticPath = path.resolve(__dirname, "..", "dist");
  app.use(express.static(staticPath));
  app.use((_req, res) => res.sendFile(path.join(staticPath, "index.html")));

  const port = Number(process.env.PORT) || 3001;
  server.listen(port, "0.0.0.0", () => {
    console.log(`SentinelOps orchestrator on :${port} (${LLM_MODEL})`);
    console.log(`ML sidecar expected at: ${ML_URL}`);
  });
}

startServer().catch((e) => { console.error(e); process.exit(1); });