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

// =========================================================================
// SUB-AGENTS
// These mirror the three agents in your architecture. Each one is a
// standalone LLM call with its own system prompt and returns strict JSON.
// Swap the implementations with your real ML models when ready — the
// orchestrator just needs the JSON shapes.
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

// ----- Agent 1: Anomaly ---------------------------------------------------
async function anomalyAgent(machine: any) {
  // NOTE: your README shows anomaly_score is already on the machine record.
  // For a real demo you'd pass sensor readings to your ML model. Here we
  // use the LLM to "classify" based on the numbers — good enough for the
  // orchestrator demo; swap with your real isolation forest / autoencoder
  // when ready.
  const system = `You are the Anomaly Detection Agent for industrial machines.
Analyze sensor readings and return ONLY JSON:
{
  "anomaly_score": number between 0 and 1,
  "classification": "normal" | "moderate" | "severe",
  "reasoning": "one sentence"
}`;

  const user = `Machine sensor readings:
${JSON.stringify({
  machine_id: machine.machine_id,
  air_temperature: machine.air_temperature,
  process_temperature: machine.process_temperature,
  rotational_speed: machine.rotational_speed,
  torque: machine.torque,
  tool_wear: machine.tool_wear,
  existing_anomaly_score: machine.anomaly_score,
})}`;

  const out = await callLLMJson(system, user);
  return {
    anomaly_score: typeof out.anomaly_score === "number"
      ? out.anomaly_score
      : machine.anomaly_score ?? 0,
    classification: out.classification ?? "normal",
    reasoning: out.reasoning ?? "No reasoning provided.",
  };
}

// ----- Agent 2: Fault Classifier -----------------------------------------
async function faultAgent(machine: any) {
  const system = `You are the Fault Classification Agent.
Given sensor readings and existing fault flags, identify which faults are active.
Return ONLY JSON:
{
  "active_faults": array of strings from ["HDF", "OSF", "PWF", "RNF", "TWF"],
  "severity": "low" | "medium" | "high",
  "procurement_flag": boolean (true if spare parts likely needed),
  "reasoning": "one sentence"
}`;

  const user = `Machine data:
${JSON.stringify({
  machine_id: machine.machine_id,
  HDF: machine.HDF, OSF: machine.OSF, PWF: machine.PWF,
  RNF: machine.RNF, TWF: machine.TWF,
  tool_wear: machine.tool_wear,
  torque: machine.torque,
  rotational_speed: machine.rotational_speed,
})}`;

  const out = await callLLMJson(system, user);
  return {
    active_faults: Array.isArray(out.active_faults) ? out.active_faults : [],
    severity: out.severity ?? "low",
    procurement_flag: !!out.procurement_flag,
    reasoning: out.reasoning ?? "No reasoning provided.",
  };
}

// ----- Agent 3: Predictive Maintenance -----------------------------------
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
  machine_id: machine.machine_id,
  tool_wear: machine.tool_wear,
  existing_rul: machine.rul_hours,
  anomaly_score: machine.anomaly_score,
})}`;

  const out = await callLLMJson(system, user);
  return {
    rul_hours: typeof out.rul_hours === "number"
      ? out.rul_hours
      : machine.rul_hours ?? 0,
    degradation_rate: out.degradation_rate ?? 0,
    urgency: out.urgency ?? "low",
    procurement_flag: !!out.procurement_flag,
    reasoning: out.reasoning ?? "No reasoning provided.",
  };
}

// =========================================================================
// ORCHESTRATOR
// Conditional routing — decides at runtime which sub-agents to invoke.
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
  let routing_log: string[] = [];

  // --- Step 1: ALWAYS call the Anomaly Agent ---
  const anomaly = await anomalyAgent(machine);
  agents_called.push("anomaly");
  routing_log.push(
    `Anomaly Agent returned score=${anomaly.anomaly_score.toFixed(2)}.`
  );

  let fault: Awaited<ReturnType<typeof faultAgent>> | undefined;
  let predictive: Awaited<ReturnType<typeof predictiveAgent>> | undefined;

  // --- Step 2: If score < 0.4, monitor only and STOP ---
  if (anomaly.anomaly_score < 0.4) {
    routing_log.push(
      "Score below 0.4 → skip Fault & Predictive agents (monitor only)."
    );
  } else {
    // --- Step 3: Score >= 0.4 → call Fault Classifier ---
    fault = await faultAgent(machine);
    agents_called.push("fault");
    routing_log.push(
      `Score >= 0.4 → Fault Agent returned [${
        fault.active_faults.join(", ") || "none"
      }], severity=${fault.severity}.`
    );

    // --- Step 4: If any fault OR score >= 0.7 → call Predictive Agent ---
    const hasFault = fault.active_faults.length > 0;
    const highAnomaly = anomaly.anomaly_score >= 0.7;

    if (hasFault || highAnomaly) {
      predictive = await predictiveAgent(machine);
      agents_called.push("predictive");
      routing_log.push(
        `${hasFault ? "Fault detected" : "Score >= 0.7"} → Predictive Agent returned RUL=${
          predictive.rul_hours.toFixed(1)
        }h, urgency=${predictive.urgency}.`
      );
    } else {
      routing_log.push(
        "No faults and score < 0.7 → skip Predictive Agent."
      );
    }
  }

  // --- Step 5: Synthesize natural-language work order ---
  const synthesis = await synthesizeWorkOrder(machine, {
    anomaly, fault, predictive,
    routing_reason: routing_log.join(" "),
  });

  return {
    machine_id: machine.machine_id,
    agents_called,
    anomaly, fault, predictive,
    routing_reason: routing_log.join(" "),
    work_order: synthesis.work_order,
    overall_urgency: synthesis.overall_urgency,
  };
}

async function synthesizeWorkOrder(
  machine: any,
  signals: {
    anomaly: any;
    fault?: any;
    predictive?: any;
    routing_reason: string;
  }
) {
  const system = `You are the SentinelOps Orchestrator.
Synthesize sub-agent outputs into a final work order.

RULES:
1. overall_urgency = highest severity across all agents (low < medium < high < critical).
2. If ANY agent indicates critical failure risk → urgency = "critical".
3. If anomaly_score < 0.4 AND no active faults → urgency = "low", work_order = monitor only.
4. If RUL < 24h OR severe faults → urgency = "critical".
5. You should not create/invent any new numbers by yourself.

Work order rules:
- 2 to 5 sentences, actionable for technicians.
- Include: machine ID, fault types (if any), ETA to failure (if available),
  required parts (if applicable), action (inspect / replace / shutdown / monitor).
- If monitoring only, specify review interval (e.g., "recheck in 12 hours").
- Make the format readible for user, given that the space for the AI Assistance is small.
- Reduce the spaces in between them.

Return ONLY JSON:
{ "overall_urgency": "low"|"medium"|"high"|"critical", "work_order": "..." }`;

  const user = `Machine: ${machine.machine_id}

ROUTING DECISION LOG:
${signals.routing_reason}

SUB-AGENT OUTPUTS:
${JSON.stringify({
  anomaly: signals.anomaly,
  fault: signals.fault ?? "not called",
  predictive: signals.predictive ?? "not called",
}, null, 2)}`;

  const out = await callLLMJson(system, user);
  return {
    overall_urgency: out.overall_urgency ?? "medium",
    work_order: out.work_order ?? "Unable to generate structured decision. Manual inspection required.",
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
- Base answers ONLY on the live machine data provided. Do notinvent machines or readings.
- Be concise. Use markdown with **bold** for machine_ids.
- If the user asks to "orchestrate" or wants a work order for a specific machine,
  tell them to use the Orchestrate button — the orchestrator tool is more accurate
  than this conversational endpoint.
`.trim();

async function startServer() {
  const app = express();
  const server = createServer(app);
  app.use(express.json({ limit: "2mb" }));

  // --- /api/chat — conversational Q&A (unchanged) -----------------------
  app.post("/api/chat", async (req, res) => {
    try {
      const { userMessage, machines, history = [] } = req.body ?? {};
      if (typeof userMessage !== "string" || !Array.isArray(machines)) {
        return res.status(400).json({ error: "Invalid body" });
      }

      const safeHistory = history
        .filter((h: any) =>
          h && (h.role === "user" || h.role === "assistant") &&
          typeof h.content === "string"
        )
        .map(({ role, content }: any) => ({ role, content }))
        .slice(-20);

      const data = await callLLM({
        model: LLM_MODEL,
        temperature: 0.3,
        max_tokens: 2048,
        messages: [
          {
            role: "system",
            content: `${CHAT_SYSTEM_PROMPT}\n\nLive machine data (JSON):\n${JSON.stringify(machines)}`,
          },
          ...safeHistory,
          { role: "user", content: userMessage },
        ],
      });

      const msg = data?.choices?.[0]?.message ?? {};
      res.json({
        reply: msg.content ?? "",
        reasoning: msg.reasoning_content ?? null,
        model: LLM_MODEL,
      });
    } catch (err: any) {
      console.error("[/api/chat]", err?.message);
      res.status(500).json({ error: "AI call failed", detail: err?.message });
    }
  });

  // --- /api/orchestrate — conditional sub-agent routing -----------------
  app.post("/api/orchestrate", async (req, res) => {
    try {
      const { machine } = req.body ?? {};
      if (!machine || !machine.machine_id) {
        return res.status(400).json({ error: "Missing machine" });
      }

      console.log(`[orchestrate] starting for ${machine.machine_id}`);
      const result = await orchestrate(machine);
      console.log(
        `[orchestrate] ${machine.machine_id} → ${result.overall_urgency} (agents: ${result.agents_called.join(", ")})`
      );
      res.json(result);
    } catch (err: any) {
      console.error("[/api/orchestrate]", err?.message);
      res.status(500).json({ error: "Orchestrate failed", detail: err?.message });
    }
  });

  // --- /api/orchestrate/fleet — run orchestrator over every machine -----
  app.post("/api/orchestrate/fleet", async (req, res) => {
    try {
      const { machines } = req.body ?? {};
      if (!Array.isArray(machines)) {
        return res.status(400).json({ error: "machines must be an array" });
      }

      // Run in parallel for speed. If you have rate limits, serialize instead.
      const results = await Promise.all(
        machines.map((m) =>
          orchestrate(m).catch((err) => ({
            machine_id: m.machine_id,
            error: err?.message,
            agents_called: [],
            routing_reason: "orchestration failed",
            work_order: "Error during orchestration — manual inspection required.",
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

  app.get("/api/health", (_req, res) => res.json({ ok: true }));

  const staticPath = path.resolve(__dirname, "..", "dist");
  app.use(express.static(staticPath));
  app.use((_req, res) => res.sendFile(path.join(staticPath, "index.html")));

  const port = Number(process.env.PORT) || 3001;
  server.listen(port, "0.0.0.0", () => {
    console.log(`SentinelOps orchestrator on :${port} (${LLM_MODEL})`);
  });
}

startServer().catch((e) => { console.error(e); process.exit(1); });

// import express from "express";
// import { createServer } from "http";
// import path from "path";
// import { fileURLToPath } from "url";

// const __filename = fileURLToPath(import.meta.url);
// const __dirname = path.dirname(__filename);

// // ---- LLM config (server-side only) --------------------------------------
// // Huawei Cloud ModelArts MaaS — OpenAI-compatible API.
// // Base URL includes /openai/v1 prefix; we append /chat/completions.
// // Region: ap-southeast-1 (Singapore).
// const LLM_BASE_URL =
//   process.env.LLM_BASE_URL ||
//   "https://api-ap-southeast-1.modelarts-maas.com/openai/v1";
// const LLM_URL = `${LLM_BASE_URL}/chat/completions`;
// const LLM_API_KEY = process.env.LLM_API_KEY;
// const LLM_MODEL = process.env.LLM_MODEL || "DeepSeek-V3";

// if (!LLM_API_KEY) {
//   console.warn(
//     "[WARN] LLM_API_KEY is not set. /api/chat will return 500."
//   );
// }

// // ---- System prompt ------------------------------------------------------
// const SYSTEM_PROMPT = `
// You are SentinelOps AI, an assistant for a machine-fleet monitoring dashboard.

// You help plant managers understand machine status, faults, and maintenance needs.

// Rules:
// - Base every answer only on the live machine data provided.
// - Respond ONLY with valid JSON — no markdown, no explanation.
// - Never invent machines or readings.

// You help plant managers:
// - understand current machine status (Normal / Warning / Critical)
// - explain anomaly scores, RUL (remaining useful life), tool wear, and active
//   faults (HDF, OSF, PWF, RNF, TWF)
// - recommend maintenance and dispatch actions
// - act as the orchestrator for sub-agents (anomaly / fault / predictive) when
//   asked for final work orders

// Rules:
// - Base every answer only on the live machine data provided below. Do not
//   invent machines or readings.
// - Be concise. Use short markdown bullet lists when listing machines.
// - Use **machine_id** in bold when referencing a specific machine.
// - Never give medical, legal, or financial advice.
// `.trim();

// // ---- Types --------------------------------------------------------------
// interface ChatHistoryItem {
//   role: "user" | "assistant";
//   content: string;
// }

// // ---- LLM caller ---------------------------------------------------------
// async function callLLM(body: Record<string, unknown>) {
//   const res = await fetch(LLM_URL, {
//     method: "POST",
//     headers: {
//       "Content-Type": "application/json",
//       Authorization: `Bearer ${LLM_API_KEY}`,
//     },
//     body: JSON.stringify(body),
//   });
//   if (!res.ok) {
//     const text = await res.text();
//     throw new Error(`LLM ${res.status}: ${text}`);
//   }
//   return res.json();
// }

// // ---- Server -------------------------------------------------------------
// async function startServer() {
//   const app = express();
//   const server = createServer(app);

//   app.use(express.json({ limit: "2mb" }));

//   app.post("/api/chat", async (req, res) => {
//     try {
//       const {
//         userMessage,
//         machines,
//         history = [],
//       }: {
//         userMessage: string;
//         machines: unknown[];
//         history?: ChatHistoryItem[];
//       } = req.body ?? {};

//       if (typeof userMessage !== "string" || !Array.isArray(machines)) {
//         return res.status(400).json({ error: "Invalid body" });
//       }

//       const safeHistory = Array.isArray(history)
//         ? history
//             .filter(
//               (h) =>
//                 h &&
//                 (h.role === "user" || h.role === "assistant") &&
//                 typeof h.content === "string"
//             )
//             .map(({ role, content }) => ({ role, content }))
//             .slice(-20)
//         : [];

//       const messages = [
//         {
//           role: "system",
//           content: `${SYSTEM_PROMPT}\n\nLive machine data (JSON):\n${JSON.stringify(
//             machines
//           )}`,
//         },
//         ...safeHistory,
//         { role: "user", content: userMessage },
//       ];

//       const data = await callLLM({
//         model: LLM_MODEL,
//         messages,
//         max_tokens: 4096,
//         temperature: 0.3,
//       });

//       const msg = data?.choices?.[0]?.message ?? {};
//       res.json({
//         reply: msg.content ?? "",
//         // GLM-5 may or may not expose reasoning_content depending on the
//         // deployment — we pass it through if present so the "Show thinking"
//         // toggle in the UI works when supported. If null, the toggle is
//         // hidden automatically.
//         reasoning: msg.reasoning_content ?? null,
//         model: LLM_MODEL,
//       });
//     } catch (err: any) {
//       console.error("[/api/chat] error:", err?.message);
//       res.status(500).json({ error: "AI call failed", detail: err?.message });
//     }
//   });

//   app.get("/api/health", (_req, res) => res.json({ ok: true }));

//   // ---- Serve the built React app -------------------------------------
//   const staticPath = path.resolve(__dirname, "..", "dist");
//   app.use(express.static(staticPath));

//   // SPA fallback — works on Express 4 and 5.
//   app.use((_req, res) => {
//     res.sendFile(path.join(staticPath, "index.html"));
//   });

//   const port = Number(process.env.PORT) || 3001;
//   server.listen(port, "0.0.0.0", () => {
//     console.log(
//       `SentinelOps server listening on :${port} (${LLM_MODEL} @ ${LLM_BASE_URL})`
//     );
//   });
// }

// startServer().catch((e) => {
//   console.error(e);
//   process.exit(1);
// });