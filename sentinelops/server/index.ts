import express from "express";
import { createServer } from "http";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";
dotenv.config();

import {
  startTelegramBot,
  notifyEngineers,
  notifyEngineer,
  notifyEngineerWithButtons,
  registerAssignment,
  getAssignmentEvents,
  updateAgentContext,
  updateEngineerRoster,
} from "./telegramAgent.js";

import {
  rankEngineers,
  pickBestEngineer,
  FAULT_NAMES,
  type FaultCode,
  type RoutableEngineer,
} from "../shared/faultRouting.js";

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

────────────────────────────────────────────────────────────────────────
SIMULATION & WORKFLOW AWARENESS
────────────────────────────────────────────────────────────────────────
You support a simulated maintenance workflow for testing the system.

Simulation flow:
  1. A machine develops a fault and is assigned to an engineer.
  2. Engineer receives the assignment via Telegram.
  3. Engineer may respond with:
       - "fixed"    → issue resolved
       - "wip"      → still working
       - "escalate" → cannot fix

When the engineer reports "fixed":
- Assume all fault flags for that machine are cleared.
- Machine status becomes Normal.
- Respond with a brief confirmation.
- Reflect that the system is now healthy.

When the engineer reports "escalate":
- Ask the engineer WHY escalation is needed.
- Request brief details (e.g. missing tools, unclear fault, severity).
- State that the issue will be passed to a senior engineer.

When describing an escalation, ALWAYS include:
- Machine ID
- Fault type
- Anomaly score and RUL
- Engineer's reason (if available)

────────────────────────────────────────────────────────────────────────
FAULT LOGGING FORMAT
────────────────────────────────────────────────────────────────────────
Whenever you mention a fault, include all of:
- machine ID
- fault type (PWF, TWF, HDF, OSF, RNF)
- anomaly score
- RUL
- a brief description of the issue

Example:
🚨 **U-03 — Power Failure (PWF)**
Anomaly: 0.41 | RUL: 150h
Issue: unstable power supply detected

────────────────────────────────────────────────────────────────────────
SIMULATION MODE
────────────────────────────────────────────────────────────────────────
The system may be running a simulation:
- Machines can transition from faulty → fixed.
- New faults may appear after resolution.
- Always reflect the LATEST state from the live machine data — do not
  assume faults persist after being marked fixed.
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

  // ── /api/chat-agentic ───────────────────────────────────────────────────────
  // Like /api/chat but the LLM has tools that wrap the existing sub-agents
  // (anomaly, fault, predictive) plus engineer notification helpers. The full
  // tool-call sequence is returned alongside the reply so the UI can render
  // a transparent agent trace.
  app.post("/api/chat-agentic", async (req, res) => {
    try {
      const { userMessage, machines, history = [] } = req.body ?? {};
      if (typeof userMessage !== "string" || !Array.isArray(machines))
        return res.status(400).json({ error: "Invalid body" });

      const machinesById = new Map<string, any>(
        (machines as any[]).map((m) => [m.machine_id, m]),
      );
      const findMachine = (id: string) => {
        const m = machinesById.get(id);
        if (!m) throw new Error(`Unknown machine_id: ${id}. Valid IDs: ${Array.from(machinesById.keys()).join(', ')}`);
        return m;
      };

      const trace: Array<{
        agent: string;
        input: Record<string, unknown>;
        output: unknown;
        ts: string;
        ms: number;
      }> = [];

      // Tool implementations — each one wraps an existing function and
      // appends an entry to `trace` so the client can show what happened.
      const tools = {
        run_anomaly_agent: async (args: { machine_id: string }) => {
          const m = findMachine(args.machine_id);
          return await anomalyAgent(m);
        },
        run_fault_agent: async (args: { machine_id: string }) => {
          const m = findMachine(args.machine_id);
          return await faultAgent(m);
        },
        run_predictive_agent: async (args: { machine_id: string }) => {
          const m = findMachine(args.machine_id);
          return await predictiveAgent(m);
        },
        summarize_fleet: async () => {
          const severe = (machines as any[]).filter((m) =>
            [m.HDF, m.OSF, m.PWF, m.RNF, m.TWF].some((v: number) => v === 1),
          );
          const moderate = (machines as any[]).filter(
            (m) => !severe.includes(m) && m.anomaly_score > 0.6,
          );
          return {
            total: machines.length,
            severe: severe.map((m) => ({ machine_id: m.machine_id, anomaly_score: m.anomaly_score, rul_hours: m.rul_hours, faults: ['HDF','OSF','PWF','RNF','TWF'].filter((f) => m[f] === 1) })),
            moderate: moderate.map((m) => ({ machine_id: m.machine_id, anomaly_score: m.anomaly_score, rul_hours: m.rul_hours })),
            normal_count: machines.length - severe.length - moderate.length,
          };
        },
        notify_engineers_about: async (args: { machine_id: string; custom_message?: string }) => {
          const m = findMachine(args.machine_id);
          const faults = ['HDF','OSF','PWF','RNF','TWF'].filter((f) => m[f] === 1);
          const message = args.custom_message ?? [
            `📌 *Status Update — ${m.machine_id}*`,
            ``,
            `*Anomaly score:* ${(m.anomaly_score ?? 0).toFixed(3)}`,
            `*RUL:* ${(m.rul_hours ?? 0).toFixed(1)}h`,
            `*Active faults:* ${faults.length > 0 ? faults.join(', ') : 'none'}`,
            `*Tool wear:* ${m.tool_wear} min`,
          ].join('\n');
          await notifyEngineers(message);
          return { ok: true, broadcast_to: 'all engineers', message };
        },
        get_machine_details: async (args: { machine_id: string }) => {
          const m = findMachine(args.machine_id);
          return {
            machine_id: m.machine_id,
            machine_type: m.machine_type,
            anomaly_score: m.anomaly_score,
            rul_hours: m.rul_hours,
            tool_wear: m.tool_wear,
            air_temperature: m.air_temperature,
            process_temperature: m.process_temperature,
            rotational_speed: m.rotational_speed,
            torque: m.torque,
            faults: ['HDF','OSF','PWF','RNF','TWF'].filter((f) => m[f] === 1),
          };
        },
        get_recent_engineer_events: async (args: { limit?: number }) => {
          // Returns recent Telegram-driven assignment events: fixes, escalations,
          // status changes. Use this when the user asks "what just happened",
          // "any escalations?", or to describe a fix the orchestrator just learned about.
          const limit = Math.min(Math.max(args.limit ?? 10, 1), 50);
          const all = getAssignmentEvents(0);
          const recent = all.slice(-limit).reverse(); // newest first
          return recent.map((ev) => {
            const m = machinesById.get(ev.machine_id);
            return {
              kind: ev.kind,
              machine_id: ev.machine_id,
              fault_types: ev.fault_types,
              engineer_name: ev.engineer_name,
              ts: ev.ts,
              reason: ev.note ?? null,
              anomaly_score: m?.anomaly_score ?? null,
              rul_hours: m?.rul_hours ?? null,
            };
          });
        },
      } as const;

      // Tool schema for the LLM
      const toolSchema = [
        {
          type: 'function',
          function: {
            name: 'run_anomaly_agent',
            description: 'Run the LOF-based anomaly detection sub-agent for a single machine. Returns the anomaly score and severity classification.',
            parameters: { type: 'object', properties: { machine_id: { type: 'string' } }, required: ['machine_id'] },
          },
        },
        {
          type: 'function',
          function: {
            name: 'run_fault_agent',
            description: 'Run the random-forest fault classifier sub-agent. Returns the active faults (HDF/OSF/PWF/RNF/TWF) and severity.',
            parameters: { type: 'object', properties: { machine_id: { type: 'string' } }, required: ['machine_id'] },
          },
        },
        {
          type: 'function',
          function: {
            name: 'run_predictive_agent',
            description: 'Run the predictive maintenance sub-agent. Returns RUL estimate, degradation rate, and urgency level.',
            parameters: { type: 'object', properties: { machine_id: { type: 'string' } }, required: ['machine_id'] },
          },
        },
        {
          type: 'function',
          function: {
            name: 'summarize_fleet',
            description: 'Get a quick summary of the entire fleet — counts and lists of severe/moderate/normal machines.',
            parameters: { type: 'object', properties: {} },
          },
        },
        {
          type: 'function',
          function: {
            name: 'notify_engineers_about',
            description: 'Broadcast a Telegram message to ALL engineers about a specific machine. Use when the user asks to "text engineers about <machine>" or similar.',
            parameters: {
              type: 'object',
              properties: {
                machine_id: { type: 'string' },
                custom_message: { type: 'string', description: 'Optional custom message body (Telegram Markdown). If omitted, a default status block is generated.' },
              },
              required: ['machine_id'],
            },
          },
        },
        {
          type: 'function',
          function: {
            name: 'get_machine_details',
            description: 'Read full sensor + fault state of a machine. Cheap — use it before running heavier agents.',
            parameters: { type: 'object', properties: { machine_id: { type: 'string' } }, required: ['machine_id'] },
          },
        },
        {
          type: 'function',
          function: {
            name: 'get_recent_engineer_events',
            description: 'List recent Telegram-driven engineer events: fixes, in-progress updates, and escalations (with reasons). Each event includes the machine_id, fault_types, engineer_name, timestamp, escalation reason if any, and the machine\'s current anomaly_score and RUL. Use this whenever the user asks about escalations, recent fixes, or "what just happened".',
            parameters: { type: 'object', properties: { limit: { type: 'number', description: 'How many recent events to return (default 10, max 50).' } } },
          },
        },
      ];

      const safeHistory = history
        .filter((h: any) => h && (h.role === "user" || h.role === "assistant") && typeof h.content === "string")
        .map(({ role, content }: any) => ({ role, content }))
        .slice(-10);

      const systemPrompt = `${CHAT_SYSTEM_PROMPT}

You are an orchestrator agent with access to specialist sub-agents as tools.
- Use tools to gather fresh data instead of guessing from the JSON snapshot.
- For status questions about a single machine, prefer get_machine_details first.
- For severity / urgency / RUL, run the matching sub-agent.
- For "text/notify engineers about <machine>", call notify_engineers_about with that exact machine_id and a tailored custom_message — never blast generic fleet stats.
- Tool calls are visible to the user as a trace, so be deliberate.

Live machine snapshot (use only for IDs and quick context — call tools for fresh values):
${JSON.stringify((machines as any[]).map((m) => ({ machine_id: m.machine_id, status: m.status })))}`;

      const messages: any[] = [
        { role: 'system', content: systemPrompt },
        ...safeHistory,
        { role: 'user', content: userMessage },
      ];

      // Tool-calling loop, capped to 6 iterations
      const MAX_HOPS = 6;
      let finalReply = '';
      let finalReasoning: string | null = null;

      for (let hop = 0; hop < MAX_HOPS; hop++) {
        const data = await callLLM({
          model: LLM_MODEL,
          temperature: 0.2,
          max_tokens: 1500,
          messages,
          tools: toolSchema,
          tool_choice: 'auto',
        });

        const msg = data?.choices?.[0]?.message;
        if (!msg) break;
        finalReasoning = msg.reasoning_content ?? finalReasoning;

        const toolCalls = msg.tool_calls ?? [];
        if (toolCalls.length === 0) {
          finalReply = msg.content ?? '';
          break;
        }

        // Push the assistant's tool-call message into history
        messages.push({ role: 'assistant', content: msg.content ?? '', tool_calls: toolCalls });

        // Execute each tool call and append the result
        for (const call of toolCalls) {
          const name: string = call.function?.name;
          let parsedArgs: Record<string, unknown> = {};
          try { parsedArgs = JSON.parse(call.function?.arguments ?? '{}'); }
          catch { parsedArgs = {}; }

          let result: unknown;
          const start = Date.now();
          try {
            const fn = (tools as any)[name];
            if (typeof fn !== 'function') throw new Error(`Unknown tool ${name}`);
            result = await fn(parsedArgs);
          } catch (err: any) {
            result = { error: err?.message ?? 'tool execution failed' };
          }
          const ms = Date.now() - start;

          trace.push({ agent: name, input: parsedArgs, output: result, ts: new Date().toISOString(), ms });

          messages.push({
            role: 'tool',
            tool_call_id: call.id,
            content: JSON.stringify(result),
          });
        }
      }

      res.json({
        reply: finalReply,
        reasoning: finalReasoning,
        agent_calls: trace,
        model: LLM_MODEL,
      });
    } catch (err: any) {
      console.error("[/api/chat-agentic]", err?.message);
      res.status(500).json({ error: "Agentic chat failed", detail: err?.message });
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

  // ── /api/engineers/sync ─────────────────────────────────────────────────────
  // Mirror the engineer roster from Firestore into the Telegram agent's cache.
  // Called by the dashboard on load so escalations can route to seniors even
  // before any assignment has happened in this session.
  app.post("/api/engineers/sync", (req, res) => {
    try {
      const { engineers } = req.body ?? {};
      if (!Array.isArray(engineers)) return res.status(400).json({ error: "engineers array required" });
      updateEngineerRoster(engineers);
      res.json({ cached: engineers.length });
    } catch (err: any) {
      res.status(500).json({ error: err?.message });
    }
  });

  // ── /api/assign-engineer ────────────────────────────────────────────────────
  // Body: {
  //   machine: Machine,                  (full machine object so we can build a rich message)
  //   engineers: RoutableEngineer[],     (caller's view of the active engineer roster)
  //   fault_types?: FaultCode[],         (defaults to whichever faults are active on the machine)
  //   engineer_id?: string,              (skip auto-routing and target this engineer)
  //   custom_message?: string,           (optional override of the default DM body)
  //   assignment_id?: string,            (Firestore doc ID — registers the assignment so button taps map back)
  // }
  // Picks the best-matching engineer (or the explicit one), DMs them with action buttons, returns the picked engineer + ranking.
  app.post("/api/assign-engineer", async (req, res) => {
    try {
      const { machine, engineers, fault_types, engineer_id, custom_message, assignment_id } = req.body ?? {};
      if (!machine?.machine_id) return res.status(400).json({ error: "machine required" });
      if (!Array.isArray(engineers)) return res.status(400).json({ error: "engineers array required" });

      // If caller didn't specify fault_types, derive from the machine's flags.
      const ALL_FAULTS: FaultCode[] = ['HDF', 'OSF', 'PWF', 'RNF', 'TWF'];
      const inferredFaults: FaultCode[] = ALL_FAULTS.filter((f) => machine[f] === 1);
      const faultsToRoute: FaultCode[] = (Array.isArray(fault_types) && fault_types.length > 0)
        ? fault_types
        : inferredFaults;

      // Cache the roster on the Telegram side so escalations can be routed
      // to senior engineers and AI replies can identify who's chatting.
      updateEngineerRoster(engineers);

      const ranked = rankEngineers(engineers as RoutableEngineer[], faultsToRoute);

      // Choose target: explicit ID > best-ranked > nothing
      let chosen = engineer_id
        ? ranked.find((r) => r.engineer.id === engineer_id) ?? null
        : pickBestEngineer(engineers as RoutableEngineer[], faultsToRoute);

      if (!chosen) {
        return res.status(409).json({ error: "No active engineer available", ranking: ranked });
      }

      // Build the DM if no custom message
      const faultLabel = faultsToRoute.length > 0
        ? faultsToRoute.map((f) => `${f} (${FAULT_NAMES[f] ?? f})`).join(', ')
        : 'general inspection';

      const message = custom_message ?? [
        `🛠 *Assignment — ${machine.machine_id}*`,
        ``,
        `*Faults:* ${faultLabel}`,
        `*Anomaly Score:* ${(machine.anomaly_score ?? 0).toFixed(3)}`,
        `*RUL:* ${(machine.rul_hours ?? 0).toFixed(1)}h`,
        `*Why you:* ${chosen.reason}`,
        ``,
        `Tap a button below — or reply with *fixed*, *wip*, or *escalate*.`,
      ].join('\n');

      // Use buttons if we have an assignment_id to track callbacks; fall back to plain DM otherwise
      let sent = false;
      let telegramError: string | undefined;
      if (chosen.engineer.telegram_chat_id) {
        if (assignment_id) {
          // Register first so the callback handler can find this assignment when the engineer taps
          registerAssignment({
            assignment_id,
            machine_id: machine.machine_id,
            engineer_chat_id: chosen.engineer.telegram_chat_id,
            engineer_name: chosen.engineer.name,
            fault_types: faultsToRoute,
            created_at: new Date().toISOString(),
          });
          const result = await notifyEngineerWithButtons(
            chosen.engineer.telegram_chat_id,
            message,
            assignment_id,
          );
          sent = result.sent;
          telegramError = result.error;
        } else {
          const result = await notifyEngineer(chosen.engineer.telegram_chat_id, message);
          sent = result.sent;
          telegramError = result.error;
        }
      } else {
        telegramError = 'Engineer has no telegram_chat_id configured';
      }

      res.json({
        sent,
        telegram_error: telegramError,
        chosen: {
          engineer: chosen.engineer,
          score: chosen.score,
          matchedKeywords: chosen.matchedKeywords,
          reason: chosen.reason,
        },
        ranking: ranked.slice(0, 5).map((r) => ({
          engineer_id: r.engineer.id,
          name: r.engineer.name,
          specialization: r.engineer.specialization,
          score: r.score,
          reason: r.reason,
          has_telegram: !!r.engineer.telegram_chat_id,
        })),
        message,
        fault_types: faultsToRoute,
      });
    } catch (err: any) {
      console.error("[/api/assign-engineer]", err?.message);
      res.status(500).json({ error: err?.message });
    }
  });

  // ── /api/assignments/events ─────────────────────────────────────────────────
  // Client polls this to pick up engineer button taps from Telegram and
  // mirror them into Firestore (resolveFaultLog, updateAssignment, addEngineerLog).
  // Query: ?since=<lastEventId>  (string; defaults to 0)
  app.get("/api/assignments/events", (req, res) => {
    const sinceParam = req.query.since;
    const since = typeof sinceParam === "string" ? Number(sinceParam) || 0 : 0;
    const events = getAssignmentEvents(since);
    res.json({ events });
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

  // Start Telegram bot (non-blocking).
  // Set DISABLE_TELEGRAM_POLLING=true in .env to prevent this instance from
  // polling. Useful when another instance is already polling the same token,
  // which causes duplicate replies. The webapp works fine without the bot.
  if (process.env.DISABLE_TELEGRAM_POLLING === 'true') {
    console.log('[Telegram] Polling disabled via DISABLE_TELEGRAM_POLLING=true — webapp-only mode');
  } else {
    startTelegramBot();
  }

  // Rolling tick every 10 minutes (kicks off if machines are available)
  setInterval(async () => {
    if (_machines.length > 0) {
      await runRollingTick(_machines).catch((e) => console.warn('[RollingTick] error:', e?.message));
    }
  }, 10 * 60 * 1000);
}

startServer().catch((e) => { console.error(e); process.exit(1); });