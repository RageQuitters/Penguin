/**
 * Frontend API client for SentinelOps.
 *
 *   POST /api/chat                — conversational Q&A
 *   POST /api/orchestrate         — agentic tool-calling orchestrator (one machine)
 *   POST /api/orchestrate/fleet   — orchestrator for every machine
 *
 * The LLM API key is never exposed to the browser — it lives on the server.
 */

import type { Machine } from './fakeData';

export interface ChatHistoryItem {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatResponse {
  reply: string;
  reasoning: string | null;
  model: string;
}

export interface AnomalyResult {
  anomaly_score: number;
  classification: 'normal' | 'moderate' | 'severe';
  reasoning: string;
}

export interface FaultResult {
  active_faults: string[];
  severity: 'low' | 'medium' | 'high';
  procurement_flag: boolean;
  reasoning: string;
}

export interface PredictiveResult {
  rul_hours: number;
  degradation_rate: number;
  urgency: 'low' | 'medium' | 'high' | 'critical';
  procurement_flag: boolean;
  reasoning: string;
}

/**
 * One tool invocation in the orchestrator trace.
 * Renders each step the LLM took to reach its decision.
 */
export interface OrchestratorTraceEntry {
  tool: 'anomaly_agent' | 'fault_classifier_agent' | 'predictive_maintenance_agent' | string;
  result: AnomalyResult | FaultResult | PredictiveResult | any;
}

export interface OrchestratorResult {
  machine_id: string;
  agents_called: string[];           // ordered list of tool names the LLM invoked
  trace: OrchestratorTraceEntry[];   // full trace of every tool call and its result
  routing_reason: string;            // human-readable summary of the path taken
  work_order: string;                // natural-language decision
  overall_urgency: 'low' | 'medium' | 'high' | 'critical';
  iterations: number;                // how many LLM round-trips the agent loop took
  error?: string;
}

async function post<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(`${url} failed (${res.status}): ${detail}`);
  }
  return (await res.json()) as T;
}

export async function chat(
  userMessage: string,
  machines: Machine[],
  history: ChatHistoryItem[] = []
): Promise<ChatResponse> {
  return post<ChatResponse>('/api/chat', { userMessage, machines, history });
}

/** Run the agentic orchestrator on a SINGLE machine. */
export async function orchestrateMachine(
  machine: Machine
): Promise<OrchestratorResult> {
  return post<OrchestratorResult>('/api/orchestrate', { machine });
}

/** Run the orchestrator on every machine in parallel. */
export async function orchestrateFleet(
  machines: Machine[]
): Promise<{ results: OrchestratorResult[] }> {
  return post<{ results: OrchestratorResult[] }>(
    '/api/orchestrate/fleet',
    { machines }
  );
}

// /**
//  * Frontend API client for SentinelOps.
//  *
//  * One endpoint: POST /api/chat
//  *   Request:  { userMessage, machines, history? }
//  *   Response: { reply, reasoning, model }
//  *
//  * The DeepSeek API key is never exposed to the browser — it lives on the
//  * Huawei ECS server and is applied there when the server proxies to
//  * https://api.deepseek.com.
//  */

// import type { Machine } from './fakeData';

// export interface ChatHistoryItem {
//   role: 'user' | 'assistant';
//   content: string;
// }

// export interface ChatResponse {
//   reply: string;
//   reasoning: string | null;
//   model: string;
// }

// /**
//  * Send a message to the AI agent.
//  *
//  * @param userMessage  The current user input (or a preset prompt from a button).
//  * @param machines     Live fleet snapshot.
//  * @param history      Prior turns in the conversation. IMPORTANT: do NOT
//  *                     include `reasoning` here — DeepSeek rejects messages
//  *                     containing reasoning_content with HTTP 400. The server
//  *                     strips it defensively but keep the payload clean.
//  */
// export async function chat(
//   userMessage: string,
//   machines: Machine[],
//   history: ChatHistoryItem[] = []
// ): Promise<ChatResponse> {
//   const res = await fetch('/api/chat', {
//     method: 'POST',
//     headers: { 'Content-Type': 'application/json' },
//     body: JSON.stringify({ userMessage, machines, history }),
//   });

//   if (!res.ok) {
//     const detail = await res.text().catch(() => '');
//     throw new Error(`Chat API failed (${res.status}): ${detail}`);
//   }

//   return (await res.json()) as ChatResponse;
// }