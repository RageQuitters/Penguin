/**
 * SentinelOps API Service
 * Centralises all HTTP calls to the FastAPI backend.
 * The React dev server proxies /api/* and /ws to localhost:8000.
 */

const BASE = '/api';

/** Fetch current machine states from GET /api/machines */
export async function fetchMachines() {
  const res = await fetch(`${BASE}/machines`);
  if (!res.ok) throw new Error(`Failed to fetch machines: ${res.status}`);
  return res.json();
}

/**
 * POST /api/chat — multi-turn AI agent chat with live plant context
 * @param {Array<{role: string, content: string}>} messages
 * @returns {Promise<{reply: string, agent: string}>}
 */
export async function sendChatMessage(messages) {
  const res = await fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages, plant_id: 'Jurong Plant A' }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Chat failed: ${res.status}`);
  }
  return res.json();
}
/*
 * @param {string} machineId
 * @param {{air_temperature, process_temperature, rotational_speed, torque, tool_wear}} reading
 */
export async function analyzeReading(machineId, reading) {
  const res = await fetch(`${BASE}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ machine_id: machineId, reading }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Analysis failed: ${res.status}`);
  }
  return res.json();
}
