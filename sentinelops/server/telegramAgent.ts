/**
 * SentinelOps Telegram Agent
 * ──────────────────────────
 * • Assignment flows: inline buttons (Fixed / WIP / Escalate), quick-resolve
 *   text, escalation routing — unchanged.
 * • LLM replies: answers engineer questions directly. No proactive summaries,
 *   no fleet overviews, no unsolicited suggestions. The bot only answers
 *   what was explicitly asked.
 */

import * as https from 'https';

const BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN ?? '';
const ENGINEER_CHAT_IDS = (process.env.TELEGRAM_ENGINEER_CHAT_IDS ?? '')
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean);

// ─── Telegram HTTP helpers ────────────────────────────────────────────────────

function telegramRequest(method: string, body: Record<string, unknown>): Promise<any> {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify(body);
    const options = {
      hostname: 'api.telegram.org',
      path: `/bot${BOT_TOKEN}/${method}`,
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) },
    };
    const req = https.request(options, (res) => {
      let raw = '';
      res.on('data', (c) => (raw += c));
      res.on('end', () => {
        try { resolve(JSON.parse(raw)); } catch { resolve({}); }
      });
    });
    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

async function sendMessage(
  chatId: string | number,
  text: string,
  parseMode: string | null = 'Markdown',
  extra: Record<string, unknown> = {},
) {
  if (!BOT_TOKEN) return null;
  try {
    const body: Record<string, unknown> = { chat_id: chatId, text, ...extra };
    if (parseMode) body.parse_mode = parseMode;
    const result = await telegramRequest('sendMessage', body);

    if (result && result.ok === false) {
      const desc: string = result.description ?? '';
      const isParseError = /can't parse entities|parse entities|MARKDOWN|MarkdownV2/i.test(desc);
      console.warn(
        `[Telegram] sendMessage rejected for chat ${chatId}: ${result.error_code ?? '?'} ${desc || '(no description)'}`,
      );
      if (isParseError && parseMode) {
        console.warn('[Telegram] Retrying same text without parse_mode…');
        const plainText = text
          .replace(/\*\*?(.*?)\*\*?/g, '$1')
          .replace(/__(.*?)__/g, '$1')
          .replace(/_(.*?)_/g, '$1')
          .replace(/`{1,3}[^`]*`{1,3}/g, '')
          .replace(/#{1,6}\s?/g, '')
          .trim();
        const retryBody: Record<string, unknown> = { chat_id: chatId, text: plainText, ...extra };
        delete retryBody.reply_markup;
        const retry = await telegramRequest('sendMessage', retryBody);
        if (retry && retry.ok === false) {
          console.warn(`[Telegram] Plain-text retry also failed: ${retry.description ?? ''}`);
        }
        return retry;
      }
    }
    return result;
  } catch (err: any) {
    console.warn('[Telegram] sendMessage failed:', err?.message);
    return null;
  }
}

/** Broadcast a notification to all configured engineer chat IDs */
export async function notifyEngineers(message: string): Promise<void> {
  if (!BOT_TOKEN || ENGINEER_CHAT_IDS.length === 0) {
    console.warn('[Telegram] notifyEngineers: no token or chat IDs configured');
    return;
  }
  await Promise.all(ENGINEER_CHAT_IDS.map((id) => sendMessage(id, message)));
  console.log(`[Telegram] Notified ${ENGINEER_CHAT_IDS.length} engineer(s)`);
}

/** Send to a single engineer's chat ID. */
export async function notifyEngineer(
  chatId: string,
  message: string,
): Promise<{ sent: boolean; error?: string }> {
  if (!BOT_TOKEN) return { sent: false, error: 'TELEGRAM_BOT_TOKEN not configured on server' };
  if (!chatId) return { sent: false, error: 'Engineer has no telegram_chat_id' };
  try {
    const result = await sendMessage(chatId, message);
    if (result?.ok) return { sent: true };
    return { sent: false, error: result?.description ?? 'Unknown Telegram error' };
  } catch (err: any) {
    return { sent: false, error: err?.message ?? 'Network error' };
  }
}

// ─── Assignment lifecycle ─────────────────────────────────────────────────────

export type AssignmentEventKind = 'resolved' | 'in_progress' | 'escalated';

export interface AssignmentEvent {
  id: string;
  assignment_id: string;
  machine_id: string;
  engineer_chat_id: string;
  engineer_name: string;
  fault_types: string[];
  kind: AssignmentEventKind;
  ts: string;
  note?: string;
}

interface PendingAssignment {
  assignment_id: string;
  machine_id: string;
  engineer_chat_id: string;
  engineer_name: string;
  fault_types: string[];
  created_at: string;
}

const _pendingAssignments = new Map<string, PendingAssignment>();
const _events: AssignmentEvent[] = [];
let _nextEventId = 1;

/**
 * When an engineer taps 🚨 Escalate we ask for a reason; their very next
 * text message is captured as that reason. Keyed by chat_id.
 */
const _pendingEscalations = new Map<number, { assignment_id: string; asked_at: number }>();
const ESCALATION_WAIT_MS = 10 * 60 * 1000; // 10 minutes

export function registerAssignment(a: PendingAssignment) {
  _pendingAssignments.set(a.assignment_id, a);
}

export function getAssignmentEvents(sinceId = 0): AssignmentEvent[] {
  return _events.filter((e) => Number(e.id) > sinceId);
}

export async function notifyEngineerWithButtons(
  chatId: string,
  message: string,
  assignmentId: string,
): Promise<{ sent: boolean; message_id?: number; error?: string }> {
  if (!BOT_TOKEN) return { sent: false, error: 'TELEGRAM_BOT_TOKEN not configured on server' };
  if (!chatId) return { sent: false, error: 'Engineer has no telegram_chat_id' };

  const inline_keyboard = [
    [
      { text: '✅ Fixed', callback_data: `fix:${assignmentId}` },
      { text: '🔧 In progress', callback_data: `wip:${assignmentId}` },
    ],
    [{ text: '🚨 Escalate', callback_data: `esc:${assignmentId}` }],
  ];

  try {
    const result = await sendMessage(chatId, message, 'Markdown', {
      reply_markup: { inline_keyboard },
    });
    if (result?.ok) {
      // Record into history so the LLM can resolve "this issue" / "this machine"
      // references against the most recent assignment notification.
      appendChatTurn(chatId, 'assistant', message, { isAssignment: true });
      return { sent: true, message_id: result?.result?.message_id };
    }
    return { sent: false, error: result?.description ?? 'Unknown Telegram error' };
  } catch (err: any) {
    return { sent: false, error: err?.message ?? 'Network error' };
  }
}

// ─── Event recording ──────────────────────────────────────────────────────────

function recordEvent(pending: PendingAssignment, kind: AssignmentEventKind, note?: string) {
  const event: AssignmentEvent = {
    id: String(_nextEventId++),
    assignment_id: pending.assignment_id,
    machine_id: pending.machine_id,
    engineer_chat_id: pending.engineer_chat_id,
    engineer_name: pending.engineer_name,
    fault_types: pending.fault_types,
    kind,
    ts: new Date().toISOString(),
    note,
  };
  _events.push(event);
  if (_events.length > 500) _events.splice(0, _events.length - 500);
  console.log(`[Assignment] ${pending.machine_id} → ${kind} by ${pending.engineer_name}`);
  return event;
}

// ─── Callback query handler (button taps) ────────────────────────────────────

async function answerCallbackQuery(callbackId: string, text?: string) {
  return telegramRequest('answerCallbackQuery', {
    callback_query_id: callbackId,
    text: text ?? '',
  }).catch(() => {});
}

async function handleCallbackQuery(query: any) {
  const data: string = query?.data ?? '';
  const callbackId: string = query?.id;
  const fromName: string =
    [query?.from?.first_name, query?.from?.last_name].filter(Boolean).join(' ') || 'Engineer';

  const [action, assignmentId] = data.split(':');
  const pending = _pendingAssignments.get(assignmentId);

  if (!pending) {
    await answerCallbackQuery(callbackId, 'This assignment is no longer tracked.');
    return;
  }

  let ackText = '';
  let publicMsg = '';
  let kind: AssignmentEventKind | null = null;

  switch (action) {
    case 'fix':
      kind = 'resolved';
      ackText = '✅ Marked as fixed';
      publicMsg = [
        `✅ *Fixed by ${fromName}*`,
        ``,
        `*Machine:* ${pending.machine_id}`,
        `*Faults:* ${pending.fault_types.join(', ') || 'general'}`,
        ``,
        `Fault logs and assignment status have been updated.`,
      ].join('\n');
      clearFaultFlagsForSimulation(pending.machine_id, pending.fault_types);
      _pendingAssignments.delete(assignmentId);
      break;

    case 'wip':
      kind = 'in_progress';
      ackText = '🔧 Marked in progress';
      publicMsg = [
        `🔧 *${fromName} is working on ${pending.machine_id}*`,
        ``,
        `Faults: ${pending.fault_types.join(', ') || 'general'}`,
      ].join('\n');
      break;

    case 'esc': {
      // Don't emit event yet — wait for engineer's reason text.
      const askChatId = Number(pending.engineer_chat_id);
      if (isNaN(askChatId)) {
        // Can't register the pending escalation — engineer_chat_id is not a valid number.
        // Fall back to recording the escalation immediately without a reason.
        console.warn(
          `[Telegram] esc: engineer_chat_id "${pending.engineer_chat_id}" is not a valid number — recording escalation immediately`,
        );
        recordEvent(pending, 'escalated', '(no reason — chat ID invalid)');
        _pendingAssignments.delete(assignmentId);
        await answerCallbackQuery(callbackId, '🚨 Escalated (no reason collected)');
        return;
      }

      _pendingEscalations.set(askChatId, { assignment_id: assignmentId, asked_at: Date.now() });
      await answerCallbackQuery(callbackId, '🚨 Escalation noted — please reply with a reason');
      await sendMessage(
        pending.engineer_chat_id,
        [
          `🚨 *Escalation requested for ${pending.machine_id}*`,
          ``,
          `Please reply with a brief reason (e.g. _"missing torque wrench"_, _"fault unclear"_, _"safety concern"_).`,
          ``,
          `Your reason will be passed to a senior engineer.`,
        ].join('\n'),
      );
      return;
    }

    default:
      await answerCallbackQuery(callbackId, 'Unknown action');
      return;
  }

  if (kind) recordEvent(pending, kind);
  await answerCallbackQuery(callbackId, ackText);
  await sendMessage(pending.engineer_chat_id, publicMsg);
}

// ─── Escalation reason capture ────────────────────────────────────────────────

function tryConsumeEscalationReason(chatId: number, text: string, fromName: string): boolean {
  const pendingEsc = _pendingEscalations.get(chatId);
  if (!pendingEsc) return false;

  if (Date.now() - pendingEsc.asked_at > ESCALATION_WAIT_MS) {
    _pendingEscalations.delete(chatId);
    return false;
  }

  const pending = _pendingAssignments.get(pendingEsc.assignment_id);
  if (!pending) {
    _pendingEscalations.delete(chatId);
    return false;
  }

  const reason = text.trim();
  recordEvent(pending, 'escalated', reason);
  _pendingAssignments.delete(pending.assignment_id);
  _pendingEscalations.delete(chatId);

  sendMessage(
    chatId,
    [
      `🚨 *Escalation recorded for ${pending.machine_id}*`,
      ``,
      `*Reason:* ${reason}`,
      ``,
      `Passed to a senior engineer. Thanks ${fromName} — you can step away from this one.`,
    ].join('\n'),
  );

  const seniors = getSeniorEngineers().filter(
    (e) => String(e.telegram_chat_id) !== String(chatId),
  );

  const broadcast = [
    `🚨 *Escalation from ${pending.engineer_name}*`,
    ``,
    `*Machine:* ${pending.machine_id}`,
    `*Faults:* ${pending.fault_types.join(', ') || 'general'}`,
    `*Reason:* ${reason}`,
    ``,
    seniors.length > 0
      ? `You are receiving this as a senior engineer. Please take over and reply *fixed ${pending.machine_id}* once resolved.`
      : `Manual support requested — please pick this up.`,
  ].join('\n');

  if (seniors.length > 0) {
    console.log(`[Telegram] Routing escalation to ${seniors.length} senior(s): ${seniors.map((s) => s.name).join(', ')}`);
    Promise.all(seniors.map((s) => sendMessage(s.telegram_chat_id!, broadcast))).catch(() => {});
  } else if (ENGINEER_CHAT_IDS.length > 0) {
    console.warn('[Telegram] No senior engineers in roster — falling back to env broadcast list');
    Promise.all(
      ENGINEER_CHAT_IDS
        .filter((id) => String(id) !== String(chatId))
        .map((id) => sendMessage(id, broadcast)),
    ).catch(() => {});
  } else {
    console.warn('[Telegram] No senior engineers AND no broadcast list — escalation will not be forwarded');
  }

  return true;
}

// ─── Quick-resolve text shortcut ──────────────────────────────────────────────

function tryQuickResolveByText(chatId: number, text: string): boolean {
  const lower = text.toLowerCase().trim();
  const fixedKeywords = ['fixed', 'done', 'resolved', 'complete', 'completed', 'closed'];
  if (!fixedKeywords.some((k) => lower.startsWith(k) || lower === k)) return false;

  const pendings = Array.from(_pendingAssignments.values()).filter(
    (p) => String(p.engineer_chat_id) === String(chatId),
  );
  if (pendings.length === 0) return false;

  const machineIdMatch = text.match(/[A-Z]-\d{1,2}/i);
  let target = pendings[0];
  if (machineIdMatch) {
    const want = machineIdMatch[0].toUpperCase();
    target = pendings.find((p) => p.machine_id.toUpperCase() === want) ?? target;
  }

  recordEvent(target, 'resolved');
  clearFaultFlagsForSimulation(target.machine_id, target.fault_types);
  _pendingAssignments.delete(target.assignment_id);

  const ackMsg = `✅ *Marked ${target.machine_id} as fixed.*\nThanks for closing the loop, ${target.engineer_name.split(' ')[0]}. The machine is now treated as Normal.`;
  sendMessage(chatId, ackMsg).then((res) => {
    if (res?.ok) appendChatTurn(chatId, 'assistant', ackMsg);
  });
  appendChatTurn(chatId, 'user', text);
  return true;
}

// ─── Machine / history context ────────────────────────────────────────────────

let _currentMachines: any[] = [];
let _currentAnomalyHistory: Record<string, any[]> = {};
let _currentFaultHistory: Record<string, any[]> = {};

export function updateAgentContext(
  machines: any[],
  anomalyHistory: Record<string, any[]>,
  faultHistory: Record<string, any[]>,
) {
  _currentMachines = machines;
  _currentAnomalyHistory = anomalyHistory;
  _currentFaultHistory = faultHistory;
}

function clearFaultFlagsForSimulation(machineId: string, faultTypes: string[]) {
  const idx = _currentMachines.findIndex((m) => m.machine_id === machineId);
  if (idx < 0) return;
  const machine = _currentMachines[idx];
  for (const f of faultTypes) {
    if (machine[f] === 1) machine[f] = 0;
  }
  if (machine.anomaly_score > 0.6) machine.anomaly_score = 0.3;
  console.log(`[Sim] Cleared fault flags on ${machineId}: ${faultTypes.join(', ')}`);
}

// ─── Engineer roster (for escalation routing + LLM identity) ─────────────────

interface CachedEngineer {
  id?: string;
  name: string;
  role?: string;
  specialization?: string;
  active: boolean;
  telegram_chat_id?: string;
}

let _engineerRoster: CachedEngineer[] = [];

export function updateEngineerRoster(engineers: CachedEngineer[]) {
  _engineerRoster = engineers.filter((e) => e?.name);
  console.log(`[Telegram] Engineer roster cached: ${_engineerRoster.length} engineers`);
}

function getSeniorEngineers(): CachedEngineer[] {
  return _engineerRoster.filter(
    (e) => e.active && e.telegram_chat_id && /senior|lead|principal/i.test(e.role ?? ''),
  );
}

function findEngineerByChatId(chatId: number | string): CachedEngineer | undefined {
  return _engineerRoster.find((e) => String(e.telegram_chat_id) === String(chatId));
}

// ─── Conversation history ─────────────────────────────────────────────────────
// Lets the LLM resolve "this machine" / "this issue" across turns.

interface ChatTurn {
  role: 'user' | 'assistant';
  content: string;
  ts: number;
  isAssignment?: boolean;
}

const _chatHistories = new Map<string, ChatTurn[]>();
const MAX_HISTORY_TURNS = 16;

function appendChatTurn(
  chatId: number | string,
  role: 'user' | 'assistant',
  content: string,
  meta: { isAssignment?: boolean } = {},
) {
  const key = String(chatId);
  const arr = _chatHistories.get(key) ?? [];
  arr.push({ role, content, ts: Date.now(), ...meta });
  if (arr.length > MAX_HISTORY_TURNS) arr.splice(0, arr.length - MAX_HISTORY_TURNS);
  _chatHistories.set(key, arr);
}

function getChatHistory(chatId: number | string): ChatTurn[] {
  return _chatHistories.get(String(chatId)) ?? [];
}

// ─── LLM configuration ────────────────────────────────────────────────────────

const LLM_BASE_URL = process.env.LLM_BASE_URL || 'https://api-ap-southeast-1.modelarts-maas.com/openai/v1';
const LLM_URL = `${LLM_BASE_URL}/chat/completions`;
const LLM_API_KEY = process.env.LLM_API_KEY;
const LLM_MODEL = process.env.LLM_MODEL || 'DeepSeek-V3';

// Brief fix guidance per fault type — used when engineer asks "how do I fix this".
const FAULT_FIX_GUIDANCE: Record<string, string> = {
  HDF: 'Heat Dissipation Failure — check coolant flow, verify fan operation, clean heat exchanger fins, inspect for blocked vents, and confirm process_temperature vs air_temperature delta.',
  OSF: 'Overstrain Failure — inspect bearings and couplings for wear, verify torque is within spec, check for misalignment, and consider reducing load until cleared.',
  PWF: 'Power Failure — check supply voltage stability, inspect contactors and relays, look for loose terminals, test grounding, and verify VFD/inverter health.',
  RNF: 'Random Failure — capture sensor traces, look for intermittent wiring faults, inspect connectors, and review event logs for correlated faults.',
  TWF: 'Tool Wear Failure — measure remaining tool life vs tool_wear value, schedule a tool change, inspect cutting edges, and verify feed/speed parameters.',
};

// ─── System prompt ────────────────────────────────────────────────────────────

const SYSTEM_PROMPT = `You are SentinelOps AI Engineer Assistant, answering maintenance engineers via Telegram.

OUTPUT RULES — non-negotiable:
- Answer ONLY the specific question asked. Nothing else.
- Keep replies under 150 words. Engineers read on a phone in the field.
- Use Telegram bold (*text*) for key values. No # headings, no backtick fences.
- Do not greet. Do not sign off. Get straight to the answer.
- Do NOT volunteer fleet summaries, machine rankings, anomaly overviews, or suggestions the engineer did not ask for.
- Do NOT append "other things to check", "by the way", or next-step recommendations unless explicitly requested.

CRITICAL: WHEN THE ENGINEER HAS AN OPEN ASSIGNMENT, NEVER PRODUCE:
- "Priority list", "priority order", or any ranked list of multiple machines
- "Severe (Immediate action)" / "Moderate (Monitor closely)" / fleet classification sections
- "Suggested actions" lists that span multiple machines
- ANY section that begins with "Here's the priority list" or similar
- Even if the engineer's question contains the words "potential work", "what else", or "other issues" — interpret it ONLY as relating to their current assignment. If they want fleet info, they'll explicitly ask "show me the fleet" or "fleet status".

If the question is ambiguous and you're tempted to produce two sections, OUTPUT ONLY THE FIX-STEPS SECTION for their current assignment.

CONTEXT USE:
- Use conversation history to resolve "this", "it", "this machine", "this issue" — they refer to whatever was most recently discussed or assigned.
- If an [ASSIGNMENT NOTIFICATION] appears in history, that machine is the subject unless the engineer names a different one.

MACHINE STATUS CLASSIFICATION (apply only when explicitly asked for fleet status):
- *Severe*: at least one fault flag active (HDF / OSF / PWF / RNF / TWF = 1)
- *Moderate*: no active faults but anomaly_score > 0.6
- *Normal*: anomaly_score ≤ 0.6, no active faults

For fix/repair questions: give concise numbered steps for the SPECIFIC MACHINE ONLY.
End fix-step replies with: Reply *fixed [machine_id]* once done.
NOTHING after that line. Stop. The reply is complete.`;

// ─── LLM reply generation ─────────────────────────────────────────────────────

async function generateTelegramReply(userText: string, chatId: number): Promise<string> {
  const myAssignments = Array.from(_pendingAssignments.values()).filter(
    (p) => String(p.engineer_chat_id) === String(chatId),
  );
  const engineer = findEngineerByChatId(chatId);

  // Build open-assignments block with live machine data + fix hints
  const myAssignmentBlock = myAssignments.length > 0
    ? myAssignments.map((a) => {
        const m = _currentMachines.find((x) => x.machine_id === a.machine_id);
        if (!m) return `- ${a.machine_id} (faults: ${a.fault_types.join(', ') || 'general'}) — machine data unavailable`;
        const fixHints = a.fault_types
          .map((f) => FAULT_FIX_GUIDANCE[f])
          .filter(Boolean)
          .join('\n  ');
        return [
          `- *${m.machine_id}* (${m.machine_type ?? 'unknown'} type)`,
          `  Faults: ${a.fault_types.join(', ') || 'general inspection'}`,
          `  anomaly_score=${(m.anomaly_score ?? 0).toFixed(3)}, RUL=${(m.rul_hours ?? 0).toFixed(1)}h, tool_wear=${m.tool_wear ?? '?'} min`,
          `  air_temp=${m.air_temperature}, process_temp=${m.process_temperature}, rotational_speed=${m.rotational_speed}, torque=${m.torque}`,
          fixHints ? `  Fix hints:\n  ${fixHints}` : '',
        ].filter(Boolean).join('\n');
      }).join('\n\n')
    : '(none)';

  // Fallback when no LLM key
  if (!LLM_API_KEY) {
    if (myAssignments.length > 0 && /how|fix|issue|problem|what/i.test(userText)) {
      const a = myAssignments[0];
      const m = _currentMachines.find((x) => x.machine_id === a.machine_id);
      const hints = a.fault_types.map((f) => FAULT_FIX_GUIDANCE[f]).filter(Boolean).join('\n\n');
      return [
        `🛠 *${a.machine_id}* — ${a.fault_types.join(', ')}`,
        m ? `Anomaly: ${m.anomaly_score?.toFixed(3)} | RUL: ${m.rul_hours?.toFixed(1)}h | Tool wear: ${m.tool_wear} min` : '',
        '',
        hints || 'No specific guidance for these faults.',
        '',
        `Reply *fixed ${a.machine_id}* once resolved.`,
      ].filter(Boolean).join('\n');
    }
    return `SentinelOps AI is unavailable (no LLM key). Check the dashboard for machine status.`;
  }

  try {
    // Trim fleet snapshot to save tokens — LLM only needs IDs + key metrics.
    // Included so the LLM can answer specific machine queries, but the system
    // prompt instructs it not to volunteer this data unprompted.
    const trimmedMachines = _currentMachines.map((m) => ({
      machine_id: m.machine_id,
      anomaly_score: m.anomaly_score,
      rul_hours: m.rul_hours,
      tool_wear: m.tool_wear,
      faults: ['HDF', 'OSF', 'PWF', 'RNF', 'TWF'].filter((f) => m[f] === 1),
    }));

    const engineerLine = engineer
      ? `Engineer: *${engineer.name}* (${engineer.role ?? 'role unknown'}, specializes in ${engineer.specialization ?? 'n/a'})`
      : 'Engineer identity unknown.';

    const userContext = [
      engineerLine,
      '',
      myAssignments.length > 0
        ? `OPEN ASSIGNMENTS (anchor "this machine" / "this issue" references here):\n${myAssignmentBlock}`
        : 'No open assignments.',
      '',
      `Fleet data (use only when the engineer explicitly asks about a specific machine or metric):\n${JSON.stringify(trimmedMachines).slice(0, 2000)}`,
    ].join('\n');

    // Replay conversation history — replace verbose assignment messages with
    // compact markers to save tokens without losing reference context.
    const history = getChatHistory(chatId);
    const replayedTurns = history.map((turn) => {
      if (turn.isAssignment) {
        const machineMatch = turn.content.match(/Assignment\s+[—-]\s+([A-Z]-\d{1,2})/i);
        const machineId = machineMatch?.[1] ?? 'a machine';
        const faultMatch = turn.content.match(/\*Faults?:\*\s*([^\n]+)/i);
        const faults = faultMatch?.[1]?.trim() ?? 'unspecified';
        return {
          role: 'assistant' as const,
          content: `[ASSIGNMENT NOTIFICATION SENT] Assigned engineer to ${machineId} with faults: ${faults}.`,
        };
      }
      return { role: turn.role, content: turn.content };
    });

    const res = await fetch(LLM_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${LLM_API_KEY}` },
      body: JSON.stringify({
        model: LLM_MODEL,
        temperature: 0.3,
        max_tokens: 400,
        messages: [
          { role: 'system', content: `${SYSTEM_PROMPT}\n\n${userContext}` },
          ...replayedTurns,
          { role: 'user', content: userText },
        ],
      }),
    });

    if (!res.ok) {
      const body = await res.text().catch(() => '');
      console.warn(`[Telegram AI] LLM HTTP ${res.status}: ${body.slice(0, 300)}`);
      return '⚠️ AI service unavailable. Try again in a moment.';
    }

    const data = await res.json();
    const content = data?.choices?.[0]?.message?.content;
    if (!content) {
      console.warn('[Telegram AI] Empty LLM response:', JSON.stringify(data).slice(0, 300));
      return 'Sorry, I could not generate a response. Try rephrasing.';
    }

    // Defense-in-depth: even if the LLM ignores the prompt, strip any fleet-ranking
    // section when the engineer is in assignment mode. We detect common headers
    // ("priority list", "Severe (Immediate", "Moderate (Monitor", "Suggested actions")
    // and truncate from the earliest match onward.
    const inAssignmentMode = myAssignments.length > 0;
    const cleaned = inAssignmentMode ? stripFleetRanking(content) : content;
    if (cleaned !== content) {
      console.log('[Telegram AI] Stripped fleet-ranking section from assignment-mode reply');
    }
    return cleaned;
  } catch (err: any) {
    console.warn('[Telegram AI] LLM call failed:', err?.message);
    return '⚠️ AI service unavailable. Check machine status in the SentinelOps dashboard.';
  }
}

/**
 * Strip fleet-ranking sections from a reply.
 * Looks for the earliest occurrence of common ranking headers and truncates
 * from there to the end. Preserves trailing whitespace cleanup.
 */
function stripFleetRanking(text: string): string {
  // Patterns ordered roughly by how clearly they signal a ranking dump
  const rankingPatterns = [
    /(?:^|\n)\s*Here'?s\s+(?:the\s+)?priority\s+list/i,
    /(?:^|\n)\s*Priority\s+list\b/i,
    /(?:^|\n)\s*Priority\s+order\s*:/i,
    /(?:^|\n)\s*\*?Severe\s+\(Immediate/i,
    /(?:^|\n)\s*\*?Moderate\s+\(Monitor/i,
    /(?:^|\n)\s*\*?Suggested\s+actions\s*:/i,
    /(?:^|\n)\s*Across\s+the\s+fleet\b/i,
    /(?:^|\n)\s*Fleet\s+(?:summary|status|overview)\s*:/i,
  ];

  let earliest = -1;
  for (const re of rankingPatterns) {
    const match = re.exec(text);
    if (match && (earliest < 0 || match.index < earliest)) earliest = match.index;
  }

  if (earliest < 0) return text;
  // Trim trailing whitespace/newlines from the kept portion
  return text.slice(0, earliest).trimEnd();
}

// ─── Long-polling loop ────────────────────────────────────────────────────────

let _lastUpdateId = 0;
let _polling = false;
const _processedUpdateIds = new Set<number>();
const PROCESSED_IDS_MAX = 200;
// Per-chat in-flight lock: prevents double-replies and the race where a
// message arrives before registerAssignment() is called from the dashboard.
const _chatInFlight = new Set<number>();

async function pollOnce() {
  if (!BOT_TOKEN) return;
  try {
    const result = await telegramRequest('getUpdates', {
      offset: _lastUpdateId + 1,
      timeout: 20,
      allowed_updates: ['message', 'callback_query'],
    });

    // Telegram returns 409 Conflict when another instance is also calling
    // getUpdates with the same token — this is the #1 cause of duplicate
    // replies during demos. Surface it loudly so the user knows to kill
    // the other process.
    if (result?.ok === false && result?.error_code === 409) {
      console.error(
        `[Telegram] CONFLICT: another bot instance is polling the same token. ` +
        `Kill any other dev/prod servers running this bot. (${result?.description ?? ''})`,
      );
      return;
    }

    if (!result.ok || !Array.isArray(result.result)) return;

    for (const update of result.result) {
      _lastUpdateId = Math.max(_lastUpdateId, update.update_id);

      if (_processedUpdateIds.has(update.update_id)) {
        console.warn(`[Telegram] Skipping duplicate update_id=${update.update_id}`);
        continue;
      }
      _processedUpdateIds.add(update.update_id);
      if (_processedUpdateIds.size > PROCESSED_IDS_MAX) {
        const oldest = _processedUpdateIds.values().next().value;
        if (oldest !== undefined) _processedUpdateIds.delete(oldest);
      }

      // Inline-button taps
      if (update.callback_query) {
        await handleCallbackQuery(update.callback_query).catch((e) =>
          console.warn('[Telegram] callback handler failed:', e?.message),
        );
        continue;
      }

      const msg = update.message;
      if (!msg || !msg.text) continue;

      const chatId: number = msg.chat.id;
      const text: string = msg.text;
      const from = msg.from?.first_name ?? 'Engineer';

      console.log(`[Telegram] Message from ${from} (${chatId}): ${text}`);

      // 1. Pending escalation reason — consume before anything else
      if (tryConsumeEscalationReason(chatId, text, from)) continue;

      // 2. Quick-resolve shortcut ("fixed U-03")
      if (tryQuickResolveByText(chatId, text)) continue;

      // 3. Serialize per-chat to avoid double-replies
      if (_chatInFlight.has(chatId)) {
        console.warn(`[Telegram] Dropping message from ${chatId} — previous reply still in flight`);
        continue;
      }

      _chatInFlight.add(chatId);
      try {
        telegramRequest('sendChatAction', { chat_id: chatId, action: 'typing' }).catch(() => {});

        const reply = await generateTelegramReply(text, chatId);
        const sendResult = await sendMessage(chatId, reply);

        appendChatTurn(chatId, 'user', text);
        if (sendResult?.ok) appendChatTurn(chatId, 'assistant', reply);
      } finally {
        _chatInFlight.delete(chatId);
      }
    }
  } catch (err: any) {
    console.warn('[Telegram] poll error:', err?.message);
  }
}

export function startTelegramBot() {
  if (!BOT_TOKEN) {
    console.warn('[Telegram] TELEGRAM_BOT_TOKEN not set — bot disabled');
    return;
  }
  if (_polling) return;
  _polling = true;
  console.log('[Telegram] Bot started, polling for messages…');
  const loop = async () => {
    while (_polling) {
      await pollOnce();
      await new Promise((r) => setTimeout(r, 1000));
    }
  };
  loop().catch((e) => console.error('[Telegram] loop crashed:', e));
}

export function stopTelegramBot() {
  _polling = false;
}