/**
 * SentinelOps Telegram AI Agent
 * ─────────────────────────────
 * Listens for incoming Telegram messages from engineers and responds with
 * AI-powered answers about machine status, anomalies, and fault history.
 *
 * Also exposes:
 *   notifyEngineers(message) — broadcast an alert to all configured engineer chats
 */

import * as https from 'https';

const BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN ?? '';
const ENGINEER_CHAT_IDS = (process.env.TELEGRAM_ENGINEER_CHAT_IDS ?? '')
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean);

// ─── Telegram HTTP helpers ─────────────────────────────────────────────────────

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

async function sendMessage(chatId: string | number, text: string, parseMode = 'Markdown') {
  if (!BOT_TOKEN) return;
  try {
    await telegramRequest('sendMessage', { chat_id: chatId, text, parse_mode: parseMode });
  } catch (err: any) {
    console.warn('[Telegram] sendMessage failed:', err?.message);
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

// ─── AI response generation for engineer queries ───────────────────────────────

const LLM_BASE_URL = process.env.LLM_BASE_URL || 'https://api-ap-southeast-1.modelarts-maas.com/openai/v1';
const LLM_URL = `${LLM_BASE_URL}/chat/completions`;
const LLM_API_KEY = process.env.LLM_API_KEY;
const LLM_MODEL = process.env.LLM_MODEL || 'DeepSeek-V3';

let _currentMachines: any[] = [];
let _currentAnomalyHistory: Record<string, any[]> = {};
let _currentFaultHistory: Record<string, any[]> = {};

/** Call this from the main server whenever machines are refreshed */
export function updateAgentContext(
  machines: any[],
  anomalyHistory: Record<string, any[]>,
  faultHistory: Record<string, any[]>
) {
  _currentMachines = machines;
  _currentAnomalyHistory = anomalyHistory;
  _currentFaultHistory = faultHistory;
}

const TELEGRAM_SYSTEM_PROMPT = `You are SentinelOps AI Engineer Assistant, communicating via Telegram.
You help maintenance engineers understand machine status, anomalies, and faults.
You have access to live machine data, recent anomaly scores, and fault history.

Keep replies concise (Telegram-friendly, under 300 words). Use bold (*text*) for emphasis.
Classification:
- *Severe*: ≥1 predicted fault (HDF/OSF/PWF/RNF/TWF = 1) → immediate action
- *Moderate*: no faults, anomaly_score > 0.6 → monitor closely
- *Normal*: anomaly_score ≤ 0.6, no faults

If asked "what should I fix" or "where should I go", rank machines by severity then RUL.`;

async function generateTelegramReply(userText: string, chatId: number): Promise<string> {
  if (!LLM_API_KEY) {
    // Fallback: simple rule-based replies
    const severe = _currentMachines.filter((m) =>
      [m.HDF, m.OSF, m.PWF, m.RNF, m.TWF].some((v) => v === 1)
    );
    const moderate = _currentMachines.filter(
      (m) => !severe.includes(m) && m.anomaly_score > 0.6
    );
    if (userText.toLowerCase().includes('status') || userText.toLowerCase().includes('hello')) {
      return `🤖 *SentinelOps Status*\n\n✅ Normal: ${_currentMachines.length - severe.length - moderate.length}\n⚠️ Moderate: ${moderate.length}\n🚨 Severe: ${severe.length}\n\nType "severe" to see critical machines.`;
    }
    if (userText.toLowerCase().includes('severe') || userText.toLowerCase().includes('critical')) {
      if (severe.length === 0) return '✅ No severe machines at this time.';
      return `🚨 *Severe Machines:*\n${severe.map((m) => `- *${m.machine_id}*: score ${m.anomaly_score.toFixed(2)}, RUL ${m.rul_hours.toFixed(0)}h`).join('\n')}`;
    }
    return `I'm the SentinelOps AI. Send me "status" for fleet overview or "severe" for critical machines.`;
  }

  try {
    const context = {
      machines: _currentMachines,
      recent_anomalies: _currentAnomalyHistory,
      recent_faults: _currentFaultHistory,
    };
    const res = await fetch(LLM_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${LLM_API_KEY}` },
      body: JSON.stringify({
        model: LLM_MODEL,
        temperature: 0.3,
        max_tokens: 500,
        messages: [
          { role: 'system', content: `${TELEGRAM_SYSTEM_PROMPT}\n\nLive data:\n${JSON.stringify(context).slice(0, 4000)}` },
          { role: 'user', content: userText },
        ],
      }),
    });
    const data = await res.json();
    return data?.choices?.[0]?.message?.content ?? 'Sorry, I could not generate a response.';
  } catch (err: any) {
    console.warn('[Telegram AI] LLM call failed:', err?.message);
    return '⚠️ AI service unavailable. Check machine status in the SentinelOps dashboard.';
  }
}

// ─── Long-polling loop ─────────────────────────────────────────────────────────

let _lastUpdateId = 0;
let _polling = false;

async function pollOnce() {
  if (!BOT_TOKEN) return;
  try {
    const result = await telegramRequest('getUpdates', {
      offset: _lastUpdateId + 1,
      timeout: 20,
      allowed_updates: ['message'],
    });
    if (!result.ok || !Array.isArray(result.result)) return;

    for (const update of result.result) {
      _lastUpdateId = Math.max(_lastUpdateId, update.update_id);
      const msg = update.message;
      if (!msg || !msg.text) continue;

      const chatId: number = msg.chat.id;
      const text: string = msg.text;
      const from = msg.from?.first_name ?? 'Engineer';

      console.log(`[Telegram] Message from ${from} (${chatId}): ${text}`);

      // Typing indicator
      telegramRequest('sendChatAction', { chat_id: chatId, action: 'typing' }).catch(() => {});

      const reply = await generateTelegramReply(text, chatId);
      await sendMessage(chatId, reply);
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
