/**
 * Assignment sync — Track C
 * ─────────────────────────
 * The server holds the source of truth for in-flight Telegram interactions
 * (callback button taps, "fixed" text shortcuts). We poll its event log,
 * translate each event into Firestore writes, and let the dashboard's
 * existing live queries pick up the change.
 *
 * Why polling? The server has no Firebase Admin SDK configured, and going
 * the other direction (server → Firestore) would require a service account.
 * The client already has Firestore credentials, so we mirror events from
 * here. Polling at 4s feels live enough for demos.
 */

import {
  resolveFaultLog,
  updateAssignment,
  addEngineerLog,
  getOpenAssignments,
  getAllUnresolvedFaults,
  type Assignment,
  type AssignmentStatus,
} from './firebaseService';

const API_BASE = (import.meta as any).env?.VITE_API_BASE_URL ?? '';
const POLL_INTERVAL_MS = 4_000;
const STORAGE_KEY = 'sentinelops_assignment_event_cursor';

export interface ServerAssignmentEvent {
  id: string;
  assignment_id: string;
  machine_id: string;
  engineer_chat_id: string;
  engineer_name: string;
  fault_types: string[];
  kind: 'resolved' | 'in_progress' | 'escalated';
  ts: string;
  note?: string;
}

export interface SyncCallbacks {
  /** Called after each batch of events is applied so the UI can refresh. */
  onChange?: (events: ServerAssignmentEvent[]) => void;
  /** Called on transport / Firestore error so the caller can toast. */
  onError?: (err: Error) => void;
}

let _timer: ReturnType<typeof setInterval> | null = null;
let _cursor = 0;

function loadCursor(): number {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? Number(raw) || 0 : 0;
  } catch { return 0; }
}

function saveCursor(n: number) {
  try { localStorage.setItem(STORAGE_KEY, String(n)); } catch { /* ignore */ }
}

const STATUS_FROM_KIND: Record<ServerAssignmentEvent['kind'], AssignmentStatus> = {
  resolved: 'resolved',
  in_progress: 'in_progress',
  escalated: 'escalated',
};

/**
 * Apply one server event to Firestore.
 *
 *   resolved    → assignment.status = resolved
 *                 + every matching unresolved fault_log marked resolved
 *                 + new engineer_log entry
 *   in_progress → assignment.status = in_progress (no fault changes)
 *   escalated   → assignment.status = escalated
 *                 + engineer_log entry with outcome=escalated
 */
async function applyEvent(event: ServerAssignmentEvent): Promise<void> {
  const status = STATUS_FROM_KIND[event.kind];
  const isResolved = event.kind === 'resolved';
  const isEscalated = event.kind === 'escalated';

  // 1. Update the assignment record itself
  const updates: Partial<Assignment> = { status };
  if (isResolved) {
    updates.resolved_at = event.ts;
    updates.notes = event.note;
  }
  if (isEscalated && event.note) {
    // Persist the escalation reason so the dashboard chat can mention it
    updates.notes = `Escalated: ${event.note}`;
  }
  if (event.assignment_id) {
    await updateAssignment(event.assignment_id, updates).catch(() => {});
  } else {
    // Fallback: locate by machine + engineer if no assignment_id was registered
    const open = await getOpenAssignments();
    const match = open.find(
      (a) => a.machine_id === event.machine_id && a.engineer_name === event.engineer_name,
    );
    if (match?.id) await updateAssignment(match.id, updates).catch(() => {});
  }

  // 2. Resolve fault logs (only for "fixed")
  if (isResolved && event.fault_types.length > 0) {
    const openFaults = await getAllUnresolvedFaults(100);
    const machineFaults = openFaults.filter((f) => f.machine_id === event.machine_id);
    const targetTypes = new Set(event.fault_types);
    await Promise.all(
      machineFaults
        .filter((f) => targetTypes.has(f.fault_type))
        .map((f) =>
          f.id ? resolveFaultLog(f.id, event.engineer_name, event.note ?? 'Confirmed via Telegram') : Promise.resolve(),
        ),
    );
  }

  // 3. Add an engineer_log entry for resolved & escalated
  if (event.kind !== 'in_progress') {
    const faultLabel = event.fault_types.join(', ') || 'general inspection';
    const action = isResolved
      ? `Confirmed fix via Telegram for ${faultLabel}`
      : `Escalated via Telegram for ${faultLabel}${event.note ? ` — reason: ${event.note}` : ''}`;
    await addEngineerLog({
      machine_id: event.machine_id,
      engineer_name: event.engineer_name,
      action,
      timestamp: event.ts,
      fault_types: event.fault_types,
      outcome: isResolved ? 'resolved' : 'escalated',
    }).catch(() => {});
  }
}

async function pollOnce(callbacks: SyncCallbacks): Promise<void> {
  try {
    const res = await fetch(`${API_BASE}/api/assignments/events?since=${_cursor}`);
    if (!res.ok) return;
    const { events } = (await res.json()) as { events: ServerAssignmentEvent[] };
    if (!Array.isArray(events) || events.length === 0) return;

    // Apply in order — server already returns ascending by id
    for (const event of events) {
      await applyEvent(event);
      _cursor = Math.max(_cursor, Number(event.id));
    }
    saveCursor(_cursor);
    callbacks.onChange?.(events);
  } catch (err: any) {
    callbacks.onError?.(err instanceof Error ? err : new Error(String(err)));
  }
}

export function startAssignmentSync(callbacks: SyncCallbacks = {}): () => void {
  if (_timer) return () => stopAssignmentSync();
  _cursor = loadCursor();
  // Run once immediately so the dashboard reflects state on mount
  pollOnce(callbacks);
  _timer = setInterval(() => pollOnce(callbacks), POLL_INTERVAL_MS);
  return () => stopAssignmentSync();
}

export function stopAssignmentSync(): void {
  if (_timer) {
    clearInterval(_timer);
    _timer = null;
  }
}