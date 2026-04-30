/**
 * Fault → engineer specialty routing.
 *
 * Used by both the client (to rank engineers in the assign dialog)
 * and the server (to pick a single recipient when /api/assign-engineer
 * is called without an explicit engineer_id).
 */

export type FaultCode = 'HDF' | 'OSF' | 'PWF' | 'RNF' | 'TWF';

/** Each fault code maps to one or more specialty keywords (lowercased). */
export const FAULT_TO_SPECIALTY: Record<FaultCode, string[]> = {
  HDF: ['hydraulic', 'bearing'],          // Heat Dissipation Failure
  OSF: ['rotational', 'sensor', 'mechanical'], // Overstrain Failure
  PWF: ['power', 'electrical'],           // Power Failure
  RNF: ['rotational', 'mechanical'],      // Random Failure
  TWF: ['tool wear', 'mechanical', 'preventive'], // Tool Wear Failure
};

/** Human-readable fault names — used in Telegram messages. */
export const FAULT_NAMES: Record<FaultCode, string> = {
  HDF: 'Heat Dissipation Failure',
  OSF: 'Overstrain Failure',
  PWF: 'Power Failure',
  RNF: 'Random Failure',
  TWF: 'Tool Wear Failure',
};

export interface RoutableEngineer {
  id?: string;
  name: string;
  specialization: string;
  active: boolean;
  telegram_chat_id?: string;
  /** Optional — if you track current open assignments for load balancing. */
  open_assignments?: number;
}

export interface ScoredEngineer<E extends RoutableEngineer = RoutableEngineer> {
  engineer: E;
  score: number;
  matchedKeywords: string[];
  reason: string;
}

/**
 * Score an engineer against a set of fault types.
 * Higher = better match.
 *
 *  +2  per specialty keyword that appears in the engineer's specialization
 *  -1  per existing open assignment (load penalty)
 *  +0.5 if the engineer has a Telegram chat ID configured (so they can actually be DM'd)
 *
 * Returns 0 for inactive engineers — they should never be picked.
 */
export function scoreEngineer<E extends RoutableEngineer>(
  engineer: E,
  faultTypes: FaultCode[],
): ScoredEngineer<E> {
  if (!engineer.active) {
    return { engineer, score: 0, matchedKeywords: [], reason: 'inactive' };
  }

  const spec = (engineer.specialization || '').toLowerCase();
  const matched = new Set<string>();
  let score = 0;

  for (const fault of faultTypes) {
    for (const kw of FAULT_TO_SPECIALTY[fault] ?? []) {
      if (spec.includes(kw) && !matched.has(kw)) {
        matched.add(kw);
        score += 2;
      }
    }
  }

  if (engineer.telegram_chat_id) score += 0.5;
  if (typeof engineer.open_assignments === 'number') {
    score -= engineer.open_assignments;
  }

  const matchedList = Array.from(matched);
  const reason = matchedList.length
    ? `Specializes in ${matchedList.join(', ')}`
    : 'General match (no specialty keyword overlap)';

  return { engineer, score, matchedKeywords: matchedList, reason };
}

/**
 * Rank a list of engineers for a given fault set. Inactive engineers are dropped.
 * Returns sorted descending by score; ties broken by name for stability.
 */
export function rankEngineers<E extends RoutableEngineer>(
  engineers: E[],
  faultTypes: FaultCode[],
): ScoredEngineer<E>[] {
  return engineers
    .filter((e) => e.active)
    .map((e) => scoreEngineer(e, faultTypes))
    .sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;
      return a.engineer.name.localeCompare(b.engineer.name);
    });
}

/** Convenience: pick the single best-matching engineer, or null if none active. */
export function pickBestEngineer<E extends RoutableEngineer>(
  engineers: E[],
  faultTypes: FaultCode[],
): ScoredEngineer<E> | null {
  const ranked = rankEngineers(engineers, faultTypes);
  return ranked[0] ?? null;
}