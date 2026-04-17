/**
 * MachineCard — single machine entry in the left panel list.
 * Color-coded: green=Normal, yellow=Warning, red=Critical.
 * Shows machine ID, type, anomaly score bar, RUL, and a
 * "SIMULATE BREAKDOWN" / "RESET" button.
 */
import React, { useState } from 'react';
import { simulateBreakdown, resetMachine } from '../services/api';

const STATUS_META = {
  Normal:   { color: 'var(--accent)',  badge: 'NORMAL', cls: 'ok'     },
  Warning:  { color: 'var(--warn)',    badge: 'WATCH',  cls: 'warn'   },
  Critical: { color: 'var(--danger)',  badge: 'ALERT',  cls: 'danger' },
};

export default function MachineCard({ machine, active, onClick, onStatusChange }) {
  const { machine_id, machine_type, status, anomaly_score, rul_hours } = machine;
  const meta = STATUS_META[status] || STATUS_META.Normal;
  const barPct = Math.round(anomaly_score * 100);

  const [busy, setBusy] = useState(false);

  const isBroken = status === 'Critical';

  async function handleBreakdown(e) {
    e.stopPropagation(); // don't trigger card selection
    setBusy(true);
    try {
      if (isBroken) {
        await resetMachine(machine_id);
      } else {
        await simulateBreakdown(machine_id);
      }
      if (onStatusChange) onStatusChange();
    } catch (err) {
      console.error('Breakdown toggle error:', err);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      onClick={onClick}
      style={{
        padding: '12px 16px',
        borderBottom: '1px solid var(--border)',
        cursor: 'pointer',
        background: active ? 'var(--bg4)' : 'transparent',
        borderLeft: `2px solid ${active ? 'var(--accent)' : meta.color === 'var(--accent)' ? 'transparent' : meta.color}`,
        transition: 'background 0.12s',
        animation: 'fadeSlideIn 0.3s ease both',
      }}
    >
      {/* Top row: ID + badge */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 12, fontWeight: 500 }}>
          {machine_id}
        </span>
        <span style={{
          fontFamily: 'var(--mono)',
          fontSize: 9,
          padding: '2px 6px',
          borderRadius: 3,
          border: `1px solid ${meta.color}`,
          color: meta.color,
          background: meta.color === 'var(--accent)' ? 'var(--ok-bg)' : meta.color === 'var(--warn)' ? 'var(--warn-bg)' : 'var(--danger-bg)',
        }}>
          {meta.badge}
        </span>
      </div>

      {/* Machine type */}
      <div style={{ fontSize: 11, color: 'var(--text3)', marginBottom: 6 }}>{machine_type}</div>

      {/* Anomaly score bar */}
      <div style={{ height: 3, background: 'var(--border)', borderRadius: 2, overflow: 'hidden', marginBottom: 5 }}>
        <div style={{
          height: '100%',
          width: `${barPct}%`,
          background: meta.color,
          borderRadius: 2,
          transition: 'width 0.6s ease',
        }} />
      </div>

      {/* Score + RUL */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text2)' }}>
          score <span style={{ color: meta.color }}>{anomaly_score.toFixed(3)}</span>
        </span>
        {rul_hours != null && (
          <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text3)' }}>
            RUL {rul_hours.toFixed(1)}h
          </span>
        )}
      </div>

      {/* Breakdown / Reset button */}
      <button
        onClick={handleBreakdown}
        disabled={busy}
        style={{
          width: '100%',
          fontFamily: 'var(--mono)',
          fontSize: 9,
          padding: '4px 0',
          borderRadius: 4,
          cursor: busy ? 'not-allowed' : 'pointer',
          border: `1px solid ${isBroken ? 'var(--accent)' : 'var(--danger)'}`,
          background: isBroken ? 'var(--ok-bg)' : 'var(--danger-bg)',
          color: isBroken ? 'var(--accent)' : 'var(--danger)',
          letterSpacing: '0.06em',
          transition: 'opacity 0.15s',
          opacity: busy ? 0.5 : 1,
        }}
      >
        {busy ? '…' : isBroken ? '↺ RESET MACHINE' : '⚡ SIMULATE BREAKDOWN'}
      </button>
    </div>
  );
}
