/**
 * MachineCard — single machine entry in the left panel list.
 * Color-coded: green=Normal, yellow=Warning, red=Critical.
 * Shows machine ID, type, anomaly score bar, and RUL.
 */
import React from 'react';

const STATUS_META = {
  Normal:   { color: 'var(--accent)',  badge: 'NORMAL', cls: 'ok'     },
  Warning:  { color: 'var(--warn)',    badge: 'WATCH',  cls: 'warn'   },
  Critical: { color: 'var(--danger)',  badge: 'ALERT',  cls: 'danger' },
};

export default function MachineCard({ machine, active, onClick }) {
  const { machine_id, machine_type, status, anomaly_score, rul_hours, tool_wear } = machine;
  const meta = STATUS_META[status] || STATUS_META.Normal;
  const barPct = Math.round(anomaly_score * 100);

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
      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text2)' }}>
          score <span style={{ color: meta.color }}>{anomaly_score.toFixed(3)}</span>
        </span>
        {rul_hours != null && (
          <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text3)' }}>
            RUL {rul_hours.toFixed(1)}h
          </span>
        )}
      </div>
    </div>
  );
}
