/**
 * NotificationToast — shows trace log entries as pop-up notifications
 * as soon as they arrive via WebSocket, rather than only in the trace panel.
 * Notifications auto-dismiss after 5s. Max 3 visible at once.
 */
import React, { useState, useEffect, useCallback, useRef } from 'react';

const AGENT_COLORS = {
  'Orchestrator':                 '#00d4aa',
  'Anomaly Detection Agent':      '#5b9cf6',
  'Fault Classification Agent':   '#f5a623',
  'Predictive Maintenance Agent': '#b085f5',
};

const ACTION_ICONS = {
  tool_call: '⚙',
  reasoning: '◈',
  decision:  '◉',
  handoff:   '→',
};

// Only show notifications for specific action types to avoid noise
const NOTIFY_ACTIONS = new Set(['decision', 'handoff']);

function Toast({ notif, onDismiss }) {
  const color = AGENT_COLORS[notif.agent] || 'var(--text2)';
  const icon = ACTION_ICONS[notif.action] || '·';
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    // Animate in
    const t1 = setTimeout(() => setVisible(true), 10);
    // Auto-dismiss
    const t2 = setTimeout(() => {
      setVisible(false);
      setTimeout(onDismiss, 300);
    }, 5000);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, [onDismiss]);

  return (
    <div
      onClick={() => { setVisible(false); setTimeout(onDismiss, 300); }}
      style={{
        cursor: 'pointer',
        background: 'var(--bg3)',
        border: `1px solid ${color}40`,
        borderLeft: `3px solid ${color}`,
        borderRadius: 6,
        padding: '10px 12px',
        maxWidth: 320,
        boxShadow: `0 4px 20px rgba(0,0,0,0.6), 0 0 0 1px ${color}10`,
        transform: visible ? 'translateX(0)' : 'translateX(340px)',
        opacity: visible ? 1 : 0,
        transition: 'transform 0.3s cubic-bezier(0.16,1,0.3,1), opacity 0.3s ease',
        willChange: 'transform, opacity',
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 5 }}>
        <span style={{
          fontFamily: 'var(--mono)',
          fontSize: 9,
          padding: '1px 5px',
          borderRadius: 3,
          background: `${color}18`,
          border: `1px solid ${color}40`,
          color,
        }}>
          {icon} {notif.action?.toUpperCase()}
        </span>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--text3)', marginLeft: 'auto' }}>
          {new Date(notif.timestamp).toLocaleTimeString('en-SG', {
            hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
          })}
        </span>
      </div>
      {/* Agent */}
      <div style={{ fontFamily: 'var(--mono)', fontSize: 9, color, marginBottom: 3 }}>
        {notif.agent}
      </div>
      {/* Explanation */}
      {notif.explanation && (
        <div style={{ fontSize: 11, color: 'var(--text)', lineHeight: 1.5 }}>
          {notif.explanation.length > 100
            ? notif.explanation.slice(0, 100) + '…'
            : notif.explanation}
        </div>
      )}
    </div>
  );
}

export default function NotificationToasts({ entries }) {
  const [toasts, setToasts] = useState([]);
  const seenIds = useRef(new Set());

  useEffect(() => {
    if (!entries || entries.length === 0) return;
    const latest = entries[entries.length - 1];
    if (!latest || seenIds.current.has(latest.id)) return;
    seenIds.current.add(latest.id);

    // Only pop notification for decision/handoff to reduce noise
    if (!NOTIFY_ACTIONS.has(latest.action)) return;

    const toastId = latest.id || Date.now();
    setToasts(prev => {
      const next = [...prev, { ...latest, toastId }];
      return next.slice(-3); // max 3 toasts
    });
  }, [entries]);

  const dismiss = useCallback((toastId) => {
    setToasts(prev => prev.filter(t => t.toastId !== toastId));
  }, []);

  if (toasts.length === 0) return null;

  return (
    <div style={{
      position: 'fixed',
      bottom: 24,
      right: 24,
      zIndex: 9000,
      display: 'flex',
      flexDirection: 'column',
      gap: 10,
      pointerEvents: 'none',
    }}>
      {toasts.map(t => (
        <div key={t.toastId} style={{ pointerEvents: 'auto' }}>
          <Toast notif={t} onDismiss={() => dismiss(t.toastId)} />
        </div>
      ))}
    </div>
  );
}
