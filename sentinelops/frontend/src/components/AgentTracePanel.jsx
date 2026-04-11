/**
 * AgentTracePanel — streams live agent trace entries as a chat/timeline.
 * Each entry shows agent name, action type, and explanation.
 * Color-coded by agent; action icons distinguish tool calls, reasoning, decisions.
 */
import React, { useEffect, useRef } from 'react';

const AGENT_COLORS = {
  'Orchestrator':                { color: '#00d4aa', short: 'ORC' },
  'Anomaly Detection Agent':     { color: '#5b9cf6', short: 'ANO' },
  'Fault Classification Agent':  { color: '#f5a623', short: 'FLT' },
  'Predictive Maintenance Agent':{ color: '#b085f5', short: 'PRD' },
};

const ACTION_ICONS = {
  tool_call:  { icon: '⚙', label: 'TOOL' },
  reasoning:  { icon: '◈', label: 'THINK' },
  decision:   { icon: '◉', label: 'DECIDE' },
  handoff:    { icon: '→', label: 'ROUTE' },
};

function TraceEntry({ entry, index }) {
  const agentMeta = AGENT_COLORS[entry.agent] || { color: 'var(--text2)', short: '???' };
  const actionMeta = ACTION_ICONS[entry.action] || { icon: '·', label: entry.action };
  const ts = entry.timestamp ? new Date(entry.timestamp).toLocaleTimeString('en-SG', {
    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false
  }) : '--:--:--';

  return (
    <div style={{
      padding: '10px 14px',
      borderBottom: '1px solid var(--border)',
      animation: 'scanIn 0.2s ease both',
      animationDelay: `${Math.min(index * 0.03, 0.3)}s`,
    }}>
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 5 }}>
        {/* Agent tag */}
        <span style={{
          fontFamily: 'var(--mono)',
          fontSize: 9,
          fontWeight: 500,
          padding: '2px 6px',
          borderRadius: 3,
          background: `${agentMeta.color}18`,
          border: `1px solid ${agentMeta.color}40`,
          color: agentMeta.color,
          letterSpacing: '0.06em',
        }}>
          {agentMeta.short}
        </span>
        {/* Action badge */}
        <span style={{
          fontFamily: 'var(--mono)',
          fontSize: 9,
          color: 'var(--text3)',
          display: 'flex',
          alignItems: 'center',
          gap: 3,
        }}>
          <span style={{ fontSize: 11 }}>{actionMeta.icon}</span>
          {actionMeta.label}
        </span>
        {/* Timestamp */}
        <span style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--text3)', marginLeft: 'auto' }}>
          {ts}
        </span>
      </div>

      {/* Full agent name */}
      <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: agentMeta.color, marginBottom: 4 }}>
        {entry.agent}
      </div>

      {/* Explanation */}
      {entry.explanation && (
        <div style={{ fontSize: 12, color: 'var(--text)', lineHeight: 1.5, marginBottom: 4 }}>
          {entry.explanation}
        </div>
      )}

      {/* Output data preview (collapsed for tool calls) */}
      {entry.output_data && (
        <div style={{
          fontFamily: 'var(--mono)',
          fontSize: 10,
          color: 'var(--text2)',
          background: 'var(--bg3)',
          border: '1px solid var(--border)',
          borderRadius: 4,
          padding: '6px 8px',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-all',
          maxHeight: 80,
          overflow: 'auto',
        }}>
          {JSON.stringify(entry.output_data, null, 2)}
        </div>
      )}
    </div>
  );
}

export default function AgentTracePanel({ entries, connected, onClear }) {
  const bottomRef = useRef(null);

  // Auto-scroll to latest entry
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [entries.length]);

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      background: 'var(--bg2)',
    }}>
      {/* Panel header */}
      <div style={{
        padding: '12px 16px',
        borderBottom: '1px solid var(--border)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text3)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
            AI AGENT TRACE
          </span>
          <span style={{
            fontFamily: 'var(--mono)',
            fontSize: 9,
            padding: '1px 6px',
            borderRadius: 10,
            border: `1px solid ${connected ? 'var(--accent)' : 'var(--danger)'}`,
            color: connected ? 'var(--accent)' : 'var(--danger)',
            background: connected ? 'var(--accent-glow)' : 'var(--danger-bg)',
          }}>
            {connected ? '● LIVE' : '○ OFFLINE'}
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{
            fontFamily: 'var(--mono)',
            fontSize: 9,
            color: 'var(--accent)',
            background: 'var(--accent-glow)',
            border: '1px solid var(--accent)',
            borderRadius: 10,
            padding: '1px 7px',
          }}>
            {entries.length}
          </span>
          {entries.length > 0 && (
            <button onClick={onClear} style={{
              fontFamily: 'var(--mono)',
              fontSize: 9,
              color: 'var(--text3)',
              background: 'none',
              border: '1px solid var(--border2)',
              borderRadius: 3,
              padding: '2px 6px',
              cursor: 'pointer',
            }}>
              CLEAR
            </button>
          )}
        </div>
      </div>

      {/* Agent legend */}
      <div style={{
        padding: '8px 14px',
        borderBottom: '1px solid var(--border)',
        display: 'flex',
        gap: 10,
        flexWrap: 'wrap',
        flexShrink: 0,
      }}>
        {Object.entries(AGENT_COLORS).map(([name, meta]) => (
          <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <div style={{ width: 6, height: 6, borderRadius: '50%', background: meta.color }} />
            <span style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--text3)' }}>{meta.short}</span>
          </div>
        ))}
      </div>

      {/* Entries list */}
      <div style={{ flex: 1, overflowY: 'auto' }}>
        {entries.length === 0 ? (
          <div style={{
            padding: '40px 20px',
            textAlign: 'center',
            fontFamily: 'var(--mono)',
            fontSize: 11,
            color: 'var(--text3)',
          }}>
            <div style={{ marginBottom: 8, fontSize: 20 }}>◈</div>
            Awaiting analysis run.<br />
            Select a machine and click ANALYZE.
          </div>
        ) : (
          entries.map((entry, i) => (
            <TraceEntry key={entry.id || i} entry={entry} index={i} />
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
