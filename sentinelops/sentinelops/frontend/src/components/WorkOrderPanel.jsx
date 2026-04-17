/**
 * WorkOrderPanel — displays the Orchestrator's final synthesised decision.
 * Shows the work order text, urgency, agents called, and enriched fault analysis.
 */
import React from 'react';

const URGENCY_STYLE = {
  low:      { color: 'var(--accent)',  bg: 'var(--ok-bg)',     border: 'var(--accent)',  label: 'LOW'      },
  medium:   { color: 'var(--warn)',    bg: 'var(--warn-bg)',   border: 'var(--warn)',    label: 'MEDIUM'   },
  high:     { color: 'var(--danger)',  bg: 'var(--danger-bg)', border: 'var(--danger)',  label: 'HIGH'     },
  critical: { color: 'var(--danger)',  bg: 'var(--danger-bg)', border: 'var(--danger)',  label: 'CRITICAL' },
};

const AGENT_COLORS = {
  'Anomaly Detection Agent':      '#5b9cf6',
  'Fault Classification Agent':   '#f5a623',
  'Predictive Maintenance Agent': '#b085f5',
};

export default function WorkOrderPanel({ decision }) {
  if (!decision) return null;

  const urgencyStyle = URGENCY_STYLE[decision.overall_urgency] || URGENCY_STYLE.low;

  return (
    <div style={{
      background: 'var(--bg2)',
      border: `1px solid ${urgencyStyle.border}`,
      borderRadius: 8,
      overflow: 'hidden',
      animation: 'fadeSlideIn 0.4s ease both',
      marginBottom: 16,
    }}>
      {/* Header */}
      <div style={{
        padding: '12px 16px',
        background: urgencyStyle.bg,
        borderBottom: `1px solid ${urgencyStyle.border}`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: urgencyStyle.color, letterSpacing: '0.1em' }}>
            ◉ ORCHESTRATOR DECISION
          </span>
        </div>
        <span style={{
          fontFamily: 'var(--mono)',
          fontSize: 9,
          color: urgencyStyle.color,
          border: `1px solid ${urgencyStyle.border}`,
          borderRadius: 3,
          padding: '2px 8px',
          fontWeight: 500,
        }}>
          {urgencyStyle.label} URGENCY
        </span>
      </div>

      {/* Work order text */}
      <div style={{ padding: '14px 16px', borderBottom: '1px solid var(--border)' }}>
        <div style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--text3)', letterSpacing: '0.08em', marginBottom: 8, textTransform: 'uppercase' }}>
          Work Order
        </div>
        <p style={{ fontSize: 13, color: 'var(--text)', lineHeight: 1.7 }}>
          {decision.work_order}
        </p>
      </div>

      {/* Agents called */}
      {decision.agents_called?.length > 0 && (
        <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <span style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--text3)', letterSpacing: '0.08em', textTransform: 'uppercase', marginRight: 4 }}>
            Agents Called
          </span>
          {decision.agents_called.map(name => (
            <span key={name} style={{
              fontFamily: 'var(--mono)',
              fontSize: 9,
              padding: '2px 8px',
              borderRadius: 3,
              border: `1px solid ${AGENT_COLORS[name] || 'var(--border2)'}40`,
              color: AGENT_COLORS[name] || 'var(--text2)',
              background: `${AGENT_COLORS[name] || '#ffffff'}10`,
            }}>
              {name}
            </span>
          ))}
        </div>
      )}

      {/* Agent summaries */}
      <div style={{ padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: 10 }}>
        {decision.anomaly && (
          <AgentSummary
            color="#5b9cf6"
            label="Anomaly Agent"
            lines={[
              `Score: ${decision.anomaly.anomaly_score?.toFixed(4)} — ${decision.anomaly.classification} (${decision.anomaly.urgency})`,
              decision.anomaly.reasoning,
            ]}
          />
        )}
        {decision.fault && (
          <AgentSummary
            color="#f5a623"
            label="Fault Agent"
            lines={[
              `Confirmed faults: ${decision.fault.active_faults?.join(', ') || 'None'} — ${decision.fault.severity}`,
              decision.fault.reasoning,
            ]}
          />
        )}
        {decision.predictive && (
          <AgentSummary
            color="#b085f5"
            label="Predictive Agent"
            lines={[
              `RUL: ${decision.predictive.rul_hours?.toFixed(1)}h @ ${decision.predictive.degradation_rate} min/h wear`,
              decision.predictive.reasoning,
            ]}
          />
        )}
      </div>
    </div>
  );
}

function AgentSummary({ color, label, lines }) {
  return (
    <div style={{
      borderLeft: `2px solid ${color}`,
      paddingLeft: 10,
    }}>
      <div style={{ fontFamily: 'var(--mono)', fontSize: 9, color, letterSpacing: '0.06em', marginBottom: 4, textTransform: 'uppercase' }}>
        {label}
      </div>
      {lines.filter(Boolean).map((line, i) => (
        <div key={i} style={{ fontSize: 12, color: i === 0 ? 'var(--text)' : 'var(--text2)', lineHeight: 1.5 }}>
          {line}
        </div>
      ))}
    </div>
  );
}
