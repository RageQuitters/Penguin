/**
 * SensorGrid — displays 5 live sensor readings with progress bars.
 */
import React from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

const SENSOR_DEFS = [
  { key: 'air_temperature',    label: 'Air Temp',    unit: 'K',   nominal: [295, 305] },
  { key: 'process_temperature',label: 'Proc Temp',   unit: 'K',   nominal: [305, 315] },
  { key: 'rotational_speed',   label: 'Rot Speed',   unit: 'rpm', nominal: [1200, 1800] },
  { key: 'torque',             label: 'Torque',      unit: 'Nm',  nominal: [10, 70] },
  { key: 'tool_wear',          label: 'Tool Wear',   unit: 'min', nominal: [0, 200] },
];

export function SensorGrid({ reading }) {
  if (!reading) return null;

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(5, 1fr)',
      gap: 8,
      marginBottom: 16,
    }}>
      {SENSOR_DEFS.map(({ key, label, unit, nominal }) => {
        const val = reading[key];
        const [lo, hi] = nominal;
        const pct = Math.min(100, Math.max(0, ((val - lo) / (hi - lo)) * 100));
        const isWarn = val > hi * 0.9 || val < lo * 1.05;
        const isDanger = val > hi || val < lo;
        const color = isDanger ? 'var(--danger)' : isWarn ? 'var(--warn)' : 'var(--accent)';

        return (
          <div key={key} style={{
            background: 'var(--bg2)',
            border: '1px solid var(--border)',
            borderRadius: 8,
            padding: '12px',
          }}>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--text3)', letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 6 }}>
              {label}
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 18, fontWeight: 500, color, marginBottom: 2 }}>
              {typeof val === 'number' ? (unit === 'rpm' ? Math.round(val) : val.toFixed(1)) : '--'}
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text3)', marginBottom: 8 }}>{unit}</div>
            <div style={{ height: 3, background: 'var(--border)', borderRadius: 2, overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${pct}%`, background: color, borderRadius: 2, transition: 'width 0.5s' }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

/**
 * FaultGrid — displays TWF/HDF/PWF/OSF/RNF fault status cells.
 */
const FAULT_LABELS = {
  TWF: 'Tool Wear Failure',
  HDF: 'Heat Dissipation',
  PWF: 'Power Failure',
  OSF: 'Overstrain',
  RNF: 'Random Failure',
};

export function FaultGrid({ activeFaults = [] }) {
  const activeSet = new Set(activeFaults);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 8, marginBottom: 16 }}>
      {Object.entries(FAULT_LABELS).map(([code, label]) => {
        const isActive = activeSet.has(code);
        return (
          <div key={code} style={{
            background: isActive ? 'var(--danger-bg)' : 'var(--bg2)',
            border: `1px solid ${isActive ? 'var(--danger)' : 'var(--border)'}`,
            borderRadius: 8,
            padding: '10px 12px',
            textAlign: 'center',
            transition: 'all 0.3s',
          }}>
            <div style={{
              fontFamily: 'var(--mono)',
              fontSize: 13,
              fontWeight: 500,
              color: isActive ? 'var(--danger)' : 'var(--text3)',
              marginBottom: 4,
            }}>
              {code}
            </div>
            <div style={{ fontSize: 10, color: isActive ? 'var(--text)' : 'var(--text3)', marginBottom: 6 }}>{label}</div>
            <div style={{
              fontFamily: 'var(--mono)',
              fontSize: 9,
              padding: '2px 6px',
              borderRadius: 3,
              border: `1px solid ${isActive ? 'var(--danger)' : 'var(--border2)'}`,
              color: isActive ? 'var(--danger)' : 'var(--text3)',
              display: 'inline-block',
            }}>
              {isActive ? 'ACTIVE' : 'CLEAR'}
            </div>
          </div>
        );
      })}
    </div>
  );
}

/**
 * AnomalyChart — 24h anomaly score trend line chart using Recharts.
 */
const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: 'var(--bg2)',
      border: '1px solid var(--border2)',
      borderRadius: 4,
      padding: '6px 10px',
      fontFamily: 'var(--mono)',
      fontSize: 10,
      color: 'var(--text)',
    }}>
      <div style={{ color: 'var(--accent)' }}>Score: {payload[0]?.value?.toFixed(4)}</div>
    </div>
  );
};

export function AnomalyChart({ baseline = [] }) {
  const data = baseline.map((score, i) => ({
    hour: i < 10 ? `0${i}:00` : `${i}:00`,
    score,
  }));

  return (
    <div style={{
      background: 'var(--bg2)',
      border: '1px solid var(--border)',
      borderRadius: 8,
      padding: '16px',
      marginBottom: 16,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 11, fontWeight: 500 }}>
          ANOMALY SCORE — 24H TREND
        </span>
        <div style={{ display: 'flex', gap: 12 }}>
          {[['var(--accent)', 'Score'], ['var(--danger)', 'Threshold']].map(([color, label]) => (
            <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <div style={{ width: 6, height: 6, borderRadius: '50%', background: color }} />
              <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text3)' }}>{label}</span>
            </div>
          ))}
        </div>
      </div>
      <ResponsiveContainer width="100%" height={90}>
        <LineChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
          <XAxis
            dataKey="hour"
            tick={{ fontFamily: 'IBM Plex Mono', fontSize: 9, fill: '#2e4f4a' }}
            tickLine={false}
            axisLine={{ stroke: '#1c2a2e' }}
            interval={3}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fontFamily: 'IBM Plex Mono', fontSize: 9, fill: '#2e4f4a' }}
            tickLine={false}
            axisLine={false}
            tickCount={3}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine y={0.8} stroke="#ff4757" strokeDasharray="4 4" strokeWidth={1} />
          <Line
            type="monotone"
            dataKey="score"
            stroke="#00d4aa"
            strokeWidth={1.5}
            dot={false}
            fill="#00d4aa12"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/**
 * KpiGrid — 4 top-level KPI cards.
 */
export function KpiGrid({ anomalyScore, rul, activeFaults, status }) {
  const scoreColor = anomalyScore >= 0.8 ? 'var(--danger)' : anomalyScore >= 0.4 ? 'var(--warn)' : 'var(--accent)';
  const rulColor = rul == null ? 'var(--text3)' : rul < 4 ? 'var(--danger)' : rul < 12 ? 'var(--warn)' : 'var(--accent)';

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginBottom: 20 }}>
      {[
        { label: 'ANOMALY SCORE', val: anomalyScore?.toFixed(3) ?? '--', sub: 'Threshold: 0.80', color: scoreColor },
        { label: 'RUL ESTIMATE', val: rul != null ? `${rul.toFixed(1)}h` : '--', sub: 'Remaining useful life', color: rulColor },
        { label: 'ACTIVE FAULTS', val: activeFaults?.length ?? '--', sub: activeFaults?.join(', ') || 'None detected', color: activeFaults?.length ? 'var(--danger)' : 'var(--accent)' },
        { label: 'STATUS', val: status ?? '--', sub: 'Current classification', color: status === 'Critical' ? 'var(--danger)' : status === 'Warning' ? 'var(--warn)' : 'var(--accent)' },
      ].map(({ label, val, sub, color }) => (
        <div key={label} style={{
          background: 'var(--bg2)',
          border: '1px solid var(--border)',
          borderRadius: 8,
          padding: 14,
          animation: 'fadeSlideIn 0.3s ease both',
        }}>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text3)', marginBottom: 6, letterSpacing: '0.05em' }}>{label}</div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 24, fontWeight: 500, lineHeight: 1, color }}>{val}</div>
          <div style={{ fontSize: 11, color: 'var(--text3)', marginTop: 4 }}>{sub}</div>
        </div>
      ))}
    </div>
  );
}
