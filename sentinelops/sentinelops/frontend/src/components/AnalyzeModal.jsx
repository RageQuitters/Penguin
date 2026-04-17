/**
 * AnalyzeModal — lets the operator enter custom sensor values before running analysis.
 * Pre-filled with the machine's current seed readings.
 */
import React, { useState } from 'react';

const FIELDS = [
  { key: 'air_temperature',     label: 'Air Temperature',     unit: 'K',   min: 280, max: 320,  step: 0.1 },
  { key: 'process_temperature', label: 'Process Temperature', unit: 'K',   min: 295, max: 325,  step: 0.1 },
  { key: 'rotational_speed',    label: 'Rotational Speed',    unit: 'rpm', min: 0,   max: 3000, step: 1   },
  { key: 'torque',              label: 'Torque',              unit: 'Nm',  min: 0,   max: 100,  step: 0.1 },
  { key: 'tool_wear',           label: 'Tool Wear',           unit: 'min', min: 0,   max: 250,  step: 1   },
];

export default function AnalyzeModal({ defaultReading, machineId, onClose, onSubmit }) {
  const [values, setValues] = useState({ ...defaultReading });

  const handleChange = (key, val) => {
    setValues(prev => ({ ...prev, [key]: parseFloat(val) }));
  };

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      background: 'rgba(7,9,10,0.85)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
      backdropFilter: 'blur(4px)',
    }}>
      <div style={{
        background: 'var(--bg2)',
        border: '1px solid var(--border2)',
        borderRadius: 10,
        width: 440,
        overflow: 'hidden',
        animation: 'fadeSlideIn 0.2s ease both',
        boxShadow: '0 32px 80px rgba(0,0,0,0.6)',
      }}>
        {/* Header */}
        <div style={{
          padding: '14px 20px',
          borderBottom: '1px solid var(--border)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          <div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 11, fontWeight: 500, marginBottom: 2 }}>
              CUSTOM SENSOR READING
            </div>
            <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text3)' }}>
              {machineId} — override values before analysis
            </div>
          </div>
          <button onClick={onClose} style={{
            background: 'none', border: 'none',
            color: 'var(--text3)', cursor: 'pointer', fontSize: 18, lineHeight: 1,
          }}>✕</button>
        </div>

        {/* Fields */}
        <div style={{ padding: '16px 20px', display: 'flex', flexDirection: 'column', gap: 12 }}>
          {FIELDS.map(({ key, label, unit, min, max, step }) => (
            <div key={key} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
              <div style={{ flex: 1 }}>
                <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text3)', marginBottom: 4 }}>{label}</div>
                <input
                  type="range"
                  min={min}
                  max={max}
                  step={step}
                  value={values[key] ?? 0}
                  onChange={e => handleChange(key, e.target.value)}
                  style={{ width: '100%', accentColor: 'var(--accent)' }}
                />
              </div>
              <div style={{ width: 90, textAlign: 'right' }}>
                <input
                  type="number"
                  min={min}
                  max={max}
                  step={step}
                  value={values[key] ?? 0}
                  onChange={e => handleChange(key, e.target.value)}
                  style={{
                    width: '100%',
                    fontFamily: 'var(--mono)',
                    fontSize: 12,
                    background: 'var(--bg3)',
                    border: '1px solid var(--border2)',
                    borderRadius: 4,
                    color: 'var(--text)',
                    padding: '4px 8px',
                    textAlign: 'right',
                  }}
                />
                <div style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--text3)', marginTop: 2 }}>{unit}</div>
              </div>
            </div>
          ))}
        </div>

        {/* Actions */}
        <div style={{
          padding: '12px 20px',
          borderTop: '1px solid var(--border)',
          display: 'flex',
          justifyContent: 'flex-end',
          gap: 8,
        }}>
          <button onClick={onClose} style={{
            fontFamily: 'var(--mono)', fontSize: 10,
            padding: '7px 16px',
            border: '1px solid var(--border2)',
            borderRadius: 5,
            background: 'none', color: 'var(--text2)', cursor: 'pointer',
          }}>
            CANCEL
          </button>
          <button onClick={() => onSubmit(values)} style={{
            fontFamily: 'var(--mono)', fontSize: 10,
            padding: '7px 18px',
            border: '1px solid var(--accent)',
            borderRadius: 5,
            background: 'var(--accent)',
            color: '#07090a',
            cursor: 'pointer',
            fontWeight: 500,
          }}>
            ▶ RUN ANALYSIS
          </button>
        </div>
      </div>
    </div>
  );
}
