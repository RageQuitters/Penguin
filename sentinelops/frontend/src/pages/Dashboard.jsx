/**
 * Dashboard — main page layout.
 * Three-column layout matching the SentinelOps mockup:
 *   LEFT:   Machine list panel
 *   CENTER: KPIs, anomaly chart, sensors, faults, work order
 *   RIGHT:  Live agent trace panel
 */
import React, { useState, useEffect, useCallback, useRef } from 'react';
import MachineCard from '../components/MachineCard';
import AgentTracePanel from '../components/AgentTracePanel';
import { SensorGrid, FaultGrid, AnomalyChart, KpiGrid } from '../components/DashboardWidgets';
import AnalyzeModal from '../components/AnalyzeModal';
import WorkOrderPanel from '../components/WorkOrderPanel';
import { fetchMachines, analyzeReading } from '../services/api';
import { useWebSocket } from '../hooks/useWebSocket';


// Default sensor readings per machine (seeded for demo)
function seedReading(machineId) {
  const seeds = {
    'U-01': { air_temperature: 298.2, process_temperature: 308.4, rotational_speed: 1451, torque: 38.2, tool_wear: 34 },
    'U-02': { air_temperature: 298.0, process_temperature: 308.2, rotational_speed: 1502, torque: 41.5, tool_wear: 56 },
    'U-03': { air_temperature: 299.1, process_temperature: 309.8, rotational_speed: 1621, torque: 68.4, tool_wear: 198 },
    'U-04': { air_temperature: 297.8, process_temperature: 307.9, rotational_speed: 1398, torque: 22.1, tool_wear: 12 },
    'U-05': { air_temperature: 298.5, process_temperature: 308.9, rotational_speed: 1555, torque: 47.3, tool_wear: 103 },
    'U-06': { air_temperature: 298.1, process_temperature: 308.3, rotational_speed: 1483, torque: 39.8, tool_wear: 29 },
    'U-07': { air_temperature: 298.1, process_temperature: 308.6, rotational_speed: 1543, torque: 42.8, tool_wear: 187 },
    'U-08': { air_temperature: 297.9, process_temperature: 308.0, rotational_speed: 1449, torque: 31.5, tool_wear: 8 },
    'U-09': { air_temperature: 298.3, process_temperature: 308.5, rotational_speed: 1598, torque: 45.1, tool_wear: 77 },
    'U-10': { air_temperature: 298.0, process_temperature: 308.1, rotational_speed: 1552, torque: 36.7, tool_wear: 41 },
    'U-11': { air_temperature: 298.6, process_temperature: 309.1, rotational_speed: 1501, torque: 53.2, tool_wear: 155 },
    'U-12': { air_temperature: 297.7, process_temperature: 307.8, rotational_speed: 1402, torque: 19.4, tool_wear: 22 },
  };
  return seeds[machineId] || { air_temperature: 298.1, process_temperature: 308.5, rotational_speed: 1500, torque: 42.0, tool_wear: 50 };
}

export default function Dashboard() {
  const [machines, setMachines] = useState([]);
  const [selectedId, setSelectedId] = useState('U-07');
  const [loadingMachines, setLoadingMachines] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeResult, setAnalyzeResult] = useState(null); // last API response
  const [showModal, setShowModal] = useState(false);
  const [error, setError] = useState(null);
  const [clock, setClock] = useState('');

  const { entries: traceEntries, connected: wsConnected, clear: clearTrace } = useWebSocket();

  // Clock
  useEffect(() => {
    const tick = () => setClock(new Date().toLocaleTimeString('en-SG', {
      hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
    }));
    tick();
    const t = setInterval(tick, 1000);
    return () => clearInterval(t);
  }, []);

  // Load machines
  useEffect(() => {
    fetchMachines()
      .then(setMachines)
      .catch(() => setError('Failed to load machines. Is the backend running?'))
      .finally(() => setLoadingMachines(false));
    const interval = setInterval(() =>
      fetchMachines().then(setMachines).catch(() => {}), 15000);
    return () => clearInterval(interval);
  }, []);

  const selectedMachine = machines.find(m => m.machine_id === selectedId) || null;

  // Derive display data — prefer analyzeResult if it matches the selected machine
  const displayResult = analyzeResult?.machine_id === selectedId ? analyzeResult : null;
  const anomalyScore = displayResult?.decision?.anomaly?.anomaly_score ?? selectedMachine?.anomaly_score ?? null;
  const rul = displayResult?.decision?.predictive?.rul_hours ?? selectedMachine?.rul_hours ?? null;
  const activeFaults = displayResult?.decision?.fault?.active_faults ?? selectedMachine?.active_faults ?? [];
  const status = displayResult?.decision?.final_status ?? selectedMachine?.status ?? null;
  const baseline = displayResult?.decision?.anomaly?.baseline_trend ?? [];
  const reading = seedReading(selectedId);

  const handleAnalyze = useCallback(async (customReading) => {
    setAnalyzing(true);
    setError(null);
    clearTrace();
    try {
      const result = await analyzeReading(selectedId, customReading || reading);
      setAnalyzeResult(result);
      // Update the machine card score live
      setMachines(prev => prev.map(m =>
        m.machine_id === selectedId
          ? { ...m,
              anomaly_score: result.decision.anomaly?.anomaly_score ?? m.anomaly_score,
              status: result.decision.final_status ?? m.status,
              active_faults: result.decision.fault?.active_faults ?? m.active_faults,
              rul_hours: result.decision.predictive?.rul_hours ?? m.rul_hours,
            }
          : m
      ));
    } catch (e) {
      setError(e.message);
    } finally {
      setAnalyzing(false);
    }
  }, [selectedId, reading, clearTrace]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* ── Header ── */}
      <Header clock={clock} wsConnected={wsConnected} />

      {/* ── Body ── */}
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>

        {/* ── LEFT: Machine list ── */}
        <div style={{
          width: 260,
          borderRight: '1px solid var(--border)',
          background: 'var(--bg2)',
          display: 'flex',
          flexDirection: 'column',
          flexShrink: 0,
          overflow: 'hidden',
        }}>
          <div style={{
            padding: '12px 16px 10px',
            borderBottom: '1px solid var(--border)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexShrink: 0,
          }}>
            <span style={{ fontFamily: 'var(--mono)', fontSize: 11, color: 'white', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
              MACHINES
            </span>
            <span style={{
              fontFamily: 'var(--mono)',
              fontSize: 9,
              color: 'var(--accent)',
              background: 'var(--accent-glow)',
              border: '1px solid var(--accent)',
              borderRadius: 10,
              padding: '1px 7px',
            }}>
              {machines.length}
            </span>
          </div>
          <div style={{ overflowY: 'auto', flex: 1 }}>
            {loadingMachines ? (
              <div style={{ padding: 24, fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text3)', textAlign: 'center' }}>
                Loading…
              </div>
            ) : (
              machines.map(m => (
                <MachineCard
                  key={m.machine_id}
                  machine={m}
                  active={m.machine_id === selectedId}
                  onClick={() => { setSelectedId(m.machine_id); setAnalyzeResult(null); }}
                />
              ))
            )}
          </div>
        </div>

        {/* ── CENTER: Detail panel ── */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '20px 24px', background: 'var(--bg)' }}>

          {error && (
            <div style={{
              marginBottom: 16,
              padding: '10px 14px',
              background: 'var(--danger-bg)',
              border: '1px solid var(--danger)',
              borderRadius: 6,
              fontFamily: 'var(--mono)',
              fontSize: 11,
              color: 'white',
            }}>
              ⚠ {error}
            </div>
          )}

          {/* Section label + analyze button */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
            <div>
              <div style={{ fontFamily: 'var(--mono)', fontSize: 13, color: 'white', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 2 }}>
                JURONG PLANT A · {selectedId}
              </div>
              <div style={{ fontFamily: 'var(--mono)', fontSize: 12, color: 'white' }}>
                {selectedMachine?.machine_type ?? '—'}
              </div>
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              <button
                onClick={() => setShowModal(true)}
                style={{
                  fontFamily: 'var(--mono)',
                  fontSize: 10,
                  padding: '6px 14px',
                  border: '1px solid var(--border2)',
                  borderRadius: 5,
                  background: 'var(--bg3)',
                  color: 'var(--text2)',
                  cursor: 'pointer',
                }}
              >
                CUSTOM READING
              </button>
              <button
                onClick={() => handleAnalyze(null)}
                disabled={analyzing}
                style={{
                  fontFamily: 'var(--mono)',
                  fontSize: 10,
                  padding: '6px 16px',
                  border: '1px solid var(--accent)',
                  borderRadius: 5,
                  background: analyzing ? 'var(--accent-glow)' : 'var(--accent)',
                  color: analyzing ? 'var(--accent)' : '#07090a',
                  cursor: analyzing ? 'not-allowed' : 'pointer',
                  fontWeight: 500,
                  transition: 'all 0.15s',
                }}
              >
                {analyzing ? '◈ ANALYZING…' : '▶ ANALYZE'}
              </button>
            </div>
          </div>

          {/* KPI row */}
          <KpiGrid
            anomalyScore={anomalyScore}
            rul={rul}
            activeFaults={activeFaults}
            status={status}
          />

          {/* Anomaly trend chart */}
          {baseline.length > 0 && <AnomalyChart baseline={baseline} />}

          {/* Live sensor readings */}
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'white', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 10 }}>
            LIVE SENSORS
          </div>
          <SensorGrid reading={reading} />

          {/* Fault classification */}
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'white', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 10 }}>
            FAULT CLASSIFICATION
          </div>
          <FaultGrid activeFaults={activeFaults} />

          {/* Work order */}
          {displayResult?.decision && (
            <WorkOrderPanel decision={displayResult.decision} />
          )}
        </div>

        {/* ── RIGHT: Agent trace ── */}
        <div style={{
          width: 340,
          borderLeft: '1px solid var(--border)',
          flexShrink: 0,
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
        }}>
          <AgentTracePanel
            entries={traceEntries}
            connected={wsConnected}
            onClear={clearTrace}
          />
        </div>
      </div>

      {/* ── Analyze modal (custom sensor input) ── */}
      {showModal && (
        <AnalyzeModal
          defaultReading={reading}
          machineId={selectedId}
          onClose={() => setShowModal(false)}
          onSubmit={(r) => { setShowModal(false); handleAnalyze(r); }}
        />
      )}
    </div>
  );
}

/* ── Header component ── */
function Header({ clock, wsConnected }) {
  return (
    <header style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '0 24px',
      height: 50,
      borderBottom: '1px solid var(--border)',
      background: 'var(--bg2)',
      flexShrink: 0,
    }}>
      {/* Logo */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <div style={{
          width: 26, height: 26,
          border: '1.5px solid var(--accent)',
          borderRadius: 4,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <div style={{
            width: 8, height: 8,
            background: 'var(--accent)',
            borderRadius: 2,
            animation: 'pulse 2s ease-in-out infinite',
          }} />
        </div>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 12, fontWeight: 500, letterSpacing: '0.1em' }}>
          SENTINEL<span style={{ color: 'var(--accent)' }}>OPS</span>
        </span>
        <span style={{
          fontFamily: 'var(--mono)',
          fontSize: 9,
          color: 'var(--text3)',
          border: '1px solid var(--border2)',
          borderRadius: 3,
          padding: '1px 6px',
          marginLeft: 4,
        }}>
          Pangu LLM · Huawei Cloud
        </span>
      </div>

      {/* Right side */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 6,
          fontFamily: 'var(--mono)', fontSize: 11,
          color: 'var(--text2)',
          border: '1px solid var(--border2)',
          borderRadius: 20, padding: '3px 10px',
        }}>
          <div style={{
            width: 5, height: 5, borderRadius: '50%',
            background: wsConnected ? 'var(--accent)' : 'var(--danger)',
            animation: wsConnected ? 'pulse 2s ease-in-out infinite' : 'none',
          }} />
          {wsConnected ? 'SYSTEM ONLINE' : 'CONNECTING…'}
        </div>
        <span style={{
          fontFamily: 'var(--mono)',
          fontSize: 11,
          color: 'white',
          letterSpacing: '0.05em',
        }}>
          Jurong Plant A
        </span>
        <span style={{ fontFamily: 'var(--mono)', fontSize: 13, color: 'white' }}>
          {clock}
        </span>
      </div>
    </header>
  );
}
