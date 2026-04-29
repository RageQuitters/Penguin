import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts';
import {
  Activity, AlertTriangle, CheckCircle2, Clock, User, Wrench, ChevronDown, ChevronRight,
} from 'lucide-react';
import type { Machine, AnomalyLog, FaultLog, EngineerLog } from '@/lib/firebaseService';
import {
  getAnomalyLogs, getFaultLogs, getEngineerLogs,
} from '@/lib/firebaseService';

interface MachineLogsProps {
  machine: Machine;
}

function fmt(ts: string) {
  try {
    const d = new Date(ts);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch { return ts; }
}

function fmtFull(ts: string) {
  try { return new Date(ts).toLocaleString(); } catch { return ts; }
}

const SCORE_COLOR = (score: number) =>
  score >= 0.6 ? '#f59e0b' : '#22c55e';

const SCORE_LABEL = (score: number) => score >= 0.6 ? 'Warning' : 'Normal';

export default function MachineLogs({ machine }: MachineLogsProps) {
  const [anomalyLogs, setAnomalyLogs] = useState<AnomalyLog[]>([]);
  const [faultLogs, setFaultLogs] = useState<FaultLog[]>([]);
  const [engineerLogs, setEngineerLogs] = useState<EngineerLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedFault, setExpandedFault] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'anomaly' | 'faults' | 'engineers'>('anomaly');

  useEffect(() => {
    setLoading(true);
    Promise.all([
      getAnomalyLogs(machine.machine_id, 24),
      getFaultLogs(machine.machine_id, 20),
      getEngineerLogs(machine.machine_id, 20),
    ]).then(([a, f, e]) => {
      setAnomalyLogs(a);
      setFaultLogs(f);
      setEngineerLogs(e);
      setLoading(false);
    });
  }, [machine.machine_id]);

  // Enrich chart data with rolling index label
  const chartData = anomalyLogs.map((log, i) => ({
    ...log,
    timeLabel: fmt(log.timestamp),
    scoreVal: +log.anomaly_score.toFixed(3),
  }));

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload?.length) return null;
    const d = payload[0].payload as AnomalyLog;
    return (
      <div className="bg-card border border-border rounded-lg p-3 shadow-xl text-xs space-y-1">
        <p className="font-bold text-sm">{fmtFull(d.timestamp)}</p>
        <p>Anomaly Score: <span className="font-bold" style={{ color: SCORE_COLOR(d.anomaly_score) }}>{d.anomaly_score.toFixed(3)}</span></p>
        <p>Classification: <span className="font-semibold">{SCORE_LABEL(d.anomaly_score)}</span></p>
        <p>Air Temp: {d.air_temperature} K</p>
        <p>Process Temp: {d.process_temperature} K</p>
        <p>Rotational: {d.rotational_speed} rpm</p>
        <p>Torque: {d.torque} Nm</p>
        <p>Tool Wear: {d.tool_wear} min</p>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
      </div>
    );
  }

  return (
    <div className="space-y-4 p-4">
      {/* Tab buttons */}
      <div className="flex gap-2 border-b border-border pb-2">
        {([
          { key: 'anomaly', label: '📈 Anomaly History', count: anomalyLogs.length },
          { key: 'faults', label: '⚠️ Fault Log', count: faultLogs.length },
          { key: 'engineers', label: '👷 Engineer Fixes', count: engineerLogs.length },
        ] as const).map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors ${
              activeTab === tab.key
                ? 'bg-primary text-primary-foreground'
                : 'bg-muted text-muted-foreground hover:bg-muted/80'
            }`}
          >
            {tab.label}
            <span className="ml-1 opacity-70">({tab.count})</span>
          </button>
        ))}
      </div>

      {/* ── Anomaly Chart ─────────────────────────────────────────────────────── */}
      {activeTab === 'anomaly' && (
        <div className="space-y-4">
          <Card className="p-4">
            <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
              <Activity className="h-4 w-4 text-primary" />
              Anomaly Score — Last 2 Hours (10-min intervals)
            </h4>
            {chartData.length === 0 ? (
              <p className="text-xs text-muted-foreground text-center py-8">No anomaly data yet. Data populates every 10 minutes.</p>
            ) : (
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="timeLabel" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
                  <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} />
                  <Tooltip content={<CustomTooltip />} />
                  <ReferenceLine y={0.6} stroke="#f59e0b" strokeDasharray="4 2" label={{ value: 'WARNING (0.6)', fill: '#f59e0b', fontSize: 9 }} />
                  <Line
                    type="monotone"
                    dataKey="scoreVal"
                    stroke="#6366f1"
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </Card>

          {/* Raw table */}
          {chartData.length > 0 && (
            <Card className="p-0 overflow-hidden">
              <div className="px-4 py-2 border-b border-border bg-muted/30">
                <p className="text-xs font-semibold text-muted-foreground">RAW READINGS</p>
              </div>
              <ScrollArea className="h-40">
                <table className="w-full text-xs">
                  <thead className="sticky top-0 bg-muted/50">
                    <tr>
                      {['Time', 'Score', 'Classification', 'Air Temp (K)', 'Proc Temp (K)', 'RPM', 'Torque', 'Wear'].map((h) => (
                        <th key={h} className="text-left px-3 py-1.5 font-semibold text-muted-foreground">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {[...chartData].reverse().map((row, i) => (
                      <tr key={i} className="border-t border-border/50 hover:bg-muted/20">
                        <td className="px-3 py-1">{row.timeLabel}</td>
                        <td className="px-3 py-1 font-mono" style={{ color: SCORE_COLOR(row.anomaly_score) }}>{row.scoreVal}</td>
                        <td className="px-3 py-1">
                          <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${row.anomaly_score >= 0.6 ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300' : 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'}`}>{SCORE_LABEL(row.anomaly_score)}</span>
                        </td>
                        <td className="px-3 py-1">{row.air_temperature}</td>
                        <td className="px-3 py-1">{row.process_temperature ?? '—'}</td>
                        <td className="px-3 py-1">{row.rotational_speed}</td>
                        <td className="px-3 py-1">{row.torque}</td>
                        <td className="px-3 py-1">{row.tool_wear}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </ScrollArea>
            </Card>
          )}
        </div>
      )}

      {/* ── Fault Log ─────────────────────────────────────────────────────────── */}
      {activeTab === 'faults' && (
        <div className="space-y-2">
          {faultLogs.length === 0 ? (
            <p className="text-xs text-muted-foreground text-center py-8">No fault history recorded.</p>
          ) : faultLogs.map((fault) => (
            <Card key={fault.id} className={`p-3 border-l-4 ${fault.resolved ? 'border-l-green-500' : 'border-l-red-500'}`}>
              <button
                className="w-full text-left"
                onClick={() => setExpandedFault(expandedFault === fault.id ? null : (fault.id ?? null))}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {fault.resolved
                      ? <CheckCircle2 className="h-4 w-4 text-green-500 flex-shrink-0" />
                      : <AlertTriangle className="h-4 w-4 text-red-500 flex-shrink-0" />}
                    <span className="font-semibold text-sm">{fault.fault_type}</span>
                    <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${fault.resolved ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300' : 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300'}`}>
                      {fault.resolved ? 'RESOLVED' : 'ACTIVE'}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <Clock className="h-3 w-3" />
                    {fmt(fault.timestamp)}
                    {expandedFault === fault.id ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                  </div>
                </div>
              </button>
              {expandedFault === fault.id && (
                <div className="mt-2 pt-2 border-t border-border/50 text-xs space-y-1 text-muted-foreground">
                  <p>Detected: {fmtFull(fault.timestamp)}</p>
                  {fault.resolved_by && <p>Resolved by: <span className="font-semibold text-foreground">{fault.resolved_by}</span></p>}
                  {fault.resolved_at && <p>Resolved at: {fmtFull(fault.resolved_at)}</p>}
                  {fault.notes && <p>Notes: {fault.notes}</p>}
                </div>
              )}
            </Card>
          ))}
        </div>
      )}

      {/* ── Engineer Logs ──────────────────────────────────────────────────────── */}
      {activeTab === 'engineers' && (
        <div className="space-y-2">
          {engineerLogs.length === 0 ? (
            <p className="text-xs text-muted-foreground text-center py-8">No engineer visits recorded.</p>
          ) : engineerLogs.map((log) => (
            <Card key={log.id} className={`p-3 border-l-4 ${log.outcome === 'resolved' ? 'border-l-green-500' : log.outcome === 'partial' ? 'border-l-yellow-500' : 'border-l-red-500'}`}>
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-start gap-2 min-w-0">
                  <Wrench className="h-4 w-4 text-primary flex-shrink-0 mt-0.5" />
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-semibold text-sm">{log.engineer_name}</span>
                      <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${log.outcome === 'resolved' ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300' : log.outcome === 'partial' ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'}`}>
                        {log.outcome.toUpperCase()}
                      </span>
                    </div>
                    <p className="text-xs text-muted-foreground mt-0.5">{log.action}</p>
                    <div className="flex gap-1 mt-1 flex-wrap">
                      {log.fault_types.map((ft) => (
                        <span key={ft} className="text-[10px] px-1 py-0.5 bg-muted rounded font-mono">{ft}</span>
                      ))}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-1 text-xs text-muted-foreground flex-shrink-0">
                  <Clock className="h-3 w-3" />
                  {fmt(log.timestamp)}
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
