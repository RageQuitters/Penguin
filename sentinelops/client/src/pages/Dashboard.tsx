import { useState, useEffect, useRef } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  AlertTriangle,
  CheckCircle2,
  AlertCircle,
  Activity,
  ChevronLeft,
  ChevronRight,
  ClipboardList,
  Send,
} from "lucide-react";
import AIChat from "@/components/AIChat";
import AllMachinesAnalysisPanel from "@/components/AllMachinesAnalysis";
import MachineLogs from "@/components/MachineLogs";
import { getAllMachines, analyzeAllMachines } from "@/lib/fakeData";
import type { Machine, AllMachinesAnalysis } from "@/lib/fakeData";
import {
  updateMachineMLResults,
  seedRollingData,
  seedEngineerAndFaultLogs,
  addAnomalyLog,
} from "@/lib/firebaseService";
import logo from "@/public/logo.png";
import { PanelGroup, Panel, PanelResizeHandle } from "react-resizable-panels";
import { toast } from "sonner";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";
const CAROUSEL_SIZE = 6;

// Machine detail view tabs
type DetailTab = "sensors" | "faults" | "health" | "logs";

export default function Dashboard() {
  const [machines, setMachines] = useState<Machine[]>([]);
  const [selectedMachineId, setSelectedMachineId] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AllMachinesAnalysis | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [carouselStart, setCarouselStart] = useState(0);
  const [detailTab, setDetailTab] = useState<DetailTab>("sensors");
  const [sendingTelegram, setSendingTelegram] = useState(false);
  const rollingTickerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Load machines on mount
  useEffect(() => {
    const loadMachines = async () => {
      let data = await getAllMachines();

      try {
        const res = await fetch(`${API_BASE}/api/predict-all`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ machines: data }),
        });
        if (res.ok) {
          const { results } = await res.json();
          data = data.map((m) => {
            const ml = results.find((r: any) => r.machine_id === m.machine_id);
            if (!ml) return m;
            return {
              ...m,
              anomaly_score: ml.anomaly_score,
              HDF: ml.failure_vector.HDF ?? m.HDF,
              OSF: ml.failure_vector.OSF ?? m.OSF,
              PWF: ml.failure_vector.PWF ?? m.PWF,
              RNF: ml.failure_vector.RNF ?? m.RNF,
              TWF: ml.failure_vector.TWF ?? m.TWF,
              status: (() => {
                const faults = [ml.failure_vector.HDF, ml.failure_vector.OSF, ml.failure_vector.PWF, ml.failure_vector.RNF, ml.failure_vector.TWF];
                if (faults.some((v) => v === 1)) return "Severe";
                if (ml.anomaly_score > 0.6) return "Moderate";
                return "Normal";
              })(),
            };
          });
          results.forEach((ml: any) => {
            updateMachineMLResults(ml.machine_id, { anomaly_score: ml.anomaly_score, failure_vector: ml.failure_vector, decision: ml.decision }).catch(() => {});
          });
        }
      } catch {
        console.warn("[ML] sidecar unavailable, using stored scores");
      }

      setMachines(data);
      setSelectedMachineId(data[0]?.machine_id || null);
      setLoading(false);

      // Seed rolling & engineer data if empty (runs only once per fresh DB)
      seedRollingData(data).catch(() => {});
      seedEngineerAndFaultLogs(data).catch(() => {});

      // Start client-side rolling ticker (every 10 min, writes a new anomaly log)
      rollingTickerRef.current = setInterval(async () => {
        const currentMachines = await getAllMachines();
        for (const m of currentMachines) {
          const rawScore = m.anomaly_score + (Math.random() - 0.5) * 0.06;
          const score = +Math.max(0, Math.min(1, rawScore)).toFixed(3);
          await addAnomalyLog({
            machine_id: m.machine_id,
            timestamp: new Date().toISOString(),
            anomaly_score: score,
            air_temperature: m.air_temperature,
            process_temperature: m.process_temperature,
            rotational_speed: m.rotational_speed,
            torque: m.torque,
            tool_wear: m.tool_wear,
            decision: score > 0.7 ? 'FAILURE' : score > 0.4 ? 'WARNING' : 'NORMAL',
          }).catch(() => {});
        }
      }, 10 * 60 * 1000);
    };
    loadMachines();

    return () => {
      if (rollingTickerRef.current) clearInterval(rollingTickerRef.current);
    };
  }, []);

  const handleAnalyzeAll = async () => {
    setAnalyzing(true);
    try {
      const result = await analyzeAllMachines();
      setAnalysisResult(result);
      setDetailTab("sensors");

      // After full analysis, trigger a Telegram notification for critical machines
      const criticals = result.engineers_to_dispatch.filter((d) => d.urgency === "IMMEDIATE");
      if (criticals.length > 0) {
        const msg = `📊 *Fleet Analysis Complete*\n\n🚨 ${criticals.length} machine(s) require IMMEDIATE dispatch:\n${criticals.map((d) => `• *${d.machine_id}*: ${d.reason}`).join("\n")}`;
        fetch(`${API_BASE}/api/telegram/notify`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: msg }),
        }).catch(() => {});
      }
    } catch (error) {
      console.error("Analysis failed:", error);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleSendTelegramUpdate = async () => {
    if (!selectedMachine) return;
    setSendingTelegram(true);
    try {
      const c = getMachineColor(selectedMachine);
      const faults = getFaultStatus(selectedMachine).filter((f) => f.value === 1).map((f) => f.name);
      const msg = `📌 *Machine Status Update*\n\n*Machine:* ${selectedMachine.machine_id} (${selectedMachine.machine_type})\n*Status:* ${c.label}\n*Anomaly Score:* ${selectedMachine.anomaly_score.toFixed(3)}\n*RUL:* ${selectedMachine.rul_hours.toFixed(1)}h\n*Active Faults:* ${faults.length > 0 ? faults.join(", ") : "None"}\n*Tool Wear:* ${selectedMachine.tool_wear} min`;
      const res = await fetch(`${API_BASE}/api/telegram/notify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg }),
      });
      if (res.ok) toast.success("Telegram notification sent to engineers!");
      else toast.error("Failed to send Telegram notification");
    } catch {
      toast.error("Telegram service unavailable");
    } finally {
      setSendingTelegram(false);
    }
  };

  const handleCarouselNext = () => { if (carouselStart + CAROUSEL_SIZE < machines.length) setCarouselStart(carouselStart + 1); };
  const handleCarouselPrev = () => { if (carouselStart > 0) setCarouselStart(carouselStart - 1); };

  const visibleMachines = machines.slice(carouselStart, carouselStart + CAROUSEL_SIZE);
  const selectedMachine = machines.find((m) => m.machine_id === selectedMachineId);

  const getMachineColor = (machine: Machine) => {
    const hasFault = [machine.HDF, machine.OSF, machine.PWF, machine.RNF, machine.TWF].some((v) => v === 1);
    if (hasFault) return { card: "bg-red-50 border-red-300 dark:bg-red-950 dark:border-red-700", dot: "bg-red-500", label: "Severe" };
    if (machine.anomaly_score > 0.6) return { card: "bg-yellow-50 border-yellow-300 dark:bg-yellow-950 dark:border-yellow-700", dot: "bg-yellow-500", label: "Moderate" };
    return { card: "bg-green-50 border-green-300 dark:bg-green-950 dark:border-green-700", dot: "bg-green-500", label: "Normal" };
  };

  const getStatusIcon = (machine: Machine) => {
    const { label } = getMachineColor(machine);
    if (label === "Severe") return <AlertTriangle className="h-4 w-4 text-red-600" />;
    if (label === "Moderate") return <AlertCircle className="h-4 w-4 text-yellow-600" />;
    return <CheckCircle2 className="h-4 w-4 text-green-600" />;
  };

  const getFaultStatus = (machine: Machine) => [
    { name: "HDF", value: machine.HDF },
    { name: "OSF", value: machine.OSF },
    { name: "PWF", value: machine.PWF },
    { name: "RNF", value: machine.RNF },
    { name: "TWF", value: machine.TWF },
  ];

  const normalCount = machines.filter((m) => getMachineColor(m).label === "Normal").length;
  const moderateCount = machines.filter((m) => getMachineColor(m).label === "Moderate").length;
  const severeCount = machines.filter((m) => getMachineColor(m).label === "Severe").length;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4" />
          <p className="text-sm text-muted-foreground">Loading machines from Firebase…</p>
          <p className="text-xs text-muted-foreground mt-1">Running ML scoring on all machines</p>
        </div>
      </div>
    );
  }

  // Machine detail inner tabs
  const DETAIL_TABS: { key: DetailTab; label: string }[] = [
    { key: "sensors", label: "📡 Sensors" },
    { key: "faults", label: "⚠️ Faults" },
    { key: "health", label: "❤️ Health" },
    { key: "logs", label: "📋 Logs" },
  ];

  return (
    <div className="flex h-screen bg-background overflow-hidden">
      {/* Left Sidebar */}
      <div className="w-72 border-r border-border flex flex-col">
        <div className="flex-shrink-0 p-4 border-b border-border">
          <h1 className="text-lg font-bold flex items-center gap-2">
            <img src={logo} alt="SentinelOps" className="h-10 w-8" />
            SentinelOps
          </h1>
          <p className="text-xs text-muted-foreground mt-1">Fleet Monitoring</p>
        </div>

        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-shrink-0 p-3 border-b border-border flex gap-2">
            <Button onClick={handleCarouselPrev} disabled={carouselStart === 0} size="sm" variant="outline" className="flex-1">
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <Button onClick={handleCarouselNext} disabled={carouselStart + CAROUSEL_SIZE >= machines.length} size="sm" variant="outline" className="flex-1">
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>

          <ScrollArea className="flex-1">
            <div className="p-4 space-y-2">
              {visibleMachines.map((machine) => {
                const colors = getMachineColor(machine);
                return (
                  <button
                    key={machine.machine_id}
                    onClick={() => { setSelectedMachineId(machine.machine_id); setAnalysisResult(null); setDetailTab("sensors"); }}
                    className={`w-full text-left p-3 rounded-lg border transition-colors ${selectedMachineId === machine.machine_id ? "bg-primary text-primary-foreground border-primary" : `${colors.card} cursor-pointer hover:opacity-80`}`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-semibold text-sm">{machine.machine_id}</span>
                      {selectedMachineId === machine.machine_id ? <Activity className="h-4 w-4" /> : getStatusIcon(machine)}
                    </div>
                    <div className="text-xs opacity-75 mb-2">{machine.machine_type}</div>
                    <div className="grid grid-cols-2 gap-1 text-xs">
                      <div>Score: {machine.anomaly_score.toFixed(2)}</div>
                      <div>RUL: {machine.rul_hours.toFixed(1)}h</div>
                    </div>
                  </button>
                );
              })}
            </div>
          </ScrollArea>
        </div>

        <div className="flex-shrink-0 p-4 border-t border-border space-y-2">
          <div className="text-xs font-semibold text-muted-foreground mb-2">FLEET STATUS</div>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="p-2 bg-green-50 dark:bg-green-950 rounded text-center">
              <div className="font-bold text-green-600 dark:text-green-400">{normalCount}</div>
              <div className="text-muted-foreground">Normal</div>
            </div>
            <div className="p-2 bg-yellow-50 dark:bg-yellow-950 rounded text-center">
              <div className="font-bold text-yellow-600 dark:text-yellow-400">{moderateCount}</div>
              <div className="text-muted-foreground">Moderate</div>
            </div>
            <div className="p-2 bg-red-50 dark:bg-red-950 rounded text-center">
              <div className="font-bold text-red-600 dark:text-red-400">{severeCount}</div>
              <div className="text-muted-foreground">Severe</div>
            </div>
          </div>
        </div>
      </div>

      <PanelGroup direction="horizontal" autoSaveId="sentinelops-layout-v3" className="flex-1 min-w-0">
        {/* Center Panel */}
        <Panel defaultSize={75} minSize={50}>
          <div className="h-full flex flex-col min-w-0">
            {/* Header */}
            <div className="flex-shrink-0 p-4 border-b border-border flex items-center justify-between gap-2 flex-wrap">
              <div>
                <h2 className="text-xl font-bold">
                  {selectedMachine ? selectedMachine.machine_id : "No Machine Selected"}
                </h2>
                {selectedMachine && (
                  <p className="text-sm text-muted-foreground mt-1">
                    Status: <span className="font-semibold">{getMachineColor(selectedMachine).label}</span>
                  </p>
                )}
              </div>
              <div className="flex items-center gap-2">
                {selectedMachine && (
                  <Button
                    onClick={handleSendTelegramUpdate}
                    disabled={sendingTelegram}
                    size="sm"
                    variant="outline"
                    className="gap-1"
                  >
                    <Send className="h-4 w-4" />
                    {sendingTelegram ? "Sending…" : "Notify Engineers"}
                  </Button>
                )}
                <Button onClick={handleAnalyzeAll} disabled={analyzing} size="lg">
                  {analyzing ? "Analyzing…" : "Analyze All Machines"}
                </Button>
              </div>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-hidden">
              {analysisResult ? (
                <ScrollArea className="h-full">
                  <AllMachinesAnalysisPanel analysis={analysisResult} loading={analyzing} />
                  {selectedMachine && (
                    <div className="border-t border-border p-6">
                      <h3 className="font-semibold mb-4 text-lg">Selected Machine Details</h3>
                      {renderMachineDetail(selectedMachine, detailTab, setDetailTab, DETAIL_TABS, getFaultStatus)}
                    </div>
                  )}
                </ScrollArea>
              ) : selectedMachine ? (
                <div className="h-full flex flex-col">
                  {/* Detail tabs */}
                  <div className="flex-shrink-0 flex gap-1 px-4 pt-3 border-b border-border pb-0">
                    {DETAIL_TABS.map((tab) => (
                      <button
                        key={tab.key}
                        onClick={() => setDetailTab(tab.key)}
                        className={`px-3 py-2 text-xs font-semibold rounded-t-md transition-colors border-b-2 ${detailTab === tab.key ? "border-primary text-primary" : "border-transparent text-muted-foreground hover:text-foreground"}`}
                      >
                        {tab.label}
                      </button>
                    ))}
                  </div>
                  <ScrollArea className="flex-1">
                    <div className="p-6">
                      {renderMachineDetail(selectedMachine, detailTab, setDetailTab, DETAIL_TABS, getFaultStatus)}
                    </div>
                  </ScrollArea>
                </div>
              ) : (
                <div className="flex items-center justify-center h-full">
                  <p className="text-muted-foreground">Select a machine to view details</p>
                </div>
              )}
            </div>
          </div>
        </Panel>

        <PanelResizeHandle className="w-1.5 bg-border hover:bg-blue-500/50 transition-colors cursor-col-resize flex items-center justify-center group">
          <div className="w-0.5 h-8 bg-muted-foreground/30 group-hover:bg-blue-500 rounded-full" />
        </PanelResizeHandle>

        <Panel defaultSize={25} minSize={25} maxSize={50}>
          <div className="h-full border-l border-border flex flex-col overflow-hidden">
            <AIChat machines={machines} onAnalyzeAll={handleAnalyzeAll} />
          </div>
        </Panel>
      </PanelGroup>
    </div>
  );
}

// ─── Machine detail renderer ──────────────────────────────────────────────────

function renderMachineDetail(
  machine: Machine,
  tab: string,
  setTab: (t: any) => void,
  tabs: { key: string; label: string }[],
  getFaultStatus: (m: Machine) => { name: string; value: number }[]
) {
  if (tab === "sensors") {
    return (
      <Card>
        <div className="p-6">
          <h3 className="font-semibold mb-4">Sensor Readings</h3>
          <div className="grid grid-cols-2 gap-4">
            {[
              { label: "Air Temperature", value: `${machine.air_temperature.toFixed(1)} K` },
              { label: "Process Temperature", value: `${machine.process_temperature.toFixed(1)} K` },
              { label: "Rotational Speed", value: `${machine.rotational_speed} rpm` },
              { label: "Torque", value: `${machine.torque.toFixed(1)} Nm` },
              { label: "Tool Wear", value: `${machine.tool_wear} min` },
              { label: "Anomaly Score", value: machine.anomaly_score.toFixed(3) },
            ].map(({ label, value }) => (
              <div key={label}>
                <p className="text-sm text-muted-foreground">{label}</p>
                <p className="text-2xl font-bold">{value}</p>
              </div>
            ))}
          </div>
        </div>
      </Card>
    );
  }

  if (tab === "faults") {
    return (
      <Card className="border-orange-200 dark:border-orange-800">
        <div className="p-6">
          <h3 className="font-semibold mb-4">Fault Status</h3>
          <div className="grid grid-cols-2 gap-3">
            {getFaultStatus(machine).map((fault) => (
              <div key={fault.name} className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-800 rounded border border-slate-200 dark:border-slate-700">
                <span className="font-semibold text-sm">{fault.name}</span>
                <span className={`text-xs font-bold px-2 py-1 rounded ${fault.value === 1 ? "bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-200" : "bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-200"}`}>
                  {fault.value === 1 ? "FAULT" : "CLEAR"}
                </span>
              </div>
            ))}
          </div>
        </div>
      </Card>
    );
  }

  if (tab === "health") {
    return (
      <Card>
        <div className="p-6">
          <h3 className="font-semibold mb-4">Health Metrics</h3>
          <div className="space-y-3">
            <div>
              <p className="text-sm text-muted-foreground mb-1">RUL (Remaining Useful Life)</p>
              <div className="w-full bg-muted rounded-full h-2">
                <div className="bg-primary h-2 rounded-full" style={{ width: `${Math.min((machine.rul_hours / 150) * 100, 100)}%` }} />
              </div>
              <p className="text-sm font-semibold mt-1">{machine.rul_hours.toFixed(1)} hours remaining</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground mb-1">Tool Wear Progress</p>
              <div className="w-full bg-muted rounded-full h-2">
                <div className="bg-orange-500 h-2 rounded-full" style={{ width: `${Math.min((machine.tool_wear / 240) * 100, 100)}%` }} />
              </div>
              <p className="text-sm font-semibold mt-1">{machine.tool_wear} / 240 min</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground mb-1">Anomaly Score</p>
              <div className="w-full bg-muted rounded-full h-2">
                <div
                  className={`h-2 rounded-full ${machine.anomaly_score > 0.7 ? "bg-red-500" : machine.anomaly_score > 0.4 ? "bg-yellow-500" : "bg-green-500"}`}
                  style={{ width: `${machine.anomaly_score * 100}%` }}
                />
              </div>
              <p className="text-sm font-semibold mt-1">{machine.anomaly_score.toFixed(3)}</p>
            </div>
          </div>
        </div>
      </Card>
    );
  }

  if (tab === "logs") {
    return <MachineLogs machine={machine} />;
  }

  return null;
}
