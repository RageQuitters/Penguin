import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { AlertTriangle, CheckCircle2, AlertCircle, Activity, Zap, ChevronLeft, ChevronRight } from 'lucide-react';
import AIChat from '@/components/AIChat';
import AllMachinesAnalysisPanel from '@/components/AllMachinesAnalysis';
import { getAllMachines, analyzeAllMachines, simulateFault } from '@/lib/fakeData';
import type { Machine, AllMachinesAnalysis } from '@/lib/fakeData';
import logo from '@/public/logo.png';
import { PanelGroup, Panel, PanelResizeHandle } from 'react-resizable-panels';

const CAROUSEL_SIZE = 6;

export default function Dashboard() {
  const [machines, setMachines] = useState<Machine[]>([]);
  const [selectedMachineId, setSelectedMachineId] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AllMachinesAnalysis | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [simulatingFault, setSimulatingFault] = useState<string | null>(null);
  const [carouselStart, setCarouselStart] = useState(0);

  // Load machines on mount
  useEffect(() => {
    const loadMachines = async () => {
      const data = await getAllMachines();
      setMachines(data);
      setSelectedMachineId(data[0]?.machine_id || null);
      setLoading(false);
    };
    loadMachines();
  }, []);

  const handleAnalyzeAll = async () => {
    setAnalyzing(true);
    try {
      const result = await analyzeAllMachines();
      setAnalysisResult(result);
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleSimulateFault = async (machineId: string) => {
    setSimulatingFault(machineId);
    try {
      await simulateFault(machineId);
      // Refresh machines data
      const data = await getAllMachines();
      setMachines(data);
    } catch (error) {
      console.error('Fault simulation failed:', error);
    } finally {
      setSimulatingFault(null);
    }
  };

  const handleCarouselNext = () => {
    if (carouselStart + CAROUSEL_SIZE < machines.length) {
      setCarouselStart(carouselStart + 1);
    }
  };

  const handleCarouselPrev = () => {
    if (carouselStart > 0) {
      setCarouselStart(carouselStart - 1);
    }
  };

  const visibleMachines = machines.slice(carouselStart, carouselStart + CAROUSEL_SIZE);

  const selectedMachine = machines.find((m) => m.machine_id === selectedMachineId);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'Critical':
        return <AlertTriangle className="h-4 w-4 text-red-600" />;
      case 'Warning':
        return <AlertCircle className="h-4 w-4 text-yellow-600" />;
      default:
        return <CheckCircle2 className="h-4 w-4 text-green-600" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'Critical':
        return 'bg-red-50 border-red-200 dark:bg-red-950 dark:border-red-800';
      case 'Warning':
        return 'bg-yellow-50 border-yellow-200 dark:bg-yellow-950 dark:border-yellow-800';
      default:
        return 'bg-green-50 border-green-200 dark:bg-green-950 dark:border-green-800';
    }
  };

  const getFaultStatus = (machine: Machine) => {
    const faults = [
      { name: 'HDF', value: machine.HDF },
      { name: 'OSF', value: machine.OSF },
      { name: 'PWF', value: machine.PWF },
      { name: 'RNF', value: machine.RNF },
      { name: 'TWF', value: machine.TWF },
    ];
    return faults;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4" />
          <p className="text-sm text-muted-foreground">Loading machines...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-background overflow-hidden">
      {/* Left Sidebar - Machine Carousel (unchanged) */}
      <div className="w-72 border-r border-border flex flex-col">
        {/* Header */}
        <div className="flex-shrink-0 p-4 border-b border-border">
          <h1 className="text-lg font-bold flex items-center gap-2">
            <img src={logo} alt="SentinelOps" className="h-10 w-8" />
            SentinelOps
          </h1>
          <p className="text-xs text-muted-foreground mt-1">Fleet Monitoring</p>
        </div>

        {/* Machine Carousel */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Carousel Buttons */}
          <div className="flex-shrink-0 p-3 border-b border-border flex gap-2">
            <Button
              onClick={handleCarouselPrev}
              disabled={carouselStart === 0}
              size="sm"
              variant="outline"
              className="flex-1"
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <Button
              onClick={handleCarouselNext}
              disabled={carouselStart + CAROUSEL_SIZE >= machines.length}
              size="sm"
              variant="outline"
              className="flex-1"
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>

          {/* Machine List */}
          <ScrollArea className="flex-1">
            <div className="p-4 space-y-2">
              {visibleMachines.map((machine) => (
                <button
                  key={machine.machine_id}
                  onClick={() => setSelectedMachineId(machine.machine_id)}
                  className={`w-full text-left p-3 rounded-lg border transition-colors ${
                    selectedMachineId === machine.machine_id
                      ? 'bg-primary text-primary-foreground border-primary'
                      : `${getStatusColor(machine.status)} border cursor-pointer hover:opacity-80`
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold text-sm">{machine.machine_id}</span>
                    {getStatusIcon(machine.status)}
                  </div>
                  <div className="text-xs opacity-75 mb-2">{machine.machine_type}</div>
                  <div className="grid grid-cols-2 gap-1 text-xs">
                    <div>Score: {machine.anomaly_score.toFixed(2)}</div>
                    <div>RUL: {machine.rul_hours.toFixed(1)}h</div>
                  </div>
                </button>
              ))}
            </div>
          </ScrollArea>
        </div>

        {/* Summary Stats */}
        <div className="flex-shrink-0 p-4 border-t border-border space-y-2">
          <div className="text-xs font-semibold text-muted-foreground mb-2">FLEET STATUS</div>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="p-2 bg-green-50 dark:bg-green-950 rounded text-center">
              <div className="font-bold text-green-600 dark:text-green-400">
                {machines.filter((m) => m.status === 'Normal').length}
              </div>
              <div className="text-muted-foreground">Normal</div>
            </div>
            <div className="p-2 bg-yellow-50 dark:bg-yellow-950 rounded text-center">
              <div className="font-bold text-yellow-600 dark:text-yellow-400">
                {machines.filter((m) => m.status === 'Warning').length}
              </div>
              <div className="text-muted-foreground">Warning</div>
            </div>
            <div className="p-2 bg-red-50 dark:bg-red-950 rounded text-center">
              <div className="font-bold text-red-600 dark:text-red-400">
                {machines.filter((m) => m.status === 'Critical').length}
              </div>
              <div className="text-muted-foreground">Critical</div>
            </div>
          </div>
        </div>
      </div>

      {/* CHANGED: wrapped center + AI Chat in a PanelGroup for resizing */}
      <PanelGroup
        direction="horizontal"
        autoSaveId="sentinelops-layout-v2"
        className="flex-1 min-w-0"
      >
        {/* Center - Machine Details & Analysis */}
        {/* CHANGED: was `flex-1`, now inside a Panel with 75% default, 50% min */}
        <Panel defaultSize={75} minSize={50}>
          <div className="h-full flex flex-col min-w-0">
            {/* Header */}
            <div className="flex-shrink-0 p-4 border-b border-border flex items-center justify-between">
              <div>
                <h2 className="text-xl font-bold">
                  {selectedMachine ? selectedMachine.machine_id : 'No Machine Selected'}
                </h2>
                {selectedMachine && (
                  <p className="text-sm text-muted-foreground mt-1">
                    Status: <span className="font-semibold">{selectedMachine.status}</span>
                  </p>
                )}
              </div>
              <Button onClick={handleAnalyzeAll} disabled={analyzing} size="lg">
                {analyzing ? 'Analyzing...' : 'Analyze All Machines'}
              </Button>
            </div>

            {/* Content Area */}
            <div className="flex-1 overflow-hidden">
              {analysisResult ? (
                <ScrollArea className="h-full">
                  <AllMachinesAnalysisPanel analysis={analysisResult} loading={analyzing} />
                  {/* Show selected machine details below analysis */}
                  {selectedMachine && (
                    <div className="border-t border-border p-6">
                      <h3 className="font-semibold mb-4 text-lg">Selected Machine Details</h3>
                      <Card>
                        <div className="p-6 space-y-6">
                          {/* Fault Simulation Button */}
                          <Card className="bg-blue-50 dark:bg-blue-950 border-blue-200 dark:border-blue-800">
                            <div className="p-4 flex items-center justify-between">
                              <div>
                                <p className="text-sm font-semibold text-blue-900 dark:text-blue-100">
                                  Simulate Fault
                                </p>
                                <p className="text-xs text-blue-800 dark:text-blue-200 mt-1">
                                  Click to simulate a fault for this machine and update Firestore
                                </p>
                              </div>
                              <Button
                                onClick={() => handleSimulateFault(selectedMachine.machine_id)}
                                disabled={simulatingFault === selectedMachine.machine_id}
                                size="sm"
                                className="bg-blue-600 hover:bg-blue-700 text-white"
                              >
                                <Zap className="h-4 w-4 mr-2" />
                                {simulatingFault === selectedMachine.machine_id ? 'Simulating...' : 'Simulate'}
                              </Button>
                            </div>
                          </Card>

                          {/* Machine Details Card */}
                          <Card>
                            <div className="p-6">
                              <h3 className="font-semibold mb-4">Sensor Readings</h3>
                              <div className="grid grid-cols-2 gap-4">
                                <div>
                                  <p className="text-sm text-muted-foreground">Air Temperature</p>
                                  <p className="text-2xl font-bold">{selectedMachine.air_temperature.toFixed(1)} K</p>
                                </div>
                                <div>
                                  <p className="text-sm text-muted-foreground">Process Temperature</p>
                                  <p className="text-2xl font-bold">{selectedMachine.process_temperature.toFixed(1)} K</p>
                                </div>
                                <div>
                                  <p className="text-sm text-muted-foreground">Rotational Speed</p>
                                  <p className="text-2xl font-bold">{selectedMachine.rotational_speed} rpm</p>
                                </div>
                                <div>
                                  <p className="text-sm text-muted-foreground">Torque</p>
                                  <p className="text-2xl font-bold">{selectedMachine.torque.toFixed(1)} Nm</p>
                                </div>
                                <div>
                                  <p className="text-sm text-muted-foreground">Tool Wear</p>
                                  <p className="text-2xl font-bold">{selectedMachine.tool_wear} min</p>
                                </div>
                                <div>
                                  <p className="text-sm text-muted-foreground">Anomaly Score</p>
                                  <p className="text-2xl font-bold">{selectedMachine.anomaly_score.toFixed(3)}</p>
                                </div>
                              </div>
                            </div>
                          </Card>

                          {/* Fault Status */}
                          <Card className="border-orange-200 dark:border-orange-800">
                            <div className="p-6">
                              <h3 className="font-semibold mb-4">Fault Status</h3>
                              <div className="grid grid-cols-2 gap-3">
                                {getFaultStatus(selectedMachine).map((fault) => (
                                  <div key={fault.name} className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-800 rounded border border-slate-200 dark:border-slate-700">
                                    <span className="font-semibold text-sm">{fault.name}</span>
                                    <span
                                      className={`text-xs font-bold px-2 py-1 rounded ${
                                        fault.value === 1
                                          ? 'bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-200'
                                          : 'bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-200'
                                      }`}
                                    >
                                      {fault.value === 1 ? 'FAULT' : 'CLEAR'}
                                    </span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          </Card>

                          {/* Health Metrics */}
                          <Card>
                            <div className="p-6">
                              <h3 className="font-semibold mb-4">Health Metrics</h3>
                              <div className="space-y-3">
                                <div>
                                  <p className="text-sm text-muted-foreground mb-1">RUL (Remaining Useful Life)</p>
                                  <div className="w-full bg-muted rounded-full h-2">
                                    <div
                                      className="bg-primary h-2 rounded-full"
                                      style={{
                                        width: `${Math.min((selectedMachine.rul_hours / 150) * 100, 100)}%`,
                                      }}
                                    />
                                  </div>
                                  <p className="text-sm font-semibold mt-1">{selectedMachine.rul_hours.toFixed(1)} hours remaining</p>
                                </div>
                              </div>
                            </div>
                          </Card>
                        </div>
                      </Card>
                    </div>
                  )}
                </ScrollArea>
              ) : selectedMachine ? (
                <ScrollArea className="h-full">
                  <div className="p-6 space-y-6">
                    {/* Fault Simulation Button */}
                    <Card className="bg-blue-50 dark:bg-blue-950 border-blue-200 dark:border-blue-800">
                      <div className="p-4 flex items-center justify-between">
                        <div>
                          <p className="text-sm font-semibold text-blue-900 dark:text-blue-100">
                            Simulate Fault
                          </p>
                          <p className="text-xs text-blue-800 dark:text-blue-200 mt-1">
                            Click to simulate a fault for this machine and update Firestore
                          </p>
                        </div>
                        <Button
                          onClick={() => handleSimulateFault(selectedMachine.machine_id)}
                          disabled={simulatingFault === selectedMachine.machine_id}
                          size="sm"
                          className="bg-blue-600 hover:bg-blue-700 text-white"
                        >
                          <Zap className="h-4 w-4 mr-2" />
                          {simulatingFault === selectedMachine.machine_id ? 'Simulating...' : 'Simulate'}
                        </Button>
                      </div>
                    </Card>

                    {/* Machine Details Card */}
                    <Card>
                      <div className="p-6">
                        <h3 className="font-semibold mb-4">Sensor Readings</h3>
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <p className="text-sm text-muted-foreground">Air Temperature</p>
                            <p className="text-2xl font-bold">{selectedMachine.air_temperature.toFixed(1)} K</p>
                          </div>
                          <div>
                            <p className="text-sm text-muted-foreground">Process Temperature</p>
                            <p className="text-2xl font-bold">{selectedMachine.process_temperature.toFixed(1)} K</p>
                          </div>
                          <div>
                            <p className="text-sm text-muted-foreground">Rotational Speed</p>
                            <p className="text-2xl font-bold">{selectedMachine.rotational_speed} rpm</p>
                          </div>
                          <div>
                            <p className="text-sm text-muted-foreground">Torque</p>
                            <p className="text-2xl font-bold">{selectedMachine.torque.toFixed(1)} Nm</p>
                          </div>
                          <div>
                            <p className="text-sm text-muted-foreground">Tool Wear</p>
                            <p className="text-2xl font-bold">{selectedMachine.tool_wear} min</p>
                          </div>
                          <div>
                            <p className="text-sm text-muted-foreground">Anomaly Score</p>
                            <p className="text-2xl font-bold">{selectedMachine.anomaly_score.toFixed(3)}</p>
                          </div>
                        </div>
                      </div>
                    </Card>

                    {/* Fault Status */}
                    <Card className="border-orange-200 dark:border-orange-800">
                      <div className="p-6">
                        <h3 className="font-semibold mb-4">Fault Status</h3>
                        <div className="grid grid-cols-2 gap-3">
                          {getFaultStatus(selectedMachine).map((fault) => (
                            <div key={fault.name} className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-800 rounded border border-slate-200 dark:border-slate-700">
                              <span className="font-semibold text-sm">{fault.name}</span>
                              <span
                                className={`text-xs font-bold px-2 py-1 rounded ${
                                  fault.value === 1
                                    ? 'bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-200'
                                    : 'bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-200'
                                }`}
                              >
                                {fault.value === 1 ? 'FAULT' : 'CLEAR'}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </Card>

                    {/* Health Metrics */}
                    <Card>
                      <div className="p-6">
                        <h3 className="font-semibold mb-4">Health Metrics</h3>
                        <div className="space-y-3">
                          <div>
                            <p className="text-sm text-muted-foreground mb-1">RUL (Remaining Useful Life)</p>
                            <div className="w-full bg-muted rounded-full h-2">
                              <div
                                className="bg-primary h-2 rounded-full"
                                style={{
                                  width: `${Math.min((selectedMachine.rul_hours / 150) * 100, 100)}%`,
                                }}
                              />
                            </div>
                            <p className="text-sm font-semibold mt-1">{selectedMachine.rul_hours.toFixed(1)} hours remaining</p>
                          </div>
                        </div>
                      </div>
                    </Card>

                    {/* Tip */}
                    <Card className="bg-blue-50 dark:bg-blue-950 border-blue-200 dark:border-blue-800">
                      <div className="p-4">
                        <p className="text-sm text-blue-900 dark:text-blue-100">
                          💡 <strong>Tip:</strong> Click "Analyze All Machines" to get a comprehensive fleet report with engineer dispatch recommendations.
                        </p>
                      </div>
                    </Card>
                  </div>
                </ScrollArea>
              ) : (
                <div className="flex items-center justify-center h-full">
                  <p className="text-muted-foreground">Select a machine to view details</p>
                </div>
              )}
            </div>
          </div>
        </Panel>

        {/* CHANGED: draggable handle between center and AI chat */}
        <PanelResizeHandle className="w-1.5 bg-border hover:bg-blue-500/50 transition-colors cursor-col-resize flex items-center justify-center group">
          <div className="w-0.5 h-8 bg-muted-foreground/30 group-hover:bg-blue-500 rounded-full" />
        </PanelResizeHandle>

        {/* CHANGED: right sidebar wrapped in a Panel — 25% default, expandable to 50% (double) */}
        <Panel defaultSize={25} minSize={25} maxSize={50}>
          <div className="h-full border-l border-border flex flex-col overflow-hidden">
            <AIChat machines={machines} onAnalyzeAll={handleAnalyzeAll} />
          </div>
        </Panel>
      </PanelGroup>
    </div>
  );
}
