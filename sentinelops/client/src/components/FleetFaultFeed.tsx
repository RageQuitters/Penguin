import { useState, useEffect, useCallback } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  AlertTriangle, Clock, RefreshCcw, UserPlus, Inbox,
} from 'lucide-react';
import {
  getAllUnresolvedFaults,
  getOpenAssignments,
  type FaultLog,
  type Assignment,
  type Machine,
  type Engineer,
} from '@/lib/firebaseService';
import AssignEngineerDialog from './AssignEngineerDialog';
import type { FaultCode } from '@shared/faultRouting';

interface FleetFaultFeedProps {
  machines: Machine[];
  engineers: Engineer[];
  /** Bumping this number forces a refresh — useful to call from the parent
   *  whenever a new fault is auto-detected from rolling ticks. */
  refreshKey?: number;
}

function timeAgo(ts: string): string {
  const ms = Date.now() - new Date(ts).getTime();
  const min = Math.floor(ms / 60000);
  if (min < 1) return 'just now';
  if (min < 60) return `${min}m ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24) return `${hr}h ago`;
  return `${Math.floor(hr / 24)}d ago`;
}

const FAULT_COLOR: Record<string, string> = {
  HDF: 'border-l-orange-500',
  OSF: 'border-l-purple-500',
  PWF: 'border-l-red-500',
  RNF: 'border-l-yellow-500',
  TWF: 'border-l-pink-500',
};

export default function FleetFaultFeed({ machines, engineers, refreshKey = 0 }: FleetFaultFeedProps) {
  const [faults, setFaults] = useState<FaultLog[]>([]);
  const [assignments, setAssignments] = useState<Assignment[]>([]);
  const [loading, setLoading] = useState(true);
  const [assignTarget, setAssignTarget] = useState<{ machine: Machine; faultTypes: FaultCode[] } | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    const [f, a] = await Promise.all([getAllUnresolvedFaults(50), getOpenAssignments()]);
    setFaults(f);
    setAssignments(a);
    setLoading(false);
  }, []);

  useEffect(() => { refresh(); }, [refresh, refreshKey]);

  // Auto-refresh every 30s so the feed stays fresh during demo
  useEffect(() => {
    const id = setInterval(refresh, 30_000);
    return () => clearInterval(id);
  }, [refresh]);

  // Find any open assignment for a (machine, fault) pair so we can show "Assigned to X"
  const findAssignment = (machineId: string, faultType: string): Assignment | undefined =>
    assignments.find(
      (a) => a.machine_id === machineId && a.fault_types.includes(faultType),
    );

  const handleAssign = (fault: FaultLog) => {
    const machine = machines.find((m) => m.machine_id === fault.machine_id);
    if (!machine) return;
    setAssignTarget({ machine, faultTypes: [fault.fault_type as FaultCode] });
  };

  // Group by machine for cleaner display
  const groupedByMachine = faults.reduce<Record<string, FaultLog[]>>((acc, f) => {
    (acc[f.machine_id] = acc[f.machine_id] ?? []).push(f);
    return acc;
  }, {});

  return (
    <div className="space-y-3 p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-red-400" />
          <h3 className="text-sm font-semibold">Live Fault Feed</h3>
          <Badge variant="secondary" className="bg-red-500/10 text-red-400 border-red-500/20">
            {faults.length} unresolved
          </Badge>
        </div>
        <Button size="sm" variant="ghost" onClick={refresh} disabled={loading} className="h-7 px-2">
          <RefreshCcw className={`h-3 w-3 ${loading ? 'animate-spin' : ''}`} />
        </Button>
      </div>

      <ScrollArea className="h-[calc(100vh-220px)]">
        {loading && faults.length === 0 ? (
          <div className="flex justify-center py-8">
            <div className="animate-spin h-6 w-6 border-b-2 border-primary rounded-full" />
          </div>
        ) : faults.length === 0 ? (
          <Card className="p-8 text-center border-dashed">
            <Inbox className="h-8 w-8 mx-auto mb-2 text-muted-foreground/50" />
            <p className="text-xs text-muted-foreground">No unresolved faults — fleet is healthy.</p>
            <p className="text-[10px] text-muted-foreground/70 mt-1">
              New faults appear here automatically as machines flip into faulted state.
            </p>
          </Card>
        ) : (
          <div className="space-y-3 pr-2">
            {Object.entries(groupedByMachine).map(([machineId, machineFaults]) => {
              const machine = machines.find((m) => m.machine_id === machineId);
              const allFaultTypes = machineFaults.map((f) => f.fault_type as FaultCode);

              return (
                <Card key={machineId} className="p-3 border-red-500/30 bg-red-500/5">
                  <div className="flex items-center justify-between mb-2 gap-2">
                    <div className="flex items-center gap-2 min-w-0">
                      <span className="font-bold text-sm">{machineId}</span>
                      {machine && (
                        <span className="text-[10px] text-muted-foreground">
                          score {machine.anomaly_score.toFixed(2)} · RUL {machine.rul_hours.toFixed(0)}h
                        </span>
                      )}
                    </div>
                    {machine && (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => setAssignTarget({ machine, faultTypes: allFaultTypes })}
                        className="h-7 px-2 gap-1 text-[11px]"
                      >
                        <UserPlus className="h-3 w-3" />
                        Assign
                      </Button>
                    )}
                  </div>

                  <div className="space-y-1.5">
                    {machineFaults.map((fault) => {
                      const assignment = findAssignment(fault.machine_id, fault.fault_type);
                      return (
                        <div
                          key={fault.id}
                          className={`flex items-center justify-between gap-2 p-2 rounded border-l-2 bg-background/40 ${FAULT_COLOR[fault.fault_type] ?? 'border-l-red-500'}`}
                        >
                          <div className="flex items-center gap-2 min-w-0">
                            <span className="font-mono text-[11px] font-bold">{fault.fault_type}</span>
                            <span className="text-[10px] text-muted-foreground flex items-center gap-1">
                              <Clock className="h-2.5 w-2.5" />
                              {timeAgo(fault.timestamp)}
                            </span>
                          </div>
                          {assignment ? (
                            <Badge
                              variant="secondary"
                              className="bg-blue-500/15 text-blue-300 border-blue-500/30 text-[9px]"
                            >
                              {assignment.status === 'in_progress' ? '🔧' : '👷'} {assignment.engineer_name}
                            </Badge>
                          ) : (
                            <button
                              onClick={() => handleAssign(fault)}
                              className="text-[10px] text-blue-400 hover:underline whitespace-nowrap"
                            >
                              Assign →
                            </button>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </Card>
              );
            })}
          </div>
        )}
      </ScrollArea>

      <AssignEngineerDialog
        open={!!assignTarget}
        onOpenChange={(o) => !o && setAssignTarget(null)}
        machine={assignTarget?.machine ?? null}
        faultTypes={assignTarget?.faultTypes}
        engineers={engineers}
        onAssigned={() => refresh()}
      />
    </div>
  );
}