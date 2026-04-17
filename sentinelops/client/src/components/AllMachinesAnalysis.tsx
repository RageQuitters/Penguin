import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { AlertTriangle, CheckCircle2, AlertCircle } from 'lucide-react';
import type { AllMachinesAnalysis } from '@/lib/fakeData';

interface AllMachinesAnalysisProps {
  analysis: AllMachinesAnalysis;
  loading?: boolean;
}

export default function AllMachinesAnalysisPanel({
  analysis,
  loading = false,
}: AllMachinesAnalysisProps) {
  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4" />
          <p className="text-sm text-muted-foreground">Analyzing all machines...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6">
      {/* Header Summary */}
      <div className="grid grid-cols-3 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Normal
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-green-500 dark:text-green-400">
              {analysis.normal_machines.length}
            </div>
            <p className="text-xs text-muted-foreground mt-1">machines operating normally</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Warning
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-yellow-500 dark:text-yellow-400">
              {analysis.warning_machines.length}
            </div>
            <p className="text-xs text-muted-foreground mt-1">machines in warning state</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Critical
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-red-500 dark:text-red-400">
              {analysis.critical_machines.length}
            </div>
            <p className="text-xs text-muted-foreground mt-1">machines critical</p>
          </CardContent>
        </Card>
      </div>

      {/* Engineers to Dispatch */}
      {analysis.engineers_to_dispatch.length > 0 && (
        <Card className="border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-950">
          <CardHeader>
            <CardTitle className="text-sm flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-red-600 dark:text-red-400" />
              Engineers to Dispatch
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {analysis.engineers_to_dispatch.map((dispatch) => (
                <div
                  key={dispatch.machine_id}
                  className="flex items-start justify-between p-3 bg-white dark:bg-slate-800 rounded border border-red-200 dark:border-red-800"
                >
                  <div className="flex-1">
                    <div className="font-semibold text-sm">{dispatch.machine_id}</div>
                    <p className="text-xs text-muted-foreground mt-1">{dispatch.reason}</p>
                  </div>
                  <Badge
                    variant={
                      dispatch.urgency === 'IMMEDIATE'
                        ? 'destructive'
                        : dispatch.urgency === 'HIGH'
                          ? 'secondary'
                          : 'outline'
                    }
                    className="ml-2 flex-shrink-0"
                  >
                    {dispatch.urgency}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Recommended Actions */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Recommended Actions</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="space-y-2">
            {analysis.recommended_actions.map((action, idx) => (
              <li key={idx} className="flex items-start gap-2 text-sm">
                <CheckCircle2 className="h-4 w-4 text-green-500 dark:text-green-400 mt-0.5 flex-shrink-0" />
                <span>{action}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>

      {/* Critical Machines */}
      {analysis.critical_machines.length > 0 && (
        <Card className="border-red-200 dark:border-red-800">
          <CardHeader>
            <CardTitle className="text-sm flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-red-600 dark:text-red-400" />
              Critical Machines
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {analysis.critical_machines.map((machine) => (
                <div
                  key={machine.machine_id}
                  className="p-3 bg-red-50 dark:bg-red-950 rounded border border-red-200 dark:border-red-800 text-sm"
                >
                  <div className="font-semibold">{machine.machine_id}</div>
                  <div className="grid grid-cols-2 gap-2 mt-2 text-xs text-muted-foreground">
                    <div>Anomaly: {machine.anomaly_score.toFixed(2)}</div>
                    <div>RUL: {machine.rul_hours.toFixed(1)}h</div>
                    <div>Tool Wear: {machine.tool_wear} min</div>
                    <div>Torque: {machine.torque.toFixed(1)} Nm</div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Warning Machines */}
      {analysis.warning_machines.length > 0 && (
        <Card className="border-yellow-200 dark:border-yellow-800">
          <CardHeader>
            <CardTitle className="text-sm flex items-center gap-2">
              <AlertCircle className="h-4 w-4 text-yellow-600 dark:text-yellow-400" />
              Warning Machines
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {analysis.warning_machines.map((machine) => (
                <div
                  key={machine.machine_id}
                  className="p-3 bg-yellow-50 dark:bg-yellow-950 rounded border border-yellow-200 dark:border-yellow-800 text-sm"
                >
                  <div className="font-semibold">{machine.machine_id}</div>
                  <div className="grid grid-cols-2 gap-2 mt-2 text-xs text-muted-foreground">
                    <div>Anomaly: {machine.anomaly_score.toFixed(2)}</div>
                    <div>RUL: {machine.rul_hours.toFixed(1)}h</div>
                    <div>Tool Wear: {machine.tool_wear} min</div>
                    <div>Torque: {machine.torque.toFixed(1)} Nm</div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Normal Machines */}
      {analysis.normal_machines.length > 0 && (
        <Card className="border-green-200 dark:border-green-800">
          <CardHeader>
            <CardTitle className="text-sm flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-green-600 dark:text-green-400" />
              Normal Machines
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-2">
              {analysis.normal_machines.map((machine) => (
                <div
                  key={machine.machine_id}
                  className="p-2 bg-green-50 dark:bg-green-950 rounded border border-green-200 dark:border-green-800 text-sm"
                >
                  <div className="font-semibold">{machine.machine_id}</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Score: {machine.anomaly_score.toFixed(2)}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Timestamp */}
      <div className="text-xs text-muted-foreground text-center pt-4 border-t">
        Analysis generated: {new Date(analysis.timestamp).toLocaleString()}
      </div>
    </div>
  );
}
