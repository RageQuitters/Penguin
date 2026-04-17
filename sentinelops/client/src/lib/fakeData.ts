/**
 * Data Service for SentinelOps
 * Uses Firebase Firestore for persistence, with fallback to defaults
 */

import { getAllMachines as getFirestoreMachines, initializeMachinesData, simulateFault as firebaseSimulateFault, Machine } from './firebaseService';

export type { Machine };

export interface AnalysisResult {
  machine_id: string;
  timestamp: string;
  decision: {
    overall_urgency: 'low' | 'medium' | 'high' | 'critical';
    work_order: string;
    agents_called: string[];
    anomaly: {
      anomaly_score: number;
      baseline_trend: number[];
    };
    fault: {
      faults: Record<string, 0 | 1>;
      fault_probabilities: Record<string, number>;
    };
    predictive: {
      rul_hours: number;
      degradation_rate: number;
    };
    final_status: string;
  };
}

export interface AllMachinesAnalysis {
  timestamp: string;
  summary: string;
  critical_machines: Machine[];
  warning_machines: Machine[];
  normal_machines: Machine[];
  recommended_actions: string[];
  engineers_to_dispatch: {
    machine_id: string;
    reason: string;
    urgency: string;
  }[];
}

// Initialize Firebase data on module load
initializeMachinesData().catch(console.error);

export async function getAllMachines(): Promise<Machine[]> {
  return getFirestoreMachines();
}

export async function getMachineById(machineId: string): Promise<Machine | undefined> {
  const machines = await getFirestoreMachines();
  return machines.find((m) => m.machine_id === machineId);
}

export async function simulateFault(machineId: string): Promise<void> {
  await firebaseSimulateFault(machineId);
}

export async function analyzeAllMachines(): Promise<AllMachinesAnalysis> {
  // Simulate API delay
  await new Promise((resolve) => setTimeout(resolve, 1500));

  const machines = await getFirestoreMachines();
  const critical = machines.filter((m) => m.status === 'Critical');
  const warning = machines.filter((m) => m.status === 'Warning');
  const normal = machines.filter((m) => m.status === 'Normal');

  const engineers_to_dispatch = [];

  // Critical machines get immediate dispatch
  for (const machine of critical) {
    engineers_to_dispatch.push({
      machine_id: machine.machine_id,
      reason: `Critical status - anomaly score ${machine.anomaly_score.toFixed(2)}, RUL ${machine.rul_hours.toFixed(1)}h`,
      urgency: 'IMMEDIATE',
    });
  }

  // Warning machines with low RUL get dispatch
  for (const machine of warning) {
    if (machine.rul_hours < 100) {
      engineers_to_dispatch.push({
        machine_id: machine.machine_id,
        reason: `Warning status with low RUL - ${machine.rul_hours.toFixed(1)}h remaining`,
        urgency: 'HIGH',
      });
    }
  }

  const summary = generateSummaryText(critical, warning, normal, engineers_to_dispatch);

  return {
    timestamp: new Date().toISOString(),
    summary,
    critical_machines: critical,
    warning_machines: warning,
    normal_machines: normal,
    recommended_actions: generateRecommendations(critical, warning),
    engineers_to_dispatch,
  };
}

function generateSummaryText(
  critical: Machine[],
  warning: Machine[],
  normal: Machine[],
  dispatch: any[]
): string {
  let text = `**Fleet Status Report**\n\n`;
  text += `**Summary:** ${normal.length} Normal | ${warning.length} Warning | ${critical.length} Critical\n\n`;

  if (critical.length > 0) {
    text += `**🚨 CRITICAL MACHINES:**\n`;
    for (const m of critical) {
      text += `- **${m.machine_id}**: Anomaly ${m.anomaly_score.toFixed(2)}, RUL ${m.rul_hours.toFixed(1)}h\n`;
    }
    text += `\n`;
  }

  if (warning.length > 0) {
    text += `**⚠️ WARNING MACHINES:**\n`;
    for (const m of warning) {
      text += `- **${m.machine_id}**: Anomaly ${m.anomaly_score.toFixed(2)}, RUL ${m.rul_hours.toFixed(1)}h\n`;
    }
    text += `\n`;
  }

  if (dispatch.length > 0) {
    text += `**👷 ENGINEERS TO DISPATCH:**\n`;
    for (const d of dispatch) {
      text += `- **${d.machine_id}** (${d.urgency}): ${d.reason}\n`;
    }
  } else {
    text += `**✅ No immediate engineer dispatch required.**\n`;
  }

  return text;
}

function generateRecommendations(critical: Machine[], warning: Machine[]): string[] {
  const recommendations: string[] = [];

  if (critical.length > 0) {
    recommendations.push(`IMMEDIATE: Dispatch maintenance team to critical machines: ${critical.map((m) => m.machine_id).join(', ')}`);
  }

  const lowRulWarnings = warning.filter((m) => m.rul_hours < 100);
  if (lowRulWarnings.length > 0) {
    recommendations.push(`URGENT: Schedule tool changes for ${lowRulWarnings.map((m) => m.machine_id).join(', ')} within next 24 hours`);
  }

  const highWearWarnings = warning.filter((m) => m.tool_wear > 40);
  if (highWearWarnings.length > 0) {
    recommendations.push(`Monitor tool wear on ${highWearWarnings.map((m) => m.machine_id).join(', ')} - consider preventive replacement`);
  }

  if (recommendations.length === 0) {
    recommendations.push('Continue normal monitoring - all machines within acceptable parameters');
  }

  return recommendations;
}

export async function generateAIChatResponse(userMessage: string, machines: Machine[]): Promise<string> {
  // Simulate AI processing delay
  await new Promise((resolve) => setTimeout(resolve, 800));

  const lowerMessage = userMessage.toLowerCase();

  // Simple pattern matching for demo purposes
  if (lowerMessage.includes('status') || lowerMessage.includes('how are')) {
    const critical = machines.filter((m) => m.status === 'Critical').length;
    const warning = machines.filter((m) => m.status === 'Warning').length;
    const normal = machines.filter((m) => m.status === 'Normal').length;

    return `Current fleet status: **${normal}** machines operating normally, **${warning}** machines in warning state, and **${critical}** machines critical. Overall fleet health is ${critical > 0 ? 'compromised' : warning > 3 ? 'degraded' : 'good'}.`;
  }

  if (lowerMessage.includes('critical') || lowerMessage.includes('alert')) {
    const criticalMachines = machines.filter((m) => m.status === 'Critical');
    if (criticalMachines.length === 0) {
      return `No critical machines detected at this time. All machines are either normal or in warning state.`;
    }
    return `Critical machines: ${criticalMachines.map((m) => m.machine_id).join(', ')}. These require immediate attention and engineer dispatch.`;
  }

  if (lowerMessage.includes('recommend') || lowerMessage.includes('suggest')) {
    const warnings = machines.filter((m) => m.status === 'Warning');
    if (warnings.length === 0) {
      return `All machines are healthy. Continue routine monitoring and maintenance schedules.`;
    }
    return `I recommend prioritizing maintenance for: ${warnings.map((m) => m.machine_id).join(', ')}. Focus on tool wear and RUL management to prevent critical failures.`;
  }

  if (lowerMessage.includes('analyze')) {
    return `To analyze all machines, click the "Analyze All Machines" button in the dashboard. This will generate a comprehensive fleet report with engineer dispatch recommendations.`;
  }

  // Default response
  return `I'm monitoring the SentinelOps fleet. Ask me about machine status, critical alerts, recommendations, or click "Analyze All Machines" for a comprehensive report.`;
}
