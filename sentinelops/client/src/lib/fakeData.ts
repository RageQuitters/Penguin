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

  // Correct classification:
  //   severe   — at least 1 predicted fault (HDF/OSF/PWF/RNF/TWF === 1)
  //   moderate — no faults, but anomaly_score > 0.6
  //   normal   — neither
  const hasFault = (m: Machine) =>
    [m.HDF, m.OSF, m.PWF, m.RNF, m.TWF].some((v) => v === 1);

  const severe = machines.filter((m) => hasFault(m));
  const moderate = machines.filter((m) => !hasFault(m) && m.anomaly_score > 0.6);
  const normal = machines.filter((m) => !hasFault(m) && m.anomaly_score <= 0.6);

  const engineers_to_dispatch = [];

  // Severe machines get immediate dispatch
  for (const machine of severe) {
    engineers_to_dispatch.push({
      machine_id: machine.machine_id,
      reason: `Severe - predicted fault detected, anomaly score ${machine.anomaly_score.toFixed(2)}, RUL ${machine.rul_hours.toFixed(1)}h`,
      urgency: 'IMMEDIATE',
    });
  }

  // Moderate machines with low RUL get dispatch
  for (const machine of moderate) {
    if (machine.rul_hours < 100) {
      engineers_to_dispatch.push({
        machine_id: machine.machine_id,
        reason: `Moderate - high anomaly score (${machine.anomaly_score.toFixed(2)}) with low RUL ${machine.rul_hours.toFixed(1)}h remaining`,
        urgency: 'HIGH',
      });
    }
  }

  const summary = generateSummaryText(severe, moderate, normal, engineers_to_dispatch);

  return {
    timestamp: new Date().toISOString(),
    summary,
    critical_machines: severe,
    warning_machines: moderate,
    normal_machines: normal,
    recommended_actions: generateRecommendations(severe, moderate),
    engineers_to_dispatch,
  };
}

function generateSummaryText(
  severe: Machine[],
  moderate: Machine[],
  normal: Machine[],
  dispatch: any[]
): string {
  let text = `**Fleet Status Report**\n\n`;
  text += `**Summary:** ${normal.length} Normal | ${moderate.length} Moderate | ${severe.length} Severe\n\n`;

  if (severe.length > 0) {
    text += `**🚨 SEVERE MACHINES (predicted fault):**\n`;
    for (const m of severe) {
      text += `- **${m.machine_id}**: Anomaly ${m.anomaly_score.toFixed(2)}, RUL ${m.rul_hours.toFixed(1)}h\n`;
    }
    text += `\n`;
  }

  if (moderate.length > 0) {
    text += `**⚠️ MODERATE MACHINES (high anomaly score):**\n`;
    for (const m of moderate) {
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

function generateRecommendations(severe: Machine[], moderate: Machine[]): string[] {
  const recommendations: string[] = [];

  if (severe.length > 0) {
    recommendations.push(`IMMEDIATE: Dispatch maintenance team to severe machines: ${severe.map((m) => m.machine_id).join(', ')}`);
  }

  const lowRulModerate = moderate.filter((m) => m.rul_hours < 100);
  if (lowRulModerate.length > 0) {
    recommendations.push(`URGENT: Schedule tool changes for ${lowRulModerate.map((m) => m.machine_id).join(', ')} within next 24 hours`);
  }

  const highWearModerate = moderate.filter((m) => m.tool_wear > 40);
  if (highWearModerate.length > 0) {
    recommendations.push(`Monitor tool wear on ${highWearModerate.map((m) => m.machine_id).join(', ')} - consider preventive replacement`);
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

  // Correct classification helpers
  const hasFault = (m: Machine) => [m.HDF, m.OSF, m.PWF, m.RNF, m.TWF].some((v) => v === 1);
  const severeMachines = machines.filter((m) => hasFault(m));
  const moderateMachines = machines.filter((m) => !hasFault(m) && m.anomaly_score > 0.6);
  const normalMachines = machines.filter((m) => !hasFault(m) && m.anomaly_score <= 0.6);

  // Simple pattern matching for demo purposes
  if (lowerMessage.includes('status') || lowerMessage.includes('how are')) {
    return `Current fleet status: **${normalMachines.length}** machines normal, **${moderateMachines.length}** moderate (high anomaly), and **${severeMachines.length}** severe (predicted fault). Overall fleet health is ${severeMachines.length > 0 ? 'compromised' : moderateMachines.length > 3 ? 'degraded' : 'good'}.`;
  }

  if (lowerMessage.includes('critical') || lowerMessage.includes('severe') || lowerMessage.includes('alert')) {
    if (severeMachines.length === 0) {
      return `No severe machines detected at this time. All machines have no active fault predictions.`;
    }
    return `Severe machines (predicted fault): ${severeMachines.map((m) => m.machine_id).join(', ')}. These require immediate attention and engineer dispatch.`;
  }

  if (lowerMessage.includes('recommend') || lowerMessage.includes('suggest')) {
    if (moderateMachines.length === 0 && severeMachines.length === 0) {
      return `All machines are healthy. Continue routine monitoring and maintenance schedules.`;
    }
    return `I recommend prioritizing maintenance for severe machines: ${severeMachines.map((m) => m.machine_id).join(', ') || 'none'}. Also monitor moderate machines: ${moderateMachines.map((m) => m.machine_id).join(', ') || 'none'}. Focus on tool wear and RUL management.`;
  }

  if (lowerMessage.includes('analyze')) {
    return `To analyze all machines, click the "Analyze All Machines" button in the dashboard. This will generate a comprehensive fleet report with engineer dispatch recommendations.`;
  }

  // Default response
  return `I'm monitoring the SentinelOps fleet. Ask me about machine status, severe alerts, recommendations, or click "Analyze All Machines" for a comprehensive report.`;
}
