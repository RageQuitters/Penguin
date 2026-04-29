import { initializeApp } from 'firebase/app';
import {
  getFirestore,
  collection,
  getDocs,
  doc,
  updateDoc,
  setDoc,
  addDoc,
  query,
  where,
  orderBy,
  limit,
} from 'firebase/firestore';

const firebaseConfig = {
  apiKey: 'AIzaSyBOphhLQ_LKWv9BEmAD7rfwMerWmCZtZ8U',
  authDomain: 'penguin-a7200.firebaseapp.com',
  projectId: 'penguin-a7200',
  storageBucket: 'penguin-a7200.firebasestorage.app',
  messagingSenderId: '525126016743',
  appId: '1:525126016743:web:e7a469b130ba21d0f44c42',
};

const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

export interface Machine {
  machine_id: string;
  machine_type: string;
  air_temperature: number;
  process_temperature: number;
  rotational_speed: number;
  torque: number;
  tool_wear: number;
  anomaly_score: number;
  rul_hours: number;
  HDF: 0 | 1;
  OSF: 0 | 1;
  PWF: 0 | 1;
  RNF: 0 | 1;
  TWF: 0 | 1;
  status: 'Normal' | 'Warning' | 'Critical';
}

export interface AnomalyLog {
  id?: string;
  machine_id: string;
  timestamp: string;
  anomaly_score: number;
  air_temperature: number;
  process_temperature: number;
  rotational_speed: number;
  torque: number;
  tool_wear: number;
  decision: 'NORMAL' | 'WARNING' | 'FAILURE';
}

export interface FaultLog {
  id?: string;
  machine_id: string;
  timestamp: string;
  fault_type: string;
  resolved: boolean;
  resolved_by?: string;
  resolved_at?: string;
  notes?: string;
}

export interface EngineerLog {
  id?: string;
  machine_id: string;
  engineer_name: string;
  action: string;
  timestamp: string;
  fault_types: string[];
  outcome: 'resolved' | 'partial' | 'escalated';
}

export const DEFAULT_MACHINES: Machine[] = [
  { machine_id: 'U-01', machine_type: 'Universal', air_temperature: 298.1, process_temperature: 308.6, rotational_speed: 1551, torque: 42.8, tool_wear: 0, anomaly_score: 0.12, rul_hours: 142.0, HDF: 0, OSF: 0, PWF: 0, RNF: 0, TWF: 0, status: 'Normal' },
  { machine_id: 'U-02', machine_type: 'Universal', air_temperature: 298.2, process_temperature: 308.7, rotational_speed: 1408, torque: 46.3, tool_wear: 3, anomaly_score: 0.18, rul_hours: 138.5, HDF: 0, OSF: 0, PWF: 0, RNF: 0, TWF: 0, status: 'Normal' },
  { machine_id: 'U-03', machine_type: 'Universal', air_temperature: 298.1, process_temperature: 308.5, rotational_speed: 1498, torque: 49.4, tool_wear: 5, anomaly_score: 0.21, rul_hours: 135.0, HDF: 0, OSF: 0, PWF: 0, RNF: 0, TWF: 0, status: 'Normal' },
  { machine_id: 'U-04', machine_type: 'Universal', air_temperature: 298.3, process_temperature: 308.8, rotational_speed: 1489, torque: 51.1, tool_wear: 8, anomaly_score: 0.29, rul_hours: 130.0, HDF: 0, OSF: 0, PWF: 0, RNF: 0, TWF: 0, status: 'Normal' },
  { machine_id: 'U-05', machine_type: 'Universal', air_temperature: 298.4, process_temperature: 309.0, rotational_speed: 1412, torque: 55.7, tool_wear: 14, anomaly_score: 0.44, rul_hours: 122.0, HDF: 0, OSF: 0, PWF: 0, RNF: 0, TWF: 0, status: 'Normal' },
  { machine_id: 'H-01', machine_type: 'High', air_temperature: 299.1, process_temperature: 309.8, rotational_speed: 1285, torque: 68.4, tool_wear: 28, anomaly_score: 0.58, rul_hours: 112.0, HDF: 0, OSF: 0, PWF: 0, RNF: 0, TWF: 0, status: 'Normal' },
  { machine_id: 'H-02', machine_type: 'High', air_temperature: 299.3, process_temperature: 310.1, rotational_speed: 1224, torque: 72.1, tool_wear: 33, anomaly_score: 0.65, rul_hours: 104.0, HDF: 0, OSF: 0, PWF: 0, RNF: 0, TWF: 0, status: 'Warning' },
  { machine_id: 'H-03', machine_type: 'High', air_temperature: 299.6, process_temperature: 310.4, rotational_speed: 1198, torque: 76.3, tool_wear: 39, anomaly_score: 0.71, rul_hours: 96.0, HDF: 0, OSF: 0, PWF: 0, RNF: 0, TWF: 0, status: 'Warning' },
  { machine_id: 'L-01', machine_type: 'Low', air_temperature: 297.8, process_temperature: 307.9, rotational_speed: 1602, torque: 31.2, tool_wear: 11, anomaly_score: 0.15, rul_hours: 148.0, HDF: 0, OSF: 0, PWF: 0, RNF: 0, TWF: 0, status: 'Normal' },
  { machine_id: 'L-02', machine_type: 'Low', air_temperature: 297.9, process_temperature: 308.1, rotational_speed: 1588, torque: 33.7, tool_wear: 17, anomaly_score: 0.19, rul_hours: 143.0, HDF: 0, OSF: 0, PWF: 0, RNF: 0, TWF: 0, status: 'Normal' },
  { machine_id: 'U-11', machine_type: 'Universal', air_temperature: 298.6, process_temperature: 309.1, rotational_speed: 1501, torque: 53.2, tool_wear: 56, anomaly_score: 0.74, rul_hours: 74.0, HDF: 0, OSF: 0, PWF: 0, RNF: 0, TWF: 0, status: 'Warning' },
  { machine_id: 'U-12', machine_type: 'Universal', air_temperature: 297.7, process_temperature: 307.8, rotational_speed: 1402, torque: 19.4, tool_wear: 22, anomaly_score: 0.11, rul_hours: 130.0, HDF: 0, OSF: 0, PWF: 0, RNF: 0, TWF: 0, status: 'Normal' },
];

function generateRollingReading(machine: Machine, intervalIndex: number) {
  const drift = Math.sin(intervalIndex * 0.3) * 0.1;
  const noise = () => (Math.random() - 0.5) * 0.05;
  return {
    air_temperature: +(machine.air_temperature + drift * 2 + noise() * 2).toFixed(1),
    process_temperature: +(machine.process_temperature + drift * 1.5 + noise() * 1.5).toFixed(1),
    rotational_speed: Math.round(machine.rotational_speed + drift * 50 + (Math.random() - 0.5) * 20),
    torque: +(machine.torque + drift * 5 + noise() * 3).toFixed(1),
    tool_wear: Math.min(machine.tool_wear + intervalIndex * 0.5, 240),
  };
}

export async function initializeMachinesData(): Promise<void> {
  try {
    const snapshot = await getDocs(collection(db, 'machines'));
    if (snapshot.empty) {
      for (const machine of DEFAULT_MACHINES) {
        await setDoc(doc(db, 'machines', machine.machine_id), machine);
      }
    }
  } catch (error) {
    console.error('Error initializing machines data:', error);
  }
}

export async function getAllMachines(): Promise<Machine[]> {
  try {
    const snapshot = await getDocs(collection(db, 'machines'));
    const machines: Machine[] = [];
    snapshot.forEach((d) => machines.push(d.data() as Machine));
    machines.sort((a, b) => a.machine_id.localeCompare(b.machine_id));
    return machines.length > 0 ? machines : DEFAULT_MACHINES;
  } catch (error) {
    console.error('Error fetching machines:', error);
    return DEFAULT_MACHINES;
  }
}

export async function getMachineById(machineId: string): Promise<Machine | undefined> {
  try {
    const snapshot = await getDocs(query(collection(db, 'machines'), where('machine_id', '==', machineId)));
    if (!snapshot.empty) return snapshot.docs[0].data() as Machine;
    return undefined;
  } catch (error) {
    return undefined;
  }
}

export async function simulateFault(machineId: string): Promise<void> {
  try {
    const snapshot = await getDocs(query(collection(db, 'machines'), where('machine_id', '==', machineId)));
    if (!snapshot.empty) {
      const docRef = snapshot.docs[0].ref;
      const machine = snapshot.docs[0].data() as Machine;
      const faultTypes: Array<'HDF' | 'OSF' | 'PWF' | 'RNF' | 'TWF'> = ['HDF', 'OSF', 'PWF', 'RNF', 'TWF'];
      const randomFault = faultTypes[Math.floor(Math.random() * faultTypes.length)];
      const newFaultValue = machine[randomFault] === 0 ? 1 : 0;
      const activeFaults = [machine.HDF, machine.OSF, machine.PWF, machine.RNF, machine.TWF].filter((f) => f === 1).length;
      const newAnomalyScore = Math.min(0.2 + activeFaults * 0.15, 1.0);
      const newRulHours = Math.max(150 - activeFaults * 20, 0);
      let newStatus: 'Normal' | 'Warning' | 'Critical' = 'Normal';
      if (newAnomalyScore > 0.7) newStatus = 'Critical';
      else if (newAnomalyScore > 0.4) newStatus = 'Warning';
      await updateDoc(docRef, { [randomFault]: newFaultValue, anomaly_score: newAnomalyScore, rul_hours: newRulHours, status: newStatus });
      if (newFaultValue === 1) {
        await addDoc(collection(db, 'fault_logs'), { machine_id: machineId, timestamp: new Date().toISOString(), fault_type: randomFault, resolved: false } as FaultLog);
      }
    }
  } catch (error) {
    console.error('Error simulating fault:', error);
  }
}

export async function updateMachine(machineId: string, updates: Partial<Machine>): Promise<void> {
  try {
    const snapshot = await getDocs(query(collection(db, 'machines'), where('machine_id', '==', machineId)));
    if (!snapshot.empty) await updateDoc(snapshot.docs[0].ref, updates);
  } catch (error) {
    console.error('Error updating machine:', error);
  }
}

export async function updateMachineMLResults(
  machineId: string,
  mlResult: { anomaly_score: number; failure_vector: { TWF: number; HDF: number; PWF: number; OSF: number; RNF: number }; decision: 'FAILURE' | 'WARNING' | 'NORMAL' }
): Promise<void> {
  try {
    const snapshot = await getDocs(query(collection(db, 'machines'), where('machine_id', '==', machineId)));
    if (!snapshot.empty) {
      const status: Machine['status'] = mlResult.decision === 'FAILURE' ? 'Critical' : mlResult.decision === 'WARNING' ? 'Warning' : 'Normal';
      await updateDoc(snapshot.docs[0].ref, { anomaly_score: mlResult.anomaly_score, TWF: mlResult.failure_vector.TWF as 0 | 1, HDF: mlResult.failure_vector.HDF as 0 | 1, PWF: mlResult.failure_vector.PWF as 0 | 1, OSF: mlResult.failure_vector.OSF as 0 | 1, RNF: mlResult.failure_vector.RNF as 0 | 1, status });
    }
  } catch (error) {
    console.error(`[Firebase] Error writing ML results for ${machineId}:`, error);
  }
}

// ─── Anomaly Logs ─────────────────────────────────────────────────────────────

export async function seedRollingData(machines: Machine[]): Promise<void> {
  try {
    const existing = await getDocs(query(collection(db, 'anomaly_logs'), limit(1)));
    if (!existing.empty) return;
    const now = Date.now();
    const INTERVALS = 12;
    for (const machine of machines) {
      for (let i = INTERVALS; i >= 0; i--) {
        const ts = new Date(now - i * 10 * 60 * 1000).toISOString();
        const reading = generateRollingReading(machine, INTERVALS - i);
        const rawScore = machine.anomaly_score + Math.sin((INTERVALS - i) * 0.5) * 0.08 + (Math.random() - 0.5) * 0.05;
        const anomaly_score = +Math.max(0, Math.min(1, rawScore)).toFixed(3);
        await addDoc(collection(db, 'anomaly_logs'), { machine_id: machine.machine_id, timestamp: ts, anomaly_score, ...reading, decision: anomaly_score > 0.7 ? 'FAILURE' : anomaly_score > 0.4 ? 'WARNING' : 'NORMAL' } as AnomalyLog);
      }
    }
    console.log('[Firebase] Seeded rolling anomaly data for', machines.length, 'machines');
  } catch (err) {
    console.warn('[Firebase] seedRollingData failed:', err);
  }
}

export async function addAnomalyLog(log: AnomalyLog): Promise<void> {
  try { await addDoc(collection(db, 'anomaly_logs'), log); } catch (err) { console.warn('[Firebase] addAnomalyLog failed:', err); }
}

export async function getAnomalyLogs(machineId: string, n = 24): Promise<AnomalyLog[]> {
  try {
    const q = query(collection(db, 'anomaly_logs'), where('machine_id', '==', machineId), orderBy('timestamp', 'desc'), limit(n));
    const snap = await getDocs(q);
    const logs: AnomalyLog[] = snap.docs.map((d) => ({ id: d.id, ...d.data() } as AnomalyLog));
    return logs.reverse();
  } catch (err) {
    console.warn('[Firebase] getAnomalyLogs failed:', err);
    return [];
  }
}

// ─── Fault Logs ───────────────────────────────────────────────────────────────

export async function addFaultLog(log: FaultLog): Promise<string> {
  try { const ref = await addDoc(collection(db, 'fault_logs'), log); return ref.id; } catch (err) { return ''; }
}

export async function getFaultLogs(machineId: string, n = 20): Promise<FaultLog[]> {
  try {
    const q = query(collection(db, 'fault_logs'), where('machine_id', '==', machineId), orderBy('timestamp', 'desc'), limit(n));
    const snap = await getDocs(q);
    return snap.docs.map((d) => ({ id: d.id, ...d.data() } as FaultLog));
  } catch (err) {
    console.warn('[Firebase] getFaultLogs failed:', err);
    return [];
  }
}

export async function resolveFaultLog(logId: string, engineerName: string, notes?: string): Promise<void> {
  try { await updateDoc(doc(db, 'fault_logs', logId), { resolved: true, resolved_by: engineerName, resolved_at: new Date().toISOString(), notes: notes ?? '' }); } catch (err) { console.warn('[Firebase] resolveFaultLog failed:', err); }
}

// ─── Engineer Logs ────────────────────────────────────────────────────────────

export async function addEngineerLog(log: EngineerLog): Promise<void> {
  try { await addDoc(collection(db, 'engineer_logs'), log); } catch (err) { console.warn('[Firebase] addEngineerLog failed:', err); }
}

export async function getEngineerLogs(machineId: string, n = 20): Promise<EngineerLog[]> {
  try {
    const q = query(collection(db, 'engineer_logs'), where('machine_id', '==', machineId), orderBy('timestamp', 'desc'), limit(n));
    const snap = await getDocs(q);
    return snap.docs.map((d) => ({ id: d.id, ...d.data() } as EngineerLog));
  } catch (err) {
    console.warn('[Firebase] getEngineerLogs failed:', err);
    return [];
  }
}

export async function seedEngineerAndFaultLogs(machines: Machine[]): Promise<void> {
  try {
    const existing = await getDocs(query(collection(db, 'engineer_logs'), limit(1)));
    if (!existing.empty) return;
    const engineers = ['Alice Tan', 'Bob Chen', 'Carlos Rivera', 'Diana Lee'];
    const actions = ['Replaced hydraulic filter', 'Recalibrated torque sensor', 'Replaced worn tool bits', 'Inspected bearings, applied lubricant', 'Adjusted rotational speed parameters', 'Power module inspection and reset', 'Full preventive maintenance cycle'];
    const faultTypes: Array<'HDF' | 'OSF' | 'PWF' | 'RNF' | 'TWF'> = ['HDF', 'OSF', 'PWF', 'RNF', 'TWF'];
    const outcomes: Array<'resolved' | 'partial' | 'escalated'> = ['resolved', 'resolved', 'partial', 'escalated'];
    const now = Date.now();
    for (const machine of machines.slice(0, 6)) {
      const visits = 2 + Math.floor(Math.random() * 3);
      for (let v = 0; v < visits; v++) {
        const hoursAgo = (visits - v) * 18 + Math.floor(Math.random() * 6);
        const ts = new Date(now - hoursAgo * 3600 * 1000).toISOString();
        const engineer = engineers[Math.floor(Math.random() * engineers.length)];
        const fts = [faultTypes[Math.floor(Math.random() * faultTypes.length)]];
        const outcome = outcomes[Math.floor(Math.random() * outcomes.length)];
        await addDoc(collection(db, 'engineer_logs'), { machine_id: machine.machine_id, engineer_name: engineer, action: actions[Math.floor(Math.random() * actions.length)], timestamp: ts, fault_types: fts, outcome } as EngineerLog);
        await addDoc(collection(db, 'fault_logs'), { machine_id: machine.machine_id, timestamp: ts, fault_type: fts[0], resolved: outcome === 'resolved', resolved_by: outcome === 'resolved' ? engineer : undefined, resolved_at: outcome === 'resolved' ? ts : undefined } as FaultLog);
      }
    }
    console.log('[Firebase] Seeded engineer & fault logs');
  } catch (err) {
    console.warn('[Firebase] seedEngineerAndFaultLogs failed:', err);
  }
}
