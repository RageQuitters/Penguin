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
  // Machine 01 — Severe (PWF + TWF active faults). Fixed by Zachary — seeded in engineer logs.
  { machine_id: 'U-01', machine_type: 'Universal', air_temperature: 312.4, process_temperature: 328.7, rotational_speed: 1840, torque: 88.6, tool_wear: 198, anomaly_score: 0.91, rul_hours: 8.2, HDF: 0, OSF: 0, PWF: 1, RNF: 0, TWF: 1, status: 'Critical' },
  { machine_id: 'U-02', machine_type: 'Universal', air_temperature: 298.2, process_temperature: 308.7, rotational_speed: 1408, torque: 46.3, tool_wear: 3, anomaly_score: 0.18, rul_hours: 138.5, HDF: 0, OSF: 0, PWF: 0, RNF: 0, TWF: 0, status: 'Normal' },
  // Machine 03 — Severe (HDF + OSF active faults)
  { machine_id: 'U-03', machine_type: 'Universal', air_temperature: 315.9, process_temperature: 332.1, rotational_speed: 1920, torque: 94.2, tool_wear: 221, anomaly_score: 0.88, rul_hours: 12.5, HDF: 1, OSF: 1, PWF: 0, RNF: 0, TWF: 0, status: 'Critical' },
  { machine_id: 'U-04', machine_type: 'Universal', air_temperature: 298.3, process_temperature: 308.8, rotational_speed: 1489, torque: 51.1, tool_wear: 8, anomaly_score: 0.29, rul_hours: 130.0, HDF: 0, OSF: 0, PWF: 0, RNF: 0, TWF: 0, status: 'Normal' },
  { machine_id: 'U-05', machine_type: 'Universal', air_temperature: 298.4, process_temperature: 309.0, rotational_speed: 1412, torque: 55.7, tool_wear: 14, anomaly_score: 0.44, rul_hours: 122.0, HDF: 0, OSF: 0, PWF: 0, RNF: 0, TWF: 0, status: 'Normal' },
  { machine_id: 'U-06', machine_type: 'Universal', air_temperature: 298.8, process_temperature: 309.3, rotational_speed: 1455, torque: 58.1, tool_wear: 19, anomaly_score: 0.51, rul_hours: 118.0, HDF: 0, OSF: 0, PWF: 0, RNF: 0, TWF: 0, status: 'Normal' },
  // Machine 07 — Severe (RNF + PWF active faults)
  { machine_id: 'U-07', machine_type: 'Universal', air_temperature: 318.3, process_temperature: 335.6, rotational_speed: 1975, torque: 97.8, tool_wear: 235, anomaly_score: 0.93, rul_hours: 5.7, HDF: 0, OSF: 0, PWF: 1, RNF: 1, TWF: 0, status: 'Critical' },
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

    // Machine U-01: Zachary fixed it (resolved)
    const u01FixTs = new Date(now - 2 * 3600 * 1000).toISOString();
    await addDoc(collection(db, 'engineer_logs'), {
      machine_id: 'U-01', engineer_name: 'Zachary Lim',
      action: 'Replaced power module and worn tool bits; recalibrated speed controller — all fault indicators cleared',
      timestamp: u01FixTs, fault_types: ['PWF', 'TWF'], outcome: 'resolved',
    } as EngineerLog);
    await addDoc(collection(db, 'fault_logs'), { machine_id: 'U-01', timestamp: u01FixTs, fault_type: 'PWF', resolved: true, resolved_by: 'Zachary Lim', resolved_at: u01FixTs, notes: 'Power module replaced. Machine back online.' } as FaultLog);
    await addDoc(collection(db, 'fault_logs'), { machine_id: 'U-01', timestamp: u01FixTs, fault_type: 'TWF', resolved: true, resolved_by: 'Zachary Lim', resolved_at: u01FixTs, notes: 'Tool bits replaced.' } as FaultLog);

    for (const machine of machines.filter(m => m.machine_id !== 'U-01').slice(0, 6)) {
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

// ─── Engineer Registry ────────────────────────────────────────────────────────

export interface Engineer {
  id?: string;
  name: string;
  role: string;
  telegram_chat_id?: string;
  phone?: string;
  specialization: string;
  active: boolean;
  added_at: string;
}

const DEFAULT_ENGINEERS: Engineer[] = [
  { name: 'Zachary Lim', role: 'Senior Maintenance Engineer', telegram_chat_id: '', phone: '+65 9123 4567', specialization: 'Power systems, tool wear', active: true, added_at: new Date(Date.now() - 90 * 24 * 3600 * 1000).toISOString() },
  { name: 'Alice Tan', role: 'Maintenance Engineer', telegram_chat_id: '', phone: '+65 9234 5678', specialization: 'Hydraulics, bearings', active: true, added_at: new Date(Date.now() - 60 * 24 * 3600 * 1000).toISOString() },
  { name: 'Bob Chen', role: 'Maintenance Technician', telegram_chat_id: '', phone: '+65 9345 6789', specialization: 'Sensor calibration', active: true, added_at: new Date(Date.now() - 45 * 24 * 3600 * 1000).toISOString() },
  { name: 'Carlos Rivera', role: 'Field Engineer', telegram_chat_id: '', phone: '+65 9456 7890', specialization: 'Rotational systems', active: true, added_at: new Date(Date.now() - 30 * 24 * 3600 * 1000).toISOString() },
  { name: 'Diana Lee', role: 'Maintenance Engineer', telegram_chat_id: '', phone: '+65 9567 8901', specialization: 'Preventive maintenance', active: false, added_at: new Date(Date.now() - 10 * 24 * 3600 * 1000).toISOString() },
];

export async function seedEngineers(): Promise<void> {
  try {
    const existing = await getDocs(query(collection(db, 'engineers'), limit(1)));
    if (!existing.empty) return;
    for (const eng of DEFAULT_ENGINEERS) {
      await addDoc(collection(db, 'engineers'), eng);
    }
    console.log('[Firebase] Seeded engineers registry');
  } catch (err) {
    console.warn('[Firebase] seedEngineers failed:', err);
  }
}

export async function getEngineers(): Promise<Engineer[]> {
  try {
    const snap = await getDocs(query(collection(db, 'engineers'), orderBy('added_at', 'asc')));
    return snap.docs.map(d => ({ id: d.id, ...d.data() } as Engineer));
  } catch (err) {
    console.warn('[Firebase] getEngineers failed:', err);
    return DEFAULT_ENGINEERS.map((e, i) => ({ ...e, id: `default-${i}` }));
  }
}

export async function addEngineer(engineer: Omit<Engineer, 'id'>): Promise<string> {
  try {
    const ref = await addDoc(collection(db, 'engineers'), engineer);
    return ref.id;
  } catch (err) {
    console.warn('[Firebase] addEngineer failed:', err);
    return '';
  }
}

export async function updateEngineer(id: string, updates: Partial<Engineer>): Promise<void> {
  try {
    await updateDoc(doc(db, 'engineers', id), updates);
  } catch (err) {
    console.warn('[Firebase] updateEngineer failed:', err);
  }
}
