import { initializeApp } from 'firebase/app';
import {
  getFirestore,
  collection,
  getDocs,
  doc,
  updateDoc,
  setDoc,
  query,
  where,
} from 'firebase/firestore';

// Firebase configuration
const firebaseConfig = {
  apiKey: 'AIzaSyBOphhLQ_LKWv9BEmAD7rfwMerWmCZtZ8U',
  authDomain: 'penguin-a7200.firebaseapp.com',
  projectId: 'penguin-a7200',
  storageBucket: 'penguin-a7200.firebasestorage.app',
  messagingSenderId: '525126016743',
  appId: '1:525126016743:web:e7a469b130ba21d0f44c42',
};

// Initialize Firebase
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
  HDF: 0 | 1; // Hydraulic Fluid Deterioration Fault
  OSF: 0 | 1; // Overstrain Fault
  PWF: 0 | 1; // Power Failure Fault
  RNF: 0 | 1; // Rotational Noise Fault
  TWF: 0 | 1; // Tool Wear Fault
  status: 'Normal' | 'Warning' | 'Critical';
}

// Default machines data
const DEFAULT_MACHINES: Machine[] = [
  {
    machine_id: 'U-01',
    machine_type: 'Universal',
    air_temperature: 298.1,
    process_temperature: 308.6,
    rotational_speed: 1551,
    torque: 42.8,
    tool_wear: 0,
    anomaly_score: 0.12,
    rul_hours: 142.0,
    HDF: 0,
    OSF: 0,
    PWF: 0,
    RNF: 0,
    TWF: 0,
    status: 'Normal',
  },
  {
    machine_id: 'U-02',
    machine_type: 'Universal',
    air_temperature: 298.2,
    process_temperature: 308.7,
    rotational_speed: 1408,
    torque: 46.3,
    tool_wear: 3,
    anomaly_score: 0.18,
    rul_hours: 138.5,
    HDF: 0,
    OSF: 0,
    PWF: 0,
    RNF: 0,
    TWF: 0,
    status: 'Normal',
  },
  {
    machine_id: 'U-03',
    machine_type: 'Universal',
    air_temperature: 298.1,
    process_temperature: 308.5,
    rotational_speed: 1498,
    torque: 49.4,
    tool_wear: 5,
    anomaly_score: 0.21,
    rul_hours: 135.0,
    HDF: 0,
    OSF: 0,
    PWF: 0,
    RNF: 0,
    TWF: 0,
    status: 'Normal',
  },
  {
    machine_id: 'U-04',
    machine_type: 'Universal',
    air_temperature: 297.8,
    process_temperature: 307.9,
    rotational_speed: 1398,
    torque: 22.1,
    tool_wear: 12,
    anomaly_score: 0.09,
    rul_hours: 128.0,
    HDF: 0,
    OSF: 0,
    PWF: 0,
    RNF: 0,
    TWF: 0,
    status: 'Normal',
  },
  {
    machine_id: 'U-05',
    machine_type: 'Universal',
    air_temperature: 298.5,
    process_temperature: 308.9,
    rotational_speed: 1555,
    torque: 47.3,
    tool_wear: 18,
    anomaly_score: 0.31,
    rul_hours: 122.0,
    HDF: 0,
    OSF: 0,
    PWF: 0,
    RNF: 0,
    TWF: 0,
    status: 'Normal',
  },
  {
    machine_id: 'U-06',
    machine_type: 'Universal',
    air_temperature: 298.1,
    process_temperature: 308.3,
    rotational_speed: 1483,
    torque: 39.8,
    tool_wear: 29,
    anomaly_score: 0.44,
    rul_hours: 110.5,
    HDF: 0,
    OSF: 0,
    PWF: 0,
    RNF: 0,
    TWF: 0,
    status: 'Warning',
  },
  {
    machine_id: 'U-07',
    machine_type: 'Universal',
    air_temperature: 298.1,
    process_temperature: 308.6,
    rotational_speed: 1543,
    torque: 42.8,
    tool_wear: 34,
    anomaly_score: 0.51,
    rul_hours: 104.0,
    HDF: 0,
    OSF: 0,
    PWF: 0,
    RNF: 0,
    TWF: 0,
    status: 'Warning',
  },
  {
    machine_id: 'U-08',
    machine_type: 'Universal',
    air_temperature: 297.9,
    process_temperature: 308.0,
    rotational_speed: 1449,
    torque: 31.5,
    tool_wear: 8,
    anomaly_score: 0.15,
    rul_hours: 140.0,
    HDF: 0,
    OSF: 0,
    PWF: 0,
    RNF: 0,
    TWF: 0,
    status: 'Normal',
  },
  {
    machine_id: 'U-09',
    machine_type: 'Universal',
    air_temperature: 298.3,
    process_temperature: 308.5,
    rotational_speed: 1598,
    torque: 45.1,
    tool_wear: 22,
    anomaly_score: 0.38,
    rul_hours: 118.0,
    HDF: 0,
    OSF: 0,
    PWF: 0,
    RNF: 0,
    TWF: 0,
    status: 'Normal',
  },
  {
    machine_id: 'U-10',
    machine_type: 'Universal',
    air_temperature: 298.0,
    process_temperature: 308.1,
    rotational_speed: 1552,
    torque: 36.7,
    tool_wear: 41,
    anomaly_score: 0.62,
    rul_hours: 92.5,
    HDF: 0,
    OSF: 0,
    PWF: 0,
    RNF: 0,
    TWF: 0,
    status: 'Warning',
  },
  {
    machine_id: 'U-11',
    machine_type: 'Universal',
    air_temperature: 298.6,
    process_temperature: 309.1,
    rotational_speed: 1501,
    torque: 53.2,
    tool_wear: 56,
    anomaly_score: 0.74,
    rul_hours: 74.0,
    HDF: 0,
    OSF: 0,
    PWF: 0,
    RNF: 0,
    TWF: 0,
    status: 'Warning',
  },
  {
    machine_id: 'U-12',
    machine_type: 'Universal',
    air_temperature: 297.7,
    process_temperature: 307.8,
    rotational_speed: 1402,
    torque: 19.4,
    tool_wear: 22,
    anomaly_score: 0.11,
    rul_hours: 130.0,
    HDF: 0,
    OSF: 0,
    PWF: 0,
    RNF: 0,
    TWF: 0,
    status: 'Normal',
  },
];

export async function initializeMachinesData(): Promise<void> {
  try {
    const machinesRef = collection(db, 'machines');
    const snapshot = await getDocs(machinesRef);

    // If no machines exist, populate with defaults
    if (snapshot.empty) {
      for (const machine of DEFAULT_MACHINES) {
        await setDoc(doc(db, 'machines', machine.machine_id), machine);
      }
    }
  } catch (error) {
    console.error('Error initializing machines data:', error);
    // Silently fail - will use defaults
  }
}

export async function getAllMachines(): Promise<Machine[]> {
  try {
    const machinesRef = collection(db, 'machines');
    const snapshot = await getDocs(machinesRef);
    const machines: Machine[] = [];

    snapshot.forEach((doc) => {
      machines.push(doc.data() as Machine);
    });

    // Sort by machine_id for consistency
    machines.sort((a, b) => a.machine_id.localeCompare(b.machine_id));

    return machines.length > 0 ? machines : DEFAULT_MACHINES;
  } catch (error) {
    console.error('Error fetching machines:', error);
    return DEFAULT_MACHINES;
  }
}

export async function getMachineById(machineId: string): Promise<Machine | undefined> {
  try {
    const machineRef = doc(db, 'machines', machineId);
    const snapshot = await getDocs(query(collection(db, 'machines'), where('machine_id', '==', machineId)));

    if (!snapshot.empty) {
      return snapshot.docs[0].data() as Machine;
    }
    return undefined;
  } catch (error) {
    console.error('Error fetching machine:', error);
    return undefined;
  }
}

export async function simulateFault(machineId: string): Promise<void> {
  try {
    const machinesRef = collection(db, 'machines');
    const snapshot = await getDocs(query(machinesRef, where('machine_id', '==', machineId)));

    if (!snapshot.empty) {
      const docRef = snapshot.docs[0].ref;
      const machine = snapshot.docs[0].data() as Machine;

      // Randomly select a fault type to toggle
      const faultTypes: Array<'HDF' | 'OSF' | 'PWF' | 'RNF' | 'TWF'> = ['HDF', 'OSF', 'PWF', 'RNF', 'TWF'];
      const randomFault = faultTypes[Math.floor(Math.random() * faultTypes.length)];

      // Toggle the selected fault
      const newFaultValue = machine[randomFault] === 0 ? 1 : 0;

      // Calculate new anomaly score and RUL based on fault count
      const activeFaults = [machine.HDF, machine.OSF, machine.PWF, machine.RNF, machine.TWF].filter((f) => f === 1).length;
      const newAnomalyScore = Math.min(0.2 + activeFaults * 0.15, 1.0);
      const newRulHours = Math.max(150 - activeFaults * 20, 0);

      let newStatus: 'Normal' | 'Warning' | 'Critical' = 'Normal';
      if (newAnomalyScore > 0.7) {
        newStatus = 'Critical';
      } else if (newAnomalyScore > 0.4) {
        newStatus = 'Warning';
      }

      const updateData: Record<string, any> = {
        [randomFault]: newFaultValue,
        anomaly_score: newAnomalyScore,
        rul_hours: newRulHours,
        status: newStatus,
      };

      await updateDoc(docRef, updateData);
    }
  } catch (error) {
    console.error('Error simulating fault:', error);
  }
}

export async function updateMachine(machineId: string, updates: Partial<Machine>): Promise<void> {
  try {
    const machinesRef = collection(db, 'machines');
    const snapshot = await getDocs(query(machinesRef, where('machine_id', '==', machineId)));

    if (!snapshot.empty) {
      const docRef = snapshot.docs[0].ref;
      await updateDoc(docRef, updates);
    }
  } catch (error) {
    console.error('Error updating machine:', error);
  }
}

/**
 * Persists the ML-computed anomaly score and fault flags back to Firestore.
 * Called after /api/predict-all returns so that the dashboard, the AI chat,
 * and any other client always read up-to-date values from the database.
 *
 * Fields written:
 *   anomaly_score  — LOF model output (0–1)
 *   TWF, HDF, PWF, OSF, RNF — Random Forest fault classifier flags (0 | 1)
 *   status         — derived from decision string ("FAILURE" | "WARNING" | "NORMAL")
 */
export async function updateMachineMLResults(
  machineId: string,
  mlResult: {
    anomaly_score: number;
    failure_vector: { TWF: number; HDF: number; PWF: number; OSF: number; RNF: number };
    decision: 'FAILURE' | 'WARNING' | 'NORMAL';
  }
): Promise<void> {
  try {
    const machinesRef = collection(db, 'machines');
    const snapshot = await getDocs(query(machinesRef, where('machine_id', '==', machineId)));

    if (!snapshot.empty) {
      const docRef = snapshot.docs[0].ref;

      const status: Machine['status'] =
        mlResult.decision === 'FAILURE'
          ? 'Critical'
          : mlResult.decision === 'WARNING'
          ? 'Warning'
          : 'Normal';

      await updateDoc(docRef, {
        anomaly_score: mlResult.anomaly_score,
        TWF: (mlResult.failure_vector.TWF as 0 | 1),
        HDF: (mlResult.failure_vector.HDF as 0 | 1),
        PWF: (mlResult.failure_vector.PWF as 0 | 1),
        OSF: (mlResult.failure_vector.OSF as 0 | 1),
        RNF: (mlResult.failure_vector.RNF as 0 | 1),
        status,
      });
    }
  } catch (error) {
    console.error(`[Firebase] Error writing ML results for ${machineId}:`, error);
    // Non-fatal — the in-memory state is already correct; the write will
    // succeed on the next predict cycle.
  }
}