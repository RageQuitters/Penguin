import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import {
  UserCheck, UserX, Plus, X, Phone, Cpu, Send, Users, CheckCircle2, ArrowLeft,
} from 'lucide-react';
import {
  getEngineers, addEngineer, updateEngineer, seedEngineers,
  type Engineer,
} from '@/lib/firebaseService';
import { toast } from 'sonner';
import { useLocation } from 'wouter';

const API_BASE = (import.meta as any).env?.VITE_API_BASE_URL ?? '';

const SPECIALIZATIONS = [
  'Power systems, tool wear',
  'Hydraulics, bearings',
  'Sensor calibration',
  'Rotational systems',
  'Preventive maintenance',
  'Electrical systems',
  'PLC & automation',
  'Mechanical systems',
];

const ROLES = [
  'Senior Maintenance Engineer',
  'Maintenance Engineer',
  'Maintenance Technician',
  'Field Engineer',
  'Shift Supervisor',
];

interface AddEngineerForm {
  name: string;
  role: string;
  phone: string;
  telegram_chat_id: string;
  specialization: string;
}

const EMPTY_FORM: AddEngineerForm = {
  name: '',
  role: ROLES[1],
  phone: '',
  telegram_chat_id: '',
  specialization: SPECIALIZATIONS[0],
};

export default function Engineers() {
  const [, navigate] = useLocation();
  const [engineers, setEngineers] = useState<Engineer[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAdd, setShowAdd] = useState(false);
  const [form, setForm] = useState<AddEngineerForm>(EMPTY_FORM);
  const [saving, setSaving] = useState(false);
  const [notifying, setNotifying] = useState(false);

  useEffect(() => {
    const load = async () => {
      await seedEngineers();
      const data = await getEngineers();
      setEngineers(data);
      setLoading(false);
    };
    load();
  }, []);

  const active = engineers.filter(e => e.active);
  const inactive = engineers.filter(e => !e.active);

  const handleToggleActive = async (eng: Engineer) => {
    if (!eng.id) return;
    const updated = { ...eng, active: !eng.active };
    setEngineers(prev => prev.map(e => e.id === eng.id ? updated : e));
    await updateEngineer(eng.id, { active: !eng.active });
    toast.success(`${eng.name} marked as ${!eng.active ? 'active' : 'inactive'}`);
  };

  const handleAdd = async () => {
    if (!form.name.trim()) { toast.error('Name is required'); return; }
    setSaving(true);
    const newEng: Omit<Engineer, 'id'> = {
      name: form.name.trim(),
      role: form.role,
      phone: form.phone.trim(),
      telegram_chat_id: form.telegram_chat_id.trim(),
      specialization: form.specialization,
      active: true,
      added_at: new Date().toISOString(),
    };
    const id = await addEngineer(newEng);
    setEngineers(prev => [...prev, { ...newEng, id: id || `local-${Date.now()}` }]);
    setForm(EMPTY_FORM);
    setShowAdd(false);
    setSaving(false);
    toast.success(`${form.name} added to the team!`);
  };

  const handleNotifyAll = async () => {
    setNotifying(true);
    try {
      const names = active.map(e => `• ${e.name} (${e.role})`).join('\n');
      const message = `👷 *SentinelOps Team Broadcast*\n\nThis is a notification to all active engineers:\n${names}\n\n_Check the SentinelOps dashboard for current machine status._`;
      const res = await fetch(`${API_BASE}/api/telegram/notify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
      });
      if (res.ok) toast.success(`Telegram notification sent to ${active.length} engineers`);
      else toast.error('Telegram service unavailable — check server config');
    } catch {
      toast.error('Could not reach notification service');
    }
    setNotifying(false);
  };

  return (
    <div className="flex flex-col h-screen bg-background overflow-hidden">
      {/* Top nav bar */}
      <div className="flex-shrink-0 flex items-center gap-3 px-6 py-3 border-b border-border">
        <button onClick={() => navigate('/')} className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors">
          <ArrowLeft className="h-3.5 w-3.5" />
          Dashboard
        </button>
        <span className="text-muted-foreground/40">/</span>
        <span className="text-xs font-medium">Engineers</span>
      </div>
      <div className="flex-1 overflow-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-blue-500/10 border border-blue-500/20">
            <Users className="h-5 w-5 text-blue-400" />
          </div>
          <div>
            <h1 className="text-lg font-bold">Engineer Registry</h1>
            <p className="text-xs text-muted-foreground">
              {active.length} active · {inactive.length} inactive
            </p>
          </div>
        </div>
        <div className="flex gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={handleNotifyAll}
            disabled={notifying || active.length === 0}
            className="gap-2"
          >
            <Send className="h-4 w-4" />
            {notifying ? 'Sending…' : 'Notify All via Telegram'}
          </Button>
          <Button
            size="sm"
            onClick={() => setShowAdd(true)}
            className="gap-2 bg-blue-600 hover:bg-blue-700 text-white"
          >
            <Plus className="h-4 w-4" />
            Add Engineer
          </Button>
        </div>
      </div>

      {/* Add Engineer Form */}
      {showAdd && (
        <Card className="p-5 border-blue-500/30 bg-blue-500/5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-bold text-blue-400">New Engineer</h2>
            <button onClick={() => { setShowAdd(false); setForm(EMPTY_FORM); }} className="text-muted-foreground hover:text-foreground">
              <X className="h-4 w-4" />
            </button>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground">Full Name *</label>
              <Input
                placeholder="e.g. Zachary Lim"
                value={form.name}
                onChange={e => setForm(f => ({ ...f, name: e.target.value }))}
                className="text-sm"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground">Role</label>
              <select
                value={form.role}
                onChange={e => setForm(f => ({ ...f, role: e.target.value }))}
                className="w-full h-9 rounded-md border border-input bg-black px-3 text-sm text-white appearance-none"
              >
                {ROLES.map(r => <option key={r} value={r}>{r}</option>)}
              </select>
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground">Phone</label>
              <Input
                placeholder="+65 9123 4567"
                value={form.phone}
                onChange={e => setForm(f => ({ ...f, phone: e.target.value }))}
                className="text-sm"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium text-muted-foreground">Telegram Chat ID</label>
              <Input
                placeholder="e.g. 123456789"
                value={form.telegram_chat_id}
                onChange={e => setForm(f => ({ ...f, telegram_chat_id: e.target.value }))}
                className="text-sm"
              />
            </div>
            <div className="space-y-1 sm:col-span-2">
              <label className="text-xs font-medium text-muted-foreground">Specialization</label>
              <select
                value={form.specialization}
                onChange={e => setForm(f => ({ ...f, specialization: e.target.value }))}
                className="w-full h-9 rounded-md border border-input bg-background px-3 text-sm appearance-none"
              >
                {SPECIALIZATIONS.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
          </div>
          <div className="flex gap-2 mt-4 justify-end">
            <Button variant="outline" size="sm" onClick={() => { setShowAdd(false); setForm(EMPTY_FORM); }}>
              Cancel
            </Button>
            <Button size="sm" onClick={handleAdd} disabled={saving} className="bg-blue-600 hover:bg-blue-700 text-white gap-2">
              <Plus className="h-4 w-4" />
              {saving ? 'Saving…' : 'Add Engineer'}
            </Button>
          </div>
        </Card>
      )}

      {loading ? (
        <div className="flex items-center justify-center py-20">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6">
          {/* Active Engineers */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <CheckCircle2 className="h-4 w-4 text-green-400" />
              <h2 className="text-sm font-semibold">Active Engineers</h2>
              <Badge variant="secondary" className="bg-green-500/10 text-green-400 border-green-500/20">{active.length}</Badge>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
              {active.map(eng => (
                <EngineerCard key={eng.id} engineer={eng} onToggle={() => handleToggleActive(eng)} />
              ))}
              {active.length === 0 && (
                <p className="text-xs text-muted-foreground col-span-3 py-4">No active engineers. Add one above.</p>
              )}
            </div>
          </div>

          {/* Inactive Engineers */}
          {inactive.length > 0 && (
            <div>
              <div className="flex items-center gap-2 mb-3">
                <UserX className="h-4 w-4 text-muted-foreground" />
                <h2 className="text-sm font-semibold text-muted-foreground">Inactive</h2>
                <Badge variant="secondary">{inactive.length}</Badge>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3 opacity-60">
                {inactive.map(eng => (
                  <EngineerCard key={eng.id} engineer={eng} onToggle={() => handleToggleActive(eng)} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
      </div>
    </div>
  );
}

function EngineerCard({ engineer, onToggle }: { engineer: Engineer; onToggle: () => void }) {
  const initials = engineer.name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2);
  const colors = ['bg-blue-500', 'bg-purple-500', 'bg-amber-500', 'bg-green-500', 'bg-pink-500', 'bg-cyan-500'];
  const color = colors[engineer.name.charCodeAt(0) % colors.length];

  return (
    <Card className="p-4 hover:border-border/80 transition-colors">
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className={`${color} flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center text-white text-sm font-bold`}>
            {initials}
          </div>
          <div className="min-w-0">
            <p className="font-semibold text-sm truncate">{engineer.name}</p>
            <p className="text-xs text-muted-foreground truncate">{engineer.role}</p>
          </div>
        </div>
        <button
          onClick={onToggle}
          title={engineer.active ? 'Mark inactive' : 'Mark active'}
          className="flex-shrink-0 mt-0.5 text-muted-foreground hover:text-foreground transition-colors"
        >
          {engineer.active ? <UserCheck className="h-4 w-4 text-green-400" /> : <UserX className="h-4 w-4" />}
        </button>
      </div>
      <div className="mt-3 space-y-1.5">
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <Cpu className="h-3 w-3 flex-shrink-0" />
          <span className="truncate">{engineer.specialization}</span>
        </div>
        {engineer.phone && (
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <Phone className="h-3 w-3 flex-shrink-0" />
            <span>{engineer.phone}</span>
          </div>
        )}
        {engineer.telegram_chat_id && (
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <Send className="h-3 w-3 flex-shrink-0" />
            <span>Chat ID: {engineer.telegram_chat_id}</span>
          </div>
        )}
      </div>
      <div className="mt-3 flex items-center justify-between">
        <Badge
          variant="secondary"
          className={engineer.active
            ? 'bg-green-500/10 text-green-400 border-green-500/20 text-[10px]'
            : 'text-[10px]'
          }
        >
          {engineer.active ? '● Active' : '○ Inactive'}
        </Badge>
        <span className="text-[10px] text-muted-foreground">
          Added {new Date(engineer.added_at).toLocaleDateString()}
        </span>
      </div>
    </Card>
  );
}
