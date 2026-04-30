import { useState, useEffect, useMemo } from 'react';
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';
import { Loader2, Send, CheckCircle2, Sparkles, User, Phone, AlertCircle } from 'lucide-react';
import { toast } from 'sonner';
import {
  rankEngineers,
  FAULT_NAMES,
  type FaultCode,
  type ScoredEngineer,
} from '@shared/faultRouting';
import {
  type Engineer,
  type Machine,
  addAssignment,
  updateAssignment,
} from '@/lib/firebaseService';

const API_BASE = (import.meta as any).env?.VITE_API_BASE_URL ?? '';
const ALL_FAULTS: FaultCode[] = ['HDF', 'OSF', 'PWF', 'RNF', 'TWF'];

interface AssignEngineerDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  machine: Machine | null;
  engineers: Engineer[];
  /** Pre-fill which faults to route on. Defaults to whichever faults are active on the machine. */
  faultTypes?: FaultCode[];
  /** Called after successful assignment, e.g. so the parent can refresh fault lists. */
  onAssigned?: (engineer: Engineer) => void;
}

export default function AssignEngineerDialog({
  open, onOpenChange, machine, engineers, faultTypes, onAssigned,
}: AssignEngineerDialogProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [customMessage, setCustomMessage] = useState('');
  const [showCustomMessage, setShowCustomMessage] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  // Active faults on the machine (for label display)
  const activeFaults = useMemo<FaultCode[]>(() => {
    if (!machine) return [];
    return ALL_FAULTS.filter((f) => machine[f] === 1);
  }, [machine]);

  const faultsForRouting = faultTypes && faultTypes.length > 0 ? faultTypes : activeFaults;

  // Rank engineers locally — purely visual; the server re-ranks for the source of truth
  const ranked: ScoredEngineer<Engineer>[] = useMemo(
    () => rankEngineers(engineers, faultsForRouting),
    [engineers, faultsForRouting],
  );

  // Auto-select the top engineer whenever the dialog reopens
  useEffect(() => {
    if (open && ranked.length > 0) {
      setSelectedId(ranked[0].engineer.id ?? null);
    }
    if (!open) {
      setCustomMessage('');
      setShowCustomMessage(false);
    }
  }, [open, ranked]);

  const handleAssign = async () => {
    if (!machine) return;
    setSubmitting(true);
    try {
      // 1) Pre-create the Firestore assignment so we have an ID to register with the
      //    Telegram callback handler. Status starts as 'assigned' — Track C will flip
      //    it via assignmentSync when the engineer taps a button.
      //    We don't know the picked engineer yet if the user didn't pre-select one,
      //    so we send a placeholder and patch after the server responds.
      const provisional = engineers.find((e) => e.id === selectedId);
      const assignmentId = await addAssignment({
        machine_id: machine.machine_id,
        engineer_id: provisional?.id ?? 'pending',
        engineer_name: provisional?.name ?? 'pending',
        engineer_telegram_chat_id: provisional?.telegram_chat_id,
        fault_types: faultsForRouting,
        status: 'assigned',
        created_at: new Date().toISOString(),
        auto_assigned: !selectedId,
      });

      const res = await fetch(`${API_BASE}/api/assign-engineer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          machine,
          engineers: engineers.map((e) => ({
            id: e.id,
            name: e.name,
            role: e.role,
            specialization: e.specialization,
            active: e.active,
            telegram_chat_id: e.telegram_chat_id,
          })),
          fault_types: faultsForRouting,
          engineer_id: selectedId,
          custom_message: customMessage.trim() || undefined,
          assignment_id: assignmentId,
        }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        toast.error(body.error ?? 'Assignment failed');
        // Roll back the placeholder doc
        if (assignmentId) {
          await updateAssignment(assignmentId, { status: 'escalated', notes: 'Server rejected the assignment' }).catch(() => {});
        }
        return;
      }

      const data = await res.json();
      const chosen = data.chosen?.engineer as Engineer | undefined;
      if (!chosen) {
        toast.error('Server did not return an engineer');
        return;
      }

      // 2) Patch the Firestore doc with the chosen engineer (in case auto-routing
      //    selected someone other than the provisional placeholder).
      if (assignmentId && (!provisional || provisional.id !== chosen.id)) {
        await updateAssignment(assignmentId, {
          engineer_id: chosen.id ?? 'unknown',
          engineer_name: chosen.name,
          engineer_telegram_chat_id: chosen.telegram_chat_id,
        }).catch(() => {});
      }

      const sent = data.sent;
      if (sent) {
        toast.success(`Assigned to ${chosen.name} — Telegram sent`);
      } else {
        const reason = data.telegram_error ?? (chosen.telegram_chat_id ? 'unknown error' : 'no Telegram chat ID configured');
        toast.warning(`Assigned to ${chosen.name}, but Telegram failed: ${reason}`, {
          duration: 8000,
        });
      }
      onAssigned?.(chosen);
      onOpenChange(false);
    } catch (err: any) {
      toast.error(`Assignment error: ${err?.message ?? 'unknown'}`);
    } finally {
      setSubmitting(false);
    }
  };

  if (!machine) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[90vh] flex flex-col gap-5">
        <DialogHeader className="flex-shrink-0">
          <DialogTitle className="flex items-center gap-2.5 text-xl">
            <Sparkles className="h-5 w-5 text-blue-400" />
            Assign Engineer — {machine.machine_id}
          </DialogTitle>
          <DialogDescription className="text-sm pt-1">
            {activeFaults.length > 0 ? (
              <span className="flex items-center gap-2 flex-wrap">
                <span>Active faults:</span>
                {activeFaults.map((f) => (
                  <Badge key={f} variant="secondary" className="font-mono text-xs px-2 py-0.5">
                    {f}
                  </Badge>
                ))}
              </span>
            ) : (
              <>No active faults — assigning for general inspection.</>
            )}
          </DialogDescription>
        </DialogHeader>

        {/* Ranked engineer list */}
        <div className="space-y-3 max-h-[420px] overflow-y-auto pr-2 -mr-2">
          {ranked.length === 0 && (
            <div className="text-center py-10 text-sm text-muted-foreground flex flex-col items-center gap-2">
              <AlertCircle className="h-6 w-6" />
              No active engineers found.
            </div>
          )}

          {ranked.map(({ engineer, score, matchedKeywords, reason }, i) => {
            const isSelected = selectedId === engineer.id;
            const isTop = i === 0 && score > 0;
            const initials = engineer.name.split(' ').map((n) => n[0]).join('').slice(0, 2).toUpperCase();

            return (
              <button
                key={engineer.id}
                onClick={() => setSelectedId(engineer.id ?? null)}
                className={`w-full text-left p-4 rounded-xl border transition-all ${
                  isSelected
                    ? 'border-blue-500 bg-blue-500/10 ring-1 ring-blue-500/30'
                    : 'border-border hover:bg-muted/50 hover:border-muted-foreground/30'
                }`}
              >
                <div className="flex items-start gap-4">
                  {/* Avatar */}
                  <div className={`flex-shrink-0 w-12 h-12 rounded-full flex items-center justify-center text-sm font-bold ${
                    isTop ? 'bg-blue-500/30 text-blue-200 ring-2 ring-blue-500/50' : 'bg-blue-500/20 text-blue-300'
                  }`}>
                    {initials}
                  </div>

                  {/* Main info */}
                  <div className="flex-1 min-w-0 space-y-1.5">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="font-semibold text-base">{engineer.name}</span>
                      {isTop && (
                        <Badge className="bg-blue-500/20 text-blue-300 border-blue-500/30 text-[10px] font-bold tracking-wide px-2 py-0.5">
                          BEST MATCH
                        </Badge>
                      )}
                      {isSelected && <CheckCircle2 className="h-4 w-4 text-blue-400 flex-shrink-0" />}
                    </div>

                    <p className="text-sm text-muted-foreground">{engineer.role}</p>

                    <p className="text-sm flex items-center gap-1.5">
                      <User className="h-3.5 w-3.5 text-muted-foreground" />
                      <span className="text-foreground/90">{engineer.specialization}</span>
                    </p>

                    {matchedKeywords.length > 0 && (
                      <div className="flex flex-wrap gap-1.5 pt-0.5">
                        {matchedKeywords.map((k) => (
                          <span key={k} className="text-xs px-2 py-0.5 bg-blue-500/15 text-blue-300 rounded border border-blue-500/20 font-mono">
                            {k}
                          </span>
                        ))}
                      </div>
                    )}

                    <p className="text-xs text-muted-foreground italic pt-0.5">{reason}</p>
                  </div>

                  {/* Right-side: score + channels */}
                  <div className="flex flex-col items-end gap-2 flex-shrink-0 min-w-[60px]">
                    <div className="text-right">
                      <div className="text-[10px] uppercase tracking-wider text-muted-foreground font-medium">Score</div>
                      <div className={`text-lg font-bold font-mono ${isTop ? 'text-blue-300' : 'text-foreground/80'}`}>
                        {score.toFixed(1)}
                      </div>
                    </div>
                    <div className="flex items-center gap-1.5">
                      {engineer.telegram_chat_id ? (
                        <Send className="h-4 w-4 text-green-400" />
                      ) : (
                        <Send className="h-4 w-4 text-muted-foreground/40" />
                      )}
                      {engineer.phone && (
                        <Phone className="h-4 w-4 text-muted-foreground/60" />
                      )}
                    </div>
                  </div>
                </div>
              </button>
            );
          })}
        </div>

        {/* Optional custom message */}
        <div className="flex-shrink-0">
          {!showCustomMessage ? (
            <button
              onClick={() => setShowCustomMessage(true)}
              className="text-sm text-blue-400 hover:text-blue-300 hover:underline transition-colors"
            >
              + Customise Telegram message
            </button>
          ) : (
            <div className="space-y-2">
              <label className="text-sm text-muted-foreground font-medium">
                Custom Telegram message <span className="text-xs">(leave blank for auto-generated)</span>
              </label>
              <Textarea
                placeholder="e.g. Hi Alice, U-03 just flipped HDF. Can you take it before lunch?"
                value={customMessage}
                onChange={(e) => setCustomMessage(e.target.value)}
                className="text-sm min-h-[100px]"
              />
            </div>
          )}
        </div>

        <DialogFooter className="flex-shrink-0 gap-2">
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={submitting}>
            Cancel
          </Button>
          <Button
            onClick={handleAssign}
            disabled={submitting || !selectedId}
            className="bg-blue-600 hover:bg-blue-700 text-white gap-2"
          >
            {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            {submitting ? 'Assigning…' : 'Assign & Send Telegram'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}