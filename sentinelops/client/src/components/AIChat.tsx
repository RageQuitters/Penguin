import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Send, Loader2, ChevronDown, ChevronRight, Brain } from 'lucide-react';
import { chat, type ChatHistoryItem } from '@/lib/api';
import type { Machine } from '@/lib/firebaseService';
import { Streamdown } from 'streamdown';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  reasoning?: string | null; // only on assistant messages
  timestamp: Date;
}

interface AIChatProps {
  machines: Machine[];
  onAnalyzeAll?: () => void;
}

// Preset prompts shown as tappable buttons above the input.
// Clicking one sends it to the AI exactly as if the user had typed it.
// Edit this list to change the suggested questions — no backend changes needed.
const QUICK_PROMPTS: { label: string; prompt: string }[] = [
  {
    label: '📋 Handover report',
    prompt:
      'Generate a concise handover report for the current fleet. Include a status summary (counts), list critical machines with anomaly score, RUL and active faults, list warning machines the same way, and end with 3 prioritized recommended actions.',
  },
  {
    label: '📍 Where to send mechanics?',
    prompt:
      'Where should I send my mechanics? Organize your answer into three priority tiers: Priority 1 (critical — immediate), Priority 2 (warning with RUL < 100h — within 24h), Priority 3 (warning with RUL ≥ 100h — within 48h). For each machine give machine_id, RUL, active faults, and the action.',
  },
  {
    label: '🔧 Orchestrate fleet decisions',
    prompt:
      'Act as the orchestrator. For every machine that is Warning or Critical, combine its anomaly score, active faults (HDF/OSF/PWF/RNF/TWF) and RUL to produce a final urgency level (low/medium/high/critical) and a 2–5 sentence work order. Return the results as a markdown table.',
  },
];

export default function AIChat({ machines, onAnalyzeAll }: AIChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '0',
      role: 'assistant',
      content:
        "Hello! I'm the SentinelOps AI Assistant. Ask me anything about your fleet, or tap one of the suggested questions below.",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [expandedReasoning, setExpandedReasoning] = useState<Set<string>>(
    new Set()
  );
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const toggleReasoning = (id: string) => {
    setExpandedReasoning((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  /**
   * Unified send path — used by both the free-form input AND the preset
   * prompt buttons. There is no "mode" or special-casing anymore; a button
   * click is just a predetermined string that goes through the same pipe
   * as user-typed text.
   */
  const sendPrompt = async (prompt: string) => {
    const text = prompt.trim();
    if (!text || loading) return;

    const userMessage: ChatMessage = {
      id: `u-${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: new Date(),
    };

    // Build history from current messages (exclude reasoning when forwarding).
    const history: ChatHistoryItem[] = messages
      .filter((m) => m.id !== '0') // skip the greeting
      .map((m) => ({ role: m.role, content: m.content }));

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const { reply, reasoning } = await chat(text, machines, history);

      const assistantMessage: ChatMessage = {
        id: `a-${Date.now()}`,
        role: 'assistant',
        content: reply || '(empty response)',
        reasoning: reasoning || null,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          id: `e-${Date.now()}`,
          role: 'assistant',
          content:
            'Sorry, I could not reach the AI service. Check the server logs and DEEPSEEK_API_KEY.',
          timestamp: new Date(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendPrompt(input);
    }
  };

  function formatMessage(text: string): string {
    return text
      // Add spacing before sections
      .replace(/\n(?=[A-Z][a-z]+:)/g, '\n')

      // Space out bullet lists
      .replace(/•/g, '\n•')

      // Add spacing before numbered lists
      .replace(/(\d+\.)/g, '\n$1')

      // Add spacing before ALL CAPS headers
      .replace(/\n([A-Z\s]{6,})\n/g, '\n**$1**\n')

      // Compress excessive newlines
      .replace(/\n{3,}/g, '\n')

      .trim();
  }
  
  return (
    <div className="flex flex-col h-full bg-background border-l border-border overflow-hidden">
      {/* Header */}
      <div className="flex-shrink-0 p-4 border-b border-border">
        <h2 className="text-sm font-semibold">AI Assistant</h2>
        <p className="text-xs text-muted-foreground mt-1">
          Fleet monitoring & analysis
        </p>
      </div>

      {/* Messages */}
      <ScrollArea className="flex-1 min-h-0">
        <div className="p-4 space-y-4">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${
                msg.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              <div
                className={`max-w-[85%] px-3 py-2 rounded-lg text-sm ${
                  msg.role === 'user'
                    ? 'bg-blue-600 dark:bg-blue-500 text-white'
                    : 'bg-slate-700 dark:bg-slate-600 text-slate-100 dark:text-slate-200'
                }`}
              >
                {/* Thinking toggle — only shown on assistant messages that
                    actually have reasoning content. Collapsed by default. */}
                {msg.role === 'assistant' && msg.reasoning && (
                  <div className="mb-2">
                    <button
                      onClick={() => toggleReasoning(msg.id)}
                      className="flex items-center gap-1 text-xs opacity-70 hover:opacity-100 transition-opacity"
                    >
                      {expandedReasoning.has(msg.id) ? (
                        <ChevronDown className="h-3 w-3" />
                      ) : (
                        <ChevronRight className="h-3 w-3" />
                      )}
                      <Brain className="h-3 w-3" />
                      <span>
                        {expandedReasoning.has(msg.id)
                          ? 'Hide thinking'
                          : 'Show thinking'}
                      </span>
                    </button>
                    {expandedReasoning.has(msg.id) && (
                      <div className="mt-2 p-2 rounded bg-slate-800/60 dark:bg-slate-900/40 text-xs italic whitespace-pre-wrap break-words border-l-2 border-slate-500/50 max-h-64 overflow-y-auto">
                        {msg.reasoning}
                      </div>
                    )}
                  </div>
                )}

                {msg.role === 'user' ? (
                  <div className="whitespace-pre-wrap break-words">{msg.content}</div>
                ) : (
                  <div
                    className="
                      break-words text-sm leading-snug
                      [&>*:first-child]:mt-0
                      [&>*:last-child]:mb-0
                      [&_p]:my-1.5 [&_p]:leading-snug
                      [&_ul]:my-1.5 [&_ul]:pl-4 [&_ul]:space-y-0
                      [&_ol]:my-1.5 [&_ol]:pl-5 [&_ol]:space-y-0
                      [&_li]:my-0 [&_li]:leading-snug
                      [&_li>p]:my-0
                      [&_strong]:text-white [&_strong]:font-semibold
                      [&_h1]:text-sm [&_h1]:font-semibold [&_h1]:mt-2 [&_h1]:mb-1
                      [&_h2]:text-sm [&_h2]:font-semibold [&_h2]:mt-2 [&_h2]:mb-1
                      [&_h3]:text-sm [&_h3]:font-semibold [&_h3]:mt-2 [&_h3]:mb-1
                      [&_hr]:my-2 [&_hr]:border-slate-500/40
                      [&_code]:bg-slate-900/60 [&_code]:px-1 [&_code]:py-0.5
                      [&_code]:rounded [&_code]:text-xs
                      [&_table]:text-xs [&_table]:my-2 [&_table]:border-collapse
                      [&_th]:px-2 [&_th]:py-1 [&_th]:border [&_th]:border-slate-500/40
                      [&_th]:bg-slate-800/40 [&_th]:font-semibold
                      [&_td]:px-2 [&_td]:py-1 [&_td]:border [&_td]:border-slate-500/40
                    "
                  >
                    <Streamdown>{msg.content}</Streamdown>
                  </div>
                )}
                <div className="text-xs opacity-70 mt-1">
                  {msg.timestamp.toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
                </div>
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="bg-slate-700 dark:bg-slate-600 text-slate-100 dark:text-slate-200 px-3 py-2 rounded-lg">
                <Loader2 className="h-4 w-4 animate-spin" />
              </div>
            </div>
          )}
          <div ref={scrollRef} />
        </div>
      </ScrollArea>

      {/* Input Area */}
      <div className="flex-shrink-0 p-4 border-t border-border space-y-3">
        {/* Suggested prompt buttons — each one just sends its prompt string. */}
        <div className="space-y-2">
          <p className="text-xs text-muted-foreground">Suggested questions</p>
          <div className="flex flex-col gap-1.5">
            {QUICK_PROMPTS.map((q) => (
              <button
                key={q.label}
                onClick={() => sendPrompt(q.prompt)}
                disabled={loading}
                className="text-left text-xs px-3 py-2 rounded-md border border-border hover:bg-muted transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {q.label}
              </button>
            ))}
          </div>
        </div>

        {onAnalyzeAll && (
          <Button
            onClick={onAnalyzeAll}
            className="w-full bg-blue-600 hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 text-white"
            variant="default"
            disabled={loading}
          >
            Analyze All Machines
          </Button>
        )}

        <div className="flex gap-2">
          <Input
            placeholder="Ask about machine status..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={loading}
            className="text-sm"
          />
          <Button
            onClick={() => sendPrompt(input)}
            disabled={loading || !input.trim()}
            size="sm"
            className="bg-blue-600 hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 text-white"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}