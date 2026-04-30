import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Send, Loader2, ChevronDown, ChevronRight, ChevronUp, Brain, Wrench } from "lucide-react";
import { chatAgentic, type ChatHistoryItem, type AgentCall } from "@/lib/api";
import type { Machine } from "@/lib/firebaseService";
import { Streamdown } from "streamdown";

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  reasoning?: string | null;
  agentCalls?: AgentCall[];
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
    label: "📲 Text engineers about U-03",
    prompt:
      "Text the engineers about machine U-03 specifically — fetch its current state and write a tailored message about that machine.",
  },
  {
    label: "📄 Generate handover report",
    prompt:
      "Generate a concise handover report for the current fleet. Use summarize_fleet first, then run the appropriate sub-agents on every severe machine. Include a status summary, list critical machines with anomaly score, RUL and active faults, and end with 3 prioritized recommended actions.",
  },
  {
    label: "📍 Where to send mechanics?",
    prompt:
      "Where should I send my mechanics? Use the agents to gather fresh data, then organize the answer into priority tiers: P1 critical (immediate), P2 warning RUL<100h (24h), P3 warning RUL≥100h (48h). For each give machine_id, RUL, active faults, and the action.",
  },
  {
    label: "🔧 Orchestrate fleet decisions",
    prompt:
      "Act as the orchestrator. For every Warning or Critical machine, run the anomaly + fault + predictive sub-agents, combine their outputs, and produce a final urgency level and 2-5 sentence work order per machine. Return as a markdown table.",
  },
];

export default function AIChat({ machines, onAnalyzeAll }: AIChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "0",
      role: "assistant",
      content:
        "Hello! I'm the SentinelOps AI Orchestrator. I can call specialist sub-agents (anomaly, fault, predictive) and notify engineers — every step shows up in the trace below my responses. Ask me anything, or tap a suggested question.",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [expandedReasoning, setExpandedReasoning] = useState<Set<string>>(new Set());
  const [expandedTrace, setExpandedTrace] = useState<Set<string>>(new Set());
  // Collapsed by default — arrow-down indicates "tap to expand". User wanted
  // these tucked away most of the time so the input is the main affordance.
  const [suggestionsOpen, setSuggestionsOpen] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const toggleReasoning = (id: string) => {
    setExpandedReasoning((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleTrace = (id: string) => {
    setExpandedTrace((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  /**
   * Single send path — used by both the input and the preset buttons.
   * Always goes through /api/chat-agentic so the LLM can decide whether to
   * call sub-agents and what to do. No more keyword routing on the client.
   */
  const sendPrompt = async (prompt: string) => {
    const text = prompt.trim();
    if (!text || loading) return;

    const userMessage: ChatMessage = {
      id: `u-${Date.now()}`,
      role: "user",
      content: text,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");

    // Build history (exclude greeting and any tool/reasoning details)
    const history: ChatHistoryItem[] = messages
      .filter((m) => m.id !== "0")
      .map((m) => ({ role: m.role, content: m.content }));

    setLoading(true);

    try {
      const { reply, reasoning, agent_calls } = await chatAgentic(text, machines, history);

      const assistantMessage: ChatMessage = {
        id: `a-${Date.now()}`,
        role: "assistant",
        content: reply || "(empty response)",
        reasoning: reasoning || null,
        agentCalls: agent_calls ?? [],
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
      // Auto-expand trace when there's interesting agent activity
      if (agent_calls && agent_calls.length > 0) {
        setExpandedTrace((prev) => {
          const next = new Set(prev);
          next.add(assistantMessage.id);
          return next;
        });
      }
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          id: `e-${Date.now()}`,
          role: "assistant",
          content:
            "Sorry, I could not reach the AI service. Check the server logs and LLM_API_KEY.",
          timestamp: new Date(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendPrompt(input);
    }
  };

  function formatMessage(text: string): string {
    return (
      text
        // Add spacing before sections
        .replace(/\n(?=[A-Z][a-z]+:)/g, "\n")

        // Space out bullet lists
        .replace(/•/g, "\n•")

        // Add spacing before numbered lists
        .replace(/(\d+\.)/g, "\n$1")

        // Add spacing before ALL CAPS headers
        .replace(/\n([A-Z\s]{6,})\n/g, "\n**$1**\n")

        // Compress excessive newlines
        .replace(/\n{3,}/g, "\n")

        .trim()
    );
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
                msg.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`max-w-[85%] px-3 py-2 rounded-lg text-sm ${
                  msg.role === "user"
                    ? "bg-blue-600 dark:bg-blue-500 text-white"
                    : "bg-slate-700 dark:bg-slate-600 text-slate-100 dark:text-slate-200"
                }`}
              >
                {/* Thinking toggle — only shown on assistant messages that
                    actually have reasoning content. Collapsed by default. */}
                {msg.role === "assistant" && msg.reasoning && (
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
                          ? "Hide thinking"
                          : "Show thinking"}
                      </span>
                    </button>
                    {expandedReasoning.has(msg.id) && (
                      <div className="mt-2 p-2 rounded bg-slate-800/60 dark:bg-slate-900/40 text-xs italic whitespace-pre-wrap break-words border-l-2 border-slate-500/50 max-h-64 overflow-y-auto">
                        {msg.reasoning}
                      </div>
                    )}
                  </div>
                )}

                {/* Agent trace — visible for assistant messages whose request
                    triggered any sub-agent calls. Each entry shows the agent
                    name, what it was given, and what it returned. */}
                {msg.role === "assistant" && msg.agentCalls && msg.agentCalls.length > 0 && (
                  <div className="mb-2">
                    <button
                      onClick={() => toggleTrace(msg.id)}
                      className="flex items-center gap-1 text-xs opacity-80 hover:opacity-100 transition-opacity"
                    >
                      {expandedTrace.has(msg.id) ? (
                        <ChevronDown className="h-3 w-3" />
                      ) : (
                        <ChevronRight className="h-3 w-3" />
                      )}
                      <Wrench className="h-3 w-3" />
                      <span className="font-semibold">
                        {expandedTrace.has(msg.id) ? "Hide" : "Show"} agent trace
                        <span className="ml-1 px-1.5 py-0.5 rounded bg-blue-500/30 text-blue-200 font-mono text-[10px]">
                          {msg.agentCalls.length} call{msg.agentCalls.length === 1 ? '' : 's'}
                        </span>
                      </span>
                    </button>
                    {expandedTrace.has(msg.id) && (
                      <div className="mt-2 space-y-1.5 max-h-80 overflow-y-auto pr-1">
                        {msg.agentCalls.map((call, i) => (
                          <div
                            key={i}
                            className="p-2 rounded bg-slate-900/60 border border-slate-600/40 text-[11px]"
                          >
                            <div className="flex items-center justify-between gap-2 mb-1">
                              <div className="flex items-center gap-1.5 flex-wrap">
                                <span className="text-blue-300 font-mono font-semibold">
                                  {i + 1}. {call.agent}
                                </span>
                                {Object.entries(call.input).slice(0, 3).map(([k, v]) => (
                                  <span key={k} className="text-[10px] px-1 py-0.5 rounded bg-slate-700/60 text-slate-300 font-mono">
                                    {k}={typeof v === 'string' ? v : JSON.stringify(v)}
                                  </span>
                                ))}
                              </div>
                              <span className="text-[10px] text-muted-foreground font-mono whitespace-nowrap">
                                {call.ms}ms
                              </span>
                            </div>
                            <pre className="text-[10px] text-slate-300 whitespace-pre-wrap break-words bg-slate-950/40 p-1.5 rounded max-h-32 overflow-y-auto">
                              {(() => {
                                try {
                                  const s = typeof call.output === 'string' ? call.output : JSON.stringify(call.output, null, 2);
                                  return s.length > 800 ? s.slice(0, 800) + '\n…' : s;
                                } catch { return String(call.output); }
                              })()}
                            </pre>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {msg.role === "user" ? (
                  <div className="whitespace-pre-wrap break-words">
                    {msg.content}
                  </div>
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
                    hour: "2-digit",
                    minute: "2-digit",
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
        {/* Suggested prompt buttons — collapsible to keep the input area uncluttered.
            Default is collapsed (arrow down). Click the header to expand. */}
        <div className="space-y-2">
          <button
            onClick={() => setSuggestionsOpen((v) => !v)}
            className="flex items-center justify-between w-full text-xs text-muted-foreground hover:text-foreground transition-colors"
            aria-expanded={suggestionsOpen}
          >
            <span>Suggested questions</span>
            {suggestionsOpen ? (
              <ChevronDown className="h-3.5 w-3.5" />
            ) : (
              <ChevronUp className="h-3.5 w-3.5" />
            )}
          </button>
          {suggestionsOpen && (
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
          )}
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