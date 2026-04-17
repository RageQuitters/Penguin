import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Send, Loader2, FileText, MapPin } from 'lucide-react';
import { generateAIChatResponse } from '@/lib/fakeData';
import type { Machine } from '@/lib/fakeData';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface AIChatProps {
  machines: Machine[];
  onAnalyzeAll?: () => void;
}

export default function AIChat({ machines, onAnalyzeAll }: AIChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '0',
      role: 'assistant',
      content:
        "Hello! I'm the SentinelOps AI Assistant. I can help you monitor your machine fleet, analyze status, and provide recommendations. Ask me anything about your machines or use the quick action buttons below.",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollRef.current) {
      setTimeout(() => {
        scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 0);
    }
  }, [messages]);

  const handleSendMessage = async () => {
    if (!input.trim()) return;

    // Add user message
    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      // Get AI response
      const response = await generateAIChatResponse(input, machines);

      const assistantMessage: ChatMessage = {
        id: `msg-${Date.now()}-ai`,
        role: 'assistant',
        content: response,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage: ChatMessage = {
        id: `msg-${Date.now()}-error`,
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleQuickAction = (action: 'handover' | 'mechanics') => {
    let userContent = '';
    let assistantContent = '';

    if (action === 'handover') {
      userContent = 'Generate a handover report';
      
      const critical = machines.filter((m) => m.status === 'Critical');
      const warning = machines.filter((m) => m.status === 'Warning');
      const normal = machines.filter((m) => m.status === 'Normal');

      assistantContent = `**HANDOVER REPORT**\n\n`;
      assistantContent += `**Fleet Summary:**\n`;
      assistantContent += `- Normal Machines: ${normal.length}\n`;
      assistantContent += `- Warning Machines: ${warning.length}\n`;
      assistantContent += `- Critical Machines: ${critical.length}\n\n`;

      if (critical.length > 0) {
        assistantContent += `**CRITICAL ATTENTION REQUIRED:**\n`;
        critical.forEach((m) => {
          const activeFaults = [m.HDF, m.OSF, m.PWF, m.RNF, m.TWF].filter((f) => f === 1);
          const faultTypes = ['HDF', 'OSF', 'PWF', 'RNF', 'TWF'].filter((_, i) => [m.HDF, m.OSF, m.PWF, m.RNF, m.TWF][i] === 1);
          assistantContent += `- ${m.machine_id}: Anomaly ${m.anomaly_score.toFixed(2)}, RUL ${m.rul_hours.toFixed(1)}h`;
          if (faultTypes.length > 0) {
            assistantContent += `, Faults: ${faultTypes.join(', ')}`;
          }
          assistantContent += `\n`;
        });
        assistantContent += `\n`;
      }

      if (warning.length > 0) {
        assistantContent += `**MACHINES IN WARNING STATE:**\n`;
        warning.forEach((m) => {
          const faultTypes = ['HDF', 'OSF', 'PWF', 'RNF', 'TWF'].filter((_, i) => [m.HDF, m.OSF, m.PWF, m.RNF, m.TWF][i] === 1);
          assistantContent += `- ${m.machine_id}: Anomaly ${m.anomaly_score.toFixed(2)}, RUL ${m.rul_hours.toFixed(1)}h`;
          if (faultTypes.length > 0) {
            assistantContent += `, Faults: ${faultTypes.join(', ')}`;
          }
          assistantContent += `\n`;
        });
        assistantContent += `\n`;
      }

      assistantContent += `**Recommended Actions:**\n`;
      if (critical.length > 0) {
        assistantContent += `1. Immediate maintenance required for: ${critical.map((m) => m.machine_id).join(', ')}\n`;
      }
      if (warning.length > 0) {
        const lowRul = warning.filter((m) => m.rul_hours < 100);
        if (lowRul.length > 0) {
          assistantContent += `2. Schedule tool changes for: ${lowRul.map((m) => m.machine_id).join(', ')}\n`;
        }
      }
      assistantContent += `3. Continue monitoring all machines for anomalies\n`;
    } else if (action === 'mechanics') {
      userContent = 'Where should I send my mechanics to?';

      const critical = machines.filter((m) => m.status === 'Critical');
      const warning = machines.filter((m) => m.status === 'Warning');

      if (critical.length === 0 && warning.length === 0) {
        assistantContent = `**No immediate dispatch required.** All machines are operating normally. Continue routine maintenance schedules.`;
      } else {
        assistantContent = `**MECHANIC DISPATCH RECOMMENDATIONS:**\n\n`;

        if (critical.length > 0) {
          assistantContent += `**PRIORITY 1 - IMMEDIATE (Critical):**\n`;
          critical.forEach((m) => {
            const faultTypes = ['HDF', 'OSF', 'PWF', 'RNF', 'TWF'].filter((_, i) => [m.HDF, m.OSF, m.PWF, m.RNF, m.TWF][i] === 1);
            assistantContent += `- **${m.machine_id}** - Anomaly: ${m.anomaly_score.toFixed(2)}, RUL: ${m.rul_hours.toFixed(1)}h`;
            if (faultTypes.length > 0) {
              assistantContent += `, Faults: ${faultTypes.join(', ')}`;
            }
            assistantContent += `\n`;
            assistantContent += `  Action: Full diagnostic and immediate repair\n`;
          });
          assistantContent += `\n`;
        }

        if (warning.length > 0) {
          const highPriority = warning.filter((m) => m.rul_hours < 100);
          const mediumPriority = warning.filter((m) => m.rul_hours >= 100);

          if (highPriority.length > 0) {
            assistantContent += `**PRIORITY 2 - URGENT (Warning + Low RUL):**\n`;
            highPriority.forEach((m) => {
              const faultTypes = ['HDF', 'OSF', 'PWF', 'RNF', 'TWF'].filter((_, i) => [m.HDF, m.OSF, m.PWF, m.RNF, m.TWF][i] === 1);
              assistantContent += `- **${m.machine_id}** - RUL: ${m.rul_hours.toFixed(1)}h, Tool Wear: ${m.tool_wear}min`;
              if (faultTypes.length > 0) {
                assistantContent += `, Faults: ${faultTypes.join(', ')}`;
              }
              assistantContent += `\n`;
              assistantContent += `  Action: Schedule tool change within 24 hours\n`;
            });
            assistantContent += `\n`;
          }

          if (mediumPriority.length > 0) {
            assistantContent += `**PRIORITY 3 - ROUTINE (Warning + Normal RUL):**\n`;
            mediumPriority.forEach((m) => {
              const faultTypes = ['HDF', 'OSF', 'PWF', 'RNF', 'TWF'].filter((_, i) => [m.HDF, m.OSF, m.PWF, m.RNF, m.TWF][i] === 1);
              assistantContent += `- **${m.machine_id}** - RUL: ${m.rul_hours.toFixed(1)}h`;
              if (faultTypes.length > 0) {
                assistantContent += `, Faults: ${faultTypes.join(', ')}`;
              }
              assistantContent += `\n`;
              assistantContent += `  Action: Monitor closely, schedule maintenance within 48 hours\n`;
            });
            assistantContent += `\n`;
          }
        }

        assistantContent += `**Summary:** Send mechanics to ${critical.length > 0 ? critical.map((m) => m.machine_id).join(', ') : 'no critical machines'} immediately.`;
      }
    }

    // Add user message
    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: userContent,
      timestamp: new Date(),
    };

    // Add assistant message
    const assistantMessage: ChatMessage = {
      id: `msg-${Date.now()}-ai`,
      role: 'assistant',
      content: assistantContent,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex flex-col h-full bg-background border-l border-border overflow-hidden">
      {/* Header */}
      <div className="flex-shrink-0 p-4 border-b border-border">
        <h2 className="text-sm font-semibold">AI Assistant</h2>
        <p className="text-xs text-muted-foreground mt-1">Fleet monitoring & analysis</p>
      </div>

      {/* Messages - Independent ScrollArea */}
      <ScrollArea className="flex-1 min-h-0">
        <div className="p-4 space-y-4">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-xs px-3 py-2 rounded-lg text-sm ${
                  msg.role === 'user'
                    ? 'bg-blue-600 dark:bg-blue-500 text-white'
                    : 'bg-slate-700 dark:bg-slate-600 text-slate-100 dark:text-slate-200'
                }`}
              >
                <div className="whitespace-pre-wrap break-words">{msg.content}</div>
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
        {/* Quick Action Buttons */}
        <div className="grid grid-cols-2 gap-2">
          <Button
            onClick={() => handleQuickAction('handover')}
            size="sm"
            variant="outline"
            className="text-xs h-auto py-2"
          >
            <FileText className="h-3 w-3 mr-1" />
            Handover Report
          </Button>
          <Button
            onClick={() => handleQuickAction('mechanics')}
            size="sm"
            variant="outline"
            className="text-xs h-auto py-2"
          >
            <MapPin className="h-3 w-3 mr-1" />
            Send Mechanics
          </Button>
        </div>

        {onAnalyzeAll && (
          <Button
            onClick={onAnalyzeAll}
            className="w-full bg-blue-600 hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 text-white"
            variant="default"
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
            onClick={handleSendMessage}
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
