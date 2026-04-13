/**
 * SentinelChat — AI Agent chat interface for SentinelOps.
 * Styled to match the dark terminal aesthetic (IBM Plex Mono, teal accent).
 * Supports multi-turn conversation with live plant context via /api/chat.
 */
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { sendChatMessage } from '../services/api';

const QUICK_QUERIES = [
  { label: 'Unit 7 anomaly status', query: 'What is the current anomaly status of Unit 7?' },
  { label: 'Active fault types', query: 'What are the active fault types across all machines?' },
  { label: 'Tool wear RUL', query: 'Show me RUL estimates for machines with tool wear faults.' },
  { label: 'Shift handover report', query: 'Generate a shift handover report.' },
  { label: 'Upcoming maintenance', query: 'What is the recommended maintenance schedule for the next 24 hours?' },
  { label: 'Which machine to prioritize', query: 'Which machine should I send my engineers to?' },
];

const AGENTS = [
  { label: '● Anomaly detector', color: '#5b9cf6' },
  { label: '● Fault classifier', color: '#f5a623' },
  { label: '● Predictive maint.', color: '#b085f5' },
  { label: '● Orchestrator (Pangu)', color: '#00d4aa' },
];

function MessageBubble({ msg }) {
  const isBot = msg.role === 'assistant';
  const lines = msg.content.split('\n');

  return (
    <div style={{
      display: 'flex',
      flexDirection: isBot ? 'row' : 'row-reverse',
      gap: 10,
      padding: '12px 14px',
      borderBottom: '1px solid var(--border)',
      animation: 'scanIn 0.18s ease both',
    }}>
      {/* Avatar */}
      <div style={{
        width: 28,
        height: 28,
        borderRadius: 4,
        background: isBot ? 'var(--accent-glow)' : 'var(--bg4)',
        border: `1px solid ${isBot ? 'var(--accent)' : 'var(--border2)'}`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
        fontFamily: 'var(--mono)',
        fontSize: 9,
        color: isBot ? 'var(--accent)' : 'var(--text2)',
        fontWeight: 600,
        letterSpacing: '0.02em',
      }}>
        {isBot ? '◈' : 'YOU'}
      </div>

      {/* Bubble */}
      <div style={{
        flex: 1,
        maxWidth: '85%',
        alignSelf: isBot ? 'flex-start' : 'flex-end',
      }}>
        {isBot && (
          <div style={{
            fontFamily: 'var(--mono)',
            fontSize: 9,
            color: 'var(--accent)',
            marginBottom: 5,
            letterSpacing: '0.06em',
          }}>
            SENTINELOPS · Pangu LLM
          </div>
        )}
        <div style={{
          background: isBot ? 'var(--bg3)' : 'var(--bg4)',
          border: `1px solid ${isBot ? 'var(--border2)' : 'var(--border)'}`,
          borderRadius: isBot ? '2px 8px 8px 8px' : '8px 2px 8px 8px',
          padding: '10px 12px',
          fontSize: 12,
          color: 'var(--text)',
          lineHeight: 1.6,
        }}>
          {lines.map((line, i) => {
            // Bold **text** support
            const parts = line.split(/\*\*(.*?)\*\*/g);
            return (
              <React.Fragment key={i}>
                {parts.map((part, j) =>
                  j % 2 === 1
                    ? <strong key={j} style={{ color: 'var(--accent)', fontWeight: 600 }}>{part}</strong>
                    : <span key={j}>{part}</span>
                )}
                {i < lines.length - 1 && <br />}
              </React.Fragment>
            );
          })}
        </div>
        {msg.timestamp && (
          <div style={{
            fontFamily: 'var(--mono)',
            fontSize: 9,
            color: 'var(--text3)',
            marginTop: 3,
            textAlign: isBot ? 'left' : 'right',
          }}>
            {new Date(msg.timestamp).toLocaleTimeString('en-SG', {
              hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
            })}
          </div>
        )}
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div style={{
      display: 'flex',
      gap: 10,
      padding: '12px 14px',
      borderBottom: '1px solid var(--border)',
    }}>
      <div style={{
        width: 28, height: 28,
        borderRadius: 4,
        background: 'var(--accent-glow)',
        border: '1px solid var(--accent)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        flexShrink: 0,
        fontFamily: 'var(--mono)',
        fontSize: 9,
        color: 'var(--accent)',
      }}>◈</div>
      <div style={{
        background: 'var(--bg3)',
        border: '1px solid var(--border2)',
        borderRadius: '2px 8px 8px 8px',
        padding: '10px 14px',
        display: 'flex',
        alignItems: 'center',
        gap: 4,
      }}>
        {[0, 1, 2].map(i => (
          <div key={i} style={{
            width: 5, height: 5,
            borderRadius: '50%',
            background: 'var(--accent)',
            animation: 'pulse 1.2s ease-in-out infinite',
            animationDelay: `${i * 0.2}s`,
          }} />
        ))}
        <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'var(--text3)', marginLeft: 6 }}>
          Pangu reasoning…
        </span>
      </div>
    </div>
  );
}

export default function SentinelChat() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'SentinelOps online. I\'m monitoring **Jurong Plant A** across all sensor channels.\n\nMy agents are running — anomaly detection, fault classification, and predictive maintenance. Ask me about plant status, specific machines, fault history, or maintenance scheduling.\n\nCurrent watch: **tool wear at 187 min** on Unit 7 is approaching threshold. Rotational speed variance noted.',
      timestamp: new Date().toISOString(),
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  // Auto-resize textarea to fit content (clamped between 36px and 100px)
  useEffect(() => {
    const ta = inputRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = `${Math.max(36, Math.min(ta.scrollHeight, 100))}px`;
  }, [input]);

  const send = useCallback(async (text) => {
    const content = (text || input).trim();
    if (!content || loading) return;

    const userMsg = { role: 'user', content, timestamp: new Date().toISOString() };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);
    setError(null);

    // Build API messages (exclude timestamps — API doesn't need them)
    const apiMessages = [...messages, userMsg].map(({ role, content }) => ({ role, content }));

    try {
      const { reply } = await sendChatMessage(apiMessages);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: reply,
        timestamp: new Date().toISOString(),
      }]);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [input, messages, loading]);

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', background: 'var(--bg2)' }}>

      {/* Live sensors summary bar */}
      <div style={{
        padding: '8px 14px',
        borderBottom: '1px solid var(--border)',
        flexShrink: 0,
        background: 'var(--bg3)',
      }}>
        <div style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'white', letterSpacing: '0.08em', marginBottom: 6 }}>
          LIVE SENSORS
        </div>
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
          {[
            { label: 'Air temp', value: '298.1 K', color: 'var(--text)' },
            { label: 'Process temp', value: '308.6 K', color: 'var(--text)' },
            { label: 'Rot. speed', value: '1559 rpm', color: 'var(--warn)' },
            { label: 'Torque', value: '42.5 Nm', color: 'var(--text)' },
            { label: 'Tool wear', value: '187 min', color: 'var(--danger)' },
          ].map(s => (
            <div key={s.label} style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              <span style={{ fontFamily: 'var(--mono)', fontSize: 8, color: 'grey' }}>{s.label}</span>
              <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: s.color, fontWeight: 500 }}>{s.value}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Agents sidebar info */}
      <div style={{
        padding: '8px 14px',
        borderBottom: '1px solid var(--border)',
        flexShrink: 0,
      }}>
        <div style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'var(--text3)', letterSpacing: '0.08em', marginBottom: 5 }}>
          AGENTS
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          {AGENTS.map(a => (
            <div key={a.label} style={{ fontFamily: 'var(--mono)', fontSize: 10, color: a.color }}>
              {a.label}
            </div>
          ))}
        </div>
      </div>

      {/* Messages area */}
      <div style={{ flex: 1, overflowY: 'auto' }}>
        {messages.map((msg, i) => (
          <MessageBubble key={i} msg={msg} />
        ))}
        {loading && <TypingIndicator />}
        <div ref={bottomRef} />
      </div>

      {/* Quick queries */}
      <div style={{
        padding: '8px 12px',
        borderTop: '1px solid var(--border)',
        borderBottom: '1px solid var(--border)',
        flexShrink: 0,
        background: 'var(--bg3)',
      }}>
        <div style={{ fontFamily: 'var(--mono)', fontSize: 9, color: 'grey', letterSpacing: '0.08em', marginBottom: 6 }}>
          QUICK QUERIES
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          {QUICK_QUERIES.map(q => (
            <button
              key={q.label}
              onClick={() => send(q.query)}
              disabled={loading}
              style={{
                textAlign: 'left',
                fontFamily: 'var(--mono)',
                fontSize: 10,
                color: loading ? 'var(--text3)' : 'var(--text2)',
                background: 'none',
                border: '1px solid var(--border)',
                borderRadius: 4,
                padding: '4px 8px',
                cursor: loading ? 'not-allowed' : 'pointer',
                transition: 'all 0.1s',
              }}
              onMouseEnter={e => { if (!loading) { e.target.style.borderColor = 'var(--accent)'; e.target.style.color = 'var(--accent)'; } }}
              onMouseLeave={e => { e.target.style.borderColor = 'var(--border)'; e.target.style.color = loading ? 'var(--text3)' : 'var(--text2)'; }}
            >
              {q.label}
            </button>
          ))}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div style={{
          margin: '6px 12px 0',
          padding: '6px 10px',
          background: 'var(--danger-bg)',
          border: '1px solid var(--danger)',
          borderRadius: 4,
          fontFamily: 'var(--mono)',
          fontSize: 10,
          color: 'var(--danger)',
          flexShrink: 0,
        }}>
          ⚠ {error}
        </div>
      )}

      {/* Input */}
      <div style={{
        padding: '10px 12px',
        borderTop: '1px solid var(--border)',
        flexShrink: 0,
        display: 'flex',
        gap: 8,
        alignItems: 'flex-end',
        background: 'var(--bg2)',
      }}>
        <textarea
          ref={inputRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKey}
          disabled={loading}
          placeholder="Ask about plant status, machines, faults…"
          rows={1}
          style={{
            flex: 1,
            fontFamily: 'var(--mono)',
            fontSize: 11,
            color: 'var(--text)',
            background: 'var(--bg3)',
            border: '1px solid var(--border2)',
            borderRadius: 6,
            padding: '8px 10px',
            resize: 'none',
            outline: 'none',
            lineHeight: 1.5,
            maxHeight: 100,
            overflowY: 'auto',
            transition: 'border-color 0.15s',
          }}
          onFocus={e => { e.target.style.borderColor = 'var(--accent)'; }}
          onBlur={e => { e.target.style.borderColor = 'var(--border2)'; }}
        />
        <button
          onClick={() => send()}
          disabled={loading || !input.trim()}
          style={{
            width: 36,
            height: 36,
            borderRadius: 6,
            border: '1px solid var(--accent)',
            background: (loading || !input.trim()) ? 'var(--accent-glow)' : 'var(--accent)',
            color: (loading || !input.trim()) ? 'var(--accent)' : '#07090a',
            cursor: (loading || !input.trim()) ? 'not-allowed' : 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 14,
            transition: 'all 0.15s',
            flexShrink: 0,
          }}
        >
          ↑
        </button>
      </div>
    </div>
  );
}