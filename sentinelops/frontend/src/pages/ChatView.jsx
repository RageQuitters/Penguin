import React, { useState, useEffect } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';

export default function ChatView() {
  const [clock, setClock] = useState('');
  const { connected: wsConnected } = useWebSocket();

  useEffect(() => {
    const tick = () => setClock(new Date().toLocaleTimeString('en-SG', {
      hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
    }));
    tick();
    const t = setInterval(tick, 1000);
    return () => clearInterval(t);
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <header style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 24px',
        height: 50,
        borderBottom: '1px solid var(--border)',
        background: 'var(--bg2)',
        flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 26, height: 26,
            border: '1.5px solid var(--accent)',
            borderRadius: 4,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <div style={{
              width: 8, height: 8,
              background: 'var(--accent)',
              borderRadius: 2,
              animation: 'pulse 2s ease-in-out infinite',
            }} />
          </div>
          <span style={{ fontFamily: 'var(--mono)', fontSize: 12, fontWeight: 500, letterSpacing: '0.1em' }}>
            SENTINEL<span style={{ color: 'var(--accent)' }}>OPS</span>
          </span>
          <span style={{
            fontFamily: 'var(--mono)',
            fontSize: 9,
            color: 'var(--text3)',
            border: '1px solid var(--border2)',
            borderRadius: 3,
            padding: '1px 6px',
            marginLeft: 4,
          }}>
            Pangu LLM · Huawei Cloud
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div style={{
            display: 'flex', alignItems: 'center', gap: 6,
            fontFamily: 'var(--mono)', fontSize: 11,
            color: 'var(--text2)',
            border: '1px solid var(--border2)',
            borderRadius: 20, padding: '3px 10px',
          }}>
            <div style={{
              width: 5, height: 5, borderRadius: '50%',
              background: wsConnected ? 'var(--accent)' : 'var(--danger)',
              animation: wsConnected ? 'pulse 2s ease-in-out infinite' : 'none',
            }} />
            {wsConnected ? 'SYSTEM ONLINE' : 'CONNECTING…'}
          </div>
          <span style={{
            fontFamily: 'var(--mono)',
            fontSize: 11,
            color: 'white',
            letterSpacing: '0.05em',
          }}>
            Jurong Plant A
          </span>
          <span style={{ fontFamily: 'var(--mono)', fontSize: 13, color: 'white' }}>
            {clock}
          </span>
        </div>
      </header>

      <div style={{
        flex: 1,
        padding: 30,
        fontFamily: 'var(--mono)',
        color: 'var(--text2)',
        overflowY: 'auto',
      }}>
        <div style={{ fontSize: 30, color: 'var(--accent)', letterSpacing: '0.1em' }}>
          AI AGENT CHAT
        </div>
        <p style={{ fontSize: 16, marginTop: 12, color: 'white' }}>
          Coming soon — orchestrator chat interface.
        </p>
      </div>
    </div>
  );
}