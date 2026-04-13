/**
 * RightPanel — tabbed panel for the right column.
 * Tabs:
 *   AI AGENT — SentinelOps chat interface (previously "under construction")
 *   TRACE    — live agent trace log (existing AgentTracePanel)
 *
 * The notification badge on TRACE tab pulses when new entries arrive.
 */
import React, { useState, useEffect, useRef } from 'react';
import SentinelChat from './SentinelChat';
import AgentTracePanel from './AgentTracePanel';

export default function RightPanel({ entries, connected, onClear }) {
  const [activeTab, setActiveTab] = useState('chat');
  const [unreadTrace, setUnreadTrace] = useState(0);
  const prevLength = useRef(entries.length);

  // Track unread trace entries when user is on chat tab
  useEffect(() => {
    if (entries.length > prevLength.current) {
      if (activeTab !== 'trace') {
        setUnreadTrace(n => n + (entries.length - prevLength.current));
      }
    }
    prevLength.current = entries.length;
  }, [entries.length, activeTab]);

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    if (tab === 'trace') setUnreadTrace(0);
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      background: 'var(--bg2)',
    }}>
      {/* Tab bar */}
      <div style={{
        display: 'flex',
        borderBottom: '1px solid var(--border)',
        flexShrink: 0,
        background: 'var(--bg2)',
      }}>
        <TabButton
          label="AI AGENT"
          active={activeTab === 'chat'}
          onClick={() => handleTabChange('chat')}
          icon="◈"
          accent
        />
        <TabButton
          label="TRACE"
          active={activeTab === 'trace'}
          onClick={() => handleTabChange('trace')}
          icon="⚙"
          badge={unreadTrace > 0 ? unreadTrace : null}
          connected={connected}
        />
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {activeTab === 'chat' ? (
          <SentinelChat />
        ) : (
          <AgentTracePanel
            entries={entries}
            connected={connected}
            onClear={() => { onClear(); setUnreadTrace(0); }}
          />
        )}
      </div>
    </div>
  );
}

function TabButton({ label, active, onClick, icon, badge, accent, connected }) {
  return (
    <button
      onClick={onClick}
      style={{
        flex: 1,
        fontFamily: 'var(--mono)',
        fontSize: 10,
        letterSpacing: '0.08em',
        padding: '10px 12px',
        background: active ? 'var(--bg)' : 'transparent',
        border: 'none',
        borderBottom: active ? `2px solid ${accent ? 'var(--accent)' : 'var(--accent)'}` : '2px solid transparent',
        color: active ? 'var(--accent)' : 'var(--text3)',
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 6,
        transition: 'all 0.15s',
        position: 'relative',
      }}
    >
      <span style={{ fontSize: 11 }}>{icon}</span>
      {label}

      {/* Live indicator for trace tab */}
      {label === 'TRACE' && (
        <span style={{
          width: 5, height: 5,
          borderRadius: '50%',
          background: connected ? 'var(--accent)' : 'var(--danger)',
          animation: connected ? 'pulse 2s ease-in-out infinite' : 'none',
          flexShrink: 0,
        }} />
      )}

      {/* Unread badge */}
      {badge && (
        <span style={{
          fontFamily: 'var(--mono)',
          fontSize: 8,
          padding: '1px 5px',
          borderRadius: 10,
          background: 'var(--warn)',
          color: '#07090a',
          fontWeight: 700,
          animation: 'pulse 1s ease-in-out infinite',
          minWidth: 16,
          textAlign: 'center',
        }}>
          {badge > 99 ? '99+' : badge}
        </span>
      )}
    </button>
  );
}
