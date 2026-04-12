// frontend/src/components/NavRail.jsx
import React from 'react';

const ITEMS = [
    { id: 'dashboard', label: 'Dashboard', glyph: '⊞' },
    { id: 'chat',      label: 'AI Agent',  glyph: '⬡' }
];

export default function Navigation({ view, setView }) {
  return (
    <nav style={{
      width: 100,
      background: 'var(--bg2)',
      borderRight: '1px solid var(--border)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      padding: '14px 0',
      gap: 6,
      flexShrink: 0,
    }}>
      {ITEMS.map(item => {
        const active = view === item.id;
        return (
          <button
            key={item.id}
            onClick={() => setView(item.id)}
            title={item.label}
            style={{
              width: 80,
              height: 44,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 2,
              fontFamily: 'var(--mono)',
              fontSize: 9,
              letterSpacing: '0.05em',
              border: `1px solid ${active ? 'var(--accent)' : 'transparent'}`,
              background: active ? 'var(--accent-glow)' : 'transparent',
              color: active ? 'var(--accent)' : 'var(--text3)',
              borderRadius: 5,
              cursor: 'pointer',
              transition: 'all 0.15s',
            }}
          >
            <span style={{ fontSize: 14 }}>{item.glyph}</span>
            {item.label}
          </button>
        );
      })}
    </nav>
  );
}