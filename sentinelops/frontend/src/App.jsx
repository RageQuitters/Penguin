import React, { useState } from 'react';
import Dashboard from './pages/Dashboard';
import ChatView from './pages/ChatView';
import Navigation from './components/Navigation';

export default function App() {
  const [view, setView] = useState('dashboard');

  return (
    <div style={{ display: 'flex', height: '100vh', width: '100vw' }}>
      <Navigation view={view} setView={setView} />
      <main style={{ flex: 1, overflow: 'hidden' }}>
        {view === 'dashboard' && <Dashboard />}
        {view === 'chat' && <ChatView />}
      </main>
    </div>
  );
}
