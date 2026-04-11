/**
 * useWebSocket — connects to /ws and streams trace entries.
 * Returns { entries, connected, clear }
 * Each entry is a TraceEntry object from the backend.
 */
import { useState, useEffect, useRef, useCallback } from 'react';

const WS_URL = `ws://${window.location.hostname}:8000/ws`;
const MAX_ENTRIES = 200; // keep last N entries to avoid memory bloat

export function useWebSocket() {
  const [entries, setEntries] = useState([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);
  const pingRef = useRef(null);

  useEffect(() => {
    function connect() {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        // Heartbeat ping every 25s
        pingRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) ws.send('ping');
        }, 25000);
      };

      ws.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data);
          // Skip heartbeat/connection messages
          if (data.type === 'heartbeat' || data.type === 'connected') return;
          // TraceEntry has an `action` field — add to entries
          setEntries((prev) => {
            const next = [...prev, data];
            return next.length > MAX_ENTRIES ? next.slice(-MAX_ENTRIES) : next;
          });
        } catch {
          // ignore malformed messages
        }
      };

      ws.onclose = () => {
        setConnected(false);
        clearInterval(pingRef.current);
        // Reconnect after 3s
        setTimeout(connect, 3000);
      };

      ws.onerror = () => ws.close();
    }

    connect();

    return () => {
      clearInterval(pingRef.current);
      wsRef.current?.close();
    };
  }, []);

  const clear = useCallback(() => setEntries([]), []);

  return { entries, connected, clear };
}
