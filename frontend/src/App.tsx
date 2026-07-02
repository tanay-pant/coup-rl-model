import { useState, useEffect, useRef } from 'react';
import './index.css';

interface Card {
  role: string;
  revealed: boolean;
}

interface Player {
  id: number;
  name: string;
  cash: number;
  cards: Card[];
  alive: boolean;
}

interface ValidAction {
  id: number;
  name: string;
}

interface GameState {
  phase: string;
  active_player: number;
  players: Player[];
  exchange_pool: { id: number; role: string }[];
  valid_actions: ValidAction[];
  log: string[];
  winner: string | null;
}

const WS_URL = import.meta.env.DEV ? "ws://127.0.0.1:7860/ws" : `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/ws`;

function App() {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const ws = useRef<WebSocket | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    connect();
    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, []);

  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [gameState?.log]);

  const connect = () => {
    ws.current = new WebSocket(WS_URL);
    ws.current.onopen = () => {
      setConnected(true);
      setError(null);
    };
    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'state_update') {
        setGameState(data.data);
      } else if (data.type === 'error') {
        setError(data.message);
      }
    };
    ws.current.onclose = () => {
      setConnected(false);
    };
  };

  const handleAction = (actionId: number) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ type: 'action', action_id: actionId }));
    }
  };

  if (!connected) {
    return (
      <div className="window" style={{ maxWidth: '400px', margin: '0 auto' }}>
        <div className="title-bar">
          <span>Connection Status</span>
        </div>
        <div className="window-content">
          <p>Connecting to AI Server...</p>
          <button onClick={connect}>Retry</button>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="window" style={{ maxWidth: '400px', margin: '0 auto' }}>
        <div className="title-bar">
          <span>Error</span>
        </div>
        <div className="window-content">
          <p>{error}</p>
          <button onClick={connect}>Reconnect</button>
        </div>
      </div>
    );
  }

  if (!gameState) {
    return (
      <div className="window" style={{ maxWidth: '400px', margin: '0 auto' }}>
        <div className="title-bar">
          <span>Loading</span>
        </div>
        <div className="window-content">
          <p>Waiting for game state...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container" style={{ maxWidth: '1000px', margin: '0 auto' }}>
      <div className="window">
        <div className="title-bar">
          <span>Coup AI - Retro Edition</span>
        </div>
        <div className="window-content">
          <h3>Phase: {gameState.phase} | Winner: {gameState.winner || "None"}</h3>
          
          <div className="player-container">
            {gameState.players.map((p) => (
              <div key={p.id} className={`player-box ${gameState.active_player === p.id ? 'active' : ''}`} style={{ opacity: p.alive ? 1 : 0.5 }}>
                <strong>{p.name}</strong>
                <p style={{ margin: '4px 0' }}>Cash: {p.cash}</p>
                <div>
                  {p.cards.map((c, i) => (
                    <div key={i}>[{c.revealed ? `${c.role} (DEAD)` : c.role}]</div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {gameState.exchange_pool.length > 0 && (
            <div style={{ marginTop: '10px' }}>
              <strong>Exchange Pool:</strong>
              {gameState.exchange_pool.map((c) => (
                <span key={c.id} style={{ marginLeft: '5px' }}>[{c.id}: {c.role}]</span>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="window">
        <div className="title-bar">
          <span>Actions (You)</span>
        </div>
        <div className="window-content">
          {gameState.valid_actions.length > 0 ? (
            <div style={{ display: 'flex', flexWrap: 'wrap' }}>
              {gameState.valid_actions.map(action => (
                <button key={action.id} onClick={() => handleAction(action.id)}>
                  {action.name}
                </button>
              ))}
            </div>
          ) : (
            <p>{gameState.winner ? "Game Over! Refresh to restart." : "Waiting for AI..."}</p>
          )}
        </div>
      </div>

      <div className="window">
        <div className="title-bar">
          <span>Game Log</span>
        </div>
        <div className="window-content">
          <div className="log-box">
            {gameState.log.map((msg, i) => (
              <div key={i}>{msg}</div>
            ))}
            <div ref={logEndRef} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
