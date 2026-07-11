import { useState, useEffect, useRef } from 'react';
import './index.css';
import { initAudio, setMuted, Sounds } from './audio';

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
  deck_count: number;
  player_0_placement: number | null;
}

type AppState = 'connecting' | 'lobby' | 'playing';

const WS_URL = "wss://ptanay-coup-rl-backend.hf.space/ws";

const toTitleCase = (str: string) => {
  if (!str || str === 'HIDDEN') return 'HIDDEN';
  return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
};

const getShortRoleName = (role: string) => {
  const t = toTitleCase(role);
  switch (t) {
    case 'Duke': return 'Duke';
    case 'Assassin': return 'Assn';
    case 'Captain': return 'Capt';
    case 'Ambassador': return 'Ambs';
    case 'Contessa': return 'Cnts';
    case 'HIDDEN': return 'HIDDEN';
    default: return t;
  }
};

const getRoleIcon = (role: string) => {
  switch (toTitleCase(role)) {
    case 'Duke': return '🎩';
    case 'Assassin': return '🗡️';
    case 'Captain': return '⚓';
    case 'Ambassador': return '📜';
    case 'Contessa': return '👑';
    default: return '❓';
  }
};

const getActionIcon = (name: string) => {
  if (name.includes('Income')) return '💰';
  if (name.includes('Foreign Aid')) return '💸';
  if (name.includes('Tax')) return '🎩';
  if (name.includes('Exchange return')) return '🗂️';
  if (name.includes('Exchange')) return '📜';
  if (name.includes('Steal')) return '⚓';
  if (name.includes('Assassinate')) return '🗡️';
  if (name.includes('Coup')) return '💀';
  if (name.includes('Challenge')) return '🗣️';
  if (name.includes('Allow')) return '👍';
  if (name.includes('Block')) return '🛡️';
  if (name.includes('Reveal')) return '👁️';
  return '▶️';
};

const getOrdinal = (n: number | null) => {
  if (n === null) return "";
  const s = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
};

const parseLogToSnippet = (msg: string) => {
  let icon = '▶️';
  // Response actions have priority for icons
  if (msg.includes('allow') || msg.includes('accept')) icon = '👍';
  else if (msg.includes('block')) icon = '🛡️';
  else if (msg.includes('challeng')) icon = '🗣️';
  else if (msg.includes('reveal')) icon = '👁️';
  // Primary actions
  else if (msg.includes('Assassinate')) icon = '🗡️';
  else if (msg.includes('Coup')) icon = '💀';
  else if (msg.includes('Income')) icon = '💰';
  else if (msg.includes('Foreign Aid')) icon = '💸';
  else if (msg.includes('Tax')) icon = '🎩';
  else if (msg.includes('Exchange')) icon = '📜';
  else if (msg.includes('Steal')) icon = '⚓';

  const shortMsg = msg.replace('decided to ', '').replace('chose: ', '').replace(/\s*\([^)]*\)/, '');
  return `${icon} ${shortMsg}`;
};

function App() {
  const [appState, setAppState] = useState<AppState>('connecting');
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [localLog, setLocalLog] = useState<string[]>([]);
  const [turnLogs, setTurnLogs] = useState<{id: number, text: string, original: string}[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [botCount, setBotCount] = useState<number>(3);
  const [isSidebarOpen, setIsSidebarOpen] = useState(window.innerWidth > 768);
  const [isMuted, setIsMutedState] = useState(false);
  const [showFullRules, setShowFullRules] = useState(false);
  const [showShortcut, setShowShortcut] = useState(false);

  const toggleMute = () => {
    const newMuted = !isMuted;
    setIsMutedState(newMuted);
    setMuted(newMuted);
    if (!newMuted) initAudio();
  };
  
  const ws = useRef<WebSocket | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);
  const lastLogLength = useRef<number>(0);
  const logIdCounter = useRef<number>(0);
  const previousTurnPhase = useRef<string | null>(null);

  useEffect(() => {
    connect();
    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, []);

  useEffect(() => {
    if (logEndRef.current && isSidebarOpen) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [localLog, isSidebarOpen]);

  useEffect(() => {
    if (!gameState) return;

    let currentLastLength = lastLogLength.current;
    
    // If localLog shrinks, the game restarted. Reset the tracker!
    if (localLog.length < currentLastLength || localLog.length === 0) {
      currentLastLength = 0;
      lastLogLength.current = 0;
      if (localLog.length === 0) {
         setTurnLogs([]);
         return;
      }
    }

    let shouldWipe = false;
    if (gameState.phase === 'START_OF_TURN' && previousTurnPhase.current !== 'START_OF_TURN') {
        shouldWipe = true;
    }
    previousTurnPhase.current = gameState.phase;

    if (localLog.length > currentLastLength) {
      const newLogs = localLog.slice(currentLastLength);
      lastLogLength.current = localLog.length;
      
      const parsedLogs = newLogs.map(log => {
        const lowerLog = log.toLowerCase();
        if (lowerLog.includes('allow') || lowerLog.includes('accept')) Sounds.success();
        else if (lowerLog.includes('block')) Sounds.thud();
        else if (lowerLog.includes('challeng') || lowerLog.includes('reveal')) Sounds.alert();
        else if (lowerLog.includes('assassinate')) Sounds.assassinate();
        else if (lowerLog.includes('coup')) Sounds.coup();
        else if (lowerLog.includes('exchange')) Sounds.shuffle();
        else if (lowerLog.includes('income')) Sounds.income();
        else if (lowerLog.includes('foreign aid')) Sounds.foreignAid();
        else if (lowerLog.includes('tax')) Sounds.duke();
        else if (lowerLog.includes('steal')) Sounds.steal();

        return { id: logIdCounter.current++, text: parseLogToSnippet(log), original: log };
      });
      
      setTurnLogs(prev => {
        if (shouldWipe) return [];
        let current = currentLastLength === 0 ? [] : [...prev];
        for (const pLog of parsedLogs) {
          if (pLog.original.includes('decided to')) {
            current = [pLog];
          } else {
            current.push(pLog);
          }
        }
        return current;
      });
    } else if (shouldWipe) {
        setTurnLogs([]);
    }
  }, [localLog, gameState]);

  const connect = () => {
    setAppState('connecting');
    const savedSessionId = localStorage.getItem('coup_session_id');
    const url = savedSessionId ? `${WS_URL}?session_id=${savedSessionId}` : WS_URL;
    ws.current = new WebSocket(url);
    ws.current.onopen = () => {
      setError(null);
    };
    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'session_id') {
        localStorage.setItem('coup_session_id', data.session_id);
      } else if (data.type === 'lobby_state') {
        setAppState('lobby');
        setGameState(null);
        setLocalLog([]);
      } else if (data.type === 'state_update') {
        setAppState('playing');
        setGameState(data.data);
        
        // Backend now returns full contextual log list every time
        setLocalLog(data.data.log as string[]);
      } else if (data.type === 'error') {
        setError(data.message);
      }
    };
    ws.current.onclose = () => {
      setAppState('connecting');
      setTimeout(() => {
        if (ws.current?.readyState === WebSocket.CLOSED) {
          connect();
        }
      }, 2000);
    };
  };

  const startGame = () => {
    initAudio(); // Required to bypass browser auto-play policy
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ type: 'start_game', bot_count: botCount }));
    }
  };

  const restartGame = () => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ type: 'start_game', bot_count: botCount }));
    }
  };

  const handleAction = (actionId: number) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ type: 'action', action_id: actionId }));
    }
  };

  if (error) {
    return (
      <div className="modal-overlay">
        <div className="modal-content">
          <h1 style={{color: 'red'}}>Error</h1>
          <p>{error}</p>
          <button onClick={connect}>Reconnect</button>
        </div>
      </div>
    );
  }

  if (appState === 'connecting') {
    return (
      <div className="modal-overlay">
        <div className="modal-content">
          <h1>Connecting...</h1>
          <p>Waking up the AI server on Hugging Face 🚀</p>
        </div>
      </div>
    );
  }

  if (appState === 'lobby') {
    return (
      <>
        <div style={{ position: 'absolute', top: '10px', left: '10px', zIndex: 2002, display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <button 
            onClick={toggleMute} 
            onMouseEnter={() => Sounds.hover()}
            style={{
              background: isMuted ? '#6c757d' : '#06d6a0', 
              color: 'white', 
              border: '2px solid #fff', 
              padding: '5px 10px', 
              borderRadius: '4px',
              cursor: 'pointer',
              fontFamily: 'VT323, monospace',
              fontSize: '1.2rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '5px'
            }}
          >
            {isMuted ? '🔇 Audio Off' : '🔊 Audio On'}
          </button>
          <button 
            onClick={() => setShowFullRules(true)} 
            onMouseEnter={() => Sounds.hover()}
            style={{
              background: '#ffb703', 
              color: '#000', 
              border: '2px solid #fff', 
              padding: '5px 10px', 
              borderRadius: '4px',
              cursor: 'pointer',
              fontFamily: 'VT323, monospace',
              fontSize: '1.2rem'
            }}
          >
            Official Coup Rules
          </button>
        </div>
        <div className="modal-overlay">
          <div className="modal-content">
            <h1>Coup RL</h1>
            <p>Select number of AI opponents:</p>
            <div style={{ display: 'flex', gap: '10px', justifyContent: 'center', margin: '20px 0' }}>
              {[2, 3, 4, 5].map(n => (
                <button 
                  key={n} 
                  onClick={() => setBotCount(n)}
                  onMouseEnter={() => Sounds.hover()}
                  style={{ backgroundColor: botCount === n ? 'var(--accent)' : '#2b2d42' }}
                >
                  {n} Bots
                </button>
              ))}
            </div>
            <button onClick={startGame} onMouseEnter={() => Sounds.hover()} style={{ fontSize: '1.5rem', padding: '15px 30px' }}>Deal Cards</button>
          </div>
        </div>

        {showFullRules && (
          <div className="modal-overlay" style={{ zIndex: 3000 }} onClick={() => setShowFullRules(false)}>
            <div className="modal-content" style={{ width: '90%', height: '90%', padding: '40px 10px 10px 10px', display: 'flex', flexDirection: 'column' }} onClick={e => e.stopPropagation()}>
              <button onClick={() => setShowFullRules(false)} style={{ position: 'absolute', top: 10, right: 10, zIndex: 3001, background: '#ef476f', color: '#fff', border: 'none', borderRadius: '4px', padding: '5px 10px', cursor: 'pointer', fontFamily: 'VT323' }}>Close</button>
              <iframe src="/rules_complete.pdf" width="100%" height="100%" style={{ border: 'none', borderRadius: '8px', flexGrow: 1 }} />
            </div>
          </div>
        )}
      </>
    );
  }

  if (!gameState) return null;

  // Split players into Opponents and You
  const you = gameState.players.find(p => p.id === 0);
  const opponents = gameState.players.filter(p => p.id !== 0);

  const resetGame = () => {
    localStorage.removeItem('coup_session_id');
    if (ws.current) {
      ws.current.close();
    }
    setTimeout(connect, 100);
  };

  return (
    <div className="main-wrapper">
      <div className="app-container">
        
        <div style={{ position: 'absolute', top: '10px', left: '10px', zIndex: 2002, display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <button 
            onClick={resetGame} 
            onMouseEnter={() => Sounds.hover()}
            style={{
              background: '#ef476f', 
              color: 'white', 
              border: '2px solid #fff', 
              padding: '5px 10px', 
              borderRadius: '4px',
              cursor: 'pointer',
              fontFamily: 'VT323, monospace',
              fontSize: '1.2rem'
            }}
          >
            Reset Game
          </button>
          
          <button 
            onClick={toggleMute} 
            onMouseEnter={() => Sounds.hover()}
            style={{
              background: isMuted ? '#6c757d' : '#06d6a0', 
              color: 'white', 
              border: '2px solid #fff', 
              padding: '5px 10px', 
              borderRadius: '4px',
              cursor: 'pointer',
              fontFamily: 'VT323, monospace',
              fontSize: '1.2rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '5px'
            }}
          >
            {isMuted ? '🔇 Audio Off' : '🔊 Audio On'}
          </button>
          
          <button 
            onClick={() => setShowShortcut(true)} 
            onMouseEnter={() => Sounds.hover()}
            style={{
              background: '#ffb703', 
              color: '#000', 
              border: '2px solid #fff', 
              padding: '5px 10px', 
              borderRadius: '4px',
              cursor: 'pointer',
              fontFamily: 'VT323, monospace',
              fontSize: '1.2rem'
            }}
          >
            How to play
          </button>
        </div>

        {/* Opponents Row */}
        <div className="opponents-row">
          {opponents.map((p) => (
            <div key={p.id} className={`player-box ${gameState.active_player === p.id ? 'active' : ''} ${!p.alive ? 'dead' : ''}`}>
              
              {/* Hover History Tooltip */}
              <div className="history-tooltip-wrapper">
                <div className="history-tooltip">
                  <div style={{textAlign: 'center', marginBottom: '4px', fontWeight: 'bold', borderBottom: '1px solid #333'}}>{p.name}'s History:</div>
                  {localLog.filter(msg => msg.startsWith(p.name)).length > 0 ? 
                    localLog.filter(msg => msg.startsWith(p.name)).map((msg, i) => <div key={i} style={{marginBottom: '3px'}}>• {msg.replace(p.name + ' ', '')}</div>)
                    : <div style={{textAlign: 'center'}}>No actions yet</div>
                  }
                </div>
              </div>

              <div className="player-header">
                {p.name}
                <span className="player-cash">💰 {p.cash}</span>
              </div>
              
              <div className="cards-container">
                {p.cards.map((c, i) => (
                  <div key={i} className={`playing-card ${c.revealed ? 'dead role-' + getShortRoleName(c.role) : 'role-HIDDEN'}`}>
                    <div className="card-icon">{c.revealed ? getRoleIcon(c.role) : '❓'}</div>
                    <div className="card-text">{c.revealed ? getShortRoleName(c.role) : 'HIDDEN'}</div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Transient Turn Logs Ticker */}
        <div className="action-toast-container">
          {turnLogs.map((log) => (
            <div key={log.id} className="action-toast">{log.text}</div>
          ))}
        </div>

        {/* Center Action Area & Deck */}
        <div className="table-center-area">

          <div className="deck-tracker">
            <div className="deck-visual"></div>
            <div className="deck-count">{gameState.deck_count ?? 15}</div>
          </div>
          <div className="actions-container">
            <div className="actions-grid">
              {(() => {
                const hasChallenge = gameState.valid_actions.some(a => a.name === "Challenge");
                const isTarget = hasChallenge && gameState.valid_actions.some(a => a.name.includes("Block"));
                return gameState.valid_actions.map(action => {
                  let tooltipText = "";
                  if (isTarget) {
                    if (action.name === "Challenge") {
                      tooltipText = "If you challenge and lose, you still lose a card! However, if you survive, you can still block.";
                    } else if (action.name.includes("Block")) {
                      tooltipText = "Blocking directly implies you do not challenge their action.";
                    }
                  }
                  
                  return (
                    <button key={action.id} className="action-btn" onClick={() => handleAction(action.id)} onMouseEnter={() => Sounds.hover()}>
                      <span style={{ fontSize: '1.5rem' }}>{getActionIcon(action.name)}</span> {action.name}
                      {tooltipText && (
                        <span className="action-tooltip-wrapper">
                          <span className="info-icon">?</span>
                          <div className="action-tooltip-text">{tooltipText}</div>
                        </span>
                      )}
                    </button>
                  );
                });
              })()}
              {gameState.valid_actions.length === 0 && <p style={{fontFamily: 'VT323, monospace'}}>Waiting for opponents...</p>}
            </div>
          </div>
        </div>

        {/* My Area */}
        {you && (
          <div className="my-area">
            <div className={`player-box ${gameState.active_player === 0 ? 'active' : ''} ${!you.alive ? 'dead' : ''}`}>
               <div className="player-header">
                You
                <span className="player-cash">💰 {you.cash}</span>
              </div>
              
              <div className="cards-container">
                {you.cards.map((c, i) => (
                  <div key={i} className={`playing-card role-${getShortRoleName(c.role)} ${c.revealed ? 'dead' : ''}`}>
                    <div className="card-icon">{getRoleIcon(c.role)}</div>
                    <div className="card-text">{getShortRoleName(c.role)}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Sidebar Overlay Backdrop */}
      {isSidebarOpen && <div className="sidebar-backdrop" onClick={() => setIsSidebarOpen(false)}></div>}

      {/* Sidebar for Logs and Phase */}
      <div className={`sidebar ${isSidebarOpen ? '' : 'collapsed'}`}>
        <button className="sidebar-toggle" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
          {isSidebarOpen ? '▶' : '◀'}
        </button>
        <div className="sidebar-content">
          <div className="sidebar-pointer">
            ℹ️ Click the arrow tab on the left to minimize or expand this log panel.
          </div>
          <div className="sidebar-header">
            Current Phase<br/>
            <span style={{color: 'var(--accent)', fontSize: '1.2rem'}}>{gameState.phase}</span>
          </div>
          <div className="global-log">
            {localLog.map((msg, i) => (
              <div key={i}>{msg}</div>
            ))}
            <div ref={logEndRef} />
          </div>
        </div>
      </div>

      {/* Exchange Phase Modal */}
      {gameState.phase === 'EXCHANGE' && gameState.active_player === 0 && gameState.exchange_pool.length > 0 && (
        <div className="modal-overlay">
          <div className="modal-content" style={{ maxWidth: '600px' }}>
            <h2>Exchange Phase</h2>
            <p style={{fontFamily: 'VT323, monospace'}}>Select a card to <strong>RETURN</strong> to the deck.</p>
            <div className="cards-container" style={{ justifyContent: 'center', margin: '20px 0' }}>
              {gameState.exchange_pool.map((c) => {
                const canReturn = gameState.valid_actions.some(a => a.id === 34 + c.id);
                return (
                  <div 
                    key={c.id} 
                    className={`playing-card role-${getShortRoleName(c.role)} ${canReturn ? 'selectable' : ''}`}
                    onClick={() => canReturn && handleAction(34 + c.id)}
                  >
                    <div className="card-icon">{getRoleIcon(c.role)}</div>
                    <div className="card-text">{getShortRoleName(c.role)}</div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Game Over Modal */}
      {gameState.winner && (
        <div className="modal-overlay">
          <div className="modal-content">
            <h1 style={{ fontSize: '4rem', color: '#ffd166', marginBottom: '10px' }}>
              {gameState.winner === 'You' ? '🎉 YOU WIN! 🎉' : '💀 GAME OVER 💀'}
            </h1>
            <p style={{ fontSize: '2rem', margin: '10px 0', fontFamily: 'VT323, monospace', fontWeight: 'bold' }}>
              {gameState.winner === 'You' ? '1st Place!' : `${getOrdinal(gameState.player_0_placement)} Place`}
            </p>
            <p style={{ fontSize: '1.2rem', fontFamily: 'VT323, monospace', color: '#ccc' }}>
              Winner: {gameState.winner}
            </p>
            <button onClick={restartGame} onMouseEnter={() => Sounds.hover()} style={{ marginTop: '20px', fontSize: '1.5rem', padding: '15px 40px' }}>
              Play Again
            </button>
          </div>
        </div>
      )}

      {showShortcut && (
        <div className="modal-overlay" style={{ zIndex: 3000 }} onClick={() => setShowShortcut(false)}>
          <div className="modal-content" style={{ width: '90%', maxWidth: '800px', padding: '10px' }}>
            <img src="/rules_shortcut.jpg" alt="Rules Shortcut" style={{ width: '100%', borderRadius: '8px', display: 'block' }} />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
