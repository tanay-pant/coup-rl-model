# Coup RL 🃏🤖

Play the classic deduction card game **Coup** against autonomous AI agents trained using Reinforcement Learning!

**🎮 Play it live here:** [coup-rl.vercel.app](https://coup-rl.vercel.app)

<img width="877" height="739" alt="Screenshot 2026-07-14 at 9 59 36 PM" src="https://github.com/user-attachments/assets/45a6272c-3fd6-4351-adb9-3cda2cdece42" />

## What is Coup RL?
This project is a full-stack implementation of the card game Coup, where human players can play directly against AI bots in a web browser. The bots were trained using deep reinforcement learning to bluff, deduce, and strategize.

The architecture consists of:
- A custom Reinforcement Learning environment where bots learned how to play Coup.
- A **Python backend** running a FastAPI WebSocket server on Hugging Face Spaces that hosts the trained RL models and manages game state logic.
- A **React frontend** hosted on Vercel that provides a beautiful, interactive casino-style web interface.

## Codebase Structure
The repository is split into several key directories:
- `frontend/`: The Vite + React web application. Handles the UI, state rendering, audio, and WebSocket communication with the backend.
- `backend/`: The FastAPI WebSocket server. Acts as the game engine, handling human inputs and querying the RL model for bot actions.
- `envs/`: Contains the custom Gym/PettingZoo environment implementation for Coup that the AI was trained on.
- `agents/` & `scripts/`: Training scripts, agent architectures, and data analysis utilities used during the reinforcement learning phase.
- `checkpoints_*/`: Saved weights for the trained neural networks.

## 🧠 How the Model Functions

The bots in this project are powered by a hybrid architecture inspired by DeepMind's AlphaZero, combining deep reinforcement learning with Monte Carlo Tree Search (MCTS) to navigate the hidden information inherent to Coup.

### 1. The Neural Network Prior (LSTM)
At the core of the AI is an LSTM-based neural network trained via self-play using RLlib. This network evaluates the current board state and provides two critical outputs:
- **Prior Probabilities (P):** A probability distribution over all legal moves, representing the network's intuition on what a good move looks like.
- **Value (V):** An evaluation of who is currently winning the game (from -1.0 to 1.0).

### 2. Information Determinization
Because Coup relies HEAVILY on hidden information (face-down cards), a standard search tree cannot simulate the future perfectly. Before the MCTS begins its search, it uses a technique called **Determinization**. The engine analyzes all publicly revealed cards, cross-references them with the roles each opponent is actively claiming (e.g., if a player stole coins, they claim a Captain), and deals out a simulated, consistent hand of cards to the opponents.

### 3. PUCT Monte Carlo Tree Search
Using the determinized board state, the bot simulates the game forward hundreds of times per turn. It balances exploring new strategies and exploiting known good paths using the PUCT (Predictor Upper Confidence Bound applied to Trees) algorithm. 
- The MCTS uses a standard AlphaZero exploration constant (`c_puct = 1.25`) to prevent over-exploring risky, low-probability branches (like random challenges). This ensures the bot relies on the actual proven win-rates of the branches it explores, making it highly stable. To keep it snappy for the user, searches are limited to 0.6 seconds.

### 4. Depth Discount Penalty
To solve a notorious problem in MCTS known as "Resignation Behavior" (where a bot cannot mathematically distinguish between an instant loss and a slow loss 10 turns away, leading to suicidal moves), the backpropagation incorporates a depth discount factor (`gamma = 0.99`). 
- This mathematically guarantees that the bot will prioritize a slow loss over an instant loss, encouraging it to take risky bluffs to survive. 
- Conversely, when the bot is winning, it forces the AI to ruthlessly close out the game as fast as possible rather than stalling.

## 🚀 Running Locally

### Backend
1. `pip install -r requirements.txt`
2. `cd backend`
3. `uvicorn app:app --host 0.0.0.0 --port 7860 --reload`

### Frontend
1. `cd frontend`
2. `npm install`
3. `npm run dev`
