# Tetris AI Player

An AI that plays Tetris using a pseudo end-to-end architecture:

**Visual Parsing → RL Decision-Making → Action Mapping**

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────┐
│  Visual Parser   │────▶│   RL Agent       │────▶│ Action Mapper │
│                 │     │  (DQN)           │     │               │
│ Board grid      │     │ Dueling DQN      │     │ Action index  │
│ → Multi-channel │     │ Double DQN       │     │ → (rotation,  │
│   tensor        │     │ Experience Replay│     │    column)    │
│ → Compact       │     │ ε-greedy explore │     │               │
│   features      │     │ Invalid masking  │     │ Valid action  │
│                 │     │                  │     │ mask          │
└─────────────────┘     └──────────────────┘     └───────────────┘
         ▲                                              │
         │              ┌──────────────────┐            │
         └──────────────│  Tetris Engine   │◀───────────┘
                        │  (Game State)    │
                        └──────────────────┘
```

### Stage 1: Visual Parser (`visual_parser.py`)
Converts raw board state into neural-network-ready representations:
- **CNN mode**: 17-channel spatial tensor (20×10) — board occupancy, piece one-hot encoding, height map, hole map
- **Compact mode**: 48-dim feature vector — per-column heights/holes, piece encoding, global metrics

### Stage 2: RL Agent (`rl_agent.py`)
Deep Q-Network with modern enhancements:
- **Dueling architecture**: Separates state value from action advantages
- **Double DQN**: Reduces Q-value overestimation
- **Experience replay**: 50K transition buffer
- **Invalid action masking**: Only considers legal placements

### Stage 3: Action Mapper (`action_mapper.py`)
Maps between the agent's discrete action space and game mechanics:
- 40 actions = 4 rotations × 10 columns
- Provides valid action masks per game state
- Bidirectional mapping (action ↔ game move)

## Usage

### Demo (no PyTorch required)
```bash
python main.py demo --games 10 --visual
```

### Train the RL Agent
```bash
# Compact mode (faster training)
python main.py train --episodes 5000 --mode compact

# CNN mode (learns spatial features)
python main.py train --episodes 10000 --mode cnn

# Resume training
python main.py train --resume checkpoints/checkpoint_2000.pt
```

### Watch Trained Agent Play
```bash
python main.py play --model checkpoints/best_model.pt --speed 0.2
```

## Files

| File | Description |
|------|-------------|
| `tetris_engine.py` | Game engine — board, pieces, scoring, 7-bag randomizer |
| `visual_parser.py` | Stage 1 — state → tensor/features |
| `rl_agent.py` | Stage 2 — DQN agent with replay buffer |
| `action_mapper.py` | Stage 3 — action index ↔ game move |
| `trainer.py` | Training loop with reward shaping |
| `main.py` | CLI entry point (train / play / demo) |

## Requirements

- Python 3.10+
- NumPy
- PyTorch (for training/playing with RL agent; demo mode works without it)
