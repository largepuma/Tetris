# AGENT.md — Tetris AI Codebase Guide

This file provides context for AI coding agents working in this repository.

## Project Overview

A Tetris AI implemented in pure Python with a three-stage pipeline:

```
Visual Parser → RL Agent (DQN) → Action Mapper
```

The agent learns to place Tetris pieces optimally using Deep Q-Learning. A heuristic demo mode requires no ML framework.

## Dependency Management

This project uses **uv** for dependency management.

- `pyproject.toml` — canonical dependency spec
- `uv.lock` — pinned lockfile (commit this)
- `.venv/` — local virtual environment (gitignored)

**Always use uv commands:**

```bash
uv sync                          # install/update all deps
uv add <package>                 # add a dependency
uv run main.py <subcommand>      # run without activating venv
uv run pytest                    # run tests
```

Do NOT edit `requirements.txt` — it is superseded by `pyproject.toml`.

## Key Files

| File | Role |
|------|------|
| `tetris_engine.py` | Self-contained game engine — `TetrisGame`, `Board`, `Piece`, scoring, 7-bag RNG |
| `visual_parser.py` | Stage 1: board state → CNN tensor (17ch × 20 × 10) or compact 48-dim vector |
| `rl_agent.py` | Stage 2: Dueling Double DQN, experience replay buffer, ε-greedy policy |
| `action_mapper.py` | Stage 3: maps discrete action index ↔ (rotation, column); generates valid action masks |
| `trainer.py` | Training loop with reward shaping, checkpoint saving, stats tracking |
| `main.py` | CLI entry point — `train`, `play`, `demo` subcommands |
| `pyproject.toml` | Project metadata and dependencies |

## Architecture Details

### Visual Parser (`visual_parser.py`)
- **CNN mode**: returns `(17, 20, 10)` float32 tensor — channels include board occupancy, 7 piece one-hots, height map, hole map
- **Compact mode**: returns 48-dim float32 vector — per-column heights/holes, piece encoding (7-dim one-hot + next piece), global metrics

### RL Agent (`rl_agent.py`)
- Dueling DQN network: shared backbone → separate Value and Advantage streams
- Double DQN target: `r + γ * Q_target(s', argmax_a Q_online(s', a))`
- Replay buffer: ring-buffer of 50K transitions
- Invalid action masking: logits for illegal placements set to `-inf` before softmax/argmax

### Action Space
- 40 discrete actions = 4 rotations × 10 columns
- `ActionMapper.action_to_move(i)` → `(rotation, col)`
- `ActionMapper.get_valid_action_mask(game)` → bool array of shape `(40,)`

### Reward Shaping (in `trainer.py`)
- Lines cleared: large positive reward (scaled by 1/4/9/16 for 1/2/3/4 lines)
- Holes created: negative penalty
- Height increase: negative penalty
- Game over: large negative reward

## Running the Code

```bash
# Quick sanity check (no GPU, no PyTorch)
uv run main.py demo --games 3

# Full training run
uv run main.py train --episodes 5000 --mode compact --save-dir checkpoints

# Watch a trained agent
uv run main.py play --model checkpoints/best_model.pt --speed 0.2
```

## Extending the Project

- **New state features**: modify `VisualParser` in `visual_parser.py`; update `compact_feature_size` or `num_channels` accordingly
- **New network architecture**: subclass or replace the network in `rl_agent.py`; the `DQNAgent` interface expects `select_action(state, valid_mask, training)` and `update(batch)`
- **New reward signals**: edit `Trainer._compute_reward()` in `trainer.py`
- **New action space**: update `ActionMapper`; `action_space_size` must stay consistent with the DQN output dimension

## Coding Conventions

- Pure Python 3.10+, typed with standard `typing` module
- NumPy for board operations; PyTorch only in `rl_agent.py` and `trainer.py`
- No external game libraries — `tetris_engine.py` is self-contained
- `demo` mode must work without PyTorch (lazy import in `cmd_train` / `cmd_play`)
