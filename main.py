#!/usr/bin/env python3
"""
Tetris AI — Main entry point.

Architecture: Visual Parsing → RL Decision-Making → Action Mapping

Usage:
  python main.py train [--episodes N] [--mode compact|cnn] [--seed S]
  python main.py play  [--model PATH] [--mode compact|cnn] [--speed DELAY]
  python main.py demo  [--games N] [--seed S]

Modes:
  train  — Train the DQN agent from scratch
  play   — Watch a trained agent play with live board display
  demo   — Quick demo with a heuristic baseline (no PyTorch needed)
"""

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np

from tetris_engine import TetrisGame, PIECE_CHARS
from visual_parser import VisualParser
from action_mapper import ActionMapper


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def render_game(game: TetrisGame, action_desc: str = "", extra: str = ""):
    """Render the game board to terminal."""
    clear_screen()
    print("╔══════════════════════════════╗")
    print("║     TETRIS AI PLAYER         ║")
    print("║  Visual → RL → Action        ║")
    print("╚══════════════════════════════╝")
    print()
    print(f"  Score: {game.score:>8d}    Lines: {game.lines_cleared:>4d}")
    print(f"  Pieces: {game.pieces_placed:>6d}    Next: {game.next_piece.value if game.next_piece else '?'}")
    if action_desc:
        print(f"  Action: {action_desc}")
    print()

    # Board
    border = "  +" + "──" * game.board.width + "─+"
    print(border)
    for row in game.board.grid:
        cells = " ".join(_cell_char(c) for c in row)
        print(f"  │ {cells} │")
    print(border)

    if extra:
        print()
        print(extra)


def _cell_char(cell: Optional[str]) -> str:
    if cell is None:
        return "·"
    colors = {"I": "█", "O": "▓", "T": "▒", "S": "░", "Z": "▞", "J": "▚", "L": "▐"}
    return colors.get(cell, "■")


# ─── Training Mode ────────────────────────────────────────────────────

def cmd_train(args):
    """Train the DQN agent."""
    from trainer import Trainer

    print("=" * 60)
    print("  Tetris AI Training")
    print(f"  Mode: {args.mode} | Episodes: {args.episodes}")
    print(f"  LR: {args.lr} | Gamma: {args.gamma}")
    print("=" * 60)
    print()

    trainer = Trainer(
        mode=args.mode,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        save_dir=args.save_dir,
        device=args.device,
    )

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        trainer.agent.load(args.resume)

    stats = trainer.train(
        num_episodes=args.episodes,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        seed=args.seed,
    )

    # Print final summary
    if stats["episode_lines"]:
        last_n = min(100, len(stats["episode_lines"]))
        print(f"\nFinal {last_n} episodes:")
        print(f"  Avg lines:  {np.mean(stats['episode_lines'][-last_n:]):.1f}")
        print(f"  Avg score:  {np.mean(stats['episode_scores'][-last_n:]):.1f}")
        print(f"  Avg pieces: {np.mean(stats['episode_pieces'][-last_n:]):.1f}")
        print(f"  Max lines:  {max(stats['episode_lines'][-last_n:])}")


# ─── Play Mode (Watch Trained Agent) ──────────────────────────────────

def cmd_play(args):
    """Watch a trained agent play Tetris with live rendering."""
    from rl_agent import DQNAgent

    parser = VisualParser()
    mapper = ActionMapper()

    compact_size = parser.compact_feature_size
    in_channels = parser.num_channels

    agent = DQNAgent(
        action_space=mapper.action_space_size,
        mode=args.mode,
        in_channels=in_channels,
        compact_size=compact_size,
        device=args.device,
    )

    if args.model and os.path.exists(args.model):
        agent.load(args.model)
        print(f"Loaded model from {args.model}")
    else:
        print("No model specified or found. Agent will use random actions.")

    agent.epsilon = 0.0  # No exploration during play

    game = TetrisGame(seed=args.seed)

    while not game.game_over:
        if args.mode == "cnn":
            state = parser.parse(game)
        else:
            state = parser.parse_compact(game)

        valid_mask = mapper.get_valid_action_mask(game)
        action = agent.select_action(state, valid_mask, training=False)
        rotation, col = mapper.action_to_move(action)

        desc = f"rot={rotation} col={col} ({game.current_piece.value})"
        render_game(game, action_desc=desc)

        game.place_piece(rotation, col)
        time.sleep(args.speed)

    render_game(game, extra="  *** GAME OVER ***")
    print(f"\n  Final Score: {game.score}")
    print(f"  Lines Cleared: {game.lines_cleared}")
    print(f"  Pieces Placed: {game.pieces_placed}")


# ─── Demo Mode (Heuristic Baseline, No PyTorch) ──────────────────────

def cmd_demo(args):
    """Run games with a heuristic baseline to demonstrate the pipeline."""
    parser = VisualParser()
    mapper = ActionMapper()

    total_lines = []
    total_scores = []
    total_pieces = []

    for g in range(1, args.games + 1):
        game = TetrisGame(seed=(args.seed + g) if args.seed is not None else None)

        max_pieces = args.max_pieces
        while not game.game_over and game.pieces_placed < max_pieces:
            # Visual parsing stage — extract features
            state = parser.parse_compact(game)

            # Decision stage — heuristic evaluation of all placements
            placements = game.get_all_placements()
            if not placements:
                break

            best_score = float("-inf")
            best_rot, best_col = 0, 0

            for rotation, col, row, cells in placements:
                # Simulate placement
                sim_board = game.board.copy()
                char = PIECE_CHARS[game.current_piece]
                sim_board.place_cells(cells, char)
                lines = sim_board.count_complete_lines()
                sim_board.clear_lines()

                # Heuristic evaluation (mimics what the RL agent learns)
                score = (
                    -0.510066 * sim_board.aggregate_height()
                    + 0.760666 * lines
                    - 0.35663 * sim_board.count_holes()
                    - 0.184483 * sim_board.bumpiness()
                )

                if score > best_score:
                    best_score = score
                    best_rot, best_col = rotation, col

            # Action mapping stage
            if args.visual:
                desc = f"rot={best_rot} col={best_col} ({game.current_piece.value})"
                render_game(game, action_desc=desc)
                time.sleep(args.speed)

            game.place_piece(best_rot, best_col)

        total_lines.append(game.lines_cleared)
        total_scores.append(game.score)
        total_pieces.append(game.pieces_placed)

        if args.visual:
            render_game(game, extra="  *** GAME OVER ***")
            time.sleep(1.0)

        print(
            f"Game {g:4d} | "
            f"Lines: {game.lines_cleared:5d} | "
            f"Score: {game.score:8d} | "
            f"Pieces: {game.pieces_placed:5d}"
        )

    print("\n" + "=" * 50)
    print(f"Results over {args.games} games:")
    print(f"  Avg Lines:  {np.mean(total_lines):8.1f}  (max: {max(total_lines)})")
    print(f"  Avg Score:  {np.mean(total_scores):8.1f}  (max: {max(total_scores)})")
    print(f"  Avg Pieces: {np.mean(total_pieces):8.1f}  (max: {max(total_pieces)})")


# ─── CLI ──────────────────────────────────────────────────────────────

def main():
    top = argparse.ArgumentParser(
        description="Tetris AI: Visual Parsing → RL Decision → Action Mapping"
    )
    sub = top.add_subparsers(dest="command")

    # Train
    p_train = sub.add_parser("train", help="Train the DQN agent")
    p_train.add_argument("--episodes", type=int, default=5000)
    p_train.add_argument("--mode", choices=["compact", "cnn"], default="compact")
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--gamma", type=float, default=0.99)
    p_train.add_argument("--epsilon-start", type=float, default=1.0)
    p_train.add_argument("--epsilon-end", type=float, default=0.01)
    p_train.add_argument("--epsilon-decay", type=float, default=0.9995)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--buffer-capacity", type=int, default=50000)
    p_train.add_argument("--save-dir", default="checkpoints")
    p_train.add_argument("--log-interval", type=int, default=100)
    p_train.add_argument("--save-interval", type=int, default=500)
    p_train.add_argument("--resume", type=str, default=None)
    p_train.add_argument("--seed", type=int, default=None)
    p_train.add_argument("--device", type=str, default=None)

    # Play
    p_play = sub.add_parser("play", help="Watch a trained agent play")
    p_play.add_argument("--model", type=str, default="checkpoints/best_model.pt")
    p_play.add_argument("--mode", choices=["compact", "cnn"], default="compact")
    p_play.add_argument("--speed", type=float, default=0.15)
    p_play.add_argument("--seed", type=int, default=None)
    p_play.add_argument("--device", type=str, default=None)

    # Demo
    p_demo = sub.add_parser("demo", help="Heuristic demo (no PyTorch needed)")
    p_demo.add_argument("--games", type=int, default=10)
    p_demo.add_argument("--seed", type=int, default=42)
    p_demo.add_argument("--visual", action="store_true", help="Show live board")
    p_demo.add_argument("--speed", type=float, default=0.05)
    p_demo.add_argument("--max-pieces", type=int, default=500)

    args = top.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "play":
        cmd_play(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        top.print_help()


if __name__ == "__main__":
    main()
