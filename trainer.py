"""
Training loop for the Tetris RL agent.

Orchestrates the full pipeline:
  Visual Parser → RL Agent → Action Mapper → Game Engine → Reward
"""

import os
import time
from typing import Optional

import numpy as np

from tetris_engine import TetrisGame
from visual_parser import VisualParser
from action_mapper import ActionMapper
from rl_agent import DQNAgent, Transition, TORCH_AVAILABLE


class RewardShaper:
    """
    Shapes rewards to provide denser learning signals.

    Reward components:
      - Lines cleared: primary positive signal (scaled by count)
      - Holes created: negative penalty
      - Height increase: negative penalty
      - Survival bonus: small positive for staying alive
      - Game over: large negative penalty
    """

    def __init__(
        self,
        line_clear_rewards: Optional[dict] = None,
        hole_penalty: float = -0.5,
        height_penalty: float = -0.01,
        survival_bonus: float = 0.05,
        game_over_penalty: float = -5.0,
        bumpiness_penalty: float = -0.01,
    ):
        self.line_clear_rewards = line_clear_rewards or {
            0: 0.0,
            1: 1.0,
            2: 3.0,
            3: 5.0,
            4: 10.0,
        }
        self.hole_penalty = hole_penalty
        self.height_penalty = height_penalty
        self.survival_bonus = survival_bonus
        self.game_over_penalty = game_over_penalty
        self.bumpiness_penalty = bumpiness_penalty

    def compute(
        self,
        lines_cleared: int,
        holes_before: int,
        holes_after: int,
        height_before: int,
        height_after: int,
        bumpiness_before: int,
        bumpiness_after: int,
        game_over: bool,
    ) -> float:
        reward = 0.0

        # Line clear reward
        reward += self.line_clear_rewards.get(lines_cleared, 0.0)

        # Hole change penalty
        hole_delta = holes_after - holes_before
        if hole_delta > 0:
            reward += hole_delta * self.hole_penalty

        # Height change penalty
        height_delta = height_after - height_before
        if height_delta > 0:
            reward += height_delta * self.height_penalty

        # Bumpiness change
        bump_delta = bumpiness_after - bumpiness_before
        if bump_delta > 0:
            reward += bump_delta * self.bumpiness_penalty

        # Survival bonus
        if not game_over:
            reward += self.survival_bonus

        # Game over penalty
        if game_over:
            reward += self.game_over_penalty

        return reward


class Trainer:
    """Trains the DQN agent to play Tetris."""

    def __init__(
        self,
        mode: str = "compact",
        board_width: int = 10,
        board_height: int = 20,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.9995,
        batch_size: int = 64,
        buffer_capacity: int = 50000,
        target_update_freq: int = 500,
        save_dir: str = "checkpoints",
        device: Optional[str] = None,
    ):
        self.board_width = board_width
        self.board_height = board_height
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.parser = VisualParser(board_height, board_width)
        self.mapper = ActionMapper(board_width)
        self.reward_shaper = RewardShaper()

        compact_size = self.parser.compact_feature_size
        in_channels = self.parser.num_channels

        self.agent = DQNAgent(
            action_space=self.mapper.action_space_size,
            mode=mode,
            in_channels=in_channels,
            compact_size=compact_size,
            lr=lr,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            target_update_freq=target_update_freq,
            batch_size=batch_size,
            buffer_capacity=buffer_capacity,
            device=device,
        )
        self.mode = mode

    def _get_state(self, game: TetrisGame) -> np.ndarray:
        if self.mode == "cnn":
            return self.parser.parse(game)
        return self.parser.parse_compact(game)

    def train(
        self,
        num_episodes: int = 5000,
        log_interval: int = 100,
        save_interval: int = 500,
        seed: Optional[int] = None,
    ) -> dict:
        """
        Run the training loop.

        Returns a dict with training statistics.
        """
        stats = {
            "episode_scores": [],
            "episode_lines": [],
            "episode_pieces": [],
            "losses": [],
            "epsilons": [],
        }

        best_avg_lines = 0.0
        start_time = time.time()

        for episode in range(1, num_episodes + 1):
            ep_seed = seed + episode if seed is not None else None
            game = TetrisGame(self.board_width, self.board_height, seed=ep_seed)

            state = self._get_state(game)
            valid_mask = self.mapper.get_valid_action_mask(game)
            total_reward = 0.0

            while not game.game_over:
                # Measure board state before action
                holes_before = game.board.count_holes()
                height_before = game.board.aggregate_height()
                bump_before = game.board.bumpiness()

                # Agent selects action
                action = self.agent.select_action(state, valid_mask, training=True)
                rotation, col = self.mapper.action_to_move(action)

                # Execute action
                lines_cleared, game_over = game.place_piece(rotation, col)

                # Measure board state after action
                holes_after = game.board.count_holes()
                height_after = game.board.aggregate_height()
                bump_after = game.board.bumpiness()

                # Compute shaped reward
                reward = self.reward_shaper.compute(
                    lines_cleared,
                    holes_before, holes_after,
                    height_before, height_after,
                    bump_before, bump_after,
                    game_over,
                )
                total_reward += reward

                # Get next state
                if not game_over:
                    next_state = self._get_state(game)
                    next_valid_mask = self.mapper.get_valid_action_mask(game)
                else:
                    next_state = None
                    next_valid_mask = None

                # Store transition
                self.agent.replay_buffer.push(
                    Transition(
                        state=state,
                        action=action,
                        reward=reward,
                        next_state=next_state,
                        done=game_over,
                        valid_actions_mask=valid_mask,
                        next_valid_actions_mask=next_valid_mask,
                    )
                )

                # Train
                loss = self.agent.train_step()
                if loss is not None:
                    stats["losses"].append(loss)

                if not game_over:
                    state = next_state
                    valid_mask = next_valid_mask

            # Episode stats
            stats["episode_scores"].append(game.score)
            stats["episode_lines"].append(game.lines_cleared)
            stats["episode_pieces"].append(game.pieces_placed)
            stats["epsilons"].append(self.agent.epsilon)

            # Logging
            if episode % log_interval == 0:
                recent_lines = stats["episode_lines"][-log_interval:]
                recent_scores = stats["episode_scores"][-log_interval:]
                recent_pieces = stats["episode_pieces"][-log_interval:]
                avg_lines = np.mean(recent_lines)
                avg_score = np.mean(recent_scores)
                avg_pieces = np.mean(recent_pieces)
                elapsed = time.time() - start_time

                print(
                    f"Episode {episode:5d} | "
                    f"Avg Lines: {avg_lines:6.1f} | "
                    f"Avg Score: {avg_score:8.1f} | "
                    f"Avg Pieces: {avg_pieces:6.1f} | "
                    f"Epsilon: {self.agent.epsilon:.4f} | "
                    f"Buffer: {len(self.agent.replay_buffer):6d} | "
                    f"Time: {elapsed:6.1f}s"
                )

                if avg_lines > best_avg_lines:
                    best_avg_lines = avg_lines
                    self.agent.save(os.path.join(self.save_dir, "best_model.pt"))
                    print(f"  → New best average lines: {best_avg_lines:.1f}")

            # Periodic save
            if episode % save_interval == 0:
                self.agent.save(
                    os.path.join(self.save_dir, f"checkpoint_{episode}.pt")
                )

        # Final save
        self.agent.save(os.path.join(self.save_dir, "final_model.pt"))
        print(f"\nTraining complete. Best avg lines: {best_avg_lines:.1f}")
        return stats
