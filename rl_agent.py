"""
RL Agent — Stage 2 of the AI pipeline.

DQN (Deep Q-Network) agent with:
  - CNN backbone for processing the visual state tensor
  - Dueling architecture (separate value and advantage streams)
  - Experience replay buffer
  - Target network for stable training
  - Epsilon-greedy exploration with decay
  - Invalid action masking
"""

import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ─── Experience Replay ───────────────────────────────────────────────

@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: Optional[np.ndarray]
    done: bool
    valid_actions_mask: list[bool]
    next_valid_actions_mask: Optional[list[bool]]


class ReplayBuffer:
    """Fixed-size circular buffer for experience replay."""

    def __init__(self, capacity: int = 50000):
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


# ─── Neural Network ──────────────────────────────────────────────────

if TORCH_AVAILABLE:

    class DuelingDQN(nn.Module):
        """
        Dueling DQN with CNN backbone for spatial board features.

        Architecture:
          Input: (batch, channels, 20, 10) visual state
          → 3 conv layers with batch norm
          → flatten
          → split into value stream and advantage stream
          → combine: Q(s,a) = V(s) + A(s,a) - mean(A)
        """

        def __init__(self, in_channels: int, action_space: int):
            super().__init__()

            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )

            # Calculate flattened size: (64, 10, 5) after stride-2 on (20,10)
            self._flat_size = 64 * 10 * 5

            # Value stream
            self.value_stream = nn.Sequential(
                nn.Linear(self._flat_size, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

            # Advantage stream
            self.advantage_stream = nn.Sequential(
                nn.Linear(self._flat_size, 256),
                nn.ReLU(),
                nn.Linear(256, action_space),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.features(x)
            features = features.view(features.size(0), -1)

            value = self.value_stream(features)
            advantage = self.advantage_stream(features)

            # Dueling combination
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
            return q_values

    class CompactDQN(nn.Module):
        """Simpler MLP-based DQN for the compact feature vector."""

        def __init__(self, input_size: int, action_space: int):
            super().__init__()

            self.net = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
            )

            self.value_stream = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )

            self.advantage_stream = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, action_space),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.net(x)
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            return value + advantage - advantage.mean(dim=1, keepdim=True)


# ─── DQN Agent ────────────────────────────────────────────────────────

class DQNAgent:
    """
    DQN Agent that combines all three pipeline stages.

    Supports two modes:
      - 'cnn': Uses DuelingDQN on the full visual state tensor
      - 'compact': Uses CompactDQN on the compact feature vector
    """

    def __init__(
        self,
        action_space: int = 40,
        mode: str = "compact",
        in_channels: int = 17,
        compact_size: int = 48,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.9995,
        target_update_freq: int = 500,
        batch_size: int = 64,
        buffer_capacity: int = 50000,
        device: Optional[str] = None,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is required for the DQN agent. "
                "Install with: pip install torch"
            )

        self.action_space = action_space
        self.mode = mode
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.train_steps = 0

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Create networks
        if mode == "cnn":
            self.policy_net = DuelingDQN(in_channels, action_space).to(self.device)
            self.target_net = DuelingDQN(in_channels, action_space).to(self.device)
        else:
            self.policy_net = CompactDQN(compact_size, action_space).to(self.device)
            self.target_net = CompactDQN(compact_size, action_space).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def select_action(
        self, state: np.ndarray, valid_mask: list[bool], training: bool = True
    ) -> int:
        """
        Select an action using epsilon-greedy with invalid action masking.
        """
        valid_actions = [i for i, v in enumerate(valid_mask) if v]
        if not valid_actions:
            return 0

        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t).squeeze(0)

            # Mask invalid actions with very negative values
            mask_t = torch.BoolTensor(valid_mask).to(self.device)
            q_values[~mask_t] = float("-inf")

            return q_values.argmax().item()

    def train_step(self) -> Optional[float]:
        """
        Perform one training step on a batch from the replay buffer.
        Returns the loss value, or None if buffer too small.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(np.array([t.state for t in batch])).to(self.device)
        actions = torch.LongTensor([t.action for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in batch]).to(self.device)
        dones = torch.BoolTensor([t.done for t in batch]).to(self.device)

        # Current Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (Double DQN)
        with torch.no_grad():
            next_states_list = []
            next_masks_list = []
            for t in batch:
                if t.next_state is not None:
                    next_states_list.append(t.next_state)
                    next_masks_list.append(t.next_valid_actions_mask)
                else:
                    next_states_list.append(t.state)  # placeholder
                    next_masks_list.append([False] * self.action_space)

            next_states = torch.FloatTensor(np.array(next_states_list)).to(self.device)

            # Double DQN: use policy net to select action, target net to evaluate
            next_q_policy = self.policy_net(next_states)
            next_q_target = self.target_net(next_states)

            # Mask invalid actions
            for i, mask in enumerate(next_masks_list):
                mask_t = torch.BoolTensor(mask).to(self.device)
                next_q_policy[i][~mask_t] = float("-inf")

            best_actions = next_q_policy.argmax(dim=1)
            next_q = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            next_q[dones] = 0.0

            target_q = rewards + self.gamma * next_q

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.train_steps += 1

        # Update target network
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save(self, path: str):
        """Save model weights and training state."""
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "train_steps": self.train_steps,
                "mode": self.mode,
            },
            path,
        )

    def load(self, path: str):
        """Load model weights and training state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.train_steps = checkpoint["train_steps"]
