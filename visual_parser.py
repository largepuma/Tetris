"""
Visual Parser — Stage 1 of the AI pipeline.

Converts raw board state into multi-channel tensor representation,
mimicking how a vision system would parse a game screen.

Channels:
  0: Board occupancy (binary grid)
  1: Current piece one-hot encoding (broadcast across spatial dims)
  2: Next piece one-hot encoding (broadcast across spatial dims)
  3: Column height map (normalized per-row encoding)
  4: Hole map (binary — marks cells that are holes)
"""

import numpy as np

from tetris_engine import Piece, TetrisGame, PIECE_SHAPES

# Fixed mapping from piece to integer index
PIECE_TO_IDX = {p: i for i, p in enumerate(Piece)}
NUM_PIECES = len(Piece)


class VisualParser:
    """Parses the Tetris game state into a tensor suitable for neural networks."""

    def __init__(self, board_height: int = 20, board_width: int = 10):
        self.board_height = board_height
        self.board_width = board_width
        # 1 (occupancy) + 7 (current piece one-hot) + 7 (next piece one-hot)
        # + 1 (height map) + 1 (hole map) = 17 channels
        self.num_channels = 1 + NUM_PIECES + NUM_PIECES + 1 + 1

    def parse(self, game: TetrisGame) -> np.ndarray:
        """
        Convert game state to a (num_channels, height, width) float32 tensor.
        """
        h, w = self.board_height, self.board_width
        state = np.zeros((self.num_channels, h, w), dtype=np.float32)

        # Channel 0: Board occupancy
        for r in range(h):
            for c in range(w):
                if game.board.grid[r][c] is not None:
                    state[0, r, c] = 1.0

        # Channels 1-7: Current piece one-hot (broadcast spatially)
        if game.current_piece is not None:
            idx = PIECE_TO_IDX[game.current_piece]
            state[1 + idx, :, :] = 1.0

        # Channels 8-14: Next piece one-hot (broadcast spatially)
        if game.next_piece is not None:
            idx = PIECE_TO_IDX[game.next_piece]
            state[1 + NUM_PIECES + idx, :, :] = 1.0

        # Channel 15: Column height map (normalized)
        ch_height = 1 + NUM_PIECES + NUM_PIECES
        heights = game.board.get_column_heights()
        for c in range(w):
            norm_h = heights[c] / h
            # Fill from bottom up to the height
            for r in range(h - heights[c], h):
                state[ch_height, r, c] = norm_h

        # Channel 16: Hole map
        ch_holes = ch_height + 1
        for c in range(w):
            found_block = False
            for r in range(h):
                if game.board.grid[r][c] is not None:
                    found_block = True
                elif found_block:
                    state[ch_holes, r, c] = 1.0

        return state

    def parse_compact(self, game: TetrisGame) -> np.ndarray:
        """
        Compact feature vector for simpler networks.
        Returns a 1D array of hand-crafted features.

        Features (per column):
          - column height (normalized)
          - height difference with left neighbor
          - holes in column
        Global features:
          - current piece index (one-hot, 7)
          - next piece index (one-hot, 7)
          - max height (normalized)
          - total holes
          - bumpiness (normalized)
          - aggregate height (normalized)
        """
        h, w = self.board_height, self.board_width
        heights = game.board.get_column_heights()

        features = []

        # Per-column features
        for c in range(w):
            features.append(heights[c] / h)
            if c > 0:
                features.append((heights[c] - heights[c - 1]) / h)
            else:
                features.append(0.0)

            # Count holes in this column
            col_holes = 0
            found = False
            for r in range(h):
                if game.board.grid[r][c] is not None:
                    found = True
                elif found:
                    col_holes += 1
            features.append(col_holes / h)

        # Current piece one-hot
        cur_oh = [0.0] * NUM_PIECES
        if game.current_piece is not None:
            cur_oh[PIECE_TO_IDX[game.current_piece]] = 1.0
        features.extend(cur_oh)

        # Next piece one-hot
        nxt_oh = [0.0] * NUM_PIECES
        if game.next_piece is not None:
            nxt_oh[PIECE_TO_IDX[game.next_piece]] = 1.0
        features.extend(nxt_oh)

        # Global features
        features.append(max(heights) / h)
        features.append(game.board.count_holes() / (h * w))
        features.append(game.board.bumpiness() / (h * w))
        features.append(game.board.aggregate_height() / (h * w))

        return np.array(features, dtype=np.float32)

    @property
    def compact_feature_size(self) -> int:
        """Size of the compact feature vector."""
        w = self.board_width
        return w * 3 + NUM_PIECES * 2 + 4
