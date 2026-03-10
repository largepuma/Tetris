"""
Action Mapper — Stage 3 of the AI pipeline.

Maps discrete action indices from the RL agent to concrete
(rotation, column) game actions, and vice versa.

The action space covers all valid (rotation, column) combinations
for all piece types. Since different pieces have different rotation
counts and column ranges, we use a universal action encoding that
covers the maximum possibilities.

Action encoding:
  action_index = rotation * board_width + column_offset
  where column_offset = column - min_valid_column

For a 10-wide board with max 4 rotations:
  Total action space = 4 * 10 = 40
"""

from tetris_engine import Piece, TetrisGame, PIECE_SHAPES


class ActionMapper:
    """Bidirectional mapping between action indices and game moves."""

    def __init__(self, board_width: int = 10, max_rotations: int = 4):
        self.board_width = board_width
        self.max_rotations = max_rotations
        self.action_space_size = max_rotations * board_width

    def action_to_move(self, action: int) -> tuple[int, int]:
        """Convert action index to (rotation, column)."""
        rotation = action // self.board_width
        column = action % self.board_width
        return rotation, column

    def move_to_action(self, rotation: int, column: int) -> int:
        """Convert (rotation, column) to action index."""
        return rotation * self.board_width + column

    def get_valid_actions(self, game: TetrisGame) -> list[int]:
        """Get list of valid action indices for the current game state."""
        placements = game.get_all_placements()
        valid = []
        for rotation, col, row, cells in placements:
            if 0 <= col < self.board_width and rotation < self.max_rotations:
                action = self.move_to_action(rotation, col)
                valid.append(action)
        return valid

    def get_valid_action_mask(self, game: TetrisGame) -> list[bool]:
        """
        Get a boolean mask of size action_space_size.
        True = valid action, False = invalid.
        """
        mask = [False] * self.action_space_size
        for action in self.get_valid_actions(game):
            mask[action] = True
        return mask

    def describe_action(self, action: int) -> str:
        """Human-readable description of an action."""
        rotation, column = self.action_to_move(action)
        return f"rotation={rotation}, column={column}"
