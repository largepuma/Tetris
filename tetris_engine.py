"""Tetris game engine with board management and piece logic."""

import random
from copy import deepcopy
from enum import Enum
from typing import Optional


class Piece(Enum):
    I = "I"
    O = "O"
    T = "T"
    S = "S"
    Z = "Z"
    J = "J"
    L = "L"


# Each piece defined as list of rotations, each rotation is list of (row, col) offsets
PIECE_SHAPES = {
    Piece.I: [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(0, 0), (1, 0), (2, 0), (3, 0)],
    ],
    Piece.O: [
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ],
    Piece.T: [
        [(0, 0), (0, 1), (0, 2), (1, 1)],
        [(0, 0), (1, 0), (2, 0), (1, 1)],
        [(1, 0), (1, 1), (1, 2), (0, 1)],
        [(0, 0), (1, 0), (2, 0), (1, -1)],
    ],
    Piece.S: [
        [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
    ],
    Piece.Z: [
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
    ],
    Piece.J: [
        [(0, 0), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (0, 1), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (2, -1)],
    ],
    Piece.L: [
        [(0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
    ],
}

PIECE_CHARS = {
    Piece.I: "I",
    Piece.O: "O",
    Piece.T: "T",
    Piece.S: "S",
    Piece.Z: "Z",
    Piece.J: "J",
    Piece.L: "L",
}


class TetrisBoard:
    """Represents the Tetris playing field."""

    def __init__(self, width: int = 10, height: int = 20):
        self.width = width
        self.height = height
        # Board is a 2D grid; None = empty, str = filled with piece char
        self.grid: list[list[Optional[str]]] = [
            [None] * width for _ in range(height)
        ]

    def is_valid_position(self, cells: list[tuple[int, int]]) -> bool:
        """Check if all cells are within bounds and unoccupied."""
        for r, c in cells:
            if r < 0 or r >= self.height or c < 0 or c >= self.width:
                return False
            if self.grid[r][c] is not None:
                return False
        return True

    def place_cells(self, cells: list[tuple[int, int]], char: str):
        """Place a piece's cells onto the board."""
        for r, c in cells:
            self.grid[r][c] = char

    def clear_lines(self) -> int:
        """Clear completed lines and return the count cleared."""
        new_grid = [row for row in self.grid if any(cell is None for cell in row)]
        lines_cleared = self.height - len(new_grid)
        for _ in range(lines_cleared):
            new_grid.insert(0, [None] * self.width)
        self.grid = new_grid
        return lines_cleared

    def get_column_heights(self) -> list[int]:
        """Get the height of the highest filled cell in each column."""
        heights = []
        for c in range(self.width):
            h = 0
            for r in range(self.height):
                if self.grid[r][c] is not None:
                    h = self.height - r
                    break
            heights.append(h)
        return heights

    def count_holes(self) -> int:
        """Count empty cells that have a filled cell above them."""
        holes = 0
        for c in range(self.width):
            found_block = False
            for r in range(self.height):
                if self.grid[r][c] is not None:
                    found_block = True
                elif found_block:
                    holes += 1
        return holes

    def aggregate_height(self) -> int:
        """Sum of all column heights."""
        return sum(self.get_column_heights())

    def bumpiness(self) -> int:
        """Sum of absolute differences between adjacent column heights."""
        heights = self.get_column_heights()
        return sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))

    def count_complete_lines(self) -> int:
        """Count how many lines are currently complete."""
        return sum(
            1 for row in self.grid if all(cell is not None for cell in row)
        )

    def copy(self) -> "TetrisBoard":
        """Create a deep copy of the board."""
        new_board = TetrisBoard(self.width, self.height)
        new_board.grid = deepcopy(self.grid)
        return new_board

    def __str__(self) -> str:
        border = "+" + "-" * (self.width * 2 + 1) + "+"
        lines = [border]
        for row in self.grid:
            cells = " ".join(c if c else "." for c in row)
            lines.append(f"| {cells} |")
        lines.append(border)
        return "\n".join(lines)


class TetrisGame:
    """Main game controller."""

    SCORING = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}

    def __init__(self, width: int = 10, height: int = 20, seed: Optional[int] = None):
        self.board = TetrisBoard(width, height)
        self.rng = random.Random(seed)
        self.score = 0
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.game_over = False
        self.current_piece: Optional[Piece] = None
        self.next_piece: Optional[Piece] = None
        self._bag: list[Piece] = []
        self._spawn_next()
        self._spawn_next()

    def _refill_bag(self):
        """7-bag randomizer: shuffle all 7 pieces."""
        bag = list(Piece)
        self.rng.shuffle(bag)
        self._bag = bag

    def _draw_piece(self) -> Piece:
        if not self._bag:
            self._refill_bag()
        return self._bag.pop()

    def _spawn_next(self):
        self.current_piece = self.next_piece
        self.next_piece = self._draw_piece()

    def get_piece_cells(
        self, piece: Piece, rotation: int, col: int, row: int
    ) -> list[tuple[int, int]]:
        """Get absolute cell positions for a piece at given position."""
        shape = PIECE_SHAPES[piece][rotation % len(PIECE_SHAPES[piece])]
        return [(row + dr, col + dc) for dr, dc in shape]

    def get_all_placements(
        self, piece: Optional[Piece] = None
    ) -> list[tuple[int, int, int, list[tuple[int, int]]]]:
        """
        Get all valid placements (hard-dropped) for a piece.
        Returns list of (rotation, column, final_row, cells).
        """
        if piece is None:
            piece = self.current_piece
        if piece is None:
            return []

        placements = []
        num_rotations = len(PIECE_SHAPES[piece])

        for rotation in range(num_rotations):
            shape = PIECE_SHAPES[piece][rotation]
            min_c = min(dc for _, dc in shape)
            max_c = max(dc for _, dc in shape)

            for col in range(-min_c, self.board.width - max_c):
                # Find the lowest valid row (hard drop)
                row = 0
                cells = self.get_piece_cells(piece, rotation, col, row)
                if not self.board.is_valid_position(cells):
                    continue

                while True:
                    next_cells = self.get_piece_cells(piece, rotation, col, row + 1)
                    if self.board.is_valid_position(next_cells):
                        row += 1
                    else:
                        break

                final_cells = self.get_piece_cells(piece, rotation, col, row)
                placements.append((rotation, col, row, final_cells))

        return placements

    def place_piece(
        self, rotation: int, col: int
    ) -> tuple[int, bool]:
        """
        Place current piece with given rotation at given column (hard drop).
        Returns (lines_cleared, game_over).
        """
        if self.game_over or self.current_piece is None:
            return 0, True

        piece = self.current_piece
        char = PIECE_CHARS[piece]

        # Find hard-drop position
        row = 0
        cells = self.get_piece_cells(piece, rotation, col, row)
        if not self.board.is_valid_position(cells):
            self.game_over = True
            return 0, True

        while True:
            next_cells = self.get_piece_cells(piece, rotation, col, row + 1)
            if self.board.is_valid_position(next_cells):
                row += 1
            else:
                break

        final_cells = self.get_piece_cells(piece, rotation, col, row)
        self.board.place_cells(final_cells, char)

        cleared = self.board.clear_lines()
        self.lines_cleared += cleared
        self.score += self.SCORING.get(cleared, 0)
        self.pieces_placed += 1

        self._spawn_next()
        return cleared, False
