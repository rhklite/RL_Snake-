"""Hamiltonian-cycle policy for SnakeEnv.

Builds a Hamiltonian cycle (a closed loop visiting every cell exactly once and
returning to its start) over the grid and follows it. Following a fixed cycle
guarantees the snake never traps itself and eventually fills the entire board,
because the head periodically revisits every cell, so any food — wherever it
spawns — is always eventually reachable.

A Hamiltonian cycle on an ``m x n`` grid graph exists iff ``m * n`` is even.
For odd-by-odd grids no such cycle exists and ``build_hamiltonian_cycle``
raises ``ValueError``.

Coordinate convention matches ``snake_env``: positions are ``(x, y) = (col, row)``.
"""

from __future__ import annotations

import numpy as np

from snake_env import DIRECTION_VECTORS, DOWN, LEFT, RIGHT, UP

_DELTA_TO_ACTION = {(0, -1): UP, (0, 1): DOWN, (-1, 0): LEFT, (1, 0): RIGHT}


def _build_even_cols(rows: int, cols: int) -> list[tuple[int, int]]:
    """Boustrophedon Hamiltonian cycle for a grid with an even number of cols.

    Returns cells as ``(row, col)`` in cycle order. Column 0 is the vertical
    return spine; row 0 is traversed left-to-right, the interior columns zigzag
    vertically over rows ``1..rows-1``, and the spine closes the loop. An even
    column count guarantees the last interior column ends at the bottom row so
    it connects to the spine.
    """
    order: list[tuple[int, int]] = []
    # Top row, left to right.
    for c in range(cols):
        order.append((0, c))
    # Interior columns (right to left), zigzag over rows 1..rows-1.
    for c in range(cols - 1, 0, -1):
        if (cols - 1 - c) % 2 == 0:
            for r in range(1, rows):  # downward
                order.append((r, c))
        else:
            for r in range(rows - 1, 0, -1):  # upward
                order.append((r, c))
    # Close the loop up the spine (column 0), rows rows-1..1.
    for r in range(rows - 1, 0, -1):
        order.append((r, 0))
    return order


def _validate_cycle(cycle: list[tuple[int, int]], rows: int, cols: int) -> None:
    n = rows * cols
    assert len(cycle) == n, f"cycle covers {len(cycle)} of {n} cells"
    assert len(set(cycle)) == n, "cycle revisits a cell"
    for i in range(n):
        (y0, x0) = cycle[i]
        (y1, x1) = cycle[(i + 1) % n]
        assert abs(y0 - y1) + abs(x0 - x1) == 1, (
            f"non-adjacent step {cycle[i]} -> {cycle[(i + 1) % n]}"
        )


def build_hamiltonian_cycle(rows: int, cols: int) -> list[tuple[int, int]]:
    """Return a Hamiltonian cycle over the grid as ``(x, y)`` positions in order.

    Requires at least one of ``rows``/``cols`` to be even.
    """
    if cols % 2 == 0:
        cells = _build_even_cols(rows, cols)  # (row, col)
    elif rows % 2 == 0:
        # Transpose: build on a (cols x rows) grid (whose col count == rows is
        # even), then swap axes back.
        transposed = _build_even_cols(cols, rows)  # (row', col') in cols x rows
        cells = [(c, r) for (r, c) in transposed]  # back to (row, col)
    else:
        raise ValueError(
            f"No Hamiltonian cycle exists for an odd-by-odd grid "
            f"({rows}x{cols}); at least one dimension must be even."
        )
    _validate_cycle(cells, rows, cols)
    # Convert (row, col) -> (x, y) = (col, row).
    return [(c, r) for (r, c) in cells]


class HamiltonianPolicy:
    """Follows a precomputed Hamiltonian cycle around the grid.

    Deterministically fills the board. ``act`` derives the snake's heading from
    its own body when possible, so it needs no external state beyond the current
    snake and (only while length 1) the last action taken.
    """

    def __init__(self, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols
        self.cycle = build_hamiltonian_cycle(rows, cols)
        self.n = len(self.cycle)
        self._index = {pos: i for i, pos in enumerate(self.cycle)}

    def _target(self, snake: list[np.ndarray], last_direction: int) -> tuple[int, int]:
        head = (int(snake[0][0]), int(snake[0][1]))
        idx = self._index[head]
        nxt = self.cycle[(idx + 1) % self.n]
        prv = self.cycle[(idx - 1) % self.n]
        if len(snake) > 1:
            neck = (int(snake[1][0]), int(snake[1][1]))
            # Move to whichever cycle-neighbour is not the neck (i.e. forward).
            if neck == prv:
                return nxt
            if neck == nxt:
                return prv
            return nxt  # invariant violated; fall back to forward
        # Length 1: pick a neighbour that is not directly behind us, so the
        # env (which blocks 180-degree reversals) accepts the move.
        dx, dy = DIRECTION_VECTORS[last_direction]
        behind = (head[0] - int(dx), head[1] - int(dy))
        return nxt if nxt != behind else prv

    def act(self, snake: list[np.ndarray], last_direction: int = RIGHT) -> int:
        head = (int(snake[0][0]), int(snake[0][1]))
        tx, ty = self._target(snake, last_direction)
        return _DELTA_TO_ACTION[(tx - head[0], ty - head[1])]
