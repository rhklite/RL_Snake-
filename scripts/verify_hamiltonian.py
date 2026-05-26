"""Verify the Hamiltonian-cycle policy deterministically fills the board.

Runs ``HamiltonianPolicy`` on one or more grid sizes and asserts the snake
reaches full length (every cell occupied) without dying. Prints steps taken
and theoretical efficiency.

Usage:
    python scripts/verify_hamiltonian.py            # default 12x12 and 32x32
    python scripts/verify_hamiltonian.py 6 8 12 32  # custom square sizes
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hamiltonian import HamiltonianPolicy  # noqa: E402
from snake_env import SnakeEnv  # noqa: E402


def run(rows: int, cols: int, seed: int = 0) -> bool:
    n = rows * cols
    # Generous step budget so truncation never pre-empts a full fill;
    # cycle-following needs ~N^2/4 steps total.
    env = SnakeEnv(
        rows=rows,
        cols=cols,
        obs_type="features",
        max_steps_factor=4 * n,
    )
    policy = HamiltonianPolicy(rows, cols)
    env.reset(seed=seed)

    last_direction = 3  # RIGHT, matches env reset
    steps = 0
    while True:
        action = policy.act(env.snake, last_direction)
        last_direction = action
        _, _, terminated, truncated, info = env.step(action)
        steps += 1
        length = len(env.snake)
        if length == n:
            print(
                f"  {rows}x{cols}: FILLED all {n} cells in {steps} steps "
                f"(ideal lower bound ~{n}; cycle overhead ~{steps / n:.1f}x)"
            )
            return True
        if terminated:
            print(
                f"  {rows}x{cols}: DIED ({info.get('cause_of_death')}) at "
                f"length {length}/{n} after {steps} steps"
            )
            return False
        if truncated:
            print(
                f"  {rows}x{cols}: TRUNCATED at length {length}/{n} "
                f"after {steps} steps"
            )
            return False


def main() -> None:
    if len(sys.argv) > 1:
        sizes = [(int(s), int(s)) for s in sys.argv[1:]]
    else:
        sizes = [(12, 12), (32, 32)]

    print("Verifying Hamiltonian-cycle fill:")
    all_ok = True
    for rows, cols in sizes:
        ok = run(rows, cols)
        all_ok = all_ok and ok

    if all_ok:
        print("\nAll grids filled successfully.")
        sys.exit(0)
    print("\nSome grids did not fill.")
    sys.exit(1)


if __name__ == "__main__":
    main()
